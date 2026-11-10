import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from pydriller import Commit, Git, ModificationType, Repository
from tqdm import tqdm

LOG = logging.getLogger(__name__)


class FileRegistry:
    # Dictionary of LookuEntry, keyed by commit hash
    # Each entry is a LookupEntry containing the hash, the parent hash and the lookup table with filepaths to file IDs
    lookup_entries: dict[str, "LookupEntry"]
    _file_counter: int
    _commit_counter: int
    _file_ids: list[int]
    _commit_ids: list[int]
    # file_id -> hash of the commit that deleted it
    _deleted_at: dict[int, str]
    _find_cache: dict[tuple[str, str], int | None] = {}

    def __init__(self):
        self.lookup_entries = {}
        self._file_counter = 0
        self._commit_counter = 0
        self._file_ids = []
        self._commit_ids = []
        self._deleted_at = {}
        self._find_cache = {}

    def register_new(self, commit_hash, path):
        LOG.debug(f"Register new file {path} at commit {commit_hash}")
        self.lookup_entries[commit_hash].lookup_table[path] = self._file_counter
        self._file_ids.append(self._file_counter)
        self._file_counter += 1

    def register_move(self, commit_hash, new_path, old_path):
        LOG.info(f"Register move {old_path} -> {new_path} at commit {commit_hash}")
        file_id = self.find_file(commit_hash, old_path)
        self.lookup_entries[commit_hash].lookup_table[new_path] = file_id

    def register_delete(self, commit_hash, path):
        LOG.info(f"Register delete {path} at commit {commit_hash}")
        file_id = self.find_file(commit_hash, path)
        self._deleted_at[file_id] = commit_hash

    def find_file(self, commit_hash, path) -> int:
        LOG.debug(f"Find file {path} at commit {commit_hash}")
        file_id = self._resolve(commit_hash, path)
        if file_id is None:
            raise KeyError(
                f"File '{path}' not found in history of commit {commit_hash}"
            )
        return file_id

    def _known(self, commit_hash, path) -> tuple[int | None, bool]:
        """What one commit says about a path, without looking at its parents.

        The question is settled when the commit records the path, when it was
        answered before, and when the commit is not part of the mined history.
        """
        cache_key = (commit_hash, path)
        if cache_key in self._find_cache:
            return self._find_cache[cache_key], True
        entry = self.lookup_entries.get(commit_hash)
        if entry is None:
            self._find_cache[cache_key] = None
            return None, True
        if path in entry.lookup_table:
            self._find_cache[cache_key] = entry.lookup_table[path]
            return entry.lookup_table[path], True
        return None, False

    def _resolve(self, commit_hash, path) -> int | None:
        """Searches a commit and its ancestors for the id of a path.

        Walks the parents depth first and takes the first that knows the path,
        over an explicit stack rather than by recursion: a repository with a
        history deeper than the interpreter's stack would otherwise end mining
        with a RecursionError.
        """
        file_id, settled = self._known(commit_hash, path)
        if settled:
            return file_id

        # One frame per commit whose parents are still being searched
        frames: list[tuple[str, Iterator[str]]] = [
            (commit_hash, iter(self.lookup_entries[commit_hash].parent_hashes))
        ]
        while frames:
            current, parents = frames[-1]
            for parent_hash in parents:
                file_id, settled = self._known(parent_hash, path)
                if settled:
                    if file_id is None:
                        continue
                    # Every commit still open reaches this ancestor
                    for open_hash, _ in frames:
                        self._find_cache[(open_hash, path)] = file_id
                    return file_id
                frames.append(
                    (parent_hash, iter(self.lookup_entries[parent_hash].parent_hashes))
                )
                break
            else:
                # No ancestor of this commit knows the path
                self._find_cache[(current, path)] = None
                frames.pop()
        return None

    def add_commit(self, commit: Commit):
        LOG.debug(f"Added commit {commit.hash}")
        self.lookup_entries[commit.hash] = LookupEntry(
            commit_hash=commit.hash,
            commit_id=self._commit_counter,
            parent_hashes=commit.parents,
        )
        self._commit_ids.append(self._commit_counter)
        self._commit_counter += 1

        for mod_file in commit.modified_files:
            change_type = mod_file.change_type
            if change_type == ModificationType.ADD:
                self.register_new(commit.hash, mod_file.new_path)
            elif change_type == ModificationType.RENAME:
                self.register_move(commit.hash, mod_file.new_path, mod_file.old_path)
            elif change_type == ModificationType.DELETE:
                self.register_delete(commit.hash, mod_file.old_path)

    @property
    def file_ids(self) -> list[int]:
        return list(self._file_ids)

    @property
    def commit_ids(self) -> list[int]:
        return list(self._commit_ids)

    @property
    def deleted_at(self) -> dict[int, str]:
        """Maps file_id -> hash of the commit that deleted it, for files that
        have been deleted. File ids not present in this dict are still alive
        as of the last mined commit.
        """
        return dict(self._deleted_at)

    @staticmethod
    def from_local_repository(repo: str) -> "FileRegistry":
        LOG.info(f"Start creating FileRegistry from repository {repo}")

        repository: Repository = Repository(repo, order="date-order")
        registry = FileRegistry()
        total_commits = Git(repo).total_commits()
        for commit in tqdm(repository.traverse_commits(), total=total_commits):
            registry.add_commit(commit)
        return registry

    def to_dict(self) -> dict:
        """Converts the registry to a plain, JSON-serializable dict. `_find_cache`
        is intentionally not included, since it's a pure memoization cache and gets
        rebuilt lazily by `find_file()` as needed after `from_dict()`.
        """
        return {
            "file_counter": self._file_counter,
            "commit_counter": self._commit_counter,
            "file_ids": self._file_ids,
            "commit_ids": self._commit_ids,
            "deleted_at": self._deleted_at,
            "lookup_entries": {
                commit_hash: asdict(entry)
                for commit_hash, entry in self.lookup_entries.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "FileRegistry":
        registry = FileRegistry()
        registry._file_counter = data["file_counter"]
        registry._commit_counter = data["commit_counter"]
        registry._file_ids = data["file_ids"]
        registry._commit_ids = data["commit_ids"]
        # Older saved registries predate deletion tracking
        registry._deleted_at = {
            int(file_id): commit_hash
            for file_id, commit_hash in data.get("deleted_at", {}).items()
        }
        registry.lookup_entries = {
            commit_hash: LookupEntry(**entry)
            for commit_hash, entry in data["lookup_entries"].items()
        }
        return registry


@dataclass
class LookupEntry:
    commit_hash: str
    commit_id: int
    parent_hashes: list[str]
    lookup_table: dict[str, int] = field(default_factory=dict)
