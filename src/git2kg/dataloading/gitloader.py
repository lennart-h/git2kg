import logging
from multiprocessing import Lock, Pool
from tempfile import TemporaryDirectory
from typing import Generator, Self

import pandas as pd
from git import Repo
from pydriller import (  # type: ignore[import-untyped]
    Commit,
    ModificationType,
    ModifiedFile,
    Repository,
)
from pydriller import git as pdgit
from textblob import TextBlob  # type: ignore[import-untyped]
from tqdm import tqdm

from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    EditDatapoint,
    FileDatapoint,
    FileInfo,
)

LOG = logging.getLogger(__name__)


# ------------------------- Orchestrating class ------------------------- #
class GitLoader:
    def __init__(self, repo_path: str):
        self.repo_path: str = repo_path
        self._clone_dir: TemporaryDirectory | None = None
        self._local_repo_path: str | None = None

    def _local_repo(self) -> str:
        """Returns a local path to a clone of `repo_path`, cloning it into a
        temporary directory on first use. Subsequent calls (across all
        mining methods of this GitLoader) reuse the same clone instead of
        cloning again. The clone is removed by `close()`.
        """
        if self._local_repo_path is None:
            self._clone_dir = TemporaryDirectory()
            Repo.clone_from(self.repo_path, to_path=self._clone_dir.name)
            self._local_repo_path = self._clone_dir.name
        return self._local_repo_path

    def close(self):
        """Removes the clone created by `_local_repo()`, if any. Safe to call
        even if no mining method that needs a local clone was ever used."""
        if self._clone_dir is not None:
            self._clone_dir.cleanup()
            self._clone_dir = None
            self._local_repo_path = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_git_edits(self, _from: str | None = None) -> Generator[EditDatapoint]:
        """
        Load edits in a git repository. Edits are stored along with information regarding the previous author of the line. If the edit creates a new line, that wasnt there before, the previous author is None.

        Returns
        -------
        Generator[EditDatapoint]
            A generator yielding the mined edits
        """

        repo: Repository = Repository(
            self._local_repo(),
            order="date-order",
            from_commit=_from,
        )
        num_commits = len(list(repo.traverse_commits()))

        for commit in tqdm(
            repo.traverse_commits(),
            total=num_commits,
            desc="Loading edits in commits",
        ):
            # Skip first commit if _from is specified. since that commit was already mined
            if commit.hash == _from:
                continue
            edit_df = _parse_edits(commit, repo)
            for _, edit_row in edit_df.iterrows():
                yield _row_to_edit_datapoint(edit_row)

    def load_git_commits(self, _from: str | None = None) -> Generator[CommitDatapoint]:
        """Mines the commits of the repository

        Returns
        -------
        Generator[CommitDatapoint]
            A generator yielding the mined commits
        """
        repo: Repository = Repository(
            self._local_repo(),
            order="date-order",
            from_commit=_from,
        )
        num_commits = len(list(repo.traverse_commits()))

        for commit in tqdm(
            repo.traverse_commits(), total=num_commits, desc="Loading commits"
        ):
            # Skip first commit if _from is specified. since that commit was already mined
            if commit.hash == _from:
                continue
            yield _create_commit_datapoint(commit)

    def load_git_commits_parallel(
        self, _from: str | None = None, num_processes=1, chunk_size: int = 20
    ) -> Generator[CommitDatapoint]:
        yield from self._load_for_hash_chunk(
            _load_commits_for_hash_chunk, _from, num_processes, chunk_size
        )

    def load_git_edits_parallel(
        self, _from: str | None = None, num_processes=1, chunk_size: int = 20
    ) -> Generator[EditDatapoint]:
        yield from self._load_for_hash_chunk(
            _load_edits_for_hash_chunk, _from, num_processes, chunk_size
        )

    def load_modified_files_parallel(
        self, _from: str | None = None, num_processes=1, chunk_size: int = 20
    ) -> Generator[FileDatapoint]:
        yield from self._load_for_hash_chunk(
            _load_modified_files_for_hash_chunk, _from, num_processes, chunk_size
        )

    def load_modified_files(self, _from: str | None = None) -> Generator[FileDatapoint]:

        repo: Repository = Repository(
            self._local_repo(), order="date-order", from_commit=_from
        )
        num_commits = len(list(repo.traverse_commits()))

        for commit in tqdm(
            repo.traverse_commits(), total=num_commits, desc="Loading modified files"
        ):
            # Skip first commit if _from is specified. since that commit was already mined
            if commit.hash == _from:
                continue
            for mod_file in commit.modified_files:
                yield _create_file_datapoint(mod_file, commit)

    def _load_for_hash_chunk(self, load_func, _from, num_processes, chunk_size):

        repo: Repository = Repository(
            self._local_repo(),
            order="date-order",
            from_commit=_from,
        )
        hashes = [c.hash for c in repo.traverse_commits()]
        if _from:
            hashes = hashes[1:]
        n_commits = len(hashes)
        hash_chunks = [
            hashes[i : i + chunk_size] for i in range(0, len(hashes), chunk_size)
        ]
        config_lock = Lock()  # Lock needed for config file writing of PyDriller, which unfortunately writes the config file in the repo dir
        pool = Pool(
            num_processes,
            initializer=_init_worker,
            initargs=(self._local_repo(), config_lock),
        )
        try:
            with tqdm(
                total=n_commits,
                desc=f"Loading parallely on {num_processes} processes",
            ) as pbar:
                for hash_chunk, chunk in zip(
                    hash_chunks, pool.imap(load_func, hash_chunks), strict=True
                ):
                    pbar.update(len(hash_chunk))
                    yield from chunk
        finally:
            pool.terminate()
            pool.join()


# ------------------------- Functions for parallel processing ------------------------- #
_repo_path = None  # Module Level variable used by workers in multiprocessing only


def _init_worker(repo_path, open_lock):
    global _repo_path
    _repo_path = repo_path

    ### MONKEYPATCH ###
    # Serialize pydriller's repo-open (which writes .git/config) across workers,
    # since it races on the config lock file when multiple workers open concurrently.
    original_open_repository = pdgit.Git._open_repository

    def _locked_open_repository(self):
        with open_lock:
            return original_open_repository(self)

    pdgit.Git._open_repository = _locked_open_repository


def _load_commits_for_hash_chunk(hashes: list[str]):

    repo: Repository = Repository(_repo_path, order="date-order", only_commits=hashes)

    res = [_create_commit_datapoint(c) for c in repo.traverse_commits()]
    return res


def _load_edits_for_hash_chunk(hashes: list[str]):
    repo: Repository = Repository(_repo_path, order="date-order", only_commits=hashes)
    res = []
    for c in repo.traverse_commits():
        parsed_edits = _parse_edits(c, repo)
        res.extend([_row_to_edit_datapoint(row) for _, row in parsed_edits.iterrows()])
    return res


def _load_modified_files_for_hash_chunk(hashes: list[str]):
    repo: Repository = Repository(_repo_path, order="date-order", only_commits=hashes)
    res = []
    for c in repo.traverse_commits():
        res.extend(
            [_create_file_datapoint(mod_file, c) for mod_file in c.modified_files]
        )
    return res


# ------------------------- Utility Functions for parsing commits and modified files and compute some features ------------------------- #


def _parse_edits(commit: Commit, repo: Repository) -> pd.DataFrame:
    author = commit.author.name
    author_email = commit.author.email
    # An edit enters the repository with its commit, i.e. at the committer date
    timestamp = commit.committer_date
    edits = []
    for mod in commit.modified_files:
        try:
            diffs = mod.diff_parsed
        except Exception:
            LOG.warning(
                "Could not read the diff of %s in %s, recording no edits for it",
                mod.new_path or mod.old_path,
                commit.hash,
                exc_info=True,
            )
            continue

        for edit_type in diffs.keys():
            # Convert diff to dataframe of changed lines
            diff_df = pd.DataFrame(
                diffs[edit_type], columns=["line_number", "new_line"]
            )
            blame_entries = []
            # Previous line authorships from git blame, per parent commit
            for parent in commit.parents:
                blame_entries.extend(_blame_entries(repo, parent, mod))
            blame_df = pd.DataFrame(
                blame_entries,
                columns=[
                    "prev_author_name",
                    "prev_author_email",
                    "prev_commit_hash",
                    "line_number",
                ],
            )
            blame_df = blame_df.explode("line_number")

            # Merge by line number to obtain edit relations
            edit_df = diff_df.merge(blame_df, on="line_number", how="left")
            edit_df["author_name"] = author
            edit_df["author_email"] = author_email
            edit_df["new_path"] = mod.new_path
            edit_df["old_path"] = mod.old_path
            edit_df["edit_type"] = edit_type
            edit_df["timestamp"] = timestamp
            edit_df["commit_hash"] = commit.hash
            edits.append(edit_df)
    return pd.concat(edits) if edits else pd.DataFrame()


def _row_to_edit_datapoint(edit_row) -> EditDatapoint:
    prev_author_name = edit_row["prev_author_name"]
    prev_author_email = edit_row["prev_author_email"]
    prev_author = (
        ActorInfo(name=prev_author_name, email=prev_author_email, type="git")
        if isinstance(prev_author_name, str) and isinstance(prev_author_email, str)
        else None
    )
    prev_commit = (
        CommitInfo(hash=edit_row["prev_commit_hash"])
        if pd.notna(edit_row["prev_commit_hash"])
        else None
    )

    return EditDatapoint(
        author=ActorInfo(
            name=str(edit_row["author_name"]),
            email=str(edit_row["author_email"]),
            type="git",
        ),
        prev_author=prev_author,
        prev_commit=prev_commit,
        line_number=edit_row["line_number"],
        new_line=edit_row["new_line"],
        new_path=edit_row["new_path"],
        old_path=edit_row["old_path"] if pd.notnull(edit_row["old_path"]) else None,
        edit_type=edit_row["edit_type"],
        timestamp=edit_row["timestamp"],
        commit=CommitInfo(edit_row["commit_hash"]),
    )


def _parse_sig_type(sig: str):
    header = sig.split("\n")[0].strip()
    if header == "-----BEGIN SSH SIGNATURE-----":
        return "ssh"
    if header == "-----BEGIN PGP SIGNATURE-----":
        return "pgp"


def _create_commit_datapoint(commit: Commit) -> CommitDatapoint:  #
    bugfix_keywords = ["fix", "bug", "issue", "problem", "error"]
    documentation_extensions = (".md", ".txt", ".rst")
    mod_files = commit.modified_files
    blob = TextBlob(commit.msg)
    return CommitDatapoint(
        commit=CommitInfo(commit.hash),
        commit_parents=[CommitInfo(hash) for hash in commit.parents],
        author=ActorInfo(
            name=commit.author.name, email=commit.author.email, type="git"
        ),
        committer=ActorInfo(
            name=commit.committer.name, email=commit.committer.email, type="git"
        ),
        author_date=commit.author_date,
        author_timezone=commit.author_timezone,
        committer_date=commit.committer_date,
        committer_timezone=commit.committer_timezone,
        n_modified_files=(len(mod_files) if mod_files else 0),
        message=commit.msg,
        is_merge=commit.merge,
        is_in_main_branch=commit.in_main_branch,
        branches=commit.branches,
        sig_type=_parse_sig_type(commit._c_object.gpgsig)
        if commit._c_object.gpgsig
        else None,
        sig=commit._c_object.gpgsig if commit._c_object.gpgsig else None,
        n_modified_documentation=len(
            [
                file
                for file in mod_files
                if file.filename.endswith(documentation_extensions)
            ]
        ),
        is_bugfix=any(keyword in commit.msg.lower() for keyword in bugfix_keywords),
        insertions=commit.insertions,
        deletions=commit.deletions,
        message_polarity=blob.sentiment.polarity,
        message_subjectivity=blob.sentiment.subjectivity,
    )


def _blame_entries(repo: Repository, parent: str, mod_file: ModifiedFile) -> list:
    """The line authorships of a file in one parent commit.

    git blame fails on a path the parent does not carry, which happens around
    renames and merges, and on a history it cannot follow. The edits are then
    recorded without a previous author rather than ending the mine.
    """
    if (
        not mod_file.old_path
        or mod_file.change_type == ModificationType.ADD
        or not _carries_source(mod_file)
    ):
        return []
    try:
        return [
            (
                blame_entry.commit.author.name,
                blame_entry.commit.author.email,
                blame_entry.commit.hexsha,
                list(blame_entry.linenos),
            )
            # Consumed here rather than returned: the generator fails while it
            # is read, not where it is created
            for blame_entry in repo.git.repo.blame_incremental(
                parent, mod_file.old_path
            )
        ]
    except Exception:
        LOG.warning(
            "Could not blame %s at %s, recording its edits without a previous author",
            mod_file.old_path,
            parent,
            exc_info=True,
        )
        return []


def _carries_source(mod_file: ModifiedFile) -> bool:
    """Whether a modification carries source, i.e. is not a submodule.

    A submodule has no source and therefore no line count, which is how one is
    told apart from a file. lizard failing to parse a file is not the same
    thing: the source is there, only its metrics cannot be computed, so such a
    modification is treated as the file it is.
    """
    try:
        return mod_file.nloc is not None
    except Exception:
        return True


def _source_metrics(mod_file: ModifiedFile, commit_hash: str) -> tuple:
    """The source metrics of a file version, or none of them.

    Reading any of these runs lizard over the file, which raises on a source
    its parser cannot follow. One such file is no reason to end a mine, so the
    metrics are then recorded as unknown, the same as for a language lizard
    does not support.
    """
    try:
        return (
            mod_file.nloc,
            len(mod_file.methods),
            mod_file.complexity,
            mod_file.token_count,
        )
    except Exception:
        LOG.warning(
            "Could not measure %s in %s, recording its source metrics as unknown",
            mod_file.filename,
            commit_hash,
            exc_info=True,
        )
        return None, None, None, None


def _create_file_datapoint(mod_file: ModifiedFile, commit: Commit):
    nloc, n_methods, complexity, n_token = _source_metrics(mod_file, commit.hash)
    return FileDatapoint(
        file=FileInfo(
            commit_hash=commit.hash, path=mod_file.new_path or mod_file.old_path
        ),
        commit=CommitInfo(commit.hash),
        nloc=nloc,
        # The committer date, matching CommitDatapoint: a file version enters
        # history when its commit does, not at the author date, which for an
        # applied patch can predate the commit object by months.
        timestamp=commit.committer_date,
        extension=mod_file.filename.split(".")[-1],
        n_methods=n_methods,
        complexity=complexity,
        n_token=n_token,
    )
