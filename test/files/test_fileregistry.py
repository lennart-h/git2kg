import datetime
import os
import shutil
import unittest
from tempfile import TemporaryDirectory

from git import Repo
from git.util import Actor

from git2kg.files.fileregistry import FileRegistry


class TestFileRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.repo_dir = f"{self.tmp_dir.name}/test_repo_files"
        if os.path.exists(self.repo_dir):
            shutil.rmtree(self.repo_dir)
        os.makedirs(self.repo_dir)
        repo = Repo.init(self.repo_dir)
        _author = Actor("author_a", "author-a@uni-wuerzburg.de")

        # Commit 0: create two new files
        with open(os.path.join(self.repo_dir, "file_a.py"), "w") as f:
            f.write("This is file a\n")
        with open(os.path.join(self.repo_dir, "file_b.py"), "w") as f:
            f.write("This is file b\n")
        repo.index.add(["file_a.py", "file_b.py"])
        self.commit_0 = repo.index.commit(
            "commit 0 - create file_a.py and file_b.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        )

        # Commit 1: move file_a.py to a different path
        os.makedirs(os.path.join(self.repo_dir, "utils"))
        repo.index.move(["file_a.py", "utils/file_a.py"])
        self.commit_1 = repo.index.commit(
            "commit 1 - move file_a.py to utils/file_a.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T01:00:00+00:00"),
        )

        # Commit 2: create a third new file
        with open(os.path.join(self.repo_dir, "file_c.py"), "w") as f:
            f.write("This is file c\n")
        repo.index.add(["file_c.py"])
        self.commit_2 = repo.index.commit(
            "commit 2 - create file_c.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T02:00:00+00:00"),
        )

        # Commit 3: delete file_b.py, freeing up its path
        repo.index.remove(["file_b.py"], working_tree=True)
        self.commit_3 = repo.index.commit(
            "commit 3 - delete file_b.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T03:00:00+00:00"),
        )

        # Commit 4: move utils/file_a.py to the now-freed file_b.py path
        repo.index.move(["utils/file_a.py", "file_b.py"])
        self.commit_4 = repo.index.commit(
            "commit 4 - move utils/file_a.py to file_b.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T04:00:00+00:00"),
        )

        # Commit 5: delete file_c.py
        repo.index.remove(["file_c.py"], working_tree=True)
        self.commit_5 = repo.index.commit(
            "commit 5 - delete file_c.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T05:00:00+00:00"),
        )

        # Commit 6: create a new file at the now-freed file_c.py path
        with open(os.path.join(self.repo_dir, "file_c.py"), "w") as f:
            f.write("This is a new file_c\n")
        repo.index.add(["file_c.py"])
        self.commit_6 = repo.index.commit(
            "commit 6 - re-create file_c.py",
            author=_author,
            committer=_author,
            author_date=datetime.datetime.fromisoformat("2025-01-01T06:00:00+00:00"),
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_file_registry_from_repository(self):

        registry = FileRegistry.from_local_repository(self.repo_dir)

        self.assertEqual(registry._file_counter, 4)
        self.assertEqual(len(registry.lookup_entries), 7)
        self.assertEqual(
            len(registry.lookup_entries[self.commit_0.hexsha].lookup_table), 2
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_1.hexsha].lookup_table), 1
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_2.hexsha].lookup_table), 1
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_3.hexsha].lookup_table), 0
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_4.hexsha].lookup_table), 1
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_5.hexsha].lookup_table), 0
        )
        self.assertEqual(
            len(registry.lookup_entries[self.commit_6.hexsha].lookup_table), 1
        )

    def test_register_delete(self):
        registry = FileRegistry.from_local_repository(self.repo_dir)

        # file_b.py (id 1) deleted at commit 3; file_c.py (id 2) deleted at
        # commit 5. Neither the original file_a.py (id 0, moved into the
        # freed file_b.py path at commit 4) nor the file re-created at the
        # freed file_c.py path (id 3, at commit 6) were ever deleted.
        self.assertEqual(
            registry.deleted_at,
            {
                1: self.commit_3.hexsha,
                2: self.commit_5.hexsha,
            },
        )

    def test_serialize_roundtrip_preserves_deletions(self):
        registry = FileRegistry.from_local_repository(self.repo_dir)

        restored = FileRegistry.from_dict(registry.to_dict())
        self.assertEqual(restored.deleted_at, registry.deleted_at)

        # Older, already-saved registries predate deletion tracking and won't
        # have a "deleted_at" key at all - must still load cleanly.
        data = registry.to_dict()
        del data["deleted_at"]
        legacy_restored = FileRegistry.from_dict(data)
        self.assertEqual(legacy_restored.deleted_at, {})

    def test_find_file_id(self):

        registry = FileRegistry.from_local_repository(self.repo_dir)

        self.assertEqual(registry.find_file(self.commit_0.hexsha, "file_a.py"), 0)
        self.assertEqual(registry.find_file(self.commit_1.hexsha, "utils/file_a.py"), 0)
        self.assertEqual(registry.find_file(self.commit_2.hexsha, "utils/file_a.py"), 0)
        self.assertEqual(registry.find_file(self.commit_3.hexsha, "utils/file_a.py"), 0)
        self.assertEqual(registry.find_file(self.commit_4.hexsha, "file_b.py"), 0)
        self.assertEqual(registry.find_file(self.commit_5.hexsha, "file_b.py"), 0)
        self.assertEqual(registry.find_file(self.commit_6.hexsha, "file_b.py"), 0)

        self.assertEqual(registry.find_file(self.commit_0.hexsha, "file_b.py"), 1)
        self.assertEqual(registry.find_file(self.commit_1.hexsha, "file_b.py"), 1)
        self.assertEqual(registry.find_file(self.commit_2.hexsha, "file_b.py"), 1)
        self.assertEqual(registry.find_file(self.commit_3.hexsha, "file_b.py"), 1)

        self.assertEqual(registry.find_file(self.commit_2.hexsha, "file_c.py"), 2)
        self.assertEqual(registry.find_file(self.commit_3.hexsha, "file_c.py"), 2)
        self.assertEqual(registry.find_file(self.commit_4.hexsha, "file_c.py"), 2)
        self.assertEqual(registry.find_file(self.commit_5.hexsha, "file_c.py"), 2)

        self.assertEqual(registry.find_file(self.commit_6.hexsha, "file_c.py"), 3)


class TestFileRegistryMergeCommits(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.repo_dir = f"{self.tmp_dir.name}/test_repo_merge"
        if os.path.exists(self.repo_dir):
            shutil.rmtree(self.repo_dir)
        os.makedirs(self.repo_dir)
        repo = Repo.init(self.repo_dir)
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "author_a")
            cw.set_value("user", "email", "author-a@uni-wuerzburg.de")

        # Commit 0 (main): create file_a.py
        with open(os.path.join(self.repo_dir, "file_a.py"), "w") as f:
            f.write("This is file a\n")
        repo.index.add(["file_a.py"])
        self.commit_0 = repo.index.commit("commit 0 - create file_a.py")
        main_branch = repo.active_branch.name

        # Branch off for the feature work
        feature = repo.create_head("feature", self.commit_0)
        feature.checkout()

        # Commit 1 (feature): move file_a.py to utils/file_a.py
        os.makedirs(os.path.join(self.repo_dir, "utils"))
        repo.index.move(["file_a.py", "utils/file_a.py"])
        self.commit_1_feature = repo.index.commit(
            "commit 1 - move file_a.py to utils/file_a.py on feature"
        )

        # Back on main: create file_b.py, diverging from feature
        repo.heads[main_branch].checkout()
        with open(os.path.join(self.repo_dir, "file_b.py"), "w") as f:
            f.write("This is file b\n")
        repo.index.add(["file_b.py"])
        self.commit_1_main = repo.index.commit("commit 1 - create file_b.py on main")

        # Merge feature into main
        repo.git.merge("feature", "--no-ff", "-m", "merge feature into main")
        self.merge_commit = repo.head.commit

        # Commit after the merge: create file_c.py
        with open(os.path.join(self.repo_dir, "file_c.py"), "w") as f:
            f.write("This is file c\n")
        repo.index.add(["file_c.py"])
        self.commit_after_merge = repo.index.commit(
            "commit 2 - create file_c.py after merge"
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_merge_commit_is_registered(self):
        registry = FileRegistry.from_local_repository(self.repo_dir)

        self.assertIn(self.merge_commit.hexsha, registry.lookup_entries)
        merge_entry = registry.lookup_entries[self.merge_commit.hexsha]
        self.assertEqual(
            set(merge_entry.parent_hashes),
            {self.commit_1_main.hexsha, self.commit_1_feature.hexsha},
        )

    def test_find_file_across_merge(self):
        registry = FileRegistry.from_local_repository(self.repo_dir)

        # file_a.py, renamed to utils/file_a.py on the feature branch, must
        # resolve to the same id after the merge and beyond.
        file_a_id = registry.find_file(self.commit_1_feature.hexsha, "utils/file_a.py")
        self.assertEqual(
            registry.find_file(self.merge_commit.hexsha, "utils/file_a.py"), file_a_id
        )
        self.assertEqual(
            registry.find_file(self.commit_after_merge.hexsha, "utils/file_a.py"),
            file_a_id,
        )

        # file_b.py, created on main after the branch point, must resolve
        # through the merge commit's other parent.
        file_b_id = registry.find_file(self.commit_1_main.hexsha, "file_b.py")
        self.assertEqual(
            registry.find_file(self.merge_commit.hexsha, "file_b.py"), file_b_id
        )
        self.assertEqual(
            registry.find_file(self.commit_after_merge.hexsha, "file_b.py"), file_b_id
        )

        self.assertNotEqual(file_a_id, file_b_id)


class _Entry:
    """A registered commit, without building a repository for it."""

    def __init__(self, commit_hash, parent_hashes, lookup_table=None):
        self.commit_hash = commit_hash
        self.parent_hashes = parent_hashes
        self.lookup_table = lookup_table or {}


class TestDeepHistory(unittest.TestCase):
    """A path is found however deep in the history it was added.

    PyDriller walks a repository commit by commit, so a long chain of commits
    that do not touch a file sits between it and the commit that added it.
    """

    def _chain(self, length: int) -> FileRegistry:
        registry = FileRegistry()
        registry.lookup_entries["c0"] = _Entry("c0", [], {"deep.py": 42})
        for i in range(1, length):
            registry.lookup_entries[f"c{i}"] = _Entry(f"c{i}", [f"c{i - 1}"])
        return registry

    def test_finds_a_file_added_deeper_than_the_recursion_limit(self):
        import sys

        length = sys.getrecursionlimit() * 3
        registry = self._chain(length)

        self.assertEqual(42, registry.find_file(f"c{length - 1}", "deep.py"))

    def test_a_missing_file_raises_however_deep_the_history(self):
        import sys

        length = sys.getrecursionlimit() * 3
        registry = self._chain(length)

        with self.assertRaises(KeyError):
            registry.find_file(f"c{length - 1}", "absent.py")

    def test_the_answer_is_cached_for_every_commit_on_the_way(self):
        registry = self._chain(50)

        registry.find_file("c49", "deep.py")

        # The next lookup of any commit on that path is answered outright
        self.assertEqual(42, registry._find_cache[("c25", "deep.py")])
        self.assertEqual(42, registry._find_cache[("c49", "deep.py")])
