import datetime
import os
import shutil
import unittest
from tempfile import TemporaryDirectory
from typing import Generator
from unittest import mock

from git import Repo
from git.util import Actor

import git2kg.dataloading.gitloader as gitloader
from git2kg.dataloading.gitloader import (
    GitLoader,
    _carries_source,
    _source_metrics,
)
from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    EditDatapoint,
    FileDatapoint,
)


class TestGitLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()

        self.repo_dir = f"{self.tmp_dir.name}/test_repo_commits"
        if os.path.exists(self.repo_dir):
            shutil.rmtree(self.repo_dir)
        os.makedirs(self.repo_dir)
        repo = Repo.init(self.repo_dir, initial_branch="main")
        # Create two different authors
        _author_a = Actor("author_a", "author-a@uni-wuerzburg.de")
        _author_b = Actor("author_b", "author-b@uni-wuerzburg.de")
        _author_c = Actor("author_c", "author-c@uni-wuerzburg.de")
        self.author_a = ActorInfo(
            name="author_a", email="author-a@uni-wuerzburg.de", type="git"
        )
        self.author_b = ActorInfo(
            name="author_b", email="author-b@uni-wuerzburg.de", type="git"
        )
        self.author_c = ActorInfo(
            name="author_c", email="author-c@uni-wuerzburg.de", type="git"
        )
        # Write file and commit with author a
        with open(os.path.join(self.repo_dir, "file.py"), "w") as f:
            f.write(
                """This is a test file
                    Created by author a
                    Third line
                    Fourth line"""
            )
        repo.index.add(["file.py"])
        self.commit_0 = repo.index.commit(
            "author_a commit 0",
            author=_author_a,
            committer=_author_a,
            author_date=datetime.datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        )

        # Alter two lines and commit with author b
        with open(os.path.join(self.repo_dir, "file.py"), "w") as f:
            f.write(
                """This is a test file
                    Created by author a
                    Third line
                    Last line"""
            )
        repo.index.add(["file.py"])
        self.commit_1 = repo.index.commit(
            "author_b commit 0",
            author=_author_b,
            committer=_author_b,
            author_date=datetime.datetime.fromisoformat("2025-01-01T05:00:00+00:00"),
        )
        # Delete last line and commit with author a
        with open(os.path.join(self.repo_dir, "file.py"), "w") as f:
            f.write(
                """This is a test file
                    Created by author a
                    Third line\n"""
            )
        repo.index.add(["file.py"])
        self.commit_2 = repo.index.commit(
            "author_a commit 1",
            author=_author_a,
            committer=_author_a,
            author_date=datetime.datetime.fromisoformat("2025-01-01T10:00:00+00:00"),
        )
        # Add new file and commit with author c
        with open(os.path.join(self.repo_dir, "file2.py"), "w") as f:
            f.write(
                """This is a new file
                    Created by author c
                    Third line"""
            )
        repo.index.add(["file2.py"])
        self.commit_3 = repo.index.commit(
            "author_c commit 0 - Fix authorship",
            author=_author_c,
            committer=_author_c,
            author_date=datetime.datetime.fromisoformat("2025-01-01T12:00:00+00:00"),
        )

        self.uut = GitLoader(self.repo_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_git_edits(self):
        edits = self.uut.load_git_edits()
        edits_list = list(edits)

        self.assertIsInstance(edits, Generator)
        self.assertTrue(all(isinstance(e, EditDatapoint) for e in edits_list))
        # Assert edits of existing lines

        edits_of_existing_lines = [
            e for e in edits_list if e.prev_author is not None
        ]  # edits[pd.isnull(edits["prev_author_name"]) == False]
        # Alter existing line -> 2 edits (1 line deleted, 1 line added) + delete last line -> 1 edit
        self.assertEqual(len(edits_of_existing_lines), 3)
        self.assertEqual(edits_of_existing_lines[0].commit.hash, self.commit_1.hexsha)
        self.assertEqual(
            edits_of_existing_lines[0].prev_commit.hash, self.commit_0.hexsha
        )
        self.assertEqual(edits_of_existing_lines[0].author, self.author_b)
        self.assertEqual(edits_of_existing_lines[0].prev_author, self.author_a)

        self.assertEqual(edits_of_existing_lines[1].commit.hash, self.commit_1.hexsha)
        self.assertEqual(
            edits_of_existing_lines[1].prev_commit.hash, self.commit_0.hexsha
        )
        self.assertEqual(edits_of_existing_lines[1].author, self.author_b)
        self.assertEqual(edits_of_existing_lines[1].prev_author, self.author_a)

        self.assertEqual(edits_of_existing_lines[2].commit.hash, self.commit_2.hexsha)
        self.assertEqual(
            edits_of_existing_lines[2].prev_commit.hash, self.commit_1.hexsha
        )
        self.assertEqual(edits_of_existing_lines[2].author, self.author_a)
        self.assertEqual(edits_of_existing_lines[2].prev_author, self.author_b)
        # Assert added edits
        added_edits = [e for e in edits_of_existing_lines if e.edit_type == "added"]

        self.assertSetEqual(set(map(lambda x: x.author, added_edits)), {self.author_b})
        self.assertSetEqual(
            set(map(lambda x: x.prev_author, added_edits)), {self.author_a}
        )
        self.assertSetEqual(set(map(lambda x: x.line_number, added_edits)), {4})

        # Assert deleted edits
        deleted_edits = [e for e in edits_of_existing_lines if e.edit_type == "deleted"]

        self.assertSetEqual(
            set(map(lambda x: x.author, deleted_edits)), {self.author_b, self.author_a}
        )

        self.assertSetEqual(
            set(map(lambda x: x.prev_author, deleted_edits)),
            {self.author_a, self.author_b},
        )

        self.assertSetEqual(set(map(lambda x: x.line_number, deleted_edits)), {4})

        # Assert edits of new lines
        edits_of_new_lines = [e for e in edits_list if e.prev_author is None]
        self.assertEqual(len(edits_of_new_lines), 7)
        self.assertSetEqual(
            set(map(lambda x: x.author, edits_of_new_lines)),
            {self.author_a, self.author_c},
        )

    def test_load_git_edits_parallel(self):
        edits = self.uut.load_git_edits_parallel(num_processes=2, chunk_size=1)
        edits_list = list(edits)

        self.assertIsInstance(edits, Generator)
        self.assertTrue(all(isinstance(e, EditDatapoint) for e in edits_list))
        # Assert edits of existing lines

        edits_of_existing_lines = [
            e for e in edits_list if e.prev_author is not None
        ]  # edits[pd.isnull(edits["prev_author_name"]) == False]
        # Alter existing line -> 2 edits (1 line deleted, 1 line added) + delete last line -> 1 edit
        self.assertEqual(len(edits_of_existing_lines), 3)

        # Assert added edits
        added_edits = [e for e in edits_of_existing_lines if e.edit_type == "added"]

        self.assertSetEqual(set(map(lambda x: x.author, added_edits)), {self.author_b})
        self.assertSetEqual(
            set(map(lambda x: x.prev_author, added_edits)), {self.author_a}
        )
        self.assertSetEqual(set(map(lambda x: x.line_number, added_edits)), {4})

        # Assert deleted edits
        deleted_edits = [e for e in edits_of_existing_lines if e.edit_type == "deleted"]

        self.assertSetEqual(
            set(map(lambda x: x.author, deleted_edits)), {self.author_b, self.author_a}
        )

        self.assertSetEqual(
            set(map(lambda x: x.prev_author, deleted_edits)),
            {self.author_a, self.author_b},
        )

        self.assertSetEqual(set(map(lambda x: x.line_number, deleted_edits)), {4})

        # Assert edits of new lines
        edits_of_new_lines = [e for e in edits_list if e.prev_author is None]
        self.assertEqual(len(edits_of_new_lines), 7)
        self.assertSetEqual(
            set(map(lambda x: x.author, edits_of_new_lines)),
            {self.author_a, self.author_c},
        )

    def test_load_edits_from_commit(self):
        edits = list(self.uut.load_git_edits(_from=self.commit_1.hexsha))
        hashes = set(map(lambda e: e.commit.hash, edits))
        self.assertSetEqual(hashes, {self.commit_2.hexsha, self.commit_3.hexsha})

    def test_load_git_commits(self):
        commits = self.uut.load_git_commits()
        commits_list = list(commits)
        self.assertIsInstance(commits, Generator)
        self.assertTrue(all(isinstance(c, CommitDatapoint) for c in commits_list))

        self.assertEqual(len(commits_list), 4)

        # First commit assertions

        self.assertEqual(commits_list[0].commit.hash, self.commit_0.hexsha)
        self.assertListEqual(commits_list[0].commit_parents, [])
        self.assertEqual(commits_list[0].author, self.author_a)
        self.assertEqual(commits_list[0].committer, self.author_a)
        self.assertEqual(commits_list[0].author_date, self.commit_0.authored_datetime)
        self.assertEqual(
            commits_list[0].author_timezone, self.commit_0.author_tz_offset
        )
        self.assertEqual(
            commits_list[0].committer_date, self.commit_0.committed_datetime
        )
        self.assertEqual(
            commits_list[0].committer_timezone, self.commit_0.committer_tz_offset
        )
        self.assertEqual(commits_list[0].n_modified_files, 1)
        self.assertEqual(commits_list[0].message, "author_a commit 0")
        self.assertEqual(commits_list[0].is_merge, False)
        self.assertEqual(commits_list[0].is_in_main_branch, True)
        self.assertEqual(commits_list[0].branches, {"main"})
        self.assertEqual(commits_list[0].sig_type, None)
        self.assertEqual(commits_list[0].sig, None)
        self.assertEqual(commits_list[0].n_modified_documentation, 0)
        self.assertEqual(commits_list[0].is_bugfix, False)
        self.assertEqual(commits_list[0].insertions, 4)
        self.assertEqual(commits_list[0].deletions, 0)

        # Second commit assertions

        self.assertEqual(commits_list[1].commit.hash, self.commit_1.hexsha)
        self.assertListEqual(
            commits_list[1].commit_parents, [CommitInfo(self.commit_0.hexsha)]
        )
        self.assertEqual(commits_list[1].author, self.author_b)
        self.assertEqual(commits_list[1].committer, self.author_b)
        self.assertEqual(commits_list[1].author_date, self.commit_1.authored_datetime)
        self.assertEqual(
            commits_list[1].author_timezone, self.commit_1.author_tz_offset
        )
        self.assertEqual(
            commits_list[1].committer_date, self.commit_1.committed_datetime
        )
        self.assertEqual(
            commits_list[1].committer_timezone, self.commit_1.committer_tz_offset
        )
        self.assertEqual(commits_list[1].n_modified_files, 1)
        self.assertEqual(commits_list[1].message, "author_b commit 0")
        self.assertEqual(commits_list[1].is_merge, False)
        self.assertEqual(commits_list[1].is_in_main_branch, True)
        self.assertEqual(commits_list[1].branches, {"main"})
        self.assertEqual(commits_list[1].sig_type, None)
        self.assertEqual(commits_list[1].sig, None)
        self.assertEqual(commits_list[1].n_modified_documentation, 0)
        self.assertEqual(commits_list[1].is_bugfix, False)
        self.assertEqual(commits_list[1].insertions, 1)
        self.assertEqual(commits_list[1].deletions, 1)

        # Third commit assertions

        self.assertEqual(commits_list[2].commit.hash, self.commit_2.hexsha)
        self.assertListEqual(
            commits_list[2].commit_parents, [CommitInfo(self.commit_1.hexsha)]
        )
        self.assertEqual(commits_list[2].author, self.author_a)
        self.assertEqual(commits_list[2].committer, self.author_a)
        self.assertEqual(commits_list[2].author_date, self.commit_2.authored_datetime)
        self.assertEqual(
            commits_list[2].author_timezone, self.commit_2.author_tz_offset
        )
        self.assertEqual(
            commits_list[2].committer_date, self.commit_2.committed_datetime
        )
        self.assertEqual(
            commits_list[2].committer_timezone, self.commit_2.committer_tz_offset
        )
        self.assertEqual(commits_list[2].n_modified_files, 1)
        self.assertEqual(commits_list[2].message, "author_a commit 1")
        self.assertEqual(commits_list[2].is_merge, False)
        self.assertEqual(commits_list[2].is_in_main_branch, True)
        self.assertEqual(commits_list[2].branches, {"main"})
        self.assertEqual(commits_list[2].sig_type, None)
        self.assertEqual(commits_list[2].sig, None)
        self.assertEqual(commits_list[2].n_modified_documentation, 0)
        self.assertEqual(commits_list[2].is_bugfix, False)
        self.assertEqual(commits_list[2].insertions, 0)
        self.assertEqual(commits_list[2].deletions, 1)

        # Fourths commit assertions

        self.assertEqual(commits_list[3].commit.hash, self.commit_3.hexsha)
        self.assertListEqual(
            commits_list[3].commit_parents, [CommitInfo(self.commit_2.hexsha)]
        )
        self.assertEqual(commits_list[3].author, self.author_c)
        self.assertEqual(commits_list[3].committer, self.author_c)
        self.assertEqual(commits_list[3].author_date, self.commit_3.authored_datetime)
        self.assertEqual(
            commits_list[3].author_timezone, self.commit_3.author_tz_offset
        )
        self.assertEqual(
            commits_list[3].committer_date, self.commit_3.committed_datetime
        )
        self.assertEqual(
            commits_list[3].committer_timezone, self.commit_3.committer_tz_offset
        )
        self.assertEqual(commits_list[3].n_modified_files, 1)
        self.assertEqual(commits_list[3].message, "author_c commit 0 - Fix authorship")
        self.assertEqual(commits_list[3].is_merge, False)
        self.assertEqual(commits_list[3].is_in_main_branch, True)
        self.assertEqual(commits_list[3].branches, {"main"})
        self.assertEqual(commits_list[3].sig_type, None)
        self.assertEqual(commits_list[3].sig, None)
        self.assertEqual(commits_list[3].n_modified_documentation, 0)
        self.assertEqual(commits_list[3].is_bugfix, True)
        self.assertEqual(commits_list[3].insertions, 3)
        self.assertEqual(commits_list[3].deletions, 0)

    def test_load_git_commits_parallel(self):
        commits = self.uut.load_git_commits_parallel(num_processes=2, chunk_size=1)
        commits_list = list(commits)
        self.assertIsInstance(commits, Generator)
        self.assertTrue(all(isinstance(c, CommitDatapoint) for c in commits_list))

        self.assertEqual(len(commits_list), 4)

        # First commit assertions

        self.assertEqual(commits_list[0].commit.hash, self.commit_0.hexsha)
        self.assertListEqual(commits_list[0].commit_parents, [])
        self.assertEqual(commits_list[0].author, self.author_a)
        self.assertEqual(commits_list[0].committer, self.author_a)
        self.assertEqual(commits_list[0].author_date, self.commit_0.authored_datetime)
        self.assertEqual(
            commits_list[0].author_timezone, self.commit_0.author_tz_offset
        )
        self.assertEqual(
            commits_list[0].committer_date, self.commit_0.committed_datetime
        )
        self.assertEqual(
            commits_list[0].committer_timezone, self.commit_0.committer_tz_offset
        )
        self.assertEqual(commits_list[0].n_modified_files, 1)
        self.assertEqual(commits_list[0].message, "author_a commit 0")
        self.assertEqual(commits_list[0].is_merge, False)
        self.assertEqual(commits_list[0].is_in_main_branch, True)
        self.assertEqual(commits_list[0].branches, {"main"})
        self.assertEqual(commits_list[0].sig_type, None)
        self.assertEqual(commits_list[0].sig, None)
        self.assertEqual(commits_list[0].n_modified_documentation, 0)
        self.assertEqual(commits_list[0].is_bugfix, False)
        self.assertEqual(commits_list[0].insertions, 4)
        self.assertEqual(commits_list[0].deletions, 0)

        # Second commit assertions

        self.assertEqual(commits_list[1].commit.hash, self.commit_1.hexsha)
        self.assertListEqual(
            commits_list[1].commit_parents, [CommitInfo(self.commit_0.hexsha)]
        )
        self.assertEqual(commits_list[1].author, self.author_b)
        self.assertEqual(commits_list[1].committer, self.author_b)
        self.assertEqual(commits_list[1].author_date, self.commit_1.authored_datetime)
        self.assertEqual(
            commits_list[1].author_timezone, self.commit_1.author_tz_offset
        )
        self.assertEqual(
            commits_list[1].committer_date, self.commit_1.committed_datetime
        )
        self.assertEqual(
            commits_list[1].committer_timezone, self.commit_1.committer_tz_offset
        )
        self.assertEqual(commits_list[1].n_modified_files, 1)
        self.assertEqual(commits_list[1].message, "author_b commit 0")
        self.assertEqual(commits_list[1].is_merge, False)
        self.assertEqual(commits_list[1].is_in_main_branch, True)
        self.assertEqual(commits_list[1].branches, {"main"})
        self.assertEqual(commits_list[1].sig_type, None)
        self.assertEqual(commits_list[1].sig, None)
        self.assertEqual(commits_list[1].n_modified_documentation, 0)
        self.assertEqual(commits_list[1].is_bugfix, False)
        self.assertEqual(commits_list[1].insertions, 1)
        self.assertEqual(commits_list[1].deletions, 1)

        # Third commit assertions

        self.assertEqual(commits_list[2].commit.hash, self.commit_2.hexsha)
        self.assertListEqual(
            commits_list[2].commit_parents, [CommitInfo(self.commit_1.hexsha)]
        )
        self.assertEqual(commits_list[2].author, self.author_a)
        self.assertEqual(commits_list[2].committer, self.author_a)
        self.assertEqual(commits_list[2].author_date, self.commit_2.authored_datetime)
        self.assertEqual(
            commits_list[2].author_timezone, self.commit_2.author_tz_offset
        )
        self.assertEqual(
            commits_list[2].committer_date, self.commit_2.committed_datetime
        )
        self.assertEqual(
            commits_list[2].committer_timezone, self.commit_2.committer_tz_offset
        )
        self.assertEqual(commits_list[2].n_modified_files, 1)
        self.assertEqual(commits_list[2].message, "author_a commit 1")
        self.assertEqual(commits_list[2].is_merge, False)
        self.assertEqual(commits_list[2].is_in_main_branch, True)
        self.assertEqual(commits_list[2].branches, {"main"})
        self.assertEqual(commits_list[2].sig_type, None)
        self.assertEqual(commits_list[2].sig, None)
        self.assertEqual(commits_list[2].n_modified_documentation, 0)
        self.assertEqual(commits_list[2].is_bugfix, False)
        self.assertEqual(commits_list[2].insertions, 0)
        self.assertEqual(commits_list[2].deletions, 1)

        # Fourths commit assertions

        self.assertEqual(commits_list[3].commit.hash, self.commit_3.hexsha)
        self.assertListEqual(
            commits_list[3].commit_parents, [CommitInfo(self.commit_2.hexsha)]
        )
        self.assertEqual(commits_list[3].author, self.author_c)
        self.assertEqual(commits_list[3].committer, self.author_c)
        self.assertEqual(commits_list[3].author_date, self.commit_3.authored_datetime)
        self.assertEqual(
            commits_list[3].author_timezone, self.commit_3.author_tz_offset
        )
        self.assertEqual(
            commits_list[3].committer_date, self.commit_3.committed_datetime
        )
        self.assertEqual(
            commits_list[3].committer_timezone, self.commit_3.committer_tz_offset
        )
        self.assertEqual(commits_list[3].n_modified_files, 1)
        self.assertEqual(commits_list[3].message, "author_c commit 0 - Fix authorship")
        self.assertEqual(commits_list[3].is_merge, False)
        self.assertEqual(commits_list[3].is_in_main_branch, True)
        self.assertEqual(commits_list[3].branches, {"main"})
        self.assertEqual(commits_list[3].sig_type, None)
        self.assertEqual(commits_list[3].sig, None)
        self.assertEqual(commits_list[3].n_modified_documentation, 0)
        self.assertEqual(commits_list[3].is_bugfix, True)
        self.assertEqual(commits_list[3].insertions, 3)
        self.assertEqual(commits_list[3].deletions, 0)

    def test_mine_commits_from_commits(self):
        commits = list(self.uut.load_git_commits(_from=self.commit_1.hexsha))
        self.assertEqual(len(commits), 2)
        self.assertListEqual(
            list(map(lambda c: c.commit.hash, commits)),
            [self.commit_2.hexsha, self.commit_3.hexsha],
        )

    def test_load_modified_files(self):
        files = self.uut.load_modified_files()
        files_list = list(files)

        self.assertIsInstance(files, Generator)
        self.assertTrue(all(isinstance(f, FileDatapoint) for f in files_list))
        self.assertEqual(len(files_list), 4)

        # Each commit in the fixture touches exactly one file
        self.assertListEqual(
            [f.file.commit_hash for f in files_list],
            [
                self.commit_0.hexsha,
                self.commit_1.hexsha,
                self.commit_2.hexsha,
                self.commit_3.hexsha,
            ],
        )
        self.assertListEqual(
            [f.file.path for f in files_list],
            ["file.py", "file.py", "file.py", "file2.py"],
        )
        self.assertTrue(all(f.extension == "py" for f in files_list))
        self.assertTrue(all(f.file.id is None for f in files_list))

        # First commit: file.py created with 4 lines
        self.assertEqual(files_list[0].nloc, 4)
        self.assertEqual(files_list[0].timestamp, self.commit_0.committed_datetime)

        # Second commit: two lines altered, file.py still has 4 lines
        self.assertEqual(files_list[1].nloc, 4)
        self.assertEqual(files_list[1].timestamp, self.commit_1.committed_datetime)

        # Third commit: last line deleted, file.py left with 3 lines
        self.assertEqual(files_list[2].nloc, 3)
        self.assertEqual(files_list[2].timestamp, self.commit_2.committed_datetime)

        # Fourth commit: new file2.py with 3 lines
        self.assertEqual(files_list[3].nloc, 3)
        self.assertEqual(files_list[3].timestamp, self.commit_3.committed_datetime)

    def test_load_modified_files_parallel(self):
        files = self.uut.load_modified_files_parallel(num_processes=2, chunk_size=1)
        files_list = list(files)

        self.assertIsInstance(files, Generator)
        self.assertTrue(all(isinstance(f, FileDatapoint) for f in files_list))
        self.assertEqual(len(files_list), 4)

        # Each commit in the fixture touches exactly one file
        self.assertListEqual(
            [f.file.commit_hash for f in files_list],
            [
                self.commit_0.hexsha,
                self.commit_1.hexsha,
                self.commit_2.hexsha,
                self.commit_3.hexsha,
            ],
        )
        self.assertListEqual(
            [f.file.path for f in files_list],
            ["file.py", "file.py", "file.py", "file2.py"],
        )
        self.assertTrue(all(f.extension == "py" for f in files_list))
        self.assertTrue(all(f.file.id is None for f in files_list))

        # First commit: file.py created with 4 lines
        self.assertEqual(files_list[0].nloc, 4)
        self.assertEqual(files_list[0].timestamp, self.commit_0.committed_datetime)

        # Second commit: two lines altered, file.py still has 4 lines
        self.assertEqual(files_list[1].nloc, 4)
        self.assertEqual(files_list[1].timestamp, self.commit_1.committed_datetime)

        # Third commit: last line deleted, file.py left with 3 lines
        self.assertEqual(files_list[2].nloc, 3)
        self.assertEqual(files_list[2].timestamp, self.commit_2.committed_datetime)

        # Fourth commit: new file2.py with 3 lines
        self.assertEqual(files_list[3].nloc, 3)
        self.assertEqual(files_list[3].timestamp, self.commit_3.committed_datetime)

    def test_load_modified_files_from_commit(self):
        files = list(self.uut.load_modified_files(_from=self.commit_1.hexsha))
        hashes = [f.file.commit_hash for f in files]
        self.assertListEqual(hashes, [self.commit_2.hexsha, self.commit_3.hexsha])


class _UnparsableFile:
    """A file whose metrics lizard cannot compute.

    lizard parses the source to produce them and raises on constructs its
    parser cannot follow, which has happened on TypeScript sources.
    """

    filename = "broken.ts"

    @property
    def nloc(self):
        raise IndexError("list index out of range")

    @property
    def methods(self):
        raise IndexError("list index out of range")

    @property
    def complexity(self):
        raise IndexError("list index out of range")

    @property
    def token_count(self):
        raise IndexError("list index out of range")


class _MeasurableFile:
    filename = "fine.py"
    nloc = 12
    methods = ["a", "b"]
    complexity = 3.5
    token_count = 40


class TestSourceMetrics(unittest.TestCase):
    def test_reads_the_metrics_of_a_file_lizard_can_parse(self):
        self.assertEqual((12, 2, 3.5, 40), _source_metrics(_MeasurableFile(), "a" * 40))

    def test_a_file_lizard_cannot_parse_has_unknown_metrics(self):
        # Recorded as unknown rather than ending the mine, and rather than
        # being written as zero, which would read as an empty file
        with mock.patch.object(gitloader, "LOG") as log:
            metrics = _source_metrics(_UnparsableFile(), "a" * 40)

        self.assertEqual((None, None, None, None), metrics)
        self.assertIn("broken.ts", log.warning.call_args.args)


class _Submodule:
    """A submodule change, which carries no source and so no line count."""

    filename = "some-submodule"
    nloc = None


class TestCarriesSource(unittest.TestCase):
    def test_a_submodule_carries_no_source(self):
        self.assertFalse(_carries_source(_Submodule()))

    def test_a_measurable_file_carries_source(self):
        self.assertTrue(_carries_source(_MeasurableFile()))

    def test_a_file_lizard_cannot_parse_still_carries_source(self):
        # Only a submodule has no source. A parser failure means the metrics
        # are unknown, not that the file is absent, so its lines are blamed
        self.assertTrue(_carries_source(_UnparsableFile()))
