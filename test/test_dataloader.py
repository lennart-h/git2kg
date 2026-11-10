import unittest
from datetime import datetime, timezone

from git2kg.dataloading.dataloader import Dataloader
from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    Datapoint,
    DatapointList,
    GithubCommitDatapoint,
)
from git2kg.persistence import PersistenceManager


def _commit(hash_: str, committed: datetime) -> CommitDatapoint:
    """A CommitDatapoint carrying only what the resume logic reads."""
    actor = ActorInfo(name="Ada", email="ada@example.org")
    return CommitDatapoint(
        commit=CommitInfo(hash=hash_),
        author=actor,
        committer=actor,
        author_date=committed,
        author_timezone=0,
        committer_date=committed,
        committer_timezone=0,
        n_modified_files=0,
        message="",
        is_merge=False,
        is_in_main_branch=True,
        is_bugfix=False,
        n_modified_documentation=0,
        insertions=0,
        deletions=0,
        message_polarity=0.0,
        message_subjectivity=0.0,
    )


def _github_commit(hash_: str, committed: datetime) -> GithubCommitDatapoint:
    """What GitHub reports about one commit, keyed by its committer date."""
    return GithubCommitDatapoint(commit_hash=hash_, committer_date=committed)


class _RecordingPersistence(PersistenceManager):
    """Hands back a fixed set of datapoints and records what is appended."""

    def __init__(self, stored: list[Datapoint]):
        self.stored = stored
        self.appended: list[Datapoint] = []

    def load(self, datapoint_class: type[Datapoint], processes=1) -> DatapointList:
        return DatapointList(list(self.stored), dp_type=datapoint_class)

    def save(self, datapoints: DatapointList):
        self.stored = list(datapoints)

    def append(self, datapoints: DatapointList):
        self.appended.extend(datapoints)


class TestCheckAndLoad(unittest.TestCase):
    """`Dataloader._check_and_load` resumes from the newest datapoint it has.

    Every loader answers that bound inclusively, so the datapoints sitting on
    it arrive again and must not be stored or returned twice.
    """

    def setUp(self):
        self.old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.new = datetime(2026, 2, 1, tzinfo=timezone.utc)
        # The one the previous run stopped on, and the one that follows it
        self.boundary = _commit("a" * 40, self.old)
        self.following = _commit("b" * 40, self.new)
        self.uut = Dataloader.__new__(Dataloader)
        self.uut.processes = 1

    def _load(self, stored, yields):
        persistence = _RecordingPersistence(stored)
        self.uut.persistence_manager = persistence
        loaded = self.uut._check_and_load(
            CommitDatapoint, lambda _from: iter(yields), chunk_size=10
        )
        return loaded, persistence

    def test_the_datapoint_on_the_boundary_is_not_stored_twice(self):
        loaded, persistence = self._load(
            stored=[self.boundary], yields=[self.boundary, self.following]
        )

        hashes = [dp.commit.hash for dp in loaded]
        self.assertEqual(hashes, ["a" * 40, "b" * 40])
        self.assertEqual([dp.commit.hash for dp in persistence.appended], ["b" * 40])

    def test_several_datapoints_share_the_boundary(self):
        """`GithubCommitDatapoint` resumes from a timestamp, which several
        commits can carry, so all of them come back and all must be skipped."""
        first = _github_commit("a" * 40, self.old)
        sibling = _github_commit("c" * 40, self.old)
        following = _github_commit("b" * 40, self.new)
        persistence = _RecordingPersistence([first, sibling])
        self.uut.persistence_manager = persistence

        loaded = self.uut._check_and_load(
            GithubCommitDatapoint,
            lambda _from: iter([first, sibling, following]),
            chunk_size=10,
        )

        self.assertEqual(
            [dp.commit_hash for dp in loaded], ["a" * 40, "c" * 40, "b" * 40]
        )
        self.assertEqual([dp.commit_hash for dp in persistence.appended], ["b" * 40])

    def test_a_first_run_stores_everything(self):
        loaded, persistence = self._load(
            stored=[], yields=[self.boundary, self.following]
        )

        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(persistence.appended), 2)

    def test_a_changed_datapoint_on_the_boundary_is_kept(self):
        """An issue is mined again whenever it was updated, so a datapoint on
        the boundary that differs from the stored one is a new version of it
        rather than a duplicate."""
        changed = _commit("a" * 40, self.old)
        changed.message = "amended"
        loaded, _ = self._load(stored=[self.boundary], yields=[changed])

        self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
