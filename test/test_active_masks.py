import unittest

import pandas as pd

from git2kg.active_masks import compute_active_masks
from git2kg.datapoint import (
    ActorInfo,
    CommitInfo,
    DatapointList,
    FileDatapoint,
    FileInfo,
)
from git2kg.disambiguation import (
    disambiguated_author_mapping,
)
from git2kg.gitdata import _load_gitdata


def _ids(actors=(), issues=(), commits=(), files=()) -> dict[str, pd.Series]:
    """Builds an id universe per node type. Tests pass only the ids of the node
    type they assert on, leaving the others empty.
    """
    return {
        "actor": pd.Series(list(actors), name=ActorInfo.ID_COLUMN, dtype="int64"),
        "issue": pd.Series(list(issues), name="issue_number", dtype="int64"),
        "commit": pd.Series(list(commits), name="commit_id", dtype="int64"),
        "file": pd.Series(list(files), name="file_id", dtype="int64"),
    }


def _bounds(*timestamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC") for ts in timestamps])


def _disambiguate(gitdata) -> dict[str, int]:
    """Assigns actor ids to `gitdata` and maps alias name -> actor id.

    Which id an actor is given is not stable across runs, so tests address
    actors by alias and build their own id universe from the result.
    """
    mapping = disambiguated_author_mapping(gitdata)
    gitdata.add_actor_ids(mapping)
    return {row.alias_name: int(row.actor_id) for row in mapping.itertuples()}


class TestActiveMasks(unittest.TestCase):
    def test_returns_one_dict_per_timeslice_with_every_node_type(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        by_alias = _disambiguate(gitdata)
        gitdata.add_file_ids()
        gitdata.add_commit_ids()
        bounds = _bounds("2008-01-01", "2016-01-01", "2025-01-01")

        masks = compute_active_masks(
            gitdata,
            bounds,
            _ids(
                actors=sorted(set(by_alias.values())),
                issues=[1, 2],
                commits=[0],
                files=[0],
            ),
        )

        self.assertEqual(len(bounds) - 1, len(masks))
        for mask in masks:
            self.assertEqual({"actor", "issue", "commit", "file"}, set(mask))
            self.assertEqual(bool, mask["actor"].dtype)
        self.assertEqual((4,), masks[0]["actor"].shape)

    def test_actors_exist_from_their_first_appearance_onwards(self):
        # Earliest datapoint each fixture actor shows up in:
        #   mariecurie, alanturing:  2008-01-14T04:33:35
        #   Konny Zuse:              2009-02-15T05:34:36
        #   rosafranklin1920:        2009-03-15T08:59:36
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        by_alias = _disambiguate(gitdata)
        actors = [
            by_alias["rosafranklin1920"],
            by_alias["mariecurie"],
            by_alias["alanturing"],
            by_alias["Konny Zuse"],
        ]

        bounds = _bounds("2008-01-01", "2008-06-01", "2009-03-01", "2009-06-01")
        masks = compute_active_masks(gitdata, bounds, _ids(actors=actors))

        self.assertListEqual([False, True, True, False], list(masks[0]["actor"]))
        self.assertListEqual([False, True, True, True], list(masks[1]["actor"]))
        self.assertListEqual([True, True, True, True], list(masks[2]["actor"]))

    def test_actors_stay_active_while_dormant(self):
        # Both actors appear only on 2008-01-14, but the last slice runs into
        # 2016, well past any datapoint of theirs.
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        by_alias = _disambiguate(gitdata)
        actors = [by_alias["mariecurie"], by_alias["alanturing"]]

        bounds = _bounds("2008-01-01", "2009-01-01", "2016-01-01")
        masks = compute_active_masks(gitdata, bounds, _ids(actors=actors))

        self.assertListEqual([True, True], list(masks[0]["actor"]))
        self.assertListEqual([True, True], list(masks[1]["actor"]))

    def test_issues_exist_from_creation_not_from_last_update(self):
        # Issue 1: created and last updated 2008-01-14T04:33:35
        # Issue 2: created 2009-02-15T05:34:36, last updated 2009-03-15T05:34:36
        # Issue 3 has no IssueDatapoint of its own in the fixture - it only ever
        # appears as the target of a "merged" event - so it never exists.
        gitdata = _load_gitdata("test/resources", "gitdata-test")

        bounds = _bounds("2008-01-01", "2009-01-01", "2009-03-01", "2009-06-01")
        masks = compute_active_masks(gitdata, bounds, _ids(issues=[1, 2, 3]))

        self.assertListEqual([True, False, False], list(masks[0]["issue"]))
        # Issue 2 was created within this slice but not updated until the next
        # one, so it exists here only because appearance_key reads created_at.
        self.assertListEqual([True, True, False], list(masks[1]["issue"]))
        self.assertListEqual([True, True, False], list(masks[2]["issue"]))

    def test_commits_exist_from_their_committer_date_onwards(self):
        # Committer dates in the fixture (2024-11-29):
        #   0=10:15:17  1=10:18:29  2=10:20:46  3=10:23:10  4=10:25:25  5=10:27:14
        # Commit 99 is not in the fixture at all and so never exists.
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_commit_ids()

        bounds = _bounds(
            "2024-11-29T10:15:00",
            "2024-11-29T10:19:00",
            "2024-11-29T10:24:00",
            "2024-11-29T10:28:00",
        )
        masks = compute_active_masks(
            gitdata, bounds, _ids(commits=[0, 1, 2, 3, 4, 5, 99])
        )

        expected = [
            [True, True, False, False, False, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False],
        ]
        for i, row in enumerate(expected):
            self.assertListEqual(row, list(masks[i]["commit"]))

    def test_files_stop_existing_after_the_slice_that_deletes_them(self):
        # Fixture files (see gitdata-test-modified_files.csv):
        #   file 0 (radiation.txt): first touched 10:15:17, never deleted
        #   file 1 (dns.txt): first touched 10:18:29
        #   file 2 (machine.py): first touched 10:27:14, never deleted
        # A synthetic deletion of file 1 at 10:24:00 is added to exercise the
        # deletion path, which the fixture doesn't otherwise cover.
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_file_ids()
        delete_dp = FileDatapoint(
            file=FileInfo(commit_hash="deleted-dns", path="dns.txt", id=1),
            commit=CommitInfo(hash="deleted-dns", id=0),
            timestamp=pd.to_datetime("2024-11-29T10:24:00.000Z", utc=True),
            nloc=0,
            extension="txt",
            n_methods=0,
            complexity=0.0,
            n_token=0,
        )
        gitdata.modified_files = gitdata.modified_files + DatapointList(
            [delete_dp], FileDatapoint
        )
        gitdata.file_registry._deleted_at = {1: "deleted-dns"}

        bounds = _bounds(
            "2024-11-29T10:15:00",
            "2024-11-29T10:20:00",
            "2024-11-29T10:23:30",
            "2024-11-29T10:24:30",
            "2024-11-29T10:30:00",
        )
        masks = compute_active_masks(gitdata, bounds, _ids(files=[0, 1, 2]))

        self.assertListEqual([True, True, False], list(masks[0]["file"]))
        self.assertListEqual([True, True, False], list(masks[1]["file"]))
        # File 1 is deleted within this slice and still exists in it.
        self.assertListEqual([True, True, False], list(masks[2]["file"]))
        # File 1 was deleted in the previous slice; file 2 has just been created.
        self.assertListEqual([True, False, True], list(masks[3]["file"]))


if __name__ == "__main__":
    unittest.main()
