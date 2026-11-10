import unittest

import pandas as pd

from git2kg.gitdata import _load_gitdata
from git2kg.timeslicing import TimesliceIterator


def _filtered(gitdata, dp_name, lower, upper):
    """The datapoints of a list inside a window, found the obvious way."""
    dp_list = gitdata.dp_lists[dp_name]
    time_of = dp_list.dp_type.appearance_key()
    return [dp for dp in dp_list if lower <= time_of(dp) < upper]


class TestTimeslicing(unittest.TestCase):
    def setUp(self):
        self.gitdata = _load_gitdata("test/resources", "gitdata-test")
        # Year starts, so the first bound precedes the fixture's earliest
        # datapoint (an issue created 2008-01-14) and nothing falls outside
        self.timeindex = pd.date_range("2007-01-01", "2026-01-01", freq="YS", tz="UTC")

    def test_every_slice_holds_what_falls_into_its_window(self):
        timeslices = TimesliceIterator(self.gitdata, self.timeindex)

        for i, timeslice in enumerate(timeslices):
            lower, upper = self.timeindex[i], self.timeindex[i + 1]
            for dp_name in self.gitdata.dp_lists:
                with self.subTest(slice=i, datapoints=dp_name):
                    self.assertListEqual(
                        _filtered(self.gitdata, dp_name, lower, upper),
                        getattr(timeslice, dp_name).datapoints,
                    )

    def test_a_datapoint_lands_in_exactly_one_slice(self):
        timeslices = list(TimesliceIterator(self.gitdata, self.timeindex))

        for dp_name, dp_list in self.gitdata.dp_lists.items():
            with self.subTest(dp_name):
                sliced = sum(len(getattr(ts, dp_name)) for ts in timeslices)
                # Everything in the fixture falls inside the index
                self.assertEqual(len(dp_list), sliced)

    def test_datapoints_outside_the_index_are_left_out(self):
        # An index starting after the fixture's first commits
        late = pd.date_range("2024-11-29T10:20:00", periods=3, freq="h", tz="UTC")

        timeslices = list(TimesliceIterator(self.gitdata, late))

        kept = sum(len(ts.commits) for ts in timeslices)
        self.assertLess(kept, len(self.gitdata.commits))
        self.assertGreater(kept, 0)

    def test_the_sequence_reports_one_slice_per_interval(self):
        timeslices = TimesliceIterator(self.gitdata, self.timeindex)

        self.assertEqual(len(self.timeindex) - 1, len(timeslices))
        self.assertEqual(len(timeslices), len(list(timeslices)))

    def test_the_partition_is_computed_once(self):
        timeslices = TimesliceIterator(self.gitdata, self.timeindex)

        # Indexing twice must not partition the data again
        self.assertIs(timeslices[0], timeslices[0])

    def test_the_disambiguation_flags_carry_over(self):
        self.gitdata.actors_disambiguated = True
        self.gitdata.files_disambiguated = True

        timeslice = TimesliceIterator(self.gitdata, self.timeindex)[0]

        self.assertTrue(timeslice.actors_disambiguated)
        self.assertTrue(timeslice.files_disambiguated)

    def test_the_file_registry_carries_over(self):
        # The commit and file id universes are read from it, so a slice
        # without one describes a graph with no commit and no file nodes
        timeslices = TimesliceIterator(self.gitdata, self.timeindex)

        for i, timeslice in enumerate(timeslices):
            with self.subTest(slice=i):
                self.assertIs(self.gitdata.file_registry, timeslice.file_registry)


if __name__ == "__main__":
    unittest.main()


class TestEdgesWithMissingEndpoints(unittest.TestCase):
    """An interaction can name an actor that is no node, since the actor of a
    deleted account is recorded as missing. Such an edge has nothing to
    connect and is left out rather than ending snapshot creation."""

    def setUp(self):
        from datetime import datetime, timezone

        from git2kg.datapoint import ActorInfo, IssueInfo

        self.when = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.actor = ActorInfo(name="marie", id=3)
        self.issue = IssueInfo(7)

    def _edges(self):
        return [
            (self.actor, self.issue, self.when),
            (None, self.issue, self.when),  # the account behind it is gone
            (self.actor, None, self.when),
        ]

    def test_collapsed_edges_keep_only_connected_interactions(self):
        from git2kg.snapshot_compute import _collapse_edges

        edges = _collapse_edges(self._edges())

        self.assertListEqual([3], list(edges["u"]))
        self.assertListEqual([7], list(edges["v"]))
        self.assertListEqual([1], list(edges["weight"]))

    def test_stacked_edges_keep_only_connected_interactions(self):
        from git2kg.snapshot_compute import _stack_edges

        edges = _stack_edges(self._edges())

        self.assertEqual(1, len(edges))
        self.assertListEqual([3], list(edges["u"]))
        self.assertListEqual([7], list(edges["v"]))
