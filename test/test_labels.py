import unittest
from unittest import mock

import numpy as np
import pandas as pd

from git2kg.core import Git2KG
from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    DatapointList,
    EditDatapoint,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    IssueInfo,
    ReviewDatapoint,
)
from git2kg.gitdata import GitData, _load_gitdata
from git2kg.labels import (
    DeveloperLeaverLabelExtractor,
    FileModifiedNextSliceLabelExtractor,
    IssueClosedNextSliceLabelExtractor,
)
from test.test_gitdata import _commit_datapoint


def _gitdata_with_commits(*commits: CommitDatapoint) -> GitData:
    """Builds a GitData holding only the given commits"""
    return GitData(
        DatapointList(list(commits), CommitDatapoint),
        DatapointList([], EditDatapoint),
        DatapointList([], IssueDatapoint),
        DatapointList([], IssueCommentDatapoint),
        DatapointList([], IssueEventDatapoint),
        DatapointList([], ReviewDatapoint),
    )


def _gitdata_with_events(*events: IssueEventDatapoint) -> GitData:
    """Builds a GitData holding only the given issue events"""
    return GitData(
        DatapointList([], CommitDatapoint),
        DatapointList([], EditDatapoint),
        DatapointList([], IssueDatapoint),
        DatapointList([], IssueCommentDatapoint),
        DatapointList(list(events), IssueEventDatapoint),
        DatapointList([], ReviewDatapoint),
    )


def _closed_event(issue_number: int, created_at: str) -> IssueEventDatapoint:
    """Builds a "closed" event for an issue, at a UTC timestamp"""
    return IssueEventDatapoint(
        event_type="closed",
        issue=IssueInfo(number=issue_number),
        requester=None,
        req_reviewer=None,
        created_at=pd.Timestamp(created_at, tz="UTC"),
        assignee=None,
        assigner=None,
        actor=ActorInfo(name="alanturing", github_id=1912, type="User"),
    )


class TestLabelExtractors(unittest.TestCase):
    def setUp(self):
        actor_mapping = pd.DataFrame(
            {
                ActorInfo.ID_COLUMN: list(range(4)),
                "alias_name": [
                    ["Marie Curie", "mariecurie"],
                    ["Rosalind Franklin", "Rosa Franklin", "rosafranklin1920"],
                    ["Alan Turing", "Alan Turing", "alanturing"],
                    ["Konrad Zuse", "Konny Zuse"],
                ],
                "alias_email": [
                    ["marie.curie@radioactive.pl", "marie.curie+github@radioactive.pl"],
                    ["franklin@kcl.ac.uk", "rosa@frankling.uk", "franklin@kcl.ac.uk"],
                    ["turingalan1912@computable.uk", "alanturing@lichess.org", None],
                    ["kzuse@z3.de", "zuse@zusekg.de"],
                ],
                ActorInfo.GITHUB_ID_COLUMN: [
                    [None, 1867],
                    [None, None, 1920],
                    [None, None, 1912],
                    [None, 1910],
                ],
            }
        )
        self.actor_mapping = actor_mapping.explode(
            ["alias_name", "alias_email", ActorInfo.GITHUB_ID_COLUMN]
        ).drop_duplicates()
        self.gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.gitdata.add_actor_ids(self.actor_mapping)
        self.git2kg = Git2KG(self.gitdata, "YE")

    def test_IssueClosedNextSliceLabelExtractor(self):
        # Fixture:
        # issue 1 is closed at 2008-01-14T04:50:35;
        # issue 2 is closed at 2009-04-15T10:34:36;

        gitdata = _load_gitdata("test/resources", "gitdata-test")
        issues = pd.Series([1, 2], name="issue_number")

        bounds = pd.DatetimeIndex(
            [
                pd.to_datetime("2008-01-13T04:40:35.000Z", utc=True),
                pd.to_datetime("2008-01-13T04:50:35.000Z", utc=True),
                pd.to_datetime("2009-04-15T10:00:00.000Z", utc=True),
                pd.to_datetime("2009-04-15T10:20:00.000Z", utc=True),  # slice 1
                pd.to_datetime(
                    "2009-04-15T10:40:00.000Z", utc=True
                ),  # slice 2: issue 2 closed within this slice
                pd.to_datetime(
                    "2009-04-15T11:00:00.000Z", utc=True
                ),  # slice 3: last slice, always False
            ]
        )
        extractor = IssueClosedNextSliceLabelExtractor(gitdata, [], bounds, issues)
        labels = extractor.extract_labels_for_datetime_slice(
            datetime_slice=(
                pd.to_datetime("2008-01-14T04:50:00.000Z", utc=True),
                pd.to_datetime("2008-01-14T05:00:00.000Z", utc=True),
            )
        )
        np.testing.assert_array_equal(
            np.array([1.0, 0.0]), labels["closed_time"].to_numpy()
        )
        # extract_labels_for_datetime_slice itself is unchanged - still "is
        # this issue closed within the given window" - verify it directly.
        labels = extractor.extract_labels_for_datetime_slice(
            datetime_slice=(
                pd.to_datetime("2009-04-15T10:30:17.000Z", utc=True),
                pd.to_datetime("2009-04-15T10:40:17.000Z", utc=True),
            )
        )
        np.testing.assert_array_equal(
            np.array([0.0, 1.0]), labels["closed_time"].to_numpy()
        )

        labels = extractor.extract_labels_for_datetime_slice(
            datetime_slice=(
                pd.to_datetime("2009-04-15T10:40:17.000Z", utc=True),
                pd.to_datetime("2009-04-15T10:50:17.000Z", utc=True),
            )
        )
        np.testing.assert_array_equal(
            np.array([0.0, 0.0]), labels["closed_time"].to_numpy()
        )

        # extract_label_sequence's next-slice lookahead wiring.
        label_seq = extractor.extract_label_sequence()
        self.assertEqual(len(bounds) - 1, len(label_seq))
        self.assertListEqual([True, False], [bool(v) for v in label_seq[0]])
        self.assertListEqual([False, False], [bool(v) for v in label_seq[1]])
        # Slice 1's label looks ahead to slice 2, where issue 2 is closed.
        self.assertListEqual([False, True], [bool(v) for v in label_seq[2]])
        # Slice 2's label looks ahead to slice 3, where nothing closes.
        self.assertListEqual([False, False], [bool(v) for v in label_seq[3]])
        # Slice 3 is the last slice - no next slice to look ahead to, always False.
        self.assertListEqual([False, False], [bool(v) for v in label_seq[4]])

    def test_IssueClosedNextSliceLabelExtractor_with_reopened_issues(self):
        # An issue can be closed, reopened and closed again, so it can carry
        # several "closed" events. Every close is a label of its own, and the
        # label vector keeps one entry per issue regardless.
        #   issue 1: closed 2020-02-10, reopened, closed again 2020-04-10
        #   issue 2: closed twice within one month, 2020-02-15 and 2020-02-20
        #   issue 3: never closed
        gitdata = _gitdata_with_events(
            _closed_event(1, "2020-02-10"),
            _closed_event(1, "2020-04-10"),
            _closed_event(2, "2020-02-15"),
            _closed_event(2, "2020-02-20"),
        )
        issues = pd.Series([1, 2, 3], name="issue_number")
        bounds = pd.DatetimeIndex(
            [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-02-01", tz="UTC"),
                pd.Timestamp("2020-03-01", tz="UTC"),  # slice 1: both issues close
                pd.Timestamp("2020-04-01", tz="UTC"),  # slice 2: nothing
                pd.Timestamp("2020-05-01", tz="UTC"),  # slice 3: issue 1 closes again
            ]
        )

        label_seq = IssueClosedNextSliceLabelExtractor(
            gitdata, [], bounds, issues
        ).extract_label_sequence()

        # One entry per issue, in every snapshot, however often an issue closed.
        self.assertListEqual([3, 3, 3, 3], [len(labels) for labels in label_seq])
        # Slice 0 looks ahead to slice 1: issue 2's two closes count once.
        self.assertListEqual([True, True, False], [bool(v) for v in label_seq[0]])
        self.assertListEqual([False, False, False], [bool(v) for v in label_seq[1]])
        # Slice 2 looks ahead to slice 3: issue 1's second close is a label again.
        self.assertListEqual([True, False, False], [bool(v) for v in label_seq[2]])
        self.assertListEqual([False, False, False], [bool(v) for v in label_seq[3]])

    def test_DeveloperLeaverLabelExtractor(self):
        # Fixture commits, by committer date on 2024-11-29:
        #   10:15:17 Marie Curie      10:18:29 Rosalind Franklin
        #   10:20:46 Alan Turing      10:23:10 Konrad Zuse
        #   10:25:25 Alan Turing      10:27:14 Rosa Franklin
        # Actor ids are assigned by Git2KG, so the universe is built from
        # aliases here to keep the expected vectors readable.
        mapping = self.git2kg.actor_mapping
        by_alias = {row.alias_name: int(row.actor_id) for row in mapping.itertuples()}
        actors = pd.Series(
            [
                by_alias["Marie Curie"],
                by_alias["Rosalind Franklin"],
                by_alias["Alan Turing"],
                by_alias["Konrad Zuse"],
            ],
            name=ActorInfo.ID_COLUMN,
        )
        bounds = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-11-29T10:15:00", tz="UTC"),
                pd.Timestamp(
                    "2024-11-29T10:19:00", tz="UTC"
                ),  # slice 1: Marie, Rosalind
                pd.Timestamp("2024-11-29T10:24:00", tz="UTC"),  # slice 2: Alan, Konrad
                pd.Timestamp(
                    "2024-11-29T10:28:00", tz="UTC"
                ),  # slice 3: Alan, Rosalind
            ]
        )

        # The extractor compares commit timestamps against the timeindex and
        # never consults the timeslices.
        extractor = DeveloperLeaverLabelExtractor(self.gitdata, [], bounds, actors)
        label_seq = extractor.extract_label_sequence()
        self.assertEqual(len(bounds) - 1, len(label_seq))

        # Marie and Rosalind commit in slice 1 and not in slice 2. Rosalind
        # returning in slice 3 does not change that: the horizon is one slice.
        np.testing.assert_array_equal([1.0, 1.0, 0.0, 0.0], label_seq[0])
        # Konrad leaves; Alan stays.
        np.testing.assert_array_equal([0.0, 0.0, 0.0, 1.0], label_seq[1])
        # Last slice has no next one to look ahead to.
        np.testing.assert_array_equal([0.0, 0.0, 0.0, 0.0], label_seq[2])

    def test_DeveloperLeaverLabelExtractor_counts_committers_as_developers(self):
        # Alice authors a change that Bob commits. Both are developers in
        # slice 1; neither commits in slice 2, so both are leavers.
        alice = ActorInfo(name="Alice", email="alice@example.com", id=0)
        bob = ActorInfo(name="Bob", email="bob@example.com", id=1)
        commit = _commit_datapoint(
            CommitInfo(hash="a" * 40, id=0),
            author=alice,
            committer=bob,
            author_date=pd.Timestamp("2024-01-01", tz="UTC"),
            committer_date=pd.Timestamp("2024-02-01", tz="UTC"),
        )
        gitdata = _gitdata_with_commits(commit)
        actors = pd.Series([0, 1], name=ActorInfo.ID_COLUMN)
        bounds = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp(
                    "2024-02-15", tz="UTC"
                ),  # slice 1 holds the committer date
                pd.Timestamp("2024-03-15", tz="UTC"),
            ]
        )

        label_seq = DeveloperLeaverLabelExtractor(
            gitdata, [], bounds, actors
        ).extract_label_sequence()

        np.testing.assert_array_equal([1.0, 1.0], label_seq[0])
        np.testing.assert_array_equal([0.0, 0.0], label_seq[1])

    def test_FileModifiedNextSliceLabelExtractor(self):
        # Fixture files (see gitdata-test-modified_files.csv):
        #   file 0 (radiation.txt): touched 10:15:17 and 10:25:25
        #   file 1 (dns.txt): touched 10:18:29, 10:20:46, 10:23:10, 10:27:14
        #   file 2 (machine.py): touched 10:27:14
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_file_ids()  # resolve the fixture's file ids

        bounds = pd.DatetimeIndex(
            [
                pd.to_datetime("2024-11-29T10:15:00.000Z", utc=True),
                pd.to_datetime(
                    "2024-11-29T10:19:00.000Z", utc=True
                ),  # slice 1: [10:15,10:19)
                pd.to_datetime(
                    "2024-11-29T10:24:00.000Z", utc=True
                ),  # slice 2: [10:19,10:24) - dns touched here
                pd.to_datetime(
                    "2024-11-29T10:26:00.000Z", utc=True
                ),  # slice 3: [10:24,10:26) - radiation touched here
                pd.to_datetime(
                    "2024-11-29T10:30:00.000Z", utc=True
                ),  # slice 4: [10:26,10:30) - dns & machine touched here
            ]
        )
        files = pd.Series([0, 1, 2], name="file_id")

        # FileModifiedNextSliceLabelExtractor compares timestamps against
        # timeindex bounds directly and never consults timeslices.
        extractor = FileModifiedNextSliceLabelExtractor(gitdata, [], bounds, files)
        label_seq = extractor.extract_label_sequence()
        self.assertEqual(len(bounds) - 1, len(label_seq))

        # Slice 1's label looks ahead to slice 2, where only dns.txt (id 1) is touched.
        self.assertListEqual([False, True, False], [bool(v) for v in label_seq[0]])
        # Slice 2's label looks ahead to slice 3, where only radiation.txt (id 0) is touched.
        self.assertListEqual([True, False, False], [bool(v) for v in label_seq[1]])
        # Slice 3's label looks ahead to slice 4, where dns.txt and machine.py are touched.
        self.assertListEqual([False, True, True], [bool(v) for v in label_seq[2]])
        # Slice 4 is the last slice - no next slice to look ahead to, always False.
        self.assertListEqual([False, False, False], [bool(v) for v in label_seq[3]])


class TestDeveloperLeaverWindows(unittest.TestCase):
    """The windows decide how long someone counts as active, and how long they
    have to stay away to count as having left."""

    def setUp(self):
        self.git2kg = Git2KG(_load_gitdata("test/resources", "gitdata-test"), freq=None)
        self.gitdata = self.git2kg.gitdata
        mapping = self.git2kg.actor_mapping
        by_alias = {row.alias_name: int(row.actor_id) for row in mapping.itertuples()}
        self.actors = pd.Series(
            [
                by_alias["Marie Curie"],
                by_alias["Rosalind Franklin"],
                by_alias["Alan Turing"],
                by_alias["Konrad Zuse"],
            ],
            name=ActorInfo.ID_COLUMN,
        )
        # slice 0: Marie, Rosalind | slice 1: Alan, Konrad | slice 2: Alan, Rosalind
        self.bounds = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-11-29T10:15:00", tz="UTC"),
                pd.Timestamp("2024-11-29T10:19:00", tz="UTC"),
                pd.Timestamp("2024-11-29T10:24:00", tz="UTC"),
                pd.Timestamp("2024-11-29T10:28:00", tz="UTC"),
            ]
        )

    def _labels(self, **windows):
        return DeveloperLeaverLabelExtractor(
            self.gitdata, [], self.bounds, self.actors, **windows
        ).extract_label_sequence()

    def test_the_default_windows_look_one_slice_each_way(self):
        self.assertListEqual(
            [list(v) for v in self._labels()],
            [list(v) for v in self._labels(active_window=1, leave_window=1)],
        )

    def _labels_for(self, developers, **windows):
        """Labels for a made-up activity pattern, one set of ids per slice.

        The fixture spans three timeslices, too few to show a developer
        leaving and returning inside a window, so the pattern is given here.
        """
        bounds = pd.date_range("2024-01-01", periods=len(developers) + 1, freq="ME")
        with mock.patch(
            "git2kg.labels._developers_per_timeslice", return_value=developers
        ):
            return DeveloperLeaverLabelExtractor(
                self.gitdata, [], bounds, self.actors, **windows
            ).extract_label_sequence()

    def test_a_longer_leave_window_demands_a_longer_absence(self):
        marie, rosalind = int(self.actors[0]), int(self.actors[1])
        # Rosalind is away for one slice and comes back; Marie never returns
        pattern = [{marie, rosalind}, set(), {rosalind}, set(), set()]

        one = self._labels_for(pattern, leave_window=1)
        two = self._labels_for(pattern, leave_window=2)

        # One slice of absence makes her a leaver
        self.assertEqual(1.0, one[0][1])
        # Two does not, because she is back within the window
        self.assertEqual(0.0, two[0][1])
        # Marie leaves either way
        self.assertEqual(1.0, one[0][0])
        self.assertEqual(1.0, two[0][0])

    def test_a_longer_leave_window_shortens_the_usable_sequence(self):
        labels = self._labels(leave_window=2)

        self.assertEqual(3, len(labels))
        # The last two have no full window to look ahead to
        self.assertListEqual([0.0] * 4, list(labels[1]))
        self.assertListEqual([0.0] * 4, list(labels[2]))

    def test_a_developer_is_labelled_only_where_they_were_active(self):
        marie, rosalind = int(self.actors[0]), int(self.actors[1])
        # Marie commits once and never returns
        pattern = [{marie, rosalind}, {rosalind}, {rosalind}, {rosalind}, {rosalind}]

        labels = self._labels_for(pattern, active_window=3, leave_window=1)

        # The departure is dated to the one timeslice she is seen in, not to
        # every timeslice the lookback still reaches it from
        self.assertListEqual([1.0, 0.0, 0.0, 0.0], [v[0] for v in labels[:4]])

    def test_passing_through_once_is_not_a_contributor_leaving(self):
        marie, rosalind = int(self.actors[0]), int(self.actors[1])
        pattern = [{marie, rosalind}, {rosalind}, {rosalind}, {rosalind}, {rosalind}]

        lenient = self._labels_for(pattern, active_window=3, leave_window=1)
        strict = self._labels_for(
            pattern, active_window=3, min_active_slices=2, leave_window=1
        )

        self.assertEqual(1.0, lenient[0][0])
        self.assertEqual(0.0, strict[0][0])

    def test_an_established_contributor_still_counts_as_leaving(self):
        marie, rosalind = int(self.actors[0]), int(self.actors[1])
        # Marie is active in two consecutive timeslices, then gone
        pattern = [{marie}, {marie, rosalind}, {rosalind}, {rosalind}, {rosalind}]

        labels = self._labels_for(
            pattern, active_window=3, min_active_slices=2, leave_window=1
        )

        self.assertListEqual([0.0, 1.0, 0.0, 0.0], [v[0] for v in labels[:4]])

    def test_a_threshold_that_cannot_be_met_is_refused(self):
        with self.assertRaises(ValueError):
            self._labels(active_window=2, min_active_slices=3)

    def test_lookahead_reports_what_a_splitter_has_to_embargo(self):
        extractor = DeveloperLeaverLabelExtractor(
            self.gitdata, [], self.bounds, self.actors, leave_window=4
        )

        self.assertEqual(4, extractor.lookahead)

    def test_a_window_shorter_than_a_timeslice_is_refused(self):
        with self.assertRaises(ValueError):
            self._labels(leave_window=0)
