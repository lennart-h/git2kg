import unittest

import numpy as np
import pandas as pd

import git2kg.features as features
from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    DatapointList,
    EditDatapoint,
    FileDatapoint,
    FileInfo,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    ReviewDatapoint,
)
from git2kg.gitdata import FIRST_SEEN, GitData, _load_gitdata
from git2kg.utils import iso8601_to_utc
from test.test_gitdata import _commit_datapoint
from test.test_labels import _gitdata_with_commits


def _file_datapoint(file_id: int, path: str, timestamp: str) -> FileDatapoint:
    """Builds a FileDatapoint, defaulting everything the caller does not care about"""
    return FileDatapoint(
        file=FileInfo(commit_hash="a" * 40, path=path, id=file_id),
        commit=CommitInfo(hash="a" * 40, id=0),
        timestamp=pd.Timestamp(timestamp, tz="UTC"),
        nloc=0,
        extension=path.rsplit(".", 1)[-1],
        n_methods=0,
        complexity=0.0,
        n_token=0,
    )


def _measured_file_version(
    path: str,
    timestamp: str,
    nloc=None,
    n_methods=None,
    complexity=None,
    n_token=None,
    file_id: int = 0,
) -> FileDatapoint:
    """Builds one version of a file, with the source metrics the caller gives.

    Everything left out is None, which is what PyDriller reports for a
    modification it hands over without source.
    """
    return FileDatapoint(
        file=FileInfo(commit_hash="a" * 40, path=path, id=file_id),
        commit=CommitInfo(hash="a" * 40, id=0),
        timestamp=pd.Timestamp(timestamp, tz="UTC"),
        nloc=nloc,
        extension=path.rsplit(".", 1)[-1],
        n_methods=n_methods,
        complexity=complexity,
        n_token=n_token,
    )


def _gitdata_with_files(*files: FileDatapoint) -> GitData:
    """Builds a GitData holding only the given modified files"""
    return GitData(
        DatapointList([], CommitDatapoint),
        DatapointList([], EditDatapoint),
        DatapointList([], IssueDatapoint),
        DatapointList([], IssueCommentDatapoint),
        DatapointList([], IssueEventDatapoint),
        DatapointList([], ReviewDatapoint),
        modified_files=DatapointList(list(files), FileDatapoint),
    )


class TestFeatureExtractors(unittest.TestCase):
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

    # Actor features aggregate development commits only. The fixture's merge
    # commit (5709501) belongs to actor 1 and carries 2 insertions, 1 deletion
    # and the bugfix flag, so actor 1 contributes one commit, not two.

    def test_ActorTenureExtractor(self):
        # Actors 0 and 2 are first seen at 2008-01-14T04:33:35, 367 days
        # before the cut
        as_of = pd.Timestamp("2009-01-15T04:33:35", tz="UTC")
        feature = features.ActorTenureExtractor(
            features.feature_data(self.gitdata).as_of(as_of), as_of=as_of
        ).extract_feature()

        # Actors 1 and 3 appear after the cut, so they have no tenure yet
        self.assertListEqual([0, 2], list(feature[ActorInfo.ID_COLUMN]))
        self.assertListEqual([367 * 86400.0] * 2, list(feature["tenure"]))

    def test_ActorTenureExtractor_counts_from_any_role_not_the_first_commit(self):
        # Every actor of the fixture commits for the first time in 2024 but
        # shows up in the issue world in 2008/2009
        as_of = pd.Timestamp("2024-11-29T10:30:00", tz="UTC")
        feature = features.ActorTenureExtractor(
            features.feature_data(self.gitdata).as_of(as_of), as_of=as_of
        ).extract_feature()

        first_seen = self.gitdata.actor_appearances()
        self.assertListEqual([0, 1, 2, 3], list(feature[ActorInfo.ID_COLUMN]))
        self.assertListEqual(
            list((as_of - first_seen[FIRST_SEEN]).dt.total_seconds()),
            list(feature["tenure"]),
        )
        self.assertTrue((feature["tenure"] > 15 * 365 * 86400).any())

    def test_ActorTenureExtractor_without_as_of_counts_to_the_end_of_history(self):
        feature = features.ActorTenureExtractor(self.gitdata).extract_feature()

        # The last thing to happen in the fixture is the merge commit, at
        # 2024-11-29T10:27:14, 166 seconds before the cut used above
        self.assertListEqual([0, 1, 2, 3], list(feature[ActorInfo.ID_COLUMN]))
        self.assertListEqual([532590819.0, 493087664.0], list(feature["tenure"])[:2])

    def test_NumCommitsActivityExtractor(self):
        feature = features.NumCommitsActivityExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1, 1, 2, 1], list(feature["num_commits"]))

    def test_TotalLinesAddedExtractor(self):
        feature = features.TotalLinesAddedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([3, 2, 2, 2], list(feature["insertions"]))

    def test_TotalLinesRemovedExtractor(self):
        feature = features.TotalLinesRemovedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0, 0, 4, 2], list(feature["deletions"]))

    def test_UniqueFilesChangedExtractor(self):
        feature = features.UniqueFilesChangedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0, 1, 2, 1], list(feature["old_path"]))

    def test_CommitTimesExtractor(self):
        feature = features.CommitTimesExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1, 1, 1, 1], list(feature["author_days"]))

    def test_FirstCommitDateExtractor(self):
        feature = features.FirstCommitDateExtractor(self.gitdata).extract_feature()
        self.assertListEqual(
            [
                iso8601_to_utc("2024-11-29 11:15:17+01:00"),
                iso8601_to_utc("2024-11-29 11:18:29+01:00"),
                iso8601_to_utc("2024-11-29 11:20:46+01:00"),
                iso8601_to_utc("2024-11-29 11:23:10+01:00"),
            ],
            list(feature["author_date"]),
        )

    def test_LastCommitDateExtractor(self):
        feature = features.LastCommitDateExtractor(self.gitdata).extract_feature()
        self.assertListEqual(
            [
                iso8601_to_utc("2024-11-29 11:15:17+01:00"),
                # Actor 1's latest development commit; the merge is later
                iso8601_to_utc("2024-11-29 11:18:29+01:00"),
                iso8601_to_utc("2024-11-29 11:25:25+01:00"),
                iso8601_to_utc("2024-11-29 11:23:10+01:00"),
            ],
            list(feature["author_date"]),
        )

    def test_AverageCommitMessagePolarityExtractor(self):
        feature = features.AverageCommitMessagePolarityExtractor(
            self.gitdata
        ).extract_feature()
        np.testing.assert_array_almost_equal(
            [0.1, 0.1, 0.3, 0.3], list(feature["message_polarity"])
        )

    def test_AverageCommitMessageSubjectivityExtractor(self):
        feature = features.AverageCommitMessageSubjectivityExtractor(
            self.gitdata
        ).extract_feature()
        np.testing.assert_array_almost_equal(
            [0.0, 0.0, 0.1, 0.1], list(feature["message_subjectivity"])
        )

    def test_AverageCommitTimeExtractor(self):
        feature = features.AverageCommitTimeExtractor(self.gitdata).extract_feature()
        np.testing.assert_array_almost_equal(
            [40517.0, 40709.0, 40985.5, 40990.0], list(feature["commit_second_of_day"])
        )

    def test_NumIssueCommentsExtractor(self):
        feature = features.NumIssueCommentsExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1, 2, 1], list(feature["num_issue_comments"]))
        self.assertListEqual([1, 2, 3], list(feature["author_id"]))

    def test_NumIssuesCreatedExtractor(self):
        feature = features.NumIssuesCreatedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([2], list(feature["num_issues_created"]))
        self.assertListEqual([0], list(feature["author_id"]))

    def test_NumIssuesAssignedExtractor(self):
        # The fixture holds a single "assigned" event: issue 1 to Alan Turing
        # (actor 2) at 2008-01-14T04:35:35. Issue 2 carries two assignees but
        # no events recording them, so they do not count - assignments are
        # taken from the events, which say when they happened.
        feature = features.NumIssuesAssignedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1], list(feature["num_issues_assigned"]))
        self.assertListEqual([2], list(feature["assignee_id"]))

    def test_NumPullRequestsCreatedExtractor(self):
        feature = features.NumPullRequestsCreatedExtractor(
            self.gitdata
        ).extract_feature()
        self.assertListEqual([1], list(feature["num_pulls_created"]))
        self.assertListEqual([0], list(feature["author_id"]))

    def test_NumPullRequestsReviewedExtractor(self):
        feature = features.NumPullRequestsReviewedExtractor(
            self.gitdata
        ).extract_feature()
        self.assertListEqual([1], list(feature["num_pulls_reviewed"]))
        self.assertListEqual([1], list(feature["author_id"]))

    def test_NumMergedPullRequestsExtractor(self):
        feature = features.NumAuthoredMergedPullRequestsExtractor(
            self.gitdata
        ).extract_feature()
        self.assertListEqual([1], list(feature["num_pulls_merged"]))
        self.assertListEqual([0], list(feature["author_id"]))

    def test_CommitConsistencyExtractor(self):
        # Only actor 2 has two development commits, so only they have a time
        # between commits at all - actor 1's second commit is the merge
        feature = features.CommitConsistencyExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0], list(feature["time_diff"]))
        self.assertListEqual([2], list(feature["author_id"]))

    def test_NumDocumentationChangesExtractor(self):
        feature = features.NumDocumentationChangesExtractor(
            self.gitdata
        ).extract_feature()
        self.assertListEqual([0, 2, 1, 0], list(feature["n_modified_documentation"]))
        self.assertListEqual([0, 1, 2, 3], list(feature["author_id"]))

    def test_NumBugfixCommitsExtractor(self):
        feature = features.NumBugfixCommitsExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0, 0, 1, 1], list(feature["is_bugfix"]))
        self.assertListEqual([0, 1, 2, 3], list(feature["author_id"]))

    def test_BugfixCommitRatioExtractor(self):
        feature = features.BugfixCommitRatioExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0.0, 0.0, 0.5, 1.0], list(feature["is_bugfix"]))
        self.assertListEqual([0, 1, 2, 3], list(feature["author_id"]))

    def test_AvgCommentPolarityExtractor(self):
        feature = features.AvgCommentPolarityExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0.1, -0.1, 0.2], list(feature["sentiment_polarity"]))
        self.assertListEqual([1, 2, 3], list(feature["author_id"]))

    def test_AvgCommentSubjectivityExtractor(self):
        feature = features.AvgCommentSubjectivityExtractor(
            self.gitdata
        ).extract_feature()
        self.assertListEqual([0.3, 0.2, 0.01], list(feature["sentiment_subjectivity"]))
        self.assertListEqual([1, 2, 3], list(feature["author_id"]))

    def test_AvgIssueResponseTimeExtractor(self):
        feature = features.AvgIssueResponseTimeExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1800.0, 2431500.0], list(feature["response_time"]))
        self.assertListEqual([2, 3], list(feature["author_id"]))

    def test_AvgReviewTimeExtractor(self):
        feature = features.AvgReviewTimeExtractor(self.gitdata).extract_feature()
        self.assertListEqual([3600.0], list(feature["review_time"]))
        self.assertListEqual([1], list(feature["author_id"]))

    # Fixture issues:
    #   issue 1: created 2008-01-14T04:33:35, closed by an event 17 minutes
    #            later although its `closed_at` is unset
    #   issue 2: created 2009-02-15T05:34:36, closed and merged
    #            2009-04-15T10:34:36, reopened a day after that
    BEFORE_ISSUE_2_CLOSED = pd.Timestamp("2009-02-15", tz="UTC")
    AFTER_ISSUE_2_CLOSED = pd.Timestamp("2009-05-15", tz="UTC")
    # Points in time at which issue 2 is in each of its three states
    ISSUE_2_OPEN = pd.Timestamp("2009-03-15", tz="UTC")
    ISSUE_2_CLOSED = pd.Timestamp("2009-04-15T20:00:00", tz="UTC")
    ISSUE_2_REOPENED = pd.Timestamp("2009-05-15", tz="UTC")

    def _issue_data(self, as_of):
        return features.feature_data(self.gitdata).as_of(as_of)

    def test_TimeOpenExtractor_counts_only_time_open_so_far(self):
        feature = features.TimeOpenExtractor(
            self._issue_data(self.ISSUE_2_OPEN), as_of=self.ISSUE_2_OPEN
        ).extract_feature()

        self.assertListEqual([1, 2], list(feature["issue_number"]))
        self.assertListEqual([1020.0, 2399124.0], list(feature["time_open"]))

    def test_TimeOpenExtractor_reads_the_close_event_not_closed_at(self):
        # Issue 1 has no `closed_at`, so reading it would drop the issue
        # although it was open for 17 minutes
        issues = self.gitdata.get_as_dataframe("issues")
        self.assertTrue(pd.isna(issues.set_index("issue_number").loc[1, "closed_at"]))

        feature = features.TimeOpenExtractor(
            self._issue_data(self.ISSUE_2_CLOSED), as_of=self.ISSUE_2_CLOSED
        ).extract_feature()

        self.assertListEqual([1, 2], list(feature["issue_number"]))
        self.assertListEqual([1020.0, 5115600.0], list(feature["time_open"]))

    def test_TimeOpenExtractor_sums_every_interval_a_reopened_issue_was_open(self):
        feature = features.TimeOpenExtractor(
            self._issue_data(self.ISSUE_2_REOPENED), as_of=self.ISSUE_2_REOPENED
        ).extract_feature()

        # The time before the close plus the time since the reopen, rather
        # than the span from creation to the latest close
        self.assertListEqual([1020.0, 7583124.0], list(feature["time_open"]))

    def test_IsOpenExtractor(self):
        def is_open(as_of):
            feature = features.IsOpenExtractor(
                self._issue_data(as_of), as_of=as_of
            ).extract_feature()
            return list(feature["is_open"])

        self.assertListEqual([0, 1], is_open(self.ISSUE_2_OPEN))
        self.assertListEqual([0, 0], is_open(self.ISSUE_2_CLOSED))
        self.assertListEqual([0, 1], is_open(self.ISSUE_2_REOPENED))

    def test_IssueAgeExtractor(self):
        as_of = self.ISSUE_2_REOPENED
        feature = features.IssueAgeExtractor(
            self._issue_data(as_of), as_of=as_of
        ).extract_feature()

        self.assertListEqual([1, 2], list(feature["issue_number"]))
        self.assertListEqual([42060385.0, 7669524.0], list(feature["issue_age"]))

    def test_IsMergedExtractor(self):
        feature = features.IsMergedExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0, 1], list(feature["is_merged"]))
        self.assertListEqual([1, 2], list(feature["issue_number"]))

    def test_IsMergedExtractor_does_not_reveal_a_merge_before_it_happens(self):
        before = features.IsMergedExtractor(
            self.gitdata, as_of=self.BEFORE_ISSUE_2_CLOSED
        ).extract_feature()
        after = features.IsMergedExtractor(
            self.gitdata, as_of=self.AFTER_ISSUE_2_CLOSED
        ).extract_feature()

        self.assertListEqual([0, 0], list(before["is_merged"]))
        self.assertListEqual([0, 1], list(after["is_merged"]))

    def test_NumAuthoredMergedPullRequestsExtractor_does_not_count_a_merge_early(self):
        before = features.NumAuthoredMergedPullRequestsExtractor(
            self.gitdata, as_of=self.BEFORE_ISSUE_2_CLOSED
        ).extract_feature()
        after = features.NumAuthoredMergedPullRequestsExtractor(
            self.gitdata, as_of=self.AFTER_ISSUE_2_CLOSED
        ).extract_feature()

        self.assertListEqual([], list(before["num_pulls_merged"]))
        self.assertListEqual([1], list(after["num_pulls_merged"]))

    def test_IsPullExtractor(self):
        feature = features.IsPullExtractor(self.gitdata).extract_feature()
        self.assertListEqual([0, 1], list(feature["is_pull"]))
        self.assertListEqual([1, 2], list(feature["issue_number"]))

    def test_AvgSentimentPolarityExtractor(self):
        feature = features.AvgSentimentPolarityExtractor(self.gitdata).extract_feature()
        np.testing.assert_array_almost_equal(
            [-0.286666, 0.133333], list(feature["sentiment_polarity"])
        )
        self.assertListEqual([1, 2], list(feature["issue_number"]))

    def test_AvgSentimentSubjectivityExtractor(self):
        feature = features.AvgSentimentSubjectivityExtractor(
            self.gitdata
        ).extract_feature()
        np.testing.assert_array_almost_equal(
            [0.433333, 0.136666], list(feature["sentiment_subjectivity"])
        )
        self.assertListEqual([1, 2], list(feature["issue_number"]))

    def test_AvgCommentTimeExtractor(self):
        feature = features.AvgCommentTimeExtractor(self.gitdata).extract_feature()
        self.assertListEqual([1800.0, 2557680.0], list(feature["comment_times"]))
        self.assertListEqual([1, 2], list(feature["issue_number"]))

    # Commit and file extractors need commit and file ids, which the actor
    # mapping in setUp does not provide.
    def _gitdata_with_scm_ids(self):
        self.gitdata.add_file_ids()
        self.gitdata.add_commit_ids()
        return self.gitdata

    def test_InsertionsExtractor(self):
        feature = features.InsertionsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 2, 3, 4, 5], list(feature["commit_id"]))
        self.assertListEqual([3, 2, 2, 2, 0, 2], list(feature["insertions"]))

    def test_DeletionsExtractor(self):
        feature = features.DeletionsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 0, 2, 2, 2, 1], list(feature["deletions"]))

    def test_NumModifiedFilesExtractor(self):
        feature = features.NumModifiedFilesExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([1, 1, 1, 1, 1, 1], list(feature["n_modified_files"]))

    def test_IsMergeCommitExtractor(self):
        feature = features.IsMergeCommitExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 0, 0, 0, 0, 1], list(feature["is_merge"]))

    def test_IsBugfixCommitExtractor(self):
        feature = features.IsBugfixCommitExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 0, 1, 1, 0, 1], list(feature["is_bugfix"]))

    def test_MessagePolarityExtractor(self):
        feature = features.MessagePolarityExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        np.testing.assert_allclose(
            [0.1, 0.1, 0.2, 0.3, 0.4, 0.0], feature["message_polarity"]
        )

    def test_MessageSubjectivityExtractor(self):
        feature = features.MessageSubjectivityExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        np.testing.assert_allclose(
            [0.0, 0.0, 0.2, 0.1, 0.0, 0.0], feature["message_subjectivity"]
        )

    def test_NumParentsExtractor(self):
        # Commit 0 is the root, 5 is the merge commit with parents 3 and 4
        feature = features.NumParentsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 1, 1, 1, 2], list(feature["num_parents"]))

    def test_IntegrationLagExtractor(self):
        # Fixture commits are authored and committed in one step
        feature = features.IntegrationLagExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0.0] * 6, list(feature["integration_lag"]))

        # A patch written on Jan 1 and applied on Mar 1
        alice = ActorInfo(name="Alice", email="alice@example.org", type="git")
        rebased = _commit_datapoint(
            CommitInfo("a" * 40, id=0),
            author=alice,
            committer=alice,
            author_date=pd.Timestamp("2024-01-01", tz="UTC"),
            committer_date=pd.Timestamp("2024-03-01", tz="UTC"),
        )
        feature = features.IntegrationLagExtractor(
            _gitdata_with_commits(rebased)
        ).extract_feature()
        self.assertListEqual([60 * 86400.0], list(feature["integration_lag"]))

    # Fixture files (see gitdata-test-modified_files.csv):
    #   file 0 (radiation.txt): nloc 3 at commit 0, then 2 at commit 4
    #   file 1 (dns.txt): touched by commits 1, 2, 3, 5; complexity 0.5, 0.6, 0.8, 0.5
    #   file 2 (machine.py): touched once, by commit 5

    def test_LinesOfCodeExtractor(self):
        feature = features.LinesOfCodeExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 2], list(feature["file_id"]))
        # The latest modification wins, not the first or the largest
        self.assertListEqual([2, 5, 5], list(feature["nloc"]))

    def test_source_metrics_keep_the_last_measured_value(self):
        # PyDriller reports no source for a modification that changes no
        # lines, so its metrics are missing although the file is unchanged
        gitdata = _gitdata_with_files(
            _measured_file_version(
                "src/app.kt", "2024-11-29T10:00:00", nloc=80, n_methods=4, n_token=300
            ),
            _measured_file_version("src/moved/app.kt", "2024-11-29T11:00:00"),
        )

        self.assertListEqual(
            [80], list(features.LinesOfCodeExtractor(gitdata).extract_feature()["nloc"])
        )
        self.assertListEqual(
            [4],
            list(features.NumMethodsExtractor(gitdata).extract_feature()["n_methods"]),
        )
        self.assertListEqual(
            [300],
            list(features.NumTokensExtractor(gitdata).extract_feature()["n_token"]),
        )

    def test_source_metrics_stay_missing_for_files_lizard_never_measures(self):
        gitdata = _gitdata_with_files(
            _measured_file_version("docs/readme.md", "2024-11-29T10:00:00"),
            _measured_file_version("docs/readme.md", "2024-11-29T11:00:00"),
        )

        feature = features.LinesOfCodeExtractor(gitdata).extract_feature()

        self.assertListEqual([0], list(feature["file_id"]))
        self.assertTrue(feature["nloc"].isna().all())

    def test_ComplexityExtractor(self):
        feature = features.ComplexityExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        np.testing.assert_allclose([0.5, 0.5, 0.5], feature["complexity"])

    def test_NumMethodsExtractor(self):
        feature = features.NumMethodsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 1], list(feature["n_methods"]))

    def test_NumTokensExtractor(self):
        feature = features.NumTokensExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([2, 5, 3], list(feature["n_token"]))

    def test_NumModificationsExtractor(self):
        feature = features.NumModificationsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 2], list(feature["file_id"]))
        self.assertListEqual([2, 4, 1], list(feature["num_modifications"]))

    def test_NumModifyingAuthorsExtractor(self):
        feature = features.NumModifyingAuthorsExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()
        self.assertListEqual([0, 1, 2], list(feature["file_id"]))
        self.assertListEqual([2, 3, 1], list(feature["author_id"]))

    def test_DirectoryDepthExtractor(self):
        # The fixture keeps every file in the repository root, so nesting is
        # exercised on files built here
        gitdata = _gitdata_with_files(
            _file_datapoint(0, "README.md", "2024-01-01"),
            _file_datapoint(1, "src/git2kg/core.py", "2024-01-01"),
            _file_datapoint(2, "test/datapoint/files/deep.py", "2024-01-01"),
        )

        feature = features.DirectoryDepthExtractor(gitdata).extract_feature()

        self.assertListEqual([0, 1, 2], list(feature["file_id"]))
        self.assertListEqual([0, 2, 3], list(feature["directory_depth"]))

    def test_DirectoryDepthExtractor_follows_renames(self):
        # A file moved into a directory is as deep as its newest path says
        gitdata = _gitdata_with_files(
            _file_datapoint(0, "core.py", "2024-01-01"),
            _file_datapoint(0, "src/git2kg/core.py", "2024-02-01"),
        )

        feature = features.DirectoryDepthExtractor(gitdata).extract_feature()

        self.assertListEqual([2], list(feature["directory_depth"]))

    def test_FileAgeExtractor(self):
        # Fixture files are first seen at 10:15:17, 10:18:29 and 10:27:14
        as_of = pd.Timestamp("2024-11-29T10:30:00", tz="UTC")
        feature = features.FileAgeExtractor(
            self._gitdata_with_scm_ids(), as_of=as_of
        ).extract_feature()

        self.assertListEqual([0, 1, 2], list(feature["file_id"]))
        self.assertListEqual([883.0, 691.0, 166.0], list(feature["file_age"]))

    def test_FileAgeExtractor_without_as_of_counts_to_the_last_modification(self):
        feature = features.FileAgeExtractor(
            self._gitdata_with_scm_ids()
        ).extract_feature()

        # The newest modification in the data is at 10:27:14, which is also
        # when file 2 first appears
        self.assertListEqual([717.0, 525.0, 0.0], list(feature["file_age"]))
