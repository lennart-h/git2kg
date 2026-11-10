from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd

from git2kg.datapoint import ActorInfo
from git2kg.filtering import development_commits
from git2kg.gitdata import FIRST_SEEN, GitData

APPEARED_AT = "_appeared_at"


class DatapointTables(Protocol):
    """What a feature is computed from: the datapoint lists as tables.

    `GitData` satisfies this, and so does `FeatureData`.
    """

    def get_as_dataframe(self, dp_name: str) -> pd.DataFrame: ...

    def actor_appearances(self) -> pd.DataFrame: ...

    def history_end(self) -> pd.Timestamp: ...


class FeatureData:
    """The datapoint tables of a GitData, as they were at a point in time.

    Snapshots pass this rather than a sliced GitData: the tables are built
    once and filtered by timestamp per snapshot, instead of rebuilding and
    shipping the datapoint objects for every snapshot. For cumulative
    features that is the difference between moving a few columnar buffers and
    moving the whole mined history once per snapshot.
    """

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        appearances: pd.DataFrame,
        history_end: pd.Timestamp,
        as_of: pd.Timestamp | None = None,
    ):
        self._tables = tables
        self._appearances = appearances
        self._history_end = history_end
        self._as_of = as_of
        self._cache: dict[str, pd.DataFrame] = {}

    def get_as_dataframe(self, dp_name: str) -> pd.DataFrame:
        if dp_name not in self._cache:
            table = self._tables[dp_name]
            if self._as_of is not None:
                table = table[table[APPEARED_AT] < self._as_of]
            self._cache[dp_name] = table.drop(columns=APPEARED_AT)
        return self._cache[dp_name]

    def actor_appearances(self) -> pd.DataFrame:
        if self._as_of is None:
            return self._appearances
        return self._appearances[self._appearances[FIRST_SEEN] < self._as_of]

    def history_end(self) -> pd.Timestamp:
        return self._history_end

    def as_of(self, timestamp: pd.Timestamp | None) -> "FeatureData":
        """A view of the same tables holding only what had appeared by then"""
        return FeatureData(
            self._tables, self._appearances, self._history_end, timestamp
        )


# Columns a feature cannot aggregate, left out so that they are not carried
# into every worker process. The changed source line alone is the better part
# of a gigabyte on a mid-sized repository
UNAGGREGATABLE_COLUMNS = ("new_line",)

# The two states an issue alternates between, signed so that summing an
# issue's transition times yields the time it spent open, see
# `_open_transitions`
OPEN, CLOSED = -1, 1


def feature_data(gitdata: GitData) -> FeatureData:
    """Builds the tables of a GitData once, each carrying the appearance time
    of its datapoints so that it can be filtered to any point in time.

    Columns holding free text that no feature aggregates are left out, see
    `UNAGGREGATABLE_COLUMNS`.
    """
    tables = {}
    for dp_name, dp_list in gitdata.dp_lists.items():
        table = gitdata.get_as_dataframe(dp_name)
        table = table.drop(columns=list(UNAGGREGATABLE_COLUMNS), errors="ignore").copy()
        appeared_at = dp_list.dp_type.appearance_key()
        table[APPEARED_AT] = pd.to_datetime(
            pd.Series([appeared_at(dp) for dp in dp_list], index=table.index), utc=True
        )
        tables[dp_name] = table
    return FeatureData(tables, gitdata.actor_appearances(), gitdata.history_end())


class FeatureExtractor(ABC):
    """
    Abstract class for feature extraction
    """

    def __init__(
        self,
        data: DatapointTables,
        as_of: pd.Timestamp | None = None,
    ):
        """
        Parameters
        ----------
        data : DatapointTables
            The data the feature is computed from, as tables. A `GitData` or a
            `FeatureData` already filtered to the snapshot's point in time
        as_of : pd.Timestamp, optional
            The point in time the feature describes. Features of entities that
            keep changing after they appear must only use what was known at
            this point, so that an outcome does not show up in a snapshot that
            precedes it. `None` means as of mining time, which is what the
            un-sliced data describes
        """
        self.data = data
        self.as_of = as_of

    @abstractmethod
    def extract_feature(self) -> pd.DataFrame:
        """
        Computes the specific feature of to extract from the data
        """
        pass


class ActorFeatureExtractor(FeatureExtractor):
    """
    Classes implementing this interface compute features of actors. The first column of dataframes returned by their `extract_feature()` method holds the actor ID, the second column the feature
    """

    def development_commits(self) -> pd.DataFrame:
        """The commits of the data that count as development activity, i.e.
        every commit except the merge commits. Aggregating over these avoids
        crediting a merger with the work of everyone whose commits they merged.
        """
        return development_commits(self.data.get_as_dataframe("commits"))


class IssueFeatureExtractor(FeatureExtractor):
    """
    Classes implementing this interface compute features of issues. The first column of dataframes returned by their `extract_feature()` method holds the issue number, the second column the feature
    """


class CommitFeatureExtractor(FeatureExtractor):
    """
    Classes implementing this interface compute features of commits. The first column of dataframes returned by their `extract_feature()` method holds the commit ID, the second column the feature
    """


class FileFeatureExtractor(FeatureExtractor):
    """
    Classes implementing this interface compute features of files. The first column of dataframes returned by their `extract_feature()` method holds the file ID, the second column the feature
    """


@ActorFeatureExtractor.register
class ActorTenureExtractor(ActorFeatureExtractor):
    """
    Feature extractor for how long an actor has been around, in seconds.

    Measured from the actor's first appearance in any role, not from their
    first commit: most actors in a repository never commit at all.
    """

    def extract_feature(self):
        appearances = self.data.actor_appearances()
        until = self.as_of if self.as_of is not None else self.data.history_end()
        tenure = appearances[[ActorInfo.ID_COLUMN]].copy()
        tenure["tenure"] = (until - appearances[FIRST_SEEN]).dt.total_seconds()
        return tenure


@ActorFeatureExtractor.register
class NumCommitsActivityExtractor(ActorFeatureExtractor):
    """
    Feature extractor for commit activity of code authors.
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return commit_df.groupby("author_id").size().reset_index(name="num_commits")


@ActorFeatureExtractor.register
class TotalLinesAddedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for total lines added by code authors.
    """

    def extract_feature(self):
        commit_df = self.development_commits()

        return commit_df.groupby("author_id").agg({"insertions": "sum"}).reset_index()


@ActorFeatureExtractor.register
class TotalLinesRemovedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for total lines removed by code authors.
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return commit_df.groupby("author_id").agg({"deletions": "sum"}).reset_index()


@ActorFeatureExtractor.register
class UniqueFilesChangedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of unique files changed by code authors.
    """

    def extract_feature(self):
        edit_df = self.data.get_as_dataframe("edits")
        return edit_df.groupby("author_id").agg({"old_path": "nunique"}).reset_index()


@ActorFeatureExtractor.register
class CommitTimesExtractor(ActorFeatureExtractor):
    """
    Feature extractor for commit times of code authors.
    """

    def extract_feature(self):
        distinct_days = self.development_commits().copy()
        distinct_days["author_days"] = distinct_days["author_date"].dt.date
        return (
            distinct_days.groupby("author_id")
            .agg({"author_days": "nunique"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class FirstCommitDateExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the first commit date of code authors.
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return commit_df.groupby("author_id").agg({"author_date": "min"}).reset_index()


@ActorFeatureExtractor.register
class LastCommitDateExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the last commit date of code authors.
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return commit_df.groupby("author_id").agg({"author_date": "max"}).reset_index()


@ActorFeatureExtractor.register
class AverageCommitMessagePolarityExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average polarity of commit messages of code authors. Provided by TextBlob.
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return (
            commit_df.groupby("author_id")
            .agg({"message_polarity": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class AverageCommitMessageSubjectivityExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average subjectivity of commit messages of code authors. Provided by TextBlob
    """

    def extract_feature(self):
        commit_df = self.development_commits()

        return (
            commit_df.groupby("author_id")
            .agg({"message_subjectivity": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class AverageCommitTimeExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average time of the day, when an author makes commits.
    Counted in seconds from midnight in authors time zone.
    """

    def extract_feature(self):
        # Unfourtunately this seems not to be possible to do vectorized, since pandas does not allow different time zones in one column
        def zoned_seconds_from_midnight(datetime, timezone):
            zoned: pd.Timestamp = datetime.tz_convert(timezone)  # type: ignore[annotation-unchecked]
            return zoned.hour * 3600 + zoned.minute * 60 + zoned.second

        commit_sod = self.development_commits().copy()

        commit_sod["commit_second_of_day"] = commit_sod.apply(
            lambda x: zoned_seconds_from_midnight(
                x["author_date"], x["author_timezone"]
            ),
            axis=1,
            result_type="reduce",
        )
        return (
            commit_sod.groupby("author_id")
            .agg({"commit_second_of_day": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class NumIssueCommentsExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of comments github users contributed to issues or pull requests.
    """

    def extract_feature(self):
        issue_comment_df = self.data.get_as_dataframe("issue_comments")
        return (
            issue_comment_df.groupby("author_id")
            .size()
            .reset_index(name="num_issue_comments")
        )


@ActorFeatureExtractor.register
class NumIssuesCreatedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of issues or pull requests created by code users.
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")
        return (
            issue_df.groupby("author_id").size().reset_index(name="num_issues_created")
        )


@ActorFeatureExtractor.register
class NumIssuesAssignedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of issues or pull requests assigned to
    github users.

    Counts the assignment events rather than the assignees an issue carries,
    which are the ones it carries at the end of mining.
    """

    def extract_feature(self):
        event_df = self.data.get_as_dataframe("events")
        assignments = event_df[event_df["event_type"] == "assigned"]
        return (
            assignments.groupby("assignee_id")
            .size()
            .reset_index(name="num_issues_assigned")
        )


@ActorFeatureExtractor.register
class NumPullRequestsCreatedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of pull requests created by github users.
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")

        created_pulls = issue_df[issue_df["is_pull"] == True]  # noqa: E712
        return (
            created_pulls.groupby("author_id")
            .size()
            .reset_index(name="num_pulls_created")
        )


@ActorFeatureExtractor.register
class NumPullRequestsReviewedExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of pull requests reviewed by github users.
    """

    def extract_feature(self):
        review_df = self.data.get_as_dataframe("reviews")
        return (
            review_df.groupby("author_id").size().reset_index(name="num_pulls_reviewed")
        )


@ActorFeatureExtractor.register
class NumAuthoredMergedPullRequestsExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number of pull requests of a github user that are
    merged as of `as_of`.
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")

        merged_pulls = issue_df[_merged_by(issue_df, self.as_of)]
        return (
            merged_pulls.groupby("author_id")
            .size()
            .reset_index(name="num_pulls_merged")
        )


@ActorFeatureExtractor.register
class CommitConsistencyExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the commit consistency.
    Measured by the standard deviation of time between commits for code authors
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        ordered_commits = commit_df.sort_values(by="author_date")
        ordered_commits["time_diff"] = (
            ordered_commits.groupby("author_id")["author_date"]
            .diff()
            .dt.total_seconds()
        )
        ordered_commits = ordered_commits[pd.notna(ordered_commits["time_diff"])]
        return (
            ordered_commits.groupby("author_id")
            .agg({"time_diff": _population_std})
            .reset_index()
        )


@ActorFeatureExtractor.register
class NumDocumentationChangesExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the number changed files per author, which are documentation files
    """

    def extract_feature(self):
        commit_df = self.development_commits()
        return (
            commit_df.groupby("author_id")
            .agg({"n_modified_documentation": "sum"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class NumBugfixCommitsExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the nuumber of commits per author, which are identified asa bugfix commits
    """

    def extract_feature(self):
        commit_df = self.development_commits()

        return commit_df.groupby("author_id").agg({"is_bugfix": "sum"}).reset_index()


@ActorFeatureExtractor.register
class BugfixCommitRatioExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the ratio of bugfix commits of total commits per author
    """

    def extract_feature(self):
        commit_df = self.development_commits()

        return commit_df.groupby("author_id").agg({"is_bugfix": "mean"}).reset_index()


@ActorFeatureExtractor.register
class AvgCommentPolarityExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average polarity of comments of code authors.
    Polarity provided by the `TextBlob` library.
    """

    def extract_feature(self):
        issue_comment_df = self.data.get_as_dataframe("issue_comments")

        return (
            issue_comment_df.groupby("author_id")
            .agg({"sentiment_polarity": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class AvgCommentSubjectivityExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average subjectivity of comments of code authors.
    Subjectivity provided by the `TextBlob` library.
    """

    def extract_feature(self):
        issue_comment_df = self.data.get_as_dataframe("issue_comments")

        return (
            issue_comment_df.groupby("author_id")
            .agg({"sentiment_subjectivity": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class AvgIssueResponseTimeExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average response time of code authors to issues or pull requests assigned to them.
    """

    def extract_feature(self):
        issue_comment_df = self.data.get_as_dataframe("issue_comments")[
            ["issue_number", "created_at", "author_id"]
        ]
        issue_df = self.data.get_as_dataframe("issues")
        sorted_issues_and_comments = pd.concat(
            [issue_comment_df, issue_df[["issue_number", "created_at", "author_id"]]]
        ).sort_values(by="created_at")
        issue_df = issue_df.explode("assignees_id")
        # Add assignees id to comments as well
        sorted_issues_and_comments = sorted_issues_and_comments.merge(
            issue_df[["issue_number", "assignees_id"]], on="issue_number", how="left"
        )
        sorted_issues_and_comments["response_time"] = (
            sorted_issues_and_comments.groupby("issue_number")["created_at"]
            .diff()
            .dt.total_seconds()
        )

        sorted_issues_and_comments = sorted_issues_and_comments[
            (
                sorted_issues_and_comments["assignees_id"]
                == sorted_issues_and_comments["author_id"]
            )
            & (pd.notna(sorted_issues_and_comments["response_time"]))
        ]
        return (
            sorted_issues_and_comments.groupby("author_id")
            .agg({"response_time": "mean"})
            .reset_index()
        )


@ActorFeatureExtractor.register
class AvgReviewTimeExtractor(ActorFeatureExtractor):
    """
    Feature extractor for the average time authors take to review a pull request after review request.
    """

    def extract_feature(self):
        review_df = self.data.get_as_dataframe("reviews")
        event_df = self.data.get_as_dataframe("events")
        submitted_reviews = review_df[pd.isna(review_df["submitted_at"]) == False]  # noqa: E712
        review_requests = event_df[event_df["event_type"] == "review_requested"][
            ["issue_number", "created_at", "req_reviewer_id"]
        ]
        submitted_reviews = submitted_reviews.merge(
            review_requests,
            left_on=["pull_number", "author_id"],
            right_on=["issue_number", "req_reviewer_id"],
            how="inner",
        )
        submitted_reviews["review_time"] = (
            submitted_reviews["submitted_at"] - submitted_reviews["created_at"]
        ).dt.total_seconds()
        return (
            submitted_reviews.groupby("author_id")
            .agg({"review_time": "mean"})
            .reset_index()
        )


@IssueFeatureExtractor.register
class IssueAgeExtractor(IssueFeatureExtractor):
    """
    Feature extractor for the time since an issue was created, in seconds.
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")
        until = self.as_of if self.as_of is not None else self.data.history_end()
        age = issue_df[["issue_number"]].copy()
        age["issue_age"] = (until - issue_df["created_at"]).dt.total_seconds()
        # One row per issue: an issue can be mined twice when it is updated
        # while its pages are being read
        return age.drop_duplicates("issue_number", keep="last")


@IssueFeatureExtractor.register
class TimeOpenExtractor(IssueFeatureExtractor):
    """
    Feature extractor for how long an issue has been open in total by `as_of`,
    in seconds. An issue that was closed and reopened contributes every
    interval it spent open, rather than the span from its creation to its
    latest close.
    """

    def extract_feature(self):
        until = self.as_of if self.as_of is not None else self.data.history_end()
        transitions = _open_transitions(self.data)
        # Summing the transition times with a sign yields the length of every
        # open interval at once, since an issue leaves the open state at a
        # close and enters it again at a reopen
        signed = (
            transitions["created_at"].astype("int64") / 10**9 * transitions["state"]
        )
        time_open = signed.groupby(transitions["issue_number"]).sum()
        last_state = transitions.groupby("issue_number")["state"].last()
        time_open[last_state == OPEN] += until.timestamp()
        return time_open.reset_index(name="time_open")


@IssueFeatureExtractor.register
class IsOpenExtractor(IssueFeatureExtractor):
    """
    Feature extractor for whether an issue is open as of `as_of`.
    """

    def extract_feature(self):
        transitions = _open_transitions(self.data)
        last_state = transitions.groupby("issue_number")["state"].last()
        return (last_state == OPEN).astype(int).reset_index(name="is_open")


@IssueFeatureExtractor.register
class IsMergedExtractor(IssueFeatureExtractor):
    """
    Feature extractor for whether a pull request is merged as of `as_of`
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")

        merged = issue_df[["issue_number"]].copy()
        merged["is_merged"] = _merged_by(issue_df, self.as_of).astype("int")
        return merged.drop_duplicates("issue_number", keep="last")


@IssueFeatureExtractor.register
class IsPullExtractor(IssueFeatureExtractor):
    """
    Feature extractor for merged property of issues
    """

    def extract_feature(self):
        issue_df = self.data.get_as_dataframe("issues")

        pull = issue_df[["issue_number", "is_pull"]].copy()
        pull["is_pull"] = pull["is_pull"].astype("int")
        return pull.drop_duplicates("issue_number", keep="last")


@IssueFeatureExtractor.register
class AvgSentimentPolarityExtractor(IssueFeatureExtractor):
    """
    Feature extractor for sentiment polarity of body and comments of issues
    """

    def extract_feature(self):
        issue_comment_df = pd.concat(
            [
                self.data.get_as_dataframe("issue_comments")[
                    ["issue_number", "sentiment_polarity"]
                ],
                self.data.get_as_dataframe("issues")[
                    ["issue_number", "sentiment_polarity"]
                ],
            ]
        )

        return (
            issue_comment_df[pd.notna(issue_comment_df["sentiment_polarity"])]
            .groupby("issue_number")
            .agg({"sentiment_polarity": "mean"})
            .reset_index()
        )


@IssueFeatureExtractor.register
class AvgSentimentSubjectivityExtractor(IssueFeatureExtractor):
    """
    Feature extractor for sentiment subjectivity of body and comments of issues
    """

    def extract_feature(self):
        issue_comment_df = pd.concat(
            [
                self.data.get_as_dataframe("issue_comments")[
                    ["issue_number", "sentiment_subjectivity"]
                ],
                self.data.get_as_dataframe("issues")[
                    ["issue_number", "sentiment_subjectivity"]
                ],
            ]
        )
        return (
            issue_comment_df[pd.notna(issue_comment_df["sentiment_subjectivity"])]
            .groupby("issue_number")
            .agg({"sentiment_subjectivity": "mean"})
            .reset_index()
        )


@IssueFeatureExtractor.register
class AvgCommentTimeExtractor(IssueFeatureExtractor):
    """
    Feature extractor for computing the average time between comment on an issue
    """

    def extract_feature(self):
        issue_comment_df = pd.concat(
            [
                self.data.get_as_dataframe("issue_comments")[
                    ["issue_number", "created_at"]
                ],
                self.data.get_as_dataframe("issues")[["issue_number", "created_at"]],
            ]
        )
        comment_times = issue_comment_df.sort_values("created_at")
        comment_times["comment_times"] = (
            comment_times.groupby("issue_number")["created_at"]
            .diff()
            .dt.total_seconds()
        )
        return (
            comment_times.groupby("issue_number")
            .agg({"comment_times": "mean"})
            .reset_index()
        )


@CommitFeatureExtractor.register
class InsertionsExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the number of lines a commit inserts
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        return commit_df[["commit_id", "insertions"]]


@CommitFeatureExtractor.register
class DeletionsExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the number of lines a commit deletes
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        return commit_df[["commit_id", "deletions"]]


@CommitFeatureExtractor.register
class NumModifiedFilesExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the number of files a commit modifies
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        return commit_df[["commit_id", "n_modified_files"]]


@CommitFeatureExtractor.register
class IsMergeCommitExtractor(CommitFeatureExtractor):
    """
    Feature extractor for whether a commit is a merge commit
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        merge = commit_df[["commit_id", "is_merge"]].copy()
        merge["is_merge"] = merge["is_merge"].astype("int")
        return merge


@CommitFeatureExtractor.register
class IsBugfixCommitExtractor(CommitFeatureExtractor):
    """
    Feature extractor for whether a commit is identified as a bugfix commit
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        bugfix = commit_df[["commit_id", "is_bugfix"]].copy()
        bugfix["is_bugfix"] = bugfix["is_bugfix"].astype("int")
        return bugfix


@CommitFeatureExtractor.register
class MessagePolarityExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the sentiment polarity of a commit message. Provided by TextBlob.
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        return commit_df[["commit_id", "message_polarity"]]


@CommitFeatureExtractor.register
class MessageSubjectivityExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the sentiment subjectivity of a commit message. Provided by TextBlob.
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        return commit_df[["commit_id", "message_subjectivity"]]


@CommitFeatureExtractor.register
class NumParentsExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the number of parents of a commit: 0 for a root commit, 1 for a regular commit, 2 or more for a merge commit
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        parents = commit_df[["commit_id"]].copy()
        parents["num_parents"] = commit_df["commit_parents_id"].apply(len)
        return parents


@CommitFeatureExtractor.register
class IntegrationLagExtractor(CommitFeatureExtractor):
    """
    Feature extractor for the time between a change being written and entering
    the repository, in seconds: the committer date minus the author date. Zero
    for a commit made and integrated in one step, large for one that sat on a
    branch, was rebased or was cherry-picked
    """

    def extract_feature(self):
        commit_df = self.data.get_as_dataframe("commits")
        lag = commit_df[["commit_id"]].copy()
        lag["integration_lag"] = (
            commit_df["committer_date"] - commit_df["author_date"]
        ).dt.total_seconds()
        return lag


@FileFeatureExtractor.register
class LinesOfCodeExtractor(FileFeatureExtractor):
    """
    Feature extractor for the lines of code of a file, as of the latest modification that measured it
    """

    def extract_feature(self):
        return _last_measured(self.data, "nloc")


@FileFeatureExtractor.register
class ComplexityExtractor(FileFeatureExtractor):
    """
    Feature extractor for the cyclomatic complexity of a file, as of the latest modification that measured it
    """

    def extract_feature(self):
        return _last_measured(self.data, "complexity")


@FileFeatureExtractor.register
class NumMethodsExtractor(FileFeatureExtractor):
    """
    Feature extractor for the number of methods in a file, as of the latest modification that measured it
    """

    def extract_feature(self):
        return _last_measured(self.data, "n_methods")


@FileFeatureExtractor.register
class NumTokensExtractor(FileFeatureExtractor):
    """
    Feature extractor for the number of tokens in a file, as of the latest modification that measured it
    """

    def extract_feature(self):
        return _last_measured(self.data, "n_token")


@FileFeatureExtractor.register
class NumModificationsExtractor(FileFeatureExtractor):
    """
    Feature extractor for the number of times a file is modified in the data
    """

    def extract_feature(self):
        modified_files_df = self.data.get_as_dataframe("modified_files")
        return (
            modified_files_df.groupby("file_id")
            .size()
            .reset_index(name="num_modifications")
        )


@FileFeatureExtractor.register
class NumModifyingAuthorsExtractor(FileFeatureExtractor):
    """
    Feature extractor for the number of distinct authors modifying a file in the data
    """

    def extract_feature(self):
        modified_files_df = self.data.get_as_dataframe("modified_files")
        commit_df = self.data.get_as_dataframe("commits")
        modifying_authors = modified_files_df[["file_id", "commit_id"]].merge(
            commit_df[["commit_id", "author_id"]], on="commit_id", how="inner"
        )
        return (
            modifying_authors.groupby("file_id")
            .agg({"author_id": "nunique"})
            .reset_index()
        )


@FileFeatureExtractor.register
class DirectoryDepthExtractor(FileFeatureExtractor):
    """
    Feature extractor for how deep a file sits in the directory tree, counted
    in path separators: 0 for a file in the repository root. Taken from the
    path the file has at `as_of`, which is the one it was last moved to by
    then
    """

    def extract_feature(self):
        latest = _latest_file_version(self.data)
        depth = latest[["file_id"]].copy()
        depth["directory_depth"] = latest["file_path"].fillna("").str.count("/")
        return depth


@FileFeatureExtractor.register
class FileAgeExtractor(FileFeatureExtractor):
    """
    Feature extractor for how long a file has been in the repository at
    `as_of`, in seconds, counted from the first modification that records it.
    Without an `as_of`, the age is counted up to the file's latest
    modification instead
    """

    def extract_feature(self):
        modified_files_df = self.data.get_as_dataframe("modified_files")
        first_seen = modified_files_df.groupby("file_id")["timestamp"].min()
        until = (
            self.as_of
            if self.as_of is not None
            else modified_files_df["timestamp"].max()
        )
        return (until - first_seen).dt.total_seconds().reset_index(name="file_age")


def _population_std(values: pd.Series) -> float:
    return float(np.std(values, ddof=0))


def _merged_by(issue_df: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.Series:
    """Determines for every issue whether it is a merged pull request as of
    `as_of`. A merge closes the pull request, so the closing time is when the
    merge becomes observable.

    Parameters
    ----------
    issue_df : pd.DataFrame
        The issues, as returned by `GitData.get_as_dataframe("issues")`
    as_of : pd.Timestamp, optional
        The point in time to judge by, or None for the end of mining
    """
    merged = issue_df["is_merged"].fillna(False).astype(bool)
    if as_of is None:
        return merged
    return merged & (issue_df["merged_at"] < as_of)


def _open_transitions(data: DatapointTables) -> pd.DataFrame:
    """The open/closed transitions of every issue, in order.

    An issue opens when it is created and changes state with every "closed"
    and "reopened" event. GitHub emits repeated events of the same kind, which
    are no transition, so they are dropped and the states alternate.

    The transitions are read from the events rather than from the issue's
    `closed_at`, which holds the latest close as of mining and says nothing
    about the state the issue was in at an earlier point in time.
    """
    issue_df = data.get_as_dataframe("issues")
    events_df = data.get_as_dataframe("events")
    state_of = {"closed": CLOSED, "reopened": OPEN}
    events_df = events_df[events_df["event_type"].isin(state_of)]
    transitions = pd.concat(
        [
            issue_df[["issue_number", "created_at"]].assign(state=OPEN),
            events_df[["issue_number", "created_at"]].assign(
                state=events_df["event_type"].map(state_of)
            ),
        ]
    ).sort_values(["issue_number", "created_at"])
    previous = transitions.groupby("issue_number")["state"].shift()
    return transitions[transitions["state"] != previous]


def _latest_file_version(data: DatapointTables) -> pd.DataFrame:
    """Reduces the modified files to one row per file, holding the file's
    latest modification in the data.
    """
    modified_files_df = data.get_as_dataframe("modified_files")
    return modified_files_df.sort_values("timestamp").drop_duplicates(
        "file_id", keep="last"
    )


def _last_measured(data: DatapointTables, column: str) -> pd.DataFrame:
    """Reduces the modified files to one row per file, holding the latest
    value of `column` that was actually measured.

    PyDriller reports no source for a modification that changes no lines, so
    the source metrics of such a record are missing. The file's content is
    unchanged in that case, which makes the preceding measurement the current
    one. Reading the latest modification instead would report a file as empty
    the moment it is touched without being edited.
    """
    modified_files_df = data.get_as_dataframe("modified_files")
    return (
        modified_files_df.sort_values("timestamp")
        .groupby("file_id")[column]
        .last()
        .reset_index()
    )
