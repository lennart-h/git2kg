from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

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
    IssueInfo,
    IssueTimelineEventDatapoint,
    ReviewDatapoint,
)
from git2kg.datapoint.utils import datapoint_list_to_dataframe
from git2kg.utils import pandas_timestamps_to_datetime


class EdgeExtractor:
    def extract_edit_add_edges(
        self, datapoints: DatapointList[EditDatapoint]
    ) -> list[tuple[int, int, datetime]]:
        return self._extract_line_edit_edges(datapoints, "added")

    def extract_edit_delete_edges(
        self, datapoints: DatapointList[EditDatapoint]
    ) -> list[tuple[int, int, datetime]]:
        return self._extract_line_edit_edges(datapoints, "deleted")

    def extract_issue_comment_edges(
        self, datapoints: DatapointList[IssueCommentDatapoint]
    ) -> list[tuple[ActorInfo, ActorInfo, datetime]]:
        _authors = np.array([dp.author for dp in datapoints])
        df = datapoint_list_to_dataframe(datapoints)
        df = df.sort_values(by=["created_at"])
        authors = _authors[df.index]
        timestamps = df.reset_index()["created_at"]
        shifted_idx = (
            df.reset_index(names="idx").groupby("issue_number").shift(1)["idx"]
        )
        # Edges are present at indices where shifted authors is not nan
        edge_mask = np.argwhere(np.isnan(shifted_idx) == False).flatten()  # noqa: E712
        authors = authors[edge_mask]
        shifted_authors = _authors[shifted_idx[edge_mask].astype("int64")]
        timestamps = timestamps[edge_mask]
        issue_comment_edges = list(
            zip(
                authors,
                shifted_authors,
                pandas_timestamps_to_datetime(timestamps),
                strict=True,
            )
        )
        return issue_comment_edges

    def extract_review_request_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[ActorInfo, ActorInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "requester",
            "req_reviewer",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "review_requested"],
        )

    def extract_issue_assigned_from_user_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[ActorInfo, ActorInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "assigner",
            "assignee",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "assigned"],
        )

    def extract_issue_assigned_to_user_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[IssueInfo, ActorInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "issue",
            "assignee",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "assigned"],
        )

    def extract_issue_merged_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "actor",
            "issue",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "merged"],
        )

    def extract_issue_closed_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "actor",
            "issue",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "closed"],
        )

    def extract_issue_reopened_edges(
        self, datapoints: DatapointList[IssueEventDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "actor",
            "issue",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "reopened"],
        )

    def extract_review_edges(
        self,
        review_datapoints: DatapointList[ReviewDatapoint],
        issue_datapoints: DatapointList[IssueDatapoint],
    ) -> list[tuple[ActorInfo, ActorInfo, datetime]]:
        _reviewers = np.array([dp.author for dp in review_datapoints])
        _pr_authors = np.array([dp.author for dp in issue_datapoints])
        review_df = datapoint_list_to_dataframe(review_datapoints).reset_index(
            names="idx"
        )
        issue_df = datapoint_list_to_dataframe(issue_datapoints).reset_index(
            names="idx"
        )[["issue_number", "idx"]]
        review_df = review_df.sort_values(by=["submitted_at"])

        review_df = review_df.merge(
            issue_df,
            left_on="pull_number",
            right_on="issue_number",
            how="inner",
            suffixes=("", "_pr"),
        )
        reviewers = _reviewers[review_df["idx"]]
        pr_authors = _pr_authors[review_df["idx_pr"]]
        timestamps = review_df["submitted_at"]
        return list(
            zip(
                reviewers,
                pr_authors,
                pandas_timestamps_to_datetime(timestamps),
                strict=True,
            )
        )

    def extract_create_issue_edges(
        self, datapoints: DatapointList[IssueDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(datapoints, "author", "issue", "created_at")

    def extract_review_pull_request_edges(
        self, datapoints: DatapointList[ReviewDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(datapoints, "author", "pull", "submitted_at")

    def extract_comment_issue_edges(
        self, datapoints: DatapointList[IssueCommentDatapoint]
    ) -> list[tuple[ActorInfo, IssueInfo, datetime]]:
        return self._extract_edges(datapoints, "author", "issue", "created_at")

    def extract_issue_referenced_on_edges(
        self, datapoints: DatapointList[IssueTimelineEventDatapoint]
    ) -> list[tuple[IssueInfo, IssueInfo, datetime]]:
        return self._extract_edges(
            datapoints,
            "issue",
            "source_issue",
            "created_at",
            filter_fn=lambda df: df[df["event_type"] == "cross-referenced"],
        )

    def extract_mentions_actor_edges(
        self, datapoints: DatapointList[IssueCommentDatapoint]
    ) -> list[tuple[ActorInfo, ActorInfo, datetime]]:
        sorted_mentions = sorted(datapoints, key=lambda ma: ma.created_at)
        return [
            (dp.author, ma, dp.created_at)
            for dp in sorted_mentions
            for ma in dp.mentioned_actors
        ]

    def extract_commit_parent_edges(
        self, datapoints: DatapointList[CommitDatapoint]
    ) -> list[tuple[CommitInfo, CommitInfo, datetime]]:
        edges = []
        for commit in datapoints:
            for parent in commit.commit_parents:
                edges.append((commit.commit, parent, commit.committer_date))

        return edges

    def extract_commit_modifies_file_edges(
        self, datapoints: DatapointList[FileDatapoint]
    ) -> list[tuple[CommitInfo, FileInfo, datetime]]:
        return list(
            map(
                lambda file: (file.commit, file.file, file.timestamp),
                datapoints,
            )
        )

    def extract_file_co_commit_edges(
        self, datapoints: DatapointList[FileDatapoint]
    ) -> list[tuple[FileInfo, FileInfo, datetime]]:
        files = np.array([dp.file for dp in datapoints])
        mod_file_df = datapoint_list_to_dataframe(datapoints).reset_index(names="idx")
        co_committed = pd.merge(
            left=mod_file_df, right=mod_file_df, on="file_commit_hash"
        )

        # Remove trivial co-commits (path is sufficient identifiers, since the path of a file is stable within the same commit)
        co_committed = co_committed[
            co_committed["file_path_x"] != co_committed["file_path_y"]
        ]

        files_x = files[co_committed["idx_x"]]
        files_y = files[co_committed["idx_y"]]
        timestamps = pandas_timestamps_to_datetime(co_committed["timestamp_x"])

        # Both directions (relation is undirected) are already covered by pd.merge
        return list(zip(files_x, files_y, timestamps, strict=True))

    def extract_author_commit_edges(
        self, datapoints: DatapointList[CommitDatapoint]
    ) -> list[tuple[ActorInfo, CommitInfo, datetime]]:
        # The authorship becomes observable when the commit enters the
        # repository, so the edge is timestamped like every other commit fact
        sorted_dps = sorted(datapoints, key=lambda c: c.committer_date)
        return [(c.author, c.commit, c.committer_date) for c in sorted_dps]

    def extract_comitter_commit_edges(
        self, datapoints: DatapointList[CommitDatapoint]
    ) -> list[tuple[ActorInfo, CommitInfo, datetime]]:
        sorted_dps = sorted(datapoints, key=lambda c: c.committer_date)
        return [(c.committer, c.commit, c.committer_date) for c in sorted_dps]

    def _extract_line_edit_edges(
        self, datapoints: DatapointList[EditDatapoint], edit_type: str
    ) -> list[tuple[int, int, datetime]]:
        return self._extract_edges(
            datapoints,
            "author",
            "prev_author",
            "timestamp",
            # Only entries with a previous author/target node represent edges
            filter_fn=lambda df: df[
                (df["edit_type"] == edit_type) & df["prev_author_name"].notna()
            ],
        )

    def _extract_edges(
        self,
        datapoints: DatapointList,
        u_field: str,
        v_field: str,
        timestamp_col: str,
        filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[tuple]:
        """Pairs up two datapoint attributes into timestamp-ordered edges.

        Shared by the extractors that reduce to "take two fields off each
        datapoint, optionally keep only rows matching some condition, order
        by a timestamp column". Extractors with a different shape (self-joins,
        merges across datapoint lists, regex/explode) are not a good fit and
        stay implemented separately.
        """
        us = np.array([getattr(dp, u_field) for dp in datapoints])
        vs = np.array([getattr(dp, v_field) for dp in datapoints])
        df = datapoint_list_to_dataframe(datapoints)
        if filter_fn is not None:
            df = filter_fn(df)
        df = df.sort_values(timestamp_col)
        return list(
            zip(
                us[df.index],
                vs[df.index],
                pandas_timestamps_to_datetime(df[timestamp_col]),
                strict=True,
            )
        )
