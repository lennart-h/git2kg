import json
import os
from functools import partial
from typing import Any, get_args, get_origin

import numpy as np
import pandas as pd

from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    Datapoint,
    DatapointList,
    EditDatapoint,
    FileDatapoint,
    FileInfo,
    GithubCommitDatapoint,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    IssueTimelineEventDatapoint,
    ReviewDatapoint,
)
from git2kg.datapoint.utils import (
    csv_to_datapoint_list,
    datapoint_list_to_dataframe,
)
from git2kg.edge_extraction import EdgeExtractor
from git2kg.files import FileRegistry
from git2kg.utils import keep_earliest, map_id_to_actor_infos
from git2kg.verification import verify_columns

#: Column holding an actor's first appearance in `GitData.actor_appearances`
FIRST_SEEN = "first_seen"


class GitData:
    def __init__(
        self,
        commits: DatapointList[CommitDatapoint],
        edits: DatapointList[EditDatapoint],
        issues: DatapointList[IssueDatapoint],
        issue_comments: DatapointList[IssueCommentDatapoint],
        events: DatapointList[IssueEventDatapoint],
        reviews: DatapointList[ReviewDatapoint],
        issue_timeline_events: DatapointList[IssueTimelineEventDatapoint] | None = None,
        modified_files: DatapointList[FileDatapoint] | None = None,
        github_commits: DatapointList[GithubCommitDatapoint] | None = None,
        actors_disambiguated: bool = False,
        files_disambiguated: bool = False,
        commits_disambiguated: bool = False,
        file_registry: FileRegistry | None = None,
    ):
        self.commits: DatapointList[CommitDatapoint] = commits
        self.edits: DatapointList[EditDatapoint] = edits
        self.issues: DatapointList[IssueDatapoint] = issues
        self.issue_comments: DatapointList[IssueCommentDatapoint] = issue_comments
        self.events: DatapointList[IssueEventDatapoint] = events
        self.reviews: DatapointList[ReviewDatapoint] = reviews
        self.issue_timeline_events: DatapointList[IssueTimelineEventDatapoint] = (
            issue_timeline_events
            if issue_timeline_events is not None
            else DatapointList([], dp_type=IssueTimelineEventDatapoint)
        )
        self.modified_files: DatapointList[FileDatapoint] = (
            modified_files
            if modified_files is not None
            else DatapointList([], FileDatapoint)
        )
        # What GitHub knows about the commits git recorded. Empty for a
        # GitData mined before this was collected, or without a token
        self.github_commits: DatapointList[GithubCommitDatapoint] = (
            github_commits
            if github_commits is not None
            else DatapointList([], GithubCommitDatapoint)
        )

        self.dp_lists: dict[str, DatapointList] = {
            "commits": self.commits,
            "edits": self.edits,
            "issues": self.issues,
            "issue_comments": self.issue_comments,
            "events": self.events,
            "reviews": self.reviews,
            "issue_timeline_events": self.issue_timeline_events,
            "modified_files": self.modified_files,
            "github_commits": self.github_commits,
        }
        self._df_cache: dict[str, pd.DataFrame] = {}
        self.actors_disambiguated = actors_disambiguated
        self.files_disambiguated = files_disambiguated
        self.commits_disambiguated = commits_disambiguated
        self.file_registry: FileRegistry = (
            file_registry if file_registry is not None else FileRegistry()
        )
        self.edge_extractor = EdgeExtractor()
        self.get_actors()

    def get_as_dataframe(self, dp_name: str) -> pd.DataFrame:
        if dp_name not in self.dp_lists:
            raise KeyError(
                f"Unknown datapoint list: '{dp_name}'. Available: {list(self.dp_lists.keys())}"
            )
        if dp_name not in self._df_cache:
            self._df_cache[dp_name] = datapoint_list_to_dataframe(
                self.dp_lists[dp_name]
            )
        return self._df_cache[dp_name]

    @property
    def issues_and_comments(self) -> DatapointList[IssueCommentDatapoint]:
        return self.issue_comments + DatapointList(
            list(map(lambda x: x.as_issue_comment(), self.issues)),
            dp_type=IssueCommentDatapoint,
        )

    def get_actors(self) -> set[ActorInfo]:
        """Every actor occurring anywhere in the data, as it was mined.

        An actor is not described by a datapoint list of its own but appears
        across all of them, in any number of roles. The roles are found by
        reflection over the datapoint types, the same way `add_actor_ids` and
        `actor_appearances` find them, so a datapoint type that carries a new
        actor field needs no change here.

        The result is not disambiguated: two entries can be the same person
        under different names or addresses.

        Returns
        -------
        set[ActorInfo]
            The actors occurring in the GitData object
        """
        actors: set[ActorInfo] = set()
        for dp_list in self.dp_lists.values():
            for field_name, dtype in dp_list.dp_type.dtypes().items():
                if dtype is ActorInfo:
                    actors.update(getattr(dp, field_name) for dp in dp_list)
                # An actor can also be held in a collection, as the assignees
                # of an issue and the accounts a comment mentions are
                elif get_origin(dtype) in (list, set) and ActorInfo in get_args(dtype):
                    actors.update(
                        actor
                        for dp in dp_list
                        for actor in getattr(dp, field_name) or []
                    )
        actors.discard(None)
        self.actors = actors
        return actors

    def actor_appearances(self) -> pd.DataFrame:
        """Maps actor id -> the earliest time the actor appears anywhere in the data.

        Unlike the other node types, an actor is not described by one datapoint
        list but shows up across all of them, in any number of roles. Those
        roles are found by reflection over the datapoint types, the same way
        `add_actor_ids` finds them.

        Returns
        -------
        pd.DataFrame
            One row per actor, with the columns "actor_id" and "first_seen"
        """
        earliest: dict[Any, pd.Timestamp] = {}
        for dp_list in self.dp_lists.values():
            dp_type = dp_list.dp_type
            time_of = dp_type.appearance_key()
            dtypes = dp_type.dtypes()
            actor_fields = [
                field_name for field_name, dtype in dtypes.items() if dtype == ActorInfo
            ]
            actor_list_fields = [
                field_name
                for field_name, dtype in dtypes.items()
                if get_origin(dtype) is list and get_args(dtype)[0] == ActorInfo
            ]
            for datapoint in dp_list:
                appeared_at = time_of(datapoint)
                for field_name in actor_fields:
                    actor = getattr(datapoint, field_name)
                    if actor is not None:
                        keep_earliest(earliest, actor.id, appeared_at)
                for field_name in actor_list_fields:
                    for actor in getattr(datapoint, field_name):
                        if actor is not None:
                            keep_earliest(earliest, actor.id, appeared_at)
        return pd.DataFrame(
            {
                ActorInfo.ID_COLUMN: pd.Series(list(earliest.keys()), dtype="int64"),
                FIRST_SEEN: pd.to_datetime(
                    pd.Series(list(earliest.values()), dtype="object"), utc=True
                ),
            }
        )

    def history_end(self) -> pd.Timestamp:
        """The latest time at which anything appears in the data.

        The reference point for features that describe an age when no point in
        time is given, i.e. when the data is the whole mined history rather
        than a timeslice of it.
        """
        latest: pd.Timestamp = pd.Timestamp("NaT", tz="UTC")
        for dp_list in self.dp_lists.values():
            time_of = dp_list.dp_type.appearance_key()
            for datapoint in dp_list:
                appeared_at = time_of(datapoint)
                if appeared_at is None or pd.isna(appeared_at):
                    continue
                appeared_at = pd.Timestamp(appeared_at).tz_convert("UTC")
                if pd.isna(latest) or appeared_at > latest:
                    latest = appeared_at
        return latest

    def add_actor_ids(self, actor_mapping: pd.DataFrame):
        """Disambiguate the actors across all data frames and actor fields

        Parameters
        ----------
        actor_mapping : pd.DataFrame
            An actor mapping containing the columns "actor_id", "alias_name", "alias_email", "actor_github_id"
        """
        verify_columns(
            actor_mapping,
            [
                ActorInfo.ID_COLUMN,
                "alias_name",
                "alias_email",
                ActorInfo.GITHUB_ID_COLUMN,
            ],
        )

        for dp_list in self.dp_lists.values():
            actor_fields = [
                field_name
                for field_name, dtype in dp_list.dp_type.dtypes().items()
                if dtype == ActorInfo
            ]
            for field_name in actor_fields:
                actor_infos = [dp.__getattribute__(field_name) for dp in dp_list]
                actor_infos = [a for a in actor_infos if a is not None]
                map_id_to_actor_infos(actor_infos, actor_mapping)

            # Handle also lists of ActorInfo, as they appear in Issue.assignees and Issue.mentioned_actors
            actor_list_fields = [
                field_name
                for field_name, dtype in dp_list.dp_type.dtypes().items()
                if get_origin(dtype) is list and get_args(dtype)[0] == ActorInfo
            ]
            for field_name in actor_list_fields:
                # Unwrap list
                actor_infos = [
                    actor for dp in dp_list for actor in dp.__getattribute__(field_name)
                ]
                # Skip None-filter, as lists of None do not exist in that context
                map_id_to_actor_infos(actor_infos, actor_mapping)
        self.actors_disambiguated = True

    def add_file_ids(self):
        for dp_list in self.dp_lists.values():
            file_fields = [
                field_name
                for field_name, dtype in dp_list.dp_type.dtypes().items()
                if dtype == FileInfo
            ]
            for field_name in file_fields:
                file_infos = [dp.__getattribute__(field_name) for dp in dp_list]
                for finfo in file_infos:
                    finfo.id = self.file_registry.find_file(
                        finfo.commit_hash, finfo.path
                    )
        self.files_disambiguated = True

    def add_commit_ids(self):
        for dp_list in self.dp_lists.values():
            commit_fields = [
                field_name
                for field_name, dtype in dp_list.dp_type.dtypes().items()
                if dtype == CommitInfo
            ]
            for field_name in commit_fields:
                commit_info = [dp.__getattribute__(field_name) for dp in dp_list]
                for cinfo in commit_info:
                    if cinfo:
                        cinfo.id = self.file_registry.lookup_entries[
                            cinfo.hash
                        ].commit_id

            # Handle also lists of CommitInfo, as they appear in commit_parents
            commit_list_fields = [
                field_name
                for field_name, dtype in dp_list.dp_type.dtypes().items()
                if get_origin(dtype) is list and get_args(dtype)[0] == CommitInfo
            ]
            for field_name in commit_list_fields:
                commit_info = [dp.__getattribute__(field_name) for dp in dp_list]
                for cinfo_list in commit_info:
                    for cinfo in cinfo_list:
                        cinfo.id = self.file_registry.lookup_entries[
                            cinfo.hash
                        ].commit_id
        self.commits_disambiguated = True

    def issue_comment_edges(self) -> list:
        """Create a list of edges representing the communication of contributers, inferred from the issue tracking system.
        Returns
        -------
        list
            A list of issue comment edges. An edge of is a tuple (u, v, t) where u is the author_id of the author commenting on the same issue directly after author with the author_id v at time t (unix seconds) on the same issue.
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_comment_edges(self.issues_and_comments)

    def review_request_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of one actor u requesting a review from another actor v at time t

        Returns
        -------
        list
            The list of edges, containing tuples (u,v,t), representing the review request edges, where u and v are actor IDs and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """

        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_review_request_edges(self.events)

    def issue_assigned_from_user_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_assigned_from_user_edges(self.events)

    def issue_assigned_to_user_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_assigned_to_user_edges(self.events)

    def issue_merged_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_merged_edges(self.events)

    def issue_closed_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_closed_edges(self.events)

    def issue_reopened_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_issue_reopened_edges(self.events)

    def review_edges(self):
        """Computes a list of time stamped edges resembling the relation of an actor u submitting a review on an issue of actor v at time t


        Returns
        -------
        list
            The list of edges, containing tuples (u,v,t), representing the review edges, where u and v are actor IDs and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_review_edges(self.reviews, self.issues)

    def create_issue_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of an actor opening an issue v at time t


        Returns
        -------
        list
            The list of edges, containing tuples (u, v, t), representing the open issue edges, where u is an actor ID, v is an issue number and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_create_issue_edges(self.issues)

    def review_pull_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of an actor u submitting a review for a pull request v at time t


        Returns
        -------
        list
            The list of edges, containing tuples (u, v, t), representing the review pull edges, where u is an actor ID,, v is an issue number and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_review_pull_request_edges(self.reviews)

    def comment_issue_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of an actor u commenting on an issue v at time t

        Returns
        -------
        list
            The list of edges, containing tuples (u, v, t), representing the open issue edges, where u is an actor ID, v is an issue number and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_comment_issue_edges(self.issue_comments)

    def issue_referenced_on_edges(self) -> list:
        return self.edge_extractor.extract_issue_referenced_on_edges(
            self.issue_timeline_events
        )

    def mentions_actor_edges(self) -> list:
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_mentions_actor_edges(
            self.issues_and_comments
        )

    def edit_delete_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of an actor u deleting a line of code at time t that was previously edited by actor v


        Returns
        -------
        list
            The list of edges, containing tuples (u,v,t), representing the edit delete edges, where u and v are actor IDs and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_edit_delete_edges(self.edits)

    def edit_add_edges(self) -> list:
        """Computes a list of time stamped edges resembling the relation of an actor u adding to a line of code at time t that was previously edited by actor v


        Returns
        -------
        list
            The list of edges, containing tuples (u,v,t), representing the edit add edges, where u and v are actor IDs and t is a timestamp

        Raises
        ------
        AttributeError
            If the GitData object is not disambiguated
        """
        self._assert_actor_disambiguation()
        return self.edge_extractor.extract_edit_add_edges(self.edits)

    def commit_parent_edges(self) -> list:
        self._assert_commit_disambiguation()
        return self.edge_extractor.extract_commit_parent_edges(self.commits)

    def commit_modifies_file_edges(self) -> list:
        self._assert_commit_disambiguation()
        self._assert_file_disambiguation()
        return self.edge_extractor.extract_commit_modifies_file_edges(
            self.modified_files
        )

    def file_co_commit_edges(self) -> list:
        self._assert_file_disambiguation()
        return self.edge_extractor.extract_file_co_commit_edges(self.modified_files)

    def author_commit_edges(self) -> list:
        self._assert_actor_disambiguation()
        self._assert_commit_disambiguation()
        return self.edge_extractor.extract_author_commit_edges(self.commits)

    def committer_edges(self) -> list:
        self._assert_actor_disambiguation()
        self._assert_commit_disambiguation()
        return self.edge_extractor.extract_comitter_commit_edges(self.commits)

    def save(self, path, name):
        commits_df = datapoint_list_to_dataframe(self.commits)
        edits_df = datapoint_list_to_dataframe(self.edits)
        issues_df = datapoint_list_to_dataframe(self.issues)
        issue_comments_df = datapoint_list_to_dataframe(self.issue_comments)
        events_df = datapoint_list_to_dataframe(self.events)
        reviews_df = datapoint_list_to_dataframe(self.reviews)
        issue_timeline_events_df = datapoint_list_to_dataframe(
            self.issue_timeline_events
        )
        modified_files_df = datapoint_list_to_dataframe(self.modified_files)
        commits_df.to_csv(f"{path}/{name}-commits.csv", lineterminator="\n")
        edits_df.to_csv(f"{path}/{name}-edits.csv", lineterminator="\n")
        issues_df.to_csv(f"{path}/{name}-issues.csv", lineterminator="\n")
        issue_comments_df.to_csv(
            f"{path}/{name}-issue_comments.csv", lineterminator="\n"
        )
        events_df.to_csv(f"{path}/{name}-review_requests.csv", lineterminator="\n")
        reviews_df.to_csv(f"{path}/{name}-reviews.csv", lineterminator="\n")
        issue_timeline_events_df.to_csv(
            f"{path}/{name}-event-timeline.csv", lineterminator="\n"
        )
        modified_files_df.to_csv(
            f"{path}/{name}-modified_files.csv", lineterminator="\n"
        )
        datapoint_list_to_dataframe(self.github_commits).to_csv(
            f"{path}/{name}-github_commits.csv", lineterminator="\n"
        )
        with open(f"{path}/{name}-file_registry.json", mode="w") as f:
            json.dump(self.file_registry.to_dict(), f)

    def first_interaction(self) -> pd.Timestamp:
        """Get the time of the first interaction in the data

        Returns
        -------
        pd.Timestamp
            The timestamp of the first interaction in the data

        """
        return self._first_or_last_interaction(np.min)

    def last_interaction(self) -> pd.Timestamp:
        """Get the time of the last interaction in the data

        Returns
        -------
        pd.Timestamp
            The timestamp of the last interaction in the data

        """
        return self._first_or_last_interaction(np.max)

    def _first_or_last_interaction(self, agg_function) -> pd.Timestamp:
        timestamps = []
        for dp_list in self.dp_lists.values():
            primary_timestamp = dp_list.dp_type.appearance_key()
            timestamps.extend([primary_timestamp(dp) for dp in dp_list])
        return agg_function(timestamps)

    def _assert_actor_disambiguation(self) -> None:
        if not self.actors_disambiguated:
            raise AttributeError(
                "Actors in data are not yet disambiguated. Either provide disabmiguated data and set the actors_disambiguated flag to `True` or call the method `add_actor_ids(self, actor_mapping: pd.DataFrame)`"
            )

    def _assert_commit_disambiguation(self) -> None:
        if not self.commits_disambiguated:
            raise AttributeError(
                "Commits in data are not yet disambiguated. Either provide disabmiguated data and set the commits_disambiguated flag to `True` or call the method `add_commit_ids()`"
            )

    def _assert_file_disambiguation(self) -> None:
        if not self.files_disambiguated:
            raise AttributeError(
                "Files in data are not yet disambiguated. Either provide disabmiguated data and set the files_disambiguated flag to `True` or call the method `add_file_ids()`"
            )

    def __str__(self):
        return f"""
GitData
-------
{len(self.commits)} Commits
{len(self.edits)} Edits
{len(self.issues)} Issues
{len(self.issue_comments)} Issue Comments
{len(self.events)} Events
{len(self.reviews)} Reviews
        """


def _load_table(
    path: str,
    name: str,
    suffix: str,
    dp_type: type[Datapoint],
    processes: int | None = None,
    optional: bool = False,
    **read_options,
) -> DatapointList:
    """Reads one stored table and rebuilds its datapoints.

    The frame belongs to this function, so it is released once its datapoints
    exist. Read inline, the frames of every table would instead stay alive
    until the last one had been read, which on a large export is several times
    the memory of the datapoints themselves.
    """
    file_path = f"{path}/{name}-{suffix}.csv"
    if optional and not os.path.isfile(file_path):
        return DatapointList([], dp_type)
    return csv_to_datapoint_list(
        file_path, dp_type, processes=processes, lineterminator="\n", **read_options
    )


def _load_gitdata(
    path,
    name,
    actors_disambiguated: bool | None = None,
    files_disambiguated: bool | None = None,
    commits_disambiguated: bool | None = None,
    processes: int | None = None,
) -> GitData:
    """Loads a GitData from a directory of CSV exports.

    Parameters
    ----------
    path : str
        The directory holding the exports
    name : str
        The name the GitData was exported under
    actors_disambiguated : bool, optional
        Whether the loaded actors already carry their ids. Inferred from the
        data when not given
    files_disambiguated : bool, optional
        Whether the loaded files already carry their ids. Inferred from the
        data when not given
    commits_disambiguated : bool, optional
        Whether the loaded commits already carry their ids. Inferred from the
        data when not given
    processes : int, optional
        How many processes to rebuild the datapoints with, one by default.
        More is faster on a large export, at the cost of holding the rows of
        a table in several processes at once
    """
    read = partial(_load_table, path, name, processes=processes)
    commits = read("commits", CommitDatapoint, index_col=0, quotechar='"')
    edits = read("edits", EditDatapoint, index_col=0)
    issues = read("issues", IssueDatapoint)
    issue_comments = read("issue_comments", IssueCommentDatapoint)
    review_requests = read("review_requests", IssueEventDatapoint)
    reviews = read("reviews", ReviewDatapoint)
    timeline_events = read("event-timeline", IssueTimelineEventDatapoint, optional=True)
    modified_files = read("modified_files", FileDatapoint, optional=True)
    github_commits = read("github_commits", GithubCommitDatapoint, optional=True)

    file_registry = None
    if os.path.isfile(f"{path}/{name}-file_registry.json"):
        with open(f"{path}/{name}-file_registry.json", mode="r") as f:
            file_registry = FileRegistry.from_dict(json.load(f))

    gitdata = GitData(
        commits,
        edits,
        issues,
        issue_comments,
        review_requests,
        reviews,
        timeline_events,
        modified_files,
        github_commits,
        file_registry=file_registry,
    )
    if actors_disambiguated is None:
        actors_disambiguated = all(
            actor.id is not None for actor in gitdata.get_actors()
        )
    gitdata.actors_disambiguated = actors_disambiguated
    # An export carries the ids it was saved with. Without this the ids are
    # resolved again on every load, which walks the commit graph once per file
    # and costs far more than reading the whole export
    if files_disambiguated is None:
        files_disambiguated = _ids_present(gitdata, FileInfo)
    gitdata.files_disambiguated = files_disambiguated
    if commits_disambiguated is None:
        commits_disambiguated = _ids_present(gitdata, CommitInfo)
    gitdata.commits_disambiguated = commits_disambiguated
    return gitdata


def _ids_present(gitdata: GitData, info_type: type) -> bool:
    """Whether every nested info of a type already carries an id."""
    for dp_list in gitdata.dp_lists.values():
        info_fields = [
            field_name
            for field_name, dtype in dp_list.dp_type.dtypes().items()
            if dtype == info_type
        ]
        for field_name in info_fields:
            for datapoint in dp_list:
                info = getattr(datapoint, field_name)
                if info is not None and info.id is None:
                    return False
    return True
