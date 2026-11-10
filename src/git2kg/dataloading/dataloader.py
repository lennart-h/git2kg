import logging
from collections.abc import Callable
from functools import partial
from typing import Self

from git2kg.dataloading.githubgraphqlloader import GithubGraphQLLoader
from git2kg.dataloading.githubloader import GithubLoader
from git2kg.dataloading.gitloader import GitLoader
from git2kg.datapoint import (
    CommitDatapoint,
    Datapoint,
    EditDatapoint,
    FileDatapoint,
    GithubCommitDatapoint,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    IssueTimelineEventDatapoint,
    ReviewDatapoint,
)
from git2kg.datapoint.base import DatapointList
from git2kg.persistence import NoOperationPersistenceManager, PersistenceManager

LOG = logging.getLogger(__name__)

#: The GitHub API a Dataloader mines with. Both loaders carry the same
#: interface, so the choice only changes how the data is fetched
REST = "rest"
GRAPHQL = "graphql"


class Dataloader:
    """Proxy class for Gitloader and GithubLoader, that can be initialized with an instance of PersistenceManager to provide persistence during dataloading"""

    def __init__(
        self,
        repo_path: str,
        organization_name: str,
        repo_name: str,
        github_token: str | None = None,
        persistence_manager: PersistenceManager | None = None,
        processes: int | None = 1,
        github_api: str = GRAPHQL,
    ):
        """
        Parameters
        ----------
        github_api : str, optional
            Which GitHub API to mine with, `GRAPHQL` (default) or `REST`.
            The GraphQL loader reaches comments, reviews and events through
            the issue they belong to instead of through an endpoint of their
            own, and reads mentions from the comment as GitHub rendered it.
            It needs the `read:user` scope to mine the email of an account
        """
        if not persistence_manager:
            raise Exception(
                "Persistence manager may not be null. If you wish no persistence pass `NoopPersistenceManager` (default)"
            )

        self.persistence_manager = persistence_manager
        if persistence_manager is None:
            self.persistence_manager = NoOperationPersistenceManager("")
        self.gitloader = GitLoader(repo_path)
        # Both loaders mine the same datapoints through the same interface, so
        # which one is used changes nothing downstream
        self.rest_githubloader = GithubLoader(
            organization_name, repo_name, github_token
        )
        self.graphql_githubloader = GithubGraphQLLoader(
            organization_name, repo_name, github_token
        )
        if github_api not in (REST, GRAPHQL):
            raise ValueError(
                f"Unknown github_api '{github_api}', expected '{REST}' or '{GRAPHQL}'"
            )
        self.github_api = github_api
        self.githubloader = (
            self.graphql_githubloader
            if github_api == GRAPHQL
            else self.rest_githubloader
        )
        self.processes = processes

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gitloader.close()

    def load_commits(self, chunk_size=10):
        commit_fun = self._prepare_parallel_load_fun(
            self.gitloader.load_git_commits, self.gitloader.load_git_commits_parallel
        )
        return self._check_and_load(CommitDatapoint, commit_fun, chunk_size=chunk_size)

    def load_edits(self, chunk_size=10):
        edit_fun = self._prepare_parallel_load_fun(
            self.gitloader.load_git_edits, self.gitloader.load_git_edits_parallel
        )
        return self._check_and_load(EditDatapoint, edit_fun, chunk_size=chunk_size)

    def load_modified_files(self, chunk_size=10):
        mod_file_fun = self._prepare_parallel_load_fun(
            self.gitloader.load_modified_files,
            self.gitloader.load_modified_files_parallel,
        )
        return self._check_and_load(FileDatapoint, mod_file_fun, chunk_size=chunk_size)

    def load_issues(self, chunk_size=10):
        return self._check_and_load(
            IssueDatapoint, self.githubloader.load_github_issues, chunk_size=chunk_size
        )

    def load_issue_comments(self, chunk_size=10):
        return self._check_and_load(
            IssueCommentDatapoint,
            self.githubloader.load_github_issues_comments,
            chunk_size=chunk_size,
        )

    def load_issue_events(self, chunk_size=10):
        return self._check_and_load(
            IssueEventDatapoint,
            self.githubloader.load_github_issue_events,
            chunk_size=chunk_size,
        )

    def load_reviews(self, chunk_size=10):
        return self._check_and_load(
            ReviewDatapoint,
            self.githubloader.load_github_reviews,
            chunk_size=chunk_size,
        )

    def load_issue_event_timeline(self, chunk_size=10):
        return self._check_and_load(
            IssueTimelineEventDatapoint,
            self.githubloader.load_github_issue_event_timeline,
            chunk_size=chunk_size,
        )

    def load_github_commits(self, chunk_size=10):
        return self._check_and_load(
            GithubCommitDatapoint,
            self.githubloader.load_github_commits,
            chunk_size=chunk_size,
        )

    def local_repo_path(self) -> str:
        return self.gitloader._local_repo()

    def _check_and_load(self, datapoint_cls: type[Datapoint], load_func, chunk_size):
        # Load datapoint which are already mined and persisted
        datapoints: DatapointList = self.persistence_manager.load(
            datapoint_class=datapoint_cls, processes=self.processes
        )
        # Find point where loading stopped last time, to only load new datapoints
        _from = (
            datapoints.find_latest().from_field()
            if len(datapoints.datapoints) > 0
            else None
        )
        # Every loader resumes inclusively: pydriller walks from the commit it
        # is given and the GitHub API treats `since` as "at or after", so
        # whatever sits exactly on the boundary is mined a second time. Only
        # those datapoints can already be present, so only they are compared
        # against what arrives, which keeps this to a handful of comparisons
        # even on a repository with a long history
        boundary = [dp for dp in datapoints if dp.from_field() == _from]

        # Buffer chunks of datapoints
        buf = DatapointList([], dp_type=datapoint_cls)
        for dp in load_func(_from=_from):
            if dp in boundary:
                continue
            buf.append(dp)
            if len(buf) >= chunk_size:
                # Flush buffer to persistence manager and clear it
                self.persistence_manager.append(buf)
                datapoints.extend(buf)
                buf = DatapointList([], dp_type=datapoint_cls)

        self.persistence_manager.append(buf)
        datapoints.extend(buf)

        return datapoints

    def _prepare_parallel_load_fun(self, sync_fun, parallel_fun: Callable):
        fun = sync_fun
        if self.processes != 1:
            LOG.info(f"{parallel_fun.__name__} with processes={self.processes}")
            fun = partial(parallel_fun, num_processes=self.processes, chunk_size=5)
        return fun
