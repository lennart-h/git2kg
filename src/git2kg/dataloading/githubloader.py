"""GitHub data loading helpers."""

import logging
import re
from datetime import datetime, timezone
from typing import Generator

from github import Auth, Github, GithubException, GithubRetry
from github.GithubObject import NotSet, Opt
from github.Issue import Issue
from github.IssueComment import IssueComment
from github.IssueEvent import IssueEvent
from github.NamedUser import NamedUser
from github.Organization import Organization
from github.PullRequestReview import PullRequestReview
from github.TimelineEvent import TimelineEvent
from textblob import TextBlob  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from git2kg.datapoint import (
    ActorInfo,
    GithubCommitDatapoint,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    IssueInfo,
    IssueTimelineEventDatapoint,
    ReviewDatapoint,
)

LOG = logging.getLogger(__name__)

SUPPORTED_EVENT_TYPES = ["review_requested", "assigned", "merged", "closed", "reopened"]
MENTION_REGEX = r"\@((([A-Za-z0-9]+)([\-]?)([A-Za-z0-9]+))+)(?=[\@\s])"

# GitHub attributes the content of a deleted account to one placeholder user.
# It is not a person and every deleted account shares it, so treating it as an
# actor would merge unrelated people into a single node. The GraphQL API
# reports the same accounts as missing
GHOST_LOGIN = "ghost"


class GithubLoader:
    def __init__(self, organization_name, repo_name, token):
        """Initializes a GithubLoader instance for a single repository

        Parameters
        ----------
        organization_name : _type_
            The organization name of the repository
        repo_name : _type_
            The repository name
        token : _type_
            A GitHub access token with priviledges >= public_repo
        """

        self.full_project_name = f"{organization_name}/{repo_name}"
        auth = Auth.Token(token)
        # Widen PyGithub's default retry budget (total=10, secondary_rate_wait=60s)
        # for large, long-running mining jobs: each new request gets its own fresh
        # retry budget (PyGithub resets it per request, not per GithubLoader), so
        # this is cheap headroom against a sustained secondary rate limit or a
        # transient GitHub outage outlasting the default, not a global counter
        # that depletes over the course of a mining run.
        retry = GithubRetry(total=20, secondary_rate_wait=90)
        self.github = Github(auth=auth, per_page=100, retry=retry)
        self.repo = self.github.get_repo(self.full_project_name)

    def load_github_issues(
        self, _from: datetime | None = None
    ) -> Generator[IssueDatapoint, None, None]:
        """
        Loads issues from GitHub using the GitHub API.

        Returns
        -------
        Generator[IssueDatapoint, None, None]
            A generator yielding the mined issues
        """
        # None type conversion for compability with PyGitHub library
        typesafe_from: Opt[datetime] = _from if _from is not None else NotSet

        issues = self.repo.get_issues(state="all", since=typesafe_from)
        for issue in tqdm(
            issues,
            total=issues.totalCount,
            desc="Downloading issues from Github API",
            unit="Issues",
        ):
            if (
                issue.user
                and not _is_deleted_account(issue.user)
                and not issue.updated_at == typesafe_from
            ):
                issue_datapoint = _safe_create(issue, _create_issue_datapoint)
                if issue_datapoint:
                    yield issue_datapoint

    def load_github_issues_comments(
        self, _from: datetime | None = None
    ) -> Generator[IssueCommentDatapoint, None, None]:
        """Load issue comments of the repository.

        Returns
        -------
        Generator[IssueCommentDatapoint, None, None]
            A generator yielding the downloaded issue comments
        """
        # None type conversion for compability with PyGitHub library
        typesafe_from: Opt[datetime] = _from if _from is not None else NotSet

        comments = self.repo.get_issues_comments(since=typesafe_from)
        for comment in tqdm(
            comments,
            total=comments.totalCount,
            desc="Downloading issue comments from Github API",
            unit="Comments",
        ):
            if (
                comment.user
                and not _is_deleted_account(comment.user)
                and not comment.updated_at == typesafe_from
            ):
                comment_datapoint = _safe_create(
                    comment, _create_issue_comment_datapoint
                )
                if comment_datapoint:
                    yield comment_datapoint

    def load_github_issue_events(
        self, _from: datetime | None = None
    ) -> Generator[IssueEventDatapoint, None, None]:
        """Load supported issue events of the repository.

        Supported event types are "review_requested", "assigned", "merged", "closed".

        Returns
        -------
        Generator[IssueEventDatapoint, None, None]
            A generator yielding the downloaded events
        """
        _from = (
            _from if _from is not None else datetime(1970, 1, 1, tzinfo=timezone.utc)
        )

        events = self.repo.get_issues_events()

        for event in tqdm(
            events,
            total=events.totalCount,
            desc="Processing events",
            unit="Event",
        ):
            if event.event in SUPPORTED_EVENT_TYPES and event.created_at > _from:
                event_datapoint = _safe_create(event, _create_issue_event_datapoint)
                if event_datapoint:
                    yield event_datapoint

    def load_github_reviews(
        self, _from: datetime | None = None
    ) -> Generator[ReviewDatapoint, None, None]:
        """Loads Pull Request Reviews from GitHub API.

        Returns
        -------
        Generator[ReviewDatapoint, None, None]
            A generator yielding the downloaded reviews
        """
        _from = (
            _from if _from is not None else datetime(1970, 1, 1, tzinfo=timezone.utc)
        )

        pulls = self.repo.get_pulls(
            state="all",
        )
        for pull in tqdm(
            pulls,
            total=pulls.totalCount,
            desc="Downloading reviews for pull requests",
            unit="PRs",
        ):
            for review in pull.get_reviews():
                if (
                    review.user
                    and not _is_deleted_account(review.user)
                    and review.submitted_at > _from
                ):
                    review_datapoint = _safe_create(
                        review, _create_review_datapoint, pull_request=pull
                    )
                    if review_datapoint and review_datapoint.review_state != "PENDING":
                        yield review_datapoint

    def load_github_issue_event_timeline(
        self, _from: datetime | None = None
    ) -> Generator[IssueTimelineEventDatapoint, None, None]:
        """Loads timeline events for issues.

        Currently only cross-referenced events that point back to the same repository
        are stored.

        Returns
        -------
        Generator[IssueTimelineEventDatapoint, None, None]
            A generator yielding the downloaded timeline events
        """
        typesafe_from: Opt[datetime] = _from if _from is not None else NotSet

        issues = self.repo.get_issues(state="all", since=typesafe_from)
        for issue in tqdm(
            issues,
            total=issues.totalCount,
            desc="Downloading timeline events for issues",
            unit="Issues",
        ):
            for event in issue.get_timeline():
                if _is_cross_referenced_event_for_repo(
                    event, self.full_project_name
                ) and (_from is None or event.created_at > _from):
                    timeline_datapoint = _safe_create(
                        event,
                        _create_issue_timeline_event_datapoint,
                        issue_number=issue.number,
                    )
                    if timeline_datapoint and timeline_datapoint.created_at:
                        yield timeline_datapoint

    def load_github_commits(
        self, _from: datetime | None = None
    ) -> Generator[GithubCommitDatapoint, None, None]:
        """Loads the accounts GitHub resolved each commit's git identity to.

        Only the default branch is walked, which is what the history of a
        repository means to GitHub.

        Returns
        -------
        Generator[GithubCommitDatapoint, None, None]
            A generator yielding one record per commit
        """
        typesafe_from: Opt[datetime] = _from if _from is not None else NotSet
        commits = self.repo.get_commits(since=typesafe_from)
        for commit in tqdm(
            commits,
            total=commits.totalCount,
            desc="Downloading commit authorship",
            unit="Commits",
        ):
            yield GithubCommitDatapoint(
                commit_hash=commit.sha,
                committer_date=commit.commit.committer.date,
                # Filled only where GitHub recognises the address git recorded
                # as one of an account's verified ones
                author=_github_login_to_actor_info(commit.author)
                if commit.author
                else None,
                committer=_github_login_to_actor_info(commit.committer)
                if commit.committer
                else None,
            )


def _safe_create(obj, creation, **kwargs):
    try:
        return creation(obj, **kwargs)
    except GithubException as ghe:
        LOG.exception(ghe)
        return None


def _is_deleted_account(login: NamedUser | Organization | None) -> bool:
    """Whether the account behind a login is gone."""
    return login is None or login.login == GHOST_LOGIN


def _github_login_to_actor_info(login: NamedUser | Organization):
    return ActorInfo(
        name=login.login, email=login.email, github_id=login.id, type=login.type
    )


def _create_issue_datapoint(issue: Issue) -> IssueDatapoint:
    pull_request = issue.as_pull_request() if issue.pull_request else None
    pull_merged = pull_request.merged if pull_request else False
    merged_at = pull_request.merged_at if pull_request and pull_merged else None
    # A merged pull request has no merged_by if the account that merged it is
    # gone, or if it was merged without one, e.g. by auto-merge
    merged_by = (
        _github_login_to_actor_info(pull_request.merged_by)
        if pull_request
        and pull_merged
        and not _is_deleted_account(pull_request.merged_by)
        else None
    )
    return IssueDatapoint(
        id=issue.id,
        body=issue.body,
        sentiment_polarity=TextBlob(
            issue.body if issue.body else ""
        ).sentiment.polarity,
        sentiment_subjectivity=TextBlob(
            issue.body if issue.body else ""
        ).sentiment.subjectivity,
        mentioned_actors=[ActorInfo(name=x) for x in _parse_at_mentions(issue.body)],
        created_at=issue.created_at,
        updated_at=issue.updated_at,
        issue=IssueInfo(issue.number),
        author=_github_login_to_actor_info(issue.user),
        assignees=[
            _github_login_to_actor_info(assignee)
            for assignee in issue.assignees or []
            if not _is_deleted_account(assignee)
        ],
        closed_at=issue.closed_at,
        is_pull=pull_request is not None,
        is_merged=pull_merged,
        merged_at=merged_at,
        merged_by=merged_by,
        state=issue.state,
    )


def _create_issue_comment_datapoint(
    comment: IssueComment,
) -> IssueCommentDatapoint:
    reactions = comment.reactions
    return IssueCommentDatapoint(
        issue=IssueInfo(number=int(comment.issue_url.split("/")[-1])),
        id=comment.id,
        author=_github_login_to_actor_info(comment.user),
        mentioned_actors=[ActorInfo(name=x) for x in _parse_at_mentions(comment.body)],
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        body=comment.body,
        sentiment_polarity=TextBlob(
            comment.body if comment.body else ""
        ).sentiment.polarity,
        sentiment_subjectivity=TextBlob(
            comment.body if comment.body else ""
        ).sentiment.subjectivity,
        n_reactions=reactions["total_count"],
        upvotes=reactions["+1"],
        downvotes=reactions["-1"],
        laugh=reactions["laugh"],
        hooray=reactions["hooray"],
        confused=reactions["confused"],
        heart=reactions["heart"],
        rocket=reactions["rocket"],
        eyes=reactions["eyes"],
    )


def _create_issue_event_datapoint(event: IssueEvent) -> IssueEventDatapoint:
    return IssueEventDatapoint(
        event_type=event.event,
        issue=IssueInfo(event.issue.number),
        requester=_github_login_to_actor_info(event.review_requester)
        if not _is_deleted_account(event.review_requester)
        else None,
        req_reviewer=_github_login_to_actor_info(event.requested_reviewer)
        if not _is_deleted_account(event.requested_reviewer)
        else None,
        created_at=event.created_at,
        assignee=_github_login_to_actor_info(event.assignee)
        if not _is_deleted_account(event.assignee)
        else None,
        assigner=_github_login_to_actor_info(event.assigner)
        if not _is_deleted_account(event.assigner)
        else None,
        actor=_github_login_to_actor_info(event.actor)
        if not _is_deleted_account(event.actor)
        else None,
    )


def _create_issue_timeline_event_datapoint(
    event: TimelineEvent,
    issue_number,
) -> IssueTimelineEventDatapoint:
    return IssueTimelineEventDatapoint(
        event_type=event.event,
        issue=IssueInfo(issue_number),
        created_at=event.created_at,
        actor=_github_login_to_actor_info(event.actor)
        if not _is_deleted_account(event.actor)
        else None,
        source_issue=IssueInfo(number=event.source.issue.number)
        if event.source
        else None,
    )


def _create_review_datapoint(
    review: PullRequestReview, pull_request
) -> ReviewDatapoint:
    return ReviewDatapoint(
        id=review.id,
        pull=IssueInfo(pull_request.number),
        body=review.body,
        sentiment_polarity=TextBlob(
            review.body if review.body else ""
        ).sentiment.polarity,
        sentiment_subjectivity=TextBlob(
            review.body if review.body else ""
        ).sentiment.subjectivity,
        submitted_at=review.submitted_at,
        author=_github_login_to_actor_info(review.user),
        review_state=review.state,
    )


def _is_cross_referenced_event_for_repo(
    event: TimelineEvent, full_project_name: str
) -> bool:
    return (
        event.event == "cross-referenced"
        and event.source is not None
        and event.source.issue is not None
        and event.source.issue.repository is not None
        and event.source.issue.repository.full_name == full_project_name
    )


def _parse_at_mentions(body: str) -> list[str]:
    # Find @-mentions of usernames (alphanumeric with single dashes in between)
    if body is None:
        return []
    matches = re.finditer(MENTION_REGEX, body)
    mentioned_users = [m.group(1) for m in matches]
    return mentioned_users
