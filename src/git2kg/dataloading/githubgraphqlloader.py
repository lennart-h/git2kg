"""GitHub data loading over the GraphQL API.

Mines the same datapoints as `GithubLoader`, which uses the REST API, and is
interchangeable with it. Two differences carry over into the data:

Mentions are read from the comment body as GitHub itself rendered it, rather
than found by a regular expression over the Markdown source. GitHub decides
what an `@token` is - it renders a real mention as a link and notifies the
account - so an annotation like `@Composable` in a code block is text, not a
mention, without this module having to recognise code.

A comment, review or event is reached through the issue or pull request it
belongs to instead of through an endpoint of its own, so the five loading
methods are served by two queries rather than five.
"""

import logging
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import urlparse

from github import Auth, Github, GithubException, GithubRetry
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

# How many nodes a page of each connection holds. The maximum is 100, but a
# page is resolved as one request on GitHub's side and times out on an issue
# carrying a long timeline, so the sizes are kept small. Halving them costs
# nothing: a page of 50 issues scores 2 points and a page of 25 scores 1
ISSUE_PAGE_SIZE = 25
PULL_PAGE_SIZE = 25
NESTED_PAGE_SIZE = 25
# GitHub allows at most 10 assignees, so they never need a second page
ASSIGNEE_PAGE_SIZE = 20

# How often a query answered with an empty body is sent again, and the seconds
# to wait before the first of those retries. The wait doubles with each one
EMPTY_BODY_ATTEMPTS = 5
EMPTY_BODY_BACKOFF = 2

# GraphQL type name -> the event type the REST API reports, which is what the
# rest of the library filters on
EVENT_TYPES = {
    "ClosedEvent": "closed",
    "ReopenedEvent": "reopened",
    "AssignedEvent": "assigned",
    "ReviewRequestedEvent": "review_requested",
    "MergedEvent": "merged",
}
CROSS_REFERENCED = "cross-referenced"

ISSUE_EVENT_ITEM_TYPES = [
    "CLOSED_EVENT",
    "REOPENED_EVENT",
    "ASSIGNED_EVENT",
    "CROSS_REFERENCED_EVENT",
]
PULL_EVENT_ITEM_TYPES = ISSUE_EVENT_ITEM_TYPES + [
    "REVIEW_REQUESTED_EVENT",
    "MERGED_EVENT",
]

# GraphQL reaction content -> the IssueCommentDatapoint field counting it
REACTIONS = {
    "THUMBS_UP": "upvotes",
    "THUMBS_DOWN": "downvotes",
    "LAUGH": "laugh",
    "HOORAY": "hooray",
    "CONFUSED": "confused",
    "HEART": "heart",
    "ROCKET": "rocket",
    "EYES": "eyes",
}

# The GraphQL API checks a token's scopes per field, where the REST API does
# not: reading an account's email needs `read:user` and anything but the login
# of an organization needs `read:org`, although both are public. The fields are
# therefore requested conditionally, so that a token holding only `public_repo`
# mines everything else rather than being rejected outright
_ACTOR_FIELDS = """
fragment ActorFields on Actor {
  __typename
  login
  ... on User { databaseId email @include(if: $withEmail) }
  ... on Organization {
    databaseId @include(if: $withOrg)
    organizationEmail: email @include(if: $withOrg)
  }
  ... on Mannequin {
    databaseId
    mannequinEmail: email @include(if: $withOrg)
  }
  ... on Bot { databaseId }
}
"""

# Declared by every document that spreads ActorFields
_SCOPE_VARIABLES = "$withEmail: Boolean!, $withOrg: Boolean!, "

#: Scopes that unlock the conditional actor fields
EMAIL_SCOPES = frozenset({"read:user", "user:email"})
ORG_SCOPES = frozenset({"read:org"})

_COMMENT_FIELDS = """
fragment CommentFields on IssueComment {
  fullDatabaseId
  body
  bodyHTML
  createdAt
  
  updatedAt
  author { ...ActorFields }
  reactionGroups { content reactors { totalCount } }
}
"""

_REVIEW_FIELDS = """
fragment ReviewFields on PullRequestReview {
  fullDatabaseId
  body
  state
  submittedAt
  author { ...ActorFields }
}
"""

_TIMELINE_FIELDS = """
fragment TimelineFields on Node {
  __typename
  ... on ClosedEvent   { createdAt actor { ...ActorFields } }
  ... on ReopenedEvent { createdAt actor { ...ActorFields } }
  ... on MergedEvent   { createdAt actor { ...ActorFields } }
  ... on AssignedEvent {
    createdAt
    actor    { ...ActorFields }
    assignee { ...ActorFields }
  }
  ... on ReviewRequestedEvent {
    createdAt
    actor             { ...ActorFields }
    requestedReviewer { ...ActorFields }
  }
  ... on CrossReferencedEvent {
    createdAt
    actor { ...ActorFields }
    isCrossRepository
    source {
      ... on Issue       { number repository { nameWithOwner } }
      ... on PullRequest { number repository { nameWithOwner } }
    }
  }
}
"""

ISSUES_QUERY = (
    _ACTOR_FIELDS
    + _COMMENT_FIELDS
    + _TIMELINE_FIELDS
    + """
query Issues($withEmail: Boolean!, $withOrg: Boolean!, $owner: String!, $name: String!, $since: DateTime, $cursor: String,
             $issues: Int!, $nested: Int!, $assignees: Int!, $types: [IssueTimelineItemsItemType!]) {
  repository(owner: $owner, name: $name) {
    issues(first: $issues, after: $cursor, filterBy: { since: $since },
           orderBy: { field: UPDATED_AT, direction: ASC }) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id number fullDatabaseId body bodyHTML state createdAt updatedAt closedAt
        author { ...ActorFields }
        assignees(first: $assignees) { nodes { ...ActorFields } }
        comments(first: $nested) {
          pageInfo { hasNextPage endCursor }
          nodes { ...CommentFields }
        }
        timelineItems(first: $nested, itemTypes: $types) {
          pageInfo { hasNextPage endCursor }
          nodes { ...TimelineFields }
        }
      }
    }
  }
}
"""
)

PULLS_QUERY = (
    _ACTOR_FIELDS
    + _COMMENT_FIELDS
    + _REVIEW_FIELDS
    + _TIMELINE_FIELDS
    + """
query Pulls($withEmail: Boolean!, $withOrg: Boolean!, $owner: String!, $name: String!, $cursor: String,
            $pulls: Int!, $nested: Int!, $assignees: Int!, $types: [PullRequestTimelineItemsItemType!]) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $pulls, after: $cursor, states: [OPEN, CLOSED, MERGED],
                 orderBy: { field: UPDATED_AT, direction: DESC }) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id number fullDatabaseId body bodyHTML state createdAt updatedAt closedAt
        merged mergedAt
        mergedBy { ...ActorFields }
        author { ...ActorFields }
        assignees(first: $assignees) { nodes { ...ActorFields } }
        comments(first: $nested) {
          pageInfo { hasNextPage endCursor }
          nodes { ...CommentFields }
        }
        reviews(first: $nested) {
          pageInfo { hasNextPage endCursor }
          nodes { ...ReviewFields }
        }
        timelineItems(first: $nested, itemTypes: $types) {
          pageInfo { hasNextPage endCursor }
          nodes { ...TimelineFields }
        }
      }
    }
  }
}
"""
)

# Reads the remaining pages of one connection of one issue or pull request.
# `connection` is interpolated rather than passed as a variable, since GraphQL
# has no variable for a field name
_MORE_QUERY = """
query More($withEmail: Boolean!, $withOrg: Boolean!, $id: ID!, $cursor: String, $nested: Int!%(types_var)s) {
  node(id: $id) {
    ... on %(parent)s {
      number
      %(connection)s(first: $nested, after: $cursor%(types_arg)s) {
        pageInfo { hasNextPage endCursor }
        nodes { ...%(fragment)s }
      }
    }
  }
}
"""

# The fragment each nested connection's nodes are read with
_CONNECTION_FRAGMENTS = {
    "comments": ("CommentFields", _COMMENT_FIELDS),
    "reviews": ("ReviewFields", _REVIEW_FIELDS),
    "timelineItems": ("TimelineFields", _TIMELINE_FIELDS),
}


def more_query(connection: str, is_pull: bool) -> str:
    """The query reading the remaining pages of one nested connection.

    Built per connection rather than kept as one document, because GraphQL
    rejects a document that defines a fragment it does not spread or declares
    a variable it does not use.
    """
    fragment_name, fragment = _CONNECTION_FRAGMENTS[connection]
    enum = (
        "PullRequestTimelineItemsItemType" if is_pull else "IssueTimelineItemsItemType"
    )
    is_timeline = connection == "timelineItems"
    return (
        _ACTOR_FIELDS
        + fragment
        + _MORE_QUERY
        % {
            "types_var": f", $types: [{enum}!]" if is_timeline else "",
            "types_arg": ", itemTypes: $types" if is_timeline else "",
            "parent": "PullRequest" if is_pull else "Issue",
            "connection": connection,
            "fragment": fragment_name,
        }
    )


_COMMIT_ACCOUNTS_QUERY = (
    _ACTOR_FIELDS
    + """
query CommitAccounts(%(scopes)s$owner: String!, $name: String!, $size: Int!,
                     $cursor: String, $since: GitTimestamp) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: $size, after: $cursor, since: $since) {
            pageInfo { hasNextPage endCursor }
            nodes {
              oid
              committedDate
              author { user { ...ActorFields } }
              committer { user { ...ActorFields } }
            }
          }
        }
      }
    }
  }
}
"""
    % {"scopes": _SCOPE_VARIABLES}
)

#: How many commits one page of the history query holds. A commit carries far
#: less than an issue does, so the full 100 is safe here
COMMIT_PAGE_SIZE = 100


@dataclass
class _Harvest:
    """Everything the two queries yield, grouped by the datapoint they feed."""

    issues: list[IssueDatapoint] = field(default_factory=list)
    comments: list[IssueCommentDatapoint] = field(default_factory=list)
    events: list[IssueEventDatapoint] = field(default_factory=list)
    reviews: list[ReviewDatapoint] = field(default_factory=list)
    timeline: list[IssueTimelineEventDatapoint] = field(default_factory=list)


class _MentionParser(HTMLParser):
    """Collects the logins GitHub rendered as user mentions.

    GitHub marks a mention it resolved to an account with the `user-mention`
    class and links it to that account, and leaves an `@token` it did not
    resolve - including anything inside a code block - as plain text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logins: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        if "user-mention" not in (attributes.get("class") or "").split():
            return
        # Team mentions carry the team-mention class instead, but guard the
        # hovercard type as well so that only accounts are collected
        if attributes.get("data-hovercard-type") not in (None, "user"):
            return
        login = _login_from_href(attributes.get("href"))
        if login:
            self.logins.append(login)


def _login_from_href(href: str | None) -> str | None:
    """The account a mention links to, which is the login GitHub resolved.

    Taken from the link target rather than from the text of the anchor, which
    GitHub does not always render as the plain login.
    """
    if not href:
        return None
    path = urlparse(href).path.strip("/")
    return path or None


def _mentioned_actors(body_html: str | None) -> list[ActorInfo]:
    """The actors mentioned in a comment, as GitHub resolved them."""
    if not body_html:
        return []
    parser = _MentionParser()
    parser.feed(body_html)
    # An actor mentioned twice in one body is one mentioned actor
    return [ActorInfo(name=login) for login in dict.fromkeys(parser.logins)]


class GithubGraphQLLoader:
    """Loads GitHub datapoints over the GraphQL API.

    Interchangeable with `GithubLoader`: same constructor and the same five
    loading methods.
    """

    def __init__(self, organization_name: str, repo_name: str, token: str | None):
        """
        Parameters
        ----------
        organization_name : str
            The owner of the repository
        repo_name : str
            The name of the repository
        token : str, optional
            A GitHub token. The GraphQL API rejects unauthenticated requests,
            so one is required in practice
        """
        self.organization_name = organization_name
        self.repo_name = repo_name
        self.full_project_name = f"{organization_name}/{repo_name}"
        auth = Auth.Token(token) if token else None
        # Matches GithubLoader's retry budget, see the comment there
        self.github = Github(
            auth=auth, retry=GithubRetry(total=20, secondary_rate_wait=90)
        )
        self._requester = self.github._Github__requester  # type: ignore[attr-defined]
        self._harvests: dict[datetime | None, _Harvest] = {}
        self._scopes: frozenset[str] | None = None

    def load_github_issues(
        self, _from: datetime | None = None
    ) -> Generator[IssueDatapoint, None, None]:
        """Loads issues and pull requests of the repository.

        Returns
        -------
        Generator[IssueDatapoint, None, None]
            A generator yielding the mined issues
        """
        for issue in self._stream_issues(_from):
            if _from is None or issue.updated_at > _from:
                yield issue

    def load_github_issues_comments(
        self, _from: datetime | None = None
    ) -> Generator[IssueCommentDatapoint, None, None]:
        """Loads the issue comments of the repository.

        Returns
        -------
        Generator[IssueCommentDatapoint, None, None]
            A generator yielding the downloaded issue comments
        """
        for comment in self._harvest(_from).comments:
            if _from is None or comment.updated_at > _from:
                yield comment

    def load_github_issue_events(
        self, _from: datetime | None = None
    ) -> Generator[IssueEventDatapoint, None, None]:
        """Loads the supported issue events of the repository.

        Supported event types are "review_requested", "assigned", "merged",
        "closed" and "reopened".

        Returns
        -------
        Generator[IssueEventDatapoint, None, None]
            A generator yielding the downloaded events
        """
        for event in self._harvest(_from).events:
            if _from is None or event.created_at > _from:
                yield event

    def load_github_reviews(
        self, _from: datetime | None = None
    ) -> Generator[ReviewDatapoint, None, None]:
        """Loads the pull request reviews of the repository.

        Returns
        -------
        Generator[ReviewDatapoint, None, None]
            A generator yielding the downloaded reviews
        """
        for review in self._harvest(_from).reviews:
            if _from is None or review.submitted_at > _from:
                yield review

    def load_github_issue_event_timeline(
        self, _from: datetime | None = None
    ) -> Generator[IssueTimelineEventDatapoint, None, None]:
        """Loads timeline events for issues.

        Currently only cross-referenced events that point back to the same
        repository are stored.

        Returns
        -------
        Generator[IssueTimelineEventDatapoint, None, None]
            A generator yielding the downloaded timeline events
        """
        for event in self._harvest(_from).timeline:
            if _from is None or event.created_at > _from:
                yield event

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
        cursor: str | None = None
        with tqdm(desc="Downloading commit authorship", unit="Commits") as progress:
            while True:
                data = self._run(
                    _COMMIT_ACCOUNTS_QUERY,
                    {
                        "owner": self.organization_name,
                        "name": self.repo_name,
                        "size": COMMIT_PAGE_SIZE,
                        "cursor": cursor,
                        "since": _from.isoformat() if _from else None,
                    },
                )
                branch = data["repository"]["defaultBranchRef"]
                # An empty repository has no default branch to walk
                if not branch:
                    break
                history = branch["target"]["history"]
                for node in history["nodes"]:
                    yield GithubCommitDatapoint(
                        commit_hash=node["oid"],
                        committer_date=_timestamp(node["committedDate"]),
                        author=_commit_account(node["author"]),
                        committer=_commit_account(node["committer"]),
                    )
                progress.update(len(history["nodes"]))
                if not history["pageInfo"]["hasNextPage"]:
                    break
                cursor = history["pageInfo"]["endCursor"]

    def _stream_issues(self, since: datetime | None) -> Iterator[IssueDatapoint]:
        """Yields every issue and pull request as its page arrives.

        Streaming rather than returning a finished harvest lets the caller
        persist what has been mined so far, so a mine that fails resumes from
        where it stopped instead of starting over. The other datapoints of a
        page have no caller yet and are kept for the methods that read them.
        """
        if since in self._harvests:
            yield from self._harvests[since].issues
            return
        harvest = _Harvest()
        yield from self._collect_issues(harvest, since)
        yield from self._collect_pulls(harvest, since)
        # Reached only by a fully consumed stream, so an abandoned one leaves
        # no half-filled harvest behind for the other methods to read
        self._harvests[since] = harvest

    def _harvest(self, since: datetime | None) -> _Harvest:
        """Everything mined from the given point in time, fetched once.

        The five loading methods read the same two queries, so the result is
        kept per point in time instead of being fetched again for each of them.
        """
        if since not in self._harvests:
            deque(self._stream_issues(since), maxlen=0)
        return self._harvests[since]

    def _token_scopes(self) -> frozenset[str]:
        """The scopes the token carries, read once from the API."""
        if self._scopes is None:
            headers, _ = self._requester.requestJsonAndCheck(
                "GET", f"{self._requester.base_url}/user"
            )
            granted = headers.get("x-oauth-scopes") or ""
            self._scopes = frozenset(
                scope.strip() for scope in granted.split(",") if scope.strip()
            )
            if not self._scopes & EMAIL_SCOPES:
                LOG.warning(
                    "The token has none of %s, so the email of a GitHub account "
                    "is not mined and actors are disambiguated by name alone",
                    sorted(EMAIL_SCOPES),
                )
        return self._scopes

    def _run(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Runs one query, retrying the answers that carry no body.

        A query GitHub does not finish in time is answered with a success
        status and an empty body rather than an error, so the retry the
        library is configured with does not cover it. Left unhandled it ends a
        mine, which on a large repository throws away hours of work.
        """
        scopes = self._token_scopes()
        payload = {
            "query": query,
            "variables": {
                **variables,
                "withEmail": bool(scopes & EMAIL_SCOPES),
                "withOrg": bool(scopes & ORG_SCOPES),
            },
        }
        for attempt in range(EMPTY_BODY_ATTEMPTS):
            _, data = self._requester.requestJsonAndCheck(
                "POST", self._requester.graphql_url, input=payload
            )
            if data is not None:
                if "errors" in data:
                    raise GithubException(400, data, None)
                return data["data"]
            if attempt + 1 < EMPTY_BODY_ATTEMPTS:
                pause = EMPTY_BODY_BACKOFF * 2**attempt
                LOG.warning(
                    "GitHub returned an empty body, retrying in %ss (%s/%s)",
                    pause,
                    attempt + 1,
                    EMPTY_BODY_ATTEMPTS - 1,
                )
                time.sleep(pause)
        raise GithubException(
            502,
            None,
            None,
            f"GitHub returned an empty body {EMPTY_BODY_ATTEMPTS} times in a row. "
            "The query is likely too large for the repository, try lowering "
            "ISSUE_PAGE_SIZE and NESTED_PAGE_SIZE",
        )

    def _collect_issues(
        self, harvest: _Harvest, since: datetime | None
    ) -> Iterator[IssueDatapoint]:
        """Reads the issues, with their comments and events."""
        cursor: str | None = None
        progress: tqdm | None = None
        while True:
            data = self._run(
                ISSUES_QUERY,
                {
                    "owner": self.organization_name,
                    "name": self.repo_name,
                    "since": since.isoformat() if since else None,
                    "cursor": cursor,
                    "issues": ISSUE_PAGE_SIZE,
                    "nested": NESTED_PAGE_SIZE,
                    "assignees": ASSIGNEE_PAGE_SIZE,
                    "types": ISSUE_EVENT_ITEM_TYPES,
                },
            )
            connection = data["repository"]["issues"]
            if progress is None:
                progress = tqdm(
                    total=connection["totalCount"],
                    desc="Downloading issues from Github GraphQL API",
                    unit="Issues",
                )
            for node in connection["nodes"]:
                issue = self._absorb(harvest, node, is_pull=False)
                if issue is not None:
                    yield issue
                progress.update(1)
            if not connection["pageInfo"]["hasNextPage"]:
                break
            cursor = connection["pageInfo"]["endCursor"]
        if progress is not None:
            progress.close()

    def _collect_pulls(
        self, harvest: _Harvest, since: datetime | None
    ) -> Iterator[IssueDatapoint]:
        """Reads the pull requests, with their comments, reviews and events.

        The pull request connection takes no `since` filter, so it is read
        newest first and stopped once it reaches what was mined before.
        """
        cursor: str | None = None
        progress: tqdm | None = None
        while True:
            data = self._run(
                PULLS_QUERY,
                {
                    "owner": self.organization_name,
                    "name": self.repo_name,
                    "cursor": cursor,
                    "pulls": PULL_PAGE_SIZE,
                    "nested": NESTED_PAGE_SIZE,
                    "assignees": ASSIGNEE_PAGE_SIZE,
                    "types": PULL_EVENT_ITEM_TYPES,
                },
            )
            connection = data["repository"]["pullRequests"]
            if progress is None:
                progress = tqdm(
                    total=connection["totalCount"],
                    desc="Downloading pull requests from Github GraphQL API",
                    unit="PRs",
                )
            exhausted = False
            for node in connection["nodes"]:
                if since is not None and _timestamp(node["updatedAt"]) <= since:
                    exhausted = True
                    break
                issue = self._absorb(harvest, node, is_pull=True)
                if issue is not None:
                    yield issue
                progress.update(1)
            if exhausted or not connection["pageInfo"]["hasNextPage"]:
                break
            cursor = connection["pageInfo"]["endCursor"]
        if progress is not None:
            progress.close()

    def _absorb(
        self, harvest: _Harvest, node: dict, is_pull: bool
    ) -> IssueDatapoint | None:
        """Turns one issue or pull request node into its datapoints.

        The issue is returned rather than kept, so that the caller can hand it
        on as soon as its page arrives. The datapoints that no caller is
        waiting for go into the harvest.

        An author is null once the account behind it is gone. Such a node is
        skipped, as it is by the REST loader, since the datapoints are keyed
        to an actor.
        """
        number = node["number"]
        author = _actor(node.get("author"))
        if author is None:
            LOG.debug("Skipping #%s, which has no author", number)
            return None
        issue = _issue_datapoint(node, is_pull, author)
        harvest.issues.append(issue)

        for comment in self._all_nodes(node, "comments", is_pull):
            comment_author = _actor(comment.get("author"))
            if comment_author is not None:
                harvest.comments.append(
                    _comment_datapoint(comment, number, comment_author)
                )

        if is_pull:
            for review in self._all_nodes(node, "reviews", is_pull):
                review_author = _actor(review.get("author"))
                if review_author is not None and review.get("submittedAt"):
                    harvest.reviews.append(
                        _review_datapoint(review, number, review_author)
                    )

        for item in self._all_nodes(node, "timelineItems", is_pull):
            kind = item.get("__typename")
            if kind in EVENT_TYPES:
                harvest.events.append(_event_datapoint(item, number))
            elif kind == "CrossReferencedEvent" and self._is_internal_reference(item):
                harvest.timeline.append(_timeline_datapoint(item, number))
        return issue

    def _all_nodes(self, node: dict, connection: str, is_pull: bool) -> Iterable[dict]:
        """Every node of a nested connection, following its further pages.

        A connection only ever returns its first page with the parent, so the
        rest is read here. Leaving it out would silently drop the comments,
        reviews or events of the busiest issues.
        """
        nodes = list(node[connection]["nodes"])
        page_info = node[connection]["pageInfo"]
        query = more_query(connection, is_pull)
        variables: dict[str, Any] = {"id": node["id"], "nested": NESTED_PAGE_SIZE}
        if connection == "timelineItems":
            variables["types"] = (
                PULL_EVENT_ITEM_TYPES if is_pull else ISSUE_EVENT_ITEM_TYPES
            )
        while page_info["hasNextPage"]:
            LOG.debug("Reading another page of %s for #%s", connection, node["number"])
            data = self._run(query, {**variables, "cursor": page_info["endCursor"]})
            further = data["node"][connection]
            nodes.extend(further["nodes"])
            page_info = further["pageInfo"]
        return nodes

    def _is_internal_reference(self, item: dict) -> bool:
        """Whether a cross-reference points back into the mined repository."""
        source = item.get("source") or {}
        repository = source.get("repository") or {}
        return (
            not item.get("isCrossRepository", False)
            and repository.get("nameWithOwner") == self.full_project_name
            and source.get("number") is not None
        )


def _timestamp(value: str) -> datetime:
    """Parses a GraphQL DateTime into a timezone aware datetime."""
    return datetime.fromisoformat(value)


def _optional_timestamp(value: str | None) -> datetime | None:
    """Parses a GraphQL DateTime that the schema allows to be null."""
    return _timestamp(value) if value else None


def _database_id(value: str | int | None) -> int:
    """The numeric id the REST API reports for the same object.

    `fullDatabaseId` replaces the deprecated `databaseId` on issues, pull
    requests, comments and reviews. It is a BigInt, which the schema encodes
    as a string.
    """
    return int(value) if value is not None else 0


def _actor(node: dict | None) -> ActorInfo | None:
    """Turns an Actor of the GraphQL schema into an ActorInfo.

    `__typename` carries what the REST API reports as the account type, so the
    type of an actor costs no request of its own here.
    """
    if not node:
        return None
    return ActorInfo(
        # The schema types an Actor's login as non-null
        name=node["login"],
        # The schema types email as non-null on a User and nullable on the
        # other actors, which conflicts unless the latter are aliased. An
        # unknown email is reported as the empty string, where the REST API
        # reports it as missing
        email=node.get("email")
        or node.get("organizationEmail")
        or node.get("mannequinEmail")
        or None,
        github_id=node.get("databaseId"),
        type=node.get("__typename"),
    )


def _commit_account(signature: dict | None) -> ActorInfo | None:
    """The account behind a commit signature, or None where GitHub resolved none.

    A signature always carries the name and address git recorded; its `user`
    is filled only where GitHub recognises that address as one of an account's
    verified ones.
    """
    user = (signature or {}).get("user")
    return _actor(user) if user else None


def _sentiment(body: str | None) -> tuple[float, float]:
    sentiment = TextBlob(body if body else "").sentiment
    return sentiment.polarity, sentiment.subjectivity


def _issue_datapoint(node: dict, is_pull: bool, author: ActorInfo) -> IssueDatapoint:
    polarity, subjectivity = _sentiment(node.get("body"))
    merged = bool(node.get("merged", False))
    return IssueDatapoint(
        id=_database_id(node["fullDatabaseId"]),
        body=node.get("body"),
        sentiment_polarity=polarity,
        sentiment_subjectivity=subjectivity,
        mentioned_actors=_mentioned_actors(node.get("bodyHTML")),
        created_at=_timestamp(node["createdAt"]),
        updated_at=_timestamp(node["updatedAt"]),
        issue=IssueInfo(node["number"]),
        author=author,
        assignees=[
            actor
            for actor in (_actor(a) for a in node["assignees"]["nodes"])
            if actor is not None
        ],
        closed_at=_optional_timestamp(node.get("closedAt")),
        is_pull=is_pull,
        is_merged=merged,
        merged_at=_optional_timestamp(node.get("mergedAt")) if merged else None,
        merged_by=_actor(node.get("mergedBy")) if merged else None,
        # The REST API reports the state in lower case, and reports a merged
        # pull request as closed
        state="closed" if node["state"] in ("CLOSED", "MERGED") else "open",
    )


def _comment_datapoint(
    node: dict, issue_number: int, author: ActorInfo
) -> IssueCommentDatapoint:
    polarity, subjectivity = _sentiment(node.get("body"))
    counts = {name: 0 for name in REACTIONS.values()}
    total = 0
    for group in node.get("reactionGroups") or []:
        count = group["reactors"]["totalCount"]
        total += count
        field_name = REACTIONS.get(group["content"])
        if field_name:
            counts[field_name] = count
    return IssueCommentDatapoint(
        issue=IssueInfo(number=issue_number),
        id=_database_id(node["fullDatabaseId"]),
        author=author,
        mentioned_actors=_mentioned_actors(node.get("bodyHTML")),
        created_at=_timestamp(node["createdAt"]),
        updated_at=_timestamp(node["updatedAt"]),
        body=node.get("body"),
        sentiment_polarity=polarity,
        sentiment_subjectivity=subjectivity,
        n_reactions=total,
        **counts,
    )


def _review_datapoint(
    node: dict, pull_number: int, author: ActorInfo
) -> ReviewDatapoint:
    polarity, subjectivity = _sentiment(node.get("body"))
    return ReviewDatapoint(
        id=_database_id(node["fullDatabaseId"]),
        pull=IssueInfo(pull_number),
        body=node.get("body"),
        sentiment_polarity=polarity,
        sentiment_subjectivity=subjectivity,
        submitted_at=_timestamp(node["submittedAt"]),
        author=author,
        review_state=node["state"],
    )


def _event_datapoint(node: dict, issue_number: int) -> IssueEventDatapoint:
    """Turns a timeline item into the event the REST API reports.

    The REST API names the actor of an event differently per event type, so
    the one GraphQL actor is carried into the matching field.
    """
    kind = node["__typename"]
    actor = _actor(node.get("actor"))
    return IssueEventDatapoint(
        event_type=EVENT_TYPES[kind],
        issue=IssueInfo(issue_number),
        requester=actor if kind == "ReviewRequestedEvent" else None,
        req_reviewer=_actor(node.get("requestedReviewer")),
        created_at=_timestamp(node["createdAt"]),
        assignee=_actor(node.get("assignee")),
        assigner=actor if kind == "AssignedEvent" else None,
        actor=actor,
    )


def _timeline_datapoint(node: dict, issue_number: int) -> IssueTimelineEventDatapoint:
    source = node.get("source") or {}
    return IssueTimelineEventDatapoint(
        event_type=CROSS_REFERENCED,
        issue=IssueInfo(issue_number),
        created_at=_timestamp(node["createdAt"]),
        actor=_actor(node.get("actor")),
        source_issue=IssueInfo(number=source["number"]),
    )
