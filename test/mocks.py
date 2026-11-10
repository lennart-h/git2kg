from github.Issue import Issue
from github.IssueComment import IssueComment
from github.IssueEvent import IssueEvent
from github.NamedUser import NamedUser
from github.PullRequest import PullRequest
from github.PullRequestReview import PullRequestReview
from github.Repository import Repository
from github.TimelineEvent import TimelineEvent

# Mock Users
curie_attributes = {
    "login": "mariecurie",
    "email": "marie.curie+github@radioactive.pl",
    "id": 1867,
    "type": "User",
}
turing_attributes = {"login": "alanturing", "email": None, "id": 1912, "type": "User"}
zuse_attributes = {
    "login": "Konny Zuse",
    "email": "zuse@zusekg.de",
    "id": 1910,
    "type": "User",
}
franklin_attributes = {
    "login": "rosafranklin1920",
    "email": "franklin@kcl.ac.uk",
    "id": 1920,
    "type": "User",
}


class NullRequester:
    def __init__(self):
        self.is_not_lazy = False


marie_curie = NamedUser(
    requester=NullRequester(), headers=None, attributes=curie_attributes, completed=True
)
alan_turing = NamedUser(
    requester=NullRequester(),
    headers=None,
    attributes=turing_attributes,
    completed=True,
)
konrad_zuse = NamedUser(
    requester=NullRequester(), headers=None, attributes=zuse_attributes, completed=True
)
rosalind_franklin = NamedUser(
    requester=NullRequester(),
    headers=None,
    attributes=franklin_attributes,
    completed=True,
)

## Mock repository
repo_attributes = {"full_name": "scientists/repo"}
repo = Repository(
    requester=NullRequester(), headers=None, attributes=repo_attributes, completed=True
)

# Mock issues
iss_1_attributes = {
    "id": 24,
    "body": "I really dislike this feature, just because I despise it",
    "number": 1,
    "created_at": "2008-01-14T04:33:35Z",
    "updated_at": None,
    "user": curie_attributes,
    "assignee": turing_attributes,
    "repository": repo_attributes,
    "closed_at": None,
    "pull_request": None,
    "state": "open",
}


iss_2_attributes = {
    "id": 42,
    "body": "We might need to alter the codebase for slightly better readability",
    "number": 2,
    "created_at": "2009-02-15T05:34:36Z",
    "updated_at": "2009-03-15T05:34:36Z",
    "user": curie_attributes,
    "repository": repo_attributes,
    "closed_at": "2009-04-15T10:34:36Z",
    "pull_request": {"some": "object"},
    "state": "closed",
}
# None user issue
iss_3_attributes = {
    "id": 42,
    "body": "We might need to alter the codebase for slightly better readability",
    "number": 2,
    "created_at": "2009-02-15T05:34:36Z",
    "updated_at": "2009-03-15T05:34:36Z",
    "assignee": zuse_attributes,
    "repository": repo_attributes,
    "closed_at": "2009-04-15T10:34:36Z",
    "pull_request": {"some": "object"},
    "state": "closed",
}

# Issue comments
iss_comment_11_attributes = {
    "issue_url": "https://api.github.com/scientists/repo/issues/1",
    "id": 45,
    "user": turing_attributes,
    "created_at": "2008-01-14T04:45:35Z",
    "updated_at": None,
    "body": "No",
    "reactions": {
        "url": "https://api.github.com/scientists/repo/issues/comments/45/reactions",
        "total_count": 2,
        "+1": 0,
        "-1": 0,
        "laugh": 0,
        "hooray": 1,
        "confused": 0,
        "heart": 0,
        "rocket": 0,
        "eyes": 1,
    },
}


iss_comment_12_attributes = {
    "issue_url": "https://api.github.com/scientists/repo/issues/1",
    "id": 46,
    "user": turing_attributes,
    "created_at": "2008-01-14T05:33:35Z",
    "updated_at": None,
    "body": "I completely disagree",
    "reactions": {
        "url": "https://api.github.com/scientists/repo/issues/comments/46/reactions",
        "total_count": 0,
        "+1": 0,
        "-1": 0,
        "laugh": 0,
        "hooray": 0,
        "confused": 0,
        "heart": 0,
        "rocket": 0,
        "eyes": 0,
    },
}


iss_comment_21_attributes = {
    "issue_url": "https://api.github.com/scientists/repo/issues/2",
    "id": 47,
    "user": zuse_attributes,
    "created_at": "2009-03-15T08:59:36Z",
    "updated_at": None,
    "body": "I willll handle it",
    "reactions": {
        "url": "https://api.github.com/scientists/repo/issues/comments/47/reactions",
        "total_count": 1,
        "+1": 1,
        "-1": 0,
        "laugh": 0,
        "hooray": 0,
        "confused": 0,
        "heart": 0,
        "rocket": 0,
        "eyes": 0,
    },
}

iss_comment_22_attributes = {
    "issue_url": "https://api.github.com/scientists/repo/issues/2",
    "id": 48,
    "user": franklin_attributes,
    "created_at": "2009-04-15T10:30:36Z",
    "updated_at": None,
    "body": "I think this is done now",
    "reactions": {
        "url": "https://api.github.com/scientists/repo/issues/comments/48/reactions",
        "total_count": 1,
        "+1": 0,
        "-1": 0,
        "laugh": 0,
        "hooray": 0,
        "confused": 0,
        "heart": 1,
        "rocket": 0,
        "eyes": 0,
    },
}
# None user comment
iss_comment_23_attributes = {
    "issue_url": "https://api.github.com/scientists/repo/issues/2",
    "id": 48,
    "created_at": "2009-04-15T10:30:36Z",
    "updated_at": None,
    "body": "I think this is done now",
    "reactions": {
        "url": "https://api.github.com/scientists/repo/issues/comments/48/reactions",
        "total_count": 1,
        "+1": 0,
        "-1": 0,
        "laugh": 0,
        "hooray": 0,
        "confused": 0,
        "heart": 1,
        "rocket": 0,
        "eyes": 0,
    },
}

## Issue events
event_1_attributes = {
    "review_requester": turing_attributes,
    "requested_reviewer": curie_attributes,
    "created_at": "2008-01-14T04:45:35Z",
    "event": "review_requested",
    "issue": iss_1_attributes,
}

event_2_attributes = {
    "actor": franklin_attributes,
    "review_requester": None,
    "requested_reviewer": None,
    "created_at": "2008-01-14T04:45:35Z",
    "event": "closed",
    "issue": iss_2_attributes,
}

cross_referenced_event_attributes = {
    "actor": turing_attributes,
    "created_at": "2012-01-14T04:45:35Z",
    "event": "cross-referenced",
    "source": {"type": "issue", "issue": iss_2_attributes},
}

review_1_attributes = {
    "pull_request_url": "https://api.github.com/scientists/repo/pulls/2",
    "body": "LGTM",
    "submitted_at": "2009-04-15T10:39:30Z",
    "user": franklin_attributes,
    "state": "APPROVED",
}

review_2_attributes = {
    "pull_request_url": "https://api.github.com/scientists/repo/pulls/1",
    "body": "",
    "submitted_at": "2009-04-15T10:39:30Z",
    "user": curie_attributes,
    "state": "PENDING",
}
review_3_attributes = {
    "pull_request_url": "https://api.github.com/scientists/repo/pulls/1",
    "body": "Do some minor changes",
    "submitted_at": "2009-04-15T10:39:30Z",
    "user": zuse_attributes,
    "state": "REQUEST_CHANGES",
}
# None user review
review_4_attributes = {
    "pull_request_url": "https://api.github.com/scientists/repo/pulls/1",
    "body": "Do some minor changes",
    "submitted_at": "2009-04-15T10:39:30Z",
    "state": "REQUEST_CHANGES",
}

## Create issue mock objects
issue_1 = Issue(
    requester=NullRequester(), headers=None, attributes=iss_1_attributes, completed=True
)
issue_2 = Issue(
    requester=NullRequester(), headers=None, attributes=iss_2_attributes, completed=True
)
issue_3 = Issue(
    requester=NullRequester(), headers=None, attributes=iss_3_attributes, completed=True
)

## Create issue comment mock objects
issue_comment_11 = IssueComment(
    requester=NullRequester(),
    headers=None,
    attributes=iss_comment_11_attributes,
    completed=True,
)
issue_comment_12 = IssueComment(
    requester=NullRequester(),
    headers=None,
    attributes=iss_comment_12_attributes,
    completed=True,
)
issue_comment_21 = IssueComment(
    requester=NullRequester(),
    headers=None,
    attributes=iss_comment_21_attributes,
    completed=True,
)
issue_comment_22 = IssueComment(
    requester=NullRequester(),
    headers=None,
    attributes=iss_comment_22_attributes,
    completed=True,
)
issue_comment_23 = IssueComment(
    requester=NullRequester(),
    headers=None,
    attributes=iss_comment_23_attributes,
    completed=True,
)
## Issues events mock objects
issue_event_1 = IssueEvent(
    requester=NullRequester(),
    headers=None,
    attributes=event_1_attributes,
    completed=True,
)
issue_event_2 = IssueEvent(
    requester=NullRequester(), 
    headers=None,
    attributes=event_2_attributes,
    completed=True,
)

issue_cross_referenced_event = TimelineEvent(
    requester=NullRequester(),
    headers=None,
    attributes=cross_referenced_event_attributes,
)


## PR revies mock objects
review_1 = PullRequestReview(
    requester=NullRequester(),
    headers=None,
    attributes=review_1_attributes,
)

review_2 = PullRequestReview(
    requester=NullRequester(),
    headers=None,
    attributes=review_2_attributes,
)

review_3 = PullRequestReview(
    requester=NullRequester(),
    headers=None,
    attributes=review_3_attributes,
)
review_4 = PullRequestReview(
    requester=NullRequester(),
    headers=None,
    attributes=review_4_attributes,
)

pull_2_attributes = iss_2_attributes.copy()
pull_2_attributes["merged"] = True
pull_2_attributes["merged_by"] = curie_attributes
pull_2_attributes["merged_at"] = "2009-02-16T10:34:36Z"
## Reuse issues as pull requests for testing
pull_1 = PullRequest(
    requester=NullRequester(), headers=None, attributes=iss_1_attributes, completed=True
)
pull_2 = PullRequest(
    requester=NullRequester(),
    headers=None,
    attributes=pull_2_attributes,
    completed=True,
)


class MockedGithubList:
    def __init__(self, _list: list):
        self._list = _list

    def __iter__(self):
        return self._list.__iter__()

    @property
    def totalCount(self):
        return len(self._list)
