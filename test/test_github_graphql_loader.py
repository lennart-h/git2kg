import unittest
from datetime import datetime, timezone

from github import GithubException

import git2kg.dataloading.githubgraphqlloader as githubgraphqlloader
from git2kg.dataloading.githubgraphqlloader import (
    GithubGraphQLLoader,
    _mentioned_actors,
)

# Verbatim from GitHub's rendering of airbnb/mavericks#726, which mentions two
# accounts in prose and contains "@Composable" inside a code block. GitHub
# links the accounts and leaves the annotation as text.
RENDERED_BODY = """
<p>Thanks <a class="user-mention notranslate" data-hovercard-type="user"
 data-hovercard-url="/users/elihart/hovercard" data-octo-click="hovercard-link-click"
 href="https://github.com/elihart">@elihart</a>, cc <a class="user-mention notranslate"
 data-hovercard-type="user" data-hovercard-url="/users/piontienbv/hovercard"
 href="https://github.com/piontienbv">@piontienbv</a>.</p>
<div class="highlight"><pre><code>@Composable
fun Screen() { }
@Composable
fun Other() { }
</code></pre></div>
<p>Also see @Composable usage.</p>
"""

TEAM_MENTION = (
    '<a class="team-mention notranslate" data-hovercard-type="team"'
    ' href="https://github.com/orgs/airbnb/teams/android">@airbnb/android</a>'
)


def _actor(login, database_id=1, typename="User", email=None):
    return {
        "__typename": typename,
        "login": login,
        "databaseId": database_id,
        "email": email or "",
    }


def _issue_node(
    number,
    *,
    node_id=None,
    comments=None,
    timeline=None,
    updated="2024-01-02T00:00:00Z",
):
    return {
        "id": node_id or f"I_{number}",
        "number": number,
        # BigInt comes back as a string
        "fullDatabaseId": str(1000 + number),
        "body": "body",
        "bodyHTML": "<p>body</p>",
        "state": "CLOSED",
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": updated,
        "closedAt": "2024-01-03T00:00:00Z",
        "author": _actor("alice"),
        "assignees": {"nodes": [_actor("bob", 2)]},
        "comments": comments
        or {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []},
        "timelineItems": timeline
        or {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []},
    }


def _comment_node(database_id, body_html="<p>hi</p>", reactions=None):
    return {
        "fullDatabaseId": str(database_id),
        "body": "hi",
        "bodyHTML": body_html,
        "createdAt": "2024-01-02T00:00:00Z",
        "updatedAt": "2024-01-02T00:00:00Z",
        "author": _actor("carol", 3),
        "reactionGroups": reactions or [],
    }


class _StubLoader(GithubGraphQLLoader):
    """A loader whose queries are answered from canned pages."""

    def __init__(self, issue_pages, pull_pages=None, more_pages=None):
        self.organization_name = "airbnb"
        self.repo_name = "mavericks"
        self.full_project_name = "airbnb/mavericks"
        self._harvests = {}
        self._issue_pages = list(issue_pages)
        self._pull_pages = list(pull_pages or [_empty_page("pullRequests")])
        self._more_pages = list(more_pages or [])
        self.queries = []
        # A token with every scope, so the conditional actor fields are asked for
        self._scopes = frozenset({"public_repo", "read:user", "read:org"})

    def _run(self, query, variables):
        self.queries.append(variables)
        if "node(id:" in query:
            return self._more_pages.pop(0)
        if "pullRequests(" in query:
            return self._pull_pages.pop(0)
        return self._issue_pages.pop(0)


def _empty_page(connection):
    return {
        "repository": {
            connection: {
                "totalCount": 0,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [],
            }
        }
    }


def _page(connection, nodes, has_next=False, cursor=None, total=None):
    return {
        "repository": {
            connection: {
                "totalCount": total if total is not None else len(nodes),
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                "nodes": nodes,
            }
        }
    }


class TestMentionExtraction(unittest.TestCase):
    def test_reads_the_accounts_github_linked(self):
        actors = _mentioned_actors(RENDERED_BODY)

        self.assertListEqual(["elihart", "piontienbv"], [a.name for a in actors])

    def test_ignores_an_annotation_github_did_not_link(self):
        # "@Composable" appears three times, in a code block and in prose, and
        # GitHub linked none of them
        self.assertNotIn(
            "Composable", [a.name for a in _mentioned_actors(RENDERED_BODY)]
        )

    def test_ignores_team_mentions(self):
        self.assertListEqual([], _mentioned_actors(TEAM_MENTION))

    def test_mentioning_an_account_twice_yields_one_actor(self):
        twice = RENDERED_BODY + RENDERED_BODY

        self.assertListEqual(
            ["elihart", "piontienbv"], [a.name for a in _mentioned_actors(twice)]
        )

    def test_empty_body(self):
        self.assertListEqual([], _mentioned_actors(None))
        self.assertListEqual([], _mentioned_actors(""))


class TestIssueLoading(unittest.TestCase):
    def test_maps_an_issue_to_the_rest_representation(self):
        loader = _StubLoader([_page("issues", [_issue_node(7)])])

        issue = list(loader.load_github_issues())[0]

        self.assertEqual(7, issue.issue.number)
        self.assertEqual(1007, issue.id)
        self.assertFalse(issue.is_pull)
        # The REST API reports the state in lower case
        self.assertEqual("closed", issue.state)
        self.assertEqual("alice", issue.author.name)
        self.assertEqual("User", issue.author.type)
        self.assertListEqual(["bob"], [a.name for a in issue.assignees])
        self.assertEqual(datetime(2024, 1, 1, tzinfo=timezone.utc), issue.created_at)

    def test_an_empty_email_becomes_none(self):
        loader = _StubLoader([_page("issues", [_issue_node(7)])])

        issue = list(loader.load_github_issues())[0]

        self.assertIsNone(issue.author.email)

    def test_skips_an_issue_whose_account_is_gone(self):
        node = _issue_node(7)
        node["author"] = None
        loader = _StubLoader([_page("issues", [node])])

        self.assertListEqual([], list(loader.load_github_issues()))

    def test_follows_the_pages_of_the_issue_connection(self):
        loader = _StubLoader(
            [
                _page("issues", [_issue_node(1)], has_next=True, cursor="c1", total=2),
                _page("issues", [_issue_node(2)]),
            ]
        )

        self.assertListEqual(
            [1, 2], [i.issue.number for i in loader.load_github_issues()]
        )


class TestNestedPagination(unittest.TestCase):
    def test_reads_the_comments_beyond_the_first_page(self):
        first = {
            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
            "nodes": [_comment_node(1)],
        }
        more = {
            "node": {
                "number": 7,
                "comments": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [_comment_node(2)],
                },
            }
        }
        loader = _StubLoader(
            [_page("issues", [_issue_node(7, comments=first)])], more_pages=[more]
        )

        comments = list(loader.load_github_issues_comments())

        # Dropping the second page would silently lose the comments of the
        # busiest issues
        self.assertListEqual([1, 2], [c.id for c in comments])
        self.assertListEqual([7, 7], [c.issue.number for c in comments])

    def test_reads_the_timeline_beyond_the_first_page(self):
        event = {
            "__typename": "ClosedEvent",
            "createdAt": "2024-01-03T00:00:00Z",
            "actor": _actor("dave", 4),
        }
        first = {"pageInfo": {"hasNextPage": True, "endCursor": "c1"}, "nodes": []}
        more = {
            "node": {
                "number": 7,
                "timelineItems": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [event],
                },
            }
        }
        loader = _StubLoader(
            [_page("issues", [_issue_node(7, timeline=first)])], more_pages=[more]
        )

        events = list(loader.load_github_issue_events())

        self.assertEqual(1, len(events))
        self.assertEqual("closed", events[0].event_type)


class TestEventMapping(unittest.TestCase):
    def _event(self, item):
        loader = _StubLoader(
            [
                _page(
                    "issues",
                    [
                        _issue_node(
                            7,
                            timeline={
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [item],
                            },
                        )
                    ],
                )
            ]
        )
        return list(loader.load_github_issue_events())

    def test_closed_and_reopened_use_the_rest_event_names(self):
        for typename, expected in (
            ("ClosedEvent", "closed"),
            ("ReopenedEvent", "reopened"),
        ):
            with self.subTest(typename):
                events = self._event(
                    {
                        "__typename": typename,
                        "createdAt": "2024-01-03T00:00:00Z",
                        "actor": _actor("dave", 4),
                    }
                )
                self.assertEqual(expected, events[0].event_type)

    def test_assigned_puts_the_actor_in_the_assigner_field(self):
        events = self._event(
            {
                "__typename": "AssignedEvent",
                "createdAt": "2024-01-03T00:00:00Z",
                "actor": _actor("dave", 4),
                "assignee": _actor("erin", 5),
            }
        )

        self.assertEqual("assigned", events[0].event_type)
        self.assertEqual("dave", events[0].assigner.name)
        self.assertEqual("erin", events[0].assignee.name)
        self.assertIsNone(events[0].requester)

    def test_review_requested_puts_the_actor_in_the_requester_field(self):
        events = self._event(
            {
                "__typename": "ReviewRequestedEvent",
                "createdAt": "2024-01-03T00:00:00Z",
                "actor": _actor("dave", 4),
                "requestedReviewer": _actor("erin", 5),
            }
        )

        self.assertEqual("review_requested", events[0].event_type)
        self.assertEqual("dave", events[0].requester.name)
        self.assertEqual("erin", events[0].req_reviewer.name)
        self.assertIsNone(events[0].assigner)


class TestCrossReferences(unittest.TestCase):
    def _timeline(self, item):
        loader = _StubLoader(
            [
                _page(
                    "issues",
                    [
                        _issue_node(
                            7,
                            timeline={
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [item],
                            },
                        )
                    ],
                )
            ]
        )
        return list(loader.load_github_issue_event_timeline())

    def _reference(self, owner, cross_repository):
        return {
            "__typename": "CrossReferencedEvent",
            "createdAt": "2024-01-04T00:00:00Z",
            "actor": _actor("dave", 4),
            "isCrossRepository": cross_repository,
            "source": {"number": 42, "repository": {"nameWithOwner": owner}},
        }

    def test_keeps_a_reference_from_the_same_repository(self):
        events = self._timeline(self._reference("airbnb/mavericks", False))

        self.assertEqual(1, len(events))
        self.assertEqual("cross-referenced", events[0].event_type)
        self.assertEqual(42, events[0].source_issue.number)

    def test_drops_a_reference_from_another_repository(self):
        self.assertListEqual([], self._timeline(self._reference("other/repo", True)))


class TestReactions(unittest.TestCase):
    def test_counts_every_reaction_kind(self):
        reactions = [
            {"content": "THUMBS_UP", "reactors": {"totalCount": 3}},
            {"content": "THUMBS_DOWN", "reactors": {"totalCount": 1}},
            {"content": "ROCKET", "reactors": {"totalCount": 2}},
        ]
        comments = {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [_comment_node(1, reactions=reactions)],
        }
        loader = _StubLoader([_page("issues", [_issue_node(7, comments=comments)])])

        comment = list(loader.load_github_issues_comments())[0]

        self.assertEqual(3, comment.upvotes)
        self.assertEqual(1, comment.downvotes)
        self.assertEqual(2, comment.rocket)
        self.assertEqual(0, comment.heart)
        self.assertEqual(6, comment.n_reactions)


class TestPullRequests(unittest.TestCase):
    def _pull_node(self, number, merged=True, reviews=None):
        node = _issue_node(number, node_id=f"PR_{number}")
        node["state"] = "MERGED" if merged else "OPEN"
        node["merged"] = merged
        node["mergedAt"] = "2024-01-05T00:00:00Z" if merged else None
        node["mergedBy"] = _actor("frank", 6) if merged else None
        node["reviews"] = reviews or {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [],
        }
        return node

    def test_a_merged_pull_request_reports_as_closed(self):
        loader = _StubLoader(
            [_empty_page("issues")], [_page("pullRequests", [self._pull_node(9)])]
        )

        pull = list(loader.load_github_issues())[0]

        self.assertTrue(pull.is_pull)
        self.assertTrue(pull.is_merged)
        self.assertEqual("closed", pull.state)
        self.assertEqual("frank", pull.merged_by.name)
        self.assertEqual(datetime(2024, 1, 5, tzinfo=timezone.utc), pull.merged_at)

    def test_reviews_carry_the_pull_request_number(self):
        reviews = {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [
                {
                    "fullDatabaseId": "55",
                    "body": "lgtm",
                    "state": "APPROVED",
                    "submittedAt": "2024-01-04T00:00:00Z",
                    "author": _actor("grace", 7),
                }
            ],
        }
        loader = _StubLoader(
            [_empty_page("issues")],
            [_page("pullRequests", [self._pull_node(9, reviews=reviews)])],
        )

        review = list(loader.load_github_reviews())[0]

        self.assertEqual(55, review.id)
        self.assertEqual(9, review.pull.number)
        self.assertEqual("APPROVED", review.review_state)
        self.assertEqual("grace", review.author.name)

    def test_stops_at_what_was_mined_before(self):
        # The pull request connection takes no since filter, so it is read
        # newest first and stopped on the watermark
        newer = self._pull_node(9)
        newer["updatedAt"] = "2024-03-01T00:00:00Z"
        older = self._pull_node(8)
        older["updatedAt"] = "2024-01-01T00:00:00Z"
        loader = _StubLoader(
            [_empty_page("issues")],
            [_page("pullRequests", [newer, older], has_next=True, cursor="c1")],
        )

        pulls = list(
            loader.load_github_issues(_from=datetime(2024, 2, 1, tzinfo=timezone.utc))
        )

        self.assertListEqual([9], [p.issue.number for p in pulls])


class TestWatermark(unittest.TestCase):
    def test_the_two_queries_run_once_per_watermark(self):
        loader = _StubLoader(
            [_page("issues", [_issue_node(7)])] * 2,
            [_empty_page("pullRequests")] * 2,
        )

        list(loader.load_github_issues())
        list(loader.load_github_issues_comments())
        list(loader.load_github_issue_events())

        # One issue query and one pull request query, shared by all three
        self.assertEqual(2, len(loader.queries))


class TestStreaming(unittest.TestCase):
    """Issues are handed on as their page arrives, so that a mine that fails
    keeps what it had persisted."""

    def _loader(self, pages):
        return _StubLoader(pages, [_empty_page("pullRequests")])

    def test_yields_the_first_page_before_fetching_the_second(self):
        loader = self._loader(
            [
                _page("issues", [_issue_node(1)], has_next=True, cursor="c1", total=2),
                _page("issues", [_issue_node(2)]),
            ]
        )

        stream = loader.load_github_issues()
        first = next(stream)

        self.assertEqual(1, first.issue.number)
        # Only the first page has been asked for at this point
        self.assertEqual(1, len(loader.queries))

    def test_an_abandoned_stream_leaves_no_harvest_behind(self):
        loader = self._loader(
            [
                _page("issues", [_issue_node(1)], has_next=True, cursor="c1", total=2),
                _page("issues", [_issue_node(2)]),
            ]
        )

        stream = loader.load_github_issues()
        next(stream)
        stream.close()

        # A half filled harvest must not be read as a complete one
        self.assertEqual({}, loader._harvests)

    def test_the_other_methods_still_share_one_fetch(self):
        comments = {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [_comment_node(1)],
        }
        loader = _StubLoader(
            [_page("issues", [_issue_node(7, comments=comments)])],
            [_empty_page("pullRequests")],
        )

        issues = list(loader.load_github_issues())
        queries_after_issues = len(loader.queries)
        comments_mined = list(loader.load_github_issues_comments())

        self.assertEqual(1, len(issues))
        self.assertEqual(1, len(comments_mined))
        # Reading the comments costs no further query
        self.assertEqual(queries_after_issues, len(loader.queries))


class TestEmptyBodyRetry(unittest.TestCase):
    """GitHub answers a query it cannot finish with a success status and no
    body, which ends a mine unless it is retried."""

    class _Requester:
        base_url = "https://api.github.com"
        graphql_url = "https://api.github.com/graphql"

        def __init__(self, empty_bodies, page):
            self.empty_bodies = empty_bodies
            self.page = page
            self.calls = 0

        def requestJsonAndCheck(self, verb, url, **kwargs):
            if url.endswith("/user"):
                return {"x-oauth-scopes": "public_repo, read:user"}, {}
            self.calls += 1
            if self.calls <= self.empty_bodies:
                return {}, None
            query = kwargs["input"]["query"]
            page = (
                _empty_page("pullRequests") if "pullRequests(" in query else self.page
            )
            return {}, {"data": page}

    def _loader(self, empty_bodies):
        loader = GithubGraphQLLoader.__new__(GithubGraphQLLoader)
        loader.organization_name, loader.repo_name = "airbnb", "mavericks"
        loader.full_project_name = "airbnb/mavericks"
        loader._harvests, loader._scopes = {}, None
        loader._requester = self._Requester(
            empty_bodies, _page("issues", [_issue_node(7)])
        )
        return loader

    def setUp(self):
        self._slept = []
        self._sleep = githubgraphqlloader.time.sleep
        githubgraphqlloader.time.sleep = self._slept.append

    def tearDown(self):
        githubgraphqlloader.time.sleep = self._sleep

    def test_retries_an_empty_body_and_carries_on(self):
        loader = self._loader(empty_bodies=2)

        issues = list(loader.load_github_issues())

        self.assertEqual([7], [i.issue.number for i in issues])
        # Two empty answers, then the issues page and the pull request page
        self.assertEqual(4, loader._requester.calls)
        self.assertEqual([2, 4], self._slept)

    def test_gives_up_with_a_clear_error(self):
        loader = self._loader(empty_bodies=99)

        with self.assertRaises(GithubException) as caught:
            list(loader.load_github_issues())

        self.assertIn("empty body", str(caught.exception))
        self.assertEqual(
            githubgraphqlloader.EMPTY_BODY_ATTEMPTS, loader._requester.calls
        )


if __name__ == "__main__":
    unittest.main()
