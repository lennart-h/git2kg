import unittest
from datetime import UTC, datetime
from typing import Generator
from unittest.mock import patch

import pandas as pd

from git2kg.dataloading.githubloader import GithubLoader
from git2kg.datapoint import ActorInfo, IssueInfo
from test import mocks

marie_curie = ActorInfo(
    "mariecurie",
    email="marie.curie+github@radioactive.pl",
    github_id=1867,
    type="User",
)
alan_turing = ActorInfo(name="alanturing", github_id=1912, type="User")
konrad_zuse = ActorInfo(name="Konny Zuse", email="zuse@zusekg.de", github_id=1910, type="User")
rosalind_franklin = ActorInfo(
    name="rosafranklin1920", email="franklin@kcl.ac.uk", github_id=1920, type="User"
)


class TestGithubloader(unittest.TestCase):
    @patch("github.Github.get_repo")
    def setUp(self, mock_repo):
        mock_repo.return_value = mocks.repo
        self.uut = GithubLoader("scientists", "repo", "sometoken")

    @patch("github.Issue.Issue.as_pull_request")
    @patch("github.Repository.Repository.get_issues")
    @patch("github.Github.get_repo")
    def test_load_issues(self, mock_repo, mock_issues, mock_as_pull):
        mock_repo.return_value = mocks.repo
        mock_issues.return_value = mocks.MockedGithubList(
            [mocks.issue_1, mocks.issue_2, mocks.issue_3]
        )
        mock_as_pull.return_value = mocks.pull_2
        issues = self.uut.load_github_issues()
        issue_list = list(issues)
        self.assertIsInstance(issues, Generator)

        # Only issues with users should be mined
        self.assertEqual(len(issue_list), 2)
     
        self.assertListEqual(
            [iss.author for iss in issue_list],
            [marie_curie, marie_curie],
        )
        self.assertEqual(list(map(lambda x: x.assignees, issue_list)), [
            [alan_turing],
            []
        ])
    

        self.assertEqual(list(map(lambda x: x.issue, issue_list)), [IssueInfo(number=1), IssueInfo(number=2)])
        self.assertEqual(
            list(map(lambda x: x.created_at, issue_list)),
            [
                datetime.strptime("2008-01-14T04:33:35+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2009-02-15T05:34:36+0000", "%Y-%m-%dT%H:%M:%S%z"),
            ],
        )
        self.assertEqual(list(map(lambda x: x.is_pull, issue_list)), [False, True])
        self.assertEqual(list(map(lambda x: x.is_merged, issue_list)), [False, True])
        self.assertEqual(list(map(lambda x: x.merged_at, issue_list)), [None, datetime(2009,2,16,10,34,36, tzinfo=UTC)])
        self.assertEqual(list(map(lambda x: x.merged_by, issue_list)), [None, marie_curie])
                
        self.assertEqual(list(map(lambda x: x.state, issue_list)), ["open", "closed"])

        self.assertEqual(
            list(map(lambda x: x.updated_at, issue_list)),
            [
                None,
                datetime.strptime("2009-03-15T05:34:36+0000", "%Y-%m-%dT%H:%M:%S%z"),
            ],
        )

    @patch("github.Repository.Repository.get_issues_comments")
    @patch("github.Github.get_repo")
    def test_load_issues_comments(self, mock_repo, mock_issues_commments):
        mock_repo.return_value = mocks.repo
        mock_issues_commments.return_value = mocks.MockedGithubList(
            [
                mocks.issue_comment_11,
                mocks.issue_comment_12,
                mocks.issue_comment_21,
                mocks.issue_comment_22,
                mocks.issue_comment_23,
            ]
        )
        issues_comments = self.uut.load_github_issues_comments()
        issue_comments_list = list(issues_comments)
        self.assertIsInstance(issues_comments, Generator)
        # Only comments with users should be mined
        self.assertEqual(len(issue_comments_list), 4)
        self.assertEqual(issue_comments_list[0].author, alan_turing)
        self.assertEqual(issue_comments_list[1].author, alan_turing)
        self.assertEqual(issue_comments_list[2].author, konrad_zuse)
        self.assertEqual(issue_comments_list[3].author, rosalind_franklin)

        self.assertEqual(
            list(map(lambda x: x.issue, issue_comments_list)), [IssueInfo(number=1), IssueInfo(number=1), IssueInfo(number=2), IssueInfo(number=2)]
        )
        self.assertEqual(
            list(map(lambda x: x.id, issue_comments_list)), [45, 46, 47, 48]
        )

        self.assertEqual(
            list(map(lambda x: x.created_at, issue_comments_list)),
            [
                datetime.strptime("2008-01-14T04:45:35+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2008-01-14T05:33:35+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2009-03-15T08:59:36+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2009-04-15T10:30:36+0000", "%Y-%m-%dT%H:%M:%S%z"),
            ],
        )
        self.assertEqual(
            list(map(lambda x: x.updated_at, issue_comments_list)),
            [None, None, None, None],
        )
        self.assertEqual(
            list(map(lambda x: x.n_reactions, issue_comments_list)), [2, 0, 1, 1]
        )

    @patch("github.Repository.Repository.get_pulls")
    @patch("github.PullRequest.PullRequest.get_reviews")
    @patch("github.Github.get_repo")
    def test_load_reviews(self, mock_repo, mock_reviews, mock_pulls):

        mock_repo.return_value = mocks.repo
        mock_reviews.side_effect = [
            mocks.MockedGithubList([mocks.review_1]),
            mocks.MockedGithubList([mocks.review_2, mocks.review_3, mocks.review_4]),
        ]
        mock_pulls.return_value = mocks.MockedGithubList([mocks.pull_1, mocks.pull_2])
        # Only reviews with user and non pending state should be mined
        reviews = self.uut.load_github_reviews()
        review_list = list(reviews)
        self.assertIsInstance(reviews, Generator)
        self.assertEqual(len(review_list), 2)

        self.assertEqual(review_list[0].author, rosalind_franklin)
        self.assertEqual(review_list[1].author, konrad_zuse)

        self.assertEqual(
            list(map(lambda x: x.review_state, review_list)),
            ["APPROVED", "REQUEST_CHANGES"],
        )

        self.assertEqual(
            list(map(lambda x: x.submitted_at, review_list)),
            [
                datetime.strptime("2009-04-15T10:39:30+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2009-04-15T10:39:30+0000", "%Y-%m-%dT%H:%M:%S%z"),
            ],
        )

    @patch("github.Repository.Repository.get_issues_events")
    @patch("github.Github.get_repo")
    def test_load_issue_events(self, mock_repo, mock_issues_events):
        mock_repo.return_value = mocks.repo
        mock_issues_events.return_value = mocks.MockedGithubList(
            [mocks.issue_event_1, mocks.issue_event_2]
        )

        review_requests = self.uut.load_github_issue_events()
        list_review_events = list(review_requests)
        self.assertEqual(len(list_review_events), 2)
        self.assertEqual(list_review_events[0].requester, alan_turing)
        self.assertEqual(list_review_events[1].requester, None)

        self.assertEqual(list_review_events[0].req_reviewer, marie_curie)
        self.assertEqual(list_review_events[1].req_reviewer, None)

        self.assertEqual(
            list(map(lambda x: x.created_at, list_review_events)),
            [
                datetime.strptime("2008-01-14T04:45:35+0000", "%Y-%m-%dT%H:%M:%S%z"),
                datetime.strptime("2008-01-14T04:45:35+0000", "%Y-%m-%dT%H:%M:%S%z"),
            ],
        )

    @patch("github.Repository.Repository.get_issues")
    @patch("github.Issue.Issue.get_timeline")
    @patch("github.Github.get_repo")
    def test_load_issue_timeline_events(
        self, mock_repo, mock_issue_timeline_events, mock_issues
    ):
        mock_repo.return_value = mocks.repo
        mock_issues.return_value = mocks.MockedGithubList(
            [mocks.issue_1, mocks.issue_2]
        )
        mock_issue_timeline_events.side_effect = [
            mocks.MockedGithubList([mocks.issue_cross_referenced_event]),
            mocks.MockedGithubList([]),
        ]

        # turing_attributes = {"login": "alanturing", "email": None, "id": 1912, "type": "User"}
        timeline_events = self.uut.load_github_issue_event_timeline()
        timeline_events_list = list(timeline_events)
        self.assertIsInstance(timeline_events, Generator)
        self.assertEqual(len(timeline_events_list), 1)
        self.assertEqual(
            timeline_events_list[0].actor,
           alan_turing,
        )
        self.assertEqual(timeline_events_list[0].issue, IssueInfo(number=mocks.issue_1.number))
        self.assertEqual(
            timeline_events_list[0].source_issue, IssueInfo(number=mocks.issue_2.number)
        )
        self.assertEqual(
            timeline_events_list[0].created_at, pd.to_datetime("2012-01-14T04:45:35Z")
        )
        self.assertEqual(timeline_events_list[0].event_type, "cross-referenced")
