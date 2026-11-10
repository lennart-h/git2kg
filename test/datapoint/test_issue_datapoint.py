import unittest
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

from git2kg.datapoint import (
    ActorInfo,
    IssueDatapoint,
    IssueInfo,
)


class TestIssueDatapoint(unittest.TestCase):
    def test_issue_datapoint_dtypes(self):
        self.maxDiff = None
        dtypes = IssueDatapoint.dtypes(flatten=False)
        expected = {
            "issue": IssueInfo,
            "id": int,
            "body": str,
            "mentioned_actors": List[ActorInfo],
            "sentiment_polarity": float,
            "sentiment_subjectivity": float,
            "created_at": datetime,
            "updated_at": datetime,
            "author": ActorInfo,
            "assignees": List[ActorInfo],
            "closed_at": datetime,
            "is_pull": bool,
            "is_merged": bool,
            "merged_at": datetime,
            "merged_by": ActorInfo,
            "state": str,
        }
        self.assertDictEqual(dtypes, expected)

    def test_issue_flat_datapoint_dtypes(self):
        self.maxDiff = None

        dtypes = IssueDatapoint.dtypes(flatten=True)
        expected = {
            "issue_number": int,
            "id": int,
            "body": str,
            "sentiment_polarity": float,
            "sentiment_subjectivity": float,
            "created_at": datetime,
            "updated_at": datetime,
            "author_name": str,
            "author_email": str,
            "author_github_id": int,
            "author_id": int,
            "author_type": str,
            "mentioned_actors_name": List[str],
            "mentioned_actors_email": List[str],
            "mentioned_actors_github_id": List[int],
            "mentioned_actors_id": List[int],
            "mentioned_actors_type": List[str],
            "assignees_name": List[str],
            "assignees_email": List[str],
            "assignees_github_id": List[int],
            "assignees_id": List[int],
            "assignees_type": List[str],
            "closed_at": datetime,
            "is_pull": bool,
            "is_merged": bool,
            "merged_at": datetime,
            "merged_by_name": str,
            "merged_by_email": str,
            "merged_by_github_id": int,
            "merged_by_id": int,
            "merged_by_type": str,
            "state": str,
        }
        self.assertDictEqual(dtypes, expected)

    def test_issue_datapoint_to_dict(self):
        self.maxDiff = None
        tz = ZoneInfo("America/Los_Angeles")
        issue_dp = IssueDatapoint(
            issue=IssueInfo(number=12),
            id=1234,
            body="probably something bad @Teller @David",
            mentioned_actors=[ActorInfo(name="Teller"), ActorInfo(name="David")],
            sentiment_polarity=-0.3,
            sentiment_subjectivity=0.0,
            created_at=datetime(2026, 8, 19, 12, 0, 0, 0, tzinfo=tz),
            updated_at=datetime(2026, 8, 20, 12, 0, 0, 0, tzinfo=tz),
            author=ActorInfo(
                name="Penn", email="penn@magic.com", github_id=12, id=0, type="User"
            ),
            assignees=[
                ActorInfo(
                    name="Teller",
                    email="teller@magic.com",
                    github_id=11,
                    id=1,
                    type="User",
                ),
                ActorInfo(
                    name="David Copperfield",
                    email="david@copperfield.com",
                    github_id=0,
                    id=2,
                    type="User",
                ),
            ],
            closed_at=datetime(2026, 8, 20, 12, 0, 0, 0, tzinfo=tz),
            is_pull=False,
            is_merged=False,
            merged_by=None,
            merged_at=None,
            state="closed",
        )
        dp_dict = issue_dp.to_dict()
        expected_dict = {
            "issue_number": 12,
            "id": 1234,
            "body": "probably something bad @Teller @David",
            "sentiment_polarity": -0.3,
            "sentiment_subjectivity": 0.0,
            "created_at": datetime(2026, 8, 19, 12, 0, 0, 0, tzinfo=tz),
            "updated_at": datetime(2026, 8, 20, 12, 0, 0, 0, tzinfo=tz),
            "author_name": "Penn",
            "author_email": "penn@magic.com",
            "author_github_id": 12,
            "author_id": 0,
            "author_type": "User",
            "mentioned_actors_name": ["Teller", "David"],
            "mentioned_actors_email": [None, None],
            "mentioned_actors_github_id": [None, None],
            "mentioned_actors_id": [None, None],
            "mentioned_actors_type": [None, None],
            "assignees_name": ["Teller", "David Copperfield"],
            "assignees_email": ["teller@magic.com", "david@copperfield.com"],
            "assignees_github_id": [11, 0],
            "assignees_id": [1, 2],
            "assignees_type": ["User", "User"],
            "closed_at": datetime(2026, 8, 20, 12, 0, 0, 0, tzinfo=tz),
            "is_pull": False,
            "is_merged": False,
            "merged_at": None,
            "merged_by_name": None,
            "merged_by_email": None,
            "merged_by_github_id": None,
            "merged_by_id": None,
            "merged_by_type": None,
            "state": "closed",
        }
        self.assertEqual(dp_dict, expected_dict)
