import unittest
from datetime import datetime, timezone

import pandas as pd

import git2kg.datapoint.utils as uut
from git2kg.datapoint import ActorInfo, CommitDatapoint, IssueDatapoint, IssueInfo
from git2kg.datapoint.base import CommitInfo


class TestDatapointUtils(unittest.TestCase):
    def test_convert_dataframe_to_commit_datapoints(self):
        df = pd.read_csv("test/resources/gitdata-test-commits.csv")
        dps = uut.dataframe_to_datapoint_list(df, CommitDatapoint)

        self.assertEqual(len(dps), 6)

        first = dps[0]
        expected_first = CommitDatapoint(
            commit=CommitInfo(hash="72d23622b3049baa9d795930d3993e2d849ed758", id=None),
            author=ActorInfo(
                name="Marie Curie",
                email="marie.curie@radioactive.pl",
                type="git",
                github_id=None,
                id=None,
            ),
            committer=ActorInfo(
                name="Marie Curie",
                email="marie.curie@radioactive.pl",
                type="git",
                github_id=None,
                id=None,
            ),
            author_date=datetime(2024, 11, 29, 10, 15, 17, tzinfo=timezone.utc),
            author_timezone=3600,
            committer_date=datetime(2024, 11, 29, 10, 15, 17, tzinfo=timezone.utc),
            committer_timezone=3600,
            n_modified_files=1,
            message="added radiation",
            is_merge=False,
            is_in_main_branch=True,
            is_bugfix=False,
            n_modified_documentation=0,
            insertions=3,
            deletions=0,
            message_polarity=0.1,
            message_subjectivity=0.0,
            commit_parents=[],
            branches={"master"},
            sig_type=None,
            sig=None,
        )
        self.assertEqual(first, expected_first)

        # Assert Parent commit list
        self.assertListEqual(
            dps[5].commit_parents,
            [
                CommitInfo("df9535c8355d46fa7bf18644fb3ef1e6be0931d2"),
                CommitInfo("cf503da3c88726090c418c93eb2bda78268ea849"),
            ],
        )

    def test_convert_dataframe_to_issue_datapoints(self):
        self.maxDiff = None
        df = pd.read_csv("test/resources/gitdata-test-issues.csv")
        dps = uut.dataframe_to_datapoint_list(df, IssueDatapoint)

        self.assertEqual(len(dps), 2)

        expected = [
            IssueDatapoint(
                issue=IssueInfo(number=1),
                id=24,
                body="I really dislike this feature, just because I despise it",
                mentioned_actors=[],
                sentiment_polarity=-0.66,
                sentiment_subjectivity=0.9,
                created_at=datetime(2008, 1, 14, 4, 33, 35, tzinfo=timezone.utc),
                updated_at=datetime(2008, 1, 14, 4, 33, 35, tzinfo=timezone.utc),
                author=ActorInfo(
                    name="mariecurie",
                    email="marie.curie+github@radioactive.pl",
                    type="User",
                    github_id=1867,
                    id=None,
                ),
                assignees=[
                    ActorInfo(
                        name="alanturing",
                        email=None,
                        type="User",
                        github_id=1912,
                        id=None,
                    )
                ],
                closed_at=None,
                is_pull=False,
                is_merged=False,
                merged_at=None,
                merged_by=None,
                state="open",
            ),
            IssueDatapoint(
                issue=IssueInfo(number=2),
                id=42,
                body="We might need to alter the codebase for slightly better readability",
                mentioned_actors=[],
                sentiment_polarity=0.1,
                sentiment_subjectivity=0.1,
                created_at=datetime(2009, 2, 15, 5, 34, 36, tzinfo=timezone.utc),
                updated_at=datetime(2009, 3, 15, 5, 34, 36, tzinfo=timezone.utc),
                author=ActorInfo(
                    name="mariecurie",
                    email="marie.curie+github@radioactive.pl",
                    type="User",
                    github_id=1867,
                    id=None,
                ),
                assignees=[
                    ActorInfo(
                        name="Konny Zuse",
                        email="zuse@zusekg.de",
                        type="User",
                        github_id=1910,
                        id=None,
                    ),
                    ActorInfo(
                        name="alanturing",
                        email=None,
                        type="User",
                        github_id=1912,
                        id=None,
                    ),
                ],
                closed_at=datetime(2009, 4, 15, 10, 34, 36, tzinfo=timezone.utc),
                is_pull=True,
                is_merged=True,
                merged_at=datetime(2009, 2, 16, 10, 34, 36, tzinfo=timezone.utc),
                merged_by=ActorInfo(
                    name="mariecurie",
                    email="marie.curie+github@radioactive.pl",
                    type="User",
                    github_id=1867,
                    id=None,
                ),
                state="closed",
            ),
        ]
        self.assertListEqual(list(dps), expected)
