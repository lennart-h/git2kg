import unittest
from datetime import UTC, datetime

import pandas as pd

from git2kg.datapoint import (
    CommitDatapoint,
    DatapointList,
)
from git2kg.datapoint.base import (
    ActorInfo,
    CommitInfo,
    FileInfo,
    IssueInfo,
)
from git2kg.edge_extraction import EdgeExtractor
from git2kg.gitdata import FIRST_SEEN, _load_gitdata


def _commit_datapoint(
    commit: CommitInfo,
    author: ActorInfo,
    committer: ActorInfo,
    author_date: datetime,
    committer_date: datetime,
) -> CommitDatapoint:
    """Builds a CommitDatapoint, defaulting everything the caller does not care about"""
    return CommitDatapoint(
        commit=commit,
        author=author,
        committer=committer,
        author_date=author_date,
        author_timezone=0,
        committer_date=committer_date,
        committer_timezone=0,
        n_modified_files=0,
        message="",
        is_merge=False,
        is_in_main_branch=True,
        is_bugfix=False,
        n_modified_documentation=0,
        insertions=0,
        deletions=0,
        message_polarity=0.0,
        message_subjectivity=0.0,
    )


class TestGitData(unittest.TestCase):
    def setUp(self):
        actor_mapping = pd.DataFrame(
            {
                ActorInfo.ID_COLUMN: list(range(4)),
                "alias_name": [
                    ["Marie Curie", "mariecurie", "mariecurie"],
                    [
                        "Rosalind Franklin",
                        "Rosa Franklin",
                        "rosafranklin1920",
                        "rosafranklin1920",
                    ],
                    ["Alan Turing", "Alan Turing", "alanturing", "alanturing"],
                    ["Konrad Zuse", "Konny Zuse", "Konny Zuse"],
                ],
                "alias_email": [
                    [
                        "marie.curie@radioactive.pl",
                        "marie.curie+github@radioactive.pl",
                        None,
                    ],
                    [
                        "franklin@kcl.ac.uk",
                        "rosa@frankling.uk",
                        "franklin@kcl.ac.uk",
                        None,
                    ],
                    [
                        "turingalan1912@computable.uk",
                        "alanturing@lichess.org",
                        None,
                        None,
                    ],
                    ["kzuse@z3.de", "zuse@zusekg.de", None],
                ],
                ActorInfo.GITHUB_ID_COLUMN: [
                    [None, 1867, None],
                    [None, None, 1920, None],
                    [None, None, 1912, None],
                    [None, 1910, None],
                ],
            }
        )
        self.actor_mapping = actor_mapping.explode(
            ["alias_name", "alias_email", ActorInfo.GITHUB_ID_COLUMN]
        ).drop_duplicates()

    def test_get_authors(self):
        print(__name__)
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        actors = gitdata.get_actors()

        self.assertIsInstance(actors, set)
        # get_actors is idempotent across disambiguation: the timeline event
        # actors are the same ActorInfo objects add_actor_ids stamps, so they
        # dedupe against their already-stamped counterparts rather than
        # entering the set a second time with id=None.
        self.assertEqual(len(actors), 13)
        self.assertEqual(len(gitdata.get_actors()), len(actors))

    def test_actor_appearances(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)

        appearances = gitdata.actor_appearances()

        # Every actor of the fixture shows up in the issue world years before
        # their first commit, so the dates are the issue ones, not the 2024
        # commit dates
        self.assertListEqual([0, 1, 2, 3], list(appearances[ActorInfo.ID_COLUMN]))
        self.assertListEqual(
            [
                pd.Timestamp("2008-01-14T04:33:35", tz="UTC"),
                pd.Timestamp("2009-03-15T08:59:36", tz="UTC"),
                pd.Timestamp("2008-01-14T04:33:35", tz="UTC"),
                pd.Timestamp("2009-02-15T05:34:36", tz="UTC"),
            ],
            list(appearances[FIRST_SEEN]),
        )

    def test_actor_appearances_leaves_out_actors_without_an_id(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)

        appearances = gitdata.actor_appearances()

        # The fixture holds 13 distinct actors but the mapping only covers 4.
        # The rest are not part of any id universe, so they are no nodes
        self.assertEqual(13, len(gitdata.get_actors()))
        self.assertEqual(4, len(appearances))

    def test_history_end(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")

        # The last thing to happen in the fixture is the merge commit
        self.assertEqual(
            pd.Timestamp("2024-11-29T10:27:14", tz="UTC"), gitdata.history_end()
        )

    def test_create_communication_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")

        gitdata.add_actor_ids(self.actor_mapping)

        communication_net = gitdata.issue_comment_edges()

        self.assertEqual(len(communication_net), 4)

        self.assertEqual(
            communication_net[0][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            communication_net[0][1],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(
            communication_net[0][2], datetime(2008, 1, 14, 4, 45, 35, tzinfo=UTC)
        )

        self.assertEqual(
            communication_net[1][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            communication_net[1][1],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            communication_net[1][2], datetime(2008, 1, 14, 5, 33, 35, tzinfo=UTC)
        )

        self.assertEqual(
            communication_net[2][0],
            ActorInfo(
                name="Konny Zuse",
                email="zuse@zusekg.de",
                github_id=1910,
                id=3,
                type="User",
            ),
        )
        self.assertEqual(
            communication_net[2][1],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(
            communication_net[2][2], datetime(2009, 3, 15, 8, 59, 36, tzinfo=UTC)
        )
        self.assertEqual(
            communication_net[3][0],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(
            communication_net[3][1],
            ActorInfo(
                name="Konny Zuse",
                email="zuse@zusekg.de",
                github_id=1910,
                id=3,
                type="User",
            ),
        )
        self.assertEqual(
            communication_net[3][2], datetime(2009, 4, 15, 10, 30, 36, tzinfo=UTC)
        )

    def test_create_delete_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        delete_net = gitdata.edit_delete_edges()

        self.assertEqual(len(delete_net), 6)

        self.assertEqual(
            delete_net[0][0],
            ActorInfo(
                name="Alan Turing",
                email="turingalan1912@computable.uk",
                id=2,
                type="git",
            ),
        )
        self.assertEqual(
            delete_net[0][1],
            ActorInfo(
                name="Rosalind Franklin", email="franklin@kcl.ac.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            delete_net[0][2], datetime(2024, 11, 29, 10, 20, 46, tzinfo=UTC)
        )

        self.assertEqual(
            delete_net[1][0],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            delete_net[1][1],
            ActorInfo(
                name="Alan Turing",
                email="turingalan1912@computable.uk",
                id=2,
                type="git",
            ),
        )
        self.assertEqual(
            delete_net[1][2], datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC)
        )

        self.assertEqual(
            delete_net[2][0],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            delete_net[2][1],
            ActorInfo(
                name="Rosalind Franklin", email="franklin@kcl.ac.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            delete_net[2][2], datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC)
        )

        self.assertEqual(
            delete_net[3][0],
            ActorInfo(
                name="Alan Turing", email="alanturing@lichess.org", id=2, type="git"
            ),
        )
        self.assertEqual(
            delete_net[3][1],
            ActorInfo(
                name="Marie Curie", email="marie.curie@radioactive.pl", id=0, type="git"
            ),
        )
        self.assertEqual(
            delete_net[3][2], datetime(2024, 11, 29, 10, 25, 25, tzinfo=UTC)
        )

        self.assertEqual(
            delete_net[4][0],
            ActorInfo(
                name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            delete_net[4][1],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            delete_net[4][2], datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC)
        )

        self.assertEqual(
            delete_net[5][0],
            ActorInfo(
                name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            delete_net[5][1],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            delete_net[5][2], datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC)
        )

    def test_create_add_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)

        add_net = gitdata.edit_add_edges()

        self.assertEqual(len(add_net), 5)

        self.assertEqual(
            add_net[0][0],
            ActorInfo(
                name="Alan Turing",
                email="turingalan1912@computable.uk",
                id=2,
                type="git",
            ),
        )
        self.assertEqual(
            add_net[0][1],
            ActorInfo(
                name="Rosalind Franklin", email="franklin@kcl.ac.uk", id=1, type="git"
            ),
        )
        self.assertEqual(add_net[0][2], datetime(2024, 11, 29, 10, 20, 46, tzinfo=UTC))

        self.assertEqual(
            add_net[1][0],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            add_net[1][1],
            ActorInfo(
                name="Alan Turing",
                email="turingalan1912@computable.uk",
                id=2,
                type="git",
            ),
        )
        self.assertEqual(add_net[1][2], datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC))

        self.assertEqual(
            add_net[2][0],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(
            add_net[2][1],
            ActorInfo(
                name="Rosalind Franklin", email="franklin@kcl.ac.uk", id=1, type="git"
            ),
        )
        self.assertEqual(add_net[2][2], datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC))

        self.assertEqual(
            add_net[3][0],
            ActorInfo(
                name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            add_net[3][1],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(add_net[3][2], datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC))

        self.assertEqual(
            add_net[4][0],
            ActorInfo(
                name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
            ),
        )
        self.assertEqual(
            add_net[4][1],
            ActorInfo(name="Konrad Zuse", email="kzuse@z3.de", id=3, type="git"),
        )
        self.assertEqual(add_net[4][2], datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC))

    def test_review_request_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        review_request_edges = gitdata.review_request_edges()

        self.assertEqual(len(review_request_edges), 2)

        self.assertEqual(
            review_request_edges[0][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            review_request_edges[0][1],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(
            review_request_edges[0][2], datetime(2008, 1, 14, 4, 45, 35, tzinfo=UTC)
        )

        self.assertEqual(
            review_request_edges[1][0],
            ActorInfo(
                name="Konny Zuse",
                email="zuse@zusekg.de",
                github_id=1910,
                id=3,
                type="User",
            ),
        )
        self.assertEqual(
            review_request_edges[1][1],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(
            review_request_edges[1][2], datetime(2009, 4, 15, 9, 39, 30, tzinfo=UTC)
        )

    def test_review_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        review_edges = gitdata.review_edges()

        self.assertEqual(len(review_edges), 1)
        self.assertEqual(
            review_edges[0][0],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(
            review_edges[0][1],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(
            review_edges[0][2], datetime(2009, 4, 15, 10, 39, 30, tzinfo=UTC)
        )

    def test_create_issue_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        create_issue_edges = gitdata.create_issue_edges()

        self.assertEqual(len(create_issue_edges), 2)

        self.assertEqual(
            create_issue_edges[0][0],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(create_issue_edges[0][1], IssueInfo(number=1))
        self.assertEqual(
            create_issue_edges[0][2], datetime(2008, 1, 14, 4, 33, 35, tzinfo=UTC)
        )

        self.assertEqual(
            create_issue_edges[1][0],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(create_issue_edges[1][1], IssueInfo(number=2))
        self.assertEqual(
            create_issue_edges[1][2], datetime(2009, 2, 15, 5, 34, 36, tzinfo=UTC)
        )

    def test_review_pull_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        review_pull_edges = gitdata.review_pull_edges()

        self.assertEqual(len(review_pull_edges), 1)
        self.assertEqual(
            review_pull_edges[0][0],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(review_pull_edges[0][1], IssueInfo(number=2))
        self.assertEqual(
            review_pull_edges[0][2], datetime(2009, 4, 15, 10, 39, 30, tzinfo=UTC)
        )

    def test_comment_issue_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        comment_issue_edges = gitdata.comment_issue_edges()

        self.assertEqual(len(comment_issue_edges), 4)

        self.assertEqual(
            comment_issue_edges[0][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(comment_issue_edges[0][1], IssueInfo(number=1))
        self.assertEqual(
            comment_issue_edges[0][2], datetime(2008, 1, 14, 4, 45, 35, tzinfo=UTC)
        )

        self.assertEqual(
            comment_issue_edges[1][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(comment_issue_edges[1][1], IssueInfo(number=1))
        self.assertEqual(
            comment_issue_edges[1][2], datetime(2008, 1, 14, 5, 33, 35, tzinfo=UTC)
        )

        self.assertEqual(
            comment_issue_edges[2][0],
            ActorInfo(
                name="Konny Zuse",
                email="zuse@zusekg.de",
                github_id=1910,
                id=3,
                type="User",
            ),
        )
        self.assertEqual(comment_issue_edges[2][1], IssueInfo(number=2))
        self.assertEqual(
            comment_issue_edges[2][2], datetime(2009, 3, 15, 8, 59, 36, tzinfo=UTC)
        )

        self.assertEqual(
            comment_issue_edges[3][0],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(comment_issue_edges[3][1], IssueInfo(number=2))
        self.assertEqual(
            comment_issue_edges[3][2], datetime(2009, 4, 15, 10, 30, 36, tzinfo=UTC)
        )

    def test_issue_referenced_on_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        issue_referenced_on_edges = gitdata.issue_referenced_on_edges()
        us = []
        vs = []
        for edge in issue_referenced_on_edges:
            us.append(edge[0])
            vs.append(edge[1])

        self.assertEqual(len(issue_referenced_on_edges), 2)

        self.assertEqual(issue_referenced_on_edges[0][0], IssueInfo(number=1))
        self.assertEqual(issue_referenced_on_edges[0][1], IssueInfo(number=2))
        self.assertEqual(
            issue_referenced_on_edges[0][2],
            datetime(2012, 1, 14, 4, 45, 35, tzinfo=UTC),
        )

        self.assertEqual(issue_referenced_on_edges[1][0], IssueInfo(number=2))
        self.assertEqual(issue_referenced_on_edges[1][1], IssueInfo(number=1))
        self.assertEqual(
            issue_referenced_on_edges[1][2],
            datetime(2012, 3, 12, 7, 37, 21, tzinfo=UTC),
        )

    def test_issue_assigned_from_user_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        assign_edges = gitdata.issue_assigned_from_user_edges()

        self.assertEqual(len(assign_edges), 1)

        self.assertEqual(
            assign_edges[0][0],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(
            assign_edges[0][1],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            assign_edges[0][2], datetime(2008, 1, 14, 4, 35, 35, tzinfo=UTC)
        )

    def test_issue_assigned_to_user_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        assigned_to_edges = gitdata.issue_assigned_to_user_edges()

        self.assertEqual(len(assigned_to_edges), 1)

        self.assertEqual(assigned_to_edges[0][0], IssueInfo(number=1))
        self.assertEqual(
            assigned_to_edges[0][1],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(
            assigned_to_edges[0][2], datetime(2008, 1, 14, 4, 35, 35, tzinfo=UTC)
        )

    def test_issue_merged_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        merged_edges = gitdata.issue_merged_edges()

        self.assertEqual(len(merged_edges), 2)

        self.assertEqual(
            merged_edges[0][0],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(merged_edges[0][1], IssueInfo(number=3))
        self.assertEqual(
            merged_edges[0][2], datetime(2008, 1, 14, 4, 50, 35, tzinfo=UTC)
        )

        self.assertEqual(
            merged_edges[1][0],
            ActorInfo(
                name="mariecurie",
                email="marie.curie+github@radioactive.pl",
                github_id=1867,
                id=0,
                type="User",
            ),
        )
        self.assertEqual(merged_edges[1][1], IssueInfo(number=1))
        self.assertEqual(
            merged_edges[1][2], datetime(2008, 1, 14, 4, 51, 35, tzinfo=UTC)
        )

    def test_issue_closed_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        closed_edges = gitdata.issue_closed_edges()

        self.assertEqual(len(closed_edges), 2)

        self.assertEqual(
            closed_edges[0][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        
        self.assertEqual(closed_edges[0][1], IssueInfo(number=1))
        self.assertEqual(
            closed_edges[0][2], datetime(2008, 1, 14, 4, 50, 35, tzinfo=UTC)
        )
        self.assertEqual(
            closed_edges[1][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        
        self.assertEqual(closed_edges[1][1], IssueInfo(number=2))
        self.assertEqual(
            closed_edges[1][2], datetime(2009,4,15,10,34,36, tzinfo=UTC)
        )

    def test_issue_reopened_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        reopened_edges = gitdata.issue_reopened_edges()
        self.assertEqual(len(reopened_edges), 1)
        self.assertEqual(reopened_edges[0][0],   ActorInfo(
                        name="mariecurie",
                        email="marie.curie+github@radioactive.pl",
                        github_id=1867,
                        id=0,
                        type="User",
                    ))
        self.assertEqual(reopened_edges[0][1],   IssueInfo(number=2))
        self.assertEqual(reopened_edges[0][2], datetime(2009,4,16,10,34,36, tzinfo=UTC))
        



    def test_mentions_actor(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        mentions_actor_edges = gitdata.mentions_actor_edges()

        self.assertEqual(len(mentions_actor_edges), 3)
        self.assertEqual(
            mentions_actor_edges[0][0],
            ActorInfo(name="alanturing", github_id=1912, id=2, type="User"),
        )
        self.assertEqual(mentions_actor_edges[0][1], ActorInfo(name="mariecurie", id=0))
        self.assertEqual(
            mentions_actor_edges[0][2], datetime(2008, 1, 14, 5, 33, 35, tzinfo=UTC)
        )

        self.assertEqual(
            mentions_actor_edges[1][0],
            ActorInfo(
                name="Konny Zuse",
                email="zuse@zusekg.de",
                github_id=1910,
                id=3,
                type="User",
            ),
        )
        self.assertEqual(
            mentions_actor_edges[1][1], ActorInfo(name="rosafranklin1920", id=1)
        )
        self.assertEqual(
            mentions_actor_edges[1][2], datetime(2009, 3, 15, 8, 59, 36, tzinfo=UTC)
        )

        self.assertEqual(
            mentions_actor_edges[2][0],
            ActorInfo(
                name="rosafranklin1920",
                email="franklin@kcl.ac.uk",
                github_id=1920,
                id=1,
                type="User",
            ),
        )
        self.assertEqual(mentions_actor_edges[2][1], ActorInfo(name="alanturing", id=2))
        self.assertEqual(
            mentions_actor_edges[2][2], datetime(2009, 4, 15, 10, 30, 36, tzinfo=UTC)
        )

    def test_commit_parent_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_commit_ids()
        commit_parent_edges = gitdata.commit_parent_edges()
        self.assertEqual(len(commit_parent_edges), 6)

        self.assertEqual(
            commit_parent_edges[0],
            (
                CommitInfo("990ccb25e4d5dd55e0c2e5ff92c0966c3b051e9e", id=1),
                CommitInfo("72d23622b3049baa9d795930d3993e2d849ed758", id=0),
                datetime(2024, 11, 29, 10, 18, 29, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            commit_parent_edges[1],
            (
                CommitInfo("3cc5a9b28956b0cfac475a565688604bd7bb6c7f", id=2),
                CommitInfo("990ccb25e4d5dd55e0c2e5ff92c0966c3b051e9e", id=1),
                datetime(2024, 11, 29, 10, 20, 46, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            commit_parent_edges[2],
            (
                CommitInfo("cf503da3c88726090c418c93eb2bda78268ea849", id=3),
                CommitInfo("3cc5a9b28956b0cfac475a565688604bd7bb6c7f", id=2),
                datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            commit_parent_edges[3],
            (
                CommitInfo("df9535c8355d46fa7bf18644fb3ef1e6be0931d2", id=4),
                CommitInfo("cf503da3c88726090c418c93eb2bda78268ea849", id=3),
                datetime(2024, 11, 29, 10, 25, 25, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            commit_parent_edges[4],
            (
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                CommitInfo("df9535c8355d46fa7bf18644fb3ef1e6be0931d2", id=4),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            commit_parent_edges[5],
            (
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                CommitInfo("cf503da3c88726090c418c93eb2bda78268ea849", id=3),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )

    def test_author_commit_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        gitdata.add_commit_ids()
        author_commit_edges = gitdata.author_commit_edges()
        self.assertEqual(len(author_commit_edges), 6)

        self.assertEqual(
            author_commit_edges[0],
            (
                ActorInfo(
                    name="Marie Curie",
                    email="marie.curie@radioactive.pl",
                    id=0,
                    type="git",
                ),
                CommitInfo("72d23622b3049baa9d795930d3993e2d849ed758", id=0),
                datetime(2024, 11, 29, 10, 15, 17, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            author_commit_edges[4],
            (
                ActorInfo(
                    name="Alan Turing", email="alanturing@lichess.org", id=2, type="git"
                ),
                CommitInfo("df9535c8355d46fa7bf18644fb3ef1e6be0931d2", id=4),
                datetime(2024, 11, 29, 10, 25, 25, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            author_commit_edges[5],
            (
                ActorInfo(
                    name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
                ),
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )

    def test_committer_commit_edges(self):
        # Every commit in the fixture was authored and committed by the same
        # actor at the same time, so this only covers the wiring. See
        # test_author_and_committer_edges_use_their_own_actor_and_date for the
        # case the two relations exist to tell apart.
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_actor_ids(self.actor_mapping)
        gitdata.add_commit_ids()
        committer_commit_edges = gitdata.committer_edges()
        self.assertEqual(len(committer_commit_edges), 6)

        self.assertEqual(
            committer_commit_edges[0],
            (
                ActorInfo(
                    name="Marie Curie",
                    email="marie.curie@radioactive.pl",
                    id=0,
                    type="git",
                ),
                CommitInfo("72d23622b3049baa9d795930d3993e2d849ed758", id=0),
                datetime(2024, 11, 29, 10, 15, 17, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            committer_commit_edges[5],
            (
                ActorInfo(
                    name="Rosa Franklin", email="rosa@frankling.uk", id=1, type="git"
                ),
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )

    def test_author_and_committer_edges_share_the_commit_date(self):
        # Author and committer differ, and so do their dates. Both relations
        # are timestamped at the committer date.
        alice = ActorInfo(name="Alice", email="alice@example.org", id=0, type="git")
        bob = ActorInfo(name="Bob", email="bob@example.org", id=1, type="git")
        carol = ActorInfo(name="Carol", email="carol@example.org", id=2, type="git")
        commit_a = CommitInfo("a" * 40, id=0)
        commit_b = CommitInfo("b" * 40, id=1)

        commits = DatapointList(
            [
                _commit_datapoint(
                    commit_a,
                    author=alice,
                    committer=bob,
                    author_date=datetime(2024, 1, 1, tzinfo=UTC),
                    committer_date=datetime(2024, 3, 1, tzinfo=UTC),
                ),
                _commit_datapoint(
                    commit_b,
                    author=carol,
                    committer=bob,
                    author_date=datetime(2024, 2, 1, tzinfo=UTC),
                    committer_date=datetime(2024, 2, 15, tzinfo=UTC),
                ),
            ],
            CommitDatapoint,
        )
        extractor = EdgeExtractor()

        self.assertEqual(
            [
                (carol, commit_b, datetime(2024, 2, 15, tzinfo=UTC)),
                (alice, commit_a, datetime(2024, 3, 1, tzinfo=UTC)),
            ],
            extractor.extract_author_commit_edges(commits),
        )
        self.assertEqual(
            [
                (bob, commit_b, datetime(2024, 2, 15, tzinfo=UTC)),
                (bob, commit_a, datetime(2024, 3, 1, tzinfo=UTC)),
            ],
            extractor.extract_comitter_commit_edges(commits),
        )

    def test_co_commit_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_file_ids()
        co_commit_edges = gitdata.file_co_commit_edges()
        self.assertEqual(len(co_commit_edges), 2)
        expected_edges = {
            (
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=1, path="dns.txt"
                ),
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=2, path="machine.py"
                ),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
            (
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=2, path="machine.py"
                ),
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=1, path="dns.txt"
                ),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        }
        self.assertSetEqual(set(co_commit_edges), expected_edges)

    def test_commit_modifies_file_edges(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        gitdata.add_commit_ids()
        gitdata.add_file_ids()
        modifies_file_edges = gitdata.commit_modifies_file_edges()  #
        self.assertEqual(len(modifies_file_edges), 7)  #
        self.assertEqual(
            modifies_file_edges[0],
            (
                CommitInfo("72d23622b3049baa9d795930d3993e2d849ed758", id=0),
                FileInfo(
                    "72d23622b3049baa9d795930d3993e2d849ed758",
                    id=0,
                    path="radiation.txt",
                ),
                datetime(2024, 11, 29, 10, 15, 17, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[1],
            (
                CommitInfo("990ccb25e4d5dd55e0c2e5ff92c0966c3b051e9e", id=1),
                FileInfo(
                    "990ccb25e4d5dd55e0c2e5ff92c0966c3b051e9e", id=1, path="dns.txt"
                ),
                datetime(2024, 11, 29, 10, 18, 29, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[2],
            (
                CommitInfo("3cc5a9b28956b0cfac475a565688604bd7bb6c7f", id=2),
                FileInfo(
                    "3cc5a9b28956b0cfac475a565688604bd7bb6c7f", id=1, path="dns.txt"
                ),
                datetime(2024, 11, 29, 10, 20, 46, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[3],
            (
                CommitInfo("cf503da3c88726090c418c93eb2bda78268ea849", id=3),
                FileInfo(
                    "cf503da3c88726090c418c93eb2bda78268ea849", id=1, path="dns.txt"
                ),
                datetime(2024, 11, 29, 10, 23, 10, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[4],
            (
                CommitInfo("df9535c8355d46fa7bf18644fb3ef1e6be0931d2", id=4),
                FileInfo(
                    "df9535c8355d46fa7bf18644fb3ef1e6be0931d2",
                    id=0,
                    path="radiation.txt",
                ),
                datetime(2024, 11, 29, 10, 25, 25, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[5],
            (
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=1, path="dns.txt"
                ),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )
        self.assertEqual(
            modifies_file_edges[6],
            (
                CommitInfo("5709501c212d5c607b06636d9594e1569613c47b", id=5),
                FileInfo(
                    "5709501c212d5c607b06636d9594e1569613c47b", id=2, path="machine.py"
                ),
                datetime(2024, 11, 29, 10, 27, 14, tzinfo=UTC),
            ),
        )
