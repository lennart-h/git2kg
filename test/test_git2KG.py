import datetime
import json
import os
import tarfile
import unittest
from tempfile import TemporaryDirectory, mkdtemp
from unittest import mock

import numpy as np
import pandas as pd

from git2kg.continuous_network import ContinuousNetwork
from git2kg.core import Git2KG
from git2kg.dataloading import GRAPHQL, REST
from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    DatapointList,
    EditDatapoint,
    IssueCommentDatapoint,
    IssueDatapoint,
    IssueEventDatapoint,
    IssueInfo,
    ReviewDatapoint,
)
from git2kg.disambiguation import (
    DISAMBIGUATION_VERSION,
    disambiguated_author_mapping,
)
from git2kg.features import ActorFeatureExtractor
from git2kg.gitdata import GitData, _load_gitdata
from git2kg.labels import (
    DeveloperLeaverLabelExtractor,
    FileModifiedNextSliceLabelExtractor,
    IssueClosedNextSliceLabelExtractor,
)
from git2kg.network_snapshots import NetworkSnapshots


def _issue_datapoint(
    number: int, created_at: datetime.datetime, updated_at: datetime.datetime
) -> IssueDatapoint:
    """Builds an IssueDatapoint, defaulting everything the caller does not care about"""
    return IssueDatapoint(
        issue=IssueInfo(number=number),
        id=number,
        body="",
        mentioned_actors=[],
        sentiment_polarity=0.0,
        sentiment_subjectivity=0.0,
        created_at=created_at,
        updated_at=updated_at,
        author=ActorInfo(
            name="Marie Curie",
            email="marie.curie@radioactive.pl",
            type="User",
            github_id=1867,
        ),
        assignees=[],
        closed_at=None,
        is_pull=False,
        is_merged=False,
        merged_at=None,
        merged_by=None,
        state="open",
    )


def _gitdata_with_issues(*issues: IssueDatapoint) -> GitData:
    """Builds a GitData holding only the given issues"""
    return GitData(
        DatapointList([], CommitDatapoint),
        DatapointList([], EditDatapoint),
        DatapointList(list(issues), IssueDatapoint),
        DatapointList([], IssueCommentDatapoint),
        DatapointList([], IssueEventDatapoint),
        DatapointList([], ReviewDatapoint),
    )


def _repack(archive: str, edit) -> str:
    """Extracts an archive, lets `edit` change the extracted files, and packs
    the result into a new archive.
    """
    extracted = mkdtemp()
    with tarfile.open(archive) as tar:
        tar.extractall(extracted, filter="data")
    edit(extracted)
    repacked = os.path.join(mkdtemp(), os.path.basename(archive))
    with tarfile.open(repacked, "w:gz") as tar:
        for file in os.listdir(extracted):
            tar.add(os.path.join(extracted, file), arcname=file)
    return repacked


def _rewrite_metadata(archive: str, changes: dict) -> str:
    def edit(extracted):
        path = os.path.join(extracted, "metadata.json")
        with open(path) as metadata_file:
            metadata = json.load(metadata_file)
        metadata.update(changes)
        with open(path, "w") as metadata_file:
            json.dump(metadata, metadata_file)

    return _repack(archive, edit)


def _drop_from_archive(archive: str, filename: str) -> str:
    return _repack(
        archive, lambda extracted: os.remove(os.path.join(extracted, filename))
    )


# A repository small enough to mine twice in a test, with issues, comments,
# reviews and cross-references
SMALL_REPO_URL = "https://github.com/torvalds/pesconvert"

# What identifies a datapoint of each list, used to compare two minings of the
# same repository without depending on the order they came back in
# `id` is left out for issues: the REST API reports the id of the issue behind
# a pull request where GraphQL reports the id of the pull request itself, and
# GraphQL does not expose the former. The field is not used anywhere
_IDENTITY = {
    "issues": lambda dp: (dp.issue.number, dp.author.name, dp.is_pull),
    "issue_comments": lambda dp: (dp.issue.number, dp.id, dp.author.name),
    "events": lambda dp: (dp.issue.number, dp.event_type, dp.created_at),
    "reviews": lambda dp: (dp.pull.number, dp.id, dp.author.name),
}


def _identifiers(datapoints, dp_name: str) -> set:
    return {_IDENTITY[dp_name](dp) for dp in datapoints}


def _logins_that_do_not_exist(logins: set, token: str) -> set:
    """Asks GitHub which of the logins are no account, in one query."""
    from github import Auth, Github

    requester = Github(auth=Auth.Token(token))._Github__requester
    aliases = {f"u{i}": login for i, login in enumerate(sorted(logins))}
    query = "query { %s }" % " ".join(
        f'{alias}: user(login: "{login}") {{ login }}'
        for alias, login in aliases.items()
    )
    _, response = requester.requestJsonAndCheck(
        "POST", requester.graphql_url, input={"query": query, "variables": {}}
    )
    resolved = response.get("data") or {}
    return {login for alias, login in aliases.items() if not resolved.get(alias)}


TOKEN_PATH = "github_token.githubtoken"

#: The tests that mine a repository over the API need a token, which a clone
#: does not carry. They are skipped rather than failed where there is none
HAS_TOKEN = os.path.exists(TOKEN_PATH)
NEEDS_TOKEN = unittest.skipUnless(HAS_TOKEN, f"needs {TOKEN_PATH}")


class TestGit2KG(unittest.TestCase):
    def setUp(self):

        self.gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.uut = Git2KG(self.gitdata, freq="YE", name="scientists_repo")
        if HAS_TOKEN:
            with open(TOKEN_PATH) as f:
                # A trailing newline ends up in the Authorization header
                self.token = f.read().strip()

    def test_disambiguate_authors(self):
        disambiguated_authors = disambiguated_author_mapping(self.gitdata)
        self.assertEqual(
            len(disambiguated_authors[ActorInfo.ID_COLUMN].drop_duplicates()), 4
        )

    def test_get_timeindex(self):
        ti = self.uut.get_timeindex()
        self.assertEqual(len(ti), 18)
        self.assertEqual(np.min(ti).year, 2007)
        self.assertEqual(np.max(ti).year, 2024)

    def test_timeslices(self):
        slices = list(self.uut.timeslices)
        self.assertEqual(len(slices), 17)
        self.assertEqual(
            slices[0].first_interaction(), self.uut.gitdata.first_interaction()
        )
        self.assertNotEqual(
            slices[0].last_interaction(), self.uut.gitdata.last_interaction()
        )
        self.assertEqual(
            slices[-1].last_interaction(), self.uut.gitdata.last_interaction()
        )
        self.assertNotEqual(
            slices[-1].first_interaction(), self.uut.gitdata.first_interaction()
        )

    def test_timeslices_hold_only_their_own_interactions(self):
        # Every datapoint belongs to exactly one timeslice: the slices
        # partition the history rather than accumulating it
        slices = list(self.uut.timeslices)

        per_slice = [sum(len(dps) for dps in s.dp_lists.values()) for s in slices]
        total = sum(len(dps) for dps in self.uut.gitdata.dp_lists.values())

        self.assertEqual(total, sum(per_slice))

    def test_issues_are_sliced_by_creation_not_by_last_update(self):
        # An issue is a snapshot of its state at mining time, so its datapoint
        # carries a much later updated_at than created_at. It belongs to the
        # snapshot in which it was created - the edges derived from it
        # (create_issue, comment_issue) carry its creation timestamp.
        issue = _issue_datapoint(
            number=1,
            created_at=datetime.datetime(2009, 3, 1, tzinfo=datetime.timezone.utc),
            updated_at=datetime.datetime(2011, 6, 1, tzinfo=datetime.timezone.utc),
        )
        gitdata = _gitdata_with_issues(issue)
        git2kg = Git2KG(gitdata, freq="YE")

        timeindex = git2kg.get_timeindex()
        holding = [
            (timeindex[i], timeindex[i + 1])
            for i, timeslice in enumerate(git2kg.timeslices)
            if len(timeslice.issues)
        ]

        self.assertEqual(1, len(holding))
        lower, upper = holding[0]
        self.assertTrue(lower <= issue.created_at < upper)
        self.assertFalse(lower <= issue.updated_at < upper)

    def test_mining_resumes_from_the_last_issue_update(self):
        # Mining is a separate concern: an issue that was updated later has to
        # be mined again, so find_latest goes by updated_at.
        created_later = _issue_datapoint(
            number=1,
            created_at=datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc),
            updated_at=datetime.datetime(2010, 2, 1, tzinfo=datetime.timezone.utc),
        )
        updated_later = _issue_datapoint(
            number=2,
            created_at=datetime.datetime(2009, 1, 1, tzinfo=datetime.timezone.utc),
            updated_at=datetime.datetime(2011, 1, 1, tzinfo=datetime.timezone.utc),
        )
        issues = DatapointList([created_later, updated_later])

        self.assertEqual(2, issues.find_latest().issue.number)

    def test_get_networks(self):
        nets = self.uut.get_networks()

        self.assertEqual(len(nets), 17)
        for net in nets:
            self.assertEqual(
                type(net), dict, 'Networks object should be of type "dict"'
            )
            actor_actor_nets = net["actor"]["actor"]
            actor_issue_nets = net["actor"]["issue"]
            issue_actor_nets = net["issue"]["actor"]
            issue_issue_nets = net["issue"]["issue"]
            self.assertEqual(
                {
                    "edit_add_edges",
                    "edit_delete_edges",
                    "issue_comment_edges",
                    "review_request_edges",
                    "review_edges",
                    "assign_issue_edges",
                    "mentions_edges",
                },
                set(actor_actor_nets.keys()),
            )
            self.assertEqual(
                {
                    "close_issue_edges",
                    "merge_pull_edges",
                    "create_issue_edges",
                    "review_pull_edges",
                    "comment_issue_edges",
                    "reopen_issue_edges",
                },
                set(actor_issue_nets.keys()),
            )
            self.assertEqual({"assigned_to_edges"}, set(issue_actor_nets.keys()))
            self.assertEqual({"referenced_on_edges"}, set(issue_issue_nets.keys()))
            for _, edges in actor_actor_nets.items():
                self.assertEqual(type(edges), pd.DataFrame)
                self.assertEqual(set(edges.columns), set(["u", "v", "weight"]))

    def test_get_networks_exclude(self):
        nets = self.uut.get_networks(
            exclude=[
                "edit_add_edges",
                "edit_delete_edges",
                "issue_comment_edges",
                "merge_pull_edges",
            ]
        )
        self.assertEqual(len(nets), 17)
        for net in nets:
            actor_actor_nets = net["actor"]["actor"]
            actor_issue_nets = net["actor"]["issue"]
            issue_actor_nets = net["issue"]["actor"]
            issue_issue_nets = net["issue"]["issue"]
            self.assertEqual(
                {
                    "review_request_edges",
                    "review_edges",
                    "assign_issue_edges",
                    "mentions_edges",
                },
                set(actor_actor_nets.keys()),
            )
            self.assertEqual(
                {
                    "close_issue_edges",
                    "create_issue_edges",
                    "review_pull_edges",
                    "comment_issue_edges",
                    "reopen_issue_edges",
                },
                set(actor_issue_nets.keys()),
            )
            self.assertEqual({"assigned_to_edges"}, set(issue_actor_nets.keys()))
            self.assertEqual({"referenced_on_edges"}, set(issue_issue_nets.keys()))
            for _, edges in actor_actor_nets.items():
                self.assertEqual(type(edges), pd.DataFrame)
                self.assertEqual(set(edges.columns), set(["u", "v", "weight"]))

    def test_get_networks_serial_matches_parallel(self):
        serial_nets = self.uut.get_networks(processes=1)
        parallel_nets = self.uut.get_networks()

        self.assertEqual(len(serial_nets), len(parallel_nets))
        for serial_net, parallel_net in zip(serial_nets, parallel_nets, strict=True):
            self.assertEqual(serial_net.keys(), parallel_net.keys())
            for src in serial_net:
                self.assertEqual(serial_net[src].keys(), parallel_net[src].keys())
                for dst in serial_net[src]:
                    self.assertEqual(
                        serial_net[src][dst].keys(), parallel_net[src][dst].keys()
                    )
                    for relation, serial_df in serial_net[src][dst].items():
                        parallel_df = parallel_net[src][dst][relation]
                        pd.testing.assert_frame_equal(
                            serial_df.sort_values(["u", "v"]).reset_index(drop=True),
                            parallel_df.sort_values(["u", "v"]).reset_index(drop=True),
                        )

    def test_get_continuous_edges(self):
        edges = self.uut.get_continuous_edges()

        self.assertEqual(type(edges), dict)
        actor_actor_edges = edges["actor"]["actor"]
        actor_issue_edges = edges["actor"]["issue"]
        issue_actor_edges = edges["issue"]["actor"]
        issue_issue_edges = edges["issue"]["issue"]
        self.assertEqual(
            {
                "edit_add_edges",
                "edit_delete_edges",
                "issue_comment_edges",
                "review_request_edges",
                "review_edges",
                "assign_issue_edges",
                "mentions_edges",
            },
            set(actor_actor_edges.keys()),
        )
        self.assertEqual(
            {
                "close_issue_edges",
                "merge_pull_edges",
                "create_issue_edges",
                "review_pull_edges",
                "comment_issue_edges",
                "reopen_issue_edges",
            },
            set(actor_issue_edges.keys()),
        )
        self.assertEqual({"assigned_to_edges"}, set(issue_actor_edges.keys()))
        self.assertEqual({"referenced_on_edges"}, set(issue_issue_edges.keys()))
        for _, df in actor_actor_edges.items():
            self.assertEqual(type(df), pd.DataFrame)
            self.assertEqual(set(df.columns), {"u", "v", "ts"})
            self.assertTrue(df["ts"].is_monotonic_increasing)

    def test_get_continuous_edges_exclude(self):
        edges = self.uut.get_continuous_edges(
            exclude=[
                "edit_add_edges",
                "edit_delete_edges",
                "issue_comment_edges",
                "merge_pull_edges",
            ]
        )
        actor_actor_edges = edges["actor"]["actor"]
        actor_issue_edges = edges["actor"]["issue"]
        self.assertEqual(
            {
                "review_request_edges",
                "review_edges",
                "assign_issue_edges",
                "mentions_edges",
            },
            set(actor_actor_edges.keys()),
        )
        self.assertEqual(
            {
                "close_issue_edges",
                "create_issue_edges",
                "review_pull_edges",
                "comment_issue_edges",
                "reopen_issue_edges",
            },
            set(actor_issue_edges.keys()),
        )

    def test_get_continuous_edges_preserve_every_interaction(self):
        """The continuous edge stream must not lose or duplicate interactions
        relative to the (aggregated, weighted) discrete networks: per relation,
        the number of continuous rows must equal the sum of discrete weights
        across all timeslices.
        """
        continuous_edges = self.uut.get_continuous_edges()
        discrete_nets = self.uut.get_networks()

        for src, dsts in continuous_edges.items():
            for dst, relations in dsts.items():
                for relation, continuous_df in relations.items():
                    weight_sum = sum(
                        net[src][dst][relation]["weight"].sum()
                        for net in discrete_nets
                        if relation in net[src][dst]
                    )
                    self.assertEqual(
                        len(continuous_df),
                        weight_sum,
                        f"Mismatch for {src}->{dst}::{relation}",
                    )

    def test_get_continuous_network(self):
        cn = self.uut.get_continuous_network()

        self.assertEqual(type(cn), ContinuousNetwork)
        edges = self.uut.get_continuous_edges()
        self.assertEqual(cn.edges.keys(), edges.keys())
        for src in cn.edges:
            self.assertEqual(cn.edges[src].keys(), edges[src].keys())
            for dst in cn.edges[src]:
                self.assertEqual(cn.edges[src][dst].keys(), edges[src][dst].keys())
                for relation, df in cn.edges[src][dst].items():
                    pd.testing.assert_frame_equal(df, edges[src][dst][relation])
        self.assertEqual(set(cn.features.keys()), {"actor", "issue", "commit", "file"})
        # Features are computed once, over the full history - not per timeslice
        self.assertEqual(len(cn.features["actor"]), 4)

        as_dict = cn.to_dict()
        self.assertEqual(set(as_dict.keys()), {"edges", "features"})
        self.assertIs(as_dict["edges"], cn.edges)
        self.assertIs(as_dict["features"], cn.features)

    def test_get_continuous_network_exclude(self):
        cn = self.uut.get_continuous_network(exclude=["edit_add_edges"])
        self.assertNotIn("edit_add_edges", cn.edges["actor"]["actor"])

    def test_continuous_network_to_unit_time_default_origin(self):
        cn = self.uut.get_continuous_network()
        earliest = min(
            df["ts"].min()
            for dsts in cn.edges.values()
            for relations in dsts.values()
            for df in relations.values()
            if len(df)
        )

        unit_time_network = cn.to_unit_time(unit="s")

        self.assertEqual(type(unit_time_network), ContinuousNetwork)
        self.assertIs(unit_time_network.features, cn.features)
        found_zero = False
        for src, dsts in unit_time_network.edges.items():
            for dst, relations in dsts.items():
                for relation, unit_df in relations.items():
                    original_df = cn.edges[src][dst][relation]
                    self.assertEqual(unit_df["ts"].dtype, np.float64)
                    expected = (original_df["ts"] - earliest) / pd.Timedelta(1, "s")
                    pd.testing.assert_series_equal(
                        unit_df["ts"], expected, check_names=False
                    )
                    if (unit_df["ts"] == 0.0).any():
                        found_zero = True
        self.assertTrue(found_zero, "Earliest interaction should map to time 0")

        # Original network is left untouched
        for dsts in cn.edges.values():
            for relations in dsts.values():
                for df in relations.values():
                    self.assertEqual(df["ts"].dtype, "datetime64[ns, UTC]")

    def test_continuous_network_to_unit_time_custom_origin_and_unit(self):
        cn = self.uut.get_continuous_network()
        origin = self.uut.gitdata.first_interaction() - pd.Timedelta(days=1)

        seconds_network = cn.to_unit_time(unit="s", origin=origin)
        days_network = cn.to_unit_time(unit="D", origin=origin)

        seconds_ts = seconds_network.edges["actor"]["actor"]["edit_add_edges"]["ts"]
        days_ts = days_network.edges["actor"]["actor"]["edit_add_edges"]["ts"]
        # 1 day = 86400 seconds, regardless of the chosen origin
        pd.testing.assert_series_equal(days_ts * 86400, seconds_ts, check_names=False)
        self.assertTrue((seconds_ts > 0).all())

    def test_get_feature_matrices(self):
        features_maps = self.uut.get_feature_matrices(processes=1)
        self.assertEqual(len(features_maps), 17)

        for feature in features_maps:
            M_actor = feature["actor"]
            self.assertEqual(
                type(M_actor),
                pd.DataFrame,
                'Featrue matrix should be of type "pandas.DataFrame"',
            )
            self.assertEqual(
                len(M_actor), 4
            )  # Constnat lenght, since all authors are alway present as nodes
            self.assertTrue("feature_NumCommitsActivity" in M_actor.columns)
            self.assertTrue("feature_TotalLinesAdded" in M_actor.columns)
            self.assertTrue("feature_TotalLinesRemoved" in M_actor.columns)
            self.assertTrue("feature_UniqueFilesChanged" in M_actor.columns)
            self.assertTrue("feature_CommitTimes" in M_actor.columns)
            self.assertTrue("feature_FirstCommitDate" in M_actor.columns)
            self.assertTrue("feature_LastCommitDate" in M_actor.columns)
            self.assertTrue("feature_AverageCommitMessagePolarity" in M_actor.columns)
            self.assertTrue(
                "feature_AverageCommitMessageSubjectivity" in M_actor.columns
            )
            self.assertTrue("feature_AverageCommitTime" in M_actor.columns)
            self.assertTrue("feature_NumIssueComments" in M_actor.columns)
            self.assertTrue("feature_NumIssuesCreated" in M_actor.columns)
            self.assertTrue("feature_NumIssuesAssigned" in M_actor.columns)
            self.assertTrue("feature_NumPullRequestsCreated" in M_actor.columns)
            self.assertTrue("feature_NumPullRequestsReviewed" in M_actor.columns)
            self.assertTrue("feature_NumAuthoredMergedPullRequests" in M_actor.columns)
            self.assertTrue("feature_CommitConsistency" in M_actor.columns)
            self.assertTrue("feature_NumDocumentationChanges" in M_actor.columns)
            self.assertTrue("feature_NumBugfixCommits" in M_actor.columns)
            self.assertTrue("feature_BugfixCommitRatio" in M_actor.columns)
            self.assertTrue("feature_AvgCommentPolarity" in M_actor.columns)
            self.assertTrue("feature_AvgCommentSubjectivity" in M_actor.columns)
            self.assertTrue("feature_AvgIssueResponseTime" in M_actor.columns)
            self.assertTrue("feature_AvgCommentSubjectivity" in M_actor.columns)
            self.assertTrue("feature_AvgReviewTime" in M_actor.columns)

            M_issue = feature["issue"]
            self.assertTrue("feature_IssueAge" in M_issue.columns)
            self.assertTrue("feature_TimeOpen" in M_issue.columns)
            self.assertTrue("feature_IsOpen" in M_issue.columns)
            self.assertTrue("feature_IsMerged" in M_issue.columns)
            self.assertTrue("feature_IsPull" in M_issue.columns)
            self.assertTrue("feature_AvgSentimentPolarity" in M_issue.columns)
            self.assertTrue("feature_AvgSentimentSubjectivity" in M_issue.columns)
            self.assertTrue("feature_AvgCommentTime" in M_issue.columns)

    def test_get_snapshots_passes_processes_to_both_heavy_stages(self):
        """Without this, a snapshot run starts one process per core whatever
        the caller asked for, and a large repository runs out of memory."""
        one_slice = pd.DatetimeIndex(["2024-01-01", "2024-02-01"], tz="UTC")
        with (
            mock.patch.object(Git2KG, "get_timeindex", return_value=one_slice),
            mock.patch.object(Git2KG, "get_networks", return_value=[{}]) as networks,
            mock.patch.object(
                Git2KG, "get_feature_matrices", return_value=[{}]
            ) as features,
            mock.patch.object(Git2KG, "get_active_masks", return_value=[{}]),
            mock.patch.object(Git2KG, "get_labels", return_value=[{}]),
        ):
            self.uut.get_snapshots(processes=2)

        self.assertEqual(2, networks.call_args.kwargs["processes"])
        self.assertEqual(2, features.call_args.kwargs["processes"])

    def test_get_feature_matrices_serial_matches_parallel(self):
        serial_features = self.uut.get_feature_matrices(processes=1)
        parallel_features = self.uut.get_feature_matrices()

        self.assertEqual(len(serial_features), len(parallel_features))
        sort_key = {
            "actor": ActorInfo.ID_COLUMN,
            "issue": "issue_number",
            "commit": "commit_id",
            "file": "file_id",
        }
        for serial_feature, parallel_feature in zip(
            serial_features, parallel_features, strict=True
        ):
            self.assertEqual(serial_feature.keys(), parallel_feature.keys())
            for node_type, serial_df in serial_feature.items():
                parallel_df = parallel_feature[node_type]
                key = sort_key[node_type]
                pd.testing.assert_frame_equal(
                    serial_df.sort_values(key).reset_index(drop=True),
                    parallel_df.sort_values(key).reset_index(drop=True),
                )

    def test_add_feature(self):
        class NewFeatureExtractor(ActorFeatureExtractor):
            def extract_feature(self) -> pd.DataFrame:

                commit_df = self.data.get_as_dataframe("commits")
                return (
                    commit_df.groupby("author_id")
                    .agg({"n_modified_files": "mean"})
                    .reset_index()
                )

        feature_matrices = self.uut.get_feature_matrices()
        M = feature_matrices[0]["actor"]
        self.assertTrue("feature_NewFeature" in M.columns)

    def test_features_masks_and_labels_share_one_row_order(self):
        # Features, active masks and labels are aligned positionally, so every
        # snapshot has to lay them out in the order of the same id universe.
        snapshots = self.uut.get_snapshots(
            actor_label_extractor=DeveloperLeaverLabelExtractor,
            issue_label_extractor=IssueClosedNextSliceLabelExtractor,
            file_label_extractor=FileModifiedNextSliceLabelExtractor,
        )
        ids = self.uut._id_universes()

        for features, masks, labels in zip(
            snapshots.features, snapshots.active_masks, snapshots.labels, strict=True
        ):
            for node_type, node_ids in ids.items():
                id_column = features[node_type].columns[0]
                self.assertListEqual(
                    list(node_ids), list(features[node_type][id_column])
                )
                self.assertEqual(len(node_ids), len(masks[node_type]))
                if len(labels[node_type]):
                    self.assertEqual(len(node_ids), len(labels[node_type]))

    def _populated_timeslice(self):
        """A timeslice of the fixture that holds commits of its own."""
        for timeslice in self.uut.timeslices:
            if len(timeslice.commits):
                return timeslice
        self.fail("the fixture has no timeslice holding commits")

    def test_a_git2kg_built_on_a_timeslice_knows_its_commits_and_files(self):
        # Zooming into one timeslice by wrapping it in a Git2KG of its own
        zoomed = Git2KG(
            self._populated_timeslice(),
            freq="D",
            name="zoomed",
            actor_mapping=self.uut.actor_mapping,
        )

        universes = zoomed._id_universes()
        for node_type in ("commit", "file"):
            with self.subTest(node_type):
                self.assertEqual(
                    list(self.uut._id_universes()[node_type]),
                    list(universes[node_type]),
                )

    def test_building_a_git2kg_on_a_timeslice_keeps_the_actor_ids(self):
        # A timeslice shares its datapoints with the whole history, so
        # disambiguating it again would renumber the actors of both
        timeslice = self._populated_timeslice()
        before = [commit.author.get_id() for commit in self.gitdata.commits]

        Git2KG(timeslice, freq="D", name="z", actor_mapping=self.uut.actor_mapping)

        after = [commit.author.get_id() for commit in self.gitdata.commits]
        self.assertListEqual(before, after)

    def test_save(self):
        with TemporaryDirectory() as tmp:
            self.uut.save(tmp)
            self.assertTrue(os.path.isfile(f"{tmp}/scientists_repo-git2kg.tar.gz"))

    def test_save_stores_the_actor_mapping_and_its_version(self):
        with TemporaryDirectory() as tmp:
            archive = self.uut.save(tmp)
            with tarfile.open(archive) as tar:
                names = tar.getnames()
                metadata = json.load(tar.extractfile("metadata.json"))

        self.assertIn("scientists_repo-actor_mapping.csv", names)
        self.assertEqual(DISAMBIGUATION_VERSION, metadata["disambiguation_version"])

    def test_load_reuses_the_stored_mapping_instead_of_disambiguating(self):
        with TemporaryDirectory() as tmp:
            archive = self.uut.save(tmp)
            with mock.patch("git2kg.core.disambiguated_author_mapping") as disambiguate:
                loaded = Git2KG.load(archive)

        disambiguate.assert_not_called()
        pd.testing.assert_frame_equal(self.uut.actor_mapping, loaded.actor_mapping)
        self.assertEqual(
            {(str(a.name), a.id) for a in self.uut.gitdata.get_actors()},
            {(str(a.name), a.id) for a in loaded.gitdata.get_actors()},
        )

    def test_load_disambiguates_again_when_the_stored_version_is_stale(self):
        with TemporaryDirectory() as tmp:
            archive = _rewrite_metadata(
                self.uut.save(tmp),
                {"disambiguation_version": DISAMBIGUATION_VERSION + 1},
            )
            loaded = Git2KG.load(archive)

        # Recomputing has to give the same answer - that is what makes the
        # stored mapping reusable in the first place.
        pd.testing.assert_frame_equal(self.uut.actor_mapping, loaded.actor_mapping)

    def test_load_disambiguates_again_for_archives_without_a_mapping(self):
        with TemporaryDirectory() as tmp:
            archive = _drop_from_archive(
                self.uut.save(tmp), "scientists_repo-actor_mapping.csv"
            )
            with mock.patch(
                "git2kg.core.disambiguated_author_mapping",
                wraps=disambiguated_author_mapping,
            ) as disambiguate:
                loaded = Git2KG.load(archive)

        disambiguate.assert_called_once()
        pd.testing.assert_frame_equal(self.uut.actor_mapping, loaded.actor_mapping)

    def test____str__(self):
        print(self.uut)

    def test_load_with_several_processes_matches_one(self):
        """`processes` only changes how the datapoints are rebuilt."""
        archive = "test/resources/saved-git2kgs/scientists_repo-git2kg.tar.gz"

        single = Git2KG.load(archive)
        parallel = Git2KG.load(archive, processes=2)

        self.assertEqual(
            {n: len(dps) for n, dps in single.gitdata.dp_lists.items()},
            {n: len(dps) for n, dps in parallel.gitdata.dp_lists.items()},
        )
        self.assertListEqual(
            [c.commit.hash for c in single.gitdata.commits],
            [c.commit.hash for c in parallel.gitdata.commits],
        )
        self.assertListEqual(
            list(single._id_universes()["actor"]),
            list(parallel._id_universes()["actor"]),
        )

    def test_load(self):
        g2kg: Git2KG = Git2KG.load(
            "test/resources/saved-git2kgs/scientists_repo-git2kg.tar.gz"
        )
        self.assertEqual(g2kg.freq, self.uut.freq)
        self.assertEqual(g2kg.name, self.uut.name)
        self.assertEqual(g2kg.mined_on, self.uut.mined_on)
        for orig, loaded in zip(
            self.uut.gitdata.commits, g2kg.gitdata.commits, strict=True
        ):
            self.assertEqual(loaded.commit.hash, orig.commit.hash)

        self.assertTrue(
            all(
                self.uut.actor_mapping.sort_values(
                    [ActorInfo.ID_COLUMN, "alias_name", "alias_email"]
                )
                .reset_index()
                .drop("index", axis=1)
                == g2kg.actor_mapping.sort_values(
                    [ActorInfo.ID_COLUMN, "alias_name", "alias_email"]
                )
                .reset_index()
                .drop("index", axis=1)
            )
        )
        self.assertTrue(all([len(dpl) > 0 for dpl in g2kg.gitdata.dp_lists]))
        self.assertIsNotNone(g2kg.gitdata.file_registry)

    def test_get_snapshots(self):
        static_g2kg = Git2KG(self.gitdata, freq=None, name="scientists_repo")
        snapshots: NetworkSnapshots = static_g2kg.get_snapshots()

        self.assertEqual(len(snapshots.networks), 1)

        network = snapshots.networks[0]
        edge_count_per_type = {}

        for _, dst in network.items():
            for __, etype in dst.items():
                for edge_type, arr in etype.items():
                    edge_count_per_type[edge_type] = arr.shape[0]
        expected_counts = {
            "edit_add_edges": 4,
            "edit_delete_edges": 5,
            "issue_comment_edges": 4,
            "review_request_edges": 2,
            "review_edges": 1,
            "assign_issue_edges": 1,
            "mentions_edges": 3,
            "create_issue_edges": 2,
            "comment_issue_edges": 3,
            "close_issue_edges": 2,
            "review_pull_edges": 1,
            "merge_pull_edges": 2,
            "assigned_to_edges": 1,
            "referenced_on_edges": 2,
        }
        for edge_type, count in expected_counts.items():
            self.assertEqual(
                edge_count_per_type[edge_type], count, f"Mismatch for {edge_type}"
            )

    @NEEDS_TOKEN
    def test_from_github_url(self):
        print("Start mining at: ", datetime.datetime.now())
        with TemporaryDirectory() as tmp:
            g2kg = Git2KG.from_github_url(
                "https://github.com/torvalds/pesconvert",
                github_token=self.token,
                persistence_path=tmp,
            )
        print("Finished mining at: ", datetime.datetime.now())
        self.assertEqual(g2kg.name, "torvalds_pesconvert")
        self.assertTrue(len(g2kg.gitdata.commits) > 0)
        self.assertTrue(len(g2kg.gitdata.edits) > 0)
        self.assertTrue(len(g2kg.gitdata.issues) > 0)
        self.assertTrue(len(g2kg.gitdata.issue_comments) > 0)
        self.assertTrue(len(g2kg.gitdata.events) > 0)
        self.assertTrue(len(g2kg.gitdata.reviews) > 0)
        self.assertTrue(len(g2kg.gitdata.issue_timeline_events) > 0)

    @NEEDS_TOKEN
    def test_from_github_url_over_graphql(self):
        print("Start mining over GraphQL at: ", datetime.datetime.now())
        with TemporaryDirectory() as tmp:
            g2kg = Git2KG.from_github_url(
                SMALL_REPO_URL,
                github_token=self.token,
                persistence_path=tmp,
                github_api=GRAPHQL,
            )
        print("Finished mining over GraphQL at: ", datetime.datetime.now())
        self.assertEqual(g2kg.name, "torvalds_pesconvert")
        self.assertTrue(len(g2kg.gitdata.commits) > 0)
        self.assertTrue(len(g2kg.gitdata.edits) > 0)
        self.assertTrue(len(g2kg.gitdata.issues) > 0)
        self.assertTrue(len(g2kg.gitdata.issue_comments) > 0)
        self.assertTrue(len(g2kg.gitdata.events) > 0)
        self.assertTrue(len(g2kg.gitdata.reviews) > 0)
        self.assertTrue(len(g2kg.gitdata.issue_timeline_events) > 0)

    @NEEDS_TOKEN
    def test_the_two_github_apis_mine_the_same_data(self):
        """The loaders are interchangeable, so the same repository has to come
        back the same through either of them."""
        mined = {}
        for api in (REST, GRAPHQL):
            started = datetime.datetime.now()
            with TemporaryDirectory() as tmp:
                mined[api] = Git2KG.from_github_url(
                    SMALL_REPO_URL,
                    github_token=self.token,
                    persistence_path=tmp,
                    github_api=api,
                ).gitdata
            print(f"Mined over {api} in {datetime.datetime.now() - started}")

        rest, graphql = mined[REST], mined[GRAPHQL]
        for dp_name in ("issues", "events", "reviews"):
            with self.subTest(dp_name):
                self.assertEqual(
                    _identifiers(getattr(rest, dp_name), dp_name),
                    _identifiers(getattr(graphql, dp_name), dp_name),
                )

        # GitHub does not always resolve a comment's author over GraphQL even
        # though the REST API still serves it, and an author it does not
        # resolve is no actor. The GraphQL loader therefore sees a subset
        over_rest = _identifiers(rest.issue_comments, "issue_comments")
        over_graphql = _identifiers(graphql.issue_comments, "issue_comments")
        self.assertTrue(over_graphql)
        self.assertSetEqual(set(), over_graphql - over_rest)

    @NEEDS_TOKEN
    def test_graphql_resolves_mentions_through_github(self):
        """Mentions come from GitHub's own rendering, so an @token that is not
        an account - an annotation or a javadoc tag - is never an actor."""
        with TemporaryDirectory() as tmp:
            gitdata = Git2KG.from_github_url(
                SMALL_REPO_URL,
                github_token=self.token,
                persistence_path=tmp,
                github_api=GRAPHQL,
            ).gitdata

        mentioned = {
            actor.name
            for dp in list(gitdata.issues) + list(gitdata.issue_comments)
            for actor in dp.mentioned_actors
        }
        self.assertTrue(mentioned, "the repository has no mentions to check")

        # Every mention has to be an account that exists. A mentioned account
        # need not contribute to this repository, so it cannot be checked
        # against the actors mined from it
        unresolved = _logins_that_do_not_exist(mentioned, self.token)
        self.assertSetEqual(set(), unresolved)
