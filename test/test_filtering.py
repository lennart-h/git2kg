import unittest

import pandas as pd

from git2kg.core import Git2KG
from git2kg.datapoint import ActorInfo, CommitInfo
from git2kg.filtering import (
    BotFilter,
    NodeFilter,
    development_commits,
    excluded_ids,
    is_development_commit,
)
from git2kg.gitdata import _load_gitdata
from test.test_gitdata import _commit_datapoint
from test.test_labels import _gitdata_with_commits


class TestBotFilter(unittest.TestCase):
    def test_excludes_accounts_github_reports_as_bots(self):
        self.assertTrue(BotFilter().excludes(ActorInfo(name="whatever", type="Bot")))

    def test_excludes_bots_arriving_without_a_type(self):
        # A bot committing through git, or mentioned in a comment, carries no
        # GitHub type - only its name or email says what it is
        for actor in (
            ActorInfo(name="dependabot[bot]"),
            ActorInfo(name="dependabot"),
            ActorInfo(name="github-actions"),
            ActorInfo(name="GitHub", email="noreply@github.com", type="git"),
        ):
            self.assertTrue(BotFilter().excludes(actor), actor.name)

    def test_keeps_humans(self):
        for actor in (
            ActorInfo(name="Marie Curie", email="marie@radioactive.pl", type="git"),
            ActorInfo(name="mariecurie", github_id=1867, type="User"),
            # "robot" is not "[bot]", and a human may well write about bots
            ActorInfo(name="robotics-fan", email="robot@example.org"),
        ):
            self.assertFalse(BotFilter().excludes(actor), actor.name)

    def test_patterns_are_configurable(self):
        only_dependabot = BotFilter(patterns=[r"^dependabot"])

        self.assertTrue(only_dependabot.excludes(ActorInfo(name="dependabot")))
        self.assertFalse(only_dependabot.excludes(ActorInfo(name="github-actions")))
        # The GitHub type still excludes, whatever the patterns say
        self.assertTrue(only_dependabot.excludes(ActorInfo(name="x", type="Bot")))


class TestExcludedIds(unittest.TestCase):
    def test_collects_ids_per_node_type(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")
        git2kg = Git2KG(gitdata, freq=None, filters=[])
        turing = next(
            a for a in gitdata.get_actors() if a.name == "alanturing" and a.github_id
        )

        class ExcludeTuring(NodeFilter):
            node_type = "actor"

            def excludes(self, entity: ActorInfo) -> bool:
                return entity.id == turing.id

        excluded = excluded_ids(gitdata, [ExcludeTuring()])

        self.assertEqual({turing.id}, excluded["actor"])
        self.assertEqual(set(), excluded["issue"])
        self.assertEqual(set(), excluded["commit"])
        self.assertEqual(set(), excluded["file"])
        # Guards against the universe being empty for an unrelated reason
        self.assertIn(turing.id, set(git2kg._id_universes()["actor"]))


class TestFiltersInGit2KG(unittest.TestCase):
    def setUp(self):
        self.gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.unfiltered = Git2KG(self.gitdata, freq=None, filters=[])
        self.excluded_id = int(self.unfiltered._id_universes()["actor"].iloc[0])

        excluded_id = self.excluded_id

        class ExcludeOne(NodeFilter):
            node_type = "actor"

            def excludes(self, entity: ActorInfo) -> bool:
                return entity.id == excluded_id

        self.filtered = Git2KG(
            _load_gitdata("test/resources", "gitdata-test"),
            freq=None,
            filters=[ExcludeOne()],
        )

    def test_excluded_actors_leave_the_id_universe(self):
        universes = self.filtered._id_universes()

        self.assertNotIn(self.excluded_id, set(universes["actor"]))
        self.assertEqual(
            len(self.unfiltered._id_universes()["actor"]) - 1, len(universes["actor"])
        )

    def test_features_masks_and_labels_follow_the_filtered_universe(self):
        snapshots = self.filtered.get_snapshots()
        expected = len(self.filtered._id_universes()["actor"])

        for features, masks in zip(
            snapshots.features, snapshots.active_masks, strict=True
        ):
            self.assertEqual(expected, len(features["actor"]))
            self.assertEqual(expected, len(masks["actor"]))

    def test_no_edge_touches_an_excluded_actor(self):
        edges = self.filtered.get_continuous_edges()

        for u_type, tos in edges.items():
            for v_type, relations in tos.items():
                for relation, edge_set in relations.items():
                    if u_type == "actor":
                        self.assertNotIn(self.excluded_id, set(edge_set["u"]), relation)
                    if v_type == "actor":
                        self.assertNotIn(self.excluded_id, set(edge_set["v"]), relation)

    def test_bots_are_excluded_by_default(self):
        self.assertEqual(
            (BotFilter.__name__,),
            tuple(
                type(f).__name__
                for f in Git2KG(
                    _load_gitdata("test/resources", "gitdata-test"), freq=None
                ).filters
            ),
        )


class TestDevelopmentCommits(unittest.TestCase):
    def _commit(self, is_merge: bool):
        commit = _commit_datapoint(
            CommitInfo("a" * 40, id=0),
            author=ActorInfo(name="Alice", email="alice@example.org", id=0),
            committer=ActorInfo(name="Alice", email="alice@example.org", id=0),
            author_date=pd.Timestamp("2024-01-01", tz="UTC"),
            committer_date=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        commit.is_merge = is_merge
        return commit

    def test_merge_commits_are_not_development(self):
        self.assertTrue(is_development_commit(self._commit(is_merge=False)))
        self.assertFalse(is_development_commit(self._commit(is_merge=True)))

    def test_dataframe_form_leaves_merge_commits_out(self):
        gitdata = _gitdata_with_commits(
            self._commit(is_merge=False), self._commit(is_merge=True)
        )

        commits = gitdata.get_as_dataframe("commits")

        self.assertEqual(2, len(commits))
        self.assertEqual(1, len(development_commits(commits)))


if __name__ == "__main__":
    unittest.main()
