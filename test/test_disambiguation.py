import unittest

import pandas as pd

from git2kg.datapoint import ActorInfo
from git2kg.disambiguation import (
    _canonical_actor_ids,
    alias_frame,
    disambiguated_author_mapping,
    mapping_matches,
)
from git2kg.gitdata import _load_gitdata
from git2kg.timeslicing import _partition

# Three actors, spread over five aliases. The actor_id values mimic what gambit
# hands back: assigned in row order, with a hole left behind by a merge.
ALIASES = [
    ("Alan Turing", "turingalan1912@computable.uk", None, 4),
    ("alanturing", None, 1912, 4),
    ("Marie Curie", "marie.curie@radioactive.pl", None, 0),
    ("mariecurie", "marie.curie+github@radioactive.pl", 1867, 0),
    ("Konrad Zuse", "kzuse@z3.de", None, 2),
]


def _mapping(rows) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "alias_name",
            "alias_email",
            ActorInfo.GITHUB_ID_COLUMN,
            ActorInfo.ID_COLUMN,
        ],
    )
    return frame.astype(
        {
            "alias_name": "string",
            "alias_email": "string",
            ActorInfo.GITHUB_ID_COLUMN: "Int64",
            # gambit leaves its id column as object dtype
            ActorInfo.ID_COLUMN: object,
        }
    )


def _ids_by_alias(mapping: pd.DataFrame) -> dict[str, int]:
    return {
        str(row.alias_name): int(getattr(row, ActorInfo.ID_COLUMN))
        for row in mapping.itertuples()
    }


class TestCanonicalActorIds(unittest.TestCase):
    def test_ids_are_contiguous(self):
        canonical = _canonical_actor_ids(_mapping(ALIASES))

        self.assertEqual([0, 1, 2], sorted(set(canonical[ActorInfo.ID_COLUMN])))

    def test_ids_do_not_depend_on_row_order(self):
        canonical = _canonical_actor_ids(_mapping(ALIASES))
        reversed_input = _canonical_actor_ids(_mapping(list(reversed(ALIASES))))
        # Same clusters, different gambit numbering: what row order changes in
        # gambit's output is only which id each group is given.
        renumbered = _canonical_actor_ids(
            _mapping([(*row[:3], {4: 7, 0: 3, 2: 9}[row[3]]) for row in ALIASES])
        )

        self.assertEqual(_ids_by_alias(canonical), _ids_by_alias(reversed_input))
        self.assertEqual(_ids_by_alias(canonical), _ids_by_alias(renumbered))

    def test_actors_are_numbered_by_their_first_alias(self):
        canonical = _canonical_actor_ids(_mapping(ALIASES))

        # "Alan Turing" < "Konrad Zuse" < "Marie Curie"
        self.assertEqual(
            {
                "Alan Turing": 0,
                "alanturing": 0,
                "Konrad Zuse": 1,
                "Marie Curie": 2,
                "mariecurie": 2,
            },
            _ids_by_alias(canonical),
        )

    def test_mapping_is_sorted_by_actor_id(self):
        canonical = _canonical_actor_ids(_mapping(ALIASES))
        ids = list(canonical[ActorInfo.ID_COLUMN])

        self.assertEqual(sorted(ids), ids)
        self.assertEqual("int64", canonical[ActorInfo.ID_COLUMN].dtype)

    def test_aliases_are_preserved(self):
        canonical = _canonical_actor_ids(_mapping(ALIASES))

        self.assertEqual(len(ALIASES), len(canonical))
        self.assertEqual(
            {row[0] for row in ALIASES}, set(canonical["alias_name"].astype(str))
        )


class TestDisambiguatedAuthorMapping(unittest.TestCase):
    def test_fixture_mapping_has_contiguous_sorted_ids(self):
        gitdata = _load_gitdata("test/resources", "gitdata-test")

        mapping = disambiguated_author_mapping(gitdata)
        ids = list(mapping[ActorInfo.ID_COLUMN])

        self.assertEqual(list(range(len(set(ids)))), sorted(set(ids)))
        self.assertEqual(sorted(ids), ids)


class TestMappingMatches(unittest.TestCase):
    def setUp(self):
        self.gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.mapping = disambiguated_author_mapping(self.gitdata)

    def test_accepts_the_mapping_computed_for_the_same_gitdata(self):
        self.assertTrue(mapping_matches(self.mapping, self.gitdata))

    def test_rejects_a_mapping_that_is_missing_an_alias(self):
        self.assertFalse(mapping_matches(self.mapping.iloc[1:], self.gitdata))

    def test_accepts_a_mapping_with_an_alias_the_data_does_not_have(self):
        extra = alias_frame(self.gitdata).iloc[[0]].assign(alias_name="Grace Hopper")
        extended = pd.concat([self.mapping, extra.assign(**{ActorInfo.ID_COLUMN: 99})])

        self.assertTrue(mapping_matches(extended, self.gitdata))

    def test_accepts_the_mapping_of_a_whole_history_for_one_of_its_timeslices(self):
        timeindex = pd.date_range("2007-01-01", "2026-01-01", freq="YS", tz="UTC")
        populated = [
            timeslice
            for timeslice in _partition(self.gitdata, timeindex)
            if len(timeslice.commits)
        ]

        self.assertTrue(populated)
        for timeslice in populated:
            self.assertTrue(mapping_matches(self.mapping, timeslice))

    def test_rejects_none_and_frames_without_alias_columns(self):
        self.assertFalse(mapping_matches(None, self.gitdata))
        self.assertFalse(mapping_matches(pd.DataFrame({"x": [1]}), self.gitdata))

    def test_ignores_row_order_and_ids(self):
        shuffled = self.mapping.iloc[::-1].assign(**{ActorInfo.ID_COLUMN: 0})

        self.assertTrue(mapping_matches(shuffled, self.gitdata))


if __name__ == "__main__":
    unittest.main()
