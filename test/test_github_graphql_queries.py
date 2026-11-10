"""Validates the GraphQL documents against GitHub's published schema.

The loader's own tests answer its queries from canned pages, so they exercise
the mapping but never the query documents. A misspelled field or a wrong enum
value would surface only on a live call, which needs a token. Checking the
documents against the schema catches that here instead.

`test/resources/github-schema.graphql.gz` is GitHub's published schema,
downloaded from https://docs.github.com/public/fpt/schema.docs.graphql on
2026-10-21.
"""

import gzip
import pathlib
import unittest

from graphql import build_schema, parse, validate

from git2kg.dataloading.githubgraphqlloader import (
    ISSUE_EVENT_ITEM_TYPES,
    ISSUES_QUERY,
    PULL_EVENT_ITEM_TYPES,
    PULLS_QUERY,
    more_query,
)

SCHEMA_PATH = pathlib.Path(__file__).parent / "resources" / "github-schema.graphql.gz"

# Every variant of the follow-up query the loader builds in `_all_nodes`
MORE_VARIANTS = [
    ("comments", False),
    ("timelineItems", False),
    ("comments", True),
    ("reviews", True),
    ("timelineItems", True),
]


class TestQueriesAgainstSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(SCHEMA_PATH, "rt", encoding="utf-8") as schema_file:
            source = schema_file.read()
        # GitHub's own schema deprecates fields that the interfaces they
        # implement do not, which the schema validator rejects. That is a
        # property of GitHub's schema, not of the queries under test
        cls.schema = build_schema(source, assume_valid=True)

    def _assert_valid(self, name, query):
        errors = validate(self.schema, parse(query))
        self.assertEqual(
            [], [str(e) for e in errors], f"{name} is not valid against the schema"
        )

    def test_issues_query(self):
        self._assert_valid("ISSUES_QUERY", ISSUES_QUERY)

    def test_pulls_query(self):
        self._assert_valid("PULLS_QUERY", PULLS_QUERY)

    def test_follow_up_queries(self):
        for connection, is_pull in MORE_VARIANTS:
            parent = "PullRequest" if is_pull else "Issue"
            with self.subTest(f"{parent}.{connection}"):
                self._assert_valid(
                    f"more_query[{parent}.{connection}]",
                    more_query(connection, is_pull),
                )

    def test_requested_timeline_item_types_exist(self):
        for enum_name, values in (
            ("IssueTimelineItemsItemType", ISSUE_EVENT_ITEM_TYPES),
            ("PullRequestTimelineItemsItemType", PULL_EVENT_ITEM_TYPES),
        ):
            with self.subTest(enum_name):
                enum_type = self.schema.get_type(enum_name)
                self.assertIsNotNone(enum_type)
                for value in values:
                    self.assertIn(value, enum_type.values)  # type: ignore[union-attr]

    def test_the_datapoint_ids_do_not_use_a_deprecated_field(self):
        # `databaseId` is deprecated on PullRequest and PullRequestReview in
        # favour of `fullDatabaseId`, so the ids are read from the latter
        for type_name in ("Issue", "PullRequest", "IssueComment", "PullRequestReview"):
            with self.subTest(type_name):
                field = self.schema.get_type(type_name).fields["fullDatabaseId"]  # type: ignore[union-attr]
                self.assertIsNone(field.deprecation_reason)


if __name__ == "__main__":
    unittest.main()
