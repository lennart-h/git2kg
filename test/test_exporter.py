import unittest

import networkx as nx
import pathpyG as pp
import torch

import git2kg.features as features
from git2kg.core import Git2KG
from git2kg.export import NetworkXExporter, PathPyGExporter, PyGExporter
from git2kg.gitdata import _load_gitdata
from git2kg.labels import (
    DeveloperLeaverLabelExtractor,
    IssueClosedNextSliceLabelExtractor,
)


class TestExporter(unittest.TestCase):
    def setUp(self):

        gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.git2kg = Git2KG(gitdata, freq=None)

    def test_pyG_exporter(self):
        self.git2kg.freq = None
        snaps = self.git2kg.get_snapshots()
        exporter = PyGExporter()
        exports = exporter.export_snapshots(snapshots=snaps)
        exports = list(exports.values())
        self.assertEqual(len(exports), 1)
        export = exports[0]
        self.assertEqual(len(export["actor"].x), 4)
        self.assertEqual(len(export["issue"].x), 2)
        self.assertEqual(len(export["actor"].active_mask), 4)
        self.assertEqual(export["actor"].active_mask.tolist(), [True, True, True, True])
        self.assertEqual(
            export["actor", "merge_pull", "issue"]["edge_index"].shape, (2, 1)
        )

    def test_exporters_do_not_treat_node_ids_as_features(self):
        snaps = self.git2kg.get_snapshots()
        feature_widths = {
            node_type: matrix.shape[1]
            for node_type, matrix in snaps.features[0].items()
        }

        pyg_export = list(PyGExporter().export_snapshots(snaps).values())[0]
        for node_type, width in feature_widths.items():
            # One column less than the feature matrix: its id column
            self.assertEqual(width - 1, pyg_export[node_type].x.shape[1])

        pathpyg_export = list(PathPyGExporter().export_snapshots(snaps).values())[0]
        node_attributes = {
            attr for attr in pathpyg_export.data.keys() if attr.startswith("node_")
        }
        for id_column in ("actor_id", "issue_number", "commit_id", "file_id"):
            self.assertNotIn(f"node_{id_column}", node_attributes)

    def test_pyG_exporter_keeps_every_feature_on_its_own_node(self):
        # Recompute two features straight from the extractors and look each
        # value up in the export by node id, so that a row or column shift in
        # the exported tensor cannot go unnoticed.
        snaps = self.git2kg.get_snapshots()
        export = list(PyGExporter().export_snapshots(snaps).values())[0]
        oracles = {
            ("actor", "actor_id", "feature_NumCommitsActivity"): (
                features.NumCommitsActivityExtractor(self.git2kg.gitdata)
            ),
            ("file", "file_id", "feature_NumModifications"): (
                features.NumModificationsExtractor(self.git2kg.gitdata)
            ),
        }

        for (node_type, id_column, feature_name), extractor in oracles.items():
            matrix = snaps.features[0][node_type]
            # Minus one, as the id column is not exported
            column = list(matrix.columns).index(feature_name) - 1
            # Extractors return (id, value) positionally, under their own names
            expected = extractor.extract_feature()
            expected = expected.set_index(expected.columns[0]).iloc[:, 0]
            self.assertGreater(len(expected), 1)
            for node_id, value in expected.items():
                row = matrix.index[matrix[id_column] == node_id][0]
                self.assertEqual(value, export[node_type].x[row, column].item())

    def test_pyG_exporter_with_relation_exclusion(self):
        self.git2kg.freq = None
        snaps = self.git2kg.get_snapshots(
            edge_exclude=[
                "edit_add_edges",
                "assigned_to_edges",
                "issue_comment_edges",
                "review_request_edges",
                "create_issue_edges",
                "close_issue_edges",
                "merge_pull_edges",
            ]
        )
        exporter = PyGExporter()
        exports = exporter.export_snapshots(snapshots=snaps)
        exports = list(exports.values())
        edge_types = {
            ("actor", "edit_delete", "actor"),
            ("actor", "review", "actor"),
            ("actor", "assign_issue", "actor"),
            ("actor", "mentions", "actor"),
            ("actor", "comment_issue", "issue"),
            ("actor", "review_pull", "issue"),
            ("actor", "author_commit", "commit"),
            ("actor", "committer_commit", "commit"),
            ("issue", "referenced_on", "issue"),
            ("commit", "child_of", "commit"),
            ("commit", "modifies", "file"),
            ("file", "co_commit", "file"),
            ("actor", "reopen_issue", "issue"),
        }
        self.assertEqual(set(exports[0].edge_types), edge_types)

    def test_pyG_exporter_with_labels(self):
        self.git2kg.freq = "YE"
        snaps = self.git2kg.get_snapshots(
            actor_label_extractor=DeveloperLeaverLabelExtractor,
            issue_label_extractor=IssueClosedNextSliceLabelExtractor,
        )
        exporter = PyGExporter()
        exports = exporter.export_snapshots(snapshots=snaps)
        exports = list(exports.values())
        self.assertEqual(len(exports), 17)
        for export in exports:
            for index in export.edge_index_dict.values():
                self.assertEqual(index.dtype, torch.int64)
            self.assertEqual(len(export["actor"].y), 4)
            self.assertEqual(len(export["issue"].y), 2)
            self.assertEqual(len(export["actor"].x), 4)
            self.assertEqual(len(export["issue"].x), 2)

    def test_pyGExporter_empty_data(self):
        self.git2kg.freq = "YE"  # Some years in test data have no interaction

        exporter = PyGExporter()
        export = exporter.export_snapshots(self.git2kg.get_snapshots())
        empty_net = list(export.values())[3]
        # Assert all edge indices are emtpy
        for rel_name in [
            "edit_add",
            "edit_delete",
            "issue_comment",
            "review_request",
            "review",
        ]:
            self.assertEqual(
                (2, 0), empty_net[("actor", rel_name, "actor")].edge_index.shape
            )

    def test_pathpyGExporter(self):
        self.git2kg.freq = None
        exporter = PathPyGExporter()

        export = exporter.export_snapshots(
            self.git2kg.get_snapshots(
                actor_label_extractor=DeveloperLeaverLabelExtractor,
                issue_label_extractor=IssueClosedNextSliceLabelExtractor,
                edge_exclude=[],
            )
        )
        g: pp.Graph = list(export.values())[0]
        self.assertListEqual(
            [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3], g["node_type"].tolist()
        )
        self.assertEqual(15, g.n)
        self.assertEqual(60, g.m)
        self.assertListEqual(
            [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                4,
                5,
                6,
                6,
                6,
                7,
                7,
                8,
                8,
                8,
                9,
                9,
                10,
                11,
                12,
                13,
                13,
                13,
                13,
                13,
                13,
                14,
                14,
                14,
                14,
                14,
                14,
                15,
                16,
                16,
                17,
                17,
                17,
                17,
                17,
                17,
                18,
                18,
                18,
                18,
                18,
                18,
                18,
                19,
                19,
            ],
            g["edge_type"].tolist(),
        )

    def test_pathpyGExporter_only_actor_labels(self):
        self.git2kg.freq = None
        exporter = PathPyGExporter()
        export = exporter.export_snapshots(
            self.git2kg.get_snapshots(
                actor_label_extractor=DeveloperLeaverLabelExtractor
            )
        )
        g: pp.Graph = list(export.values())[0]
        self.assertEqual([0, 0], g["node_label"].tolist()[-2:])

    def test_networkXExporter(self):
        self.git2kg.freq = None
        exporter = NetworkXExporter()

        export = exporter.export_snapshots(
            self.git2kg.get_snapshots(
                actor_label_extractor=DeveloperLeaverLabelExtractor,
                issue_label_extractor=IssueClosedNextSliceLabelExtractor,
                edge_exclude=[],
            )
        )
        g: nx.MultiDiGraph = list(export.values())[0]
        self.assertEqual(15, g.number_of_nodes())
        self.assertEqual(60, g.number_of_edges())
        print(g)

    def test_networkXExporter_only_actor_labels(self):
        self.git2kg.freq = None
        exporter = NetworkXExporter()

        export = exporter.export_snapshots(
            self.git2kg.get_snapshots(
                actor_label_extractor=DeveloperLeaverLabelExtractor,
                edge_exclude=[],
            )
        )
        g: nx.MultiDiGraph = list(export.values())[0]
        self.assertEqual(15, g.number_of_nodes())
        self.assertEqual(60, g.number_of_edges())
        print(g)


class TestExportersAgreeOnFilteredEdges(unittest.TestCase):
    """Filtering gaps an edge frame's index, which is what misaligns the row
    lookup. Every exporter has to survive it, and to agree on the result.
    """

    def setUp(self):
        from git2kg.filtering import NodeFilter

        class ExcludeOneActor(NodeFilter):
            node_type = "actor"

            def excludes(self, entity):
                return entity.get_id() == 0

        gitdata = _load_gitdata("test/resources", "gitdata-test")
        self.git2kg = Git2KG(gitdata, freq=None, filters=[ExcludeOneActor()])

    def test_networkx_and_pyg_report_the_same_edges_per_relation(self):
        snapshots = self.git2kg.get_snapshots(edge_exclude=[])
        graph = list(NetworkXExporter().export_snapshots(snapshots).values())[0]
        hetero = list(PyGExporter().export_snapshots(snapshots).values())[0]

        in_networkx: dict[str, int] = {}
        for _, _, attributes in graph.edges(data=True):
            edge_type = attributes["type"]
            in_networkx[edge_type] = in_networkx.get(edge_type, 0) + 1

        for u_type, relation, v_type in hetero.edge_types:
            edge_type = f"{u_type}__{relation}__{v_type}"
            with self.subTest(edge_type):
                self.assertEqual(
                    hetero[u_type, relation, v_type]["edge_index"].shape[1],
                    in_networkx.get(edge_type, 0),
                )

    def test_the_filtered_actor_is_no_endpoint(self):
        snapshots = self.git2kg.get_snapshots(edge_exclude=[])
        edge_sets = snapshots.networks[0]["actor"]

        gapped = [
            edge_set
            for relation in edge_sets.values()
            for edge_set in relation.values()
            if len(edge_set) and list(edge_set.index) != list(range(len(edge_set)))
        ]
        self.assertTrue(gapped, "the fixture no longer produces a gapped edge frame")


class TestEdgesKeepTheirNodes(unittest.TestCase):
    """An edge must reach the nodes it names, whatever the edge frame's index.

    Filtering an edge set leaves its original row labels behind, and looking
    the nodes up with a merge answers with a fresh index. Pairing the two by
    label rather than by position silently points edges at the wrong nodes.
    """

    def setUp(self):
        import pandas as pd

        from git2kg.export import _row_of_node

        self.lookup = _row_of_node
        self.feature_matrix = pd.DataFrame(
            {"actor_id": [10, 11, 12, 13], "feature_x": [0.0, 1.0, 2.0, 3.0]}
        )
        self.pd = pd

    def test_a_gapped_index_still_finds_the_right_rows(self):
        edges = self.pd.DataFrame({"u": [13, 11, 10]}, index=[3, 4, 7])

        rows = self.lookup(edges, "u", self.feature_matrix, "actor_id")

        # actor 13 is row 3, actor 11 is row 1, actor 10 is row 0
        self.assertListEqual([3, 1, 0], list(rows))
        self.assertListEqual([3, 4, 7], list(rows.index))

    def test_a_clean_index_is_unaffected(self):
        edges = self.pd.DataFrame({"u": [12, 10]})

        self.assertListEqual(
            [2, 0], list(self.lookup(edges, "u", self.feature_matrix, "actor_id"))
        )

    def test_a_node_outside_the_matrix_is_reported_as_missing(self):
        edges = self.pd.DataFrame({"u": [10, 99]}, index=[5, 6])

        rows = self.lookup(edges, "u", self.feature_matrix, "actor_id")

        self.assertEqual(0, rows.iloc[0])
        self.assertTrue(self.pd.isna(rows.iloc[1]))

    def test_a_feature_matrix_with_duplicate_ids_is_refused(self):
        duplicated = self.pd.DataFrame({"actor_id": [10, 10], "feature_x": [0.0, 1.0]})
        edges = self.pd.DataFrame({"u": [10]})

        with self.assertRaises(ValueError):
            self.lookup(edges, "u", duplicated, "actor_id")
