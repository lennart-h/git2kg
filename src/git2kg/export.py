from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

from git2kg.network_snapshots import NetworkSnapshots

NODE_IDS = {
    "actor": "actor_id",
    "issue": "issue_number",
    "file": "file_id",
    "commit": "commit_id",
}


def _row_of_node(
    edges: pd.DataFrame,
    endpoint: str,
    feature_matrix: pd.DataFrame,
    id_column: str | None,
) -> pd.Series:
    """The row each edge endpoint has in a node type's feature matrix.

    Returned as a Series over the edge frame's own index, so that it can be
    assigned to it however that index is numbered. A node the matrix does not
    hold is answered with NA and its edge is dropped by the caller.
    """
    rows = edges.merge(
        feature_matrix.reset_index(names="idx"),
        left_on=endpoint,
        right_on=id_column,
        how="left",
    )
    if len(rows) != len(edges):
        raise ValueError(
            f"Looking up the {endpoint} of {len(edges)} edges answered with "
            f"{len(rows)} rows, so '{id_column}' is not unique in the feature "
            "matrix and the edges can no longer be matched to their nodes"
        )
    return pd.Series(rows["idx"].to_numpy(), index=edges.index)


class NetworkExporter(ABC):
    def export_snapshots(self, snapshots: NetworkSnapshots) -> dict:
        """_summary_

        Parameters
        ----------
        snapshots : NetworkSnapshots
            The snapshots to export as computed by Git2KG.get_snapshots()

        Returns
        -------
        dict
            A dictionary containing the snapshots as exported by the concrete implementation via do_export(). Keyed by the timestamp of the snapshot
        """
        res = {}
        for ts, snapshot in tqdm(
            snapshots.to_dict().items(), desc="Exporting Snapshots"
        ):
            res[ts] = self.do_export(
                snapshot["network"],
                snapshot["features"],
                snapshot["labels"],
                snapshot["active_masks"],
            )
        return res

    @abstractmethod
    def do_export(
        self,
        network: dict,
        feature_matrices: dict[str, pd.DataFrame],
        labels: dict,
        active_masks: dict,
    ):
        """Exports a single snapshot in the format specified by the concrete implementation

        Parameters
        ----------
        network : dict
            The dictionary of edges
        feature_matrices : dict[str, pd.DataFrame]
            The feature matrices for each node type
        labels : dict
            The labels for each node type
        active_masks : dict
            The active masks for each node type
        """
        pass


class PyGExporter(NetworkExporter):
    """Exporter implementation to export snapshots to `torch_geometric.data.HeteroData` objects.
    Features are stored in the `x` attribute of each node type and labels in the corresponding `y` attribute. Active masks are stored in the nodes `acitve_mask` attribute
    """

    def do_export(
        self, network: dict, feature_matrices: dict, labels: dict, active_masks
    ):
        import torch
        from torch_geometric.data import HeteroData  # type: ignore[import-untyped]

        data = HeteroData()
        for node_type, feature_matrix in feature_matrices.items():
            data[node_type].x = torch.tensor(
                _convert_feature_matrix(feature_matrix, NODE_IDS[node_type])
            )
        for node_type, labels_vec in labels.items():
            if len(labels_vec) > 0:
                data[node_type].y = torch.tensor(labels_vec)
        for node_type, active_mask in active_masks.items():
            data[node_type].active_mask = torch.tensor(active_mask, dtype=torch.bool)
        for _from, _tos in network.items():
            # Add indices of feature matrix to node
            for _to, edge_sets in _tos.items():
                for name, edge_set in edge_sets.items():
                    # Filtered edge sets carry the labels they had before
                    # the filtering, so the rows are renumbered before the
                    # row of each node is looked up: a merge answers with a
                    # fresh index, and assigning that to a gapped one aligns
                    # the two by label and silently pairs the wrong rows
                    id_edges = edge_set.reset_index(drop=True)

                    # Add index of feature map to edge dataframe, since pyG operated on these indices
                    id_edges["u_id"] = _row_of_node(
                        id_edges, "u", feature_matrices[_from], NODE_IDS.get(_from)
                    )
                    id_edges["v_id"] = _row_of_node(
                        id_edges, "v", feature_matrices[_to], NODE_IDS.get(_to)
                    )

                    # Filter Edges where no index is found (e.g. when issue of event has been deleted)
                    id_edges = id_edges[
                        pd.notna(id_edges["u_id"]) & pd.notna(id_edges["v_id"])
                    ]

                    # Create edge index and edge weight tensors
                    rel_name = name.replace("_edges", "")
                    edge_index = torch.tensor(
                        id_edges[["u_id", "v_id"]].T.to_numpy(dtype=np.int64)
                    )
                    edge_weight = torch.tensor(
                        id_edges["weight"].to_numpy(dtype=np.int64)
                    )
                    data[_from, rel_name, _to].edge_index = edge_index
                    data[_from, rel_name, _to].edge_weight = edge_weight
        return data


class PathPyGExporter(NetworkExporter):
    """Exporter implementation to export snapshots to `pathpyG.Graph` objects.
    Features, labels, active masks and node types are stored as node attributes, edge types and edge weights are stored as edge attributes
    """

    def do_export(self, network, feature_matrices, labels, active_masks):
        import pathpyG as ppG  # type: ignore[import-untyped]
        import torch

        pad = torch.nn.functional.pad
        edge_indices = torch.empty(2, 0, dtype=torch.int64)
        edge_weights = torch.empty(0, dtype=torch.int64)
        edge_type_dict = {}
        edge_types = torch.empty(0, dtype=torch.int64)
        node_type_dict = {}
        node_types = torch.empty(0, dtype=torch.int64)
        node_active_list: list[torch.Tensor] = []
        node_label_list: list[torch.Tensor] = []
        node_features_dict: dict[str, torch.Tensor] = {}
        node_type_offsets = {}
        mapping = []
        last_offset = 0
        for idx, (node_type, feature_matrix) in enumerate(feature_matrices.items()):
            node_type_offsets[node_type] = last_offset
            last_offset += len(feature_matrix)
            node_type_dict[idx] = node_type
            node_types = torch.cat(
                [node_types, torch.tensor(np.repeat(idx, len(feature_matrix)))]
            )
            ids = feature_matrix[NODE_IDS.get(node_type)].to_list()
            mapping.extend([f"{node_type}_{id}" for id in ids])
            # The id identifies the node and is not a feature of it
            feature_names = feature_matrix.columns.drop(NODE_IDS[node_type])
            # Append feature for nodes of node_type or create new left-padded feature tensor, if the feature was not present before
            for feature_name in feature_names:
                # Check if feature exists for other node types
                feature = node_features_dict.get(feature_name)
                feature_tensor = torch.tensor(
                    _convert_feature_series(feature_matrix[feature_name])
                )
                if feature:
                    node_features_dict[feature_name] = torch.stack(
                        [feature, feature_tensor]
                    )
                else:
                    node_features_dict[feature_name] = pad(
                        feature_tensor, (node_type_offsets[node_type], 0)
                    )
            # Pad all other features, which are not available for the node type
            for feature_name in set(node_features_dict.keys()) - set(feature_names):
                node_features_dict[feature_name] = pad(
                    node_features_dict[feature_name], (0, len(feature_matrix))
                )

        for node_type, _labels in labels.items():
            if len(_labels) == 0:
                _labels = np.repeat(0, len(feature_matrices[node_type]))
            node_label_list.extend(torch.tensor(_labels))
        for _, active_mask in active_masks.items():
            node_active_list.extend(torch.tensor(active_mask, dtype=torch.bool))

        node_label = torch.stack(node_label_list)
        node_active = torch.tensor(node_active_list)
        edge_type_idx = 0
        for _from, _tos in network.items():
            # Add indices of feature matrix to node
            for _to, edge_sets in _tos.items():
                for name, edge_set in edge_sets.items():
                    # See the note in the heterogeneous export: a filtered
                    # edge set keeps its original labels, which must not meet
                    # the fresh index a merge answers with
                    id_edges = edge_set.reset_index(drop=True)

                    # Add index of feature map to edge dataframe, since pyG operated on these indices
                    id_edges["u_id"] = (
                        _row_of_node(
                            id_edges, "u", feature_matrices[_from], NODE_IDS.get(_from)
                        )
                        + node_type_offsets[_from]
                    )
                    id_edges["v_id"] = (
                        _row_of_node(
                            id_edges, "v", feature_matrices[_to], NODE_IDS.get(_to)
                        )
                        + node_type_offsets[_to]
                    )

                    # Filter Edges where no index is found (e.g. when issue of event has been deleted)
                    id_edges = id_edges[
                        pd.notna(id_edges["u_id"]) & pd.notna(id_edges["v_id"])
                    ]

                    edge_index = torch.tensor(
                        id_edges[["u_id", "v_id"]].T.to_numpy(dtype=np.int64)
                    )
                    edge_indices = torch.cat([edge_indices, edge_index], dim=1)
                    edge_weight = torch.tensor(
                        id_edges["weight"].to_numpy(dtype=np.int64)
                    )
                    edge_weights = torch.cat([edge_weights, edge_weight])

                    # Create edge index and edge weight tensors
                    rel_name = name.replace("_edges", "")

                    edge_type_dict[edge_type_idx] = f"{_from}__{rel_name}__{_to}"
                    edge_types = torch.cat(
                        [
                            edge_types,
                            torch.tensor(
                                np.repeat(edge_type_idx, edge_index.size()[1])
                            ),
                        ]
                    )
                    edge_type_idx += 1

        g = ppG.Graph.from_edge_index(
            edge_index=edge_indices,
            mapping=ppG.IndexMap(mapping),
            num_nodes=last_offset,
        )

        # Node attributes
        for key, val in node_features_dict.items():
            g[f"node_{key}"] = val

        g["node_active"] = node_active
        g["node_label"] = node_label
        g["node_type"] = node_types

        # Edge attributes
        g["edge_type"] = edge_types
        g["edge_weight"] = edge_weights

        # Graph attributes
        g["graph_node_type_mapping"] = node_type_dict
        g["graph_edge_type_mapping"] = edge_type_dict
        return g


class NetworkXExporter(NetworkExporter):
    """Exporter implementation to export snapshots to `networkx.MultiDiGraph` objects.
    Features, labels, active masks and node types are stored as node attributes, edge types and edge weights are stored as edge attributes
    """

    def do_export(self, network, feature_matrices, labels, active_masks):
        import networkx as nx

        g = nx.MultiDiGraph()
        node_offset = 0
        node_type_offsets = {}
        for node_type, feature_matrix in feature_matrices.items():
            records = feature_matrix.to_dict("records")
            label = labels.get(node_type)
            active_mask = active_masks.get(node_type)
            record_tuples = []
            for idx, record in enumerate(records):
                record["type"] = node_type
                record["active"] = active_mask[idx]
                if len(label) == len(feature_matrix):
                    record["label"] = label[idx]

                record_tuples.append((idx + node_offset, record))
            g.add_nodes_from(record_tuples)
            node_type_offsets[node_type] = node_offset
            node_offset += len(feature_matrix)
        for _from, _tos in network.items():
            # Add indices of feature matrix to node
            for _to, edge_sets in _tos.items():
                for name, edge_set in edge_sets.items():
                    # Filtered edge sets carry the labels they had before
                    # the filtering, so the rows are renumbered before the
                    # row of each node is looked up: a merge answers with a
                    # fresh index, and assigning that to a gapped one aligns
                    # the two by label and silently pairs the wrong rows
                    id_edges = edge_set.reset_index(drop=True)
                    rel_name = name.replace("_edges", "")
                    edge_type = f"{_from}__{rel_name}__{_to}"

                    # Add index of feature map to edge dataframe, since pyG operated on these indices
                    id_edges["u_id"] = (
                        _row_of_node(
                            id_edges, "u", feature_matrices[_from], NODE_IDS.get(_from)
                        )
                        + node_type_offsets[_from]
                    )
                    id_edges["v_id"] = (
                        _row_of_node(
                            id_edges, "v", feature_matrices[_to], NODE_IDS.get(_to)
                        )
                        + node_type_offsets[_to]
                    )
                    # Filter Edges where no index is found (e.g. when issue of event has been deleted)
                    id_edges = id_edges[
                        pd.notna(id_edges["u_id"]) & pd.notna(id_edges["v_id"])
                    ]

                    ebunch = list(
                        id_edges[["u_id", "v_id", "weight"]].itertuples(
                            index=False, name=None
                        )
                    )
                    g.add_weighted_edges_from(ebunch_to_add=ebunch, type=edge_type)

        return g


def _convert_feature_matrix(feature_matrix: pd.DataFrame, id_column: str):
    """Converts a feature matrix to a float array, one row per node. The id
    column identifies the node and is not a feature of it, so it is left out.
    """
    feature_matrix = feature_matrix.drop(columns=id_column)
    datetime_cols = feature_matrix.select_dtypes(
        include=["datetime64[ns, UTC]"]
    ).columns
    for col in datetime_cols:
        feature_matrix[col] = (
            feature_matrix[col] - pd.Timestamp("1970-01-01T00:00:00.000Z")
        ) // pd.Timedelta("1s")  # type: ignore
    return np.nan_to_num(feature_matrix.to_numpy(dtype=np.float32))


def _convert_feature_series(feature_series: pd.Series):
    if feature_series.dtype == pd.DatetimeTZDtype("ns", "utc"):
        feature_series = (
            feature_series - pd.Timestamp("1970-01-01T00:00:00.000Z")
        ) // pd.Timedelta("1s")  # type: ignore
    return np.nan_to_num(feature_series.to_numpy(dtype=np.float32))
