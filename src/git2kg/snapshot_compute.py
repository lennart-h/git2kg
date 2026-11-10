"""Per-timeslice computation of feature matrices and edges.

These are module-level functions rather than Git2KG methods so they can be
dispatched to a multiprocessing.Pool without pickling the whole Git2KG
instance (and its potentially large gitdata) per task - each only needs the
small, precomputed values passed to it explicitly.
"""

import logging
from datetime import datetime

import pandas as pd

from git2kg.datapoint import ActorInfo, NestedNodeInfo
from git2kg.features import (
    ActorFeatureExtractor,
    CommitFeatureExtractor,
    DatapointTables,
    FeatureExtractor,
    FileFeatureExtractor,
    IssueFeatureExtractor,
)
from git2kg.gitdata import GitData

LOG = logging.getLogger(__name__)


def _compile_feature_matrices_for_timeslice(
    snapshot: tuple[DatapointTables, pd.Timestamp | None], ids: dict[str, pd.Series]
) -> dict[str, pd.DataFrame]:
    """Compiles one feature matrix per node type for a snapshot.

    Parameters
    ----------
    snapshot : tuple[DatapointTables, pd.Timestamp or None]
        The data the features are computed from and the point in time they
        describe. Passed as one argument so that the function can be mapped
        over a multiprocessing.Pool. `None` as the point in time means the
        data is not restricted, so the features describe it as of the end of
        mining
    ids : dict[str, pd.Series]
        Maps node type to its id universe, as computed by `Git2KG._id_universes`.
        The rows of each feature matrix follow the order of these ids
    """
    data, as_of = snapshot
    extractors = {
        "actor": ActorFeatureExtractor,
        "issue": IssueFeatureExtractor,
        "commit": CommitFeatureExtractor,
        "file": FileFeatureExtractor,
    }
    id_columns = {
        "actor": ActorInfo.ID_COLUMN,
        "issue": "issue_number",
        "commit": "commit_id",
        "file": "file_id",
    }
    return {
        # The bases are abstract on purpose: only their subclasses are run
        node_type: _compile_features(
            extractors[node_type],  # type: ignore[type-abstract]
            data,
            node_ids,
            id_columns[node_type],
            as_of,
        )
        for node_type, node_ids in ids.items()
    }


def _compile_features(
    extractor_base: type[FeatureExtractor],
    data: DatapointTables,
    ids: pd.Series,
    id_column: str,
    as_of: pd.Timestamp | None,
) -> pd.DataFrame:
    """Runs every extractor derived from `extractor_base` on the data and
    merges the results into one feature matrix, one row per id.

    Parameters
    ----------
    extractor_base : type[FeatureExtractor]
        The node type's extractor base class, whose subclasses are run
    data : DatapointTables
        The data the features are computed from
    ids : pd.Series
        The ids (from the full gitdata) seeding the rows of the feature matrix
    id_column : str
        The name of the id column, both in the seeded matrix and in what the
        extractors return
    as_of : pd.Timestamp, optional
        The point in time the features describe, passed on to every extractor

    Returns
    -------
    pd.DataFrame
        The feature matrix, one `feature_<Name>` column per extractor. Ids
        without a value for a feature hold NaN
    """
    feature_map = ids.rename(id_column).to_frame().reset_index(drop=True)
    for extractor in extractor_base.__subclasses__():
        feature_name = extractor.__name__.replace("Extractor", "")
        feature = extractor(data, as_of).extract_feature().copy()  # type: ignore[abstract]
        feature.columns = pd.Index([id_column, f"feature_{feature_name}"])
        feature[id_column] = feature[id_column].astype("int64")
        duplicated = int(feature[id_column].duplicated().sum())
        if duplicated:
            raise ValueError(
                f"{extractor.__name__} returned {duplicated} further rows for "
                f"ids it had already described. A feature is one value per "
                f"node, and merging several would add rows to the matrix, "
                f"which is aligned to the labels and masks by position"
            )
        feature_map = pd.merge(feature_map, feature, on=id_column, how="left")
    return feature_map


def _get_edges_for_timeslice(
    timeslice: GitData,
    actor_mapping: pd.DataFrame,
    exclude: list[str] | None = None,
    excluded_ids: dict[str, set] | None = None,
    edge_fn=None,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Computes the edges for each relation for a timeslice.

    Parameters
    ----------
    timeslice : GitData
        The timeslice of data the edges are computed for. Since a "timeslice"
        is just a GitData object, the full, un-sliced gitdata can be passed
        here too, e.g. to compute a continuous-time edge stream over the
        whole mined history
    actor_mapping : pd.DataFrame
        The actor mapping used to resolve actor ids
    exclude : list[str], optional
        Relation names to leave out entirely
    excluded_ids : dict[str, set], optional
        Node ids to leave out, per node type, as computed by
        `filtering.excluded_ids`. Every edge touching one of them is dropped
    edge_fn : callable, optional
        Turns a relation's raw list of (u, v, timestamp) tuples into the
        per-relation edge_set DataFrame. Defaults to `_collapse_edges`, which
        aggregates same-pair interactions into a single weighted edge; pass
        `_stack_edges` instead to keep every interaction as its own
        timestamped row

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing the edges of of structure u_type -> v_type -> relation -> edge_set
    """
    if exclude is None:
        exclude = []
    if excluded_ids is None:
        excluded_ids = {}
    if edge_fn is None:
        edge_fn = _collapse_edges
    edit_add_edges = edge_fn(timeslice.edit_add_edges())
    edit_delete_edges = edge_fn(timeslice.edit_delete_edges())
    issue_comment_edges = edge_fn(timeslice.issue_comment_edges())
    review_request_edges = edge_fn(timeslice.review_request_edges())
    review_edges = edge_fn(timeslice.review_edges())
    assign_user_edges = edge_fn(timeslice.issue_assigned_from_user_edges())
    mentions_actor_edge_tuples = timeslice.mentions_actor_edges()

    mentions_actor_edges = edge_fn(mentions_actor_edge_tuples)
    assigned_to_edges = edge_fn(timeslice.issue_assigned_to_user_edges())
    create_issue_edges = edge_fn(timeslice.create_issue_edges())
    comment_issue_edges = edge_fn(timeslice.comment_issue_edges())
    close_issue_edges = edge_fn(timeslice.issue_closed_edges())
    reopen_issue_edges = edge_fn(timeslice.issue_reopened_edges())
    review_pull_edges = edge_fn(timeslice.review_pull_edges())
    merge_issue_edges = edge_fn(timeslice.issue_merged_edges())

    issue_referenced_on_edges = edge_fn(timeslice.issue_referenced_on_edges())

    commit_parent_edges = edge_fn(timeslice.commit_parent_edges())
    commit_modifies_edges = edge_fn(timeslice.commit_modifies_file_edges())
    co_commit_edges = edge_fn(timeslice.file_co_commit_edges())

    author_commit_edges = edge_fn(timeslice.author_commit_edges())
    committer_commit_edges = edge_fn(timeslice.committer_edges())

    edge_dict = {
        "actor": {
            "actor": {
                "edit_add_edges": edit_add_edges,
                "edit_delete_edges": edit_delete_edges,
                "issue_comment_edges": issue_comment_edges,
                "review_request_edges": review_request_edges,
                "review_edges": review_edges,
                "assign_issue_edges": assign_user_edges,
                "mentions_edges": mentions_actor_edges,
            },
            "issue": {
                "create_issue_edges": create_issue_edges,
                "comment_issue_edges": comment_issue_edges,
                "close_issue_edges": close_issue_edges,
                "reopen_issue_edges": reopen_issue_edges,
                "review_pull_edges": review_pull_edges,
                "merge_pull_edges": merge_issue_edges,
            },
            "commit": {
                "author_commit_edges": author_commit_edges,
                "committer_commit_edges": committer_commit_edges,
            },
        },
        "issue": {
            "actor": {"assigned_to_edges": assigned_to_edges},
            "issue": {"referenced_on_edges": issue_referenced_on_edges},
        },
        "commit": {
            "commit": {"child_of_edges": commit_parent_edges},
            "file": {"modifies_edges": commit_modifies_edges},
        },
        "file": {"file": {"co_commit_edges": co_commit_edges}},
    }
    # Remove edges that are not wanted
    for u_type, nets in edge_dict.items():
        for v_type, rels in nets.items():
            for excluded_rel in exclude:
                rels.pop(excluded_rel, None)
            for relation, edges in rels.items():
                rels[relation] = edges[
                    ~edges["u"].isin(excluded_ids.get(u_type, ()))
                    & ~edges["v"].isin(excluded_ids.get(v_type, ()))
                ]
    return edge_dict


def _connected(
    edges: list[tuple[NestedNodeInfo, NestedNodeInfo, datetime]],
) -> list[tuple]:
    """The endpoint ids of the edges whose nodes both exist.

    The actor of a deleted account is recorded as missing, so an interaction
    can name an endpoint that is no node. An edge needs two, so those are left
    out, as the edges of a filtered node are.
    """
    connected = [
        (u.get_id(), v.get_id(), pd.Timestamp(timestamp))
        for u, v, timestamp in edges
        if u is not None and v is not None
    ]
    if len(connected) != len(edges):
        LOG.debug(
            "Left out %s of %s edges with a missing endpoint",
            len(edges) - len(connected),
            len(edges),
        )
    return connected


def _collapse_edges(
    edges: list[tuple[NestedNodeInfo, NestedNodeInfo, datetime]],
) -> pd.DataFrame:
    """Aggregates multi edges to single edge, weighted by number of occurence

    Parameters
    ----------
    edges : list[tuple[int, int, pd.Timestamp]]
        The edge set to be aggregated

    Returns
    -------
    pd.DataFrame
        Dataframe with columns \"u\", \"v\" and weight, where each row represent one edge from $u$ to $v$ with weight $weight$
    """
    df = pd.DataFrame.from_records(_connected(edges), columns=["u", "v", "ts"])
    return df.groupby(["u", "v"]).size().reset_index(name="weight")


def _stack_edges(
    edges: list[tuple[NestedNodeInfo, NestedNodeInfo, datetime]],
) -> pd.DataFrame:
    """Keeps every interaction as its own row, instead of aggregating same-pair
    interactions into a single weighted edge like `_collapse_edges` does.

    Parameters
    ----------
    edges : list[tuple[int, int, pd.Timestamp]]
        The edge set to be converted

    Returns
    -------
    pd.DataFrame
        Dataframe with columns \"u\", \"v\" and \"ts\", one row per interaction,
        sorted by timestamp
    """
    df = pd.DataFrame.from_records(_connected(edges), columns=["u", "v", "ts"])
    return df.sort_values("ts", ignore_index=True)


def _get_continuous_edges(
    gitdata: GitData,
    actor_mapping: pd.DataFrame,
    exclude: list[str] | None = None,
    excluded_ids: dict[str, set] | None = None,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Computes the continuous-time edges (one row per interaction, exact
    timestamps preserved) for the full, un-sliced gitdata.

    Parameters
    ----------
    gitdata : GitData
        The (typically full, un-sliced) gitdata the edges are computed for
    actor_mapping : pd.DataFrame
        The actor mapping used to resolve actor ids

    Returns
    -------
    dict[str, dict[str, dict[str, pd.DataFrame]]]
        Dictionary of structure u_type -> v_type -> relation -> edge_set, where
        each edge_set has columns \"u\", \"v\" and \"ts\"
    """
    if exclude is None:
        exclude = []
    return _get_edges_for_timeslice(
        gitdata, actor_mapping, exclude, excluded_ids, edge_fn=_stack_edges
    )
