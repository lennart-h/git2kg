"""Existence masks marking which nodes are part of the graph in each snapshot.

Snapshots share one node set - the union over the whole mined history - so that
features, labels and masks can be aligned positionally and a model can carry
per-node state from one time step to the next. A snapshot therefore contains
rows for nodes that do not exist yet at that point in time, and the masks say
which rows are real.

"Real" here means existence, not activity: an actor who contributed in an early
snapshot and is dormant in a later one still exists in the later one. Dormancy
is signal a model should see rather than something a mask hides. Activity,
where it is wanted, belongs in a feature.

Only files are ever removed from a repository, so actor, issue and commit masks
are step functions: false until the node first appears, true from then on.
"""

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from git2kg.datapoint import ActorInfo, DatapointList
from git2kg.gitdata import FIRST_SEEN, GitData
from git2kg.utils import keep_earliest


def compute_active_masks(
    gitdata: GitData,
    timeindex: pd.DatetimeIndex,
    ids: dict[str, pd.Series],
) -> list[dict[str, np.ndarray]]:
    """Computes the existence mask of every node, for every timeslice.

    Parameters
    ----------
    gitdata : GitData
        The full, un-sliced gitdata. Existence is a property of the whole
        history, so the timeslices themselves are never needed
    timeindex : pd.DatetimeIndex
        The timeslice boundaries, as computed by `Git2KG.get_timeindex()`
    ids : dict[str, pd.Series]
        Maps node type to its id universe, in the same order as the rows of
        that node type's feature matrix - masks are aligned to features
        positionally, not by id

    Returns
    -------
    list[dict[str, np.ndarray]]
        One dict per timeslice, keyed by the same node types as `ids`, holding
        a boolean array over that node type's id universe
    """
    appearances = {
        "actor": _as_appearance_map(gitdata.actor_appearances()),
        "issue": _appearances(gitdata.issues, lambda dp: dp.issue.number),
        "commit": _appearances(gitdata.commits, lambda dp: dp.commit.id),
        "file": _appearances(gitdata.modified_files, lambda dp: dp.file.id),
    }
    removals = {"file": _file_removals(gitdata)}

    masks = {
        node_type: _existence_matrix(
            node_ids, appearances[node_type], removals.get(node_type), timeindex
        )
        for node_type, node_ids in ids.items()
    }
    return [
        {node_type: mask[i] for node_type, mask in masks.items()}
        for i in range(len(timeindex) - 1)
    ]


def _appearances(
    datapoints: DatapointList, id_of: Callable[[Any], Any]
) -> dict[Any, pd.Timestamp]:
    """Maps entity id -> the earliest time any of its datapoints appears.

    Parameters
    ----------
    datapoints : DatapointList
        The datapoints describing the entities
    id_of : Callable
        Extracts the id of the entity a datapoint belongs to
    """
    time_of = datapoints.dp_type.appearance_key()
    earliest: dict[Any, pd.Timestamp] = {}
    for datapoint in datapoints:
        keep_earliest(earliest, id_of(datapoint), time_of(datapoint))
    return earliest


def _as_appearance_map(appearances: pd.DataFrame) -> dict[Any, pd.Timestamp]:
    """Turns the actor appearance frame into the mapping the masks work from.

    Actors are the one node type whose appearances are not derived here: they
    show up across every datapoint list in any number of roles, and the same
    first appearance is what an actor's tenure feature is measured from, so
    `GitData` owns the scan and both read it.
    """
    return dict(
        zip(appearances[ActorInfo.ID_COLUMN], appearances[FIRST_SEEN], strict=True)
    )


def _file_removals(gitdata: GitData) -> dict[Any, pd.Timestamp]:
    """Maps file id -> the time the commit that deleted it entered history.

    Files missing from the result were never deleted and live on until the end
    of the mined history.
    """
    time_of = gitdata.modified_files.dp_type.appearance_key()
    commit_times: dict[str, pd.Timestamp] = {}
    for datapoint in gitdata.modified_files:
        commit_times.setdefault(datapoint.commit.hash, time_of(datapoint))
    return {
        file_id: commit_times[commit_hash]
        for file_id, commit_hash in gitdata.file_registry.deleted_at.items()
        if commit_hash in commit_times
    }


def _existence_matrix(
    ids: pd.Series,
    appearances: dict[Any, pd.Timestamp],
    removals: Optional[dict[Any, pd.Timestamp]],
    timeindex: pd.DatetimeIndex,
) -> np.ndarray:
    """Computes the (n_timeslices, n_ids) matrix of node existence.

    A node is part of every timeslice from the one containing its first
    appearance through the one containing its removal, both inclusive. Ids
    without a known appearance never exist.

    Parameters
    ----------
    ids : pd.Series
        The id universe of one node type, defining the column order
    appearances : dict
        Maps id -> time of first appearance
    removals : dict, optional
        Maps id -> time of removal, for the ids that were removed at all. `None`
        for node types that are never removed
    timeindex : pd.DatetimeIndex
        The timeslice boundaries
    """
    lower = timeindex[:-1].to_numpy()[:, None]
    upper = timeindex[1:].to_numpy()[:, None]

    appeared = _timestamps_for(ids, appearances)
    exists = appeared[None, :] < upper

    if removals:
        removed = _timestamps_for(ids, removals)
        never_removed = pd.isna(removed)[None, :]
        outlived_slice_start = removed[None, :] >= lower
        exists &= never_removed | outlived_slice_start
    return exists


def _timestamps_for(ids: pd.Series, lookup: dict[Any, pd.Timestamp]) -> np.ndarray:
    """Looks every id up in `lookup`, as a UTC datetime64 array with NaT for misses."""
    return pd.to_datetime(ids.map(lookup), utc=True).to_numpy()
