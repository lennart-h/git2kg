"""Time-windowing of GitData into per-timeslice views."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from git2kg.datapoint import DatapointList
from git2kg.gitdata import GitData


class TimesliceIterator(Sequence):
    """The timeslices of a GitData, as a sequence of GitData objects.

    The data is partitioned once, on first access, rather than filtered per
    timeslice: filtering reads every datapoint of the whole history, so doing
    it per slice costs the history once per slice. Every caller reads the
    whole sequence anyway.
    """

    def __init__(self, gitdata, timeindex):
        self.gitdata = gitdata
        self.timeindex = timeindex
        self._timeslices: list[GitData] | None = None

    def _partitioned(self) -> list[GitData]:
        if self._timeslices is None:
            self._timeslices = _partition(self.gitdata, self.timeindex)
        return self._timeslices

    def __iter__(self):
        return iter(self._partitioned())

    def __getitem__(self, index):
        return self._partitioned()[index]

    def __len__(self):
        return len(self.timeindex) - 1


def _slice_of(dp_list: DatapointList, timeindex: pd.DatetimeIndex) -> np.ndarray:
    """The timeslice each datapoint of a list belongs to.

    A datapoint belongs to the slice whose lower bound is the last one at or
    before it, which is what `searchsorted` returns. Datapoints outside the
    index are given an index of their own and dropped by the caller.
    """
    time_of = dp_list.dp_type.appearance_key()
    appeared_at = pd.to_datetime(
        pd.Series([time_of(datapoint) for datapoint in dp_list]), utc=True
    ).to_numpy()
    bounds = pd.DatetimeIndex(timeindex).tz_convert("UTC").to_numpy()
    return np.searchsorted(bounds, appeared_at, side="right") - 1


def _partition(gitdata: GitData, timeindex: pd.DatetimeIndex) -> list[GitData]:
    """Splits a GitData into one GitData per timeslice.

    The slices share their datapoint objects and their file registry with the
    data they are taken from, so anything stamping ids into a slice stamps
    them into the whole history.

    Parameters
    ----------
    gitdata : GitData
        The data to split
    timeindex : pd.DatetimeIndex
        The timeslice boundaries, each slice reaching from its own bound
        (inclusive) to the next one (exclusive)

    Returns
    -------
    list[GitData]
        One GitData per timeslice, in the order of the boundaries
    """
    n_slices = len(timeindex) - 1
    buckets: dict[str, list[list]] = {
        dp_name: [[] for _ in range(n_slices)] for dp_name in gitdata.dp_lists
    }
    for dp_name, dp_list in gitdata.dp_lists.items():
        if len(dp_list) == 0:
            continue
        of_slice = buckets[dp_name]
        for datapoint, index in zip(
            dp_list.datapoints, _slice_of(dp_list, timeindex), strict=True
        ):
            if 0 <= index < n_slices:
                of_slice[index].append(datapoint)

    timeslices = []
    for i in range(n_slices):
        init_params: dict[str, Any] = {
            dp_name: DatapointList(buckets[dp_name][i], dp_list.dp_type)
            for dp_name, dp_list in gitdata.dp_lists.items()
        }
        init_params["actors_disambiguated"] = gitdata.actors_disambiguated
        init_params["files_disambiguated"] = gitdata.files_disambiguated
        init_params["commits_disambiguated"] = gitdata.commits_disambiguated
        # The registry describes the whole repository rather than a window of
        # it, and it is what the commit and file id universes are read from,
        # so a slice without it has no commit and no file nodes at all
        init_params["file_registry"] = gitdata.file_registry
        timeslices.append(GitData(**init_params))
    return timeslices
