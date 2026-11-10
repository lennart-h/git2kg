from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from git2kg.datapoint.utils import datapoint_list_to_dataframe
from git2kg.filtering import is_development_commit
from git2kg.gitdata import GitData


class LabelExtractor(ABC):
    """Base class for label extraction. Initialized with global GitData, time sliced GitData series and the corresponding time index.

    Every subclass is a prediction target describing what happens in the
    *next* timeframe, not the current one - see the README's "Temporal
    Semantics" section for why.
    """

    def __init__(
        self,
        global_data: GitData,
        timeslices: Iterable[GitData],
        timeindex: pd.DatetimeIndex,
    ):
        self.global_data = global_data
        self.timeslices = timeslices
        self.timeindex = timeindex

    @abstractmethod
    def extract_label_sequence(self) -> list[np.ndarray]:
        """Computes the desired label sequence on basis of the global data and timesliced data.
        Returns
        -------
        list
            A list of length len(self.timeslices) containing the label vectors
        """
        pass


class ActorLabelExtractor(LabelExtractor):
    """Specified subclass of LabelExtractor used for extraction of actor labels. Additionally to the initialization parameters of the suer class, as series of actor IDs is passed upon initialization"""

    def __init__(self, global_data: GitData, timeslices, timeindex, actors: pd.Series):
        super().__init__(global_data, timeslices, timeindex)
        self.actors = actors


class IssueLabelExtractor(LabelExtractor):
    """Specified subclass of LabelExtractor used for extraction of issue labels. Additionally to the initialization parameters of the suer class, as series of issue numbers is passed upon initialization"""

    def __init__(self, global_data: GitData, timeslices, timeindex, issues: pd.Series):
        super().__init__(global_data, timeslices, timeindex)
        self.issues = issues


class FileLabelExtractor(LabelExtractor):
    def __init__(self, global_data: GitData, timeslices, timeindex, files: pd.Series):
        super().__init__(global_data, timeslices, timeindex)
        self.files = files


class CommitLabelExtractor(LabelExtractor):
    def __init__(self, global_data: GitData, timeslices, timeindex, commits: pd.Series):
        super().__init__(global_data, timeslices, timeindex)
        self.commits = commits


class DeveloperLeaverLabelExtractor(ActorLabelExtractor):
    """
    Developer turnover prediction target, in the sense of Foucault et al.
    (2015) and Lin et al. (2017). A developer is labeled 1 for the timeslice
    in which they were last active before going away. Only commits count as
    developer activity, and merges are integration rather than development.

    Three conditions have to hold for the label of timeslice `i`::

        active in timeslice i                              the departure is
                                                           dated to the last
                                                           timeslice the
                                                           developer is seen in

        active in at least `min_active_slices` of the       someone passing
        `active_window` timeslices ending with i            through once is no
                                                           contributor leaving

        active in none of the `leave_window` timeslices     being away briefly
        that follow i                                       is not leaving

    Anchoring on the current timeslice matters: without it one departure is
    labeled once per timeslice the lookback still reaches it from, and those
    later labels sit on timeslices where the developer did nothing at all.

    By default each window spans one timeslice, which labels a developer that
    commits in one timeslice and not in the next. Since the extractor is
    constructed by `Git2KG.get_labels`, other windows are given by partially
    applying it::

        from functools import partial

        leaver = partial(
            DeveloperLeaverLabelExtractor,
            active_window=3,
            min_active_slices=2,
            leave_window=6,
        )
        snapshots = g2kg.get_snapshots(actor_label_extractor=leaver)

    The last `leave_window` timeslices have no full window to look ahead to
    and are labeled 0, which is not a statement about those developers. Pass
    `lookahead` to a splitter's `embargo` to leave them out.
    """

    def __init__(
        self,
        global_data: GitData,
        timeslices,
        timeindex,
        actors: pd.Series,
        active_window: int = 1,
        min_active_slices: int = 1,
        leave_window: int = 1,
    ):
        """
        Parameters
        ----------
        active_window : int, optional
            How many timeslices, ending with the current one, are looked at
            to decide whether a developer is a contributor, by default 1
        min_active_slices : int, optional
            In how many of those timeslices the developer has to have been
            active, by default 1. Raising it leaves out those who pass
            through a repository once
        leave_window : int, optional
            How many of the following timeslices a developer has to be absent
            from to count as having left, by default 1
        """
        super().__init__(global_data, timeslices, timeindex, actors)
        if active_window < 1 or leave_window < 1 or min_active_slices < 1:
            raise ValueError(
                f"The windows span timeslices, so all must be at least 1, got "
                f"active_window={active_window}, "
                f"min_active_slices={min_active_slices}, "
                f"leave_window={leave_window}"
            )
        if min_active_slices > active_window:
            raise ValueError(
                f"min_active_slices={min_active_slices} cannot be met inside an "
                f"active_window of {active_window} timeslices, so nothing would "
                f"ever be labeled"
            )
        self.active_window = active_window
        self.min_active_slices = min_active_slices
        self.leave_window = leave_window

    @property
    def lookahead(self) -> int:
        """How many timeslices past its own a label describes.

        A split has to leave this many timeslices between its parts, or the
        labels of one describe what happens in the next.
        """
        return self.leave_window

    def extract_label_sequence(self):
        developers = _developers_per_timeslice(self.global_data, self.timeindex)
        n_slices = len(developers)
        absent = np.zeros(len(self.actors), dtype=np.float32)

        seq: list[np.ndarray] = []
        for i in range(n_slices):
            if i + self.leave_window >= n_slices:
                # Not enough timeslices left to tell a leaver from someone
                # who is simply not active yet
                seq.append(absent.copy())
                continue
            lookback = developers[max(0, i - self.active_window + 1) : i + 1]
            contributors = {
                actor
                for actor in developers[i]
                if sum(actor in slice_ for slice_ in lookback) >= self.min_active_slices
            }
            following = set().union(*developers[i + 1 : i + 1 + self.leave_window])
            seq.append(
                self.actors.isin(contributors - following).to_numpy(dtype=np.float32)
            )
        return seq


class FileModifiedNextSliceLabelExtractor(FileLabelExtractor):
    """
    Change-proneness / hotspot prediction target: extracts labels indicating
    whether a file is modified (including creation, rename or deletion) in
    the *next* timeslice. The last timeslice has no next slice to look ahead
    to and is always labeled False, mirroring DeveloperLeaverLabelExtractor's
    convention for its final entry.
    """

    def extract_label_sequence(self):
        modified_files = datapoint_list_to_dataframe(self.global_data.modified_files)
        # file_id comes back as nullable Int64 from the dataframe conversion;
        # cast to plain int64 so it matches self.files for the .isin() below.
        modified_files["file_id"] = modified_files["file_id"].astype("int64")

        n_slices = len(self.timeindex) - 1
        seq = []
        for i in tqdm(
            range(n_slices), desc="Computing file modified-next-slice labels"
        ):
            if i + 1 >= n_slices:
                # No next slice to look ahead to
                seq.append(np.repeat(False, len(self.files)))
                continue
            next_lower, next_upper = self.timeindex[i + 1], self.timeindex[i + 2]
            touched = modified_files[
                (modified_files["timestamp"] >= next_lower)
                & (modified_files["timestamp"] < next_upper)
            ]["file_id"]
            seq.append(self.files.isin(touched).to_numpy())
        return seq


class IssueClosedNextSliceLabelExtractor(IssueLabelExtractor):
    """
    Prediction target: whether an issue is closed in the *next* timeslice.
    The last timeslice has no next slice to look ahead to and is always
    labeled False, mirroring DeveloperLeaverLabelExtractor's convention for
    its final entry.

    Deliberately not "closed in the current timeslice": that shape would be
    trivially reconstructable from the current snapshot's own graph, since
    a close_issue edge already exists for the same event in the same
    snapshot - not a genuine prediction target. Framing it one slice ahead
    keeps it consistent with the other predictive labels (e.g.
    DeveloperLeaverLabelExtractor, FileModifiedNextSliceLabelExtractor) and
    ensures the underlying close_issue edge only ever appears in a later
    snapshot than the one the label is attached to.
    """

    def extract_labels_for_datetime_slice(self, datetime_slice, close_events=None):
        """Labels every issue with whether a "closed" event falls into the
        given window.

        Parameters
        ----------
        datetime_slice : tuple[pd.Timestamp, pd.Timestamp]
            The lower (inclusive) and upper (exclusive) bound of the window
        close_events : pd.DataFrame, optional
            The "closed" events of the global data, as computed by
            `_close_events`. Computed when not given
        """
        if close_events is None:
            close_events = self._close_events()
        lower, upper = datetime_slice
        closed_in_slice = close_events[
            (close_events["created_at"] >= lower) & (close_events["created_at"] < upper)
        ]["issue_number"]
        return pd.DataFrame(
            {
                "issue_number": self.issues,
                "closed_time": self.issues.isin(closed_in_slice),
            }
        )

    def extract_label_sequence(self):
        close_events = self._close_events()
        n_slices = len(self.timeindex) - 1
        seq = []
        for i in tqdm(
            range(n_slices),
            desc="Computing issue closed next-slice labels",
        ):
            if i + 1 >= n_slices:
                seq.append(np.repeat(False, len(self.issues)).astype(np.float32))
                continue
            issue_closes = self.extract_labels_for_datetime_slice(
                (self.timeindex[i + 1], self.timeindex[i + 2]), close_events
            )
            seq.append(issue_closes["closed_time"].to_numpy(dtype=np.float32))
        return seq

    def _close_events(self) -> pd.DataFrame:
        events = datapoint_list_to_dataframe(self.global_data.events)
        return events[events["event_type"] == "closed"][["issue_number", "created_at"]]


def _developers_per_timeslice(
    gitdata: GitData, timeindex: pd.DatetimeIndex
) -> list[set[int]]:
    """Collects the ids of the actors authoring or committing code in each
    timeslice, in the order of the timeslices. Merging is integration rather
    than development, so merge commits do not make anyone a developer.

    Parameters
    ----------
    gitdata : GitData
        The full, un-sliced gitdata
    timeindex : pd.DatetimeIndex
        The timeslice boundaries
    """
    appeared_at = gitdata.commits.dp_type.appearance_key()
    commit_times = pd.DatetimeIndex([appeared_at(commit) for commit in gitdata.commits])
    timeslice_of = np.searchsorted(timeindex, commit_times, side="right") - 1

    developers: list[set[int]] = [set() for _ in range(len(timeindex) - 1)]
    for commit, timeslice in zip(gitdata.commits, timeslice_of, strict=True):
        if 0 <= timeslice < len(developers) and is_development_commit(commit):
            developers[timeslice].update((commit.author.id, commit.committer.id))
    return developers
