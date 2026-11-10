"""Time-respecting splitting of `NetworkSnapshots` into train/validation/test sets.

Splits are always contiguous along the time axis - every timestamp in the
training set is strictly earlier than every timestamp in the validation set,
which in turn is strictly earlier than every timestamp in the test set. Whole
snapshots are assigned to a split; a snapshot is never divided, so quadruples
sharing a timestamp never end up in different sets.

Where the strategies differ is *where the cut points land*. Splitting by
snapshot count implicitly assumes every snapshot carries a comparable number of
quadruples. Software engineering activity is bursty, so it does not: a single
snapshot containing a release or an incident can hold a large fraction of a
repository's whole history. `VolumeBalancedSplitter` therefore places the cuts
by cumulative quadruple volume instead, which keeps the *data* ratio close to
the target while letting the number of snapshots per split vary.
"""

from abc import ABC, abstractmethod

from git2kg.network_snapshots import NetworkSnapshots


class QuadrupleCounter(ABC):
    """Counts the quadruples (subject, relation, object, timestamp) held by one
    snapshot's network, i.e. one entry of `NetworkSnapshots.networks`.
    """

    @abstractmethod
    def count(self, network: dict) -> int:
        """Computes the quadruple count of a single snapshot's network

        Parameters
        ----------
        network : dict
            One snapshot's edges, of structure u_type -> v_type -> relation -> edge_set

        Returns
        -------
        int
            The number of quadruples the snapshot contains
        """
        pass


class UnweightedQuadrupleCounter(QuadrupleCounter):
    """Counts every distinct (subject, relation, object, timestamp) once,
    regardless of how often the interaction occurred within the snapshot.

    `_collapse_edges` already aggregates repeated interactions between the same
    pair into a single row per relation, so one row is one quadruple.
    """

    def count(self, network: dict) -> int:
        return sum(
            len(edge_set)
            for _tos in network.values()
            for rels in _tos.values()
            for edge_set in rels.values()
        )


class WeightedQuadrupleCounter(QuadrupleCounter):
    """Counts every individual interaction, i.e. weighs each (subject, relation,
    object, timestamp) by how often it occurred within the snapshot.

    This reflects raw event volume rather than distinct relations, and is the
    closer match to how bursty activity actually accumulates.
    """

    def count(self, network: dict) -> int:
        return sum(
            int(edge_set["weight"].sum())
            for _tos in network.values()
            for rels in _tos.values()
            for edge_set in rels.values()
        )


class SnapshotSplitter(ABC):
    """Base class for splitting a `NetworkSnapshots` sequence into a training,
    validation and test set.

    Subclasses only decide *where* the two cut points fall by implementing
    `_cut_indices`. Validation of the resulting splits, the optional embargo and
    the actual slicing are handled here, so every strategy behaves identically
    in those respects.
    """

    def __init__(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        embargo: bool | int = False,
        allow_empty_splits: bool = False,
    ):
        """
        Parameters
        ----------
        train_size : float, optional
            Target share of the training set, by default 0.8
        val_size : float, optional
            Target share of the validation set, by default 0.1. The test set
            receives the remainder, i.e. 1 - train_size - val_size
        embargo : bool or int, optional
            How many snapshots to drop from the end of the training and of
            the validation set, by default none. Labels describe what happens
            *after* the snapshot they belong to (see the README's "Temporal
            Semantics" section), so the final snapshots of a split carry
            labels about the split that follows. True means one snapshot,
            which is what a label reaching one timeframe ahead needs; a label
            reaching further needs as many, and reports that as its
            extractor's `lookahead`
        allow_empty_splits : bool, optional
            If True, permit splits that contain no snapshots instead of raising,
            by default False

        Raises
        ------
        ValueError
            If the sizes are not positive or do not leave room for a test set
        """
        if train_size <= 0 or val_size <= 0:
            raise ValueError("train_size and val_size must be positive")
        if train_size + val_size >= 1:
            raise ValueError(
                f"train_size + val_size must be < 1 to leave room for a test set, "
                f"got {train_size} + {val_size} = {train_size + val_size}"
            )
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - train_size - val_size
        if embargo is not True and embargo is not False and embargo < 0:
            raise ValueError(f"embargo cannot be negative, got {embargo}")
        # True is the one snapshot a label reaching one timeframe ahead needs
        self.embargo = int(embargo)
        self.allow_empty_splits = allow_empty_splits

    @abstractmethod
    def _cut_indices(self, snapshots: NetworkSnapshots) -> tuple[int, int]:
        """Determines where to cut the snapshot sequence.

        Parameters
        ----------
        snapshots : NetworkSnapshots
            The snapshots to split

        Returns
        -------
        tuple[int, int]
            The exclusive end index of the training set and of the validation
            set, i.e. train is [:train_end], validation is [train_end:val_end]
            and test is [val_end:]
        """
        pass

    def split(
        self, snapshots: NetworkSnapshots
    ) -> tuple[NetworkSnapshots, NetworkSnapshots, NetworkSnapshots]:
        """Splits the snapshots into a training, validation and test set

        Parameters
        ----------
        snapshots : NetworkSnapshots
            The snapshots to split, as computed by `Git2KG.get_snapshots()`

        Returns
        -------
        tuple[NetworkSnapshots, NetworkSnapshots, NetworkSnapshots]
            The training, validation and test set, in temporal order

        Raises
        ------
        ValueError
            If any of the resulting splits is empty and `allow_empty_splits`
            is False
        """
        train_end, val_end = self._cut_indices(snapshots)
        n_slices = len(snapshots.time_index)

        train_stop = max(train_end - self.embargo, 0)
        val_stop = max(val_end - self.embargo, train_end)

        splits = {
            "training": _slice(snapshots, 0, train_stop),
            "validation": _slice(snapshots, train_end, val_stop),
            "test": _slice(snapshots, val_end, n_slices),
        }
        self._verify_non_empty(splits)
        return splits["training"], splits["validation"], splits["test"]

    def _verify_non_empty(self, splits: dict[str, NetworkSnapshots]) -> None:
        empty = [name for name, split in splits.items() if not len(split.time_index)]
        if not empty or self.allow_empty_splits:
            return
        raise ValueError(
            f"The {', '.join(empty)} set would contain no snapshots. This usually "
            "means single snapshots hold a large share of all quadruples, which "
            "happens when the snapshot frequency is coarse relative to how bursty "
            "the repository's activity is. Try a finer `freq` on the Git2KG object "
            "for a more granular timeline, or pass allow_empty_splits=True to "
            "split anyway."
        )


class VolumeBalancedSplitter(SnapshotSplitter):
    """Places the cut points by cumulative quadruple volume, so that each split
    holds roughly its target *share of the data* rather than its target share of
    the snapshots.

    Both cuts are found by the same greedy walk: starting at the latest snapshot
    and moving backwards in time, snapshots are added to the later split until
    it holds at least the targeted share of quadruples. The first walk runs over
    the whole sequence and separates the training set from the combined
    validation and test set; the second repeats it within that combined set to
    separate validation from test.

    Because whole snapshots are assigned, the resulting share overshoots the
    target whenever the snapshot that crosses the threshold is a large one -
    which is exactly the bursty case this is meant to handle.
    """

    def __init__(
        self,
        counter: QuadrupleCounter | None = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
        embargo: bool | int = False,
        allow_empty_splits: bool = False,
    ):
        """
        Parameters
        ----------
        counter : QuadrupleCounter, optional
            How to count the quadruples of a snapshot, by default
            `UnweightedQuadrupleCounter()`
        """
        super().__init__(train_size, val_size, embargo, allow_empty_splits)
        self.counter = counter if counter is not None else UnweightedQuadrupleCounter()

    def _cut_indices(self, snapshots: NetworkSnapshots) -> tuple[int, int]:
        counts = [self.counter.count(network) for network in snapshots.networks]
        total = sum(counts)

        # First walk: separate training from the combined validation and test set
        train_end = _backward_cut(counts, (self.val_size + self.test_size) * total)

        # Second walk: separate validation from test within that combined set
        pool = counts[train_end:]
        test_share_of_pool = self.test_size / (self.val_size + self.test_size)
        val_end = train_end + _backward_cut(pool, test_share_of_pool * sum(pool))
        return train_end, val_end


class SnapshotCountSplitter(SnapshotSplitter):
    """Places the cut points by snapshot count, i.e. the first `train_size` of
    all snapshots form the training set, and so on.

    This is the plain time-based split, and a reasonable choice wherever
    activity is spread evenly over the timeline. It is included as a baseline:
    on bursty data it produces splits whose share of the actual quadruples can
    differ substantially from the targeted ratio, which is what
    `VolumeBalancedSplitter` addresses.
    """

    def _cut_indices(self, snapshots: NetworkSnapshots) -> tuple[int, int]:
        n_slices = len(snapshots.time_index)
        train_end = int(n_slices * self.train_size)
        val_end = train_end + int(n_slices * self.val_size)
        return train_end, val_end


def _backward_cut(counts: list[int], target: float) -> int:
    """Walks a sequence of per-snapshot quadruple counts from the latest to the
    earliest entry and reports where the later split starts.

    Parameters
    ----------
    counts : list[int]
        The quadruple count of every snapshot, in temporal order
    target : float
        The number of quadruples the later split must hold at least

    Returns
    -------
    int
        The index at which the later split starts, i.e. counts[index:] is the
        first tail of the sequence holding at least `target` quadruples
    """
    cumulative = 0
    for i in range(len(counts) - 1, -1, -1):
        cumulative += counts[i]
        if cumulative >= target:
            return i
    return 0


def _slice(snapshots: NetworkSnapshots, start: int, stop: int) -> NetworkSnapshots:
    """Builds a new `NetworkSnapshots` from a contiguous range of another one"""
    return NetworkSnapshots(
        snapshots.time_index[start:stop],
        snapshots.networks[start:stop],
        snapshots.features[start:stop],
        snapshots.active_masks[start:stop],
        snapshots.labels[start:stop],
    )
