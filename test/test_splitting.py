import unittest

import pandas as pd

from git2kg.network_snapshots import NetworkSnapshots
from git2kg.splitting import (
    SnapshotCountSplitter,
    UnweightedQuadrupleCounter,
    VolumeBalancedSplitter,
    WeightedQuadrupleCounter,
)


def _network(n_edges: int, weight: int = 1) -> dict:
    """Builds one snapshot's network holding `n_edges` distinct edges"""
    return {
        "actor": {
            "actor": {
                "edit_add_edges": pd.DataFrame(
                    {
                        "u": range(n_edges),
                        "v": range(n_edges),
                        "weight": [weight] * n_edges,
                    }
                )
            }
        }
    }


def _snapshots(edge_counts: list[int], weight: int | list[int] = 1) -> NetworkSnapshots:
    """Builds a NetworkSnapshots whose i-th snapshot holds edge_counts[i] edges.

    `weight` is either one weight for every snapshot, or one weight per snapshot.
    """
    n_slices = len(edge_counts)
    weights = weight if isinstance(weight, list) else [weight] * n_slices
    return NetworkSnapshots(
        pd.date_range("2024-01-01", periods=n_slices, freq="ME"),
        [_network(count, w) for count, w in zip(edge_counts, weights, strict=True)],
        [{} for _ in range(n_slices)],
        [{} for _ in range(n_slices)],
        [{} for _ in range(n_slices)],
    )


def _quadruples(snapshots: NetworkSnapshots) -> int:
    return sum(UnweightedQuadrupleCounter().count(net) for net in snapshots.networks)


class TestQuadrupleCounters(unittest.TestCase):
    def test_unweighted_counts_distinct_edges(self):
        network = _network(5, weight=3)
        self.assertEqual(5, UnweightedQuadrupleCounter().count(network))

    def test_weighted_counts_individual_interactions(self):
        network = _network(5, weight=3)
        self.assertEqual(15, WeightedQuadrupleCounter().count(network))

    def test_counters_handle_empty_relations(self):
        network = _network(0)
        self.assertEqual(0, UnweightedQuadrupleCounter().count(network))
        self.assertEqual(0, WeightedQuadrupleCounter().count(network))


class TestVolumeBalancedSplitter(unittest.TestCase):
    def test_evenly_sized_snapshots_split_cleanly(self):
        # 20 snapshots of 10 quadruples each. No burstiness, so a volume based
        # split lands on the same cuts a count based one would.
        snapshots = _snapshots([10] * 20)
        train, val, test = VolumeBalancedSplitter().split(snapshots)

        self.assertEqual(
            (16, 2, 2),
            (len(train.time_index), len(val.time_index), len(test.time_index)),
        )
        self.assertEqual(
            (160, 20, 20), (_quadruples(train), _quadruples(val), _quadruples(test))
        )

    def test_burst_shifts_cuts_to_keep_volume_balanced(self):
        # One early snapshot holds 100 quadruples, the remaining 19 hold 10 each.
        # A count based split would hand the training set 86% of all quadruples;
        # placing the cuts by volume keeps the ratios close to 80/10/10.
        snapshots = _snapshots([100] + [10] * 19)
        train, val, test = VolumeBalancedSplitter().split(snapshots)

        self.assertEqual(
            (14, 3, 3),
            (len(train.time_index), len(val.time_index), len(test.time_index)),
        )
        self.assertEqual(
            (230, 30, 30), (_quadruples(train), _quadruples(val), _quadruples(test))
        )

        total = _quadruples(snapshots)
        self.assertAlmostEqual(0.8, _quadruples(train) / total, places=1)
        self.assertAlmostEqual(0.1, _quadruples(val) / total, places=1)
        self.assertAlmostEqual(0.1, _quadruples(test) / total, places=1)

    def test_weighted_counter_changes_the_cuts(self):
        # Every snapshot holds the same 10 distinct edges, but the first one
        # repeats each of them 50 times. Counting distinct edges sees a flat
        # sequence; counting interactions sees a burst - so the two schemes
        # place the cuts in completely different places.
        snapshots = _snapshots([10] * 20, weight=[50] + [1] * 19)

        unweighted_train, _, _ = VolumeBalancedSplitter(
            counter=UnweightedQuadrupleCounter()
        ).split(snapshots)
        weighted_train, _, _ = VolumeBalancedSplitter(
            counter=WeightedQuadrupleCounter()
        ).split(snapshots)

        self.assertEqual(16, len(unweighted_train.time_index))
        self.assertEqual(6, len(weighted_train.time_index))

    def test_uniform_weights_do_not_change_the_cuts(self):
        # Scaling every snapshot by the same weight shifts no ratios, so both
        # counting schemes agree.
        snapshots = _snapshots([10] * 20, weight=5)
        unweighted_train, _, _ = VolumeBalancedSplitter(
            counter=UnweightedQuadrupleCounter()
        ).split(snapshots)
        weighted_train, _, _ = VolumeBalancedSplitter(
            counter=WeightedQuadrupleCounter()
        ).split(snapshots)

        self.assertEqual(
            len(unweighted_train.time_index), len(weighted_train.time_index)
        )

    def test_splits_stay_in_temporal_order(self):
        snapshots = _snapshots([10] * 20)
        train, val, test = VolumeBalancedSplitter().split(snapshots)

        self.assertLess(train.time_index.max(), val.time_index.min())
        self.assertLess(val.time_index.max(), test.time_index.min())

    def test_embargo_drops_last_snapshot_of_train_and_validation(self):
        snapshots = _snapshots([10] * 20)
        train, val, test = VolumeBalancedSplitter(embargo=True).split(snapshots)

        # Without the embargo this is 16/2/2 (see test above).
        self.assertEqual(
            (15, 1, 2),
            (len(train.time_index), len(val.time_index), len(test.time_index)),
        )
        # The test set is untouched - nothing follows it to leak into.
        self.assertEqual(snapshots.time_index[-1], test.time_index[-1])

    def test_embargo_drops_as_many_snapshots_as_it_is_given(self):
        # Enough snapshots that a three-snapshot embargo leaves a validation
        # set behind
        snapshots = _snapshots([10] * 100)
        plain = VolumeBalancedSplitter().split(snapshots)

        train, val, test = VolumeBalancedSplitter(embargo=3).split(snapshots)

        self.assertEqual(len(plain[0].time_index) - 3, len(train.time_index))
        self.assertEqual(len(plain[1].time_index) - 3, len(val.time_index))
        # Nothing follows the test set, so it keeps everything
        self.assertEqual(len(plain[2].time_index), len(test.time_index))

    def test_embargo_true_still_means_one_snapshot(self):
        snapshots = _snapshots([10] * 20)

        by_flag = VolumeBalancedSplitter(embargo=True).split(snapshots)
        by_count = VolumeBalancedSplitter(embargo=1).split(snapshots)

        self.assertEqual(
            [len(part.time_index) for part in by_flag],
            [len(part.time_index) for part in by_count],
        )

    def test_no_embargo_is_the_same_as_zero(self):
        snapshots = _snapshots([10] * 20)

        by_flag = VolumeBalancedSplitter(embargo=False).split(snapshots)
        by_count = VolumeBalancedSplitter(embargo=0).split(snapshots)

        self.assertEqual(
            [len(part.time_index) for part in by_flag],
            [len(part.time_index) for part in by_count],
        )

    def test_a_negative_embargo_is_refused(self):
        with self.assertRaises(ValueError):
            VolumeBalancedSplitter(embargo=-1)

    def test_an_embargo_never_reaches_past_a_split(self):
        # More snapshots embargoed than the validation set holds
        snapshots = _snapshots([10] * 20)

        train, val, test = VolumeBalancedSplitter(
            embargo=10, allow_empty_splits=True
        ).split(snapshots)

        self.assertEqual(0, len(val.time_index))
        self.assertEqual(snapshots.time_index[-1], test.time_index[-1])

    def test_empty_split_raises_with_actionable_message(self):
        # A single snapshot holding almost everything forces both cuts to the
        # same index, leaving the validation set empty.
        snapshots = _snapshots([1, 1, 1, 1, 500])
        with self.assertRaises(ValueError) as ctx:
            VolumeBalancedSplitter().split(snapshots)
        self.assertIn("validation", str(ctx.exception))
        self.assertIn("freq", str(ctx.exception))

    def test_empty_split_allowed_when_requested(self):
        snapshots = _snapshots([1, 1, 1, 1, 500])
        train, val, test = VolumeBalancedSplitter(allow_empty_splits=True).split(
            snapshots
        )
        self.assertEqual(0, len(val.time_index))
        self.assertEqual(504, _quadruples(train) + _quadruples(val) + _quadruples(test))


class TestSnapshotCountSplitter(unittest.TestCase):
    def test_splits_by_snapshot_count(self):
        snapshots = _snapshots([10] * 20)
        train, val, test = SnapshotCountSplitter().split(snapshots)

        self.assertEqual(
            (16, 2, 2),
            (len(train.time_index), len(val.time_index), len(test.time_index)),
        )

    def test_burst_leaves_quadruple_ratios_unbalanced(self):
        # The baseline this exists to contrast with: identical data to
        # test_burst_shifts_cuts_to_keep_volume_balanced, but cutting by
        # snapshot count hands the training set 86% of all quadruples.
        snapshots = _snapshots([100] + [10] * 19)
        train, val, test = SnapshotCountSplitter().split(snapshots)

        self.assertEqual(
            (16, 2, 2),
            (len(train.time_index), len(val.time_index), len(test.time_index)),
        )
        self.assertEqual(
            (250, 20, 20), (_quadruples(train), _quadruples(val), _quadruples(test))
        )

        total = _quadruples(snapshots)
        self.assertAlmostEqual(0.862, _quadruples(train) / total, places=2)


class TestSplitterConfiguration(unittest.TestCase):
    def test_rejects_sizes_leaving_no_room_for_test_set(self):
        with self.assertRaises(ValueError):
            VolumeBalancedSplitter(train_size=0.9, val_size=0.1)

    def test_rejects_non_positive_sizes(self):
        with self.assertRaises(ValueError):
            VolumeBalancedSplitter(train_size=0.0, val_size=0.1)

    def test_custom_ratios_are_respected(self):
        snapshots = _snapshots([10] * 20)
        train, val, test = VolumeBalancedSplitter(train_size=0.6, val_size=0.2).split(
            snapshots
        )

        self.assertEqual(
            (120, 40, 40), (_quadruples(train), _quadruples(val), _quadruples(test))
        )
