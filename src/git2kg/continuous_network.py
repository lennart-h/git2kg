import pandas as pd


class ContinuousNetwork:
    """Continuous-time counterpart to `NetworkSnapshots`.

    Instead of discretizing interactions into a fixed-frequency series of
    aggregated, weighted graph snapshots, every interaction is kept as its
    own timestamped edge, giving a continuous-time dynamic graph (CTDG) of
    the mined repository. Node features are computed once, over the full
    history, since there is no per-snapshot state to track - the temporal
    signal lives entirely in the edges' timestamps.
    """

    def __init__(self, edges: dict, features: dict):
        self.edges = edges
        self.features = features

    def to_dict(self) -> dict:
        return {"edges": self.edges, "features": self.features}

    def to_unit_time(
        self, unit: str = "s", origin: pd.Timestamp | None = None
    ) -> "ContinuousNetwork":
        """Returns a copy where every edge's \"ts\" column is replaced by the
        elapsed time since `origin`, expressed as a float multiple of `unit`,
        instead of an absolute timestamp.

        This numeric "unit time" axis is what continuous-time graph learning
        frameworks typically expect edge times to look like - e.g.
        `pathpyG.TemporalGraph`, whose `time` attribute must be int/float
        rather than a datetime.

        Parameters
        ----------
        unit : str, optional
            The pandas offset alias defining one unit of time (e.g. "s", "h",
            "D"), by default "s"
        origin : pd.Timestamp | None, optional
            The zero point of the resulting time axis, by default the
            timestamp of the earliest interaction across all edges

        Returns
        -------
        ContinuousNetwork
            A new instance with unit-time edges; `features` is shared, unchanged
        """
        if origin is None:
            origin = min(
                df["ts"].min()
                for _tos in self.edges.values()
                for rels in _tos.values()
                for df in rels.values()
                if len(df)
            )
        one_unit = pd.Timedelta(f"1{unit}")
        new_edges = {
            _from: {
                _to: {
                    name: df.assign(ts=(df["ts"] - origin) / one_unit)
                    for name, df in rels.items()
                }
                for _to, rels in _tos.items()
            }
            for _from, _tos in self.edges.items()
        }
        return ContinuousNetwork(new_edges, self.features)
