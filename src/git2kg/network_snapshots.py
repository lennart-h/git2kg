import pandas as pd


class NetworkSnapshots:
    """Wrapper class for created snapshots. Used to be passed to exporter"""

    def __init__(
        self,
        time_index: pd.DatetimeIndex,
        networks: list[dict],
        features: list[dict],
        active_masks: list[dict],
        labels: list[dict],
    ):
        if not len(time_index) == len(networks) == len(features):
            raise Exception("time_index, networks and features must be of same length")
        self.time_index = time_index
        self.networks = networks
        self.features = features
        self.labels = labels
        self.active_masks = active_masks

    def to_dict(self) -> dict:
        res = {}
        for i in range(len(self.time_index)):
            res[self.time_index[i]] = {
                "network": self.networks[i],
                "features": self.features[i],
                "labels": self.labels[i],
                "active_masks": self.active_masks[i],
            }
        return res
