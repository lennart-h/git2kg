import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from git2kg.datapoint.base import Datapoint, DatapointList
from git2kg.datapoint.utils import (
    csv_to_datapoint_list,
    datapoint_list_to_dataframe,
)


class PersistenceManager(ABC):
    def __init__(self, repository_id: Any):
        self.repository_id = repository_id

    @abstractmethod
    def load(self, datapoint_class: type[Datapoint], processes=1) -> DatapointList:
        pass

    @abstractmethod
    def save(self, datapoints: DatapointList):
        pass

    @abstractmethod
    def append(self, datapoints: DatapointList):
        pass


class CSVPersistenceManager(PersistenceManager):
    python_to_pandas_types = {
        int: pd.Int64Dtype(),
        float: pd.Float64Dtype(),
        str: pd.StringDtype(),
        bool: pd.BooleanDtype(),
        datetime: "datetime64[ns]",
    }

    def __init__(self, repository_id: Any, base_path: str):
        super().__init__(repository_id)
        self.base_path = base_path

    def load(self, datapoint_class: type[Datapoint], processes=1) -> DatapointList:
        file_path = (
            f"{self.base_path}/{self.repository_id}_{datapoint_class.__name__}.csv"
        )
        if not os.path.isfile(path=file_path):
            return DatapointList([], dp_type=datapoint_class)
        # Read a block of rows at a time: resuming a mine of a large
        # repository otherwise holds the whole table next to its datapoints
        return csv_to_datapoint_list(
            file_path,
            dp_type=datapoint_class,
            processes=processes,
            lineterminator="\n",
        )

    def save(self, datapoints: DatapointList):
        if not datapoints:
            return
        datapoint_class = datapoints.dp_type
        file_path = (
            f"{self.base_path}/{self.repository_id}_{datapoint_class.__name__}.csv"
        )
        df = datapoint_list_to_dataframe(datapoints)
        df.to_csv(file_path, index=False, lineterminator="\n")

    def append(self, datapoints: DatapointList):
        if not datapoints:
            return
        datapoint_class = datapoints.dp_type
        file_path = (
            f"{self.base_path}/{self.repository_id}_{datapoint_class.__name__}.csv"
        )
        try:
            df = datapoint_list_to_dataframe(datapoints)
            file_exists = os.path.exists(file_path)
            if file_exists:
                df.to_csv(
                    file_path, mode="a", header=False, index=False, lineterminator="\n"
                )
            else:
                df.to_csv(
                    file_path, mode="w", header=True, index=False, lineterminator="\n"
                )
        except FileNotFoundError:
            self.save(datapoints)


class NoOperationPersistenceManager(PersistenceManager):
    def load(self, datapoint_class: type[Datapoint], processes=0) -> DatapointList:
        return DatapointList([], datapoint_class)

    def save(self, datapoints: DatapointList):
        pass

    def append(self, datapoints: DatapointList):
        pass
