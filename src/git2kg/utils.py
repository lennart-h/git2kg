import warnings
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from git2kg.datapoint import ActorInfo
from git2kg.datapoint.utils import convert_to_pandas_types


def iso8601_to_utc(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, format="ISO8601", utc=True)


def pandas_timestamps_to_datetime(timestamps: pd.Series) -> Iterable[datetime]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DatetimeProperties.to_pydatetime",
            category=FutureWarning,
        )
        return np.array(timestamps.dt.to_pydatetime())


def keep_earliest(earliest: dict, key, timestamp) -> None:
    """Records `timestamp` for `key`, unless something earlier is already known.

    Entities without an id are skipped: they are not part of any id universe,
    so nothing downstream can be keyed to them.
    """
    if key is None or pd.isna(key) or timestamp is None or pd.isna(timestamp):
        return
    current = earliest.get(key)
    if current is None or timestamp < current:
        earliest[key] = timestamp


def map_id_to_actor_infos(actor_infos: list[ActorInfo], actor_mapping: pd.DataFrame):
    actor_info_df = pd.DataFrame([a.to_dict() for a in actor_infos])
    actor_info_df = convert_to_pandas_types(actor_info_df, ActorInfo)
    actor_info_df.fillna({"name": "", "email": ""}, inplace=True)
    actor_mapping = actor_mapping.fillna({"alias_name": "", "alias_email": ""})
    merged = actor_info_df.merge(
        actor_mapping,
        how="left",
        left_on=["name", "email", "github_id"],
        right_on=["alias_name", "alias_email", ActorInfo.GITHUB_ID_COLUMN],
    )
    for actor, id in zip(actor_infos, merged[ActorInfo.ID_COLUMN], strict=True):
        actor.id = id
    return actor_infos
