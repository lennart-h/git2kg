import warnings
from ast import literal_eval
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from typing import Callable, List, Set, get_origin

import pandas as pd

from git2kg.datapoint.base import Convertable, Datapoint, DatapointList

# Rows turned into dicts per step, bounding what a conversion holds beyond the
# frame it reads and the datapoints it builds
CONVERSION_CHUNK_SIZE = 50_000
# Rows per task when converting with a pool
POOL_CHUNK_SIZE = 1_000
# Rows read from a stored table at a time
CSV_CHUNK_SIZE = 100_000


def convert_to_pandas_types(df, datapoint_class: type[Convertable]) -> pd.DataFrame:
    """
    Convert the columns of a DataFrame to the specified pandas data types based on provided type hints.

        Parameters:
        - df: The input DataFrame to be converted.
    - type_hints: A dictionary mapping column names to their expected Python data types.

    Returns:
    - A new DataFrame with columns converted to the specified pandas data types.
    """
    python_to_pandas_types = {
        int: pd.Int64Dtype(),
        float: pd.Float64Dtype(),
        str: pd.StringDtype(),
        bool: pd.BooleanDtype(),
        Set[str]: object,
        List[str]: object,
        List[int]: object,
    }
    type_hints = datapoint_class.dtypes(flatten=True)
    cols_not_present = set(type_hints.keys()) - set(df.columns)
    for col in cols_not_present:
        df[col] = pd.NA
    datatypes = {
        column: python_to_pandas_types.get(dtype, dtype)
        for column, dtype in type_hints.items()
    }

    # Handle datetime columns separately with to_datetime (parses tz offsets properly)
    datetime_cols = [col for col, dtype in type_hints.items() if dtype == datetime]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], utc=True)

    # astype everything else
    other_datatypes = {
        col: dt for col, dt in datatypes.items() if col not in datetime_cols
    }
    df = df.astype(other_datatypes)
    return df


def convert_from_pandas_types(row_dict, datapoint_class: type[Datapoint]):
    constructor_mapping: dict[type, Callable] = {
        list: literal_eval,
        set: literal_eval,
        datetime: lambda x: x.to_pydatetime(),
    }
    type_hints = datapoint_class.dtypes(flatten=True)

    converted_dict = {}

    for field, dtype in type_hints.items():
        origin_dtype: type | None = get_origin(dtype)
        origin_dtype = origin_dtype if origin_dtype is not None else dtype
        val = row_dict[field]
        constructor = constructor_mapping.get(origin_dtype, origin_dtype)
        converted_dict[field] = None if pd.isna(val) else constructor(val)

    return converted_dict


def datapoint_list_to_dataframe(datapoints: DatapointList) -> pd.DataFrame:
    datapoint_class = datapoints.dp_type
    if len(datapoints) == 0:
        colnames = list(datapoint_class.dtypes(flatten=True).keys())
        df = pd.DataFrame(columns=colnames)
    else:
        df = pd.DataFrame(datapoints.to_dict_list())
    df = convert_to_pandas_types(df, datapoint_class)
    return df


def _row_to_datapoint(row: dict, datapoint_class: type[Datapoint]) -> Datapoint:
    """Builds one datapoint from one row of a stored table.

    The two conversions are one step rather than two passes: a pass over a
    table of millions of rows keeps every intermediate alive until it ends,
    and with a process pool it also pays to move them between processes.
    """
    return datapoint_class.from_flat_dict(
        convert_from_pandas_types(row, datapoint_class)
    )


def _row_dicts(df: pd.DataFrame, chunk_size: int) -> Iterator[dict]:
    """Yields the rows of a frame as dicts, a chunk at a time.

    Converting the whole frame at once holds a dict per row of the table in
    memory on top of the frame itself and of the datapoints being built.
    """
    for start in range(0, len(df), chunk_size):
        yield from df.iloc[start : start + chunk_size].to_dict(orient="records")


def _convert_frame(
    df: pd.DataFrame, dp_type: type[Datapoint], pool: PoolType | None
) -> list:
    """Rebuilds the datapoints of one frame."""
    df = convert_to_pandas_types(df, dp_type)
    convert = partial(_row_to_datapoint, datapoint_class=dp_type)
    rows = _row_dicts(df, CONVERSION_CHUNK_SIZE)
    if pool is None:
        return [convert(row) for row in rows]
    # imap over the generator, so that neither the rows nor their datapoints
    # are all in flight at once
    return list(pool.imap(convert, rows, chunksize=POOL_CHUNK_SIZE))


def dataframe_to_datapoint_list(
    df: pd.DataFrame, dp_type: type[Datapoint], processes: int | None = None
) -> DatapointList:
    """Rebuilds the datapoints of a table that is already in memory.

    Parameters
    ----------
    processes : int, optional
        How many processes to convert with. One by default: a row carries
        little work, so sending it to another process and its datapoint back
        costs more than the conversion itself
    """
    if processes is not None and processes > 1:
        with Pool(processes) as pool:
            datapoints = _convert_frame(df, dp_type, pool)
    else:
        datapoints = _convert_frame(df, dp_type, None)
    return DatapointList(datapoints, dp_type=dp_type)


def csv_to_datapoint_list(
    file_path: str,
    dp_type: type[Datapoint],
    processes: int | None = None,
    **read_options,
) -> DatapointList:
    """Rebuilds the datapoints of a stored table, a block of rows at a time.

    Reading the table in one piece holds the whole frame alongside the
    datapoints built from it. On a large export that frame costs several times
    what the datapoints do, and it is not needed once they exist, so only a
    block of it is read at a time.
    """
    datapoints: list = []
    pool = Pool(processes) if processes is not None and processes > 1 else None
    try:
        # The parser infers a dtype per block and warns where a column looks
        # like one type in one block and another in the next. Nothing reads
        # those dtypes: `_convert_frame` casts every column from the
        # datapoint's own annotations before building anything. Reading the
        # file in one piece would settle the inference instead, but that is
        # what this function reads in blocks to avoid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.DtypeWarning)
            with pd.read_csv(
                file_path, chunksize=CSV_CHUNK_SIZE, **read_options
            ) as reader:
                for chunk in reader:
                    datapoints.extend(_convert_frame(chunk, dp_type, pool))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return DatapointList(datapoints, dp_type=dp_type)
