import pandas as pd


class VerificationException(Exception):
    pass


def verify_columns(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Verifies whether the provided dataframe's column index contains a defined set of names

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to verify
    column_names : list[str]
        The columns that are required to be contained in the dataframes column index

    Returns
    -------
    pd.DataFrame
        The provided dataframe upon successfull verification
    Raises
    ------
    VerificationExceptionIf the verification is not successful

    """
    if len(set(column_names)) != len(set(df.columns).intersection(set(column_names))):
        missing_columns = set(column_names) - set(df.columns)
        raise VerificationException(f"""Dataframe {df} must have all colums {column_names} contained in it's column index
                                    Columns {missing_columns} are missing""")
    return df
