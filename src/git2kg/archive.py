"""Serialization of a Git2KG instance to a compressed tar archive."""

from __future__ import annotations

import json
import os
import tarfile
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pandas as pd

from git2kg.datapoint import ActorInfo
from git2kg.disambiguation import DISAMBIGUATION_VERSION

if TYPE_CHECKING:
    from git2kg.core import Git2KG


def _actor_mapping_path(path: str, name: str) -> str:
    return f"{path}/{name}-actor_mapping.csv"


def load_actor_mapping(path: str, name: str) -> pd.DataFrame | None:
    """Reads the actor mapping stored alongside a GitData export.

    Parameters
    ----------
    path : str
        The directory the archive was extracted to
    name : str
        The name of the Git2KG the archive holds

    Returns
    -------
    pd.DataFrame or None
        The stored mapping, or None for archives written before mappings were
        stored
    """
    mapping_path = _actor_mapping_path(path, name)
    if not os.path.isfile(mapping_path):
        return None
    # Read as text so that an alias that happens to read like a missing value
    # ("NA", "null", ...) survives the round trip, then convert the id columns
    # back. The result matches what `alias_frame` produces for the same actors.
    mapping = pd.read_csv(
        mapping_path, lineterminator="\n", keep_default_na=False, dtype=str
    )
    github_ids = mapping[ActorInfo.GITHUB_ID_COLUMN].replace("", None)
    typed_columns: dict[str, pd.Series] = {
        "alias_name": mapping["alias_name"].astype("string"),
        "alias_email": mapping["alias_email"].astype("string"),
        ActorInfo.GITHUB_ID_COLUMN: pd.to_numeric(github_ids).astype("Int64"),
        ActorInfo.ID_COLUMN: mapping[ActorInfo.ID_COLUMN].astype("int64"),
    }
    return mapping.assign(**typed_columns)


def save_git2kg(g2kg: Git2KG, path: str) -> str:
    """Saves the provided Git2KG to a compressed tar archive, containing the GitData export and metadata of the Git2KG object. The tar archive can be loaded again with `git2kg.load()`

    Parameters
    ----------
    g2kg : Git2KG
        The Git2KG object to be saved
    path : str
        The directory where the tar archive is created

    Returns
    -------
    str
        The path of the created tar archive

    Raises
    ------
    AttributeError
        If the Git2KG object has `None` value for the name attribute
    """
    if not g2kg.name:
        raise AttributeError(
            "Saving Git2KG objects without a name is not supported. Set a name before saving"
        )
    metadata = {
        "freq": g2kg.freq,
        "mined_on": g2kg.mined_on.isoformat() if g2kg.mined_on else None,
        "name": g2kg.name,
        "disambiguation_version": DISAMBIGUATION_VERSION,
    }
    with TemporaryDirectory() as tmp_dir:
        g2kg.gitdata.save(tmp_dir, g2kg.name)
        g2kg.actor_mapping.to_csv(
            _actor_mapping_path(tmp_dir, g2kg.name), index=False, lineterminator="\n"
        )
        with open(f"{tmp_dir}/metadata.json", "w") as meta_file:
            meta_file.write(json.dumps(metadata))
        tar_path = f"{path}/{g2kg.name}-git2kg.tar.gz"
        with tarfile.open(tar_path, mode="w:gz") as tar:
            for file in os.listdir(tmp_dir):
                tar.add(f"{tmp_dir}/{file}", arcname=file)
        return tar_path
