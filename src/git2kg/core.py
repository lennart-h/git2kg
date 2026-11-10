from __future__ import annotations

import json
import logging
import os
import re
import tarfile
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from git2kg.active_masks import compute_active_masks
from git2kg.archive import load_actor_mapping, save_git2kg
from git2kg.continuous_network import ContinuousNetwork
from git2kg.dataloading import GRAPHQL, Dataloader
from git2kg.datapoint import ActorInfo
from git2kg.disambiguation import (
    DISAMBIGUATION_VERSION,
    disambiguated_author_mapping,
    mapping_matches,
)
from git2kg.features import FeatureData, feature_data
from git2kg.files import FileRegistry
from git2kg.filtering import DEFAULT_FILTERS, NodeFilter, excluded_ids
from git2kg.gitdata import GitData, _load_gitdata
from git2kg.labels import (
    ActorLabelExtractor,
    CommitLabelExtractor,
    FileLabelExtractor,
    IssueLabelExtractor,
)
from git2kg.network_snapshots import NetworkSnapshots
from git2kg.persistence import (
    CSVPersistenceManager,
    NoOperationPersistenceManager,
    PersistenceManager,
)
from git2kg.snapshot_compute import (
    _compile_feature_matrices_for_timeslice,
    _get_continuous_edges,
    _get_edges_for_timeslice,
)
from git2kg.timeslicing import TimesliceIterator

LOG = logging.getLogger(__name__)

# co_commit connects every pair of files touched by the same commit, so one
# commit contributes n(n-1)/2 edges and a few large commits dominate the graph.
# It is fully derivable from commit -modifies-> file, so it is left out unless
# asked for.
DEFAULT_EDGE_EXCLUDE = ["co_commit_edges"]


class Git2KG:
    def __init__(
        self,
        gitdata: GitData,
        freq: str | None,
        mined_on: datetime | None = None,
        name: str | None = None,
        actor_mapping: pd.DataFrame | None = None,
        filters: Iterable[NodeFilter] | None = None,
    ):
        self.gitdata: GitData = gitdata
        self._freq: pd.offsets.BaseOffset | None = None
        if freq is not None:
            self._freq = pd.tseries.frequencies.to_offset(freq)

        reuse_mapping = mapping_matches(actor_mapping, gitdata)
        if actor_mapping is not None and not reuse_mapping:
            LOG.info(
                "The provided actor mapping does not describe the actors of this "
                "GitData, disambiguating again"
            )
        self.actor_mapping: pd.DataFrame = (
            actor_mapping
            if actor_mapping is not None and reuse_mapping
            else disambiguated_author_mapping(gitdata)
        )
        # A freshly computed mapping may number the actors differently than
        # whatever stamped the ids currently in the data.
        if not (reuse_mapping and gitdata.actors_disambiguated):
            self.gitdata.add_actor_ids(self.actor_mapping)
        if not gitdata.files_disambiguated:
            self.gitdata.add_file_ids()
        if not gitdata.commits_disambiguated:
            self.gitdata.add_commit_ids()
        # Exclusion is by id, so it can only be decided once ids are stamped
        self.filters = tuple(DEFAULT_FILTERS if filters is None else filters)
        self._excluded_ids = excluded_ids(self.gitdata, self.filters)

        self.mined_on = mined_on
        self.name = name

    @property
    def timeslices(self) -> Sequence[GitData]:
        return TimesliceIterator(self.gitdata, self.get_timeindex())

    @property
    def freq(self):
        return self._freq.freqstr if self._freq else None

    @freq.setter
    def freq(self, value: str):
        self._freq = None
        if value is not None:
            self._freq = pd.tseries.frequencies.to_offset(value)

    def get_snapshots(
        self,
        actor_label_extractor: Callable[..., ActorLabelExtractor] | None = None,
        issue_label_extractor: Callable[..., IssueLabelExtractor] | None = None,
        commit_label_extractor: Callable[..., CommitLabelExtractor] | None = None,
        file_label_extractor: Callable[..., FileLabelExtractor] | None = None,
        edge_exclude: list[str] | None = None,
        processes: int | None = 1,
    ):
        """Computes the snapshot sequence.

        Parameters
        ----------
        edge_exclude : list[str], optional
            Relation names to leave out, `DEFAULT_EDGE_EXCLUDE` when not given
        processes : int, optional
            How many processes to compute the edges and the feature matrices
            with, one per core when not given. Each process is sent the data
            of the timeslice it works on, so a large repository is worked on
            with fewer processes than a machine has cores
        """
        if edge_exclude is None:
            edge_exclude = DEFAULT_EDGE_EXCLUDE
        return NetworkSnapshots(
            self.get_timeindex()[1:],
            self.get_networks(exclude=edge_exclude, processes=processes),
            self.get_feature_matrices(processes=processes),
            self.get_active_masks(),
            self.get_labels(
                actor_label_extractor,
                issue_label_extractor,
                commit_label_extractor,
                file_label_extractor,
            ),
        )

    def get_continuous_network(
        self, exclude: list[str] | None = None
    ) -> ContinuousNetwork:
        """Continuous-time counterpart to `get_snapshots`: exports the full
        interaction history as one continuous stream of timestamped edges,
        instead of a fixed-frequency series of discretized, weighted graph
        snapshots. Use `ContinuousNetwork.to_unit_time()` on the result to get
        a numeric (rather than datetime) time axis, e.g. for frameworks like
        `pathpyG.TemporalGraph`.

        Parameters
        ----------
        exclude : list[str], optional
            Relation names to exclude from the resulting edge set, by default
            `DEFAULT_EDGE_EXCLUDE`

        Returns
        -------
        ContinuousNetwork
            The continuous-time network, bundling the edge stream with node
            features computed once over the full history
        """
        if exclude is None:
            exclude = DEFAULT_EDGE_EXCLUDE
        # The whole history is one unsliced piece, so its features describe it
        # as of the end of mining
        features = _compile_feature_matrices_for_timeslice(
            (feature_data(self.gitdata), None), self._id_universes()
        )
        return ContinuousNetwork(self.get_continuous_edges(exclude=exclude), features)

    def get_continuous_edges(self, exclude: list[str] | None = None) -> dict:
        """Computes the full interaction history as a continuous-time edge
        stream, preserving the exact timestamp of every individual interaction
        instead of aggregating same-pair interactions into a single weighted
        edge per timeslice (see `get_networks`).

        Parameters
        ----------
        exclude : list[str], optional
            Relation names to exclude from the resulting edge set, by default
            `DEFAULT_EDGE_EXCLUDE`

        Returns
        -------
        dict
            Dictionary of structure u_type -> v_type -> relation -> edge_set,
            where each edge_set is a DataFrame with columns "u", "v" and "ts",
            one row per interaction, sorted by time
        """
        if exclude is None:
            exclude = DEFAULT_EDGE_EXCLUDE
        return _get_continuous_edges(
            self.gitdata, self.actor_mapping, exclude, self._excluded_ids
        )

    def get_networks(
        self, exclude: list[str] | None = None, processes: int | None = None
    ):
        if exclude is None:
            exclude = DEFAULT_EDGE_EXCLUDE
        timeslices = list(self.timeslices)
        worker = partial(
            _get_edges_for_timeslice,
            actor_mapping=self.actor_mapping,
            exclude=exclude,
            excluded_ids=self._excluded_ids,
        )
        if processes == 1:
            return [
                worker(timeslice)
                for timeslice in tqdm(timeslices, desc="Creating edge snapshots")
            ]
        with Pool(processes) as pool:
            return list(
                tqdm(
                    pool.imap(worker, timeslices),
                    total=len(timeslices),
                    desc="Creating edge snapshots",
                )
            )

    def get_feature_matrices(self, processes: int | None = None):
        """Compiles the feature matrices of every snapshot.

        Features describe the state of each node as of the end of its
        timeslice, so they are computed over the history up to that point: a
        node that exists but is idle in a timeslice keeps the features it has
        accumulated, where computing them from the timeslice alone would
        describe every idle node as a row of zeros. The edges of a snapshot,
        in contrast, are the interactions of that timeslice alone.
        """
        # The tables are built once and each snapshot only says how far to
        # read them, so a worker never receives a copy of the mined history
        data = feature_data(self.gitdata)
        ids = self._id_universes()
        bounds = list(self.get_timeindex()[1:])
        desc = "Compiling feature matrices for snapshots"

        if processes == 1:
            return [
                _compile_feature_matrices_for_timeslice((data.as_of(as_of), as_of), ids)
                for as_of in tqdm(bounds, desc=desc)
            ]
        with Pool(
            processes, initializer=_init_feature_worker, initargs=(data, ids)
        ) as pool:
            return list(
                tqdm(
                    pool.imap(_compile_feature_matrices_as_of, bounds),
                    total=len(bounds),
                    desc=desc,
                )
            )

    def get_labels(
        self,
        actor_label_extractor: Callable[..., ActorLabelExtractor] | None,
        issue_label_extractor: Callable[..., IssueLabelExtractor] | None,
        commit_label_extractor: Callable[..., CommitLabelExtractor] | None = None,
        file_label_extractor: Callable[..., FileLabelExtractor] | None = None,
    ):
        n_slices = len(self.timeslices)
        ids_by_node_type = self._id_universes()
        extractors_by_node_type = {
            "actor": actor_label_extractor,
            "issue": issue_label_extractor,
            "commit": commit_label_extractor,
            "file": file_label_extractor,
        }

        sequences = {}
        for node_type, extractor in extractors_by_node_type.items():
            sequences[node_type] = (
                extractor(
                    self.gitdata,
                    self.timeslices,
                    self.get_timeindex(),
                    ids_by_node_type[node_type],
                ).extract_label_sequence()
                if extractor
                else [np.array([]) for _ in range(n_slices)]
            )

        # Only actor/issue are checked, as commit/file extractors are optional
        for node_type in ("actor", "issue"):
            if len(sequences[node_type]) != n_slices:
                raise ValueError(
                    f"The {node_type} label extractor returned "
                    f"{len(sequences[node_type])} label vectors for {n_slices} "
                    "timeslices. Labels must be aligned to snapshots positionally, "
                )

        return [
            {node_type: seq[i] for node_type, seq in sequences.items()}
            for i in range(n_slices)
        ]

    def get_active_masks(self) -> list[dict]:
        """Computes which nodes exist in each timeslice, as one boolean array
        per node type per timeslice. See `git2kg.active_masks`.
        """
        return compute_active_masks(
            self.gitdata, self.get_timeindex(), self._id_universes()
        )

    def _id_universes(self) -> dict[str, pd.Series]:
        """Maps node type to its id universe, which defines the row order of
        that node type's feature matrices, labels and active masks. Ids the
        filters exclude are not part of it.
        """
        universes = {
            "actor": self.actor_mapping.copy()[ActorInfo.ID_COLUMN].drop_duplicates(),
            "issue": pd.Series(
                [i.issue.number for i in self.gitdata.issues], name="issue_number"
            ).drop_duplicates(),
            "commit": pd.Series(
                self.gitdata.file_registry.commit_ids, name="commit_id"
            ).sort_values(),
            "file": pd.Series(
                self.gitdata.file_registry.file_ids, name="file_id"
            ).sort_values(),
        }
        return {
            node_type: ids[~ids.isin(self._excluded_ids[node_type])].reset_index(
                drop=True
            )
            for node_type, ids in universes.items()
        }

    def save(self, path) -> str:
        """Saves the Git2KG to a compressed tar archive, containing the GitData export and metadata of the Git2KG object. The tar archive can be loaded again with `git2kg.load()`

        Parameters
        ----------
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
        return save_git2kg(self, path)

    def get_timeindex(self) -> pd.DatetimeIndex:
        """Creates a timeindex based on first and last interactions in the global data and the freq parameter of self

        Returns
        -------
        pd.DatetimeIndex
            The resulting datetime index for creating time slices of data

        Raises
        ------
        Exception
            if the length of the resulting datetime index is less than two
        """
        dt_idx = pd.DatetimeIndex(
            [
                self.gitdata.first_interaction(),
                self.gitdata.last_interaction() + pd.Timedelta(seconds=1),
            ]
        )
        if self._freq:
            dt_idx = pd.date_range(
                start=self.gitdata.first_interaction()
                - self._freq,  # Time of first interaction must be in interval
                end=self.gitdata.last_interaction()
                + self._freq,  # Time of last interaction must be in interval
                freq=self._freq,
            )
        if len(dt_idx) < 2:
            raise Exception(
                "Provided datetime frequency must yield a DatetimeIndex of at least 2"
            )
        return dt_idx

    @staticmethod
    def load(
        path: str,
        filters: Iterable[NodeFilter] | None = None,
        processes: int | None = None,
    ) -> Git2KG:
        """Loads a Git2KG from the archive `save` wrote.

        Parameters
        ----------
        path : str
            The archive to read
        filters : Iterable[NodeFilter], optional
            The filters deciding which nodes are left out, `DEFAULT_FILTERS`
            when not given
        processes : int, optional
            How many processes to rebuild the datapoints with, one by default.
            More is faster on a large export, at the cost of holding the rows
            of a table in several processes at once
        """
        if not (path.endswith(".tar.gz") or path.endswith(".tar")):
            raise ValueError("File must be a (compressed) tar archive")
        with TemporaryDirectory() as temp_dir:
            with tarfile.open(path, mode="r") as tar:
                tar.extractall(temp_dir, filter="data")
            with open(f"{temp_dir}/metadata.json", "r") as meta_file:
                metadata = json.load(meta_file)
            name = metadata["name"]
            gitdata: GitData = _load_gitdata(temp_dir, name, processes=processes)
            actor_mapping = _stored_actor_mapping(temp_dir, name, metadata)
        mined_on = (
            datetime.fromisoformat(metadata["mined_on"])
            if metadata.get("mined_on")
            else None
        )
        freq = metadata["freq"]
        # Archives written before snapshots became non-incremental carry an
        # "incremental" flag, which no longer describes anything
        return Git2KG(gitdata, freq, mined_on, name, actor_mapping, filters)

    @staticmethod
    def from_github_url(
        url: str,
        github_token: str,
        freq="ME",
        persistence_path=None,
        processes: int | None = 1,
        github_api: str = GRAPHQL,
    ) -> Git2KG:
        """Mines a Git2KG object from a given GitHub Repository

        Parameters
        ----------
        url : str
            The URL of the GitHub Repository
        github_token : str
            A GitHub Access token with public:repo permission (used for mining via GitHub API)
        freq : str, optional
            The frequency of the datetime index of the Git2KG object, by default "ME"
        github_api : str, optional
            Which GitHub API to mine with, `GRAPHQL` (default) or `REST`
        Returns
        -------
        Git2KG
            The mined Git2KG object

        Raises
        ------
        Exception
            If the provided GitHub URL is not valid
        """
        match = re.match(
            r"^(?P<base>https://github.com)/(?P<vendor>.+?)/(?P<product>.+?)(?P<suffix>\.git|/)?$",
            url,
        )
        if not match:
            raise Exception(f"{url} is not a valid github URL")
        base = match.group("base")
        vendor = match.group("vendor")
        product = match.group("product")
        git_url = f"{base}/{vendor}/{product}.git"
        repo_name = f"{vendor}_{product}"

        LOG.info(f"Mining repository {vendor}/{product}")

        persistence_manager: PersistenceManager = NoOperationPersistenceManager(
            repository_id=repo_name
        )
        if persistence_path:
            if os.path.isdir(persistence_path):
                persistence_manager = CSVPersistenceManager(repo_name, persistence_path)
            else:
                LOG.warning(
                    f"Persistence path '{persistence_path}' is not a directory. Falling back to no persistence"
                )

        with Dataloader(
            git_url,
            vendor,
            product,
            github_token,
            persistence_manager,
            processes=processes,
            github_api=github_api,
        ) as dataloader:
            # Load data
            commits = dataloader.load_commits()
            edits = dataloader.load_edits()
            issues = dataloader.load_issues()
            issue_comments = dataloader.load_issue_comments()
            review_requests = dataloader.load_issue_events()
            reviews = dataloader.load_reviews()
            issue_timeline_events = dataloader.load_issue_event_timeline()
            files = dataloader.load_modified_files()
            github_commits = dataloader.load_github_commits()

            # Create file registry to track renaming of files
            file_registry = FileRegistry.from_local_repository(
                dataloader.local_repo_path()
            )
        gitdata = GitData(
            commits,
            edits,
            issues,
            issue_comments,
            review_requests,
            reviews,
            modified_files=files,
            issue_timeline_events=issue_timeline_events,
            github_commits=github_commits,
            file_registry=file_registry,
        )
        return Git2KG(
            gitdata,
            freq=freq,
            mined_on=datetime.now(),
            name=repo_name,
        )

    def __str__(self):
        return f"""
Git2KG
------
Name: \t \t {self.name}
Mined On: \t {self.mined_on.isoformat() if self.mined_on else None}
Frequency: \t {self.freq}
{self.gitdata.__str__()}
        """


def _stored_actor_mapping(path: str, name: str, metadata: dict) -> pd.DataFrame | None:
    """Reads the actor mapping of an extracted archive, unless it was written by
    a disambiguation that would no longer produce it.

    Parameters
    ----------
    path : str
        The directory the archive was extracted to
    name : str
        The name of the Git2KG the archive holds
    metadata : dict
        The archive's metadata

    Returns
    -------
    pd.DataFrame or None
        The stored mapping, or None if there is none to reuse
    """
    stored_version = metadata.get("disambiguation_version")
    if stored_version != DISAMBIGUATION_VERSION:
        LOG.info(
            "Archive was written with disambiguation version %s, but this is "
            "version %s - disambiguating again",
            stored_version,
            DISAMBIGUATION_VERSION,
        )
        return None
    return load_actor_mapping(path, name)


# Set once per worker process by `_init_feature_worker`, so that the tables and
# id universes are sent to a worker once instead of with every snapshot
_FEATURE_WORKER: dict[str, Any] = {}


def _init_feature_worker(data: FeatureData, ids: dict[str, pd.Series]) -> None:
    _FEATURE_WORKER["data"] = data
    _FEATURE_WORKER["ids"] = ids


def _compile_feature_matrices_as_of(as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    data: FeatureData = _FEATURE_WORKER["data"]
    return _compile_feature_matrices_for_timeslice(
        (data.as_of(as_of), as_of), _FEATURE_WORKER["ids"]
    )
