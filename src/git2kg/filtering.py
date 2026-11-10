"""Graph construction policy: which mined entities become nodes, and which
commits count as development activity.

Both are applied after mining and disambiguation, so they never require a
repository to be mined again.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable

import pandas as pd

from git2kg.datapoint import ActorInfo, CommitDatapoint
from git2kg.gitdata import GitData

DEFAULT_BOT_PATTERNS = (
    r"\[bot\]$",
    r"^dependabot",
    r"^github-actions",
    r"noreply@github\.com",
)


class NodeFilter(ABC):
    """Excludes entities of one node type from the graph.

    An excluded entity is left out of that node type's id universe, so it has
    no row in the feature matrices, masks and labels, and every edge touching
    it is dropped. The datapoints themselves are not changed.

    Subclasses set `node_type` and decide per entity. The entity passed to
    `excludes` is an `ActorInfo` for "actor", a `CommitDatapoint` for
    "commit", an `IssueDatapoint` for "issue" and a `FileInfo` for "file".
    """

    node_type: ClassVar[str]

    @abstractmethod
    def excludes(self, entity: Any) -> bool:
        pass


class BotFilter(NodeFilter):
    """Excludes actors that are bots: accounts GitHub reports with type
    "Bot", and aliases matching any of the given patterns in their name or
    email, which catches bots arriving through git or as mentions.
    """

    node_type = "actor"

    def __init__(self, patterns: Iterable[str] = DEFAULT_BOT_PATTERNS):
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def excludes(self, entity: ActorInfo) -> bool:
        if entity.type == "Bot":
            return True
        alias = f"{entity.name or ''} {entity.email or ''}"
        return any(pattern.search(alias) for pattern in self.patterns)


DEFAULT_FILTERS: tuple[NodeFilter, ...] = (BotFilter(),)


def excluded_ids(gitdata: GitData, filters: Iterable[NodeFilter]) -> dict[str, set]:
    """Collects the ids every filter excludes, per node type.

    Parameters
    ----------
    gitdata : GitData
        The disambiguated gitdata, i.e. its entities carry their ids
    filters : Iterable[NodeFilter]
        The filters to apply

    Returns
    -------
    dict[str, set]
        Maps each node type to the set of excluded ids. Node types without
        excluded ids map to an empty set
    """
    entities = {
        "actor": [(actor, actor.id) for actor in gitdata.get_actors()],
        "commit": [(commit, commit.commit.id) for commit in gitdata.commits],
        "issue": [(issue, issue.issue.number) for issue in gitdata.issues],
        "file": [(file.file, file.file.id) for file in gitdata.modified_files],
    }
    excluded: dict[str, set] = {node_type: set() for node_type in entities}
    for node_filter in filters:
        for entity, entity_id in entities[node_filter.node_type]:
            if entity_id is not None and node_filter.excludes(entity):
                excluded[node_filter.node_type].add(entity_id)
    return excluded


def is_development_commit(commit: CommitDatapoint) -> bool:
    """Whether a commit counts as development activity, i.e. is not a merge
    commit.

    A merge commit carries the line statistics of everything it merges, so
    counting it credits the merger with work that is already attributed to
    the commits on the merged branch. Merge commits stay in the graph as
    nodes, and keep their own features; only aggregates over commits leave
    them out.
    """
    return not commit.is_merge


def development_commits(commit_df: pd.DataFrame) -> pd.DataFrame:
    """Restricts commits to those that count as development activity, see
    `is_development_commit`.

    Parameters
    ----------
    commit_df : pd.DataFrame
        Commits, as returned by `GitData.get_as_dataframe("commits")`
    """
    return commit_df[~commit_df["is_merge"].fillna(False).astype(bool)]
