"""Entry point into gambit-based actor disambiguation for a GitData instance."""

import pandas as pd

from git2kg.datapoint import ActorInfo
from git2kg.datapoint.utils import convert_to_pandas_types
from git2kg.filtering import BotFilter, NodeFilter
from git2kg.gambit import gambit
from git2kg.gitdata import GitData

DISAMBIGUATION_VERSION = 2
"""Identifies the disambiguation that produced a stored actor mapping.

Bump this whenever a change to `gambit`, its threshold or the id assignment
would give a different mapping for the same aliases, so that mappings stored by
an older version are recomputed instead of reused.
"""

ALIAS_COLUMNS = ["alias_name", "alias_email", ActorInfo.GITHUB_ID_COLUMN]


def disambiguated_author_mapping(
    gitdata: GitData, bot_filter: NodeFilter | None = None
) -> pd.DataFrame:
    """Create an actor mapping for all actors interacting in the GitData. Computed using `gambit`

    Bots are disambiguated in a pass of their own, so that no bot alias can be
    merged with a human one. A bot's address names the forge rather than a
    person, and `gambit` compares those against the addresses of people who
    have done the same: the GitHub web-flow identity matches every contributor
    writing from `github@their-own-domain`, and one such merge puts all of them
    into a single actor.

    Separating the two passes only keeps bots from bridging human aliases.
    They keep an actor id of their own and remain nodes of the graph, so which
    of them are drawn stays a matter for the filters.

    Parameters
    ----------
    gitdata : GitData
        The GitData object containing the actors
    bot_filter : NodeFilter, optional
        Decides which aliases are disambiguated as bots, `BotFilter` by default

    Returns
    -------
    pd.DataFrame
        The expanded author mapping, containing each occuring combination of \"alias_name\", \"alias_email\" and \"actor_github_id\" and \"actor_id\"
    """
    bot_filter = BotFilter() if bot_filter is None else bot_filter
    # One list, so that the mask and the frame are built from the same actors
    # in the same order - `get_actors` answers with a set
    actors = list(gitdata.get_actors())
    aliases = _alias_frame(actors)
    is_bot = pd.Series(
        [bot_filter.excludes(actor) for actor in actors], index=aliases.index
    )

    links = _observed_links(gitdata)
    mappings, offset = [], 0
    for part in (aliases[~is_bot], aliases[is_bot]):
        mapping = _merge_on_links(_disambiguate(part), links)
        if mapping.empty:
            continue
        # Each pass numbers from zero, so one part's ids are moved clear of
        # the other's before the two are put back together
        mapping[ActorInfo.ID_COLUMN] += offset
        offset = int(mapping[ActorInfo.ID_COLUMN].max()) + 1
        mappings.append(mapping)

    if not mappings:
        return _canonical_actor_ids(_disambiguate(aliases))
    return _canonical_actor_ids(pd.concat(mappings, ignore_index=True))


def _alias_key(name, email, github_id) -> tuple[str, str, int | None]:
    """The identity an alias has in the alias frame.

    Built from both an `ActorInfo` and a row of the frame, which have been
    through different missing value conventions on the way, so each field is
    normalized before the two are compared.
    """
    return (
        "" if name is None or pd.isna(name) else str(name),
        "" if email is None or pd.isna(email) else str(email),
        None if github_id is None or pd.isna(github_id) else int(github_id),
    )


def _observed_links(gitdata: GitData) -> list[tuple[tuple, tuple]]:
    """Pairs of aliases that GitHub states are the same person.

    A `GithubCommitDatapoint` carries the account GitHub resolved a commit's
    git identity to, so for every commit it covers, that commit's signature
    and that account are two aliases of one person. Nothing else in the mined
    data says so: the git side never sees an account and the GitHub side never
    sees the address a commit was made with.
    """
    records = {record.commit_hash: record for record in gitdata.github_commits}
    links = []
    for commit in gitdata.commits:
        record = records.get(commit.commit.hash)
        if record is None:
            continue
        for signature, account in (
            (commit.author, record.author),
            (commit.committer, record.committer),
        ):
            if signature is None or account is None:
                continue
            links.append(
                (
                    _alias_key(signature.name, signature.email, signature.github_id),
                    _alias_key(account.name, account.email, account.github_id),
                )
            )
    return links


def _merge_on_links(mapping: pd.DataFrame, links) -> pd.DataFrame:
    """Merges the actors of aliases an observed link puts together.

    Applied per part, so a link is only honoured where both of its aliases
    were disambiguated in the same pass. One reaching from a bot into a human
    alias means the two disagree about what a bot is, and is dropped rather
    than undoing the separation.
    """
    if mapping.empty or not links:
        return mapping

    actor_of: dict[tuple, int] = {
        _alias_key(name, email, account): int(actor)
        for name, email, account, actor in zip(
            mapping["alias_name"],
            mapping["alias_email"],
            mapping[ActorInfo.GITHUB_ID_COLUMN],
            mapping[ActorInfo.ID_COLUMN],
            strict=True,
        )
    }
    parent: dict[int, int] = {}

    def find(actor: int) -> int:
        parent.setdefault(actor, actor)
        while parent[actor] != actor:
            parent[actor] = parent[parent[actor]]
            actor = parent[actor]
        return actor

    for left, right in links:
        if left not in actor_of or right not in actor_of:
            continue
        roots = sorted((find(actor_of[left]), find(actor_of[right])))
        # The lower id absorbs the higher, which is how `gambit` merges too
        parent[roots[1]] = roots[0]

    if not parent:
        return mapping
    return mapping.assign(
        **{ActorInfo.ID_COLUMN: [find(actor) for actor in mapping[ActorInfo.ID_COLUMN]]}
    )


def _disambiguate(aliases: pd.DataFrame) -> pd.DataFrame:
    """Runs `gambit` over one part of the alias frame, ids numbered from zero."""
    if aliases.empty:
        return aliases.assign(**{ActorInfo.ID_COLUMN: pd.Series(dtype="int64")})
    # gambit reindexes and adds its comparison columns in place
    mapping = gambit(aliases.copy(), thresh=0.9).drop_duplicates()
    mapping = mapping.rename(columns={"author_id": ActorInfo.ID_COLUMN})
    return mapping.astype({ActorInfo.ID_COLUMN: "int64"})


def alias_frame(gitdata: GitData) -> pd.DataFrame:
    """Collects the actors of a GitData into the one-row-per-alias frame that
    `gambit` runs on.

    Parameters
    ----------
    gitdata : GitData
        The GitData object containing the actors

    Returns
    -------
    pd.DataFrame
        A frame with the columns "alias_name", "alias_email" and
        "actor_github_id"
    """
    return _alias_frame(list(gitdata.get_actors()))


def _alias_frame(actors: list[ActorInfo]) -> pd.DataFrame:
    """Builds the alias frame from actors already collected, so that a caller
    needing both the frame and a per-alias verdict can keep the two aligned."""
    aliases = pd.DataFrame([actor.to_dict() for actor in actors])
    aliases = convert_to_pandas_types(aliases, ActorInfo)
    aliases = aliases.drop(["id", "type"], axis=1)
    aliases = aliases.fillna({"name": "", "email": ""})
    aliases.rename(
        columns={name: f"actor_{name}" for name in aliases.columns}, inplace=True
    )
    aliases.rename(
        columns={"actor_name": "alias_name", "actor_email": "alias_email"},
        inplace=True,
    )
    return aliases


def mapping_matches(mapping: pd.DataFrame | None, gitdata: GitData) -> bool:
    """Checks whether an actor mapping covers every alias occurring in a
    GitData, and can therefore be used instead of disambiguating again.

    A mapping of the whole history covers each of its timeslices, so the
    aliases only have to be contained in it, not to exhaust it. Requiring
    them to match exactly would disambiguate a slice again under ids of its
    own, which the aliases of the whole history no longer agree with.

    Parameters
    ----------
    mapping : pd.DataFrame, optional
        The actor mapping to check, typically one loaded from an archive
    gitdata : GitData
        The GitData the mapping would be applied to

    Returns
    -------
    bool
        True if the mapping covers the GitData's aliases
    """
    if mapping is None or not set(ALIAS_COLUMNS).issubset(mapping.columns):
        return False
    return _alias_rows(alias_frame(gitdata)) <= _alias_rows(mapping)


def _alias_rows(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Reduces the alias columns of a frame to a set of comparable rows.

    Values are normalized to strings, since a mapping that has been through a
    CSV round trip comes back with different dtypes and missing value markers
    than one that was just computed.
    """
    aliases = frame[ALIAS_COLUMNS]
    normalized = {
        "alias_name": aliases["alias_name"].fillna("").astype(str),
        "alias_email": aliases["alias_email"].fillna("").astype(str),
        ActorInfo.GITHUB_ID_COLUMN: aliases[ActorInfo.GITHUB_ID_COLUMN]
        .astype("Int64")
        .astype(str),
    }
    return set(pd.DataFrame(normalized).itertuples(index=False, name=None))


def _canonical_actor_ids(mapping: pd.DataFrame) -> pd.DataFrame:
    """Renumbers actor ids to 0..n-1, ordering actors by their alphabetically
    first alias, and sorts the mapping by the result.

    `gambit` hands its ids out in the row order of its input and drops the
    higher of the two whenever it merges a pair, so its numbering depends on
    the order the aliases arrive in and leaves gaps. Which aliases it groups
    together does not depend on that order, so renumbering by alias makes the
    ids a function of the alias set alone.

    Parameters
    ----------
    mapping : pd.DataFrame
        An actor mapping as returned by `gambit`, one row per alias

    Returns
    -------
    pd.DataFrame
        The mapping with renumbered ids, sorted by id
    """
    by_alias = mapping.sort_values(ALIAS_COLUMNS)[ActorInfo.ID_COLUMN]
    canonical = {
        old_id: new_id for new_id, old_id in enumerate(by_alias.drop_duplicates())
    }
    # gambit leaves the column as object dtype, which makes it unusable as the
    # first key of a multi-column sort.
    mapping = mapping.assign(
        **{
            ActorInfo.ID_COLUMN: mapping[ActorInfo.ID_COLUMN]
            .map(canonical)
            .astype("int64")
        }
    )
    return mapping.sort_values([ActorInfo.ID_COLUMN, *ALIAS_COLUMNS]).reset_index(
        drop=True
    )
