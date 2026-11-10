from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

from git2kg.datapoint import ActorInfo, CommitInfo
from git2kg.datapoint.base import Datapoint


@dataclass
class CommitDatapoint(Datapoint):
    """Datapoint representing commit information.

    `committer_date` is when the commit entered the repository and places it,
    and everything derived from it, on the timeline. `author_date` is metadata
    the commit carries about when the change was written - like its message,
    it can be rewritten without the history moving - and is used for features
    only.
    """

    commit: CommitInfo
    author: ActorInfo
    committer: ActorInfo
    author_date: datetime
    author_timezone: int
    committer_date: datetime
    committer_timezone: int
    n_modified_files: int
    message: str
    is_merge: bool
    is_in_main_branch: bool

    is_bugfix: bool
    n_modified_documentation: int
    insertions: int
    deletions: int
    message_polarity: float
    message_subjectivity: float
    commit_parents: List[CommitInfo] = field(default_factory=list)
    branches: Set[str] = field(default_factory=set)
    sig_type: Optional[str] = None
    sig: Optional[str] = None

    def from_field(self) -> str:
        return self.commit.hash

    @classmethod
    def appearance_key(cls):
        # A commit object comes into being when it is committed. author_date is
        # metadata it carries about when the change was originally written,
        # which for a rebased, amended or cherry-picked commit can predate the
        # commit object - and even the branch it sits on - by months.
        return lambda dp: dp.committer_date


@dataclass
class GithubCommitDatapoint(Datapoint):
    """What GitHub knows about a commit that git does not record.

    Git stores an author as the name and address configured on the machine the
    commit was written on. GitHub resolves that address against its own
    verified-email records and reports the account behind it, which is
    evidence no comparison of names and addresses can reconstruct.

    `author` and `committer` are those accounts, not the git signatures the
    `CommitDatapoint` already carries, and are missing where GitHub resolved
    none - an address that was never verified, or one predating the account.

    The commit is named by its hash rather than held as a `CommitInfo`,
    because this is a record *about* a commit and not a second node for it.
    GitHub also lists commits a clone does not carry, which have no commit id
    to be stamped with.
    """

    commit_hash: str
    committer_date: datetime
    author: Optional[ActorInfo] = None
    committer: Optional[ActorInfo] = None

    def from_field(self) -> datetime:
        return self.committer_date

    @classmethod
    def appearance_key(cls):
        # The key `CommitDatapoint` is placed by, so that a slice holds this
        # record beside the commit it describes
        return lambda dp: dp.committer_date


@dataclass
class EditDatapoint(Datapoint):
    """Datapoint representing a single edit/line change in a commit."""

    author: ActorInfo
    prev_author: Optional[ActorInfo]
    commit: CommitInfo
    prev_commit: Optional[CommitInfo]
    new_path: Optional[str]
    old_path: Optional[str]
    line_number: Optional[int]
    new_line: Optional[str]
    edit_type: str
    timestamp: datetime

    def from_field(self):
        return self.commit.hash

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.timestamp
