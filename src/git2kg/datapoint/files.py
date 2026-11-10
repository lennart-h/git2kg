from dataclasses import dataclass
from datetime import datetime

from git2kg.datapoint.base import CommitInfo, Datapoint, FileInfo


@dataclass
class FileDatapoint(Datapoint):
    file: FileInfo
    commit: CommitInfo
    timestamp: datetime
    nloc: int
    extension: str
    n_methods: int
    complexity: float
    n_token: int

    def from_field(self):
        return self.file.commit_hash

    @classmethod
    def appearance_key(cls):
        # `timestamp` is the committer date of the commit that touched the
        # file, so a file version enters history together with its commit
        # rather than at the (possibly much older) author date.
        return lambda dp: dp.timestamp
