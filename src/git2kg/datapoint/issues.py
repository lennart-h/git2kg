from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from git2kg.datapoint.base import ActorInfo, Datapoint, IssueInfo


@dataclass
class IssueDatapoint(Datapoint):
    """Datapoint representing a GitHub issue."""

    issue: IssueInfo
    id: int
    body: Optional[str]

    mentioned_actors: List[ActorInfo]
    sentiment_polarity: float
    sentiment_subjectivity: float
    created_at: datetime
    updated_at: datetime
    author: ActorInfo
    assignees: List[ActorInfo]
    closed_at: Optional[datetime]
    is_pull: bool
    is_merged: bool
    merged_at: Optional[datetime]
    merged_by: Optional[ActorInfo]
    state: str

    def from_field(self) -> datetime:
        return self.updated_at

    def as_issue_comment(self) -> "IssueCommentDatapoint":
        return IssueCommentDatapoint(
            issue=self.issue,
            id=self.id,
            author=self.author,
            created_at=self.created_at,
            updated_at=self.updated_at,
            body=self.body,
            mentioned_actors=self.mentioned_actors,
            sentiment_polarity=self.sentiment_polarity,
            sentiment_subjectivity=self.sentiment_subjectivity,
            n_reactions=0,
            upvotes=0,
            downvotes=0,
            laugh=0,
            hooray=0,
            confused=0,
            heart=0,
            rocket=0,
            eyes=0,
        )

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.created_at

    @classmethod
    def mining_key(cls):
        # An issue is mined again whenever its state changed, so mining
        # resumes from the newest update rather than the newest creation.
        return lambda dp: dp.updated_at


@dataclass
class IssueCommentDatapoint(Datapoint):
    """Datapoint representing a GitHub issue comment."""

    issue: IssueInfo
    id: int
    author: ActorInfo
    created_at: datetime
    updated_at: datetime
    body: Optional[str]
    mentioned_actors: List[ActorInfo]

    sentiment_polarity: float
    sentiment_subjectivity: float
    n_reactions: int
    upvotes: int
    downvotes: int
    laugh: int
    hooray: int
    confused: int
    heart: int
    rocket: int
    eyes: int

    def from_field(self) -> datetime:
        return self.updated_at

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.created_at

    @classmethod
    def mining_key(cls):
        # As for IssueDatapoint: a comment is mined again when it is edited.
        return lambda dp: dp.updated_at


@dataclass
class IssueEventDatapoint(Datapoint):
    """Datapoint representing a supported GitHub issue event."""

    event_type: str
    issue: IssueInfo
    requester: Optional[ActorInfo]
    req_reviewer: Optional[ActorInfo]
    created_at: datetime
    assignee: Optional[ActorInfo]
    assigner: Optional[ActorInfo]
    actor: Optional[ActorInfo]

    def from_field(self) -> datetime:
        return self.created_at

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.created_at


@dataclass
class IssueTimelineEventDatapoint(Datapoint):
    """Datapoint representing an issue timeline event."""

    event_type: str
    issue: IssueInfo
    created_at: datetime
    actor: Optional[ActorInfo]
    source_issue: Optional[IssueInfo]

    def from_field(self) -> datetime:
        return self.created_at

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.created_at


@dataclass
class ReviewDatapoint(Datapoint):
    """Datapoint representing a GitHub pull request review."""

    id: int
    pull: IssueInfo
    body: Optional[str]
    sentiment_polarity: float
    sentiment_subjectivity: float
    submitted_at: datetime
    author: ActorInfo
    review_state: str

    def from_field(self) -> datetime:
        return self.submitted_at

    @classmethod
    def appearance_key(cls):
        return lambda dp: dp.submitted_at
