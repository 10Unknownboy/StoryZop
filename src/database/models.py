"""
Domain models and enumerations for StoryZop.

Every entity that is persisted to the database has a corresponding Pydantic
model defined here.  Enums capture the fixed set of valid states.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================


class IdentityStatus(str, enum.Enum):
    """Confidence level for a person's identity link."""

    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    UNVERIFIED = "UNVERIFIED"


class CapturePass(str, enum.Enum):
    """Which capture pass a frame belongs to."""

    INITIAL = "initial"
    REVISIT = "revisit"


class CaptureStatus(str, enum.Enum):
    """Overall capture status for a story."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AnalysisStatus(str, enum.Enum):
    """Status of an analysis stage."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SamplingDecision(str, enum.Enum):
    """Result of the initial screening model's sufficiency check."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVISIT = "REVISIT"


class RevisitStatus(str, enum.Enum):
    """Status of a revisit queue entry."""

    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EventType(str, enum.Enum):
    """Types of processing events logged for each story."""

    STORY_DETECTED = "STORY_DETECTED"
    STORY_OPENED = "STORY_OPENED"
    INITIAL_CAPTURE_STARTED = "INITIAL_CAPTURE_STARTED"
    INITIAL_CAPTURE_COMPLETED = "INITIAL_CAPTURE_COMPLETED"
    OCR_COMPLETED = "OCR_COMPLETED"
    AI_4B_COMPLETED = "AI_4B_COMPLETED"
    REVISIT_REQUESTED = "REVISIT_REQUESTED"
    REVISIT_STARTED = "REVISIT_STARTED"
    REVISIT_COMPLETED = "REVISIT_COMPLETED"
    AI_8B_COMPLETED = "AI_8B_COMPLETED"
    AI_32B_REVIEWED = "AI_32B_REVIEWED"
    STORY_COMPLETED = "STORY_COMPLETED"
    ERROR = "ERROR"


# =============================================================================
# Entity models
# =============================================================================


class Person(BaseModel):
    """A unique person tracked across multiple stories and possible username changes."""

    person_id: str = Field(..., description="Internal stable ID, e.g. P_000001")
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    current_username: str = Field(..., description="Most recently observed username")
    current_display_name: str | None = None
    current_pfp_path: str | None = None
    identity_status: IdentityStatus = Field(
        default=IdentityStatus.UNVERIFIED,
        description=(
            "UNVERIFIED for new records, CONFIRMED after manual verification, "
            "UNCERTAIN if a username-change scenario was detected."
        ),
    )


class Username(BaseModel):
    """Historical record of a username associated with a person."""

    id: int | None = None
    person_id: str
    username: str
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)


class Story(BaseModel):
    """A single Instagram Story belonging to a person."""

    story_id: str = Field(..., description="Internal stable ID, e.g. S_000001")
    person_id: str
    username_at_capture: str
    story_reference: str | None = Field(
        default=None,
        description="JSON blob of navigation/reference information for revisiting.",
    )
    detected_at: datetime = Field(default_factory=datetime.now)
    opened_at: datetime | None = None
    completed_at: datetime | None = None
    story_position: int | None = Field(
        default=None, description="Position in the stories tray."
    )
    capture_status: CaptureStatus = CaptureStatus.PENDING
    initial_analysis_status: AnalysisStatus = AnalysisStatus.PENDING
    revisit_status: RevisitStatus | None = None
    final_analysis_status: AnalysisStatus = AnalysisStatus.PENDING


class Frame(BaseModel):
    """A single captured frame (screenshot) from a story."""

    frame_id: str = Field(..., description="Internal stable ID, e.g. F_000001")
    story_id: str
    capture_pass: CapturePass
    frame_number: int
    captured_at: datetime = Field(default_factory=datetime.now)
    file_path: str
    width: int | None = None
    height: int | None = None


class OCRResult(BaseModel):
    """OCR extraction result for a single frame."""

    ocr_id: str = Field(..., description="Internal stable ID, e.g. O_000001")
    frame_id: str
    text: str
    confidence: float | None = None
    bounding_data: str | None = Field(
        default=None, description="JSON-serialised bounding box data."
    )
    created_at: datetime = Field(default_factory=datetime.now)


class InitialAnalysis(BaseModel):
    """Result of the Qwen3-VL-4B initial screening pass."""

    analysis_id: str = Field(..., description="Internal stable ID, e.g. IA_000001")
    story_id: str
    model: str
    summary: str | None = None
    visible_information: str | None = None
    confidence: float | None = None
    sampling_decision: SamplingDecision | None = None
    revisit_priority: int | None = Field(
        default=None, description="Higher = more urgent."
    )
    revisit_reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class Revisit(BaseModel):
    """An entry in the revisit queue for a story needing additional captures."""

    revisit_id: str = Field(..., description="Internal stable ID, e.g. R_000001")
    story_id: str
    required: bool = True
    priority: int = Field(default=5, description="Higher = more urgent.")
    reason: str | None = None
    queued_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempts: int = 0
    status: RevisitStatus = RevisitStatus.QUEUED


class FinalAnalysis(BaseModel):
    """Result of the Qwen3-VL-8B detailed analysis."""

    analysis_id: str = Field(..., description="Internal stable ID, e.g. FA_000001")
    story_id: str
    model: str
    content_type: str | None = None
    description: str | None = None
    important_information: str | None = None
    visible_text: str | None = None
    people_detected: str | None = Field(
        default=None, description="JSON list of detected people."
    )
    objects_detected: str | None = Field(
        default=None, description="JSON list of detected objects."
    )
    context: str | None = None
    confidence: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class ExpertReview(BaseModel):
    """Result of the optional Qwen3-VL-32B expert review."""

    review_id: str = Field(..., description="Internal stable ID, e.g. ER_000001")
    story_id: str
    model: str
    reason: str | None = None
    analysis: str | None = None
    confidence: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class StoryEvent(BaseModel):
    """Processing-event log entry for a story."""

    event_id: str = Field(..., description="Internal stable ID, e.g. EV_000001")
    story_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: str | None = Field(
        default=None, description="JSON blob of extra event data."
    )
