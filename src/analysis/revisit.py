from __future__ import annotations

from typing import Any

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import Revisit, RevisitStatus, EventType, SamplingDecision, Frame
from src.logger import get_logger

logger = get_logger(__name__)


class RevisitManager:
    """Manages the revisit queue and execution."""

    def __init__(self, config: StoryZopConfig) -> None:
        self.config = config

    def should_revisit(self, analysis_result: dict) -> bool:
        """Check if sampling_decision is REVISIT."""
        decision = analysis_result.get("sampling_decision", "").upper()
        return decision == SamplingDecision.REVISIT.value

    def get_priority(self, analysis_result: dict) -> int:
        """Extract revisit_priority (default 5)."""
        try:
            return int(analysis_result.get("revisit_priority", 5))
        except (ValueError, TypeError):
            return 5

    def queue_story(self, story_id: str, analysis_result: dict, db: Database) -> Revisit:
        """Queue a story for revisit based on initial analysis result."""
        priority = self.get_priority(analysis_result)
        reason = analysis_result.get("revisit_reason")

        revisit = db.queue_revisit(story_id, priority, reason)
        db.update_story_status(story_id, revisit_status=RevisitStatus.QUEUED)
        return revisit

    def get_sorted_queue(self, db: Database) -> list[Revisit]:
        """Return the pending revisit queue, sorted by priority."""
        return db.get_revisit_queue()

    async def process_revisit(
        self,
        revisit: Revisit,
        story_navigator: Any,
        sampler: Any,
        ocr_engine: Any,
        db: Database,
    ) -> list[Frame]:
        """Execute a revisit for a queued story."""
        story_id = revisit.story_id
        logger.info("Processing revisit for story %s", story_id)
        
        try:
            from datetime import datetime, timezone
            now = datetime.now(tz=timezone.utc)
            db.update_revisit_status(revisit.revisit_id, RevisitStatus.IN_PROGRESS, started_at=now)
            db.update_story_status(story_id, revisit_status=RevisitStatus.IN_PROGRESS)
            db.log_event(story_id, EventType.REVISIT_STARTED)

            story = db.get_story(story_id)
            if not story:
                raise ValueError(f"Story {story_id} not found in database.")

            # Reopen story
            import json
            ref_dict = json.loads(story.story_reference) if story.story_reference else None
            await story_navigator.reopen_story_by_reference(ref_dict)

            # Capture additional frames
            frames = await sampler.revisit_capture(story_id)

            # Run OCR on new frames
            # Assuming ocr_engine.extract_text_for_story extracts and saves to DB, then returns results
            # The spec says "Run OCR on new frames via ocr_engine.extract_text_for_story()"
            await ocr_engine.extract_text_for_story(story_id)

            # Update status to completed
            now_completed = datetime.now(tz=timezone.utc)
            db.update_revisit_status(
                revisit.revisit_id,
                RevisitStatus.COMPLETED,
                completed_at=now_completed,
                increment_attempts=True
            )
            db.update_story_status(story_id, revisit_status=RevisitStatus.COMPLETED)
            db.log_event(story_id, EventType.REVISIT_COMPLETED)

            return frames

        except Exception as e:
            logger.error("Revisit failed for story %s: %s", story_id, e)
            db.update_revisit_status(
                revisit.revisit_id,
                RevisitStatus.FAILED,
                increment_attempts=True
            )
            db.update_story_status(story_id, revisit_status=RevisitStatus.FAILED)
            db.log_event(story_id, EventType.ERROR, {"error": str(e), "stage": "revisit"})
            return []
