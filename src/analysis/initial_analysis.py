from __future__ import annotations

import json
from typing import Any

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import (
    InitialAnalysis,
    SamplingDecision,
    AnalysisStatus,
    EventType,
    Frame,
    OCRResult,
)
from src.logger import get_logger

logger = get_logger(__name__)


class InitialAnalyzer:
    """Handles the fast screening pass for stories."""

    def __init__(self, config: StoryZopConfig) -> None:
        self.config = config

    def analyze_story(
        self,
        story_id: str,
        frames: list[Frame],
        ocr_results: list[OCRResult],
        screener_model: Any,
        db: Database,
    ) -> InitialAnalysis | None:
        """Run the initial 4B screening on a story."""
        try:
            # Build OCR context string
            ocr_texts = []
            for r in ocr_results:
                if r.text and (r.confidence is None or r.confidence >= self.config.ocr_confidence_threshold):
                    ocr_texts.append(r.text.strip())
            ocr_context = "\n".join(ocr_texts)

            # Get image paths
            image_paths = [f.file_path for f in frames]

            logger.info("Starting initial analysis for story %s", story_id)

            # Call the screening model
            result_dict = screener_model.screen_story(image_paths, ocr_context)

            # Parse result
            summary = result_dict.get("summary")
            visible_info = result_dict.get("visible_information")
            confidence = result_dict.get("confidence")
            sampling_decision_str = result_dict.get("sampling_decision", "ACCEPT")
            
            try:
                sampling_decision = SamplingDecision(sampling_decision_str.upper())
            except ValueError:
                sampling_decision = SamplingDecision.ACCEPT
                
            priority = result_dict.get("revisit_priority", 5)
            reason = result_dict.get("revisit_reason")

            # Save to database
            analysis = db.save_initial_analysis(
                story_id=story_id,
                model=self.config.initial_model,
                summary=summary,
                visible_information=visible_info,
                confidence=confidence,
                sampling_decision=sampling_decision,
                revisit_priority=priority,
                revisit_reason=reason,
            )

            # Update story status
            db.update_story_status(
                story_id,
                initial_analysis_status=AnalysisStatus.COMPLETED
            )

            # Log event
            db.log_event(story_id, EventType.AI_4B_COMPLETED)

            # If revisit needed, queue it
            if sampling_decision == SamplingDecision.REVISIT:
                logger.info("Story %s flagged for revisit (priority %s)", story_id, priority)
                db.queue_revisit(story_id, priority, reason)
                db.update_story_status(story_id, revisit_status=AnalysisStatus.PENDING) # Use QUEUED from RevisitStatus but let's queue via db methods properly
                # Actually, queue_revisit does log the event, but we should update story revisit status
                from src.database.models import RevisitStatus
                db.update_story_status(story_id, revisit_status=RevisitStatus.QUEUED)

            return analysis

        except Exception as e:
            logger.error("Failed initial analysis for story %s: %s", story_id, e)
            db.update_story_status(story_id, initial_analysis_status=AnalysisStatus.FAILED)
            db.log_event(story_id, EventType.ERROR, {"error": str(e), "stage": "initial_analysis"})
            return None
