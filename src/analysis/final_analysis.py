from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import json

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import FinalAnalysis, AnalysisStatus, EventType
from src.logger import get_logger

logger = get_logger(__name__)


class FinalAnalyzer:
    """Handles the detailed 8B analysis and optional 32B expert review."""

    def __init__(self, config: StoryZopConfig) -> None:
        self.config = config

    def analyze_story(
        self,
        story_id: str,
        analyzer_model: Any,
        expert_model: Any | None,
        db: Database,
    ) -> FinalAnalysis | None:
        """Run the final analysis pass on a story."""
        try:
            logger.info("Starting final analysis for story %s", story_id)
            
            # Gather all evidence
            story = db.get_story(story_id)
            if not story:
                raise ValueError(f"Story {story_id} not found.")
                
            person = db.get_person_by_id(story.person_id)
            if not person:
                raise ValueError(f"Person {story.person_id} not found.")

            frames = db.get_frames_for_story(story_id)
            ocr_results = db.get_ocr_for_story(story_id)
            initial_analysis = db.get_initial_analysis(story_id)

            # Build image paths
            image_paths = [f.file_path for f in frames]

            # Build OCR context
            ocr_texts = []
            for r in ocr_results:
                if r.text and (r.confidence is None or r.confidence >= self.config.ocr_confidence_threshold):
                    ocr_texts.append(r.text.strip())
            ocr_context = "\n".join(ocr_texts)

            # Build person info
            person_info = {
                "person_id": person.person_id,
                "current_username": person.current_username,
                "current_display_name": person.current_display_name,
            }

            # Call analyzer_model
            result_dict = analyzer_model.analyze_story(
                image_paths=image_paths,
                person_info=person_info,
                ocr_context=ocr_context,
                initial_analysis=initial_analysis.model_dump() if initial_analysis else None
            )

            confidence = result_dict.get("confidence", 1.0)
            
            # Save 8B result
            analysis = db.save_final_analysis(
                story_id=story_id,
                model=self.config.primary_model,
                content_type=result_dict.get("content_type"),
                description=result_dict.get("description"),
                important_information=result_dict.get("important_information"),
                visible_text=result_dict.get("visible_text"),
                people_detected=result_dict.get("people_detected"),
                objects_detected=result_dict.get("objects_detected"),
                context=result_dict.get("context"),
                confidence=confidence,
            )

            db.log_event(story_id, EventType.AI_8B_COMPLETED)

            # Check for expert review
            if confidence < self.config.expert_review_threshold:
                if expert_model is not None:
                    logger.info("Low confidence (%.2f) on story %s. Triggering expert review.", confidence, story_id)
                    expert_result = expert_model.expert_review(
                        image_paths=image_paths,
                        analysis_result=result_dict,
                        ocr_context=ocr_context
                    )
                    if expert_result is not None:
                        db.save_expert_review(
                            story_id=story_id,
                            model=self.config.expert_model,
                            reason=expert_result.get("reason"),
                            analysis=expert_result.get("analysis"),
                            confidence=expert_result.get("confidence")
                        )
                        db.log_event(story_id, EventType.AI_32B_REVIEWED)
                else:
                    logger.warning("Low confidence on story %s, but expert model is unavailable. Retaining 8B result.", story_id)

            # Update story status
            now = datetime.now(tz=timezone.utc)
            db.update_story_status(
                story_id,
                final_analysis_status=AnalysisStatus.COMPLETED,
                completed_at=now
            )
            db.log_event(story_id, EventType.STORY_COMPLETED)

            return analysis

        except Exception as e:
            logger.error("Final analysis failed for story %s: %s", story_id, e)
            db.update_story_status(story_id, final_analysis_status=AnalysisStatus.FAILED)
            db.log_event(story_id, EventType.ERROR, {"error": str(e), "stage": "final_analysis"})
            return None
