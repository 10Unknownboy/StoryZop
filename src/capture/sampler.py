from __future__ import annotations

import asyncio
from typing import Any

from src.capture.frame_manager import FrameManager
from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import CapturePass, EventType, Frame
from src.logger import get_logger

logger = get_logger(__name__)


class StorySampler:
    """Samples frames from a story and manages capture timing."""

    def __init__(self, config: StoryZopConfig, frame_manager: FrameManager):
        self.config = config
        self.frame_manager = frame_manager

    async def initial_capture(
        self, story_navigator: Any, story_id: str, person_id: str, db: Database
    ) -> list[Frame]:
        """Capture the initial set of frames for a story."""
        db.log_event(story_id, EventType.INITIAL_CAPTURE_STARTED)
        frames: list[Frame] = []

        # Try to detect if photo (single static) vs video
        story_type = await story_navigator.detect_story_type()
        is_photo = story_type == "photo"
        max_frames = 2 if is_photo else self.config.initial_max_frames

        for i in range(1, max_frames + 1):
            frame_path = self.frame_manager.get_frame_path(
                person_id, story_id, CapturePass.INITIAL, i
            )
            success = await story_navigator.capture_current_frame(frame_path)
            if not success:
                logger.warning("Capture failed for story %s frame %d", story_id, i)
                break

            width, height = self.frame_manager.get_frame_dimensions(frame_path)
            frame = db.save_frame(
                story_id=story_id,
                capture_pass=CapturePass.INITIAL,
                frame_number=i,
                file_path=str(frame_path),
                width=width,
                height=height,
            )
            frames.append(frame)

            if i < max_frames:
                await asyncio.sleep(self.config.capture_interval_ms / 1000.0)

        db.log_event(story_id, EventType.INITIAL_CAPTURE_COMPLETED)
        return frames

    async def revisit_capture(
        self,
        story_navigator: Any,
        story_id: str,
        person_id: str,
        db: Database,
        reason: str | None = None,
    ) -> list[Frame]:
        """Perform a revisit capture pass for a story."""
        db.log_event(story_id, EventType.REVISIT_STARTED, data={"reason": reason})
        frames: list[Frame] = []

        story_type = await story_navigator.detect_story_type()
        is_photo = story_type == "photo"
        max_frames = 2 if is_photo else self.config.revisit_max_frames

        interval = self.config.capture_interval_ms / 1000.0
        is_text_incomplete = reason and "text incomplete" in reason.lower()
        if is_text_incomplete:
            interval *= 1.5

        for i in range(1, max_frames + 1):
            frame_path = self.frame_manager.get_frame_path(
                person_id, story_id, CapturePass.REVISIT, i
            )
            success = await story_navigator.capture_current_frame(frame_path)
            if not success:
                logger.warning("Revisit capture failed for story %s frame %d", story_id, i)
                break

            width, height = self.frame_manager.get_frame_dimensions(frame_path)
            frame = db.save_frame(
                story_id=story_id,
                capture_pass=CapturePass.REVISIT,
                frame_number=i,
                file_path=str(frame_path),
                width=width,
                height=height,
            )
            frames.append(frame)

            if i < max_frames:
                if is_text_incomplete:
                    await story_navigator.pause_story()
                await asyncio.sleep(interval)
                if is_text_incomplete:
                    await story_navigator.resume_story()

        db.log_event(story_id, EventType.REVISIT_COMPLETED)
        return frames
