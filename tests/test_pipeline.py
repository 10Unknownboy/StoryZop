"""
Tests for the StoryPipeline orchestrator.

Uses mocked browser, models, and OCR. Database is real (tmp_path).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import AnalysisStatus, CaptureStatus
from src.pipeline import StoryPipeline


@pytest.fixture()
def config():
    return StoryZopConfig()


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


@pytest.mark.asyncio
async def test_pipeline_resume_skips_completed(config, db):
    """Pipeline final-analysis pass should skip already-completed stories."""
    pipeline = StoryPipeline(config, db)

    # Pre-populate DB
    person = db.get_or_create_person("test1")

    story_completed = db.create_story(person.person_id, "test1")
    db.update_story_status(
        story_completed.story_id,
        capture_status=CaptureStatus.COMPLETED,
        initial_analysis_status=AnalysisStatus.COMPLETED,
        final_analysis_status=AnalysisStatus.COMPLETED,
    )

    story_pending = db.create_story(person.person_id, "test1")
    db.update_story_status(
        story_pending.story_id,
        capture_status=CaptureStatus.COMPLETED,
        initial_analysis_status=AnalysisStatus.COMPLETED,
        final_analysis_status=AnalysisStatus.PENDING,
    )

    mock_nav = AsyncMock()
    mock_nav.get_stories_tray = AsyncMock(return_value=[])
    mock_sampler = AsyncMock()
    pipeline.set_components(mock_nav, mock_sampler)

    mock_analyzer = MagicMock()
    mock_analyzer.analyze_story.return_value = {
        "description": "Test",
        "confidence": 0.9,
    }
    pipeline.set_models(None, mock_analyzer, None)

    stats = await pipeline.resume()

    # The pending story should have been finalized
    assert stats["completed"] == 1
    assert stats["discovered"] == 0

    s = db.get_story(story_pending.story_id)
    assert s.final_analysis_status == AnalysisStatus.COMPLETED


@pytest.mark.asyncio
async def test_pipeline_discovery_creates_stories(config, db):
    """Pipeline should discover stories and create DB records."""
    pipeline = StoryPipeline(config, db)

    mock_nav = AsyncMock()
    mock_nav.get_stories_tray = AsyncMock(
        return_value=[
            {"username": "user_a", "index": 0},
            {"username": "user_b", "index": 1},
        ]
    )
    mock_nav.open_story = AsyncMock(return_value=True)
    mock_nav.close_story = AsyncMock()
    mock_nav.capture_current_frame = AsyncMock(return_value=True)
    mock_nav.detect_story_type = AsyncMock(return_value="photo")

    mock_sampler = AsyncMock()
    mock_sampler.initial_capture = AsyncMock(return_value=[])

    pipeline.set_components(mock_nav, mock_sampler)
    pipeline.set_models(None, None, None)

    stats = await pipeline.run()

    assert stats["discovered"] == 2

    # Both persons should exist
    p_a = db.get_person_by_username("user_a")
    p_b = db.get_person_by_username("user_b")
    assert p_a is not None
    assert p_b is not None

    # Stories should be created
    stories_a = db.get_stories_for_person(p_a.person_id)
    stories_b = db.get_stories_for_person(p_b.person_id)
    assert len(stories_a) == 1
    assert len(stories_b) == 1


@pytest.mark.asyncio
async def test_pipeline_handles_error_gracefully(config, db):
    """A single story failure should not crash the entire pipeline."""
    pipeline = StoryPipeline(config, db)

    mock_nav = AsyncMock()
    mock_nav.get_stories_tray = AsyncMock(
        return_value=[
            {"username": "fail_user", "index": 0},
            {"username": "good_user", "index": 1},
        ]
    )

    async def open_story_side_effect(item):
        if item.get("username") == "fail_user":
            raise RuntimeError("Simulated failure")
        return True

    mock_nav.open_story = AsyncMock(side_effect=open_story_side_effect)
    mock_nav.close_story = AsyncMock()
    mock_nav.capture_current_frame = AsyncMock(return_value=True)
    mock_nav.detect_story_type = AsyncMock(return_value="photo")

    mock_sampler = AsyncMock()
    mock_sampler.initial_capture = AsyncMock(return_value=[])

    pipeline.set_components(mock_nav, mock_sampler)
    pipeline.set_models(None, None, None)

    stats = await pipeline.run()

    assert stats["discovered"] == 2
    # The first story should have failed, the second should proceed
    assert stats["failed"] >= 1
