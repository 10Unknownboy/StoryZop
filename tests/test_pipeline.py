from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import AnalysisStatus
from src.pipeline import StoryPipeline

@pytest.fixture
def config():
    return StoryZopConfig()

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.create_tables()
    return d

@pytest.mark.asyncio
async def test_pipeline_resume_skips_completed(config, db):
    pipeline = StoryPipeline(config, db)
    
    # Pre-populate DB
    person = db.get_or_create_person("test1")
    story_completed = db.create_story(person.person_id, "test1")
    db.update_story_status(story_completed.story_id, 
                           initial_analysis_status=AnalysisStatus.COMPLETED,
                           final_analysis_status=AnalysisStatus.COMPLETED)
                           
    story_pending = db.create_story(person.person_id, "test1")
    db.update_story_status(story_pending.story_id,
                           initial_analysis_status=AnalysisStatus.COMPLETED,
                           final_analysis_status=AnalysisStatus.PENDING)
                           
    mock_nav = AsyncMock()
    mock_nav.get_story_tray_items.return_value = [] # no new stories
    pipeline.set_components(mock_nav, AsyncMock())
    
    mock_analyzer = MagicMock()
    mock_analyzer.analyze_story.return_value = {"confidence": 0.9}
    pipeline.set_models(None, mock_analyzer, None)
    
    stats = await pipeline.resume()
    
    assert stats["completed"] == 1
    assert stats["discovered"] == 0
    
    s = db.get_story(story_pending.story_id)
    assert s.final_analysis_status == AnalysisStatus.COMPLETED

@pytest.mark.asyncio
async def test_pipeline_single_story_failure(config, db):
    pipeline = StoryPipeline(config, db)
    
    mock_nav = AsyncMock()
    mock_nav.get_story_tray_items.return_value = [
        {"username": "fail_user"},
        {"username": "good_user"}
    ]
    
    # Make initial capture fail for first user
    mock_sampler = AsyncMock()
    
    def side_effect(story_id):
        if "fail" in str(story_id):
            raise ValueError("Intentional crash")
        return []
    
    mock_sampler.initial_capture.side_effect = side_effect
    
    pipeline.set_components(mock_nav, mock_sampler)
    
    stats = await pipeline.run()
    
    assert stats["discovered"] == 2
    assert stats["failed"] == 1
