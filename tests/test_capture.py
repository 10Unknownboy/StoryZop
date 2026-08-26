from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import AsyncMock, MagicMock

from src.config import StoryZopConfig
from src.capture.frame_manager import FrameManager
from src.capture.sampler import StorySampler
from src.database.database import Database
from src.database.models import CapturePass

@pytest.fixture
def config(tmp_path):
    c = StoryZopConfig()
    c.data_dir = tmp_path
    c.initial_max_frames = 5
    return c

@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return Database(db_path)

def test_frame_manager_ensure_directories(config):
    manager = FrameManager(config)
    story_dir = manager.ensure_directories("person123", "story456")
    assert story_dir.exists()
    assert (story_dir / "initial").exists()
    assert (story_dir / "revisit").exists()

def test_frame_manager_get_frame_path(config):
    manager = FrameManager(config)
    path = manager.get_frame_path("person123", "story456", CapturePass.INITIAL, 1)
    assert path.name == "frame_001.png"
    assert "initial" in str(path)

def test_frame_manager_get_frame_dimensions(config, tmp_path):
    # Create small image
    img_path = tmp_path / "test.png"
    img = Image.new('RGB', (10, 10))
    img.save(img_path)
    
    manager = FrameManager(config)
    width, height = manager.get_frame_dimensions(img_path)
    assert width == 10
    assert height == 10

@pytest.mark.asyncio
async def test_story_sampler_initial_capture_video(config, db):
    db.create_tables()
    manager = FrameManager(config)
    sampler = StorySampler(config, manager)
    
    mock_navigator = AsyncMock()
    mock_navigator.detect_story_type.return_value = "video"
    mock_navigator.capture_current_frame.return_value = True
    
    # Mock dimensions to avoid loading real files
    manager.get_frame_dimensions = MagicMock(return_value=(100, 100))
    
    # Set interval very low for fast tests
    config.capture_interval_ms = 1
    
    frames = await sampler.initial_capture(mock_navigator, "story1", "person1", db)
    
    assert len(frames) == config.initial_max_frames
    assert mock_navigator.capture_current_frame.call_count == config.initial_max_frames

@pytest.mark.asyncio
async def test_story_sampler_initial_capture_photo(config, db):
    db.create_tables()
    manager = FrameManager(config)
    sampler = StorySampler(config, manager)
    
    mock_navigator = AsyncMock()
    mock_navigator.detect_story_type.return_value = "photo"
    mock_navigator.capture_current_frame.return_value = True
    
    manager.get_frame_dimensions = MagicMock(return_value=(100, 100))
    
    config.capture_interval_ms = 1
    
    frames = await sampler.initial_capture(mock_navigator, "story2", "person2", db)
    
    # Photos should be limited to 2 captures
    assert len(frames) == 2
    assert mock_navigator.capture_current_frame.call_count == 2
