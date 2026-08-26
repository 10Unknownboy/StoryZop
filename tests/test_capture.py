"""
Tests for the capture system (FrameManager, StorySampler).

Uses temp directories and mocked story navigators.
"""

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


@pytest.fixture()
def config(tmp_path):
    return StoryZopConfig(data_dir=tmp_path)


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


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
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)

    manager = FrameManager(config)
    width, height = manager.get_frame_dimensions(img_path)
    assert width == 10
    assert height == 10


@pytest.mark.asyncio
async def test_story_sampler_initial_capture_video(config, db):
    """Video stories should capture up to initial_max_frames."""
    # Create person + story in DB (required for foreign keys)
    person = db.create_person("test_person")
    story = db.create_story(person.person_id, "test_person")

    manager = FrameManager(config)
    sampler = StorySampler(config, manager)

    mock_navigator = AsyncMock()
    mock_navigator.detect_story_type.return_value = "video"
    mock_navigator.capture_current_frame.return_value = True

    # Mock dimensions to avoid needing real image files
    manager.get_frame_dimensions = MagicMock(return_value=(100, 100))

    config.capture_interval_ms = 1  # fast for tests

    frames = await sampler.initial_capture(
        mock_navigator, story.story_id, person.person_id, db
    )

    assert len(frames) == config.initial_max_frames
    assert mock_navigator.capture_current_frame.call_count == config.initial_max_frames


@pytest.mark.asyncio
async def test_story_sampler_initial_capture_photo(config, db):
    """Photo stories should be limited to 2 captures."""
    person = db.create_person("test_person2")
    story = db.create_story(person.person_id, "test_person2")

    manager = FrameManager(config)
    sampler = StorySampler(config, manager)

    mock_navigator = AsyncMock()
    mock_navigator.detect_story_type.return_value = "photo"
    mock_navigator.capture_current_frame.return_value = True

    manager.get_frame_dimensions = MagicMock(return_value=(100, 100))

    config.capture_interval_ms = 1

    frames = await sampler.initial_capture(
        mock_navigator, story.story_id, person.person_id, db
    )

    assert len(frames) == 2
    assert mock_navigator.capture_current_frame.call_count == 2
