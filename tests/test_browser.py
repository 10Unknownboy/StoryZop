from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.config import StoryZopConfig
from src.browser.session import BrowserSession
from src.browser.instagram import InstagramNavigator
from src.browser.stories import StoryNavigator

@pytest.fixture
def config():
    return StoryZopConfig()

@pytest.mark.asyncio
async def test_browser_session_init(config):
    session = BrowserSession(config)
    assert session.config == config

@pytest.mark.asyncio
async def test_instagram_navigator_verify_authentication_logged_in(config):
    mock_page = MagicMock()
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_page.locator = MagicMock(return_value=mock_locator)
    
    navigator = InstagramNavigator(mock_page, config)
    result = await navigator.verify_authentication()
    
    assert result is True
    assert mock_page.locator.called

@pytest.mark.asyncio
async def test_instagram_navigator_verify_authentication_logged_out(config):
    mock_page = MagicMock()
    
    mock_page.locator = MagicMock(side_effect=lambda s: MagicMock(count=AsyncMock(return_value=1 if "username" in s else 0)))
    
    navigator = InstagramNavigator(mock_page, config)
    result = await navigator.verify_authentication()
    
    assert result is False

@pytest.mark.asyncio
async def test_story_navigator_get_story_reference(config):
    mock_page = MagicMock()
    mock_page.url = "https://www.instagram.com/stories/testuser/123456789/"
    
    navigator = StoryNavigator(mock_page, config)
    ref = await navigator.get_story_reference()
    
    assert ref["username"] == "testuser"
    assert ref["story_id"] == "123456789"

@pytest.mark.asyncio
async def test_story_navigator_detect_story_type_video(config):
    mock_page = MagicMock()
    mock_page.locator = MagicMock(side_effect=lambda s: MagicMock(count=AsyncMock(return_value=1 if "video" in s else 0)))
    
    navigator = StoryNavigator(mock_page, config)
    story_type = await navigator.detect_story_type()
    
    assert story_type == "video"

@pytest.mark.asyncio
async def test_story_navigator_detect_story_type_photo(config):
    mock_page = MagicMock()
    mock_page.locator = MagicMock(side_effect=lambda s: MagicMock(count=AsyncMock(return_value=1 if "img" in s else 0)))
    
    navigator = StoryNavigator(mock_page, config)
    story_type = await navigator.detect_story_type()
    
    assert story_type == "photo"
