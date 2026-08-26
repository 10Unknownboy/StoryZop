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
    mock_page = AsyncMock()
    mock_locator = AsyncMock()
    mock_locator.count.return_value = 1
    mock_page.locator.return_value = mock_locator
    
    navigator = InstagramNavigator(mock_page, config)
    result = await navigator.verify_authentication()
    
    assert result is True
    # We should have checked locator for Home or /
    assert mock_page.locator.called

@pytest.mark.asyncio
async def test_instagram_navigator_verify_authentication_logged_out(config):
    mock_page = AsyncMock()
    
    async def mock_count_responses(selector):
        if "Home" in selector or "href='/'" in selector:
            m = AsyncMock()
            m.count.return_value = 0
            return m
        if "username" in selector:
            m = AsyncMock()
            m.count.return_value = 1
            return m
        m = AsyncMock()
        m.count.return_value = 0
        return m

    # Mock locator to return something that returns 0 or 1 based on selector
    mock_page.locator.side_effect = lambda s: MagicMock(count=AsyncMock(return_value=1 if "username" in s else 0))
    
    navigator = InstagramNavigator(mock_page, config)
    result = await navigator.verify_authentication()
    
    assert result is False

@pytest.mark.asyncio
async def test_story_navigator_get_story_reference(config):
    mock_page = AsyncMock()
    mock_page.url = "https://www.instagram.com/stories/testuser/123456789/"
    
    navigator = StoryNavigator(mock_page, config)
    ref = await navigator.get_story_reference()
    
    assert ref["username"] == "testuser"
    assert ref["story_id"] == "123456789"

@pytest.mark.asyncio
async def test_story_navigator_detect_story_type_video(config):
    mock_page = AsyncMock()
    
    # Mock locator so that locator("video").count() returns 1
    mock_page.locator.side_effect = lambda s: MagicMock(count=AsyncMock(return_value=1 if "video" in s else 0))
    
    navigator = StoryNavigator(mock_page, config)
    story_type = await navigator.detect_story_type()
    
    assert story_type == "video"

@pytest.mark.asyncio
async def test_story_navigator_detect_story_type_photo(config):
    mock_page = AsyncMock()
    
    # Mock locator so that locator("video").count() returns 0, and img returns 1
    mock_page.locator.side_effect = lambda s: MagicMock(count=AsyncMock(return_value=1 if "img" in s else 0))
    
    navigator = StoryNavigator(mock_page, config)
    story_type = await navigator.detect_story_type()
    
    assert story_type == "photo"
