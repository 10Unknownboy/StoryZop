"""
Unit tests for the browser automation layer.

All Playwright interactions are mocked — no real browser is launched.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from src.config import StoryZopConfig
from src.browser.session import BrowserSession
from src.browser.instagram import InstagramNavigator
from src.browser.stories import StoryNavigator


@pytest.fixture()
def config():
    return StoryZopConfig()


def _make_locator_mock(count_value: int) -> MagicMock:
    """Create a mock locator whose .count() is an awaitable returning *count_value*."""
    loc = MagicMock()
    loc.count = AsyncMock(return_value=count_value)
    loc.first = MagicMock()
    loc.first.click = AsyncMock()
    return loc


# ── BrowserSession ───────────────────────────────────────────────────────


def test_browser_session_init(config):
    session = BrowserSession(config)
    assert session.config == config


# ── InstagramNavigator ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_instagram_verify_auth_logged_in(config):
    """When home indicator is found, verify_authentication returns True."""
    mock_page = AsyncMock()

    # page.locator("svg[aria-label='Home']").count() -> 1
    def locator_side_effect(selector: str):
        if "Home" in selector:
            return _make_locator_mock(1)
        return _make_locator_mock(0)

    mock_page.locator = MagicMock(side_effect=locator_side_effect)

    nav = InstagramNavigator(mock_page, config)
    assert await nav.verify_authentication() is True


@pytest.mark.asyncio
async def test_instagram_verify_auth_logged_out(config):
    """When login form is found, verify_authentication returns False."""
    mock_page = AsyncMock()

    def locator_side_effect(selector: str):
        if "username" in selector:
            return _make_locator_mock(1)
        return _make_locator_mock(0)

    mock_page.locator = MagicMock(side_effect=locator_side_effect)

    nav = InstagramNavigator(mock_page, config)
    assert await nav.verify_authentication() is False


# ── StoryNavigator ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_story_navigator_get_story_reference(config):
    mock_page = AsyncMock()
    mock_page.url = "https://www.instagram.com/stories/testuser/123456789/"

    nav = StoryNavigator(mock_page, config)
    ref = await nav.get_story_reference()

    assert isinstance(ref, dict)
    assert ref.get("url") == mock_page.url


@pytest.mark.asyncio
async def test_story_detect_type_video(config):
    """When a <video> element is present, detect_story_type returns 'video'."""
    mock_page = AsyncMock()

    def locator_side_effect(selector: str):
        if selector == "video":
            return _make_locator_mock(1)
        return _make_locator_mock(0)

    mock_page.locator = MagicMock(side_effect=locator_side_effect)

    nav = StoryNavigator(mock_page, config)
    assert await nav.detect_story_type() == "video"


@pytest.mark.asyncio
async def test_story_detect_type_photo(config):
    """When only an <img> element is present, detect_story_type returns 'photo'."""
    mock_page = AsyncMock()

    def locator_side_effect(selector: str):
        if "img" in selector:
            return _make_locator_mock(1)
        return _make_locator_mock(0)

    mock_page.locator = MagicMock(side_effect=locator_side_effect)

    nav = StoryNavigator(mock_page, config)
    assert await nav.detect_story_type() == "photo"
