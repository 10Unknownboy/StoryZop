from __future__ import annotations

from typing import Any
from pathlib import Path
from playwright.async_api import Page
from src.config import StoryZopConfig
from src.logger import get_logger

logger = get_logger(__name__)

class StoryNavigator:
    """Handles interaction with the Instagram story viewer."""

    def __init__(self, page: Page, config: StoryZopConfig) -> None:
        self.page = page
        self.config = config

    async def open_story(self, story_item: dict[str, Any]) -> bool:
        """Click to open a story from the tray."""
        try:
            username = story_item.get("username", "unknown")
            element = story_item.get("element")
            if not element:
                logger.error("No element provided in story_item.")
                return False

            logger.info(f"Opening story for user: {username}")
            await element.click()
            # Wait for story view to appear
            await self.wait_for_story_load()
            return True
        except Exception as e:
            logger.error(f"Failed to open story: {e}")
            return False

    async def capture_current_frame(self, output_path: str) -> str:
        """Screenshot the current story view."""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Capturing frame to {output_path}")
            await self.page.screenshot(path=str(path))
            return str(path)
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return ""

    async def advance_story(self) -> bool:
        """Tap/click to advance to next story slide."""
        try:
            logger.info("Advancing to next story...")
            # Click on the right side of the screen
            width = self.config.viewport_width
            height = self.config.viewport_height
            await self.page.mouse.click(width * 0.8, height * 0.5)
            await self.page.wait_for_timeout(500) # Give it a moment to load
            return True
        except Exception as e:
            logger.error(f"Failed to advance story: {e}")
            return False

    async def close_story(self) -> None:
        """Exit story viewer."""
        try:
            logger.info("Closing story viewer...")
            # Look for close button or press Escape
            close_btn = self.page.locator("svg[aria-label='Close']")
            if await close_btn.count() > 0:
                await close_btn.first.click()
            else:
                await self.page.keyboard.press("Escape")
            await self.page.wait_for_timeout(500)
        except Exception as e:
            logger.error(f"Failed to close story: {e}")

    async def get_story_reference(self) -> dict[str, Any]:
        """Extract available reference info suitable for JSON serialization."""
        try:
            url = self.page.url
            # We can extract username from URL if it's a standard story URL
            # e.g., https://www.instagram.com/stories/username/123456789/
            parts = url.split('/')
            username = parts[4] if len(parts) > 4 else None
            story_id = parts[5] if len(parts) > 5 else None
            
            return {
                "url": url,
                "username": username,
                "story_id": story_id,
            }
        except Exception as e:
            logger.error(f"Failed to get story reference: {e}")
            return {"url": self.page.url}

    async def reopen_story_by_reference(self, reference: dict[str, Any]) -> bool:
        """Navigate back to a specific story using stored reference."""
        try:
            url = reference.get("url")
            if not url:
                logger.error("No URL found in story reference.")
                return False
                
            logger.info(f"Reopening story at URL: {url}")
            await self.page.goto(url, wait_until="networkidle")
            await self.wait_for_story_load()
            return True
        except Exception as e:
            logger.error(f"Failed to reopen story: {e}")
            return False

    async def wait_for_story_load(self, timeout_ms: int = 5000) -> bool:
        """Wait for story content to be visible."""
        try:
            # Look for typical story container elements or loading spinners
            await self.page.wait_for_selector("header", timeout=timeout_ms)
            await self.page.wait_for_timeout(500) # Small buffer
            return True
        except Exception as e:
            logger.warning(f"Story load wait timed out: {e}")
            return False

    async def detect_story_type(self) -> str:
        """Try to detect if it's 'photo' or 'video'."""
        try:
            # Check for video tag within the story section
            has_video = await self.page.locator("video").count() > 0
            if has_video:
                return "video"
            
            has_img = await self.page.locator("img[decoding='sync']").count() > 0
            if has_img:
                return "photo"
                
            return "unknown"
        except Exception as e:
            logger.error(f"Error detecting story type: {e}")
            return "unknown"

    async def pause_story(self) -> None:
        """Pause story playback (hold/long-press)."""
        try:
            logger.info("Pausing story...")
            width = self.config.viewport_width
            height = self.config.viewport_height
            await self.page.mouse.move(width / 2, height / 2)
            await self.page.mouse.down()
        except Exception as e:
            logger.error(f"Failed to pause story: {e}")

    async def resume_story(self) -> None:
        """Resume story playback."""
        try:
            logger.info("Resuming story...")
            await self.page.mouse.up()
        except Exception as e:
            logger.error(f"Failed to resume story: {e}")
