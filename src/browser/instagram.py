from __future__ import annotations

from typing import Any
from playwright.async_api import Page, ElementHandle
from src.config import StoryZopConfig
from src.logger import get_logger

logger = get_logger(__name__)

class InstagramNavigator:
    """Handles navigation and element interaction for Instagram."""

    def __init__(self, page: Page, config: StoryZopConfig) -> None:
        self.page = page
        self.config = config

    async def navigate_to_instagram(self) -> None:
        """Go to instagram.com, wait for page load."""
        try:
            logger.info("Navigating to Instagram...")
            await self.page.goto("https://www.instagram.com/", wait_until="networkidle")
            logger.info("Navigation complete.")
        except Exception as e:
            logger.error(f"Failed to navigate to Instagram: {e}")
            raise

    async def verify_authentication(self) -> bool:
        """Check if logged in by looking for feed indicators vs login form."""
        try:
            logger.info("Verifying authentication status...")
            # Look for feed elements or bottom nav which indicate logged in state
            is_logged_in = await self.page.locator("svg[aria-label='Home']").count() > 0 or \
                           await self.page.locator("a[href='/']").count() > 0
            if is_logged_in:
                logger.info("Successfully authenticated.")
                return True
                
            # Check for login form
            is_logged_out = await self.page.locator("input[name='username']").count() > 0
            if is_logged_out:
                logger.warning("Not authenticated. Login form detected.")
                return False
                
            logger.warning("Could not clearly determine authentication status.")
            return False
        except Exception as e:
            logger.error(f"Error during authentication verification: {e}")
            return False

    async def dismiss_dialogs(self) -> None:
        """Handle common Instagram popup dialogs."""
        try:
            logger.info("Checking for dialogs to dismiss...")
            
            # Dismiss Cookie consent
            cookie_btn = self.page.locator("button:has-text('Allow all cookies'), button:has-text('Accept')")
            if await cookie_btn.count() > 0:
                await cookie_btn.first.click()
                logger.info("Dismissed cookie consent.")
                await self.page.wait_for_timeout(1000)

            # Dismiss "Turn on Notifications" dialog
            notif_btn = self.page.locator("button:has-text('Not Now')")
            if await notif_btn.count() > 0:
                await notif_btn.first.click()
                logger.info("Dismissed notifications dialog.")
                await self.page.wait_for_timeout(1000)

            # Dismiss "Add to Home Screen" or other similar dialogs
            cancel_btn = self.page.locator("button:has-text('Cancel')")
            if await cancel_btn.count() > 0:
                await cancel_btn.first.click()
                logger.info("Dismissed generic cancelable dialog.")
                await self.page.wait_for_timeout(1000)
                
        except Exception as e:
            logger.error(f"Error while dismissing dialogs: {e}")

    async def get_stories_tray(self) -> list[dict[str, Any]]:
        """Find the stories tray element, return list of story items."""
        try:
            logger.info("Extracting stories from tray...")
            # Wait for stories tray to load
            tray_locator = self.page.locator("ul[data-visualcompletion='ignore-dynamic'] li, div[role='menu'] button")
            await tray_locator.first.wait_for(state="visible", timeout=10000)
            
            elements = await tray_locator.element_handles()
            stories = []
            
            for i, element in enumerate(elements):
                text_content = await element.text_content()
                username = text_content.strip() if text_content else f"unknown_{i}"
                stories.append({
                    "username": username.split('\n')[0].strip(), # Get just the username part
                    "element": element,
                    "index": i
                })
                
            logger.info(f"Found {len(stories)} stories in the tray.")
            return stories
        except Exception as e:
            logger.error(f"Failed to get stories tray: {e}")
            return []
