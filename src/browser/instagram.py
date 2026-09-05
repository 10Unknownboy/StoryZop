"""
InstagramNavigator — handles navigation and story discovery on Instagram.

Uses multiple fallback selectors and JavaScript evaluation to find stories
since Instagram's DOM changes frequently.
"""

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
            # Extra wait for React hydration
            await self.page.wait_for_timeout(3000)
            logger.info("Navigation complete.")
        except Exception as e:
            logger.error(f"Failed to navigate to Instagram: {e}")
            raise

    async def verify_authentication(self) -> bool:
        """Check if logged in by detecting logged-out indicators first, then feed indicators.

        The bottom navbar (Home, Search, Reels, etc.) exists on BOTH the logged-out
        landing page and the logged-in feed, so it cannot be used for auth verification.
        Instead, we check for definitive logged-out indicators first.
        """
        try:
            logger.info("Verifying authentication status...")
            await self.page.wait_for_timeout(2000)  # Let page settle

            # --- Check for LOGGED-OUT indicators first (these are definitive) ---

            # 1. Login form
            login_form = await self.page.locator("input[name='username']").count()
            if login_form > 0:
                logger.warning("Not authenticated. Login form detected.")
                return False

            # 2. "Log in or sign up" text on landing page
            login_text = await self.page.locator("text='Log in'").count()
            signup_text = await self.page.locator("text='sign up'").count()
            if login_text > 0 and signup_text > 0:
                logger.warning("Not authenticated. Landing page detected ('Log in or sign up').")
                return False

            # 3. "Open Instagram" button on mobile landing
            open_btn = await self.page.locator("text='Open Instagram'").count()
            if open_btn > 0:
                logger.warning("Not authenticated. 'Open Instagram' button detected.")
                return False

            # 4. URL-based check
            url = self.page.url
            if "/accounts/login" in url:
                logger.warning("Not authenticated. Redirected to login page.")
                return False

            # --- Check for LOGGED-IN indicators ---

            # These elements appear ONLY on the authenticated feed, not the landing page
            feed_selectors = [
                "svg[aria-label='New post']",      # Create post icon (feed only)
                "a[href='/direct/inbox/']",         # DM link (feed only)
                "svg[aria-label='Messenger']",      # Messenger icon
                "img[data-testid='user-avatar']",   # User avatar
                "article",                          # Feed posts
            ]

            for sel in feed_selectors:
                try:
                    count = await self.page.locator(sel).count()
                    if count > 0:
                        logger.info("Successfully authenticated (matched: %s).", sel)
                        return True
                except Exception:
                    continue

            # Check page content for feed-like elements via JS
            has_feed = await self.page.evaluate("""
                () => {
                    // Check if there's a stories tray or feed content
                    const articles = document.querySelectorAll('article');
                    const imgs = document.querySelectorAll('img[alt]');
                    // Feed has many images (profile pics, post images)
                    return articles.length > 0 || imgs.length > 5;
                }
            """)
            if has_feed:
                logger.info("Successfully authenticated (feed content detected via JS).")
                return True

            # If we're on instagram.com but didn't match any logged-out indicators,
            # and there are some interactive elements, cautiously assume logged in
            if "instagram.com" in url and "/accounts/" not in url:
                logger.warning("Authentication uncertain — no definitive logged-in or logged-out indicators found.")
                return False

            logger.warning("Could not determine authentication status.")
            return False
        except Exception as e:
            logger.error(f"Error during authentication verification: {e}")
            return False

    async def dismiss_dialogs(self) -> None:
        """Handle common Instagram popup dialogs."""
        try:
            logger.info("Checking for dialogs to dismiss...")

            # Dismiss Cookie consent
            cookie_btn = self.page.locator("button:has-text('Allow all cookies'), button:has-text('Accept'), button:has-text('Allow essential and optional cookies')")
            if await cookie_btn.count() > 0:
                await cookie_btn.first.click()
                logger.info("Dismissed cookie consent.")
                await self.page.wait_for_timeout(1000)

            # Dismiss "Turn on Notifications" dialog
            notif_btn = self.page.locator("button:has-text('Not Now'), button:has-text('Not now')")
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
        """Find stories in the tray using multiple fallback strategies.

        Instagram's DOM changes frequently, so we use a layered approach:
        1. Try role-based / accessible selectors
        2. Try structural selectors for the horizontal story list
        3. Fallback to JavaScript-based DOM traversal
        """
        try:
            logger.info("Extracting stories from tray...")
            await self.page.wait_for_timeout(2000)  # Let the feed load

            stories = []

            # Strategy 1: Find story buttons/links by aria-label or role
            stories = await self._find_stories_by_role()
            if stories:
                logger.info(f"Strategy 1 (role-based): Found {len(stories)} stories.")
                return stories

            # Strategy 2: Find by the horizontal scrollable container near top
            stories = await self._find_stories_by_structure()
            if stories:
                logger.info(f"Strategy 2 (structural): Found {len(stories)} stories.")
                return stories

            # Strategy 3: JavaScript DOM traversal
            stories = await self._find_stories_by_js()
            if stories:
                logger.info(f"Strategy 3 (JS traversal): Found {len(stories)} stories.")
                return stories

            logger.warning("Could not find any stories in the tray with any strategy.")
            return []

        except Exception as e:
            logger.error(f"Failed to get stories tray: {e}")
            return []

    async def _find_stories_by_role(self) -> list[dict[str, Any]]:
        """Strategy 1: Use accessible role-based selectors."""
        stories = []

        # Instagram story items often have role="button" or are clickable elements
        # with text containing usernames, inside a horizontal scroll area
        selectors = [
            # Role-based: buttons with story-related labels
            "button[aria-label*='Story']",
            "button[aria-label*='story']",
            # Canvas/img inside circle indicators (story rings)
            "canvas[height='56'], canvas[height='64'], canvas[height='66']",
            # The story tray items as list items or buttons
            "li button img[alt]",
            "div[role='listbox'] button",
            "div[role='list'] button",
        ]

        for sel in selectors:
            try:
                loc = self.page.locator(sel)
                count = await loc.count()
                if count > 1:  # Need at least 2 (one might be "Your Story")
                    logger.info(f"Selector '{sel}' matched {count} elements.")
                    elements = await loc.element_handles()
                    for i, el in enumerate(elements):
                        username = await self._extract_username_from_element(el)
                        if username and username.lower() not in ("your story", "your story ", ""):
                            stories.append({
                                "username": username,
                                "element": el,
                                "index": i,
                                "position": i,
                            })
                    if stories:
                        return stories
            except Exception as e:
                logger.debug(f"Selector '{sel}' failed: {e}")
                continue

        return stories

    async def _find_stories_by_structure(self) -> list[dict[str, Any]]:
        """Strategy 2: Find stories by structural DOM patterns."""
        stories = []

        # The stories tray is typically a horizontal scrollable container
        # near the top of the page, containing circular profile pictures
        try:
            # Look for a scrollable horizontal container with multiple child items
            containers = await self.page.query_selector_all(
                "div[style*='overflow'][style*='scroll'], "
                "div[style*='overflow-x'], "
                "ul, "
                "div[role='tablist']"
            )

            for container in containers:
                # Check if it's near the top of the page
                bbox = await container.bounding_box()
                if not bbox or bbox["y"] > 300:
                    continue

                # Find clickable children (likely story items)
                children = await container.query_selector_all(
                    "button, a, div[role='button'], li"
                )

                if len(children) >= 2:
                    for i, child in enumerate(children):
                        username = await self._extract_username_from_element(child)
                        if username and username.lower() not in ("your story", ""):
                            stories.append({
                                "username": username,
                                "element": child,
                                "index": i,
                                "position": i,
                            })

                if stories:
                    return stories

        except Exception as e:
            logger.debug(f"Structural search failed: {e}")

        return stories

    async def _find_stories_by_js(self) -> list[dict[str, Any]]:
        """Strategy 3: JavaScript-based DOM traversal to find story items."""
        try:
            # Use JS to find elements that look like story tray items:
            # - Circular images near the top of the page
            # - With gradient ring borders (indicating unwatched stories)
            # - Containing username text
            result = await self.page.evaluate("""
                () => {
                    const stories = [];
                    // Find all images that could be profile pics in the story tray
                    const imgs = document.querySelectorAll('img[alt]');
                    for (const img of imgs) {
                        const rect = img.getBoundingClientRect();
                        // Story tray is near the top, images are small circles
                        if (rect.top < 250 && rect.width < 100 && rect.width > 20) {
                            const alt = img.getAttribute('alt') || '';
                            // Find the closest clickable parent
                            const clickable = img.closest('button, a, [role="button"], li');
                            if (clickable && alt) {
                                // Extract username from alt text
                                let username = alt;
                                // Common patterns: "username's profile picture" or just "username"
                                username = username.replace("'s profile picture", "")
                                                   .replace("'s Profile Photo", "")
                                                   .replace("profile picture", "")
                                                   .trim();
                                if (username && username.toLowerCase() !== 'your story') {
                                    stories.push({
                                        username: username,
                                        index: stories.length,
                                        position: stories.length,
                                        // Store a selector path for re-finding the element
                                        selector_path: clickable.tagName.toLowerCase() +
                                            (clickable.className ? '.' + clickable.className.split(' ').join('.') : '')
                                    });
                                }
                            }
                        }
                    }
                    return stories;
                }
            """)

            if not result:
                return []

            stories = []
            for item in result:
                username = item.get("username", "")
                index = item.get("index", 0)

                # Re-locate the element for clicking
                # Find profile images matching this username
                img_loc = self.page.locator(f"img[alt*='{username}']").first
                try:
                    el = await img_loc.element_handle(timeout=2000)
                    # Get the clickable parent
                    parent = await el.evaluate_handle(
                        "el => el.closest('button, a, [role=\"button\"], li') || el"
                    )
                    stories.append({
                        "username": username,
                        "element": parent.as_element(),
                        "index": index,
                        "position": index,
                    })
                except Exception:
                    # If we can't get element handle, store info for fallback
                    stories.append({
                        "username": username,
                        "element": None,
                        "index": index,
                        "position": index,
                    })

            return stories

        except Exception as e:
            logger.debug(f"JS traversal failed: {e}")
            return []

    async def _extract_username_from_element(self, element: ElementHandle) -> str:
        """Extract a username from a story tray element."""
        try:
            # Try img alt text first (most reliable)
            img = await element.query_selector("img[alt]")
            if img:
                alt = await img.get_attribute("alt") or ""
                username = (
                    alt.replace("'s profile picture", "")
                    .replace("'s Profile Photo", "")
                    .replace("profile picture", "")
                    .strip()
                )
                if username:
                    return username

            # Try aria-label
            aria = await element.get_attribute("aria-label") or ""
            if aria:
                cleaned = aria.replace("Story by", "").replace("story by", "").strip()
                if cleaned:
                    return cleaned

            # Try text content (last resort)
            text = await element.text_content() or ""
            text = text.strip()
            if text:
                # Take first line, first word-like segment
                first_line = text.split("\n")[0].strip()
                if first_line and len(first_line) < 50:
                    return first_line

        except Exception:
            pass

        return ""

    async def open_story(self, story_item: dict[str, Any]) -> bool:
        """Click to open a story from the tray."""
        try:
            username = story_item.get("username", "unknown")
            element = story_item.get("element")

            logger.info(f"Opening story for user: {username}")

            if element:
                await element.click()
            else:
                # Fallback: try to find by username
                loc = self.page.locator(f"img[alt*='{username}']").first
                await loc.click()

            # Wait for story viewer to appear
            await self.page.wait_for_timeout(2000)
            return True
        except Exception as e:
            logger.error(f"Failed to open story for {story_item.get('username', '?')}: {e}")
            return False

    async def close_story(self) -> None:
        """Exit story viewer."""
        try:
            logger.info("Closing story viewer...")
            close_btn = self.page.locator("svg[aria-label='Close'], button[aria-label='Close']")
            if await close_btn.count() > 0:
                await close_btn.first.click()
            else:
                await self.page.keyboard.press("Escape")
            await self.page.wait_for_timeout(500)
        except Exception as e:
            logger.error(f"Failed to close story: {e}")
