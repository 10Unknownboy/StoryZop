from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Playwright, BrowserContext, Page
from src.config import StoryZopConfig
from src.logger import get_logger

logger = get_logger(__name__)

class BrowserSession:
    """Async Playwright browser session manager."""

    def __init__(self, config: StoryZopConfig) -> None:
        self.config = config
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session not launched. Call launch() first.")
        return self._page

    async def launch(self) -> None:
        """Start headless Chromium with mobile viewport and custom user-agent."""
        try:
            logger.info("Launching browser session...")
            self._playwright = await async_playwright().start()
            browser = await self._playwright.chromium.launch(
                headless=self.config.headless
            )
            self._context = await browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                user_agent=self.config.user_agent,
                is_mobile=True,
                has_touch=True,
            )
            self._context.set_default_timeout(self.config.browser_timeout_ms)
            self._page = await self._context.new_page()
            logger.info("Browser session launched successfully.")
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            raise

    async def load_cookies(self, cookies_json_path: str | Path) -> None:
        """Load cookies from exported JSON file (e.g., data/session.json).

        This is the RECOMMENDED way to authenticate. The JSON file should
        contain a list of cookie dicts with at minimum: name, value, domain.
        Use extract_cookies.py to create this file from your browser.
        """
        if self._context is None:
            raise RuntimeError("Context not initialized. Call launch() first.")
        try:
            path = Path(cookies_json_path)
            if not path.exists():
                logger.warning(f"Cookies file not found: {path}")
                return

            logger.info(f"Loading cookies from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                cookies = json.load(f)

            # Ensure cookies have required fields for Playwright
            cleaned = []
            for c in cookies:
                cookie = {
                    "name": c["name"],
                    "value": c["value"],
                    "domain": c.get("domain", ".instagram.com"),
                    "path": c.get("path", "/"),
                }
                # Optional fields
                if "secure" in c:
                    cookie["secure"] = c["secure"]
                if "httpOnly" in c:
                    cookie["httpOnly"] = c["httpOnly"]
                if "sameSite" in c:
                    cookie["sameSite"] = c["sameSite"]
                cleaned.append(cookie)

            await self._context.add_cookies(cleaned)
            logger.info(f"Successfully loaded {len(cleaned)} cookies.")

            # Log cookie names (not values!) for debugging
            names = [c["name"] for c in cleaned]
            logger.info(f"Cookie names: {names}")

        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            raise

    async def load_sessionid(self, sessionid: str) -> None:
        """Log into Instagram using just a sessionid.

        NOTE: This is a FALLBACK method. Instagram often requires multiple
        cookies (csrftoken, ds_user_id, mid, etc.) for a valid session.
        Prefer load_cookies() with a full cookie export from extract_cookies.py.
        """
        if self._context is None:
            raise RuntimeError("Context not initialized. Call launch() first.")
        try:
            logger.info("Injecting sessionid cookie...")

            cookies = [
                {
                    "name": "sessionid",
                    "value": sessionid,
                    "domain": ".instagram.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": True,
                    "sameSite": "Lax",
                },
            ]
            await self._context.add_cookies(cookies)
            logger.info("Injected sessionid cookie.")
        except Exception as e:
            logger.error(f"Failed to inject sessionid: {e}")
            raise

    async def navigate_and_verify(self) -> bool:
        """Navigate to Instagram and return True if authenticated.

        Call this AFTER loading cookies. It navigates, waits for
        the page to settle, and checks if the feed loaded.
        """
        if self._page is None:
            raise RuntimeError("Browser not launched.")

        logger.info("Navigating to Instagram...")
        await self._page.goto("https://www.instagram.com/", wait_until="networkidle")
        await self._page.wait_for_timeout(3000)

        # Check if we're on the logged-out page
        url = self._page.url
        open_btn = await self._page.locator("text='Open Instagram'").count()
        login_text = await self._page.locator("text='Log in'").count()

        if open_btn > 0 or "/accounts/login" in url:
            logger.warning("Not authenticated after navigation.")
            return False

        logger.info("Navigation complete. Page appears authenticated.")
        return True

    async def load_state(self, state_dir: str | Path) -> None:
        """Load Playwright persistent state."""
        logger.info("load_state is a no-op in non-persistent context.")

    async def save_state(self, state_dir: str | Path) -> None:
        """Persist browser state for reuse."""
        if self._context is None:
            raise RuntimeError("Context not initialized.")
        try:
            path = Path(state_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(path))
            logger.info(f"Successfully saved browser state to {path}.")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def close(self) -> None:
        """Cleanup browser and playwright."""
        logger.info("Closing browser session...")
        try:
            if self._context:
                await self._context.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")
        finally:
            self._page = None
            self._context = None
            self._playwright = None

    async def __aenter__(self) -> BrowserSession:
        await self.launch()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
