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
        """Load cookies from exported JSON file."""
        if self._context is None:
            raise RuntimeError("Context not initialized. Call launch() first.")
        try:
            path = Path(cookies_json_path)
            if not path.exists():
                logger.warning("Cookies file not found. Skipping cookie load.")
                return
            
            logger.info("Loading cookies from file...")
            with open(path, "r", encoding="utf-8") as f:
                cookies = json.load(f)
                
            await self._context.add_cookies(cookies)
            logger.info(f"Successfully loaded {len(cookies)} cookies.")
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            raise

    async def load_sessionid(self, sessionid: str) -> None:
        """Log into Instagram by injecting sessionid and required companion cookies.

        Instagram requires multiple cookies to maintain a valid session.
        The sessionid alone is not sufficient — we also need to set the
        domain and navigate to Instagram first so the cookies are accepted.
        """
        if self._context is None:
            raise RuntimeError("Context not initialized. Call launch() first.")
        try:
            logger.info("Injecting Instagram authentication cookies...")

            # Navigate to Instagram first so cookies are set on the correct domain
            await self._page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
            await self._page.wait_for_timeout(2000)

            # Set the essential cookies
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
                {
                    "name": "ig_did",
                    "value": "",  # Will be auto-populated
                    "domain": ".instagram.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": False,
                    "sameSite": "Lax",
                },
            ]
            await self._context.add_cookies(cookies)

            # Reload the page so Instagram recognizes the session
            await self._page.reload(wait_until="networkidle")
            await self._page.wait_for_timeout(3000)

            logger.info("Successfully injected session cookies and reloaded page.")
        except Exception as e:
            logger.error(f"Failed to inject sessionid: {e}")
            raise

    async def load_state(self, state_dir: str | Path) -> None:
        """Load Playwright persistent state."""
        logger.info("Loading state is not fully supported in non-persistent context without relaunching.")
        pass

    async def save_state(self, state_dir: str | Path) -> None:
        """Persist browser state for reuse."""
        if self._context is None:
            raise RuntimeError("Context not initialized.")
        try:
            path = Path(state_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            state = await self._context.storage_state(path=str(path))
            logger.info("Successfully saved browser state.")
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
