"""
StoryPipeline — end-to-end orchestrator for the StoryZop analysis system.

Coordinates: browser → capture → OCR → 4B screening → revisit → 8B analysis → 32B expert → report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import AnalysisStatus, RevisitStatus, CapturePass
from src.analysis.initial_analysis import InitialAnalyzer
from src.analysis.revisit import RevisitManager
from src.analysis.final_analysis import FinalAnalyzer
from src.analysis.report import ReportGenerator
from src.logger import get_logger

logger = get_logger(__name__)


class StoryPipeline:
    """The main orchestrator for the StoryZop analysis pipeline."""

    def __init__(self, config: StoryZopConfig, db: Database) -> None:
        self.config = config
        self.db = db

        # Injected components
        self.browser_session: Any = None
        self.instagram_nav: Any = None   # InstagramNavigator
        self.story_nav: Any = None       # StoryNavigator
        self.sampler: Any = None         # StorySampler
        self.frame_manager: Any = None   # FrameManager

        # AI models
        self.screener: Any = None        # Qwen4BScreener
        self.analyzer: Any = None        # Qwen8BAnalyzer
        self.expert: Any = None          # Qwen32BExpert
        self.ocr_engine: Any = None      # OCREngine

        # Internal analysis helpers
        self.initial_analyzer = InitialAnalyzer(config)
        self.revisit_manager = RevisitManager(config)
        self.final_analyzer = FinalAnalyzer(config)
        self.report_generator = ReportGenerator(db)

    # ── Dependency injection ─────────────────────────────────────────────

    def set_browser(self, session: Any) -> None:
        """Inject browser session."""
        self.browser_session = session

    def set_navigators(self, instagram_nav: Any, story_nav: Any) -> None:
        """Inject both navigator objects."""
        self.instagram_nav = instagram_nav
        self.story_nav = story_nav

    def set_components(self, navigator: Any, sampler: Any) -> None:
        """Legacy injection: navigator = InstagramNavigator, sampler = StorySampler.

        Also sets story_nav = navigator for backward compat with tests.
        """
        self.instagram_nav = navigator
        self.sampler = sampler

    def set_sampler(self, sampler: Any, frame_manager: Any | None = None) -> None:
        """Inject capture components."""
        self.sampler = sampler
        self.frame_manager = frame_manager

    def set_models(
        self,
        screener: Any | None,
        analyzer: Any | None,
        expert: Any | None = None,
    ) -> None:
        """Inject the initialized AI models."""
        self.screener = screener
        self.analyzer = analyzer
        self.expert = expert

    def set_ocr(self, ocr_engine: Any) -> None:
        """Inject OCR engine."""
        self.ocr_engine = ocr_engine

    # ── Pipeline execution ───────────────────────────────────────────────

    async def run(self) -> dict:
        """Run the full end-to-end pipeline.

        Flow: discover → capture → OCR → 4B screen → revisit → 8B analyze → 32B expert → report
        """
        stats = {"discovered": 0, "completed": 0, "revisited": 0, "failed": 0}

        state = self.db.get_processing_state()
        logger.info(
            "Pipeline starting. Resume state: %s completed, %s pending, %s incomplete.",
            len(state["completed"]),
            len(state["pending"]),
            len(state["incomplete"]),
        )

        nav = self.instagram_nav
        story_nav = self.story_nav or nav  # fallback for tests

        if not nav or not self.sampler:
            logger.error("Navigator and sampler must be injected before running.")
            return stats

        try:
            # ── Phase 1: Discover stories from the tray ──────────────────
            logger.info("Discovering stories from tray...")
            if hasattr(nav, "get_stories_tray"):
                discovered_items = await nav.get_stories_tray()
            elif hasattr(nav, "get_story_tray_items"):
                discovered_items = await nav.get_story_tray_items()
            else:
                discovered_items = []
                logger.warning("Navigator has no story tray discovery method.")

            stats["discovered"] = len(discovered_items)
            logger.info("Discovered %d stories.", len(discovered_items))

            # ── Phase 2: First pass — capture + OCR + 4B screening ───────
            for idx, item in enumerate(discovered_items, start=1):
                try:
                    username = item.get("username")
                    if not username:
                        logger.warning("Story item %d has no username, skipping.", idx)
                        continue

                    logger.info("Processing Story %d/%d (@%s)...", idx, len(discovered_items), username)

                    # Get or create person
                    person = self.db.get_or_create_person(username)

                    # Check duplicate
                    if self.db.find_duplicate_story(
                        person.person_id, story_position=item.get("position", item.get("index"))
                    ):
                        logger.info("Skipping duplicate story for @%s", username)
                        continue

                    # Create a JSON-serializable copy of the reference (remove Playwright ElementHandles)
                    reference_data = {k: v for k, v in item.items() if k != "element"}

                    # Create story record
                    story = self.db.create_story(
                        person_id=person.person_id,
                        username_at_capture=username,
                        story_reference=reference_data,
                        story_position=item.get("position", item.get("index")),
                    )

                    # Open story in browser
                    opener = story_nav if hasattr(story_nav, "open_story") else nav
                    opened = await opener.open_story(item)
                    if not opened:
                        logger.warning("Failed to open story for @%s, skipping.", username)
                        stats["failed"] += 1
                        continue

                    now = datetime.now(tz=timezone.utc)
                    self.db.update_story_status(story.story_id, opened_at=now)

                    # ── Capture frames ───────────────────────────────────
                    frames = await self.sampler.initial_capture(
                        story_navigator=story_nav if hasattr(story_nav, "capture_current_frame") else opener,
                        story_id=story.story_id,
                        person_id=person.person_id,
                        db=self.db,
                    )
                    logger.info("Captured %d frames for story %s.", len(frames), story.story_id)

                    # ── OCR ──────────────────────────────────────────────
                    ocr_results = []
                    if self.ocr_engine and frames:
                        ocr_results = self.ocr_engine.extract_text_for_story(
                            frames=frames,
                            db=self.db,
                            story_id=story.story_id,
                        )
                        logger.info("OCR extracted %d text results.", len(ocr_results))

                    # ── 4B screening ─────────────────────────────────────
                    if self.screener and frames:
                        self.initial_analyzer.analyze_story(
                            story_id=story.story_id,
                            frames=frames,
                            ocr_results=ocr_results,
                            screener_model=self.screener,
                            db=self.db,
                        )

                    # Close story viewer
                    closer = story_nav if hasattr(story_nav, "close_story") else opener
                    await closer.close_story()

                except Exception as e:
                    logger.error("Error processing story %d: %s", idx, e, exc_info=True)
                    stats["failed"] += 1
                    # Try to close the viewer so we don't get stuck
                    try:
                        if story_nav and hasattr(story_nav, "close_story"):
                            await story_nav.close_story()
                    except Exception:
                        pass
                    continue

            # ── Phase 3: Process revisit queue ───────────────────────────
            queue = self.revisit_manager.get_sorted_queue(self.db)
            logger.info("Processing %d revisit requests.", len(queue))

            for rev in queue:
                try:
                    story = self.db.get_story(rev.story_id)
                    if not story:
                        continue

                    person = self.db.get_person_by_id(story.person_id)

                    # Reopen story
                    import json as _json
                    ref_dict = _json.loads(story.story_reference) if story.story_reference else None
                    if ref_dict and story_nav and hasattr(story_nav, "reopen_story_by_reference"):
                        await story_nav.reopen_story_by_reference(ref_dict)

                    # Revisit capture
                    revisit_frames = await self.sampler.revisit_capture(
                        story_navigator=story_nav,
                        story_id=rev.story_id,
                        person_id=story.person_id,
                        db=self.db,
                        reason=rev.reason,
                    )

                    # OCR on new frames
                    if self.ocr_engine and revisit_frames:
                        self.ocr_engine.extract_text_for_story(
                            frames=revisit_frames,
                            db=self.db,
                            story_id=rev.story_id,
                        )

                    # Mark revisit complete
                    now = datetime.now(tz=timezone.utc)
                    self.db.update_revisit_status(
                        rev.revisit_id, RevisitStatus.COMPLETED, completed_at=now
                    )
                    self.db.update_story_status(rev.story_id, revisit_status=RevisitStatus.COMPLETED)
                    stats["revisited"] += 1

                    # Close story
                    if story_nav and hasattr(story_nav, "close_story"):
                        await story_nav.close_story()

                except Exception as e:
                    logger.error("Revisit failed for story %s: %s", rev.story_id, e, exc_info=True)

            # ── Phase 4: Final analysis (8B + optional 32B) ──────────────
            pending = self.db.get_pending_stories()
            logger.info("Running final analysis on %d pending stories.", len(pending))

            for p_story in pending:
                if p_story.initial_analysis_status == AnalysisStatus.FAILED:
                    continue
                if p_story.final_analysis_status == AnalysisStatus.COMPLETED:
                    continue
                try:
                    if self.analyzer:
                        self.final_analyzer.analyze_story(
                            story_id=p_story.story_id,
                            analyzer_model=self.analyzer,
                            expert_model=self.expert,
                            db=self.db,
                        )
                        stats["completed"] += 1
                except Exception as e:
                    logger.error(
                        "Final analysis failed for story %s: %s", p_story.story_id, e, exc_info=True
                    )

            # ── Phase 5: Generate reports ────────────────────────────────
            try:
                report = self.report_generator.generate_text_report()
                if report:
                    logger.info("Report generated (%d chars).", len(report))
            except Exception as e:
                logger.warning("Report generation failed: %s", e)

        except Exception as e:
            logger.error("Pipeline run failed: %s", e, exc_info=True)

        logger.info(
            "Pipeline finished. discovered=%d completed=%d revisited=%d failed=%d",
            stats["discovered"], stats["completed"], stats["revisited"], stats["failed"],
        )
        return stats

    async def resume(self) -> dict:
        """Resume pipeline from last incomplete state."""
        logger.info("Resuming pipeline...")
        return await self.run()
