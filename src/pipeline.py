from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import AnalysisStatus
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
        
        self.browser_session: Any = None
        
        self.screener = None
        self.analyzer = None
        self.expert = None
        
        self.ocr_engine = None
        self.sampler = None
        self.navigator = None
        
        self.initial_analyzer = InitialAnalyzer(config)
        self.revisit_manager = RevisitManager(config)
        self.final_analyzer = FinalAnalyzer(config)
        self.report_generator = ReportGenerator(db)

    def set_browser(self, session: Any) -> None:
        """Inject browser session and initialize components that depend on it."""
        self.browser_session = session

    def set_models(self, screener: Any, analyzer: Any, expert: Any | None = None) -> None:
        """Inject the initialized AI models."""
        self.screener = screener
        self.analyzer = analyzer
        self.expert = expert

    def set_ocr(self, ocr_engine: Any) -> None:
        """Inject the initialized OCR engine."""
        self.ocr_engine = ocr_engine
        
    def set_components(self, navigator: Any, sampler: Any) -> None:
        """Inject browser interaction components."""
        self.navigator = navigator
        self.sampler = sampler

    async def run(self) -> dict:
        """Run the full end-to-end pipeline."""
        stats = {"discovered": 0, "completed": 0, "revisited": 0, "failed": 0}
        
        # 1. Check resume state
        state = self.db.get_processing_state()
        logger.info("Pipeline starting. Resume state: %s completed, %s pending, %s incomplete.",
                    len(state["completed"]), len(state["pending"]), len(state["incomplete"]))
                    
        if not self.navigator or not self.sampler:
            logger.error("Navigator and sampler components must be injected before running.")
            return stats
            
        try:
            # 4. Discover stories
            logger.info("Discovering stories from tray...")
            discovered_items = await self.navigator.get_story_tray_items()
            stats["discovered"] = len(discovered_items)
            
            # 5. First pass
            for idx, item in enumerate(discovered_items, start=1):
                logger.info("Processing Story %d/%d...", idx, len(discovered_items))
                try:
                    # a. Extract person info
                    username = item.get("username")
                    if not username:
                        continue
                        
                    # b. get or create person
                    person = self.db.get_or_create_person(username)
                    
                    # c. Check duplicate
                    if self.db.find_duplicate_story(person.person_id, story_position=item.get("position")):
                        logger.info("Skipping duplicate story for @%s", username)
                        continue
                        
                    # d. Create story
                    story = self.db.create_story(
                        person_id=person.person_id,
                        username_at_capture=username,
                        story_reference=item,
                        story_position=item.get("position")
                    )
                    
                    # e. Open story
                    await self.navigator.open_story(item)
                    now = datetime.now(tz=timezone.utc)
                    self.db.update_story_status(story.story_id, opened_at=now)
                    
                    # f. Initial capture
                    frames = await self.sampler.initial_capture(story.story_id)
                    
                    # g. OCR
                    ocr_results = []
                    if self.ocr_engine:
                        for f in frames:
                            res = await self.ocr_engine.extract_frame(f.file_path)
                            if res:
                                ocr_res = self.db.save_ocr_result(
                                    f.frame_id, 
                                    res.get("text", ""),
                                    confidence=res.get("confidence")
                                )
                                ocr_results.append(ocr_res)
                                
                    # h. 4B initial analysis
                    if self.screener:
                        self.initial_analyzer.analyze_story(
                            story.story_id, frames, ocr_results, self.screener, self.db
                        )
                        
                    # i. Close story
                    await self.navigator.close_story()
                    
                except Exception as e:
                    logger.error("Error in first pass for item %s: %s", item, e)
                    stats["failed"] += 1
                    continue
                    
            # 6 & 7. Process Revisit Queue
            queue = self.revisit_manager.get_sorted_queue(self.db)
            for rev in queue:
                try:
                    await self.revisit_manager.process_revisit(
                        rev, self.navigator, self.sampler, self.ocr_engine, self.db
                    )
                    stats["revisited"] += 1
                except Exception as e:
                    logger.error("Error in revisit pass for story %s: %s", rev.story_id, e)
                    
            # 8. Final analysis
            pending = self.db.get_pending_stories()
            for p_story in pending:
                if p_story.initial_analysis_status == AnalysisStatus.FAILED:
                    continue
                try:
                    if self.analyzer:
                        self.final_analyzer.analyze_story(
                            p_story.story_id, self.analyzer, self.expert, self.db
                        )
                        stats["completed"] += 1
                except Exception as e:
                    logger.error("Error in final analysis for story %s: %s", p_story.story_id, e)
                    
            # 9. Generate report
            self.report_generator.generate_text_report()
            
        except Exception as e:
            logger.error("Pipeline run failed: %s", e)
            
        return stats

    async def resume(self) -> dict:
        """Resume pipeline processing from the last incomplete state."""
        # Simple implementation that falls back to run() since it checks state
        logger.info("Resuming pipeline...")
        return await self.run()
