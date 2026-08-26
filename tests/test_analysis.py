"""
Tests for the analysis pipeline (InitialAnalyzer, RevisitManager,
FinalAnalyzer, ReportGenerator).

All tests use an in-memory/tmp_path database and mocked AI models.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.config import StoryZopConfig
from src.database.database import Database
from src.database.models import (
    SamplingDecision,
    AnalysisStatus,
    RevisitStatus,
    CapturePass,
    Frame,
)
from src.analysis.initial_analysis import InitialAnalyzer
from src.analysis.revisit import RevisitManager
from src.analysis.final_analysis import FinalAnalyzer
from src.analysis.report import ReportGenerator


@pytest.fixture()
def config():
    return StoryZopConfig()


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


@pytest.fixture()
def prep_story(db):
    person = db.get_or_create_person("testuser")
    story = db.create_story(person.person_id, "testuser")
    frame = db.save_frame(
        story.story_id, CapturePass.INITIAL, 1, "path/to/img.png", 100, 100
    )
    return story, [frame]


def test_initial_analyzer_sufficient(config, db, prep_story):
    story, frames = prep_story
    analyzer = InitialAnalyzer(config)

    mock_model = MagicMock()
    mock_model.screen_story.return_value = {
        "summary": "A test summary",
        "confidence": 0.95,
        "sampling_decision": "SUFFICIENT",
        "revisit_priority": 5,
        "revisit_reason": None,
    }

    res = analyzer.analyze_story(story.story_id, frames, [], mock_model, db)

    assert res is not None
    assert res.sampling_decision == SamplingDecision.SUFFICIENT

    updated_story = db.get_story(story.story_id)
    assert updated_story.initial_analysis_status == AnalysisStatus.COMPLETED


def test_initial_analyzer_revisit(config, db, prep_story):
    story, frames = prep_story
    analyzer = InitialAnalyzer(config)

    mock_model = MagicMock()
    mock_model.screen_story.return_value = {
        "summary": "Needs revisit",
        "confidence": 0.5,
        "sampling_decision": "REVISIT",
        "revisit_priority": 1,
        "revisit_reason": "Low confidence",
    }

    res = analyzer.analyze_story(story.story_id, frames, [], mock_model, db)

    assert res.sampling_decision == SamplingDecision.REVISIT

    queue = db.get_revisit_queue()
    assert len(queue) == 1
    assert queue[0].story_id == story.story_id
    assert queue[0].priority == 1

    updated_story = db.get_story(story.story_id)
    assert updated_story.revisit_status == RevisitStatus.QUEUED


def test_revisit_manager_logic(config):
    manager = RevisitManager(config)
    assert manager.should_revisit({"sampling_decision": "REVISIT"}) is True
    assert manager.should_revisit({"sampling_decision": "SUFFICIENT"}) is False

    assert manager.get_priority({"revisit_priority": 2}) == 2
    assert manager.get_priority({"revisit_priority": "invalid"}) == 5


def test_final_analyzer_no_expert(config, db, prep_story):
    story, frames = prep_story
    db.save_initial_analysis(
        story.story_id,
        model="model",
        summary="sum",
        visible_information="vis",
        confidence=0.9,
        sampling_decision=SamplingDecision.SUFFICIENT,
    )

    config.expert_review_threshold = 0.8
    analyzer = FinalAnalyzer(config)

    mock_analyzer = MagicMock()
    mock_analyzer.analyze_story.return_value = {
        "description": "Good",
        "confidence": 0.9,
    }

    res = analyzer.analyze_story(story.story_id, mock_analyzer, None, db)
    assert res is not None
    assert res.confidence == 0.9

    updated_story = db.get_story(story.story_id)
    assert updated_story.final_analysis_status == AnalysisStatus.COMPLETED


def test_final_analyzer_expert_triggered(config, db, prep_story):
    story, frames = prep_story
    db.save_initial_analysis(
        story.story_id,
        model="model",
        summary="sum",
        visible_information="vis",
        confidence=0.9,
        sampling_decision=SamplingDecision.SUFFICIENT,
    )

    config.expert_review_threshold = 0.8
    analyzer = FinalAnalyzer(config)

    mock_analyzer = MagicMock()
    mock_analyzer.analyze_story.return_value = {
        "description": "Bad",
        "confidence": 0.5,
    }

    mock_expert = MagicMock()
    mock_expert.expert_review.return_value = {
        "reason": "Reviewing",
        "analysis": "Expert says ok",
        "confidence": 0.9,
    }

    res = analyzer.analyze_story(story.story_id, mock_analyzer, mock_expert, db)

    assert res is not None
    assert res.confidence == 0.5

    reviews = db._conn.execute("SELECT * FROM expert_reviews").fetchall()
    assert len(reviews) == 1


def test_report_generator_text(db, prep_story):
    story, _ = prep_story
    db.save_initial_analysis(
        story.story_id,
        model="model",
        summary="sum",
        visible_information="vis",
        confidence=0.9,
        sampling_decision=SamplingDecision.SUFFICIENT,
    )
    db.save_final_analysis(
        story.story_id,
        model="model",
        content_type="type",
        description="Desc",
        important_information="info",
        visible_text="text",
        confidence=0.9,
    )

    gen = ReportGenerator(db)
    report = gen.generate_text_report()
    assert "Person: @testuser" in report
    assert "Desc" in report


def test_report_generator_export_json_csv(tmp_path, db, prep_story):
    story, _ = prep_story
    db.save_initial_analysis(
        story.story_id,
        model="model",
        summary="sum",
        visible_information="vis",
        confidence=0.9,
        sampling_decision=SamplingDecision.SUFFICIENT,
    )
    db.save_final_analysis(
        story.story_id,
        model="model",
        content_type="type",
        description="Desc",
        important_information="info",
        visible_text="text",
        confidence=0.9,
    )

    gen = ReportGenerator(db)

    json_path = tmp_path / "report.json"
    gen.export_json(json_path)
    assert json_path.exists()

    csv_path = tmp_path / "report.csv"
    gen.export_csv(csv_path)
    assert csv_path.exists()
