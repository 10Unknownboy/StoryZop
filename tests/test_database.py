"""
Comprehensive tests for the StoryZop database layer.

All tests use an in-memory SQLite database so nothing touches the filesystem.
"""

from __future__ import annotations

import pytest

from src.database.database import Database
from src.database.models import (
    AnalysisStatus,
    CapturePass,
    CaptureStatus,
    EventType,
    IdentityStatus,
    RevisitStatus,
    SamplingDecision,
)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def db() -> Database:
    """Yield a fresh in-memory database for each test."""
    database = Database(":memory:")
    database.initialize()
    return database


# ── table creation ───────────────────────────────────────────────────────


class TestInitialization:
    """Verify that the schema is created correctly."""

    def test_tables_exist(self, db: Database) -> None:
        tables = {
            r[0]
            for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "persons",
            "usernames",
            "stories",
            "frames",
            "ocr_results",
            "initial_analyses",
            "revisits",
            "final_analyses",
            "expert_reviews",
            "story_events",
            "id_counters",
            "_schema_version",
        }
        assert expected.issubset(tables)

    def test_idempotent_initialize(self, db: Database) -> None:
        """Calling initialize() twice should not raise."""
        db.initialize()


# ── ID generation ────────────────────────────────────────────────────────


class TestIDGeneration:
    def test_sequential_person_ids(self, db: Database) -> None:
        p1 = db.create_person("user_a")
        p2 = db.create_person("user_b")
        assert p1.person_id == "P_000001"
        assert p2.person_id == "P_000002"

    def test_sequential_story_ids(self, db: Database) -> None:
        person = db.create_person("user_a")
        s1 = db.create_story(person.person_id, "user_a")
        s2 = db.create_story(person.person_id, "user_a")
        assert s1.story_id == "S_000001"
        assert s2.story_id == "S_000002"

    def test_ids_are_unique_across_calls(self, db: Database) -> None:
        ids = {db._next_id("person") for _ in range(50)}
        assert len(ids) == 50


# ── persons ──────────────────────────────────────────────────────────────


class TestPersons:
    def test_create_person(self, db: Database) -> None:
        person = db.create_person("alice")
        assert person.person_id.startswith("P_")
        assert person.current_username == "alice"
        assert person.identity_status == IdentityStatus.UNVERIFIED

    def test_get_person_by_id(self, db: Database) -> None:
        created = db.create_person("alice")
        found = db.get_person_by_id(created.person_id)
        assert found is not None
        assert found.person_id == created.person_id
        assert found.current_username == "alice"

    def test_get_person_by_id_not_found(self, db: Database) -> None:
        assert db.get_person_by_id("P_999999") is None

    def test_get_person_by_username(self, db: Database) -> None:
        db.create_person("bob")
        found = db.get_person_by_username("bob")
        assert found is not None
        assert found.current_username == "bob"

    def test_get_person_by_username_not_found(self, db: Database) -> None:
        assert db.get_person_by_username("nobody") is None

    def test_get_or_create_existing(self, db: Database) -> None:
        p1 = db.get_or_create_person("charlie")
        p2 = db.get_or_create_person("charlie")
        assert p1.person_id == p2.person_id

    def test_get_or_create_new(self, db: Database) -> None:
        p1 = db.get_or_create_person("alice")
        p2 = db.get_or_create_person("bob")
        assert p1.person_id != p2.person_id

    def test_confirm_identity(self, db: Database) -> None:
        person = db.create_person("dana")
        assert person.identity_status == IdentityStatus.UNVERIFIED
        db.confirm_identity(person.person_id)
        updated = db.get_person_by_id(person.person_id)
        assert updated is not None
        assert updated.identity_status == IdentityStatus.CONFIRMED


# ── strict identity logic ───────────────────────────────────────────────


class TestStrictIdentity:
    """Verify that username changes are flagged as UNCERTAIN, not merged."""

    def test_username_change_marks_uncertain(self, db: Database) -> None:
        """Calling update_person_username should mark identity UNCERTAIN."""
        person = db.create_person("old_name")
        db.update_person_username(person.person_id, "new_name")

        updated = db.get_person_by_id(person.person_id)
        assert updated is not None
        assert updated.current_username == "new_name"
        assert updated.identity_status == IdentityStatus.UNCERTAIN

    def test_username_change_creates_history(self, db: Database) -> None:
        person = db.create_person("original")
        db.update_person_username(person.person_id, "renamed")

        history = db.get_username_history(person.person_id)
        usernames = [h.username for h in history]
        assert "original" in usernames
        assert "renamed" in usernames
        assert len(history) == 2

    def test_new_username_creates_new_person(self, db: Database) -> None:
        """An unknown username must never merge with existing persons."""
        db.create_person("alice")
        p2 = db.get_or_create_person("bob")
        # bob is completely new → separate person
        assert db.get_person_by_username("alice") is not None
        assert p2.current_username == "bob"
        assert db.get_person_by_username("alice").person_id != p2.person_id

    def test_get_or_create_detects_username_mismatch(self, db: Database) -> None:
        """If we look up a historical username but current_username differs,
        the identity should be marked UNCERTAIN."""
        person = db.create_person("name_v1")
        # Simulate the username changing in person record
        db.update_person_username(person.person_id, "name_v2")

        # Now look up the OLD username — should find the person but mark uncertain
        found = db.get_or_create_person("name_v1")
        assert found.person_id == person.person_id
        assert found.identity_status == IdentityStatus.UNCERTAIN


# ── username history ─────────────────────────────────────────────────────


class TestUsernameHistory:
    def test_initial_history_entry(self, db: Database) -> None:
        person = db.create_person("eve")
        history = db.get_username_history(person.person_id)
        assert len(history) == 1
        assert history[0].username == "eve"

    def test_multiple_entries(self, db: Database) -> None:
        person = db.create_person("name_a")
        db.update_person_username(person.person_id, "name_b")
        db.update_person_username(person.person_id, "name_c")
        history = db.get_username_history(person.person_id)
        assert len(history) == 3

    def test_empty_for_unknown(self, db: Database) -> None:
        assert db.get_username_history("P_999999") == []


# ── stories ──────────────────────────────────────────────────────────────


class TestStories:
    def test_create_story(self, db: Database) -> None:
        person = db.create_person("frank")
        story = db.create_story(person.person_id, "frank")
        assert story.story_id.startswith("S_")
        assert story.person_id == person.person_id
        assert story.capture_status == CaptureStatus.PENDING

    def test_get_story(self, db: Database) -> None:
        person = db.create_person("grace")
        created = db.create_story(person.person_id, "grace")
        found = db.get_story(created.story_id)
        assert found is not None
        assert found.story_id == created.story_id

    def test_get_story_not_found(self, db: Database) -> None:
        assert db.get_story("S_999999") is None

    def test_person_story_relationship(self, db: Database) -> None:
        person = db.create_person("hank")
        db.create_story(person.person_id, "hank")
        db.create_story(person.person_id, "hank")
        db.create_story(person.person_id, "hank")
        stories = db.get_stories_for_person(person.person_id)
        assert len(stories) == 3
        for s in stories:
            assert s.person_id == person.person_id

    def test_update_story_status(self, db: Database) -> None:
        person = db.create_person("ivy")
        story = db.create_story(person.person_id, "ivy")
        db.update_story_status(
            story.story_id,
            capture_status=CaptureStatus.COMPLETED,
            initial_analysis_status=AnalysisStatus.COMPLETED,
        )
        updated = db.get_story(story.story_id)
        assert updated is not None
        assert updated.capture_status == CaptureStatus.COMPLETED
        assert updated.initial_analysis_status == AnalysisStatus.COMPLETED

    def test_story_creates_detected_event(self, db: Database) -> None:
        person = db.create_person("jack")
        story = db.create_story(person.person_id, "jack")
        events = db.get_events_for_story(story.story_id)
        assert len(events) >= 1
        assert events[0].event_type == EventType.STORY_DETECTED

    def test_pending_stories(self, db: Database) -> None:
        person = db.create_person("kate")
        s1 = db.create_story(person.person_id, "kate")
        s2 = db.create_story(person.person_id, "kate")
        # Mark s1 as completed
        db.update_story_status(
            s1.story_id, final_analysis_status=AnalysisStatus.COMPLETED
        )
        pending = db.get_pending_stories()
        pending_ids = [s.story_id for s in pending]
        assert s2.story_id in pending_ids
        assert s1.story_id not in pending_ids


# ── duplicate detection ──────────────────────────────────────────────────


class TestDuplicateDetection:
    def test_find_by_reference(self, db: Database) -> None:
        person = db.create_person("lucy")
        ref = {"url": "https://instagram.com/stories/lucy/123"}
        db.create_story(person.person_id, "lucy", story_reference=ref)
        dup = db.find_duplicate_story(person.person_id, story_reference=ref)
        assert dup is not None

    def test_find_by_position(self, db: Database) -> None:
        person = db.create_person("mike")
        db.create_story(person.person_id, "mike", story_position=3)
        dup = db.find_duplicate_story(person.person_id, story_position=3)
        assert dup is not None

    def test_no_false_duplicate(self, db: Database) -> None:
        person = db.create_person("nancy")
        db.create_story(person.person_id, "nancy", story_position=1)
        dup = db.find_duplicate_story(person.person_id, story_position=99)
        assert dup is None


# ── frames ───────────────────────────────────────────────────────────────


class TestFrames:
    def test_save_frame(self, db: Database) -> None:
        person = db.create_person("oscar")
        story = db.create_story(person.person_id, "oscar")
        frame = db.save_frame(
            story.story_id, CapturePass.INITIAL, 1, "/path/frame_001.png", 412, 915
        )
        assert frame.frame_id.startswith("F_")
        assert frame.capture_pass == CapturePass.INITIAL

    def test_frames_for_story(self, db: Database) -> None:
        person = db.create_person("pat")
        story = db.create_story(person.person_id, "pat")
        db.save_frame(story.story_id, CapturePass.INITIAL, 1, "/f1.png")
        db.save_frame(story.story_id, CapturePass.INITIAL, 2, "/f2.png")
        db.save_frame(story.story_id, CapturePass.REVISIT, 3, "/f3.png")
        all_frames = db.get_frames_for_story(story.story_id)
        assert len(all_frames) == 3
        initial = db.get_frames_for_story(story.story_id, CapturePass.INITIAL)
        assert len(initial) == 2
        revisit = db.get_frames_for_story(story.story_id, CapturePass.REVISIT)
        assert len(revisit) == 1


# ── OCR ──────────────────────────────────────────────────────────────────


class TestOCR:
    def test_save_and_retrieve_ocr(self, db: Database) -> None:
        person = db.create_person("quinn")
        story = db.create_story(person.person_id, "quinn")
        frame = db.save_frame(story.story_id, CapturePass.INITIAL, 1, "/f.png")
        ocr = db.save_ocr_result(
            frame.frame_id,
            "Hello World",
            confidence=0.95,
            bounding_data=[[10, 20, 100, 50]],
        )
        assert ocr.ocr_id.startswith("O_")
        assert ocr.text == "Hello World"
        assert ocr.confidence == 0.95

    def test_ocr_for_frame(self, db: Database) -> None:
        person = db.create_person("rick")
        story = db.create_story(person.person_id, "rick")
        frame = db.save_frame(story.story_id, CapturePass.INITIAL, 1, "/f.png")
        db.save_ocr_result(frame.frame_id, "Line 1", confidence=0.9)
        db.save_ocr_result(frame.frame_id, "Line 2", confidence=0.8)
        results = db.get_ocr_for_frame(frame.frame_id)
        assert len(results) == 2

    def test_ocr_for_story(self, db: Database) -> None:
        person = db.create_person("sara")
        story = db.create_story(person.person_id, "sara")
        f1 = db.save_frame(story.story_id, CapturePass.INITIAL, 1, "/f1.png")
        f2 = db.save_frame(story.story_id, CapturePass.INITIAL, 2, "/f2.png")
        db.save_ocr_result(f1.frame_id, "Text A")
        db.save_ocr_result(f2.frame_id, "Text B")
        all_ocr = db.get_ocr_for_story(story.story_id)
        assert len(all_ocr) == 2


# ── initial analysis ─────────────────────────────────────────────────────


class TestInitialAnalysis:
    def test_save_initial_analysis(self, db: Database) -> None:
        person = db.create_person("tom")
        story = db.create_story(person.person_id, "tom")
        analysis = db.save_initial_analysis(
            story.story_id,
            model="Qwen3-VL-4B-Instruct",
            summary="A photo of a sunset",
            confidence=0.9,
            sampling_decision=SamplingDecision.ACCEPT,
        )
        assert analysis.analysis_id.startswith("IA_")
        assert analysis.sampling_decision == SamplingDecision.ACCEPT

    def test_get_initial_analysis(self, db: Database) -> None:
        person = db.create_person("uma")
        story = db.create_story(person.person_id, "uma")
        db.save_initial_analysis(
            story.story_id,
            model="Qwen3-VL-4B-Instruct",
            summary="test",
            sampling_decision=SamplingDecision.REVISIT,
            revisit_priority=8,
            revisit_reason="Text partially obscured",
        )
        found = db.get_initial_analysis(story.story_id)
        assert found is not None
        assert found.sampling_decision == SamplingDecision.REVISIT
        assert found.revisit_priority == 8

    def test_initial_analysis_never_overwritten(self, db: Database) -> None:
        """Multiple initial analyses can coexist (append-only)."""
        person = db.create_person("vera")
        story = db.create_story(person.person_id, "vera")
        a1 = db.save_initial_analysis(
            story.story_id, model="4B", summary="first pass"
        )
        a2 = db.save_initial_analysis(
            story.story_id, model="4B", summary="second pass"
        )
        assert a1.analysis_id != a2.analysis_id


# ── revisit queue ────────────────────────────────────────────────────────


class TestRevisitQueue:
    def test_queue_revisit(self, db: Database) -> None:
        person = db.create_person("wendy")
        story = db.create_story(person.person_id, "wendy")
        revisit = db.queue_revisit(story.story_id, priority=8, reason="text unclear")
        assert revisit.revisit_id.startswith("R_")
        assert revisit.status == RevisitStatus.QUEUED

    def test_queue_sorted_by_priority(self, db: Database) -> None:
        person = db.create_person("xena")
        s1 = db.create_story(person.person_id, "xena")
        s2 = db.create_story(person.person_id, "xena")
        s3 = db.create_story(person.person_id, "xena")
        db.queue_revisit(s1.story_id, priority=3)
        db.queue_revisit(s2.story_id, priority=10)
        db.queue_revisit(s3.story_id, priority=6)
        queue = db.get_revisit_queue()
        priorities = [r.priority for r in queue]
        assert priorities == [10, 6, 3]

    def test_update_revisit_status(self, db: Database) -> None:
        person = db.create_person("yara")
        story = db.create_story(person.person_id, "yara")
        revisit = db.queue_revisit(story.story_id, priority=5)
        from datetime import datetime, timezone

        db.update_revisit_status(
            revisit.revisit_id,
            RevisitStatus.COMPLETED,
            completed_at=datetime.now(tz=timezone.utc),
            increment_attempts=True,
        )
        queue = db.get_revisit_queue()
        # Completed revisit should no longer appear in pending queue
        assert all(r.revisit_id != revisit.revisit_id for r in queue)

    def test_revisit_creates_event(self, db: Database) -> None:
        person = db.create_person("zoe")
        story = db.create_story(person.person_id, "zoe")
        db.queue_revisit(story.story_id, priority=7, reason="need more detail")
        events = db.get_events_for_story(story.story_id)
        types = [e.event_type for e in events]
        assert EventType.REVISIT_REQUESTED in types


# ── final analysis ───────────────────────────────────────────────────────


class TestFinalAnalysis:
    def test_save_final_analysis(self, db: Database) -> None:
        person = db.create_person("alice2")
        story = db.create_story(person.person_id, "alice2")
        analysis = db.save_final_analysis(
            story.story_id,
            model="Qwen3-VL-8B-Instruct",
            content_type="photo",
            description="A beach at sunset",
            confidence=0.85,
            people_detected=["person_in_foreground"],
            objects_detected=["palm_tree", "ocean"],
        )
        assert analysis.analysis_id.startswith("FA_")

    def test_get_final_analysis(self, db: Database) -> None:
        person = db.create_person("bob2")
        story = db.create_story(person.person_id, "bob2")
        db.save_final_analysis(
            story.story_id, model="8B", description="test", confidence=0.72
        )
        found = db.get_final_analysis(story.story_id)
        assert found is not None
        assert found.confidence == 0.72


# ── expert review ────────────────────────────────────────────────────────


class TestExpertReview:
    def test_save_expert_review(self, db: Database) -> None:
        person = db.create_person("charlie2")
        story = db.create_story(person.person_id, "charlie2")
        review = db.save_expert_review(
            story.story_id,
            model="Qwen3-VL-32B-Instruct",
            reason="low confidence",
            analysis="Detailed analysis...",
            confidence=0.88,
        )
        assert review.review_id.startswith("ER_")

    def test_get_expert_review(self, db: Database) -> None:
        person = db.create_person("dana2")
        story = db.create_story(person.person_id, "dana2")
        db.save_expert_review(story.story_id, model="32B", reason="ambiguous")
        found = db.get_expert_review(story.story_id)
        assert found is not None
        assert found.reason == "ambiguous"

    def test_no_expert_review(self, db: Database) -> None:
        person = db.create_person("eve2")
        story = db.create_story(person.person_id, "eve2")
        assert db.get_expert_review(story.story_id) is None


# ── events ───────────────────────────────────────────────────────────────


class TestEvents:
    def test_log_event(self, db: Database) -> None:
        person = db.create_person("frank2")
        story = db.create_story(person.person_id, "frank2")
        event = db.log_event(
            story.story_id,
            EventType.STORY_OPENED,
            {"detail": "opened successfully"},
        )
        assert event.event_id.startswith("EV_")
        assert event.event_type == EventType.STORY_OPENED

    def test_events_chronological(self, db: Database) -> None:
        person = db.create_person("grace2")
        story = db.create_story(person.person_id, "grace2")
        db.log_event(story.story_id, EventType.STORY_OPENED)
        db.log_event(story.story_id, EventType.INITIAL_CAPTURE_STARTED)
        db.log_event(story.story_id, EventType.INITIAL_CAPTURE_COMPLETED)
        events = db.get_events_for_story(story.story_id)
        # First event is STORY_DETECTED (auto-created by create_story)
        assert len(events) >= 4


# ── resume ───────────────────────────────────────────────────────────────


class TestResume:
    def test_processing_state(self, db: Database) -> None:
        person = db.create_person("hank2")
        s1 = db.create_story(person.person_id, "hank2")
        s2 = db.create_story(person.person_id, "hank2")
        s3 = db.create_story(person.person_id, "hank2")

        # s1: completed
        db.update_story_status(
            s1.story_id,
            capture_status=CaptureStatus.COMPLETED,
            final_analysis_status=AnalysisStatus.COMPLETED,
        )
        # s2: in progress (capture done, analysis pending)
        db.update_story_status(
            s2.story_id, capture_status=CaptureStatus.COMPLETED
        )
        # s3: untouched (pending)

        state = db.get_processing_state()
        assert s1.story_id in state["completed"]
        assert s2.story_id in state["incomplete"]
        assert s3.story_id in state["pending"]
        assert state["total"] == 3

    def test_incomplete_stories(self, db: Database) -> None:
        person = db.create_person("ivy2")
        s1 = db.create_story(person.person_id, "ivy2")
        s2 = db.create_story(person.person_id, "ivy2")
        db.update_story_status(
            s1.story_id,
            capture_status=CaptureStatus.COMPLETED,
            final_analysis_status=AnalysisStatus.COMPLETED,
        )
        incomplete = db.get_incomplete_stories()
        ids = [s.story_id for s in incomplete]
        assert s2.story_id in ids
        assert s1.story_id not in ids


# ── foreign key enforcement ──────────────────────────────────────────────


class TestForeignKeys:
    def test_story_requires_valid_person(self, db: Database) -> None:
        with pytest.raises(Exception):
            db.create_story("P_NONEXISTENT", "nobody")

    def test_frame_requires_valid_story(self, db: Database) -> None:
        with pytest.raises(Exception):
            db.save_frame("S_NONEXISTENT", CapturePass.INITIAL, 1, "/f.png")

    def test_ocr_requires_valid_frame(self, db: Database) -> None:
        with pytest.raises(Exception):
            db.save_ocr_result("F_NONEXISTENT", "text")


# ── person → story history (spec §27) ───────────────────────────────────


class TestPersonStoryHistory:
    def test_full_history(self, db: Database) -> None:
        person = db.create_person("person_x")
        db.create_story(person.person_id, "person_x")
        db.create_story(person.person_id, "person_x")
        db.create_story(person.person_id, "person_x")
        db.create_story(person.person_id, "person_x")

        stories = db.get_stories_for_person(person.person_id)
        assert len(stories) == 4
        assert all(s.person_id == person.person_id for s in stories)

    def test_multiple_persons_isolated(self, db: Database) -> None:
        p1 = db.create_person("alice3")
        p2 = db.create_person("bob3")
        db.create_story(p1.person_id, "alice3")
        db.create_story(p1.person_id, "alice3")
        db.create_story(p2.person_id, "bob3")

        assert len(db.get_stories_for_person(p1.person_id)) == 2
        assert len(db.get_stories_for_person(p2.person_id)) == 1
