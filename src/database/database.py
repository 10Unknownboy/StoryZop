"""
SQLite database layer for StoryZop.

Provides a ``Database`` class that wraps all CRUD operations for the ten
core tables.  ID generation uses an atomic counter table to produce stable,
human-readable identifiers (``P_000001``, ``S_000042``, …).

Usage::

    from src.database import Database

    db = Database(":memory:")  # or a file path
    db.initialize()

    person = db.get_or_create_person("some_user")
    story  = db.create_story(person.person_id, "some_user")
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.database.models import (
    AnalysisStatus,
    CapturePass,
    CaptureStatus,
    EventType,
    ExpertReview,
    FinalAnalysis,
    Frame,
    IdentityStatus,
    InitialAnalysis,
    OCRResult,
    Person,
    Revisit,
    RevisitStatus,
    SamplingDecision,
    Story,
    StoryEvent,
    Username,
)
from src.logger import get_logger

logger = get_logger("database")

# Prefix → counter-name mapping for ID generation
_ID_PREFIXES: dict[str, str] = {
    "person": "P",
    "story": "S",
    "frame": "F",
    "ocr": "O",
    "initial_analysis": "IA",
    "revisit": "R",
    "final_analysis": "FA",
    "expert_review": "ER",
    "event": "EV",
}


class Database:
    """Thin wrapper around a SQLite database for StoryZop.

    All public methods that mutate state commit automatically.  The class can
    be used as a context manager::

        with Database("storyzop.db") as db:
            ...
    """

    # ── lifecycle ────────────────────────────────────────────────────────

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # ── initialisation ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create all tables if they do not already exist."""
        from src.database.migrations import apply_migrations

        apply_migrations(self._conn)
        logger.info("Database initialised at %s", self._db_path)

    # ── ID generation ────────────────────────────────────────────────────

    def _next_id(self, entity: str) -> str:
        """Return the next sequential ID for *entity* (e.g. ``P_000001``)."""
        prefix = _ID_PREFIXES[entity]
        cur = self._conn.execute(
            "INSERT INTO id_counters (entity) VALUES (?) "
            "ON CONFLICT(entity) DO UPDATE SET counter = counter + 1 "
            "RETURNING counter",
            (entity,),
        )
        row = cur.fetchone()
        if row is None:
            # Fallback for SQLite versions without RETURNING
            cur2 = self._conn.execute(
                "SELECT counter FROM id_counters WHERE entity = ?", (entity,)
            )
            row = cur2.fetchone()
        counter: int = row[0]  # type: ignore[index]
        self._conn.commit()
        return f"{prefix}_{counter:06d}"

    # ── persons ──────────────────────────────────────────────────────────

    def create_person(
        self,
        username: str,
        display_name: str | None = None,
        pfp_path: str | None = None,
        identity_status: IdentityStatus = IdentityStatus.UNVERIFIED,
    ) -> Person:
        """Create a new person record and their first username-history entry."""
        person_id = self._next_id("person")
        now = datetime.now(tz=timezone.utc)
        self._conn.execute(
            """
            INSERT INTO persons
                (person_id, created_at, last_seen,
                 current_username, current_display_name, current_pfp_path,
                 identity_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                now.isoformat(),
                now.isoformat(),
                username,
                display_name,
                pfp_path,
                identity_status.value,
            ),
        )
        # Record first username in history
        self._conn.execute(
            """
            INSERT INTO usernames (person_id, username, first_seen, last_seen)
            VALUES (?, ?, ?, ?)
            """,
            (person_id, username, now.isoformat(), now.isoformat()),
        )
        self._conn.commit()
        logger.info("Created person %s ← @%s", person_id, username)
        return Person(
            person_id=person_id,
            created_at=now,
            last_seen=now,
            current_username=username,
            current_display_name=display_name,
            current_pfp_path=pfp_path,
            identity_status=identity_status,
        )

    def get_person_by_id(self, person_id: str) -> Person | None:
        """Look up a person by their stable internal ID."""
        row = self._conn.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def get_person_by_username(self, username: str) -> Person | None:
        """Look up a person by any username they have ever used.

        Returns the person whose *most recent* username-history entry matches.
        """
        row = self._conn.execute(
            """
            SELECT p.* FROM persons p
            JOIN usernames u ON p.person_id = u.person_id
            WHERE u.username = ?
            ORDER BY u.last_seen DESC
            LIMIT 1
            """,
            (username,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def get_or_create_person(
        self,
        username: str,
        display_name: str | None = None,
        pfp_path: str | None = None,
    ) -> Person:
        """Find an existing person by *username* or create a new one.

        **Strict identity logic**: if the username is found in the history
        table, the existing ``person_id`` is reused and ``last_seen`` is
        updated.  However, if the existing person's ``current_username``
        differs from *username* (i.e. the username changed), the record is
        marked ``UNCERTAIN`` rather than silently merged.

        If the username is completely new, a fresh person record is created.
        """
        existing = self.get_person_by_username(username)
        if existing is not None:
            now = datetime.now(tz=timezone.utc)
            # Check for username-change scenario
            if existing.current_username != username:
                # The username we found was historical — the person's current
                # username is different.  Mark UNCERTAIN so it isn't silently
                # merged.
                self._conn.execute(
                    """
                    UPDATE persons
                    SET last_seen = ?, identity_status = ?
                    WHERE person_id = ?
                    """,
                    (
                        now.isoformat(),
                        IdentityStatus.UNCERTAIN.value,
                        existing.person_id,
                    ),
                )
                existing.identity_status = IdentityStatus.UNCERTAIN
                logger.warning(
                    "Person %s: username mismatch (current=%s, lookup=%s) → "
                    "identity marked UNCERTAIN",
                    existing.person_id,
                    existing.current_username,
                    username,
                )
            else:
                self._conn.execute(
                    "UPDATE persons SET last_seen = ? WHERE person_id = ?",
                    (now.isoformat(), existing.person_id),
                )
            # Update username last_seen
            self._conn.execute(
                """
                UPDATE usernames SET last_seen = ?
                WHERE person_id = ? AND username = ?
                """,
                (now.isoformat(), existing.person_id, username),
            )
            self._conn.commit()
            existing.last_seen = now
            return existing

        # Completely new person
        return self.create_person(
            username=username,
            display_name=display_name,
            pfp_path=pfp_path,
        )

    def update_person_username(
        self,
        person_id: str,
        new_username: str,
    ) -> None:
        """Record a username change for an existing person.

        Adds a new username-history entry and marks identity as UNCERTAIN.
        Does **not** silently merge — callers should verify identity
        separately.
        """
        now = datetime.now(tz=timezone.utc)
        # Mark person as uncertain
        self._conn.execute(
            """
            UPDATE persons
            SET current_username = ?, last_seen = ?, identity_status = ?
            WHERE person_id = ?
            """,
            (
                new_username,
                now.isoformat(),
                IdentityStatus.UNCERTAIN.value,
                person_id,
            ),
        )
        # Add new username to history
        self._conn.execute(
            """
            INSERT INTO usernames (person_id, username, first_seen, last_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id, username) DO UPDATE SET last_seen = ?
            """,
            (
                person_id,
                new_username,
                now.isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        self._conn.commit()
        logger.warning(
            "Person %s username updated to @%s — identity marked UNCERTAIN",
            person_id,
            new_username,
        )

    def confirm_identity(self, person_id: str) -> None:
        """Manually confirm a person's identity (clear UNCERTAIN status)."""
        self._conn.execute(
            "UPDATE persons SET identity_status = ? WHERE person_id = ?",
            (IdentityStatus.CONFIRMED.value, person_id),
        )
        self._conn.commit()

    def get_username_history(self, person_id: str) -> list[Username]:
        """Return all usernames historically associated with a person."""
        rows = self._conn.execute(
            "SELECT * FROM usernames WHERE person_id = ? ORDER BY first_seen",
            (person_id,),
        ).fetchall()
        return [
            Username(
                id=r["id"],
                person_id=r["person_id"],
                username=r["username"],
                first_seen=datetime.fromisoformat(r["first_seen"]),
                last_seen=datetime.fromisoformat(r["last_seen"]),
            )
            for r in rows
        ]

    # ── stories ──────────────────────────────────────────────────────────

    def create_story(
        self,
        person_id: str,
        username_at_capture: str,
        story_reference: dict | None = None,
        story_position: int | None = None,
    ) -> Story:
        """Create a new story record."""
        story_id = self._next_id("story")
        now = datetime.now(tz=timezone.utc)
        ref_json = json.dumps(story_reference) if story_reference else None
        self._conn.execute(
            """
            INSERT INTO stories
                (story_id, person_id, username_at_capture,
                 story_reference, detected_at, story_position,
                 capture_status, initial_analysis_status,
                 final_analysis_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                story_id,
                person_id,
                username_at_capture,
                ref_json,
                now.isoformat(),
                story_position,
                CaptureStatus.PENDING.value,
                AnalysisStatus.PENDING.value,
                AnalysisStatus.PENDING.value,
            ),
        )
        self._conn.commit()
        self.log_event(story_id, EventType.STORY_DETECTED)
        logger.info("Created story %s for person %s", story_id, person_id)
        return Story(
            story_id=story_id,
            person_id=person_id,
            username_at_capture=username_at_capture,
            story_reference=ref_json,
            detected_at=now,
            story_position=story_position,
        )

    def get_story(self, story_id: str) -> Story | None:
        """Look up a story by ID."""
        row = self._conn.execute(
            "SELECT * FROM stories WHERE story_id = ?", (story_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_story(row)

    def get_stories_for_person(self, person_id: str) -> list[Story]:
        """Return all stories for a person, ordered by detection time."""
        rows = self._conn.execute(
            "SELECT * FROM stories WHERE person_id = ? ORDER BY detected_at",
            (person_id,),
        ).fetchall()
        return [self._row_to_story(r) for r in rows]

    def update_story_status(
        self,
        story_id: str,
        *,
        capture_status: CaptureStatus | None = None,
        initial_analysis_status: AnalysisStatus | None = None,
        revisit_status: RevisitStatus | None = None,
        final_analysis_status: AnalysisStatus | None = None,
        opened_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """Update one or more status fields on a story."""
        updates: list[str] = []
        values: list[object] = []
        if capture_status is not None:
            updates.append("capture_status = ?")
            values.append(capture_status.value)
        if initial_analysis_status is not None:
            updates.append("initial_analysis_status = ?")
            values.append(initial_analysis_status.value)
        if revisit_status is not None:
            updates.append("revisit_status = ?")
            values.append(revisit_status.value)
        if final_analysis_status is not None:
            updates.append("final_analysis_status = ?")
            values.append(final_analysis_status.value)
        if opened_at is not None:
            updates.append("opened_at = ?")
            values.append(opened_at.isoformat())
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at.isoformat())
        if not updates:
            return
        values.append(story_id)
        self._conn.execute(
            f"UPDATE stories SET {', '.join(updates)} WHERE story_id = ?",
            values,
        )
        self._conn.commit()

    def find_duplicate_story(
        self,
        person_id: str,
        story_reference: dict | None = None,
        story_position: int | None = None,
    ) -> Story | None:
        """Attempt to find a duplicate story by reference or position.

        Returns the matching story or ``None``.
        """
        if story_reference is not None:
            ref_json = json.dumps(story_reference)
            row = self._conn.execute(
                """
                SELECT * FROM stories
                WHERE person_id = ? AND story_reference = ?
                ORDER BY detected_at DESC LIMIT 1
                """,
                (person_id, ref_json),
            ).fetchone()
            if row is not None:
                return self._row_to_story(row)
        if story_position is not None:
            row = self._conn.execute(
                """
                SELECT * FROM stories
                WHERE person_id = ? AND story_position = ?
                ORDER BY detected_at DESC LIMIT 1
                """,
                (person_id, story_position),
            ).fetchone()
            if row is not None:
                return self._row_to_story(row)
        return None

    def get_pending_stories(self) -> list[Story]:
        """Return stories that have not completed final analysis."""
        rows = self._conn.execute(
            """
            SELECT * FROM stories
            WHERE final_analysis_status IN (?, ?)
            ORDER BY detected_at
            """,
            (AnalysisStatus.PENDING.value, AnalysisStatus.IN_PROGRESS.value),
        ).fetchall()
        return [self._row_to_story(r) for r in rows]

    def get_incomplete_stories(self) -> list[Story]:
        """Return stories with incomplete capture or analysis for resume."""
        rows = self._conn.execute(
            """
            SELECT * FROM stories
            WHERE capture_status != ? OR final_analysis_status != ?
            ORDER BY detected_at
            """,
            (CaptureStatus.COMPLETED.value, AnalysisStatus.COMPLETED.value),
        ).fetchall()
        return [self._row_to_story(r) for r in rows]

    # ── frames ───────────────────────────────────────────────────────────

    def save_frame(
        self,
        story_id: str,
        capture_pass: CapturePass,
        frame_number: int,
        file_path: str,
        width: int | None = None,
        height: int | None = None,
    ) -> Frame:
        """Save a captured frame record."""
        frame_id = self._next_id("frame")
        now = datetime.now(tz=timezone.utc)
        self._conn.execute(
            """
            INSERT INTO frames
                (frame_id, story_id, capture_pass, frame_number,
                 captured_at, file_path, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                story_id,
                capture_pass.value,
                frame_number,
                now.isoformat(),
                file_path,
                width,
                height,
            ),
        )
        self._conn.commit()
        return Frame(
            frame_id=frame_id,
            story_id=story_id,
            capture_pass=capture_pass,
            frame_number=frame_number,
            captured_at=now,
            file_path=file_path,
            width=width,
            height=height,
        )

    def get_frames_for_story(
        self,
        story_id: str,
        capture_pass: CapturePass | None = None,
    ) -> list[Frame]:
        """Return frames for a story, optionally filtered by pass."""
        if capture_pass is not None:
            rows = self._conn.execute(
                """
                SELECT * FROM frames
                WHERE story_id = ? AND capture_pass = ?
                ORDER BY frame_number
                """,
                (story_id, capture_pass.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM frames WHERE story_id = ? ORDER BY frame_number",
                (story_id,),
            ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    # ── OCR ──────────────────────────────────────────────────────────────

    def save_ocr_result(
        self,
        frame_id: str,
        text: str,
        confidence: float | None = None,
        bounding_data: dict | list | None = None,
    ) -> OCRResult:
        """Save an OCR extraction result for a frame."""
        ocr_id = self._next_id("ocr")
        now = datetime.now(tz=timezone.utc)
        bd_json = json.dumps(bounding_data) if bounding_data is not None else None
        self._conn.execute(
            """
            INSERT INTO ocr_results
                (ocr_id, frame_id, text, confidence, bounding_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ocr_id, frame_id, text, confidence, bd_json, now.isoformat()),
        )
        self._conn.commit()
        return OCRResult(
            ocr_id=ocr_id,
            frame_id=frame_id,
            text=text,
            confidence=confidence,
            bounding_data=bd_json,
            created_at=now,
        )

    def get_ocr_for_frame(self, frame_id: str) -> list[OCRResult]:
        """Return OCR results for a single frame."""
        rows = self._conn.execute(
            "SELECT * FROM ocr_results WHERE frame_id = ?", (frame_id,)
        ).fetchall()
        return [self._row_to_ocr(r) for r in rows]

    def get_ocr_for_story(self, story_id: str) -> list[OCRResult]:
        """Return all OCR results across all frames of a story."""
        rows = self._conn.execute(
            """
            SELECT o.* FROM ocr_results o
            JOIN frames f ON o.frame_id = f.frame_id
            WHERE f.story_id = ?
            ORDER BY f.frame_number
            """,
            (story_id,),
        ).fetchall()
        return [self._row_to_ocr(r) for r in rows]

    # ── initial analysis ─────────────────────────────────────────────────

    def save_initial_analysis(
        self,
        story_id: str,
        model: str,
        summary: str | None = None,
        visible_information: list | dict | str | None = None,
        confidence: float | None = None,
        sampling_decision: SamplingDecision | None = None,
        revisit_priority: int | None = None,
        revisit_reason: str | None = None,
    ) -> InitialAnalysis:
        """Save the 4B screening result.  Never overwrites existing records."""
        analysis_id = self._next_id("initial_analysis")
        now = datetime.now(tz=timezone.utc)

        # Ensure list/dict is serialized to JSON string for SQLite
        if isinstance(visible_information, (list, dict)):
            visible_information = json.dumps(visible_information)

        self._conn.execute(
            """
            INSERT INTO initial_analyses
                (analysis_id, story_id, model, summary, visible_information,
                 confidence, sampling_decision, revisit_priority, revisit_reason,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                story_id,
                model,
                summary,
                visible_information,
                confidence,
                sampling_decision.value if sampling_decision else None,
                revisit_priority,
                revisit_reason,
                now.isoformat(),
            ),
        )
        self._conn.commit()
        return InitialAnalysis(
            analysis_id=analysis_id,
            story_id=story_id,
            model=model,
            summary=summary,
            visible_information=visible_information,
            confidence=confidence,
            sampling_decision=sampling_decision,
            revisit_priority=revisit_priority,
            revisit_reason=revisit_reason,
            created_at=now,
        )

    def get_initial_analysis(self, story_id: str) -> InitialAnalysis | None:
        """Return the initial analysis for a story (most recent)."""
        row = self._conn.execute(
            """
            SELECT * FROM initial_analyses
            WHERE story_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (story_id,),
        ).fetchone()
        if row is None:
            return None
        return InitialAnalysis(
            analysis_id=row["analysis_id"],
            story_id=row["story_id"],
            model=row["model"],
            summary=row["summary"],
            visible_information=row["visible_information"],
            confidence=row["confidence"],
            sampling_decision=(
                SamplingDecision(row["sampling_decision"])
                if row["sampling_decision"]
                else None
            ),
            revisit_priority=row["revisit_priority"],
            revisit_reason=row["revisit_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── revisits ─────────────────────────────────────────────────────────

    def queue_revisit(
        self,
        story_id: str,
        priority: int,
        reason: str | None = None,
    ) -> Revisit:
        """Add a story to the revisit queue."""
        revisit_id = self._next_id("revisit")
        now = datetime.now(tz=timezone.utc)
        self._conn.execute(
            """
            INSERT INTO revisits
                (revisit_id, story_id, required, priority, reason,
                 queued_at, attempts, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revisit_id,
                story_id,
                1,
                priority,
                reason,
                now.isoformat(),
                0,
                RevisitStatus.QUEUED.value,
            ),
        )
        self._conn.commit()
        self.log_event(
            story_id,
            EventType.REVISIT_REQUESTED,
            {"priority": priority, "reason": reason},
        )
        return Revisit(
            revisit_id=revisit_id,
            story_id=story_id,
            required=True,
            priority=priority,
            reason=reason,
            queued_at=now,
        )

    def get_revisit_queue(self) -> list[Revisit]:
        """Return pending revisits sorted by priority (highest first)."""
        rows = self._conn.execute(
            """
            SELECT * FROM revisits
            WHERE status = ?
            ORDER BY priority DESC
            """,
            (RevisitStatus.QUEUED.value,),
        ).fetchall()
        return [self._row_to_revisit(r) for r in rows]

    def update_revisit_status(
        self,
        revisit_id: str,
        status: RevisitStatus,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        increment_attempts: bool = False,
    ) -> None:
        """Update a revisit queue entry."""
        updates = ["status = ?"]
        values: list[object] = [status.value]
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at.isoformat())
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at.isoformat())
        if increment_attempts:
            updates.append("attempts = attempts + 1")
        values.append(revisit_id)
        self._conn.execute(
            f"UPDATE revisits SET {', '.join(updates)} WHERE revisit_id = ?",
            values,
        )
        self._conn.commit()

    # ── final analysis ───────────────────────────────────────────────────

    def save_final_analysis(
        self,
        story_id: str,
        model: str,
        content_type: str | None = None,
        description: str | None = None,
        important_information: str | None = None,
        visible_text: str | None = None,
        people_detected: list | None = None,
        objects_detected: list | None = None,
        context: str | None = None,
        confidence: float | None = None,
    ) -> FinalAnalysis:
        """Save the 8B detailed analysis result."""
        analysis_id = self._next_id("final_analysis")
        now = datetime.now(tz=timezone.utc)
        self._conn.execute(
            """
            INSERT INTO final_analyses
                (analysis_id, story_id, model, content_type, description,
                 important_information, visible_text, people_detected,
                 objects_detected, context, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                story_id,
                model,
                content_type,
                description,
                important_information,
                visible_text,
                json.dumps(people_detected) if people_detected else None,
                json.dumps(objects_detected) if objects_detected else None,
                context,
                confidence,
                now.isoformat(),
            ),
        )
        self._conn.commit()
        return FinalAnalysis(
            analysis_id=analysis_id,
            story_id=story_id,
            model=model,
            content_type=content_type,
            description=description,
            important_information=important_information,
            visible_text=visible_text,
            people_detected=json.dumps(people_detected) if people_detected else None,
            objects_detected=json.dumps(objects_detected) if objects_detected else None,
            context=context,
            confidence=confidence,
            created_at=now,
        )

    def get_final_analysis(self, story_id: str) -> FinalAnalysis | None:
        """Return the final analysis for a story (most recent)."""
        row = self._conn.execute(
            """
            SELECT * FROM final_analyses
            WHERE story_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (story_id,),
        ).fetchone()
        if row is None:
            return None
        return FinalAnalysis(
            analysis_id=row["analysis_id"],
            story_id=row["story_id"],
            model=row["model"],
            content_type=row["content_type"],
            description=row["description"],
            important_information=row["important_information"],
            visible_text=row["visible_text"],
            people_detected=row["people_detected"],
            objects_detected=row["objects_detected"],
            context=row["context"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── expert reviews ───────────────────────────────────────────────────

    def save_expert_review(
        self,
        story_id: str,
        model: str,
        reason: str | None = None,
        analysis: str | None = None,
        confidence: float | None = None,
    ) -> ExpertReview:
        """Save the optional 32B expert review."""
        review_id = self._next_id("expert_review")
        now = datetime.now(tz=timezone.utc)
        self._conn.execute(
            """
            INSERT INTO expert_reviews
                (review_id, story_id, model, reason, analysis,
                 confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (review_id, story_id, model, reason, analysis, confidence, now.isoformat()),
        )
        self._conn.commit()
        return ExpertReview(
            review_id=review_id,
            story_id=story_id,
            model=model,
            reason=reason,
            analysis=analysis,
            confidence=confidence,
            created_at=now,
        )

    def get_expert_review(self, story_id: str) -> ExpertReview | None:
        """Return the expert review for a story (most recent)."""
        row = self._conn.execute(
            """
            SELECT * FROM expert_reviews
            WHERE story_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (story_id,),
        ).fetchone()
        if row is None:
            return None
        return ExpertReview(
            review_id=row["review_id"],
            story_id=row["story_id"],
            model=row["model"],
            reason=row["reason"],
            analysis=row["analysis"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── events ───────────────────────────────────────────────────────────

    def log_event(
        self,
        story_id: str,
        event_type: EventType,
        data: dict | None = None,
    ) -> StoryEvent:
        """Log a processing event for a story."""
        event_id = self._next_id("event")
        now = datetime.now(tz=timezone.utc)
        data_json = json.dumps(data) if data else None
        self._conn.execute(
            """
            INSERT INTO story_events
                (event_id, story_id, event_type, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, story_id, event_type.value, now.isoformat(), data_json),
        )
        self._conn.commit()
        return StoryEvent(
            event_id=event_id,
            story_id=story_id,
            event_type=event_type,
            timestamp=now,
            data=data_json,
        )

    def get_events_for_story(self, story_id: str) -> list[StoryEvent]:
        """Return all events for a story in chronological order."""
        rows = self._conn.execute(
            "SELECT * FROM story_events WHERE story_id = ? ORDER BY timestamp",
            (story_id,),
        ).fetchall()
        return [
            StoryEvent(
                event_id=r["event_id"],
                story_id=r["story_id"],
                event_type=EventType(r["event_type"]),
                timestamp=datetime.fromisoformat(r["timestamp"]),
                data=r["data"],
            )
            for r in rows
        ]

    # ── resume helpers ───────────────────────────────────────────────────

    def get_processing_state(self) -> dict:
        """Return a summary of processing progress for resume logic.

        Returns::

            {
                "completed": [list of story_ids],
                "incomplete": [list of story_ids],
                "pending": [list of story_ids],
                "total": int,
            }
        """
        rows = self._conn.execute("SELECT * FROM stories").fetchall()
        completed = []
        incomplete = []
        pending = []
        for r in rows:
            sid = r["story_id"]
            fa = r["final_analysis_status"]
            cs = r["capture_status"]
            if fa == AnalysisStatus.COMPLETED.value:
                completed.append(sid)
            elif cs == CaptureStatus.PENDING.value and fa == AnalysisStatus.PENDING.value:
                pending.append(sid)
            else:
                incomplete.append(sid)
        return {
            "completed": completed,
            "incomplete": incomplete,
            "pending": pending,
            "total": len(rows),
        }

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> Person:
        return Person(
            person_id=row["person_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            current_username=row["current_username"],
            current_display_name=row["current_display_name"],
            current_pfp_path=row["current_pfp_path"],
            identity_status=IdentityStatus(row["identity_status"]),
        )

    @staticmethod
    def _row_to_story(row: sqlite3.Row) -> Story:
        return Story(
            story_id=row["story_id"],
            person_id=row["person_id"],
            username_at_capture=row["username_at_capture"],
            story_reference=row["story_reference"],
            detected_at=datetime.fromisoformat(row["detected_at"]),
            opened_at=(
                datetime.fromisoformat(row["opened_at"]) if row["opened_at"] else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            story_position=row["story_position"],
            capture_status=CaptureStatus(row["capture_status"]),
            initial_analysis_status=AnalysisStatus(row["initial_analysis_status"]),
            revisit_status=(
                RevisitStatus(row["revisit_status"]) if row["revisit_status"] else None
            ),
            final_analysis_status=AnalysisStatus(row["final_analysis_status"]),
        )

    @staticmethod
    def _row_to_frame(row: sqlite3.Row) -> Frame:
        return Frame(
            frame_id=row["frame_id"],
            story_id=row["story_id"],
            capture_pass=CapturePass(row["capture_pass"]),
            frame_number=row["frame_number"],
            captured_at=datetime.fromisoformat(row["captured_at"]),
            file_path=row["file_path"],
            width=row["width"],
            height=row["height"],
        )

    @staticmethod
    def _row_to_ocr(row: sqlite3.Row) -> OCRResult:
        return OCRResult(
            ocr_id=row["ocr_id"],
            frame_id=row["frame_id"],
            text=row["text"],
            confidence=row["confidence"],
            bounding_data=row["bounding_data"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_revisit(row: sqlite3.Row) -> Revisit:
        return Revisit(
            revisit_id=row["revisit_id"],
            story_id=row["story_id"],
            required=bool(row["required"]),
            priority=row["priority"],
            reason=row["reason"],
            queued_at=datetime.fromisoformat(row["queued_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"])
                if row["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            attempts=row["attempts"],
            status=RevisitStatus(row["status"]),
        )
