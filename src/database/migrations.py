"""
Database schema migrations for StoryZop.

Provides incremental, versioned schema management.  Each migration is a
function that receives a ``sqlite3.Connection`` and applies DDL statements.
The current schema version is tracked in a ``_schema_version`` table.
"""

from __future__ import annotations

import sqlite3

from src.logger import get_logger

logger = get_logger("migrations")

CURRENT_VERSION = 1


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to bring the database up to date."""
    # Ensure the version-tracking table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _schema_version (
            version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    row = conn.execute("SELECT version FROM _schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO _schema_version (version) VALUES (0)")
        conn.commit()
        current = 0
    else:
        current = row[0]

    migrations = {
        1: _migrate_v1,
    }

    for ver in range(current + 1, CURRENT_VERSION + 1):
        if ver in migrations:
            logger.info("Applying migration v%d", ver)
            migrations[ver](conn)
            conn.execute("UPDATE _schema_version SET version = ?", (ver,))
            conn.commit()
            logger.info("Migration v%d applied", ver)


# ── Migration v1: initial schema ─────────────────────────────────────────


def _migrate_v1(conn: sqlite3.Connection) -> None:
    """Create all initial tables for StoryZop."""

    # Atomic ID counter table
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS id_counters (
            entity   TEXT PRIMARY KEY,
            counter  INTEGER NOT NULL DEFAULT 1
        );

        -- ────────────────────────────────────────────────────────────────
        -- persons
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS persons (
            person_id            TEXT PRIMARY KEY,
            created_at           TEXT NOT NULL,
            last_seen            TEXT NOT NULL,
            current_username     TEXT NOT NULL,
            current_display_name TEXT,
            current_pfp_path     TEXT,
            identity_status      TEXT NOT NULL DEFAULT 'UNVERIFIED'
        );
        CREATE INDEX IF NOT EXISTS idx_persons_username
            ON persons (current_username);

        -- ────────────────────────────────────────────────────────────────
        -- usernames (history)
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS usernames (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  TEXT NOT NULL REFERENCES persons(person_id),
            username   TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen  TEXT NOT NULL,
            UNIQUE(person_id, username)
        );
        CREATE INDEX IF NOT EXISTS idx_usernames_username
            ON usernames (username);

        -- ────────────────────────────────────────────────────────────────
        -- stories
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS stories (
            story_id                TEXT PRIMARY KEY,
            person_id               TEXT NOT NULL REFERENCES persons(person_id),
            username_at_capture     TEXT NOT NULL,
            story_reference         TEXT,
            detected_at             TEXT NOT NULL,
            opened_at               TEXT,
            completed_at            TEXT,
            story_position          INTEGER,
            capture_status          TEXT NOT NULL DEFAULT 'PENDING',
            initial_analysis_status TEXT NOT NULL DEFAULT 'PENDING',
            revisit_status          TEXT,
            final_analysis_status   TEXT NOT NULL DEFAULT 'PENDING'
        );
        CREATE INDEX IF NOT EXISTS idx_stories_person
            ON stories (person_id);

        -- ────────────────────────────────────────────────────────────────
        -- frames
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS frames (
            frame_id     TEXT PRIMARY KEY,
            story_id     TEXT NOT NULL REFERENCES stories(story_id),
            capture_pass TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            captured_at  TEXT NOT NULL,
            file_path    TEXT NOT NULL,
            width        INTEGER,
            height       INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_frames_story
            ON frames (story_id);

        -- ────────────────────────────────────────────────────────────────
        -- ocr_results
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS ocr_results (
            ocr_id        TEXT PRIMARY KEY,
            frame_id      TEXT NOT NULL REFERENCES frames(frame_id),
            text          TEXT NOT NULL,
            confidence    REAL,
            bounding_data TEXT,
            created_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ocr_frame
            ON ocr_results (frame_id);

        -- ────────────────────────────────────────────────────────────────
        -- initial_analyses
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS initial_analyses (
            analysis_id        TEXT PRIMARY KEY,
            story_id           TEXT NOT NULL REFERENCES stories(story_id),
            model              TEXT NOT NULL,
            summary            TEXT,
            visible_information TEXT,
            confidence         REAL,
            sampling_decision  TEXT,
            revisit_priority   INTEGER,
            revisit_reason     TEXT,
            created_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_initial_story
            ON initial_analyses (story_id);

        -- ────────────────────────────────────────────────────────────────
        -- revisits
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS revisits (
            revisit_id   TEXT PRIMARY KEY,
            story_id     TEXT NOT NULL REFERENCES stories(story_id),
            required     INTEGER NOT NULL DEFAULT 1,
            priority     INTEGER NOT NULL DEFAULT 5,
            reason       TEXT,
            queued_at    TEXT NOT NULL,
            started_at   TEXT,
            completed_at TEXT,
            attempts     INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT 'QUEUED'
        );
        CREATE INDEX IF NOT EXISTS idx_revisit_story
            ON revisits (story_id);
        CREATE INDEX IF NOT EXISTS idx_revisit_status
            ON revisits (status, priority);

        -- ────────────────────────────────────────────────────────────────
        -- final_analyses
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS final_analyses (
            analysis_id           TEXT PRIMARY KEY,
            story_id              TEXT NOT NULL REFERENCES stories(story_id),
            model                 TEXT NOT NULL,
            content_type          TEXT,
            description           TEXT,
            important_information TEXT,
            visible_text          TEXT,
            people_detected       TEXT,
            objects_detected      TEXT,
            context               TEXT,
            confidence            REAL,
            created_at            TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_final_story
            ON final_analyses (story_id);

        -- ────────────────────────────────────────────────────────────────
        -- expert_reviews
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS expert_reviews (
            review_id  TEXT PRIMARY KEY,
            story_id   TEXT NOT NULL REFERENCES stories(story_id),
            model      TEXT NOT NULL,
            reason     TEXT,
            analysis   TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_expert_story
            ON expert_reviews (story_id);

        -- ────────────────────────────────────────────────────────────────
        -- story_events
        -- ────────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS story_events (
            event_id   TEXT PRIMARY KEY,
            story_id   TEXT NOT NULL REFERENCES stories(story_id),
            event_type TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            data       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_story
            ON story_events (story_id);
        CREATE INDEX IF NOT EXISTS idx_events_type
            ON story_events (event_type);
        """
    )
