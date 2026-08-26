"""
Central configuration for StoryZop.

All configurable values are defined here and loaded from environment variables
or .env files. Defaults are provided for all settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class StoryZopConfig(BaseSettings):
    """Application-wide configuration.

    Values can be overridden via environment variables prefixed with ``SZ_``
    or via a ``.env`` file in the project root.
    """

    model_config = {"env_prefix": "SZ_", "env_file": ".env", "extra": "ignore"}

    # ── Project paths ────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Root directory of the project.",
    )
    data_dir: Path | None = Field(default=None, description="Root data directory.")
    db_path: Path | None = Field(default=None, description="SQLite database path.")

    def model_post_init(self, __context: object) -> None:
        """Resolve derived paths after init."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.db_path is None:
            self.db_path = self.data_dir / "storyzop.db"

    # ── Capture settings ─────────────────────────────────────────────────
    initial_max_frames: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum frames to capture in the initial pass.",
    )
    revisit_max_frames: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Maximum frames to capture during a revisit.",
    )
    capture_interval_ms: int = Field(
        default=1500,
        ge=500,
        description="Milliseconds between frame captures.",
    )

    # ── AI model identifiers ─────────────────────────────────────────────
    initial_model: str = Field(
        default="Qwen/Qwen3-VL-4B-Instruct",
        description="HuggingFace model ID for the fast screening model.",
    )
    primary_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="HuggingFace model ID for the primary analysis model.",
    )
    expert_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="HuggingFace model ID for the difficult-case expert model.",
    )

    # ── Analysis thresholds ──────────────────────────────────────────────
    expert_review_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers expert (32B) review.",
    )
    ocr_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence to keep a result.",
    )

    # ── Browser settings ─────────────────────────────────────────────────
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode.",
    )
    browser_timeout_ms: int = Field(
        default=30_000,
        ge=5_000,
        description="Default browser operation timeout in milliseconds.",
    )
    viewport_width: int = Field(default=412, description="Browser viewport width.")
    viewport_height: int = Field(default=915, description="Browser viewport height.")
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Mobile Safari/537.36"
        ),
        description="Browser user-agent string.",
    )

    # ── Authentication ───────────────────────────────────────────────────
    session_cookie_path: Path | None = Field(
        default=None,
        description="Path to exported cookies JSON file.",
    )
    browser_state_path: Path | None = Field(
        default=None,
        description="Path to Playwright persistent browser state directory.",
    )

    # ── OCR settings ─────────────────────────────────────────────────────
    ocr_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for OCR recognition.",
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level.",
    )

    # ── GPU / model loading ──────────────────────────────────────────────
    model_dtype: str = Field(
        default="auto",
        description="Torch dtype for model loading (auto, float16, bfloat16).",
    )
    use_4bit_quantization: bool = Field(
        default=False,
        description="Use 4-bit quantization via bitsandbytes.",
    )
    max_model_retries: int = Field(
        default=2,
        ge=0,
        description="Max retries for malformed model JSON output.",
    )


def get_config(**overrides: object) -> StoryZopConfig:
    """Create a config instance with optional overrides.

    Example::

        cfg = get_config(initial_max_frames=6, headless=False)
    """
    return StoryZopConfig(**overrides)  # type: ignore[arg-type]
