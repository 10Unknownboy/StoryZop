"""
Centralized logging configuration for StoryZop.

Usage::

    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Browser initialized")
"""

from __future__ import annotations

import logging
import re


# Patterns that should never appear in logs
_SENSITIVE_PATTERNS = re.compile(
    r"(session[_-]?id|cookie|token|password|secret|api[_-]?key)"
    r"\s*[:=]\s*\S+",
    re.IGNORECASE,
)


class _SensitiveFilter(logging.Filter):
    """Redact sensitive values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SENSITIVE_PATTERNS.sub("[REDACTED]", record.msg)
        return True


_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root ``storyzop`` logger.

    Call once at application start-up (idempotent).
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    root = logging.getLogger("storyzop")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_SensitiveFilter())
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``storyzop`` namespace.

    Automatically calls :func:`setup_logging` if not yet configured.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"storyzop.{name}")
