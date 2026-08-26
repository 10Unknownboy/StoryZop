from __future__ import annotations

from pathlib import Path
from typing import Any

from src.database.database import Database
from src.database.models import EventType, Frame, OCRResult
from src.logger import get_logger

logger = get_logger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr is not installed. OCR capabilities will be disabled.")


class OCREngine:
    """Extracts text from images using EasyOCR."""

    def __init__(
        self, languages: list[str] | None = None, confidence_threshold: float = 0.3
    ):
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self._reader_instance: Any | None = None

        if not EASYOCR_AVAILABLE:
            logger.warning("OCREngine initialized, but easyocr is unavailable.")

    @property
    def is_available(self) -> bool:
        """Return whether easyocr is installed and available."""
        return EASYOCR_AVAILABLE

    @property
    def _reader(self) -> Any | None:
        if self._reader_instance is None and EASYOCR_AVAILABLE:
            self._reader_instance = easyocr.Reader(self.languages, gpu=True)
        return self._reader_instance

    def extract_text(self, image_path: str | Path) -> list[dict]:
        """Extract text from a single image.

        Returns a list of dicts: {'text': str, 'confidence': float, 'bounding_box': list}
        """
        if not self.is_available or self._reader is None:
            return []

        try:
            results = self._reader.readtext(str(image_path))
        except Exception as e:
            logger.error("OCR failed for %s: %s", image_path, e)
            return []

        extracted = []
        for bbox, text, conf in results:
            if conf >= self.confidence_threshold:
                # Convert coordinates for JSON serialization
                bbox_native = [[float(p[0]), float(p[1])] for p in bbox]
                extracted.append(
                    {
                        "text": text,
                        "confidence": float(conf),
                        "bounding_box": bbox_native,
                    }
                )

        return extracted

    def extract_text_for_story(
        self, frames: list[Frame], db: Database, story_id: str
    ) -> list[OCRResult]:
        """Process all frames for a story and save results to the DB."""
        ocr_results: list[OCRResult] = []
        for frame in frames:
            text_data = self.extract_text(frame.file_path)
            for item in text_data:
                res = db.save_ocr_result(
                    frame_id=frame.frame_id,
                    text=item["text"],
                    confidence=item["confidence"],
                    bounding_data=item["bounding_box"],
                )
                ocr_results.append(res)

        db.log_event(story_id, EventType.OCR_COMPLETED)
        return ocr_results

    def format_ocr_for_prompt(self, ocr_results: list[OCRResult]) -> str:
        """Format OCR results as a human-readable context string for VLM prompts."""
        if not ocr_results:
            return "No text detected."

        grouped: dict[str, list[OCRResult]] = {}
        for res in ocr_results:
            grouped.setdefault(res.frame_id, []).append(res)

        lines = []
        for i, (frame_id, results) in enumerate(grouped.items(), start=1):
            for res in results:
                conf = res.confidence if res.confidence is not None else 0.0
                lines.append(f'Frame {i} ({frame_id}, confidence {conf:.2f}): "{res.text}"')

        return "\n".join(lines)
