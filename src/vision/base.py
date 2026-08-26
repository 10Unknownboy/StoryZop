from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from src.logger import get_logger
from src.config import StoryZopConfig

logger = get_logger(__name__)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class VisionModel(ABC):
    def __init__(self, model_id: str, config: StoryZopConfig) -> None:
        self.model_id = model_id
        self.config = config
        self._is_loaded = False
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    @abstractmethod
    def load_model(self) -> None:
        pass
        
    @abstractmethod
    def unload_model(self) -> None:
        pass
        
    @abstractmethod
    def _run_inference(self, images: list, prompt: str, max_tokens: int) -> str:
        pass
        
    def analyze(self, images: list[str | Path], prompt: str, max_tokens: int = 2048) -> dict:
        loaded_images = self._load_images(images)
        for attempt in range(self.config.max_model_retries + 1):
            try:
                raw_text = self._run_inference(loaded_images, prompt, max_tokens)
                return self.parse_json_response(raw_text)
            except ValueError as e:
                logger.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_model_retries:
                    return {"error": str(e), "raw_response": raw_text}
        return {"error": "Max retries exceeded", "raw_response": ""}

    def parse_json_response(self, raw_text: str) -> dict:
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        import re
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start:end+1])
            except json.JSONDecodeError:
                pass
                
        try:
            repaired = re.sub(r",\s*}", "}", raw_text[start:end+1])
            return json.loads(repaired)
        except Exception:
            pass

        raise ValueError("Could not parse JSON from response")

    def _load_images(self, image_paths: list[str | Path]) -> list:
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is not available.")
        loaded = []
        for p in image_paths:
            img = Image.open(str(p)).convert("RGB")
            loaded.append(img)
        return loaded
