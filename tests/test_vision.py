"""
Tests for the vision model layer.

All tests run without torch/transformers — models are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.config import StoryZopConfig
from src.vision.base import VisionModel
from src.vision.gpu import GPUManager
from src.vision.qwen4b import Qwen4BScreener
from src.vision import qwen4b as qwen4b_module


class DummyVisionModel(VisionModel):
    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def _run_inference(self, images: list, prompt: str, max_tokens: int) -> str:
        return ""


@pytest.fixture()
def config():
    return StoryZopConfig()


# ── JSON parsing ─────────────────────────────────────────────────────────


def test_parse_json_response_valid(config):
    model = DummyVisionModel("test", config)
    res = model.parse_json_response('{"test": 123}')
    assert res == {"test": 123}


def test_parse_json_response_markdown(config):
    model = DummyVisionModel("test", config)
    res = model.parse_json_response('Here is the json: ```json\n{"test": 123}\n```')
    assert res == {"test": 123}


def test_parse_json_response_embedded(config):
    model = DummyVisionModel("test", config)
    res = model.parse_json_response('Some text before {"test": 123} and after.')
    assert res == {"test": 123}


def test_parse_json_response_malformed(config):
    model = DummyVisionModel("test", config)
    res = model.parse_json_response('{"test": 123, }')
    assert res == {"test": 123}


def test_parse_json_response_raises_value_error(config):
    model = DummyVisionModel("test", config)
    with pytest.raises(ValueError):
        model.parse_json_response("No json here")


# ── GPU manager ──────────────────────────────────────────────────────────


def test_gpu_manager_detect_gpu_no_torch():
    """Without torch, detect_gpu should return CPU info."""
    info = GPUManager.detect_gpu()
    assert info["device"] == "cpu"
    assert info["gpu_name"] is None


def test_gpu_manager_estimate_model_fit_no_torch():
    """Without torch, estimate_model_fit should return False."""
    assert GPUManager.estimate_model_fit("4b") is False
    assert GPUManager.estimate_model_fit("32b") is False


def test_gpu_manager_get_device_no_torch():
    """Without torch, get_device should return 'cpu'."""
    assert GPUManager.get_device() == "cpu"


def test_gpu_manager_clear_cache_no_torch():
    """Without torch, clear_gpu_cache should not raise."""
    GPUManager.clear_gpu_cache()  # Should be a no-op


# ── Model availability ───────────────────────────────────────────────────


def test_qwen4b_deps_available():
    """Without torch, _DEPS_AVAILABLE should be False."""
    assert not qwen4b_module._DEPS_AVAILABLE


def test_vision_model_subclass_instantiation(config):
    """Qwen4BScreener can be created even without torch (it won't load)."""
    model = Qwen4BScreener(config)
    assert model.config == config
    assert model.is_loaded is False


def test_vision_model_analyze_retry(config):
    """analyze() should retry on JSON parse failure."""
    model = DummyVisionModel("test", config)
    model._is_loaded = True

    # Make _run_inference return invalid JSON first, then valid
    responses = iter(["not json", "still bad", '{"ok": true}'])
    model._run_inference = lambda imgs, prompt, mt: next(responses)

    # With max_model_retries=2, it should succeed on the 3rd attempt
    config.max_model_retries = 2
    result = model.analyze([], "test")
    assert result == {"ok": True}


def test_vision_model_analyze_returns_error_on_exhausted_retries(config):
    """analyze() should return error dict when all retries fail."""
    model = DummyVisionModel("test", config)
    model._is_loaded = True
    model._run_inference = lambda imgs, prompt, mt: "never valid json"

    config.max_model_retries = 1
    result = model.analyze([], "test")
    assert "error" in result
    assert "raw_response" in result
