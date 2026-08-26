from __future__ import annotations

import pytest
import sys
from unittest.mock import patch, MagicMock

from src.config import StoryZopConfig
from src.vision.base import VisionModel
from src.vision.gpu import GPUManager
from src.vision.qwen4b import Qwen4BScreener
from src.vision import qwen4b

class DummyVisionModel(VisionModel):
    def load_model(self) -> None:
        pass
        
    def unload_model(self) -> None:
        pass
        
    def _run_inference(self, images: list, prompt: str, max_tokens: int) -> str:
        return ""

@pytest.fixture
def config():
    return StoryZopConfig()

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
    # trailing comma
    res = model.parse_json_response('{"test": 123, }')
    assert res == {"test": 123}

def test_parse_json_response_raises_value_error(config):
    model = DummyVisionModel("test", config)
    with pytest.raises(ValueError):
        model.parse_json_response('No json here')

@patch.dict('sys.modules', {'torch': MagicMock()})
def test_gpu_manager_detect_gpu():
    import src.vision.gpu as gpu_module
    gpu_module._TORCH_AVAILABLE = True
    gpu_module.torch.cuda.is_available.return_value = False
    
    info = gpu_module.GPUManager.detect_gpu()
    assert info["device"] == "cpu"

@patch.dict('sys.modules', {'torch': MagicMock()})
def test_gpu_manager_estimate_model_fit():
    import src.vision.gpu as gpu_module
    gpu_module._TORCH_AVAILABLE = True
    gpu_module.torch.cuda.is_available.return_value = True
    
    # Mock get_device_properties
    mock_props = MagicMock()
    mock_props.total_memory = 24.0 * (1024**3)
    gpu_module.torch.cuda.get_device_properties.return_value = mock_props
    gpu_module.torch.cuda.memory_allocated.return_value = 0.0
    
    assert gpu_module.GPUManager.estimate_model_fit("4b") is True
    # 32b unquantized needs 32GB, we have 24
    assert gpu_module.GPUManager.estimate_model_fit("32b", quantized=False) is False

def test_qwen4b_deps_available():
    # Without torch, _DEPS_AVAILABLE should be False
    assert not qwen4b._DEPS_AVAILABLE

def test_vision_model_subclass_instantiation(config):
    model = Qwen4BScreener(config)
    assert model.config == config
