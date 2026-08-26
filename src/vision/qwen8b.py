from __future__ import annotations

from pathlib import Path

from src.logger import get_logger
from src.config import StoryZopConfig
from src.vision.base import VisionModel
from src.vision.gpu import GPUManager

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class Qwen8BAnalyzer(VisionModel):
    def __init__(self, config: StoryZopConfig) -> None:
        super().__init__(model_id=config.primary_model, config=config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        if not _DEPS_AVAILABLE:
            raise RuntimeError("torch/transformers/qwen_vl_utils not installed.")
        logger.info(f"Loading {self.model_id}...")
        
        dtype = torch.float16 if self.config.model_dtype == "float16" else torch.bfloat16 if self.config.model_dtype == "bfloat16" else "auto"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._is_loaded = True
        
    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self._is_loaded = False
        GPUManager.clear_gpu_cache()

    def _run_inference(self, images: list, prompt: str, max_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(GPUManager.get_device())

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def analyze_story(self, image_paths: list[str | Path], person_info: dict, ocr_context: str, initial_analysis: dict | None = None) -> dict:
        if not self.is_loaded:
            self.load_model()
            
        prompt = (
            "Analyze these story frames. Return a JSON object with the following keys exactly: "
            "'content_type', 'description', 'important_information', 'visible_text', "
            "'people_detected', 'objects_detected', 'context', 'confidence'.\n"
            "DO NOT produce a timeline, focus on meaningful content.\n"
        )
        prompt += f"Person Info: {person_info}\n"
        if ocr_context:
            prompt += f"OCR Context:\n{ocr_context}\n"
        if initial_analysis:
            prompt += f"Initial Analysis:\n{initial_analysis}\n"
            
        return self.analyze(image_paths, prompt)
