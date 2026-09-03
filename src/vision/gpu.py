from __future__ import annotations

from src.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not found, GPU features disabled.")

class GPUManager:
    @staticmethod
    def detect_gpu() -> dict:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return {
                "gpu_name": None,
                "vram_total_gb": 0.0,
                "vram_available_gb": 0.0,
                "cuda_version": None,
                "pytorch_version": torch.__version__ if _TORCH_AVAILABLE else None,
                "device": "cpu"
            }
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory / (1024**3)
        allocated_vram = torch.cuda.memory_allocated(device) / (1024**3)
        available_vram = total_vram - allocated_vram

        return {
            "gpu_name": props.name,
            "vram_total_gb": total_vram,
            "vram_available_gb": available_vram,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device": "cuda"
        }

    @staticmethod
    def print_gpu_status() -> None:
        info = GPUManager.detect_gpu()
        if info["device"] == "cpu":
            print("CPU only")
        else:
            print(f"GPU: {info['gpu_name']} | VRAM: {info['vram_available_gb']:.1f}GB / {info['vram_total_gb']:.1f}GB | "
                  f"CUDA: {info['cuda_version']} | PyTorch: {info['pytorch_version']}")
            
    @staticmethod
    def estimate_model_fit(model_name: str, quantized: bool = False) -> bool:
        """Estimate whether a model can fit in available GPU VRAM.

        Rough VRAM requirements (including overhead):
            4B fp16 ~9GB,  4B 4-bit ~4GB
            8B fp16 ~17GB, 8B 4-bit ~6GB
            32B fp16 ~65GB, 32B 4-bit ~20GB
        """
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return False

        info = GPUManager.detect_gpu()
        vram = info["vram_available_gb"]

        if "32b" in model_name.lower():
            req = 20.0 if quantized else 65.0
        elif "8b" in model_name.lower():
            req = 6.0 if quantized else 17.0
        elif "4b" in model_name.lower():
            req = 4.0 if quantized else 9.0
        else:
            req = 8.0

        return vram >= req

    @staticmethod
    def clear_gpu_cache() -> None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @staticmethod
    def get_device() -> str:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
