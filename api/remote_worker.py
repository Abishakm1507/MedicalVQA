"""Remote cloud GPU inference worker for free Colab/Kaggle runtimes.

Serves Qwen2.5-VL-3B and MedGemma-4B in 4-bit precision via Hugging Face
Transformers and bitsandbytes. Designed for Google Colab (Tesla T4, 15.3 GB
VRAM) or Kaggle (P100) runtimes.

Strict GPU Memory Management (15.3 GB VRAM Ceiling):
  - Models loaded sequentially on demand.
  - Prior model unallocated and torch.cuda.empty_cache() called on switch.
  - MedSAM can run locally on CPU or on Colab before VLM loading.

Traceability:
  - Feature Specification: FS-013 (Remote Cloud GPU Inference Worker)
  - Research Paper: Section IV-B (Models and Deployment)
"""

import time

import torch
from PIL import Image

from vlm.qwen_client import QwenVLMClient


class RemoteModelWorker:
    """Standalone server/worker for remote VLM inference on free GPU runtimes.

    Loads models in 4-bit quantization and manages VRAM strictly to avoid OOM
    on free Colab T4 (15.3 GB) or Kaggle P100 instances.

    Args:
        auth_token: Authentication token for the worker (simple bearer token).
        hf_token: Optional Hugging Face token for gated models
            (e.g. google/medgemma-4b-it).
    """

    def __init__(
        self,
        auth_token: str,
        hf_token: str | None = None,
    ) -> None:
        self.auth_token = auth_token
        self.hf_token = hf_token
        self.active_model_name: str | None = None
        self.client: QwenVLMClient | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self, model_name: str) -> None:
        """Load requested model in 4-bit, unloading prior model if needed.

        Args:
            model_name: Model identifier ('qwen2.5-vl-3b' or 'medgemma-4b').
        """
        if self.active_model_name == model_name and self.client is not None:
            return

        # Unload prior model to preserve VRAM
        self.unload_model()

        if model_name == "qwen2.5-vl-3b":
            self.client = QwenVLMClient(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                device="cuda",
                quantization="4bit",
                trust_remote_code=True,
            )
            self.client._load()
            self.active_model_name = model_name
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                "Only 'qwen2.5-vl-3b' is currently supported."
            )

    def unload_model(self) -> None:
        """Unload the active model and clear CUDA cache."""
        if self.client is not None:
            self.client._model = None
            self.client._processor = None
            self.client = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.active_model_name = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        image_b64: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate a single response for a base64-encoded image.

        Args:
            model_name: Model to use for inference.
            image_b64: Base64-encoded image string.
            prompt: The VQA question / prompt text.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum new tokens.

        Returns:
            Raw JSON string from the model.
        """
        self.load_model(model_name)

        import base64
        import io

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        t0 = time.perf_counter()
        response = self.client.generate(
            marked_image=image,
            question=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - t0

        print(f"Inference completed in {elapsed:.2f}s")
        return response

    def sample_n(
        self,
        model_name: str,
        image_b64: str,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Generate N sampled responses for self-consistency.

        Args:
            model_name: Model to use for inference.
            image_b64: Base64-encoded image string.
            prompt: The VQA question / prompt text.
            n: Number of samples.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens per sample.

        Returns:
            List of N raw JSON strings from the model.
        """
        self.load_model(model_name)

        import base64
        import io

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        t0 = time.perf_counter()
        responses = self.client.sample_n(
            marked_image=image,
            question=prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - t0

        print(f"Sampled {n} responses in {elapsed:.2f}s")
        return responses

    def get_memory_usage(self) -> dict:
        """Return current GPU memory usage statistics."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(total - reserved, 2),
        }
