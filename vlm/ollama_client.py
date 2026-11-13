"""Ollama-backed VLM client for Qwen2.5-VL-3B via local Ollama instance.

Implements the VLMClient interface using Ollama's REST API
(http://localhost:11434/api/generate). Supports vision input via base64-encoded
images in the ``images`` field.

Does NOT replace QwenVLMClient — both clients coexist under the same interface.

Traceability:
  - Feature Specification: FS-005 (Unified VLM Interface & Adapters)
  - Research Paper: Section IV-B (Models and Deployment)
"""

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from vlm.base import VLMClient

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"

# System prompt matching QwenVLMClient for consistent structured output
SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Analyze the medical image and answer the question accurately. "
    "Respond ONLY with a valid JSON object matching this schema: "
    '{"answer": "string", "findings": [{"claim": "string", '
    '"region_id": "string", "status": "present|absent|uncertain", '
    '"confidence": "high|medium|low"}]} '
    "Ground findings to candidate region IDs visible in the image. "
    "Use 'NONE' for region_id if no region is relevant. "
    "Do not include any text outside the JSON object."
)


class OllamaVLMClient(VLMClient):
    """Ollama-backed VLM client for vision-language models (e.g. qwen2.5vl:3b).

    Communicates with a local Ollama instance via HTTP REST API. Images are
    base64-encoded as JPEG and passed in the ``images`` field of the generate
    request. Model configuration and runtime metrics are recorded in
    ``last_metadata`` for experiment traceability.

    Args:
        model: Ollama model tag (default: 'qwen2.5vl:3b').
        host: Ollama server base URL (default: 'http://localhost:11434').
        temperature: Default sampling temperature.
        max_tokens: Default maximum new tokens.
        timeout_s: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 300,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._device = "ollama"
        self.last_metadata: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return f"ollama-{self._model}"

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Ollama API helpers
    # ------------------------------------------------------------------

    def _image_to_b64(self, image: Image.Image) -> str:
        """Convert a PIL Image to base64-encoded JPEG string."""
        buf = io.BytesIO()
        # Convert to RGB if needed (Ollama expects standard RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Cap max dimension to avoid excessive payloads
        max_dim = 1024
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_prompt(self, question: str) -> str:
        """Build a single-prompt string combining system instructions + question."""
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

    def _call_generate(
        self,
        image_b64: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        """Call Ollama /api/generate and return (response_text, metadata)."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama API request failed: {type(exc).__name__}: {exc}. "
                f"Ensure Ollama is running at {self._host} and model '{self._model}' is installed."
            ) from exc
        wall_time = time.perf_counter() - t0

        result = json.loads(raw)
        response_text = str(result.get("response", ""))

        metadata = {
            "model": self._model,
            "host": self._host,
            "wall_time_s": round(wall_time, 4),
            "total_duration_s": round(result.get("total_duration", 0) / 1e9, 4),
            "load_duration_s": round(result.get("load_duration", 0) / 1e9, 4),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "eval_duration_s": round(result.get("eval_duration", 0) / 1e9, 4),
            "done": result.get("done"),
        }
        self.last_metadata = metadata
        return response_text, metadata

    # ------------------------------------------------------------------
    # VLMClient interface
    # ------------------------------------------------------------------

    def generate(
        self,
        marked_image: Image.Image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate a single response from the Ollama model."""
        if not isinstance(marked_image, Image.Image):
            raise TypeError(
                f"marked_image must be a PIL.Image.Image, got {type(marked_image).__name__}"
            )

        image_b64 = self._image_to_b64(marked_image)
        prompt = self._build_prompt(question)
        response_text, _ = self._call_generate(
            image_b64, prompt, temperature, max_tokens
        )
        return response_text

    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Generate N sampled responses from the Ollama model."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        image_b64 = self._image_to_b64(marked_image)
        prompt = self._build_prompt(question)
        responses: list[str] = []
        for _ in range(n):
            response_text, _ = self._call_generate(
                image_b64, prompt, temperature, max_tokens
            )
            responses.append(response_text)
        return responses
