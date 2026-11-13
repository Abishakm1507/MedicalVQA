"""Mock VLM client for lightweight local testing.

Produces deterministic JSON responses without loading any model weights.
Intended for CI/CD and unit testing only — never used as a silent fallback
for a real model.

Traceability:
  - Feature Specification: FS-005 (Unified VLM Interface & Adapters)
"""

import json

from PIL import Image

from vlm.base import VLMClient


class MockVLMClient(VLMClient):
    """Deterministic mock VLM client for testing.

    Returns structured JSON responses without any model weights. Suitable
    for CI/CD, unit tests, and offline development.

    Args:
        model_name: Identifier reported by model_name property.
        device: Device string (informational only, no actual device used).
        default_answer: Answer text to return in all responses.
        findings: Optional list of finding dicts to include in responses.
    """

    def __init__(
        self,
        model_name: str = "mock-vlm",
        device: str = "cpu",
        default_answer: str = "Mock answer — not a real model prediction.",
        findings: list[dict] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._default_answer = default_answer
        self._findings = findings or []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def _make_response(self, question: str) -> str:
        """Build a deterministic JSON response string."""
        output = {
            "answer": self._default_answer,
            "findings": self._findings,
        }
        return json.dumps(output)

    def generate(
        self,
        marked_image: Image.Image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Return a deterministic mock JSON response."""
        return self._make_response(question)

    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Return N identical mock JSON responses."""
        response = self._make_response(question)
        return [response] * n
