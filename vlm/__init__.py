"""VLM package providing unified interfaces for Qwen2.5-VL-3B, MedGemma-4B, and test mocks.

Lazy-export pattern: importing `vlm` does not pull in torch or transformers.
Heavyweight clients (QwenVLMClient) are only imported when explicitly accessed.
"""

from typing import Any

__all__ = [
    # VLM interface
    "VLMClient",
    # Concrete clients
    "QwenVLMClient",
    "MockVLMClient",
    "OllamaVLMClient",
]


def __getattr__(name: str) -> Any:
    # Lightweight interface — no heavy ML dependencies required
    if name == "VLMClient":
        from vlm.base import VLMClient

        globals()["VLMClient"] = VLMClient
        return VLMClient

    # QwenVLMClient — requires torch + transformers (lazy)
    if name == "QwenVLMClient":
        from vlm.qwen_client import QwenVLMClient

        globals()["QwenVLMClient"] = QwenVLMClient
        return QwenVLMClient

    # MockVLMClient — lightweight, no heavy dependencies
    if name == "MockVLMClient":
        from vlm.mock_client import MockVLMClient

        globals()["MockVLMClient"] = MockVLMClient
        return MockVLMClient

    # OllamaVLMClient — lightweight, only urllib + PIL
    if name == "OllamaVLMClient":
        from vlm.ollama_client import OllamaVLMClient

        globals()["OllamaVLMClient"] = OllamaVLMClient
        return OllamaVLMClient

    raise AttributeError(f"module 'vlm' has no attribute '{name}'")
