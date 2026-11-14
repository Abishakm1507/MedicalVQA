"""VLM package for MedicalVQA (minimal bundle for Kaggle inference).

Lazy-export pattern: importing `vlm` does not pull in torch or transformers.
MedGemmaVLMClient is only imported when explicitly accessed.
"""

from typing import Any

__all__ = [
    "VLMClient",
    "MedGemmaVLMClient",
]


def __getattr__(name: str) -> Any:
    if name == "VLMClient":
        from vlm.base import VLMClient

        globals()["VLMClient"] = VLMClient
        return VLMClient

    if name == "MedGemmaVLMClient":
        from vlm.medgemma_client import MedGemmaVLMClient

        globals()["MedGemmaVLMClient"] = MedGemmaVLMClient
        return MedGemmaVLMClient

    raise AttributeError(f"module 'vlm' has no attribute '{name}'")
