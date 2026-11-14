"""Unified Vision-Language Model client interface.

Defines the abstract VLMClient interface and concrete implementations for
Qwen2.5-VL-3B and a deterministic mock for testing.

Traceability:
  - Feature Specification: FS-005 (Unified VLM Interface & Adapters)
  - Research Paper: Section III-C, Section IV-B
"""

from abc import ABC, abstractmethod

from PIL import Image

from core.schemas import VLMStructuredOutput


class VLMClient(ABC):
    """Abstract base class for vision-language model clients.

    Provides a common, model-agnostic interface for prompting VLMs with
    identical prompts, images, sampling parameters, and schema constraints.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model (e.g. 'qwen2.5-vl-3b', 'medgemma-4b')."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Inference device (e.g. 'cpu', 'cuda')."""

    @abstractmethod
    def generate(
        self,
        marked_image: Image.Image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generates a single response string adhering to the JSON schema.

        Args:
            marked_image: PIL Image with Set-of-Mark visual overlays.
            question: The medical VQA question.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum new tokens to generate.

        Returns:
            Raw JSON string from the model.
        """

    @abstractmethod
    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Generates N sampled response strings for self-consistency.

        Args:
            marked_image: PIL Image with Set-of-Mark visual overlays.
            question: The medical VQA question.
            n: Number of samples to generate.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens per sample.

        Returns:
            List of N raw JSON strings from the model.
        """

    def generate_structured(
        self,
        marked_image: Image.Image,
        question: str,
        region_ids: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> VLMStructuredOutput:
        """Generates and parses a structured VLM response.

        Args:
            marked_image: PIL Image with Set-of-Mark visual overlays.
            question: The medical VQA question.
            region_ids: Candidate region IDs available for grounding (e.g. ['R1', 'R2']).
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Parsed VLMStructuredOutput object.
        """
        from core.schemas import VLMClaim

        raw = self.generate(
            marked_image=marked_image,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse JSON from model output (handle markdown code fences)
        import json
        import re

        text = raw.strip()
        # Extract JSON from ```json ... ``` blocks if present
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Try to find a JSON object anywhere in the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Fallback: wrap raw text as answer with no findings
            return VLMStructuredOutput(answer=text, findings=[])

        # Build claims with region validation
        findings: list[VLMClaim] = []
        valid_region_ids = set(region_ids or [])
        valid_region_ids.add("NONE")

        for item in data.get("findings", []):
            region_id = str(item.get("region_id", "NONE"))
            if region_id not in valid_region_ids:
                region_id = "NONE"
            findings.append(
                VLMClaim(
                    claim=str(item.get("claim", "")),
                    region_id=region_id,
                    status=str(item.get("status", "present")),
                    confidence=str(item.get("confidence", "high")),
                )
            )

        return VLMStructuredOutput(
            answer=str(data.get("answer", "")),
            findings=findings,
        )
