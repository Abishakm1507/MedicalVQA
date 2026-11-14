"""Claim extraction from structured VLM responses.

Extracts atomic, verifiable claims from an already-Pydantic-validated
``VLMStructuredOutput`` into canonical internal claim records. Performs
deterministic parsing only -- no external LLM or API dependency is used for
extraction.

Traceability:
  - Feature Specification: FS-005 (Phase 1), section 4.1
  - Research Paper: Section III-C (Region Labelling and Structured Answers)
"""

from dataclasses import dataclass, field
from typing import Literal

from core.schemas import VLMStructuredOutput


@dataclass
class ExtractedClaim:
    """A single atomic claim extracted from a structured VLM response."""

    claim: str
    region_id: str  # 'R1'..'Rk' or 'NONE'
    status: Literal["present", "absent", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    raw_claim_text: str | None  # verbatim claim text, pre-normalization


@dataclass
class ExtractionResult:
    """Result of extracting claims from a structured VLM response."""

    answer_text: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    parse_success: bool = True
    error: str | None = None


def _normalize_answer_text(text: str) -> str:
    """Lightly normalize answer text for downstream comparison.

    Strips surrounding whitespace and collapses internal runs of whitespace.
    This mirrors the conservative normalization used at the FS-004 metrics
    layer (strip, collapse whitespace) without applying aggressive transforms
    (lowercasing, punctuation stripping) that belong to the evaluation layer.
    """
    return " ".join(text.split())


def extract_claims(
    vlm_output: VLMStructuredOutput | None,
    raw_response: str | None = None,
) -> ExtractionResult:
    """Extract claims from a structured VLM response.

    Args:
        vlm_output: Pydantic-validated structured VLM output. If ``None``,
            extraction is recorded as a failure.
        raw_response: The raw model-generated string, used only for failure
            diagnostics when ``vlm_output`` is unavailable.

    Returns:
        An ``ExtractionResult`` containing the extracted claims and the
        normalized answer text. On failure, ``parse_success`` is ``False``
        and ``error`` describes the problem; no exception is raised.
    """
    if vlm_output is None:
        return ExtractionResult(
            answer_text="",
            claims=[],
            parse_success=False,
            error=(
                "missing_structured_output"
                if raw_response is None or raw_response.strip() == ""
                else "parse_failure"
            ),
        )

    claims: list[ExtractedClaim] = []
    for finding in vlm_output.findings:
        claims.append(
            ExtractedClaim(
                claim=finding.claim,
                region_id=finding.region_id,
                status=finding.status,  # type: ignore[arg-type]
                confidence=finding.confidence,  # type: ignore[arg-type]
                raw_claim_text=finding.claim,
            )
        )

    return ExtractionResult(
        answer_text=_normalize_answer_text(vlm_output.answer),
        claims=claims,
        parse_success=True,
        error=None,
    )
