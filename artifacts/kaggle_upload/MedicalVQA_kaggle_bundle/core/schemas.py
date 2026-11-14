"""Core schemas and data models for MedicalVQA.

Defines standardized data representations for datasets, localization regions,
VLM structured outputs, and API request/response payloads.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Research Evaluation Tracks (Section V-A, Table 1, & Phase 0 Audit)
# ============================================================================


class EvaluationTrack(StrEnum):
    """The three orthogonal evaluation tracks defined in the research protocol."""

    TRACK_A_PREDICTED_CXR = "track_a_predicted_cxr"  # CXR with detector-proposed candidate regions
    TRACK_B_ORACLE_SLAKE = (
        "track_b_oracle_slake"  # All SLAKE modalities using ground-truth boxes as candidate regions
    )
    TRACK_C_ANSWER_ONLY = (
        "track_c_answer_only"  # Ungrounded answer accuracy & F1 baseline (VQA-RAD, SLAKE free-form)
    )


# ============================================================================
# Dataset and Grounding Models
# ============================================================================


@dataclass
class GroundTruthRegion:
    """Represents a ground-truth annotated bounding box and optional mask."""

    label: str
    box_2d: tuple[float, float, float, float]  # Normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0]
    mask_path: str | None = None
    mask_rle: str | None = None

    def __post_init__(self) -> None:
        ymin, xmin, ymax, xmax = self.box_2d
        if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
            raise ValueError(
                f"Invalid normalized box coordinates [ymin, xmin, ymax, xmax]: {self.box_2d}. "
                "Must satisfy 0.0 <= min < max <= 1.0."
            )


@dataclass
class MedicalVQASample:
    """Standardized representation of a medical VQA benchmark sample across all datasets."""

    sample_id: str
    dataset_name: str  # 'slake', 'vqarad', 'rsna', 'vindr'
    split: str  # 'train', 'validate', 'test'
    image_id: str  # Image-level identifier (e.g. 'source.jpg', 'patient_001.png')
    image_path: str
    modality: str  # e.g., 'CXR', 'CT', 'MRI'
    organ: str  # e.g., 'chest', 'abdomen', 'brain', 'head'
    question: str
    answer: str
    question_type: str  # 'closed' or 'open'
    ground_truth_regions: list[GroundTruthRegion] = field(default_factory=list)
    has_ground_truth_regions: bool = (
        True  # False for datasets without region annotations (e.g. VQA-RAD)
    )
    patient_id: str | None = None  # None if dataset does not provide clinical patient identity
    compatible_tracks: list[EvaluationTrack] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_compatible_with(self, track: EvaluationTrack) -> bool:
        """Checks if this sample is scientifically compatible with an evaluation track."""
        return track in self.compatible_tracks

    def assert_track_compatibility(self, track: EvaluationTrack) -> None:
        """Raises ValueError if sample is incompatible with the requested track."""
        if not self.is_compatible_with(track):
            raise ValueError(
                f"Sample {self.sample_id} ({self.dataset_name}, modality={self.modality}, organ={self.organ}) "
                f"is incompatible with evaluation track '{track.value}'. "
                f"Compatible tracks: {[t.value for t in self.compatible_tracks]}"
            )


# ============================================================================
# Localization Models
# ============================================================================


@dataclass
class CandidateBox:
    """Represents a candidate bounding box proposed by the detector."""

    region_id: str  # e.g., 'R1', 'R2', ...
    box_2d: tuple[int, int, int, int]  # Absolute pixel coordinates [ymin, xmin, ymax, xmax]
    normalized_box: tuple[
        float, float, float, float
    ]  # Normalized [ymin, xmin, ymax, xmax] in [0, 1]
    score: float  # Detector confidence score rho in [0, 1]
    label: str = "lesion"
    mask_rle: str | None = None


# ============================================================================
# VLM Structured JSON Contract Models (Section III-C)
# ============================================================================


class VLMClaim(BaseModel):
    """Represents an individual clinical finding emitted by a VLM."""

    claim: str = Field(..., description="The clinical finding or pathology asserted.")
    region_id: str = Field(
        ...,
        description="The candidate region ID cited (e.g. 'R1', 'R2', or 'NONE' for negative finding).",
    )
    status: Literal["present", "absent", "uncertain"] = Field(
        "present", description="Clinical status of the asserted finding."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        "high", description="VLM self-reported certainty level."
    )


class VLMStructuredOutput(BaseModel):
    """The strict JSON contract required from all VLM responses."""

    answer: str = Field(..., description="Natural language response to the clinical question.")
    findings: list[VLMClaim] = Field(
        default_factory=list, description="List of clinical findings grounded to candidate regions."
    )


# ============================================================================
# Verification & Confidence Gate Models (Section III-D, III-E)
# ============================================================================


@dataclass
class VerifiedFinding:
    """A finding that has passed runtime schema and region contract checks."""

    claim: str
    region_id: str
    status: str  # 'present', 'absent', 'uncertain'
    is_grounded: bool
    localizer_score: float | None = None
    iou_with_ground_truth: float | None = None
    is_supported: bool = True


@dataclass
class ConfidenceGateResult:
    """Detailed breakdown of the confidence gate calculation."""

    majority_answer: str
    agreement: float  # a in [0, 1]
    raw_entropy: float  # Shannon entropy H
    normalized_entropy: float  # H_n = H / ln(N) in [0, 1]
    localizer_score: float | None  # rho in [0, 1], or None for pure negative claim
    fused_confidence: float  # c = w1*a + w2*(1-Hn) + w3*rho in [0, 1]
    is_abstaining: bool  # True if c < tau
    decision_threshold: float  # tau
    num_samples: int = 5  # Standardized N=5


# ============================================================================
# API Request & Response Schemas (Section IV-A)
# ============================================================================


class AnalyzeResponseRegion(BaseModel):
    region_id: str
    box_2d: tuple[int, int, int, int]  # [ymin, xmin, ymax, xmax]
    normalized_box: tuple[float, float, float, float]
    score: float
    label: str
    mask_rle: str | None = None


class AnalyzeResponseFinding(BaseModel):
    claim: str
    region_id: str
    status: str
    is_grounded: bool
    localizer_score: float | None
    is_supported: bool


class ConfidenceBreakdownPayload(BaseModel):
    agreement: float
    normalized_entropy: float
    localizer_score: float | None
    fused_confidence: float
    threshold: float
    num_samples: int


class AnalyzeResponse(BaseModel):
    """Response payload for POST /api/v1/analyze."""

    request_id: str
    question: str
    model: str
    status: Literal["ACCEPTED", "ABSTAINED"]
    final_answer: str
    is_abstaining: bool
    confidence_score: float
    confidence_breakdown: ConfidenceBreakdownPayload
    regions: list[AnalyzeResponseRegion]
    findings: list[AnalyzeResponseFinding]
