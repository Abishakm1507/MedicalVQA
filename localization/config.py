"""Configuration parameters for candidate region localization detectors.

Traceability:
- Research Paper: Section III-B (Region Proposal)
- Feature Specification: FS-002
"""

from dataclasses import dataclass, field
from pathlib import Path

# Maximum number of candidate regions allowed per paper/spec (Section III-B: k <= 7)
MAX_ALLOWED_REGIONS: int = 7


@dataclass
class DetectorConfig:
    """Configuration settings for candidate region detectors."""

    score_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    max_k: int = 5
    device: str = "cpu"
    checkpoint_path: str | Path | None = None
    class_names: list[str] = field(default_factory=lambda: ["background", "lesion"])
    weights: str | None = None
    num_classes: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError(
                f"score_threshold must be in range [0.0, 1.0], got {self.score_threshold}"
            )
        if not (0.0 <= self.nms_iou_threshold <= 1.0):
            raise ValueError(
                f"nms_iou_threshold must be in range [0.0, 1.0], got {self.nms_iou_threshold}"
            )
        if self.max_k <= 0:
            raise ValueError(f"max_k must be a positive integer, got {self.max_k}")
        if self.max_k > MAX_ALLOWED_REGIONS:
            # Enforce upper bound of 7 regions per paper specification
            self.max_k = MAX_ALLOWED_REGIONS
        if self.device not in ("cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ValueError(
                f"Unsupported device: {self.device}. Must be 'cpu' or 'cuda' / 'cuda:X'"
            )
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {self.num_classes}")
