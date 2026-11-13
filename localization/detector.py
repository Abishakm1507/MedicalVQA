"""Base and Mock Region Detector interfaces.

Traceability:
- Research Paper: Section III-B (Region Proposal)
- Feature Specification: FS-002
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from core.schemas import CandidateBox
from localization.config import DetectorConfig
from localization.postprocessing import apply_nms, build_candidate_boxes

logger = logging.getLogger(__name__)


class BaseRegionDetector(ABC):
    """Abstract base class for all candidate region localization detectors."""

    @abstractmethod
    def detect(self, image: Image.Image) -> list[CandidateBox]:
        """Proposes candidate bounding boxes for an input radiograph.

        Args:
            image: Input PIL Image.

        Returns:
            List of CandidateBox objects sorted by confidence score descending,
            up to max_k. Returns [] if no candidate exceeds score_threshold.
        """
        pass


class MockRegionDetector(BaseRegionDetector):
    """Deterministic mock detector for unit testing, CI/CD, and offline development.

    CRITICAL NOTICE:
    This mock detector is intended strictly for development, testing, and TDD pipelines.
    It MUST NEVER be confused with or used in place of an empirically validated
    medical detection model.
    """

    def __init__(
        self,
        mock_candidates: list[dict[str, Any]] | None = None,
        config: DetectorConfig | None = None,
        simulate_empty: bool = False,
    ) -> None:
        """Initializes the mock detector.

        Args:
            mock_candidates: Optional list of dictionaries with keys:
                - 'box_2d': (ymin, xmin, ymax, xmax) absolute pixel coordinates
                - 'score': float confidence rho in [0, 1]
                - 'label': str class label
            config: Optional DetectorConfig instance.
            simulate_empty: If True, always returns R = \u2205 regardless of inputs.
        """
        self.config = config or DetectorConfig()
        self.mock_candidates = mock_candidates or []
        self.simulate_empty = simulate_empty

    def detect(self, image: Image.Image) -> list[CandidateBox]:
        """Returns deterministic candidate boxes adhering to CandidateBox schema."""
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        # Strict empty detection simulation
        if self.simulate_empty or not self.mock_candidates:
            return []

        # Filter by score_threshold
        filtered = [
            c
            for c in self.mock_candidates
            if float(c.get("score", 0.0)) >= self.config.score_threshold
        ]
        if not filtered:
            return []

        # Convert to tensors to apply NMS if overlapping boxes exist
        boxes_list = []
        scores_list = []
        labels_list = []

        for cand in filtered:
            # mock box_2d is [ymin, xmin, ymax, xmax]
            # Convert to [x1, y1, x2, y2] for NMS
            ymin, xmin, ymax, xmax = cand.get("box_2d", (50, 50, 150, 150))
            boxes_list.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            scores_list.append(float(cand.get("score", 0.5)))
            lbl = cand.get("label", "lesion")
            lbl_idx = (
                self.config.class_names.index(lbl)
                if lbl in self.config.class_names
                else (len(self.config.class_names) - 1)
            )
            labels_list.append(lbl_idx)

        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

        # Apply NMS
        keep = apply_nms(boxes_tensor, scores_tensor, self.config.nms_iou_threshold)
        nms_boxes = boxes_tensor[keep]
        nms_scores = scores_tensor[keep]
        nms_labels = labels_tensor[keep]

        return build_candidate_boxes(
            boxes=nms_boxes,
            scores=nms_scores,
            labels=nms_labels,
            image_width=width,
            image_height=height,
            class_names=self.config.class_names,
            max_k=self.config.max_k,
        )
