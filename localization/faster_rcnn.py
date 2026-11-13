"""Torchvision Faster R-CNN adapter for chest radiograph candidate region detection.

Traceability:
- Research Paper: Section III-B (Region Proposal)
- Feature Specification: FS-002
"""

import logging
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from core.schemas import CandidateBox
from localization.config import DetectorConfig
from localization.detector import BaseRegionDetector
from localization.postprocessing import apply_nms, build_candidate_boxes, filter_by_score
from localization.preprocessing import prepare_image_for_detection

logger = logging.getLogger(__name__)


class FasterRCNNRegionDetector(BaseRegionDetector):
    """Torchvision Faster R-CNN adapter for chest radiograph lesion detection.

    Generates candidate bounding boxes b_i with confidence scores rho_i in [0, 1].
    Candidate boxes are indexed as R1, R2, ... Rk (k <= 7).
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        checkpoint_path: str | Path | None = None,
        weights: FasterRCNN_ResNet50_FPN_Weights | None = None,
    ) -> None:
        """Initializes the Faster R-CNN region detector.

        Args:
            config: Optional DetectorConfig. If omitted, uses defaults.
            checkpoint_path: Optional path to a medical fine-tuned checkpoint.
                Overrides config.checkpoint_path if specified.
            weights: Optional Torchvision pretrained weights enum.
                Must not be assumed to detect medical abnormalities.
        """
        self.config = config or DetectorConfig()
        if checkpoint_path is not None:
            self.config.checkpoint_path = checkpoint_path

        # Verify device availability
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{self.config.device}' but CUDA is not available on this system. "
                "Use device='cpu' for local execution or run on a GPU-enabled remote instance."
            )

        self.device = torch.device(self.config.device)
        self.model = self._build_model(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(
        self,
        weights: FasterRCNN_ResNet50_FPN_Weights | None,
    ) -> torch.nn.Module:
        """Constructs Faster R-CNN architecture and loads weights if specified."""
        num_classes = self.config.num_classes or len(self.config.class_names)

        if self.config.checkpoint_path is not None:
            ckpt_path = Path(self.config.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned medical detector checkpoint not found at: {ckpt_path}"
                )

            logger.info("Initializing Faster R-CNN architecture for medical checkpoint...")
            model = fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes
            )

            logger.info("Loading medical weights from checkpoint: %s", ckpt_path)
            state_dict: dict[str, Any] = torch.load(ckpt_path, map_location=self.device)  # nosec B614
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
            return cast(torch.nn.Module, model)

        if weights is not None:
            logger.warning(
                "CRITICAL RESEARCH NOTICE: Loading default COCO pretrained weights for Faster R-CNN. "
                "COCO weights detect everyday objects (cars, persons, bicycles) and CANNOT detect medical "
                "pathologies. Fine-tuning on medical datasets (RSNA/VinDr-CXR) is required for clinical localization."
            )
            model = fasterrcnn_resnet50_fpn(weights=weights)
            return cast(torch.nn.Module, model)

        logger.warning(
            "Initializing Faster R-CNN with uninitialized random weights (no checkpoint specified). "
            "Suitable only for offline test suites or training initialization. "
            "Medical fine-tuning is required before reporting localization accuracy."
        )
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes
        )
        return cast(torch.nn.Module, model)

    def detect(self, image: Image.Image) -> list[CandidateBox]:
        """Proposes candidate bounding boxes for an input radiograph.

        Args:
            image: PIL Image to analyze.

        Returns:
            List of CandidateBox objects sorted by confidence score descending,
            up to max_k. Returns [] if no candidate exceeds score_threshold.
        """
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        # Preprocess PIL image to tensor
        tensor_img = prepare_image_for_detection(image, device=self.device)

        # Deterministic inference mode
        with torch.no_grad():
            predictions = self.model([tensor_img])

        if not predictions or len(predictions) == 0:
            return []

        pred = predictions[0]
        raw_boxes = pred.get("boxes", torch.empty((0, 4), device=self.device))
        raw_scores = pred.get("scores", torch.empty((0,), device=self.device))
        raw_labels = pred.get("labels", torch.empty((0,), dtype=torch.int64, device=self.device))

        if len(raw_boxes) == 0:
            return []

        # 1. Filter candidates by score threshold: rho_i >= score_threshold
        filtered_boxes, filtered_scores, filtered_labels = filter_by_score(
            boxes=raw_boxes,
            scores=raw_scores,
            labels=raw_labels,
            score_threshold=self.config.score_threshold,
        )

        # STRICT SCIENTIFIC RULE: Zero artificial box generation
        # If no candidates exceed score_threshold, return R = \u2205
        if len(filtered_boxes) == 0:
            return []

        # 2. Non-Maximum Suppression (NMS)
        keep_indices = apply_nms(
            boxes=filtered_boxes,
            scores=filtered_scores,
            iou_threshold=self.config.nms_iou_threshold,
        )

        selected_boxes = filtered_boxes[keep_indices]
        selected_scores = filtered_scores[keep_indices]
        selected_labels = filtered_labels[keep_indices]

        # 3. Coordinate conversion, clamping, and CandidateBox construction
        return build_candidate_boxes(
            boxes=selected_boxes,
            scores=selected_scores,
            labels=selected_labels,
            image_width=width,
            image_height=height,
            class_names=self.config.class_names,
            max_k=self.config.max_k,
        )
