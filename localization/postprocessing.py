"""Postprocessing utilities for candidate region localization.

Performs confidence score threshold filtering, Non-Maximum Suppression (NMS),
coordinate normalization, boundary clamping, and CandidateBox schema construction.

Traceability:
- Research Paper: Section III-B (Region Proposal)
- Feature Specification: FS-002
"""

from collections.abc import Sequence
from typing import cast

import torch
import torchvision.ops

from core.schemas import CandidateBox
from localization.config import MAX_ALLOWED_REGIONS


def filter_by_score(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filters candidate detections by minimum confidence score threshold.

    Args:
        boxes: Tensor of shape [N, 4] in pixel coordinates [x1, y1, x2, y2].
        scores: Tensor of shape [N] with confidence scores in [0, 1].
        labels: Tensor of shape [N] with integer class labels.
        score_threshold: Minimum confidence score rho required.

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels).
        If no candidate satisfies the threshold, returns empty tensors.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    mask = scores >= float(score_threshold)
    return boxes[mask], scores[mask], labels[mask]


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Applies Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes.

    Args:
        boxes: Tensor of shape [N, 4] in pixel coordinates [x1, y1, x2, y2].
        scores: Tensor of shape [N] with confidence scores.
        iou_threshold: Intersection-over-Union threshold above which overlapping
            boxes are suppressed.

    Returns:
        Tensor of 1D indices of retained bounding boxes, sorted descending by score.
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    nms_result = torchvision.ops.nms(
        boxes=boxes,
        scores=scores,
        iou_threshold=float(iou_threshold),
    )
    return cast(torch.Tensor, nms_result)


def build_candidate_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_width: int,
    image_height: int,
    class_names: Sequence[str],
    max_k: int = 5,
) -> list[CandidateBox]:
    """Constructs standardized CandidateBox objects from post-NMS detections.

    STRICT ANTI-FABRICATION RULE:
    If boxes is empty, returns R = \u2205 (empty list). Never generates synthetic
    or artificial quadrant boxes.

    Args:
        boxes: Tensor of shape [K, 4] in pixel coordinates [x1, y1, x2, y2].
        scores: Tensor of shape [K] with confidence scores.
        labels: Tensor of shape [K] with class indices.
        image_width: Native image width in pixels.
        image_height: Native image height in pixels.
        class_names: List of class label strings.
        max_k: Maximum candidate regions to retain (capped at MAX_ALLOWED_REGIONS = 7).

    Returns:
        List of CandidateBox schemas labeled R1, R2, ... Rk (k <= max_k).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")

    # Enforce upper bound of 7 regions per paper specification
    effective_k = min(len(boxes), int(max_k), MAX_ALLOWED_REGIONS)
    if effective_k == 0:
        return []

    boxes_cpu = boxes[:effective_k].detach().cpu()
    scores_cpu = scores[:effective_k].detach().cpu()
    labels_cpu = labels[:effective_k].detach().cpu()

    candidates: list[CandidateBox] = []
    region_idx = 1

    for box, score, label_idx in zip(boxes_cpu, scores_cpu, labels_cpu, strict=True):
        x1, y1, x2, y2 = box.tolist()

        # Clamp coordinates to image boundaries
        x1 = max(0.0, min(float(x1), float(image_width)))
        x2 = max(0.0, min(float(x2), float(image_width)))
        y1 = max(0.0, min(float(y1), float(image_height)))
        y2 = max(0.0, min(float(y2), float(image_height)))

        # Skip degenerate boxes (zero area or negative coordinates)
        if x2 <= x1 or y2 <= y1:
            continue

        # Standardized coordinate ordering: [ymin, xmin, ymax, xmax]
        box_2d = (int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
        normalized_box = (
            round(y1 / image_height, 4),
            round(x1 / image_width, 4),
            round(y2 / image_height, 4),
            round(x2 / image_width, 4),
        )

        idx = int(label_idx)
        label_str = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"

        candidates.append(
            CandidateBox(
                region_id=f"R{region_idx}",
                box_2d=box_2d,
                normalized_box=normalized_box,
                score=round(float(score), 4),
                label=label_str,
            )
        )
        region_idx += 1

    return candidates
