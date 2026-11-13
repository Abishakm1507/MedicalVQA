"""Localization package for candidate region detection, MedSAM segmentation, and visual marking."""

from localization.config import MAX_ALLOWED_REGIONS, DetectorConfig
from localization.detector import BaseRegionDetector, MockRegionDetector
from localization.faster_rcnn import FasterRCNNRegionDetector
from localization.postprocessing import apply_nms, build_candidate_boxes, filter_by_score
from localization.preprocessing import (
    normalize_ct_pixels,
    normalize_cxr_pixels,
    prepare_image_for_detection,
)

__all__ = [
    "MAX_ALLOWED_REGIONS",
    "DetectorConfig",
    "BaseRegionDetector",
    "MockRegionDetector",
    "FasterRCNNRegionDetector",
    "prepare_image_for_detection",
    "normalize_cxr_pixels",
    "normalize_ct_pixels",
    "filter_by_score",
    "apply_nms",
    "build_candidate_boxes",
]
