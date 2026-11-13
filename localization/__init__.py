"""Localization package for candidate region detection, MedSAM segmentation, and visual marking.

The heavy detector modules depend on optional ML libraries (torch, torchvision). To keep the
package importable without those optional dependencies – which is required for the RLE utilities
and their unit tests – we expose a lazy import mechanism. Public symbols that do not require the
optional dependencies are imported eagerly; symbols that need the heavy stack are loaded only when
accessed via attribute access.
"""

from typing import Any

# Light‑weight imports – always safe
from localization.config import MAX_ALLOWED_REGIONS, DetectorConfig

# Define the public API. Heavy symbols are listed but will be resolved lazily.
__all__ = [
    # FS-002 — region detection
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
    # FS-003 — region segmentation
    "SegmenterConfig",
    "SegmentationResult",
    "BaseRegionSegmenter",
    "MockRegionSegmenter",
    "MedSAMRegionSegmenter",
]

# Lazy import helper for heavy modules
def __getattr__(name: str) -> Any:
    if name in ("BaseRegionDetector", "MockRegionDetector"):
        from .detector import BaseRegionDetector, MockRegionDetector
        globals().update({"BaseRegionDetector": BaseRegionDetector, "MockRegionDetector": MockRegionDetector})
        return globals()[name]
    if name == "FasterRCNNRegionDetector":
        from .faster_rcnn import FasterRCNNRegionDetector
        globals()["FasterRCNNRegionDetector"] = FasterRCNNRegionDetector
        return FasterRCNNRegionDetector
    if name == "prepare_image_for_detection":
        from .preprocessing import prepare_image_for_detection
        globals()["prepare_image_for_detection"] = prepare_image_for_detection
        return prepare_image_for_detection
    if name == "normalize_cxr_pixels":
        from .preprocessing import normalize_cxr_pixels
        globals()["normalize_cxr_pixels"] = normalize_cxr_pixels
        return normalize_cxr_pixels
    if name == "normalize_ct_pixels":
        from .preprocessing import normalize_ct_pixels
        globals()["normalize_ct_pixels"] = normalize_ct_pixels
        return normalize_ct_pixels
    if name == "filter_by_score":
        from .postprocessing import filter_by_score
        globals()["filter_by_score"] = filter_by_score
        return filter_by_score
    if name == "apply_nms":
        from .postprocessing import apply_nms
        globals()["apply_nms"] = apply_nms
        return apply_nms
    if name == "build_candidate_boxes":
        from .postprocessing import build_candidate_boxes
        globals()["build_candidate_boxes"] = build_candidate_boxes
        return build_candidate_boxes
    # FS-003 segmentation symbols — no heavy ML dependencies required
    if name in ("SegmenterConfig", "SegmentationResult",
                "BaseRegionSegmenter", "MockRegionSegmenter"):
        from .segmenter import (
            BaseRegionSegmenter,
            MockRegionSegmenter,
            SegmentationResult,
            SegmenterConfig,
        )
        globals().update({
            "SegmenterConfig": SegmenterConfig,
            "SegmentationResult": SegmentationResult,
            "BaseRegionSegmenter": BaseRegionSegmenter,
            "MockRegionSegmenter": MockRegionSegmenter,
        })
        return globals()[name]
    # MedSAMRegionSegmenter — requires torch + segment_anything (lazy)
    if name == "MedSAMRegionSegmenter":
        from .segmenter import MedSAMRegionSegmenter
        globals()["MedSAMRegionSegmenter"] = MedSAMRegionSegmenter
        return MedSAMRegionSegmenter
    raise AttributeError(f"module 'localization' has no attribute '{name}'")
