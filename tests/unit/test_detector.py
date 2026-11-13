"""Comprehensive unit tests for candidate region localization and detectors.

Tests:
1. Detector interface compliance (BaseRegionDetector).
2. Mock detector deterministic behavior.
3. Image input validation (RGB, RGBA, Grayscale, invalid dimensions).
4. Bounding-box normalization and round-trip consistency.
5. Confidence threshold filtering (score >= score_threshold).
6. Non-Maximum Suppression (NMS) behavior.
7. Empty detection output (R = \u2205 strict anti-fabrication enforcement).
8. Multiple detections and max_k truncation (capped at 7).
9. Invalid checkpoint handling (FileNotFoundError).
10. Uninitialized weights warning.
11. CPU device execution.
12. CUDA availability handling.
13. Deterministic evaluation reproducibility.
14. Coordinate boundary clamping and degenerate box rejection.
15. Modality-aware preprocessing (CXR MONOCHROME1 vs CT HU windowing).
16. DetectorConfig validation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from core.schemas import CandidateBox
from localization.config import MAX_ALLOWED_REGIONS, DetectorConfig
from localization.detector import BaseRegionDetector, MockRegionDetector
from localization.faster_rcnn import FasterRCNNRegionDetector
from localization.postprocessing import apply_nms, build_candidate_boxes, filter_by_score
from localization.preprocessing import (
    normalize_ct_pixels,
    normalize_cxr_pixels,
    prepare_image_for_detection,
)

# ============================================================================
# 1. Interface Compliance Tests
# ============================================================================


def test_base_region_detector_is_abstract() -> None:
    """Verifies that BaseRegionDetector cannot be directly instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseRegionDetector()  # type: ignore[abstract]


# ============================================================================
# 2. Config Validation Tests
# ============================================================================


def test_detector_config_valid() -> None:
    """Tests valid DetectorConfig instantiation."""
    cfg = DetectorConfig(score_threshold=0.3, nms_iou_threshold=0.4, max_k=4, device="cpu")
    assert cfg.score_threshold == 0.3
    assert cfg.nms_iou_threshold == 0.4
    assert cfg.max_k == 4
    assert cfg.device == "cpu"


def test_detector_config_clamps_max_k_to_seven() -> None:
    """Tests that max_k > 7 is clamped to MAX_ALLOWED_REGIONS = 7."""
    cfg = DetectorConfig(max_k=15)
    assert cfg.max_k == MAX_ALLOWED_REGIONS == 7


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        ({"score_threshold": -0.1}, "score_threshold must be in range"),
        ({"score_threshold": 1.5}, "score_threshold must be in range"),
        ({"nms_iou_threshold": -0.5}, "nms_iou_threshold must be in range"),
        ({"nms_iou_threshold": 1.1}, "nms_iou_threshold must be in range"),
        ({"max_k": 0}, "max_k must be a positive integer"),
        ({"device": "tpu"}, "Unsupported device"),
        ({"num_classes": 0}, "num_classes must be a positive integer"),
    ],
)
def test_detector_config_invalid_raises(kwargs: dict[str, object], err_msg: str) -> None:
    """Tests that invalid config values raise ValueError."""
    with pytest.raises(ValueError, match=err_msg):
        DetectorConfig(**kwargs)  # type: ignore[arg-type]


# ============================================================================
# 3. Image Input & Preprocessing Tests
# ============================================================================


def test_prepare_image_for_detection_rgb() -> None:
    """Tests converting standard RGB PIL image to tensor."""
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    tensor = prepare_image_for_detection(img, device="cpu")
    assert tensor.shape == (3, 100, 200)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_prepare_image_for_detection_grayscale_and_rgba() -> None:
    """Tests that Grayscale and RGBA images are converted to 3-channel RGB tensors."""
    img_gray = Image.new("L", (150, 120), color=100)
    tensor_gray = prepare_image_for_detection(img_gray, device="cpu")
    assert tensor_gray.shape == (3, 120, 150)

    img_rgba = Image.new("RGBA", (150, 120), color=(100, 100, 100, 255))
    tensor_rgba = prepare_image_for_detection(img_rgba, device="cpu")
    assert tensor_rgba.shape == (3, 120, 150)


def test_prepare_image_invalid_dimensions() -> None:
    """Tests that non-positive image dimensions raise ValueError."""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.size = (0, 100)
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        prepare_image_for_detection(mock_img)


# ============================================================================
# 4. Modality-Aware Preprocessing Tests (CXR vs CT Decoupling)
# ============================================================================


def test_normalize_cxr_pixels_monochrome2() -> None:
    """Tests standard CXR MONOCHROME2 normalization."""
    raw = np.array([[100, 200], [300, 400]], dtype=np.int32)
    norm = normalize_cxr_pixels(raw, photometric_interpretation="MONOCHROME2")
    assert norm.dtype == np.uint8
    assert norm.shape == (2, 2)
    assert norm[0, 0] <= norm[1, 1]


def test_normalize_cxr_pixels_monochrome1_inversion() -> None:
    """Tests CXR MONOCHROME1 inversion (air becomes dark, bone becomes bright)."""
    raw = np.array([[100, 200], [300, 400]], dtype=np.int32)
    norm_m1 = normalize_cxr_pixels(raw, photometric_interpretation="MONOCHROME1")
    norm_m2 = normalize_cxr_pixels(raw, photometric_interpretation="MONOCHROME2")
    assert norm_m1.dtype == np.uint8
    # Under MONOCHROME1, the previously highest pixel (400) should now be the lowest value (0)
    assert norm_m1[1, 1] == 0
    assert norm_m2[1, 1] == 255


def test_normalize_cxr_pixels_with_windowing() -> None:
    """Tests CXR normalization with explicit VOI LUT window center and width."""
    raw = np.linspace(0, 1000, 100).reshape(10, 10).astype(np.float32)
    norm = normalize_cxr_pixels(raw, window_center=500.0, window_width=400.0)
    assert norm.dtype == np.uint8
    # Values <= 300 should be clipped to 0; values >= 700 clipped to 255
    assert norm[0, 0] == 0
    assert norm[9, 9] == 255


def test_normalize_ct_pixels_hu_and_windowing() -> None:
    """Tests CT-specific Hounsfield Unit scaling and windowing (strictly decoupled from CXR)."""
    raw_ct = np.array([[0, 1000], [2000, 3000]], dtype=np.int16)
    # RescaleSlope = 1.0, RescaleIntercept = -1024 (standard CT conversion: HU = raw - 1024)
    # Lung window: WC = -600, WW = 1500 -> Range [-1350, 150]
    norm_ct = normalize_ct_pixels(
        raw_ct,
        rescale_slope=1.0,
        rescale_intercept=-1024.0,
        window_center=-600.0,
        window_width=1500.0,
    )
    assert norm_ct.dtype == np.uint8
    # raw 0 -> -1024 HU (within [-1350, 150])
    # raw 2000 -> 976 HU (above 150, clipped to 255)
    assert norm_ct[0, 0] > 0
    assert norm_ct[1, 0] == 255


# ============================================================================
# 5. Postprocessing Tests (Score Filtering, NMS, Bounding Box Construction)
# ============================================================================


def test_filter_by_score() -> None:
    """Tests filtering candidate boxes by score threshold."""
    boxes = torch.tensor(
        [[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]], dtype=torch.float32
    )
    scores = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    labels = torch.tensor([1, 1, 1], dtype=torch.int64)

    f_boxes, f_scores, f_labels = filter_by_score(boxes, scores, labels, score_threshold=0.5)
    assert len(f_boxes) == 2
    assert torch.equal(f_scores, torch.tensor([0.8, 0.6]))


def test_apply_nms() -> None:
    """Tests that overlapping boxes above IoU threshold are suppressed."""
    # Box 1 and Box 2 have ~0.8 IoU
    boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.7], dtype=torch.float32)

    keep = apply_nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 1
    assert keep[0].item() == 0  # Higher score box retained


def test_build_candidate_boxes_strict_anti_fabrication() -> None:
    """STRICT SCIENTIFIC RULE: Zero artificial box generation. Empty inputs return []."""
    res = build_candidate_boxes(
        boxes=torch.empty((0, 4)),
        scores=torch.empty((0,)),
        labels=torch.empty((0,), dtype=torch.int64),
        image_width=500,
        image_height=500,
        class_names=["background", "lesion"],
    )
    assert res == []


def test_build_candidate_boxes_clamping_and_normalization() -> None:
    """Tests coordinate normalization and out-of-bounds boundary clamping."""
    # Box extends outside 400x200 image: x2=450 (should clamp to 400), y2=250 (clamp to 200)
    boxes = torch.tensor([[-20.0, -10.0, 450.0, 250.0]], dtype=torch.float32)
    scores = torch.tensor([0.85], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)

    candidates = build_candidate_boxes(
        boxes=boxes,
        scores=scores,
        labels=labels,
        image_width=400,
        image_height=200,
        class_names=["background", "opacity"],
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert c.region_id == "R1"
    # Clamped pixel box: [ymin, xmin, ymax, xmax] -> [0, 0, 200, 400]
    assert c.box_2d == (0, 0, 200, 400)
    # Normalized box: [0.0, 0.0, 1.0, 1.0]
    assert c.normalized_box == (0.0, 0.0, 1.0, 1.0)
    assert c.score == 0.85
    assert c.label == "opacity"


def test_build_candidate_boxes_skips_degenerate() -> None:
    """Tests that degenerate zero-area boxes are skipped."""
    boxes = torch.tensor([[10.0, 10.0, 10.0, 50.0]], dtype=torch.float32)  # zero height
    scores = torch.tensor([0.9], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)

    candidates = build_candidate_boxes(
        boxes=boxes,
        scores=scores,
        labels=labels,
        image_width=200,
        image_height=200,
        class_names=["background", "lesion"],
    )
    assert candidates == []


# ============================================================================
# 6. MockRegionDetector Tests
# ============================================================================


def test_mock_region_detector_empty() -> None:
    """Tests mock detector with simulate_empty=True returning R = \u2205."""
    detector = MockRegionDetector(simulate_empty=True)
    img = Image.new("RGB", (300, 300))
    res = detector.detect(img)
    assert res == []


def test_mock_region_detector_deterministic_candidates() -> None:
    """Tests mock detector returning deterministic CandidateBox objects."""
    mock_boxes = [
        {"box_2d": (20, 30, 80, 90), "score": 0.88, "label": "pneumonia"},
        {"box_2d": (100, 120, 200, 220), "score": 0.72, "label": "cardiomegaly"},
    ]
    cfg = DetectorConfig(
        score_threshold=0.5, max_k=5, class_names=["background", "pneumonia", "cardiomegaly"]
    )
    detector = MockRegionDetector(mock_candidates=mock_boxes, config=cfg)

    img = Image.new("RGB", (400, 400))
    res = detector.detect(img)

    assert len(res) == 2
    assert res[0].region_id == "R1"
    assert res[0].score == 0.88
    assert res[0].label == "pneumonia"
    assert res[1].region_id == "R2"
    assert res[1].score == 0.72
    assert res[1].label == "cardiomegaly"


def test_mock_region_detector_score_filtering() -> None:
    """Tests mock detector filtering candidates below score_threshold."""
    mock_boxes = [
        {"box_2d": (10, 10, 50, 50), "score": 0.20, "label": "lesion"},
    ]
    cfg = DetectorConfig(score_threshold=0.5)
    detector = MockRegionDetector(mock_candidates=mock_boxes, config=cfg)

    img = Image.new("RGB", (100, 100))
    res = detector.detect(img)
    assert res == []


def test_mock_region_detector_max_k_truncation() -> None:
    """Tests mock detector truncating to max_k candidates."""
    mock_boxes = [
        {"box_2d": (i * 20, 10, i * 20 + 15, 50), "score": 0.90 - i * 0.05, "label": "lesion"}
        for i in range(10)
    ]
    cfg = DetectorConfig(score_threshold=0.1, max_k=3)
    detector = MockRegionDetector(mock_candidates=mock_boxes, config=cfg)

    img = Image.new("RGB", (300, 300))
    res = detector.detect(img)
    assert len(res) == 3
    assert [r.region_id for r in res] == ["R1", "R2", "R3"]


# ============================================================================
# 7. FasterRCNNRegionDetector Tests
# ============================================================================


def test_faster_rcnn_missing_checkpoint_raises() -> None:
    """Tests that specifying a non-existent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        FasterRCNNRegionDetector(checkpoint_path="non_existent_weights_path.pth")


def test_faster_rcnn_cuda_unavailable_handling() -> None:
    """Tests that requesting CUDA on a non-CUDA system raises RuntimeError."""
    cfg = DetectorConfig(device="cuda")
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            FasterRCNNRegionDetector(config=cfg)


def test_faster_rcnn_cpu_inference_with_uninitialized_weights() -> None:
    """Tests initializing Faster R-CNN on CPU and running forward pass without downloading weights."""
    cfg = DetectorConfig(score_threshold=0.5, max_k=5, device="cpu")
    detector = FasterRCNNRegionDetector(config=cfg)

    assert detector.device.type == "cpu"
    assert not detector.model.training  # In eval mode

    img = Image.new("RGB", (128, 128), color=(100, 100, 100))
    res = detector.detect(img)
    # Output should be a list of CandidateBox schemas (or empty if uninitialized scores < 0.5)
    assert isinstance(res, list)
    for c in res:
        assert isinstance(c, CandidateBox)
        assert c.score >= 0.5
        assert 0.0 <= c.normalized_box[0] <= 1.0


def test_faster_rcnn_mocked_forward_pass_reproducibility() -> None:
    """Tests deterministic postprocessing and empty detection output on mocked model forward pass."""
    cfg = DetectorConfig(score_threshold=0.4, nms_iou_threshold=0.5, max_k=3, device="cpu")
    detector = FasterRCNNRegionDetector(config=cfg)

    mock_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]])
    mock_scores = torch.tensor([0.85, 0.65])
    mock_labels = torch.tensor([1, 1], dtype=torch.int64)

    mock_predictions = [{"boxes": mock_boxes, "scores": mock_scores, "labels": mock_labels}]

    # Patch model forward pass
    with patch.object(detector.model, "forward", return_value=mock_predictions):
        img = Image.new("RGB", (200, 200))
        res1 = detector.detect(img)
        res2 = detector.detect(img)

        # Reproducibility check
        assert len(res1) == 2
        assert len(res2) == 2
        assert res1[0].region_id == "R1"
        assert res1[0].score == 0.85
        assert res1[0].box_2d == res2[0].box_2d
        assert res1[0].normalized_box == res2[0].normalized_box


def test_faster_rcnn_mocked_forward_pass_empty_result() -> None:
    """STRICT ANTI-FABRICATION RULE: When no candidate passes threshold, returns []."""
    cfg = DetectorConfig(score_threshold=0.9, device="cpu")
    detector = FasterRCNNRegionDetector(config=cfg)

    mock_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    mock_scores = torch.tensor([0.30])  # Below 0.9
    mock_labels = torch.tensor([1], dtype=torch.int64)

    mock_predictions = [{"boxes": mock_boxes, "scores": mock_scores, "labels": mock_labels}]

    with patch.object(detector.model, "forward", return_value=mock_predictions):
        img = Image.new("RGB", (200, 200))
        res = detector.detect(img)
        assert res == []


def test_faster_rcnn_load_checkpoint_file(tmp_path: Path) -> None:
    """Tests loading a model state_dict from a local checkpoint file."""
    # Create a dummy model checkpoint
    cfg = DetectorConfig(num_classes=2, device="cpu")
    temp_detector = FasterRCNNRegionDetector(config=cfg)
    ckpt_file = tmp_path / "dummy_medical_detector.pth"
    torch.save({"model": temp_detector.model.state_dict()}, ckpt_file)

    # Load from the created checkpoint
    loaded_detector = FasterRCNNRegionDetector(checkpoint_path=ckpt_file)
    assert loaded_detector.model is not None
    assert not loaded_detector.model.training
