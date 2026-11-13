"""Unit tests for coordinate normalization, round-trip transformations, and patient isolation."""

import pytest

from core.schemas import EvaluationTrack, MedicalVQASample
from datasets.base import (
    check_patient_leakage,
    denormalize_bounding_box,
    normalize_bounding_box,
)

# ============================================================================
# Coordinate Normalization & Round-Trip Tests (Section 2)
# ============================================================================


def test_full_image_bounding_box() -> None:
    # Full image [0, 0, 800, 600] on non-square image 800x600
    norm_box = normalize_bounding_box([0, 0, 800, 600], 800, 600, input_format="xywh")
    assert norm_box is not None
    ymin, xmin, ymax, xmax = norm_box
    assert pytest.approx(ymin, 1e-6) == 0.0
    assert pytest.approx(xmin, 1e-6) == 0.0
    assert pytest.approx(ymax, 1e-6) == 1.0
    assert pytest.approx(xmax, 1e-6) == 1.0

    # Round-trip back to pixels
    denorm = denormalize_bounding_box(norm_box, 800, 600, output_format="xywh")
    assert pytest.approx(denorm[0], 1e-4) == 0.0
    assert pytest.approx(denorm[1], 1e-4) == 0.0
    assert pytest.approx(denorm[2], 1e-4) == 800.0
    assert pytest.approx(denorm[3], 1e-4) == 600.0


def test_top_left_and_bottom_right_boundaries() -> None:
    w, h = 1000, 500  # Non-square
    # Top-left box: 50x50 at [0, 0]
    tl_box = normalize_bounding_box([0, 0, 50, 50], w, h, input_format="xywh")
    assert tl_box is not None
    assert pytest.approx(tl_box[0], 1e-6) == 0.0
    assert pytest.approx(tl_box[1], 1e-6) == 0.0
    assert pytest.approx(tl_box[2], 1e-6) == 50 / 500
    assert pytest.approx(tl_box[3], 1e-6) == 50 / 1000

    # Bottom-right box: 50x50 at [950, 450]
    br_box = normalize_bounding_box([950, 450, 50, 50], w, h, input_format="xywh")
    assert br_box is not None
    assert pytest.approx(br_box[0], 1e-6) == 450 / 500
    assert pytest.approx(br_box[1], 1e-6) == 950 / 1000
    assert pytest.approx(br_box[2], 1e-6) == 1.0
    assert pytest.approx(br_box[3], 1e-6) == 1.0


def test_non_square_aspect_ratios() -> None:
    # 1920 x 1080 High-def radiograph
    box_xyxy = [100.0, 200.0, 500.0, 800.0]
    norm_box = normalize_bounding_box(box_xyxy, 1920, 1080, input_format="xyxy")
    assert norm_box is not None
    ymin, xmin, ymax, xmax = norm_box
    assert pytest.approx(ymin, 1e-6) == 200.0 / 1080
    assert pytest.approx(xmin, 1e-6) == 100.0 / 1920
    assert pytest.approx(ymax, 1e-6) == 800.0 / 1080
    assert pytest.approx(xmax, 1e-6) == 500.0 / 1920

    # Round-trip back to xyxy
    denorm = denormalize_bounding_box(norm_box, 1920, 1080, output_format="xyxy")
    assert pytest.approx(denorm[0], 1e-4) == 100.0
    assert pytest.approx(denorm[1], 1e-4) == 200.0
    assert pytest.approx(denorm[2], 1e-4) == 500.0
    assert pytest.approx(denorm[3], 1e-4) == 800.0


def test_invalid_image_dimensions() -> None:
    assert normalize_bounding_box([10, 10, 50, 50], 0, 500) is None
    assert normalize_bounding_box([10, 10, 50, 50], 500, -10) is None
    assert normalize_bounding_box([10, 10, 50, 50], -500, -500) is None


def test_reversed_and_degenerate_coordinates() -> None:
    # xyxy format where xmax <= xmin
    assert normalize_bounding_box([200, 100, 150, 300], 500, 500, input_format="xyxy") is None
    # xyxy format where ymax <= ymin
    assert normalize_bounding_box([100, 300, 250, 200], 500, 500, input_format="xyxy") is None
    # Zero width or zero height
    assert normalize_bounding_box([100, 100, 0, 50], 500, 500, input_format="xywh") is None
    assert normalize_bounding_box([100, 100, 50, 0], 500, 500, input_format="xywh") is None


def test_out_of_bounds_rejection_vs_clipping() -> None:
    w, h = 500, 500
    # Coordinates significantly outside boundary (e.g. xmin = -50)
    out_box = [-50.0, 20.0, 100.0, 100.0]

    # Strict mode (default clip_out_of_bounds=False) MUST reject
    assert (
        normalize_bounding_box(out_box, w, h, input_format="xywh", clip_out_of_bounds=False) is None
    )

    # Explicit clip mode allows controlled clamping
    clipped = normalize_bounding_box(out_box, w, h, input_format="xywh", clip_out_of_bounds=True)
    assert clipped is not None
    # xmin should clamp to 0.0
    assert clipped[1] == 0.0


# ============================================================================
# Patient-Level Split Isolation Tests (Section 1 & Section 5)
# ============================================================================


def make_sample(
    sample_id: str, image_id: str, patient_id: str | None, split: str
) -> MedicalVQASample:
    return MedicalVQASample(
        sample_id=sample_id,
        dataset_name="test_data",
        split=split,
        image_id=image_id,
        patient_id=patient_id,
        image_path="test.png",
        modality="CXR",
        organ="chest",
        question="Is there lesion?",
        answer="yes",
        question_type="closed",
        ground_truth_regions=[],
        compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
    )


def test_patient_leakage_when_patient_id_is_none() -> None:
    # Datasets like SLAKE and VQA-RAD do not provide patient IDs (patient_id=None)
    train = [make_sample("s1", "img1", None, "train"), make_sample("s2", "img2", None, "train")]
    test = [make_sample("s3", "img3", None, "test")]

    report = check_patient_leakage(train, test)
    # Must explicitly flag that patient IDs are unavailable, NOT claim leakage prevention!
    assert report.patient_ids_available is False
    assert report.is_leakage_detected is False
    assert len(report.leaked_patient_ids) == 0


def test_patient_leakage_detected_with_multiple_views() -> None:
    # A patient has multiple radiographs (frontal and lateral)
    train = [
        make_sample("s1", "view_frontal_001", "patient_A", "train"),
        make_sample("s2", "view_frontal_002", "patient_B", "train"),
    ]
    # Inadvertent leakage: patient_A lateral view placed in test split!
    test_leaked = [
        make_sample("s3", "view_lateral_001", "patient_A", "test"),
        make_sample("s4", "view_frontal_003", "patient_C", "test"),
    ]

    report = check_patient_leakage(train, test_leaked)
    assert report.patient_ids_available is True
    assert report.is_leakage_detected is True
    assert "patient_A" in report.leaked_patient_ids
    assert report.total_patients_train == 2
    assert report.total_patients_test == 2


def test_patient_isolation_clean_split() -> None:
    train = [make_sample("s1", "img1", "patient_A", "train")]
    val = [make_sample("s2", "img2", "patient_B", "validate")]
    test = [make_sample("s3", "img3", "patient_C", "test")]

    report = check_patient_leakage(train, test, val)
    assert report.patient_ids_available is True
    assert report.is_leakage_detected is False
    assert len(report.leaked_patient_ids) == 0
