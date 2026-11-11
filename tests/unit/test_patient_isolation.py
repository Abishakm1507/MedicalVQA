"""Unit tests for patient-level split isolation and coordinate normalization."""

import pytest

from core.schemas import EvaluationTrack, MedicalVQASample
from datasets.base import check_patient_leakage, normalize_bounding_box


def test_normalize_bounding_box_xywh() -> None:
    # 512x512 image, box [100, 100, 200, 200]
    norm_box = normalize_bounding_box([100, 100, 200, 200], 512, 512, input_format="xywh")
    assert norm_box is not None
    ymin, xmin, ymax, xmax = norm_box
    assert pytest.approx(ymin, 1e-4) == 100 / 512
    assert pytest.approx(xmin, 1e-4) == 100 / 512
    assert pytest.approx(ymax, 1e-4) == 300 / 512
    assert pytest.approx(xmax, 1e-4) == 300 / 512


def test_normalize_bounding_box_xyxy() -> None:
    norm_box = normalize_bounding_box([50, 60, 150, 180], 500, 500, input_format="xyxy")
    assert norm_box is not None
    ymin, xmin, ymax, xmax = norm_box
    assert pytest.approx(ymin, 1e-4) == 60 / 500
    assert pytest.approx(xmin, 1e-4) == 50 / 500
    assert pytest.approx(ymax, 1e-4) == 180 / 500
    assert pytest.approx(xmax, 1e-4) == 150 / 500


def test_normalize_bounding_box_invalid_or_degenerate() -> None:
    # Zero area or inverted
    assert normalize_bounding_box([100, 100, 0, 50], 500, 500, input_format="xywh") is None
    assert normalize_bounding_box([200, 200, 100, 100], 500, 500, input_format="xyxy") is None
    # Invalid coordinate length
    assert normalize_bounding_box([100, 100], 500, 500) is None


def test_check_patient_leakage_detected() -> None:
    def make_sample(sample_id: str, patient_id: str, split: str) -> MedicalVQASample:
        return MedicalVQASample(
            sample_id=sample_id,
            dataset_name="test_data",
            split=split,
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

    train = [make_sample("s1", "pid_001", "train"), make_sample("s2", "pid_002", "train")]
    test_clean = [make_sample("s3", "pid_003", "test")]
    test_leaked = [make_sample("s4", "pid_001", "test")]  # pid_001 in both train and test!

    is_leak, leaked_pids = check_patient_leakage(train, test_clean)
    assert not is_leak
    assert len(leaked_pids) == 0

    is_leak, leaked_pids = check_patient_leakage(train, test_leaked)
    assert is_leak
    assert "pid_001" in leaked_pids
