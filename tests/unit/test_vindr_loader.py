"""Unit tests for VinDr-CXR dataset loader."""

from pathlib import Path

import pytest

from core.schemas import EvaluationTrack
from datasets.vindr_loader import VinDrCXRDatasetLoader

FIXTURES_DIR = Path("tests/fixtures")
VINDR_DIR = FIXTURES_DIR / "datasets" / "vindr"
IMAGES_DIR = FIXTURES_DIR / "images"


def test_vindr_loader_parses_annotations() -> None:
    loader = VinDrCXRDatasetLoader(dataset_dir=VINDR_DIR, images_dir=IMAGES_DIR)
    # Using 'train' split which loads annotations_train.csv
    samples = loader.load_split("train")

    assert len(samples) == 2

    # Patient 001 has 2 radiologist annotations for Pleural effusion
    p1 = next(s for s in samples if "patient_001" in s.sample_id)
    assert p1.answer == "yes"
    assert len(p1.ground_truth_regions) == 2
    assert p1.ground_truth_regions[0].label == "Pleural effusion"

    # CRITICAL: Verify patient_id is explicitly None (MD5 hash is not patient ID)
    assert p1.patient_id is None
    assert p1.image_id == "patient_001"

    # Verify normalized box bounds
    ymin, xmin, ymax, xmax = p1.ground_truth_regions[0].box_2d
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0

    # Patient 002 has 'No finding'
    p2 = next(s for s in samples if "patient_002" in s.sample_id)
    assert p2.answer == "no"
    assert len(p2.ground_truth_regions) == 0

    assert p1.is_compatible_with(EvaluationTrack.TRACK_A_PREDICTED_CXR)
    assert not p1.is_compatible_with(EvaluationTrack.TRACK_B_ORACLE_SLAKE)


def test_vindr_loader_missing_csv() -> None:
    loader = VinDrCXRDatasetLoader(dataset_dir="missing_vindr", annotations_file="missing.csv")
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("train"))


def test_vindr_loader_loads_pre_split_json() -> None:
    loader = VinDrCXRDatasetLoader(dataset_dir=VINDR_DIR, images_dir=IMAGES_DIR)
    # Using 'test' split which loads vindr_test.json
    samples = loader.load_split("test")
    assert len(samples) == 1
    sample = samples[0]
    assert sample.patient_id is None
    assert sample.image_id == "patient_001"
    assert sample.answer == "yes"
    assert len(sample.ground_truth_regions) == 1
    assert sample.metadata.get("source") == "pre_split_json"
