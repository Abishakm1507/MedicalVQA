"""Unit tests for RSNA Pneumonia challenge dataset loader."""

from pathlib import Path

import pytest

from core.schemas import EvaluationTrack
from datasets.rsna_loader import RSNADatasetLoader

FIXTURES_DIR = Path("tests/fixtures")
RSNA_DIR = FIXTURES_DIR / "datasets" / "rsna"
IMAGES_DIR = FIXTURES_DIR / "images"


def test_rsna_loader_parses_labels_csv() -> None:
    loader = RSNADatasetLoader(dataset_dir=RSNA_DIR, images_dir=IMAGES_DIR)
    samples = loader.load_split("train")

    assert len(samples) == 3

    # Patient 001 has Target=1 and 1 box
    p1 = next(s for s in samples if s.patient_id == "patient_001")
    assert p1.answer == "yes"
    assert len(p1.ground_truth_regions) == 1
    assert p1.ground_truth_regions[0].label == "lung opacity"
    ymin, xmin, ymax, xmax = p1.ground_truth_regions[0].box_2d
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0

    # Patient 002 has Target=0 and 0 boxes
    p2 = next(s for s in samples if s.patient_id == "patient_002")
    assert p2.answer == "no"
    assert len(p2.ground_truth_regions) == 0

    # RSNA is compatible with Track A
    assert EvaluationTrack.TRACK_A_PREDICTED_CXR in p1.compatible_tracks


def test_rsna_loader_missing_labels_file() -> None:
    loader = RSNADatasetLoader(dataset_dir=RSNA_DIR, labels_file="missing_labels.csv")
    # Using 'train' split which falls back to labels_file (no rsna_train.json exists)
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("train"))


def test_rsna_loader_loads_pre_split_json() -> None:
    loader = RSNADatasetLoader(dataset_dir=RSNA_DIR, images_dir=IMAGES_DIR)
    # Using 'test' split which loads rsna_test.json
    samples = loader.load_split("test")
    assert len(samples) == 1
    sample = samples[0]
    assert sample.patient_id == "patient_003"
    assert sample.answer == "yes"
    assert len(sample.ground_truth_regions) == 1
    assert sample.metadata.get("source") == "pre_split_json"
