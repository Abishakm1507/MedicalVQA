"""Unit tests for VQA-RAD dataset loader."""

from pathlib import Path

import pytest

from core.schemas import EvaluationTrack
from datasets.vqarad_loader import VQARADDatasetLoader

FIXTURES_DIR = Path("tests/fixtures")
VQARAD_DIR = FIXTURES_DIR / "datasets" / "vqarad"
IMAGES_DIR = FIXTURES_DIR / "images"


def test_vqarad_loader_loads_samples() -> None:
    loader = VQARADDatasetLoader(dataset_dir=VQARAD_DIR, images_dir=IMAGES_DIR)
    samples = loader.load_split("Dataset")
    assert len(samples) == 3

    # Check question types
    closed_samples = [s for s in samples if s.question_type == "closed"]
    open_samples = [s for s in samples if s.question_type == "open"]
    assert len(closed_samples) == 2
    assert len(open_samples) == 1

    # Verify that VQA-RAD samples explicitly have no bounding boxes
    for s in samples:
        assert s.has_ground_truth_regions is False
        assert len(s.ground_truth_regions) == 0
        assert s.compatible_tracks == [EvaluationTrack.TRACK_C_ANSWER_ONLY]


def test_vqarad_loader_missing_split() -> None:
    loader = VQARADDatasetLoader(dataset_dir="non_existent_dir")
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("test"))
