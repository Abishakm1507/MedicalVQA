"""Unit tests for SLAKE dataset loader."""

from pathlib import Path

import pytest

from core.schemas import EvaluationTrack
from datasets.slake_loader import SlakeDatasetLoader

FIXTURES_DIR = Path("tests/fixtures")
SLAKE_DIR = FIXTURES_DIR / "datasets" / "slake"
IMAGES_DIR = FIXTURES_DIR / "images"


def test_slake_loader_loads_test_split() -> None:
    loader = SlakeDatasetLoader(dataset_dir=SLAKE_DIR, images_dir=IMAGES_DIR, language="en")
    samples = loader.load_split("test")

    # Entry 5 is Chinese ('zh'), should be filtered out
    assert len(samples) == 3

    # Check first sample (CXR consolidation)
    s1 = samples[0]
    assert s1.dataset_name == "slake"
    assert s1.split == "test"
    assert s1.modality == "CXR"
    assert s1.organ == "chest"
    assert s1.question_type == "open"
    assert len(s1.ground_truth_regions) == 1

    # Check normalized coordinate bounds
    region = s1.ground_truth_regions[0]
    assert region.label == "consolidation"
    ymin, xmin, ymax, xmax = region.box_2d
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0

    # CXR sample should be compatible with both Track A and Track B
    assert EvaluationTrack.TRACK_A_PREDICTED_CXR in s1.compatible_tracks
    assert EvaluationTrack.TRACK_B_ORACLE_SLAKE in s1.compatible_tracks
    assert EvaluationTrack.TRACK_C_ANSWER_ONLY in s1.compatible_tracks


def test_slake_loader_modality_track_separation() -> None:
    loader = SlakeDatasetLoader(dataset_dir=SLAKE_DIR, images_dir=IMAGES_DIR, language="en")
    samples = loader.load_split("test")

    # Find the Brain MRI sample (id=4 in test.json)
    brain_sample = next(s for s in samples if s.organ == "brain")
    assert brain_sample.modality == "MRI"
    # Brain MRI must NOT be assigned to Track A (CXR predicted detector track)
    assert EvaluationTrack.TRACK_A_PREDICTED_CXR not in brain_sample.compatible_tracks
    # Must be present in Track B (Oracle GT boxes across all SLAKE)
    assert EvaluationTrack.TRACK_B_ORACLE_SLAKE in brain_sample.compatible_tracks


def test_slake_loader_missing_split_file() -> None:
    loader = SlakeDatasetLoader(dataset_dir=SLAKE_DIR)
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("non_existent_split"))


def test_base_loader_strict_image_exists_raises() -> None:
    loader = SlakeDatasetLoader(
        dataset_dir=SLAKE_DIR,
        images_dir=Path("completely_missing_images_folder"),
        strict_image_exists=True,
    )
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("test"))
