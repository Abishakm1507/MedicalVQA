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

    # CRITICAL: Verify patient identity is explicitly None (not img_id proxy)
    assert s1.patient_id is None
    assert s1.image_id is not None

    # Check normalized coordinate bounds
    region = s1.ground_truth_regions[0]
    assert region.label == "consolidation"
    ymin, xmin, ymax, xmax = region.box_2d
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0

    # CXR sample should be compatible with Track A, Track B, and Track C
    assert s1.is_compatible_with(EvaluationTrack.TRACK_A_PREDICTED_CXR)
    assert s1.is_compatible_with(EvaluationTrack.TRACK_B_ORACLE_SLAKE)
    assert s1.is_compatible_with(EvaluationTrack.TRACK_C_ANSWER_ONLY)


def test_slake_loader_modality_track_separation() -> None:
    loader = SlakeDatasetLoader(dataset_dir=SLAKE_DIR, images_dir=IMAGES_DIR, language="en")
    samples = loader.load_split("test")

    # Find the Brain MRI sample (id=4 in test.json)
    brain_sample = next(s for s in samples if s.organ == "brain")
    assert brain_sample.modality == "MRI"

    # Brain MRI must NOT be assigned to Track A (CXR predicted detector track)
    assert not brain_sample.is_compatible_with(EvaluationTrack.TRACK_A_PREDICTED_CXR)
    with pytest.raises(ValueError, match="incompatible with evaluation track"):
        brain_sample.assert_track_compatibility(EvaluationTrack.TRACK_A_PREDICTED_CXR)

    # Must be valid for Track B (Oracle GT boxes across all SLAKE) and Track C
    assert brain_sample.is_compatible_with(EvaluationTrack.TRACK_B_ORACLE_SLAKE)
    assert brain_sample.is_compatible_with(EvaluationTrack.TRACK_C_ANSWER_ONLY)


def test_slake_loader_missing_split_file() -> None:
    loader = SlakeDatasetLoader(dataset_dir=SLAKE_DIR)
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("non_existent_split"))


def test_slake_loader_strict_image_exists_raises() -> None:
    loader = SlakeDatasetLoader(
        dataset_dir=SLAKE_DIR,
        images_dir=Path("completely_missing_images_folder"),
        strict_image_exists=True,
    )
    with pytest.raises(FileNotFoundError):
        list(loader.stream_split("test"))


def test_slake_loader_malformed_json(tmp_path: Path) -> None:
    bad_json = tmp_path / "test.json"
    with open(bad_json, "w", encoding="utf-8") as f:
        f.write('{"not_a_list": true}')

    loader = SlakeDatasetLoader(dataset_dir=tmp_path)
    with pytest.raises(ValueError, match="expected list"):
        list(loader.stream_split("test"))
