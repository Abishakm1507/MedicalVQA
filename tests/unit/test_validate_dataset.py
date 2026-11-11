"""Unit tests for the real dataset validation smoke-test tool."""

from pathlib import Path

import pytest

from datasets.validate_dataset import validate_dataset_sample

FIXTURES_DIR = Path("tests/fixtures/datasets")


def test_validate_slake_sample() -> None:
    res = validate_dataset_sample("slake", FIXTURES_DIR / "slake", split="test", max_samples=10)
    assert res["status"] in ("PASSED", "WARNING")
    assert res["dataset"] == "slake"
    assert res["samples_checked"] == 3
    assert res["total_regions"] == 1
    assert res["patient_ids_available"] is False


def test_validate_vqarad_sample() -> None:
    res = validate_dataset_sample(
        "vqarad", FIXTURES_DIR / "vqarad", split="Dataset", max_samples=10
    )
    assert res["dataset"] == "vqarad"
    assert res["samples_checked"] == 3
    assert res["total_regions"] == 0
    assert res["patient_ids_available"] is False


def test_validate_rsna_sample() -> None:
    res = validate_dataset_sample("rsna", FIXTURES_DIR / "rsna", split="train", max_samples=10)
    assert res["dataset"] == "rsna"
    assert res["samples_checked"] == 3
    assert res["patient_ids_available"] is True


def test_validate_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        validate_dataset_sample("slake", "completely_missing_directory")


def test_validate_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        validate_dataset_sample("imagenet", FIXTURES_DIR / "slake")
