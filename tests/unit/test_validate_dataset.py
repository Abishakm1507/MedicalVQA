"""Unit tests for the real dataset validation smoke-test tool."""

import sys
from pathlib import Path

import pytest

from datasets.validate_dataset import main, validate_dataset_sample

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


def test_validate_vindr_sample() -> None:
    res = validate_dataset_sample("vindr", FIXTURES_DIR / "vindr", split="train", max_samples=10)
    assert res["dataset"] == "vindr"
    assert res["samples_checked"] == 2
    assert res["patient_ids_available"] is False


def test_validate_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        validate_dataset_sample("slake", "completely_missing_directory")


def test_validate_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        validate_dataset_sample("imagenet", FIXTURES_DIR / "slake")


def test_validate_missing_files_and_dirs(tmp_path: Path) -> None:
    # Empty directory missing required files and directories
    res = validate_dataset_sample("slake", tmp_path, split="test")
    assert res["status"] in ("WARNING", "FAILED")


def test_validate_streaming_failure(tmp_path: Path) -> None:
    # Directory with corrupted JSON file
    bad_json = tmp_path / "test.json"
    with open(bad_json, "w", encoding="utf-8") as f:
        f.write("corrupted json syntax {{{")

    res = validate_dataset_sample("slake", tmp_path, split="test")
    assert res["status"] == "FAILED"
    assert "error" in res


def test_validate_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test CLI invocation through main()
    test_args = [
        "validate_dataset.py",
        "--dataset",
        "slake",
        "--dir",
        str(FIXTURES_DIR / "slake"),
        "--split",
        "test",
        "--max-samples",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()
