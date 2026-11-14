"""Unit tests for resumable execution: interruption, resume, duplicate prevention, corrupt artifacts."""

import json
from pathlib import Path

import pytest

from core.generation_artifact import (
    GenerationRecord,
    GenerationSample,
    GenerationStatus,
    ValidationStatus,
    compute_generation_id,
)
from scripts.fs005_phase5_generation import GenerationRunner
from vlm.mock_client import MockVLMClient


def _runner(tmp_path: Path, track: str = "C"):
    return GenerationRunner(
        vlm_client=MockVLMClient(model_name="mock-vlm"),
        model_name="mock-vlm",
        model_revision=None,
        track=track,
        output_dir=tmp_path,
        n_samples=5,
        temperature=0.7,
        random_seed=11,
    )


def _samples():
    return [
        {"question_id": "q_resume_1", "image_id": "img_resume_1", "question": "Q1?", "region_ids": []},
        {"question_id": "q_resume_2", "image_id": "img_resume_2", "question": "Q2?", "region_ids": []},
        {"question_id": "q_resume_3", "image_id": "img_resume_3", "question": "Q3?", "region_ids": []},
    ]


def test_first_run_generates_all(tmp_path) -> None:
    runner = _runner(tmp_path)
    summary = runner.run_batch(_samples(), {s["image_id"]: None for s in _samples()})
    assert summary["n_completed"] == 3
    assert summary["n_skipped"] == 0
    assert summary["n_failed"] == 0
    assert summary["n_retried"] == 0


def test_second_run_skips_all_duplicate_prevention(tmp_path) -> None:
    runner = _runner(tmp_path)
    samples = _samples()
    images = {s["image_id"]: None for s in samples}
    first = runner.run_batch(samples, images)
    assert first["n_completed"] == 3

    second = runner.run_batch(samples, images)
    # Counters reset per call: all samples skipped, no new completions.
    assert second["n_completed"] == 0
    assert second["n_skipped"] == 3  # all valid completed records skipped
    assert second["n_failed"] == 0
    assert second["n_retried"] == 0


def test_retry_failed_record(tmp_path) -> None:
    runner = _runner(tmp_path)
    samples = _samples()
    images = {s["image_id"]: None for s in samples}
    # Pre-seed a FAILED record for the first sample.
    qid = samples[0]["question_id"]
    iid = samples[0]["image_id"]
    gen_id = compute_generation_id(
        "mock-vlm", None, "C", qid, iid, 5, 0.7, 11
    )
    failed_record = GenerationRecord(
        generation_id=gen_id,
        model_name="mock-vlm",
        model_revision=None,
        track="C",
        question_id=qid,
        image_id=iid,
        n_samples=5,
        temperature=0.7,
        random_seed=11,
        samples=[
            GenerationSample(sample_index=i, raw_response=None, status="failed", error="transient")
            for i in range(5)
        ],
        status=GenerationStatus.FAILED,
        validation_status=ValidationStatus.INVALID,
        total_duration_s=None,
        error="transient failure",
    )
    runner.store.save(failed_record, validate=False)

    summary = runner.run_batch(samples, images)
    # The failed record is retried and replaced with a completed one.
    assert summary["n_retried"] >= 1
    assert summary["n_completed"] == 3
    record = runner.store.load("mock-vlm", "C", gen_id)
    assert record is not None
    assert record.status == GenerationStatus.COMPLETED


def test_corrupt_artifact_detected_and_regenerated(tmp_path) -> None:
    runner = _runner(tmp_path)
    samples = _samples()
    images = {s["image_id"]: None for s in samples}
    first = runner.run_batch(samples, images)
    assert first["n_completed"] == 3

    # Corrupt the first sample's artifact file.
    qid = samples[0]["question_id"]
    iid = samples[0]["image_id"]
    gen_id = compute_generation_id("mock-vlm", None, "C", qid, iid, 5, 0.7, 11)
    path = runner.store._record_path("mock-vlm", "C", gen_id)
    path.write_text("{corrupt", encoding="utf-8")

    # Corrupt artifact is detected as not valid-completed.
    assert runner.store.is_completed_valid("mock-vlm", "C", gen_id) is False
    with pytest.raises(ValueError, match="Corrupt or unparseable"):
        runner.store.load("mock-vlm", "C", gen_id)

    # Re-running regenerates the corrupt record (no permanent data loss).
    repaired = runner.run_batch(samples, images)
    assert repaired["n_completed"] == 1
    assert repaired["n_skipped"] == 2
    assert repaired["n_retried"] == 1
    assert repaired["n_corrupt"] == 1
    assert repaired["progress"]["corrupt_question_ids"] == [qid]
    record = runner.store.load("mock-vlm", "C", gen_id)
    assert record is not None
    assert record.status == GenerationStatus.COMPLETED
    assert runner.store.is_completed_valid("mock-vlm", "C", gen_id) is True


def test_interruption_resume_progress_persisted(tmp_path) -> None:
    runner = _runner(tmp_path)
    samples = _samples()
    images = {s["image_id"]: None for s in samples}
    runner.run_batch(samples, images)
    progress_path = runner._progress_path()
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["n_completed"] == 3
    assert progress["updated_at_utc"]

    # Resume: same run should load progress and skip all.
    second = runner.run_batch(samples, images)
    # Counters reset per call: progress file reflects this run's stats.
    assert second["progress"]["n_completed"] == 0
    assert second["progress"]["n_skipped"] == 3
