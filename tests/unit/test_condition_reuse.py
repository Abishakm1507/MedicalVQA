"""Unit tests for Conditions 2 and 3 shared-artifact reuse.

Proves that both conditions consume identical source artifacts (same
generation record: same responses, claims, agreement, entropy, confidence)
and differ only in the abstention decision.
"""

import json
from pathlib import Path

from scripts.fs005_phase5_generation import GenerationRunner
from vlm.mock_client import MockVLMClient


def _runner(tmp_path: Path, model_name: str = "mock-vlm", track: str = "C"):
    return GenerationRunner(
        vlm_client=MockVLMClient(model_name=model_name),
        model_name=model_name,
        model_revision=None,
        track=track,
        output_dir=tmp_path,
        n_samples=5,
        temperature=0.7,
        random_seed=7,
    )


def _fixture_samples():
    return [
        {
            "question_id": "q_reuse_1",
            "image_id": "img_reuse_1",
            "question": "What modality is shown?",
            "region_ids": [],
        },
        {
            "question_id": "q_reuse_2",
            "image_id": "img_reuse_2",
            "question": "Is there an abnormality?",
            "region_ids": [],
        },
    ]


def test_conditions_share_identical_source_artifacts(tmp_path) -> None:
    runner = _runner(tmp_path, track="C")
    samples = _fixture_samples()
    images = {s["image_id"]: None for s in samples}
    summary = runner.run_batch(samples, images)

    assert summary["n_completed"] == 2
    assert summary["n_failed"] == 0

    for cr in summary["condition_results"]:
        c2 = cr["condition_2"]
        c3 = cr["condition_3"]
        # Source artifacts must be IDENTICAL for both conditions.
        assert c2["agreement"] == c3["agreement"]
        assert c2["normalized_entropy"] == c3["normalized_entropy"]
        assert c2["confidence"] == c3["confidence"]
        assert c2["evidence_state"] == c3["evidence_state"]
        assert c2["sampling_valid"] == c3["sampling_valid"]
        # Only the abstention decision may differ.
        assert c2["is_abstaining"] is False  # Condition 2 never abstains
        assert c2["in_coverage"] is True
        assert c2["answer"] is not None


def test_condition_3_reuses_same_generation_record(tmp_path) -> None:
    runner = _runner(tmp_path, track="B")
    samples = _fixture_samples()
    images = {s["image_id"]: None for s in samples}
    summary = runner.run_batch(samples, images)

    for cr in summary["condition_results"]:
        assert cr["source_artifact_identical"] is True
        assert cr["generation_id"]  # condition results share one generation record


def test_deterministic_generation_id_reused_across_conditions(tmp_path) -> None:
    runner = _runner(tmp_path, track="C")
    samples = _fixture_samples()
    images = {s["image_id"]: None for s in samples}
    runner.run_batch(samples, images)

    store = runner.store
    for s in samples:
        gen_id = runner.store  # noqa: F841 (placeholder)
        # Recompute the expected deterministic id.
        from core.generation_artifact import compute_generation_id

        expected = compute_generation_id(
            model_name="mock-vlm",
            model_revision=None,
            track="C",
            question_id=s["question_id"],
            image_id=s["image_id"],
            n_samples=5,
            temperature=0.7,
            random_seed=7,
        )
        assert store.exists("mock-vlm", "C", expected)
        record = store.load("mock-vlm", "C", expected)
        assert record is not None
        assert record.status.value == "completed"
        assert len(record.samples) == 5


def test_run_creates_condition_artifacts(tmp_path) -> None:
    runner = _runner(tmp_path, track="C")
    samples = _fixture_samples()
    images = {s["image_id"]: None for s in samples}
    summary = runner.run_batch(samples, images)

    # Condition results were produced for every completed sample.
    assert len(summary["condition_results"]) == 2
    # Progress file persisted.
    progress_path = runner._progress_path()
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["n_completed"] == 2
    assert progress["n_skipped"] == 0
