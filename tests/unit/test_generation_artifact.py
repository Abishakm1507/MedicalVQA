"""Unit tests for the FS-005 Phase 5 generation-artifact system.

Covers: record serialization/validation, deterministic condition-independent
generation identifiers, artifact store persistence, skip/retry/corrupt
handling, and duplicate prevention.
"""

import json

import pytest

from core.generation_artifact import (
    GenerationArtifactStore,
    GenerationRecord,
    GenerationSample,
    GenerationStatus,
    ValidationStatus,
    compute_generation_id,
    validate_generation_record,
)


def _make_record(
    generation_id: str = "gen_abc123",
    status: GenerationStatus = GenerationStatus.COMPLETED,
    n_samples: int = 2,
    error: str | None = None,
) -> GenerationRecord:
    return GenerationRecord(
        generation_id=generation_id,
        model_name="mock-vlm",
        model_revision=None,
        track="B",
        question_id="q1",
        image_id="img1",
        n_samples=n_samples,
        temperature=0.7,
        random_seed=42,
        samples=[
            GenerationSample(
                sample_index=0,
                raw_response='{"answer": "Yes", "findings": []}',
                parsed_answer="Yes",
                claims=[],
                inference_duration_s=1.0,
                status="ok",
            ),
            GenerationSample(
                sample_index=1,
                raw_response='{"answer": "Yes", "findings": []}',
                parsed_answer="Yes",
                claims=[],
                inference_duration_s=1.1,
                status="ok",
            ),
        ],
        status=status,
        validation_status=ValidationStatus.VALID,
        total_duration_s=2.1,
        error=error,
    )


def test_record_serialization_roundtrip() -> None:
    record = _make_record()
    data = json.loads(record.model_dump_json())
    restored = GenerationRecord.model_validate(data)
    assert restored.generation_id == record.generation_id
    assert restored.n_samples == 2
    assert len(restored.samples) == 2
    assert restored.status == GenerationStatus.COMPLETED
    assert restored.validation_status == ValidationStatus.VALID


def test_validation_requires_sample_count_match() -> None:
    record = _make_record(n_samples=2)
    record.samples = record.samples[:1]  # mismatch: 1 sample vs n_samples=2
    violations = validate_generation_record(record)
    assert any("n_samples" in v for v in violations)


def test_validation_rejects_non_contiguous_indices() -> None:
    record = _make_record()
    record.n_samples = 3
    # Rebuild with 3 samples matching n_samples, then use a non-contiguous index.
    record.samples = [
        GenerationSample(sample_index=0, raw_response="a", status="ok"),
        GenerationSample(sample_index=1, raw_response="b", status="ok"),
        GenerationSample(sample_index=5, raw_response="c", status="ok"),
    ]
    violations = validate_generation_record(record)
    assert any("contiguous" in v for v in violations)


def test_validation_rejects_completed_with_error() -> None:
    record = _make_record()
    # Construct valid, then mutate into an invalid business state (pydantic does
    # not re-validate on plain attribute assignment by default).
    record.status = GenerationStatus.COMPLETED
    record.error = "boom"
    violations = validate_generation_record(record)
    assert any("Completed record carries a generation error" in v for v in violations)


def test_validation_rejects_failed_without_error() -> None:
    record = _make_record()
    record.status = GenerationStatus.FAILED
    record.error = None
    violations = validate_generation_record(record)
    assert any("Failed record carries no error message" in v for v in violations)


def test_deterministic_id_same_inputs_same_output() -> None:
    id1 = compute_generation_id(
        "medgemma-4b", None, "B", "q1", "img1", 5, 0.7, 42
    )
    id2 = compute_generation_id(
        "medgemma-4b", None, "B", "q1", "img1", 5, 0.7, 42
    )
    assert id1 == id2
    assert len(id1) == 64  # full SHA-256 hex


def test_deterministic_id_different_inputs_different_output() -> None:
    id_a = compute_generation_id("medgemma-4b", None, "B", "q1", "img1", 5, 0.7, 42)
    id_b = compute_generation_id("medgemma-4b", None, "B", "q1", "img1", 5, 0.7, 43)
    id_c = compute_generation_id("qwen2.5-vl-3b", None, "B", "q1", "img1", 5, 0.7, 42)
    id_d = compute_generation_id("medgemma-4b", None, "C", "q1", "img1", 5, 0.7, 42)
    assert len({id_a, id_b, id_c, id_d}) == 4


def test_store_save_load_exists(tmp_path) -> None:
    store = GenerationArtifactStore(root=tmp_path / "generations")
    record = _make_record()
    path = store.save(record)
    assert path.exists()
    assert store.exists("mock-vlm", "B", record.generation_id)
    loaded = store.load("mock-vlm", "B", record.generation_id)
    assert loaded is not None
    assert loaded.generation_id == record.generation_id
    assert loaded.status == GenerationStatus.COMPLETED


def test_store_load_missing_returns_none(tmp_path) -> None:
    store = GenerationArtifactStore(root=tmp_path / "generations")
    assert store.load("mock-vlm", "B", "nonexistent") is None
    assert store.exists("mock-vlm", "B", "nonexistent") is False


def test_store_is_completed_valid(tmp_path) -> None:
    store = GenerationArtifactStore(root=tmp_path / "generations")
    record = _make_record()
    store.save(record)
    assert store.is_completed_valid("mock-vlm", "B", record.generation_id) is True
    assert store.is_completed_valid("mock-vlm", "C", record.generation_id) is False


def test_store_detects_corrupt_artifact(tmp_path) -> None:
    store = GenerationArtifactStore(root=tmp_path / "generations")
    record = _make_record()
    path = store.save(record)
    # Corrupt the file.
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or unparseable"):
        store.load("mock-vlm", "B", record.generation_id)
    assert store.is_completed_valid("mock-vlm", "B", record.generation_id) is False


def test_store_rejects_invalid_record(tmp_path) -> None:
    store = GenerationArtifactStore(root=tmp_path / "generations")
    record = _make_record()
    record.samples = []  # invalid: 0 samples vs n_samples=2
    with pytest.raises(ValueError, match="n_samples"):
        store.save(record, validate=True)
    # Nothing should have been written.
    assert not store.exists("mock-vlm", "B", record.generation_id)
