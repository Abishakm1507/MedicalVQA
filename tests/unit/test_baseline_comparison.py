"""Unit tests for the FS-004 baseline comparison logic.

Covers answer normalization, correctness evaluation, sample matching against
the original manifest, and handling of failed / missing predictions.
"""

from pathlib import Path

import pytest

from scripts.compare_slake_baselines import (
    evaluate,
    is_correct,
    load_manifest,
    load_qwen_predictions,
    normalize,
    verify_consistency,
)

QWEN_DIR = Path("experiments/runs/slake_baseline_ollama")
MANIFEST_PATH = QWEN_DIR / "selection_manifest.json"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_normalize_strips_and_lowercases() -> None:
    assert normalize("  Chest  ") == "chest"


def test_normalize_removes_code_fences() -> None:
    assert normalize('```json\n{"answer": "chest"}\n```') == '{"answer": "chest"}'


def test_normalize_removes_surrounding_quotes() -> None:
    assert normalize('"Yes"') == "yes"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("Lower   Left   Lung") == "lower left lung"


def test_normalize_removes_trailing_punctuation() -> None:
    assert normalize("X-Ray.") == "x-ray"
    assert normalize("No!") == "no"


def test_normalize_none_returns_empty() -> None:
    assert normalize(None) == ""


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def test_is_correct_exact_match() -> None:
    assert is_correct("chest", "Chest") is True
    assert is_correct("yes", "Yes") is True
    assert is_correct("x-ray", "X-Ray") is True


def test_is_correct_mismatch() -> None:
    assert is_correct("lungs", "Lung") is False  # no stemming
    assert is_correct("Right Lung", "Lower Left Lung") is False


def test_is_correct_empty_prediction_is_incorrect() -> None:
    assert is_correct(None, "Chest") is False
    assert is_correct("", "Chest") is False


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def test_load_manifest_contains_ten_samples() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert len(manifest) == 10
    assert manifest[11981]["question"] == "Which part of the body does this image belong to?"
    assert manifest[11981]["ground_truth_answer"] == "Chest"
    assert manifest[11981]["answer_type"] == "OPEN"
    assert manifest[12914]["img_id"] == 605  # shared image with 12919


def test_manifest_category_counts() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    open_qids = [q for q, m in manifest.items() if m["answer_type"] == "OPEN"]
    closed_qids = [q for q, m in manifest.items() if m["answer_type"] == "CLOSED"]
    assert len(open_qids) == 5
    assert len(closed_qids) == 5


# ---------------------------------------------------------------------------
# Evaluation: missing / failed predictions retained in denominator
# ---------------------------------------------------------------------------
def _pred(
    qid,
    ground_truth_answer,
    prediction=None,
    raw_response="",
    status="ok",
    inference_time=1.0,
    missing_file=False,
    category="OPEN",
):
    return {
        "qid": qid,
        "sample_id": f"s{qid}",
        "image_id": qid,
        "image_path": f"{qid}.jpg",
        "modality": "X-RAY",
        "answer_type": category,
        "question": f"q{qid}",
        "ground_truth_answer": ground_truth_answer,
        "raw_response": raw_response,
        "extracted_answer": prediction,
        "parsed_answer": prediction,
        "prediction": prediction,
        "recorded_correct": prediction == ground_truth_answer if prediction is not None else False,
        "inference_time_s": inference_time,
        "error": None,
        "status": status,
        "source_file": None if missing_file else f"q{qid}_response.txt",
    }


def _manifest() -> dict[int, dict]:
    return {
        1: {
            "sample_id": "s1",
            "img_id": 1,
            "img_name": "a.jpg",
            "modality": "X-RAY",
            "location": "lung",
            "answer_type": "OPEN",
            "question": "q1",
            "ground_truth_answer": "Chest",
        },
        2: {
            "sample_id": "s2",
            "img_id": 2,
            "img_name": "b.jpg",
            "modality": "CT",
            "location": "abdomen",
            "answer_type": "CLOSED",
            "question": "q2",
            "ground_truth_answer": "Yes",
        },
        3: {
            "sample_id": "s3",
            "img_id": 3,
            "img_name": "c.jpg",
            "modality": "CT",
            "location": "abdomen",
            "answer_type": "OPEN",
            "question": "q3",
            "ground_truth_answer": "0",
        },
    }


def test_evaluation_counts_failed_and_missing_in_denominator() -> None:
    manifest = _manifest()
    predictions = {
        1: _pred(1, "Chest", prediction="Chest"),  # correct
        2: _pred(2, "Yes", prediction="No"),  # incorrect
        3: _pred(3, "0", prediction=None, raw_response="", status="failed"),  # failed
    }
    result = evaluate(manifest, predictions, "TestModel")
    assert result["total_samples"] == 3
    assert result["correct"] == 1
    assert result["incorrect"] == 1
    assert result["failed"] == 1
    assert result["missing"] == 0
    assert result["accuracy"] == pytest.approx(round(1 / 3, 4))


def test_evaluation_counts_missing_file_in_denominator() -> None:
    manifest = _manifest()
    predictions = {
        1: _pred(1, "Chest", prediction="Chest"),
        2: None,  # no raw file on disk
        3: _pred(3, "0", prediction="0"),
    }
    result = evaluate(manifest, predictions, "TestModel")
    assert result["missing"] == 1
    assert result["correct"] == 2
    assert result["accuracy"] == pytest.approx(round(2 / 3, 4))


def test_evaluation_per_category() -> None:
    manifest = _manifest()
    predictions = {
        1: _pred(1, "Chest", prediction="Chest"),
        2: _pred(2, "Yes", prediction="No"),
        3: _pred(3, "0", prediction="0"),
    }
    result = evaluate(manifest, predictions, "TestModel")
    assert result["open_correct"] == 2
    assert result["open_total"] == 2
    assert result["open_accuracy"] == 1.0
    assert result["closed_correct"] == 0
    assert result["closed_total"] == 1
    assert result["closed_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Consistency verification
# ---------------------------------------------------------------------------
def test_verify_consistency_matching_manifest() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    qwen = load_qwen_predictions(QWEN_DIR)
    verification = verify_consistency(manifest, qwen, dict(qwen))
    assert verification["manifest_samples"] == 10
    assert verification["qwen_samples_on_disk"] == 10
    assert verification["medgemma_samples_on_disk"] == 10
    assert verification["sample_ids_match_manifest"] is True
    assert verification["questions_match_manifest"] is True
    assert verification["ground_truth_matches_manifest"] is True
    assert verification["open_closed_consistent"] is True
    assert verification["valid_paired_experiment"] is True


def test_verify_consistency_flags_missing_medgemma_files() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    qwen = load_qwen_predictions(QWEN_DIR)
    medgemma = {qid: pred for qid, pred in qwen.items() if qid != 12129}
    verification = verify_consistency(manifest, qwen, medgemma)
    assert 12129 in verification["medgemma_missing_files"]
    assert verification["valid_paired_experiment"] is False
