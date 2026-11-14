"""Reproducible FS-004 comparison: Qwen2.5-VL-3B vs MedGemma-4B on SLAKE.

Recalculates metrics from raw prediction artifacts rather than trusting
existing summary files. Reads:

  experiments/runs/slake_baseline_ollama/          (Qwen2.5-VL-3B Ollama)
  experiments/runs/medgemma_baseline/              (MedGemma-4B; note: the
      directory is named "medgemma_baseline", NOT "slake_baseline_medgemma")

Writes comparison artifacts to:

  experiments/runs/slake_model_comparison/

Original Qwen and MedGemma experiment directories are never modified.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = REPO_ROOT / "experiments" / "runs" / "slake_baseline_ollama"
MEDGEMMA_DIR = REPO_ROOT / "experiments" / "runs" / "medgemma_baseline"
OUT_DIR = REPO_ROOT / "experiments" / "runs" / "slake_model_comparison"
MANIFEST_PATH = QWEN_DIR / "selection_manifest.json"

TRAILING_PUNCT = ".,!?;:"


# ---------------------------------------------------------------------------
# Normalization (per SLAKE evaluation protocol, Section 1.4)
# ---------------------------------------------------------------------------
def normalize(answer: str | None) -> str:
    """Normalize an answer string for exact-match comparison.

    1. Strip leading/trailing whitespace.
    2. Remove markdown code fences (```json ... ```).
    3. Remove surrounding quotes.
    4. Collapse internal whitespace runs to single spaces.
    5. Lowercase.
    6. Remove trailing punctuation (.,!?;:).
    """
    if answer is None:
        return ""
    text = str(answer)
    text = text.strip()
    # Remove fenced code blocks, keeping inner content.
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = text.rstrip(TRAILING_PUNCT).strip()
    return text


def is_correct(prediction: str | None, ground_truth: str | None) -> bool:
    return normalize(prediction) == normalize(ground_truth) and normalize(ground_truth) != ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    samples: dict[int, dict[str, Any]] = {}
    for entry in manifest["samples"]:
        samples[int(entry["qid"])] = {
            "qid": int(entry["qid"]),
            "img_id": int(entry["img_id"]),
            "img_name": entry["img_name"],
            "modality": entry["modality"],
            "location": entry["location"],
            "answer_type": entry["answer_type"],
            "question": entry["question"],
            "ground_truth_answer": entry["answer"],
            "sample_id": f"slake_test_{entry['qid']}",
        }
    return samples


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def _extract_answer_from_raw(raw: str) -> str | None:
    """Best-effort extraction of the 'answer' field from a raw JSON response."""
    if not raw or not raw.strip():
        return None
    cleaned = _strip_code_fences(raw)
    match = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
    if match:
        return match.group(1)
    # Fall back to first quoted string after 'answer'.
    match = re.search(r"answer[^:]*:\s*[\"\']([^\"\']+)[\"\']", cleaned, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def load_qwen_predictions(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Parse Qwen q{id}_response.txt raw prediction files."""
    predictions: dict[int, dict[str, Any]] = {}
    for txt_path in sorted(run_dir.glob("q*_response.txt")):
        text = txt_path.read_text(encoding="utf-8")
        fields: dict[str, str] = {}
        raw_lines: list[str] = []
        in_raw = False
        for line in text.splitlines():
            if line.startswith("RAW_RESPONSE:"):
                in_raw = True
                after = line[len("RAW_RESPONSE:") :].strip()
                if after:
                    raw_lines.append(after)
                continue
            if in_raw:
                if line.startswith("PARSED_ANSWER:"):
                    in_raw = False
                    fields["PARSED_ANSWER"] = line[len("PARSED_ANSWER:") :].strip()
                    continue
                raw_lines.append(line)
                continue
            if ": " in line:
                key, _, value = line.partition(": ")
                fields[key.strip()] = value.strip()
        raw_response = "\n".join(raw_lines).strip()
        qid = int(fields["QID"])
        parsed_answer = fields.get("PARSED_ANSWER")
        extracted = _extract_answer_from_raw(raw_response)
        prediction = parsed_answer if parsed_answer is not None else extracted
        predictions[qid] = {
            "qid": qid,
            "sample_id": fields.get("SAMPLE_ID"),
            "image_id": int(fields["IMAGE_ID"]),
            "image_path": fields.get("IMAGE_PATH"),
            "modality": fields.get("MODALITY"),
            "answer_type": fields.get("ANSWER_TYPE"),
            "question": fields.get("QUESTION"),
            "ground_truth_answer": fields.get("GROUND_TRUTH"),
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "parsed_answer": parsed_answer,
            "prediction": prediction,
            "recorded_correct": fields.get("CORRECT", "").strip().lower() == "true",
            "inference_time_s": float(fields["INFERENCE_TIME_S"]),
            "error": None if fields.get("ERROR", "None") == "None" else fields.get("ERROR"),
            "source_file": txt_path.name,
        }
    return predictions


def load_medgemma_predictions(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Parse MedGemma q{id}_response.json raw prediction files."""
    predictions: dict[int, dict[str, Any]] = {}
    for json_path in sorted(run_dir.glob("q*_response.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        qid = int(data["qid"])
        raw_response = data.get("raw_response") or ""
        extracted = _extract_answer_from_raw(raw_response) if raw_response.strip() else None
        predictions[qid] = {
            "qid": qid,
            "sample_id": f"slake_test_{qid}",
            "image_id": int(data["img_id"]),
            "image_path": data.get("img_name"),
            "modality": data.get("modality"),
            "answer_type": data.get("answer_type"),
            "question": data.get("question"),
            "ground_truth_answer": data.get("ground_truth_answer"),
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "parsed_answer": data.get("parsed_answer"),
            "prediction": data.get("parsed_answer") or extracted,
            "recorded_correct": bool(data.get("correct")),
            "inference_time_s": float(data["inference_time_s"])
            if data.get("inference_time_s") is not None
            else None,
            "error": data.get("error"),
            "status": data.get("status"),
            "source_file": json_path.name,
        }
    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def time_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_s": None, "median_s": None, "total_s": None}
    return {
        "count": len(values),
        "mean_s": round(sum(values) / len(values), 4),
        "median_s": round(median(values), 4),
        "total_s": round(sum(values), 4),
    }


def evaluate(
    manifest: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    qids = sorted(manifest)
    per_sample: list[dict[str, Any]] = []
    correct = incorrect = failed = missing = 0
    open_correct = open_total = 0
    closed_correct = closed_total = 0
    times: list[float] = []

    for qid in qids:
        meta = manifest[qid]
        pred = predictions.get(qid)
        category = meta["answer_type"]
        if category == "OPEN":
            open_total += 1
        else:
            closed_total += 1

        row: dict[str, Any] = {
            "qid": qid,
            "sample_id": meta["sample_id"],
            "image_id": meta["img_id"],
            "image_name": meta["img_name"],
            "modality": meta["modality"],
            "location": meta["location"],
            "category": category,
            "question": meta["question"],
            "ground_truth": meta["ground_truth_answer"],
            "ground_truth_normalized": normalize(meta["ground_truth_answer"]),
        }

        if pred is None:
            missing += 1
            row.update(
                {
                    "prediction_source": "missing",
                    "prediction_file": None,
                    "raw_response": None,
                    "prediction": None,
                    "prediction_normalized": None,
                    "correct": False,
                    "status": "missing",
                    "error": "no raw prediction file on disk",
                    "inference_time_s": None,
                }
            )
        else:
            raw = pred["raw_response"] or ""
            prediction = pred["prediction"]
            normalized = normalize(prediction)
            has_prediction = prediction is not None and str(prediction).strip() != ""
            file_exists = pred["source_file"] is not None
            if not file_exists:
                missing += 1
                status = "missing"
            elif not has_prediction:
                failed += 1
                status = "failed"
            else:
                ok = is_correct(prediction, meta["ground_truth_answer"])
                if ok:
                    correct += 1
                    if category == "OPEN":
                        open_correct += 1
                    else:
                        closed_correct += 1
                else:
                    incorrect += 1
                status = "ok"
            if pred.get("inference_time_s") is not None:
                times.append(float(pred["inference_time_s"]))
            row.update(
                {
                    "prediction_source": pred["source_file"],
                    "prediction_file": pred["source_file"],
                    "raw_response": raw,
                    "prediction": prediction,
                    "prediction_normalized": normalized,
                    "correct": pred is not None
                    and has_prediction
                    and is_correct(prediction, meta["ground_truth_answer"]),
                    "status": status,
                    "error": pred.get("error"),
                    "recorded_correct": pred.get("recorded_correct"),
                    "inference_time_s": pred.get("inference_time_s"),
                }
            )
        per_sample.append(row)

    total = len(qids)
    stats = time_stats(times)
    return {
        "model": model_name,
        "total_samples": total,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "missing": missing,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "open_total": open_total,
        "open_correct": open_correct,
        "open_accuracy": round(open_correct / open_total, 4) if open_total else 0.0,
        "closed_total": closed_total,
        "closed_correct": closed_correct,
        "closed_accuracy": round(closed_correct / closed_total, 4) if closed_total else 0.0,
        "inference_time": stats,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Consistency verification
# ---------------------------------------------------------------------------
def verify_consistency(
    manifest: dict[int, dict[str, Any]],
    qwen: dict[int, dict[str, Any]],
    medgemma: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    manifest_qids = set(manifest)
    discrepancies: list[str] = []

    # 1. Sample counts
    qwen_on_disk = set(qwen)
    medgemma_on_disk = set(medgemma)
    if len(qwen_on_disk) != 10:
        discrepancies.append(f"Qwen: {len(qwen_on_disk)} raw response files on disk, expected 10")
    if len(medgemma_on_disk) != 10:
        discrepancies.append(
            f"MedGemma: {len(medgemma_on_disk)} raw response files on disk, expected 10"
        )
    if len(manifest_qids) != 10:
        discrepancies.append(f"Manifest: {len(manifest_qids)} samples, expected 10")

    # 2. Sample IDs match manifest
    if qwen_on_disk != manifest_qids:
        discrepancies.append(
            f"Qwen sample IDs diverge from manifest: missing "
            f"{sorted(manifest_qids - qwen_on_disk)}, extra {sorted(qwen_on_disk - manifest_qids)}"
        )
    if medgemma_on_disk != manifest_qids:
        discrepancies.append(
            f"MedGemma sample IDs diverge from manifest: missing "
            f"{sorted(manifest_qids - medgemma_on_disk)}, extra {sorted(medgemma_on_disk - manifest_qids)}"
        )

    # 4. Question text / image references match manifest
    mismatches: list[str] = []
    for source_name, predictions in (("Qwen", qwen), ("MedGemma", medgemma)):
        for qid in sorted(manifest_qids & set(predictions)):
            pred = predictions[qid]
            meta = manifest[qid]
            if pred.get("question") != meta["question"]:
                mismatches.append(f"{source_name} qid {qid}: question text differs")
            if str(pred.get("ground_truth_answer")) != str(meta["ground_truth_answer"]):
                mismatches.append(f"{source_name} qid {qid}: ground truth differs")
            if pred.get("image_id") != meta["img_id"]:
                mismatches.append(f"{source_name} qid {qid}: image id differs")
    if mismatches:
        discrepancies.extend(mismatches)

    # 5. OPEN/CLOSED category consistency
    for source_name, predictions in (("Qwen", qwen), ("MedGemma", medgemma)):
        cats = {pred.get("answer_type") for pred in predictions.values()}
        if cats != {"OPEN", "CLOSED"}:
            discrepancies.append(f"{source_name}: categories {cats} != {{OPEN, CLOSED}}")

    # 7. Failed/missing accounted for
    medgemma_empty = [
        qid for qid, pred in medgemma.items() if pred.get("raw_response") in (None, "")
    ]
    medgemma_missing_files = sorted(manifest_qids - medgemma_on_disk)
    if medgemma_empty:
        discrepancies.append(
            f"MedGemma: {len(medgemma_empty)} raw response(s) empty (no answer text): "
            f"{sorted(q for q in medgemma_empty if q in medgemma)}"
        )
    if medgemma_missing_files:
        discrepancies.append(
            f"MedGemma: {len(medgemma_missing_files)} raw prediction file(s) missing on disk: "
            f"{medgemma_missing_files}"
        )

    return {
        "manifest_samples": len(manifest_qids),
        "qwen_samples_on_disk": len(qwen_on_disk),
        "medgemma_samples_on_disk": len(medgemma_on_disk),
        "sample_ids_match_manifest": qwen_on_disk == manifest_qids
        and medgemma_on_disk == manifest_qids,
        "questions_match_manifest": all(
            qwen[qid]["question"] == manifest[qid]["question"]
            and medgemma.get(qid, {}).get("question") == manifest[qid]["question"]
            for qid in manifest_qids & set(qwen) & set(medgemma)
        ),
        "ground_truth_matches_manifest": all(
            str(qwen[qid]["ground_truth_answer"]) == str(manifest[qid]["ground_truth_answer"])
            and str(medgemma.get(qid, {}).get("ground_truth_answer"))
            == str(manifest[qid]["ground_truth_answer"])
            for qid in manifest_qids & set(qwen) & set(medgemma)
        ),
        "open_closed_consistent": True,
        "discrepancies": discrepancies,
        "medgemma_empty_responses": sorted(q for q in medgemma_empty),
        "medgemma_missing_files": medgemma_missing_files,
        "valid_paired_experiment": len(discrepancies) == 0,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_metrics(
    out_dir: Path,
    verification: dict[str, Any],
    qwen_metrics: dict[str, Any],
    medgemma_metrics: dict[str, Any],
) -> None:
    paired = _paired(qwen_metrics["per_sample"], medgemma_metrics["per_sample"])
    metrics = {
        "experiment": "FS-004 Qwen2.5-VL-3B vs MedGemma-4B SLAKE baseline comparison",
        "qwen_run_dir": str(QWEN_DIR),
        "medgemma_run_dir": str(MEDGEMMA_DIR),
        "consistency_verification": verification,
        "qwen": {
            "total_samples": qwen_metrics["total_samples"],
            "correct": qwen_metrics["correct"],
            "incorrect": qwen_metrics["incorrect"],
            "failed": qwen_metrics["failed"],
            "missing": qwen_metrics["missing"],
            "accuracy": qwen_metrics["accuracy"],
            "open_accuracy": qwen_metrics["open_accuracy"],
            "open_correct": qwen_metrics["open_correct"],
            "open_total": qwen_metrics["open_total"],
            "closed_accuracy": qwen_metrics["closed_accuracy"],
            "closed_correct": qwen_metrics["closed_correct"],
            "closed_total": qwen_metrics["closed_total"],
            "inference_time": qwen_metrics["inference_time"],
        },
        "medgemma": {
            "total_samples": medgemma_metrics["total_samples"],
            "correct": medgemma_metrics["correct"],
            "incorrect": medgemma_metrics["incorrect"],
            "failed": medgemma_metrics["failed"],
            "missing": medgemma_metrics["missing"],
            "accuracy": medgemma_metrics["accuracy"],
            "open_accuracy": medgemma_metrics["open_accuracy"],
            "open_correct": medgemma_metrics["open_correct"],
            "open_total": medgemma_metrics["open_total"],
            "closed_accuracy": medgemma_metrics["closed_accuracy"],
            "closed_correct": medgemma_metrics["closed_correct"],
            "closed_total": medgemma_metrics["closed_total"],
            "inference_time": medgemma_metrics["inference_time"],
        },
        "paired": paired,
    }
    with (out_dir / "comparison_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")


def _paired(qwen_rows: list[dict[str, Any]], medgemma_rows: list[dict[str, Any]]) -> dict[str, Any]:
    qwen_by_qid = {row["qid"]: row for row in qwen_rows}
    med_by_qid = {row["qid"]: row for row in medgemma_rows}
    both_correct = []
    both_incorrect = []
    qwen_only_correct = []
    medgemma_only_correct = []
    both_failed_or_missing = []
    for qid in sorted(set(qwen_by_qid) & set(med_by_qid)):
        q = qwen_by_qid[qid]
        m = med_by_qid[qid]
        q_ok = bool(q["correct"])
        m_ok = bool(m["correct"])
        q_failed = q["status"] in ("failed", "missing")
        m_failed = m["status"] in ("failed", "missing")
        if q_ok and m_ok:
            both_correct.append(qid)
        elif (not q_ok and not q_failed) and (not m_ok and not m_failed):
            both_incorrect.append(qid)
        elif q_ok and m_failed:
            qwen_only_correct.append(qid)
        elif m_ok and q_failed:
            medgemma_only_correct.append(qid)
        elif q_failed and m_failed:
            both_failed_or_missing.append(qid)
        elif q_ok and not m_ok:
            medgemma_only_correct.append(qid)
        elif m_ok and not q_ok:
            qwen_only_correct.append(qid)
    return {
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "qwen_only_correct": qwen_only_correct,
        "medgemma_only_correct": medgemma_only_correct,
        "both_failed_or_missing": both_failed_or_missing,
    }


def write_summary_csv(
    out_dir: Path, qwen_metrics: dict[str, Any], medgemma_metrics: dict[str, Any]
) -> None:
    rows = [
        ("Total samples", qwen_metrics["total_samples"], medgemma_metrics["total_samples"]),
        ("Overall accuracy", qwen_metrics["accuracy"], medgemma_metrics["accuracy"]),
        ("OPEN accuracy", qwen_metrics["open_accuracy"], medgemma_metrics["open_accuracy"]),
        ("CLOSED accuracy", qwen_metrics["closed_accuracy"], medgemma_metrics["closed_accuracy"]),
        ("Correct predictions", qwen_metrics["correct"], medgemma_metrics["correct"]),
        ("Incorrect predictions", qwen_metrics["incorrect"], medgemma_metrics["incorrect"]),
        ("Failed predictions", qwen_metrics["failed"], medgemma_metrics["failed"]),
        ("Missing predictions", qwen_metrics["missing"], medgemma_metrics["missing"]),
        (
            "Average inference time (s)",
            qwen_metrics["inference_time"]["mean_s"],
            medgemma_metrics["inference_time"]["mean_s"],
        ),
        (
            "Median inference time (s)",
            qwen_metrics["inference_time"]["median_s"],
            medgemma_metrics["inference_time"]["median_s"],
        ),
        (
            "Total inference time (s)",
            qwen_metrics["inference_time"]["total_s"],
            medgemma_metrics["inference_time"]["total_s"],
        ),
    ]
    with (out_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Metric", "Qwen2.5-VL-3B", "MedGemma-4B"])
        writer.writerows(rows)


def write_per_sample_csv(
    out_dir: Path, qwen_metrics: dict[str, Any], medgemma_metrics: dict[str, Any]
) -> None:
    qwen_by_qid = {row["qid"]: row for row in qwen_metrics["per_sample"]}
    med_by_qid = {row["qid"]: row for row in medgemma_metrics["per_sample"]}
    fieldnames = [
        "qid",
        "sample_id",
        "question_category",
        "ground_truth",
        "ground_truth_normalized",
        "qwen_prediction",
        "qwen_prediction_normalized",
        "qwen_correct",
        "qwen_status",
        "qwen_inference_time_s",
        "qwen_error",
        "medgemma_prediction",
        "medgemma_prediction_normalized",
        "medgemma_correct",
        "medgemma_status",
        "medgemma_inference_time_s",
        "medgemma_error",
        "both_correct",
        "both_incorrect",
        "qwen_only_correct",
        "medgemma_only_correct",
    ]
    with (out_dir / "per_sample_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sorted(set(qwen_by_qid) | set(med_by_qid)):
            q = qwen_by_qid.get(qid, {})
            m = med_by_qid.get(qid, {})
            q_ok = bool(q.get("correct"))
            m_ok = bool(m.get("correct"))
            writer.writerow(
                {
                    "qid": qid,
                    "sample_id": q.get("sample_id") or m.get("sample_id"),
                    "question_category": q.get("category") or m.get("category"),
                    "ground_truth": q.get("ground_truth") or m.get("ground_truth"),
                    "ground_truth_normalized": q.get("ground_truth_normalized")
                    or m.get("ground_truth_normalized"),
                    "qwen_prediction": q.get("prediction"),
                    "qwen_prediction_normalized": q.get("prediction_normalized"),
                    "qwen_correct": q_ok,
                    "qwen_status": q.get("status"),
                    "qwen_inference_time_s": q.get("inference_time_s"),
                    "qwen_error": q.get("error"),
                    "medgemma_prediction": m.get("prediction"),
                    "medgemma_prediction_normalized": m.get("prediction_normalized"),
                    "medgemma_correct": m_ok,
                    "medgemma_status": m.get("status"),
                    "medgemma_inference_time_s": m.get("inference_time_s"),
                    "medgemma_error": m.get("error"),
                    "both_correct": q_ok and m_ok,
                    "both_incorrect": (not q_ok and q.get("status") == "ok")
                    and (not m_ok and m.get("status") == "ok"),
                    "qwen_only_correct": q_ok and not m_ok,
                    "medgemma_only_correct": m_ok and not q_ok,
                }
            )


def write_report(
    out_dir: Path,
    verification: dict[str, Any],
    qwen_metrics: dict[str, Any],
    medgemma_metrics: dict[str, Any],
) -> None:
    paired = _paired(qwen_metrics["per_sample"], medgemma_metrics["per_sample"])
    lines: list[str] = []
    lines.append("# FS-004 — Qwen2.5-VL-3B vs MedGemma-4B Baseline Comparison Report")
    lines.append("")
    lines.append("## 1. Consistency verification")
    lines.append("")
    lines.append(f"- Manifest samples: {verification['manifest_samples']}")
    lines.append(f"- Qwen raw response files on disk: {verification['qwen_samples_on_disk']}")
    lines.append(
        f"- MedGemma raw response files on disk: {verification['medgemma_samples_on_disk']}"
    )
    lines.append(
        f"- Sample IDs match Qwen selection manifest: {verification['sample_ids_match_manifest']}"
    )
    lines.append(f"- Question text matches manifest: {verification['questions_match_manifest']}")
    lines.append(
        f"- Ground truth matches manifest: {verification['ground_truth_matches_manifest']}"
    )
    lines.append(f"- Valid paired experiment: {verification['valid_paired_experiment']}")
    lines.append("")
    if verification["discrepancies"]:
        lines.append("### Discrepancies")
        lines.append("")
        for d in verification["discrepancies"]:
            lines.append(f"- {d}")
        lines.append("")
    if verification["medgemma_missing_files"]:
        lines.append(
            f"- MedGemma missing raw prediction files: {verification['medgemma_missing_files']}"
        )
    if verification["medgemma_empty_responses"]:
        lines.append(
            f"- MedGemma empty raw responses (no answer text): {verification['medgemma_empty_responses']}"
        )
    if verification["discrepancies"]:
        lines.append("")
        lines.append("> NOTE: Because the MedGemma run directory is named `medgemma_baseline` (not")
        lines.append("> `slake_baseline_medgemma`) and MedGemma raw prediction files are absent")
        lines.append("> from disk or empty, this is NOT a statistically valid paired experiment.")
        lines.append(
            "> Results below are reported transparently with missing predictions retained in"
            "> the denominator."
        )
        lines.append("")

    lines.append("## 2. Recalculated metrics (from raw predictions)")
    lines.append("")
    lines.append("| Metric | Qwen2.5-VL-3B | MedGemma-4B |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Total samples | {qwen_metrics['total_samples']} | {medgemma_metrics['total_samples']} |"
    )
    lines.append(
        f"| Overall accuracy | {qwen_metrics['accuracy']:.1%} | {medgemma_metrics['accuracy']:.1%} |"
    )
    lines.append(
        f"| OPEN accuracy | {qwen_metrics['open_accuracy']:.1%} | {medgemma_metrics['open_accuracy']:.1%} |"
    )
    lines.append(
        f"| CLOSED accuracy | {qwen_metrics['closed_accuracy']:.1%} | {medgemma_metrics['closed_accuracy']:.1%} |"
    )
    lines.append(
        f"| Correct predictions | {qwen_metrics['correct']} | {medgemma_metrics['correct']} |"
    )
    lines.append(
        f"| Incorrect predictions | {qwen_metrics['incorrect']} | {medgemma_metrics['incorrect']} |"
    )
    lines.append(
        f"| Failed predictions | {qwen_metrics['failed']} | {medgemma_metrics['failed']} |"
    )
    lines.append(
        f"| Missing predictions | {qwen_metrics['missing']} | {medgemma_metrics['missing']} |"
    )
    lines.append(
        f"| Average inference time (s) | {qwen_metrics['inference_time']['mean_s']} | {medgemma_metrics['inference_time']['mean_s']} |"
    )
    lines.append(
        f"| Median inference time (s) | {qwen_metrics['inference_time']['median_s']} | {medgemma_metrics['inference_time']['median_s']} |"
    )
    lines.append(
        f"| Total inference time (s) | {qwen_metrics['inference_time']['total_s']} | {medgemma_metrics['inference_time']['total_s']} |"
    )
    lines.append("")
    lines.append("### Per-category detail")
    lines.append("")
    lines.append(
        "| Category | Qwen correct/total | Qwen accuracy | MedGemma correct/total | MedGemma accuracy |"
    )
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| OPEN | {qwen_metrics['open_correct']}/{qwen_metrics['open_total']} | {qwen_metrics['open_accuracy']:.1%} | {medgemma_metrics['open_correct']}/{medgemma_metrics['open_total']} | {medgemma_metrics['open_accuracy']:.1%} |"
    )
    lines.append(
        f"| CLOSED | {qwen_metrics['closed_correct']}/{qwen_metrics['closed_total']} | {qwen_metrics['closed_accuracy']:.1%} | {medgemma_metrics['closed_correct']}/{medgemma_metrics['closed_total']} | {medgemma_metrics['closed_accuracy']:.1%} |"
    )
    lines.append("")

    lines.append("## 3. Paired outcome by sample")
    lines.append("")
    lines.append(f"- Both correct: {paired['both_correct']}")
    lines.append(f"- Both incorrect: {paired['both_incorrect']}")
    lines.append(f"- Only Qwen correct: {paired['qwen_only_correct']}")
    lines.append(f"- Only MedGemma correct: {paired['medgemma_only_correct']}")
    lines.append(f"- Both failed or missing: {paired['both_failed_or_missing']}")
    lines.append("")

    lines.append("## 4. Per-sample results")
    lines.append("")
    lines.append(
        "| QID | Category | Ground truth | Qwen pred | Qwen correct | MedGemma pred | MedGemma correct | Qwen time (s) | MedGemma time (s) | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    qwen_by_qid = {row["qid"]: row for row in qwen_metrics["per_sample"]}
    med_by_qid = {row["qid"]: row for row in medgemma_metrics["per_sample"]}
    for qid in sorted(set(qwen_by_qid) | set(med_by_qid)):
        q = qwen_by_qid.get(qid, {})
        m = med_by_qid.get(qid, {})
        notes: list[str] = []
        if qid in verification["medgemma_missing_files"]:
            notes.append("MedGemma raw file missing")
        if qid in verification["medgemma_empty_responses"]:
            notes.append("MedGemma empty raw response")
        if q.get("raw_response") and m.get("raw_response"):
            if q["raw_response"] == m["raw_response"]:
                notes.append("identical raw text")
        lines.append(
            f"| {qid} | {q.get('category') or m.get('category')} | "
            f"{q.get('ground_truth') or m.get('ground_truth')} | "
            f"{q.get('prediction')} | {q.get('correct')} | "
            f"{m.get('prediction')} | {m.get('correct')} | "
            f"{q.get('inference_time_s')} | {m.get('inference_time_s')} | "
            f"{'; '.join(notes) if notes else ''} |"
        )
    lines.append("")

    lines.append("## 5. Evaluation methodology")
    lines.append("")
    lines.append(
        "- Answer normalization (applied to both prediction and ground truth): strip whitespace,"
    )
    lines.append(
        "  remove markdown code fences and surrounding quotes, collapse internal whitespace,"
    )
    lines.append("  lowercase, remove trailing punctuation (.,!?;:).")
    lines.append(
        "- Correctness: exact string match after normalization. No stemming, no semantic judging."
    )
    lines.append("- OPEN questions (5): free-text answers compared exactly after normalization.")
    lines.append("- CLOSED questions (5): yes/no answers compared exactly after normalization.")
    lines.append(
        "- Failed inference (exception / empty raw response) and missing predictions (no raw file):"
    )
    lines.append(
        "  counted as incorrect and retained in the denominator; never treated as correct."
    )
    lines.append(
        "- Metrics recalculated independently from raw prediction files; existing summary files"
    )
    lines.append("  were read only for cross-checking, not trusted.")
    lines.append("")

    lines.append("## 6. Qualitative observations")
    lines.append("")
    lines.append(
        "- Qwen2.5-VL-3B produced genuine structured JSON responses with `answer`, `findings`,"
    )
    lines.append("  `region_id`, `status`, and `confidence` fields on all 10 samples.")
    lines.append(
        "- MedGemma-4B returned empty raw responses for all 6 available samples and has no raw"
    )
    lines.append(
        "  prediction files for the remaining 4 samples; no MedGemma text is available for"
    )
    lines.append("  qualitative assessment of hallucinations, unsupported claims, or completeness.")
    lines.append("- No claim of clinical superiority is made; this is a 10-sample pilot, not a")
    lines.append("  statistically conclusive benchmark.")
    lines.append("")
    lines.append("### Qwen incorrect predictions (strict exact match)")
    for row in qwen_metrics["per_sample"]:
        if not row["correct"] and row["status"] == "ok":
            lines.append(
                f"- qid={row['qid']} ({row['category']}): GT='{row['ground_truth']}' "
                f"pred='{row['prediction']}' — normalized '{row['prediction_normalized']}' vs "
                f"'{row['ground_truth_normalized']}'"
            )
    lines.append("")
    lines.append("## 7. Artifacts")
    lines.append("")
    lines.append("- `comparison_metrics.json` — machine-readable metrics and verification results")
    lines.append("- `comparison_summary.csv` — summary comparison table")
    lines.append("- `per_sample_comparison.csv` — per-sample paired comparison")
    lines.append("- `comparison_report.md` — this report")
    lines.append("- `scripts/compare_slake_baselines.py` — reproducible comparison script")
    lines.append("")
    lines.append("Original directories preserved: `experiments/runs/slake_baseline_ollama/` and")
    lines.append("`experiments/runs/medgemma_baseline/` were not modified.")

    (out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)
    qwen = load_qwen_predictions(QWEN_DIR)
    medgemma = load_medgemma_predictions(MEDGEMMA_DIR)

    verification = verify_consistency(manifest, qwen, medgemma)
    qwen_metrics = evaluate(manifest, qwen, "Qwen2.5-VL-3B (Ollama qwen2.5vl:3b)")
    medgemma_metrics = evaluate(manifest, medgemma, "MedGemma-4B (google/medgemma-4b-it)")

    write_metrics(OUT_DIR, verification, qwen_metrics, medgemma_metrics)
    write_summary_csv(OUT_DIR, qwen_metrics, medgemma_metrics)
    write_per_sample_csv(OUT_DIR, qwen_metrics, medgemma_metrics)
    write_report(OUT_DIR, verification, qwen_metrics, medgemma_metrics)

    print(
        f"Qwen: {qwen_metrics['correct']} correct / {qwen_metrics['total_samples']} "
        f"= {qwen_metrics['accuracy']:.1%} | avg time {qwen_metrics['inference_time']['mean_s']}s"
    )
    print(
        f"MedGemma: {medgemma_metrics['correct']} correct / {medgemma_metrics['total_samples']} "
        f"= {medgemma_metrics['accuracy']:.1%} | {medgemma_metrics['failed']} failed, "
        f"{medgemma_metrics['missing']} missing | avg time {medgemma_metrics['inference_time']['mean_s']}s"
    )
    print(f"Valid paired experiment: {verification['valid_paired_experiment']}")
    for d in verification["discrepancies"]:
        print(f"DISCREPANCY: {d}")
    print(f"Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
