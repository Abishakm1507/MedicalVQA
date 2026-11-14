"""FS-005 Phase 5 Stage 3: answer-accuracy evaluation of the FS-004 baselines.

Reads the FS-004 baseline artifacts (Qwen2.5-VL-3B via Ollama and MedGemma-4B on
10 SLAKE test questions) and reports only what those artifacts can support:
overall, OPEN and CLOSED answer accuracy.

The FS-004 baselines were free-form answers with a single sample. They contain
no region citations, no sampled answers, no semantic clusters and no confidence
scores. The following metrics therefore CANNOT be computed from them and are
reported as ``null`` with a reason, never as a number:

  - USR_raw / USR_filtered   (no claims with region citations)
  - pointing accuracy        (no cited regions)
  - sampling agreement and normalized entropy (N = 1, no sampling)
  - ECE, AUROC, selective accuracy, coverage, accuracy-vs-coverage
                             (no confidence score, no abstention decision)

An earlier version of this script filled these inputs with placeholders
(confidence = 0.8 if correct else 0.3, agreement = 0.8, entropy = 0.2,
pointing = correct). That made every derived metric circular, so it was removed.
Those metrics must come from the integrated pipeline (generation harness plus
confidence gate) run with real models.

Traceability:
  - Feature Specification: FS-005 (Phase 5), sections 7-11
  - AGENTS.md section 3.3: never fabricate scores or calibration results
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (  # noqa: E402
    SampleRecord,
    compute_accuracy,
    compute_closed_accuracy,
    compute_open_accuracy,
)

NOT_COMPUTABLE_REASON = (
    "Not computable from FS-004 baseline artifacts: free-form single-sample "
    "answers with no region citations, no sampled answers and no confidence "
    "score. Requires the integrated pipeline run with real models."
)

NOT_COMPUTABLE_METRICS = [
    "usr_raw",
    "usr_filtered",
    "pointing_accuracy",
    "sampling_agreement",
    "normalized_entropy",
    "ece",
    "selective_accuracy",
    "coverage",
    "auroc",
]


def _load_records(run_dir: Path, track: str, condition: int) -> list[SampleRecord]:
    """Convert a baseline_summary.json into records usable for accuracy only.

    Fields that the baseline cannot provide (claims, uncertainty) are set to
    neutral zero values. They are never read: only ``correct`` and
    ``question_type`` feed the metrics computed by this script.
    """
    with open(run_dir / "baseline_summary.json", encoding="utf-8") as f:
        baseline = json.load(f)

    records = []
    for result in baseline.get("results", []):
        records.append(
            SampleRecord(
                sample_id=str(result.get("qid", "")),
                dataset_name="SLAKE",
                split="test",
                track=track,
                condition=condition,
                ground_truth_answer=result.get("ground_truth_answer", ""),
                model_answer=result.get("parsed_answer", ""),
                correct=bool(result.get("correct", False)),
                question_type=result.get("answer_type", "CLOSED"),
                has_gt_regions=False,
                pointing_correct=False,
                n_claims=0,
                n_positive_claims=0,
                n_supported=0,
                n_unsupported=0,
                n_unverifiable=0,
                n_absence=0,
                n_uncertain=0,
                agreement=0.0,
                normalized_entropy=0.0,
                confidence=0.0,
                threshold=0.0,
                is_abstaining=False,
                in_coverage=True,
                evidence_state="present_ungrounded",
                sampling_valid=False,
            )
        )
    return records


def _write(output_dir: Path, name: str, model: str, track: str, condition: int,
           records: list[SampleRecord]) -> dict:
    metrics: dict = {
        "accuracy": compute_accuracy(records),
        "open_accuracy": compute_open_accuracy(records),
        "closed_accuracy": compute_closed_accuracy(records),
    }
    for key in NOT_COMPUTABLE_METRICS:
        metrics[key] = None
    out = {
        "experiment": "FS-005 Phase 5 Stage 3: answer accuracy of FS-004 baselines",
        "model": model,
        "track": track,
        "condition": condition,
        "dataset": "SLAKE",
        "split": "test",
        "n_samples": len(records),
        "metrics": metrics,
        "not_computable": {
            "metrics": NOT_COMPUTABLE_METRICS,
            "reason": NOT_COMPUTABLE_REASON,
        },
    }
    with open(output_dir / name, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    """Report answer accuracy for the two FS-004 baselines."""
    print("FS-005 Phase 5 Stage 3: answer accuracy of FS-004 baselines")
    print("=" * 60)

    runs_dir = PROJECT_ROOT / "experiments" / "runs"
    output_dir = runs_dir / "fs005_phase5" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The track/condition labels describe how each baseline was actually run.
    medgemma = _load_records(runs_dir / "medgemma_baseline", "track_b_oracle_slake", 1)
    qwen = _load_records(runs_dir / "slake_baseline_ollama", "track_c_answer_only", 1)

    for label, fname, model, track, recs in (
        ("MedGemma-4B", "medgemma_metrics.json", "MedGemma-4B", "track_b_oracle_slake", medgemma),
        ("Qwen2.5-VL-3B", "qwen_metrics.json", "Qwen2.5-VL-3B", "track_c_answer_only", qwen),
    ):
        out = _write(output_dir, fname, model, track, 1, recs)
        m = out["metrics"]
        print(f"\n--- {label} ---")
        print(f"  Samples: {out['n_samples']}")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  OPEN accuracy: {m['open_accuracy']}")
        print(f"  CLOSED accuracy: {m['closed_accuracy']}")
        print("  Grounding / uncertainty metrics: not computable from these artifacts")

    print(f"\nEvaluation artifacts saved to: {output_dir}")
    print()
    print("NOTE: Computed from FS-004 baseline artifacts (existing model "
          "inferences). No new model inference was performed. USR, pointing "
          "accuracy, agreement, entropy, ECE, AUROC and coverage need the "
          "integrated pipeline run with real models.")


if __name__ == "__main__":
    main()
