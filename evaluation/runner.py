"""Evaluation experiment runner and artifact generation for FS-005 Phase 3.

Orchestrates metric computation and writes deterministic experiment artifacts
(JSON, CSV, Markdown) consistent with project conventions (see
scripts/compare_slake_baselines.py for the FS-004 artifact format pattern).

All functions are pure: they take per-sample ``SampleRecord`` objects
(produced by Phases 1-2) and compute aggregate metrics + write artifacts.
No model inference, no VLM calls, no ground-truth feedback into runtime.

Traceability:
  - Feature Specification: FS-005 (Phase 3), sections 4.8, 6
  - ADR 0004: Negative/Absence Claims (Option 3 accepted)
  - ADR 0005: Confidence Gate Formulation
"""

import csv
import json
from pathlib import Path

from evaluation.metrics import (
    AggregateMetrics,
    SampleRecord,
    compute_aggregate_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Per-sample row serialization
# ---------------------------------------------------------------------------


def _sample_to_dict(record: SampleRecord) -> dict:
    """Serialize a SampleRecord to a flat dict for CSV/JSON output."""
    return {
        "sample_id": record.sample_id,
        "dataset_name": record.dataset_name,
        "split": record.split,
        "track": record.track,
        "condition": record.condition,
        "ground_truth_answer": record.ground_truth_answer,
        "model_answer": record.model_answer,
        "correct": record.correct,
        "question_type": record.question_type,
        "has_gt_regions": record.has_gt_regions,
        "pointing_correct": record.pointing_correct,
        "n_claims": record.n_claims,
        "n_positive_claims": record.n_positive_claims,
        "n_supported": record.n_supported,
        "n_unsupported": record.n_unsupported,
        "n_unverifiable": record.n_unverifiable,
        "n_absence": record.n_absence,
        "n_uncertain": record.n_uncertain,
        "agreement": record.agreement,
        "normalized_entropy": record.normalized_entropy,
        "confidence": record.confidence,
        "threshold": record.threshold,
        "is_abstaining": record.is_abstaining,
        "in_coverage": record.in_coverage,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_metrics_json(
    out_dir: Path,
    metrics: AggregateMetrics,
    config: dict,
    experiment_name: str = "FS-005 Phase 3 evaluation",
) -> None:
    """Write experiment_metrics.json with aggregate metrics and config."""
    payload = {
        "experiment": experiment_name,
        "n_samples": metrics.n_samples,
        "n_positive_claims_total": metrics.n_positive_claims_total,
        "metrics": {
            "accuracy": metrics.accuracy,
            "open_accuracy": metrics.open_accuracy,
            "closed_accuracy": metrics.closed_accuracy,
            "usr_raw": metrics.usr_raw,
            "usr_filtered": metrics.usr_filtered,
            "pointing_accuracy": metrics.pointing_accuracy,
            "sampling_agreement": metrics.sampling_agreement,
            "normalized_entropy": metrics.normalized_entropy,
            "ece": metrics.ece,
            "selective_accuracy": metrics.selective_accuracy,
            "coverage": metrics.coverage,
            "auroc": metrics.auroc,
        },
        "accuracy_vs_coverage": metrics.accuracy_vs_coverage,
        "config": config,
        "exclusions": metrics.exclusions,
    }
    with (out_dir / "experiment_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def write_summary_csv(
    out_dir: Path, metrics: AggregateMetrics
) -> None:
    """Write experiment_summary.csv — one row per metric."""
    rows = [
        ("n_samples", metrics.n_samples),
        ("n_positive_claims_total", metrics.n_positive_claims_total),
        ("accuracy", metrics.accuracy),
        ("open_accuracy", metrics.open_accuracy),
        ("closed_accuracy", metrics.closed_accuracy),
        ("usr_raw", metrics.usr_raw),
        ("usr_filtered", metrics.usr_filtered),
        ("pointing_accuracy", metrics.pointing_accuracy),
        ("sampling_agreement", metrics.sampling_agreement),
        ("normalized_entropy", metrics.normalized_entropy),
        ("ece", metrics.ece),
        ("selective_accuracy", metrics.selective_accuracy),
        ("coverage", metrics.coverage),
        ("auroc", metrics.auroc),
    ]
    with (out_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_per_sample_csv(out_dir: Path, records: list[SampleRecord]) -> None:
    """Write per_sample_results.csv — one row per sample."""
    fieldnames = [
        "sample_id", "dataset_name", "split", "track", "condition",
        "ground_truth_answer", "model_answer", "correct", "question_type",
        "has_gt_regions", "pointing_correct", "n_claims",
        "n_positive_claims", "n_supported", "n_unsupported",
        "n_unverifiable", "n_absence", "n_uncertain",
        "agreement", "normalized_entropy", "confidence",
        "threshold", "is_abstaining", "in_coverage",
    ]
    with (out_dir / "per_sample_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(_sample_to_dict(record))


def write_accuracy_vs_coverage_csv(
    out_dir: Path, metrics: AggregateMetrics
) -> None:
    """Write accuracy_vs_coverage.csv — (tau, coverage, selective_accuracy)."""
    fieldnames = ["tau", "coverage", "selective_accuracy"]
    with (out_dir / "accuracy_vs_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for point in metrics.accuracy_vs_coverage:
            writer.writerow(point)


def write_report(
    out_dir: Path,
    metrics: AggregateMetrics,
    config: dict,
    experiment_name: str = "FS-005 Phase 3 evaluation",
) -> None:
    """Write a human-readable Markdown evaluation report."""
    lines: list[str] = []
    lines.append(f"# {experiment_name}")
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append("")
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 2. Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| n_samples | {metrics.n_samples} |")
    lines.append(
        f"| n_positive_claims_total | {metrics.n_positive_claims_total} |"
    )
    lines.append(f"| accuracy | {metrics.accuracy} |")
    lines.append(f"| open_accuracy | {metrics.open_accuracy} |")
    lines.append(f"| closed_accuracy | {metrics.closed_accuracy} |")
    lines.append(f"| usr_raw | {metrics.usr_raw} |")
    lines.append(f"| usr_filtered | {metrics.usr_filtered} |")
    lines.append(f"| pointing_accuracy | {metrics.pointing_accuracy} |")
    lines.append(f"| sampling_agreement | {metrics.sampling_agreement} |")
    lines.append(f"| normalized_entropy | {metrics.normalized_entropy} |")
    lines.append(f"| ece | {metrics.ece} |")
    lines.append(f"| selective_accuracy | {metrics.selective_accuracy} |")
    lines.append(f"| coverage | {metrics.coverage} |")
    lines.append(f"| auroc | {metrics.auroc} |")
    lines.append("")
    lines.append("## 3. Exclusions")
    lines.append("")
    if metrics.exclusions:
        for exclusion in metrics.exclusions:
            lines.append(f"- {exclusion}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## 4. Accuracy-vs-Coverage Curve (first 5 and last 5 points)")
    lines.append("")
    lines.append("| tau | coverage | selective_accuracy |")
    lines.append("|---|---:|---:|")
    curve = metrics.accuracy_vs_coverage
    display_points = curve[:5] + curve[-5:] if len(curve) > 10 else curve
    for point in display_points:
        lines.append(
            f"| {point['tau']} | {point['coverage']:.4f} | "
            f"{point['selective_accuracy']:.4f} |"
        )
    if len(curve) > 10:
        lines.append(f"| ... ({len(curve) - 10} more points) | | |")
    lines.append("")
    lines.append("## 5. Notes")
    lines.append("")
    lines.append(
        "- USR metrics computed over positive claims only (absence and uncertain "
        "claims excluded per ADR 0004 Option 3)."
    )
    lines.append(
        "- Pointing accuracy denominator excludes samples with no ground-truth "
        "regions."
    )
    lines.append(
        "- ECE, selective accuracy, coverage, and AUROC computed over accepted "
        "(in-coverage) samples only."
    )
    lines.append(
        "- Abstained samples counted as incorrect in overall accuracy but "
        "excluded from coverage/selective accuracy."
    )
    lines.append(
        "- No model inference was run in this phase; metrics computed from "
        "per-sample records produced by Phases 1-2."
    )

    (out_dir / "comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# EvaluationRunner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Orchestrates Phase 3 metric computation and artifact generation.

    The runner takes per-sample ``SampleRecord`` objects (produced by Phases
    1-2 on actual data, or synthetic fixtures for testing) and computes all
    aggregate metrics, then writes experiment artifacts.

    No model inference, no VLM calls, no ground-truth feedback into runtime.
    """

    def __init__(
        self,
        out_dir: Path | str,
        experiment_name: str = "FS-005 Phase 3 evaluation",
        config: dict | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            out_dir: Output directory for artifacts (created if needed).
            experiment_name: Human-readable experiment name.
            config: Configuration dict (weights, threshold, sampling params,
                etc.) recorded in artifacts for reproducibility.
        """
        self.out_dir = Path(out_dir)
        self.experiment_name = experiment_name
        self.config = config or {}
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        records: list[SampleRecord],
        exclusions: list[str] | None = None,
    ) -> AggregateMetrics:
        """Compute all metrics and write artifacts.

        Args:
            records: Per-sample evaluation records.
            exclusions: Human-readable exclusion notes.

        Returns:
            AggregateMetrics with all computed metrics.
        """
        metrics = compute_aggregate_metrics(records, exclusions=exclusions)

        write_metrics_json(self.out_dir, metrics, self.config, self.experiment_name)
        write_summary_csv(self.out_dir, metrics)
        write_per_sample_csv(self.out_dir, records)
        write_accuracy_vs_coverage_csv(self.out_dir, metrics)
        write_report(self.out_dir, metrics, self.config, self.experiment_name)

        return metrics
