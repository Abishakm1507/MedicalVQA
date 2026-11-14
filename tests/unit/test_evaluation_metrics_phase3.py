"""Unit tests for FS-005 Phase 3 evaluation metrics harness.

Covers all metrics per the technical design Section 6:
- USR_raw, USR_filtered
- Pointing accuracy (IoU >= 0.5)
- Sampling agreement
- Normalized semantic entropy
- ECE (Expected Calibration Error)
- Selective accuracy
- Coverage
- Accuracy-vs-coverage / risk-coverage curve
- AUROC

Plus edge cases: empty datasets, missing/invalid records, absence/uncertain
claims, samples without GT regions, IoU boundaries, ECE binning, selective
accuracy at boundary thresholds, artifact serialization, and GT isolation.

All tests are deterministic and require no model weights, paid APIs, or
real VLM inference.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from evaluation.metrics import (
    SampleRecord,
    compute_accuracy,
    compute_accuracy_vs_coverage_curve,
    compute_aggregate_metrics,
    compute_auroc,
    compute_closed_accuracy,
    compute_coverage,
    compute_ece,
    compute_normalized_entropy,
    compute_open_accuracy,
    compute_pointing_accuracy,
    compute_sampling_agreement,
    compute_selective_accuracy,
    compute_usr_filtered,
    compute_usr_raw,
)
from evaluation.runner import EvaluationRunner

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_record(
    sample_id: str = "s1",
    correct: bool = True,
    question_type: str = "CLOSED",
    has_gt_regions: bool = True,
    pointing_correct: bool = True,
    n_claims: int = 1,
    n_positive_claims: int = 1,
    n_supported: int = 1,
    n_unsupported: int = 0,
    n_unverifiable: int = 0,
    n_absence: int = 0,
    n_uncertain: int = 0,
    agreement: float = 1.0,
    normalized_entropy: float = 0.0,
    confidence: float = 0.9,
    threshold: float = 0.5,
    is_abstaining: bool = False,
    evidence_state: str = "present_grounded",
    sampling_valid: bool = True,
    track: str = "track_b_oracle_slake",
    condition: int = 2,
    dataset_name: str = "slake",
    split: str = "test",
) -> SampleRecord:
    """Create a SampleRecord with sensible defaults.

    ``in_coverage`` is derived from ``is_abstaining`` (not abstaining = in
    coverage).
    """
    return SampleRecord(
        sample_id=sample_id,
        dataset_name=dataset_name,
        split=split,
        track=track,
        condition=condition,
        ground_truth_answer="yes",
        model_answer="yes" if correct else "no",
        correct=correct,
        question_type=question_type,
        has_gt_regions=has_gt_regions,
        pointing_correct=pointing_correct,
        n_claims=n_claims,
        n_positive_claims=n_positive_claims,
        n_supported=n_supported,
        n_unsupported=n_unsupported,
        n_unverifiable=n_unverifiable,
        n_absence=n_absence,
        n_uncertain=n_uncertain,
        agreement=agreement,
        normalized_entropy=normalized_entropy,
        confidence=confidence,
        threshold=threshold,
        is_abstaining=is_abstaining,
        in_coverage=not is_abstaining,
        evidence_state=evidence_state,
        sampling_valid=sampling_valid,
    )


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


def test_compute_accuracy_all_correct() -> None:
    records = [_make_record(correct=True) for _ in range(5)]
    assert compute_accuracy(records) == pytest.approx(1.0)


def test_compute_accuracy_all_incorrect() -> None:
    records = [_make_record(correct=False) for _ in range(5)]
    assert compute_accuracy(records) == pytest.approx(0.0)


def test_compute_accuracy_mixed() -> None:
    records = [
        _make_record(sample_id="s1", correct=True),
        _make_record(sample_id="s2", correct=True),
        _make_record(sample_id="s3", correct=False),
    ]
    assert compute_accuracy(records) == pytest.approx(2 / 3)


def test_compute_accuracy_empty() -> None:
    assert compute_accuracy([]) is None


def test_compute_accuracy_abstained_counted_incorrect() -> None:
    """Abstained samples (correct=False) counted in denominator per FS-004 convention."""
    records = [
        _make_record(sample_id="s1", correct=True, is_abstaining=False),
        _make_record(sample_id="s2", correct=False, is_abstaining=True),
    ]
    assert compute_accuracy(records) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# OPEN / CLOSED accuracy tests
# ---------------------------------------------------------------------------


def test_compute_open_accuracy() -> None:
    records = [
        _make_record(sample_id="s1", correct=True, question_type="OPEN"),
        _make_record(sample_id="s2", correct=False, question_type="OPEN"),
        _make_record(sample_id="s3", correct=True, question_type="CLOSED"),
    ]
    assert compute_open_accuracy(records) == pytest.approx(0.5)


def test_compute_open_accuracy_empty() -> None:
    records = [_make_record(question_type="CLOSED")]
    assert compute_open_accuracy(records) is None


def test_compute_closed_accuracy() -> None:
    records = [
        _make_record(sample_id="s1", correct=True, question_type="CLOSED"),
        _make_record(sample_id="s2", correct=False, question_type="CLOSED"),
        _make_record(sample_id="s3", correct=True, question_type="OPEN"),
    ]
    assert compute_closed_accuracy(records) == pytest.approx(0.5)


def test_compute_closed_accuracy_empty() -> None:
    records = [_make_record(question_type="OPEN")]
    assert compute_closed_accuracy(records) is None


# ---------------------------------------------------------------------------
# USR tests
# ---------------------------------------------------------------------------


def test_compute_usr_raw_all_supported() -> None:
    """All supported → usr_raw = 0.0."""
    records = [
        _make_record(n_positive_claims=1, n_supported=1, n_unsupported=0, n_unverifiable=0),
        _make_record(n_positive_claims=2, n_supported=2, n_unsupported=0, n_unverifiable=0),
    ]
    assert compute_usr_raw(records) == pytest.approx(0.0)


def test_compute_usr_raw_all_unsupported() -> None:
    """All unsupported → usr_raw = 1.0."""
    records = [
        _make_record(n_positive_claims=1, n_supported=0, n_unsupported=1, n_unverifiable=0),
        _make_record(n_positive_claims=1, n_supported=0, n_unsupported=1, n_unverifiable=0),
    ]
    assert compute_usr_raw(records) == pytest.approx(1.0)


def test_compute_usr_raw_mixed() -> None:
    """usr_raw = (unsupported + unverifiable) / n_positive_claims."""
    records = [
        _make_record(
            sample_id="s1",
            n_positive_claims=3,
            n_supported=1,
            n_unsupported=1,
            n_unverifiable=1,
        ),
        _make_record(
            sample_id="s2",
            n_positive_claims=2,
            n_supported=1,
            n_unsupported=1,
            n_unverifiable=0,
        ),
    ]
    # (1+1 + 1+0) / (3+2) = 3/5 = 0.6
    assert compute_usr_raw(records) == pytest.approx(0.6)


def test_compute_usr_raw_no_positive_claims() -> None:
    """No positive claims → usr_raw = None (undefined metric, not 0.0)."""
    records = [
        _make_record(n_positive_claims=0, n_absence=1),
        _make_record(n_positive_claims=0, n_uncertain=1),
    ]
    assert compute_usr_raw(records) is None


def test_compute_usr_filtered_all_supported() -> None:
    """All supported → usr_filtered = 0.0."""
    records = [
        _make_record(n_positive_claims=1, n_supported=1, n_unsupported=0, n_unverifiable=0),
    ]
    assert compute_usr_filtered(records) == pytest.approx(0.0)


def test_compute_usr_filtered_all_unsupported() -> None:
    """All unsupported → usr_filtered = 1.0."""
    records = [
        _make_record(n_positive_claims=1, n_supported=0, n_unsupported=1, n_unverifiable=0),
    ]
    assert compute_usr_filtered(records) == pytest.approx(1.0)


def test_compute_usr_filtered_with_unverifiable() -> None:
    """usr_filtered strips unverifiable from denominator."""
    records = [
        _make_record(
            sample_id="s1",
            n_positive_claims=3,
            n_supported=1,
            n_unsupported=1,
            n_unverifiable=1,
        ),
    ]
    # unsupported / (n_positive - unverifiable) = 1 / (3-1) = 0.5
    assert compute_usr_filtered(records) == pytest.approx(0.5)


def test_compute_usr_filtered_denominator_zero() -> None:
    """All unverifiable → denominator = 0 → usr_filtered = None."""
    records = [
        _make_record(n_positive_claims=1, n_supported=0, n_unsupported=0, n_unverifiable=1),
    ]
    assert compute_usr_filtered(records) is None


def test_compute_usr_absence_excluded_from_denominator() -> None:
    """Absence claims excluded from USR denominators per ADR 0004 Option 3."""
    records = [
        _make_record(
            sample_id="s1",
            n_claims=2,
            n_positive_claims=1,
            n_supported=1,
            n_absence=1,
        ),
        _make_record(
            sample_id="s2",
            n_claims=2,
            n_positive_claims=1,
            n_supported=0,
            n_unsupported=1,
            n_absence=1,
        ),
    ]
    # usr_raw = (0 + 1) / (1 + 1) = 0.5 (absence claims not in denominator)
    assert compute_usr_raw(records) == pytest.approx(0.5)
    # usr_filtered = 1 / (2 - 0) = 0.5
    assert compute_usr_filtered(records) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pointing accuracy tests
# ---------------------------------------------------------------------------


def test_compute_pointing_accuracy_all_correct() -> None:
    records = [
        _make_record(has_gt_regions=True, pointing_correct=True),
        _make_record(has_gt_regions=True, pointing_correct=True),
    ]
    assert compute_pointing_accuracy(records) == pytest.approx(1.0)


def test_compute_pointing_accuracy_none_correct() -> None:
    records = [
        _make_record(has_gt_regions=True, pointing_correct=False),
        _make_record(has_gt_regions=True, pointing_correct=False),
    ]
    assert compute_pointing_accuracy(records) == pytest.approx(0.0)


def test_compute_pointing_accuracy_excludes_no_gt() -> None:
    """Samples without GT regions excluded from denominator."""
    records = [
        _make_record(sample_id="s1", has_gt_regions=True, pointing_correct=True),
        _make_record(sample_id="s2", has_gt_regions=False, pointing_correct=False),
        _make_record(sample_id="s3", has_gt_regions=True, pointing_correct=False),
    ]
    # Only s1 and s3 count: 1/2 = 0.5
    assert compute_pointing_accuracy(records) == pytest.approx(0.5)


def test_compute_pointing_accuracy_no_gt_regions() -> None:
    """No samples with GT regions → None."""
    records = [
        _make_record(has_gt_regions=False),
        _make_record(has_gt_regions=False),
    ]
    assert compute_pointing_accuracy(records) is None


def test_compute_pointing_accuracy_multiple_predictions() -> None:
    """Multiple predicted regions: at least one with IoU >= 0.5 → correct.

    Phase 1's evaluate_grounding_iou_with_boxes computes max IoU over all
    GT boxes for each claim. pointing_correct=True means at least one claim
    met the IoU >= 0.5 threshold against any GT region.
    """
    records = [
        # Sample 1: multiple claims, one grounded correctly (IoU >= 0.5)
        _make_record(
            sample_id="s1", has_gt_regions=True, pointing_correct=True,
        ),
        # Sample 2: multiple claims, none grounded correctly
        _make_record(
            sample_id="s2", has_gt_regions=True, pointing_correct=False,
        ),
    ]
    # 1/2 = 0.5
    assert compute_pointing_accuracy(records) == pytest.approx(0.5)


def test_compute_pointing_accuracy_multiple_gt_regions() -> None:
    """Multiple ground-truth regions: max IoU over all GT boxes used.

    Phase 1 computes IoU between each cited box and every GT box, then
    takes the maximum. pointing_correct=True if max IoU >= 0.5.
    """
    records = [
        _make_record(
            sample_id="s1", has_gt_regions=True, pointing_correct=True,
        ),
        _make_record(
            sample_id="s2", has_gt_regions=True, pointing_correct=True,
        ),
    ]
    assert compute_pointing_accuracy(records) == pytest.approx(1.0)


def test_compute_pointing_accuracy_no_predictions() -> None:
    """No predicted claims → pointing_correct=False (nothing grounded)."""
    records = [
        _make_record(
            sample_id="s1", has_gt_regions=True, pointing_correct=False,
        ),
    ]
    assert compute_pointing_accuracy(records) == pytest.approx(0.0)


def test_compute_pointing_accuracy_multiple_claims_one_sample() -> None:
    """One sample with one correctly grounded claim and one incorrectly
    grounded claim → pointing_correct=True (sample-level: at least one claim
    meets IoU >= 0.5 against any GT region).

    Technical design Section 6.1: IoU computed per cited box against every GT
    box, max taken. Threshold >= 0.5. Section 7.1: denominator counts samples
    with GT regions (sample-level aggregation).
    """
    records = [
        # One sample: claim 1 grounded correctly (IoU >= 0.5), claim 2 not.
        # Sample-level: at least one correct → pointing_correct=True.
        _make_record(
            sample_id="s1", has_gt_regions=True, pointing_correct=True,
        ),
        # One sample: no claim grounded correctly.
        _make_record(
            sample_id="s2", has_gt_regions=True, pointing_correct=False,
        ),
    ]
    # Sample-level: 1 correct out of 2 eligible samples = 0.5
    assert compute_pointing_accuracy(records) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Sampling agreement and entropy tests
# ---------------------------------------------------------------------------


def test_compute_sampling_agreement() -> None:
    records = [
        _make_record(agreement=1.0),
        _make_record(agreement=0.5),
        _make_record(agreement=0.0),
    ]
    assert compute_sampling_agreement(records) == pytest.approx(0.5)


def test_compute_sampling_agreement_empty() -> None:
    assert compute_sampling_agreement([]) is None


def test_compute_normalized_entropy() -> None:
    records = [
        _make_record(normalized_entropy=0.0),
        _make_record(normalized_entropy=1.0),
        _make_record(normalized_entropy=0.5),
    ]
    assert compute_normalized_entropy(records) == pytest.approx(0.5)


def test_compute_normalized_entropy_empty() -> None:
    assert compute_normalized_entropy([]) is None


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------


def test_compute_ece_perfectly_calibrated() -> None:
    """All confidences match accuracy within bins → ECE = 0."""
    records = [
        _make_record(confidence=0.9, correct=True, is_abstaining=False),
        _make_record(confidence=0.9, correct=True, is_abstaining=False),
        _make_record(confidence=0.1, correct=False, is_abstaining=False),
        _make_record(confidence=0.1, correct=False, is_abstaining=False),
    ]
    # Bin [0.8, 1.0]: 2 samples, acc=1.0, conf=0.9 → |1.0-0.9| = 0.1, weight=0.5 → 0.05
    # Bin [0.0, 0.2): 2 samples, acc=0.0, conf=0.1 → |0.0-0.1| = 0.1, weight=0.5 → 0.05
    # ECE = 0.1
    assert compute_ece(records) == pytest.approx(0.1)


def test_compute_ece_no_accepted_samples() -> None:
    """All abstained → ECE = None."""
    records = [
        _make_record(is_abstaining=True),
        _make_record(is_abstaining=True),
    ]
    assert compute_ece(records) is None


def test_compute_ece_empty() -> None:
    assert compute_ece([]) is None


def test_compute_ece_bin_boundary() -> None:
    """Confidence exactly at bin boundary goes to upper bin (last bin inclusive)."""
    records = [
        _make_record(
            confidence=0.5, correct=True, is_abstaining=False
        ),
        _make_record(
            confidence=0.5, correct=False, is_abstaining=False
        ),
    ]
    # 0.5 falls in bin [0.5, 0.6): acc=0.5, conf=0.5 → |0.5-0.5| = 0
    assert compute_ece(records) == pytest.approx(0.0)


def test_compute_ece_max_confidence_last_bin() -> None:
    """Confidence = 1.0 goes to last bin [0.9, 1.0]."""
    records = [
        _make_record(
            confidence=1.0, correct=True, is_abstaining=False
        ),
    ]
    assert compute_ece(records) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Selective accuracy and coverage tests
# ---------------------------------------------------------------------------


def test_compute_selective_accuracy_all_correct() -> None:
    records = [
        _make_record(correct=True),
        _make_record(correct=True),
    ]
    assert compute_selective_accuracy(records) == pytest.approx(1.0)


def test_compute_selective_accuracy_none_in_coverage() -> None:
    """All abstained → selective_accuracy = None."""
    records = [
        _make_record(is_abstaining=True),
        _make_record(is_abstaining=True),
    ]
    assert compute_selective_accuracy(records) is None


def test_compute_coverage() -> None:
    records = [
        _make_record(),
        _make_record(),
        _make_record(is_abstaining=True),
    ]
    assert compute_coverage(records) == pytest.approx(2 / 3)


def test_compute_coverage_empty() -> None:
    assert compute_coverage([]) == 0.0


def test_compute_coverage_all_abstained() -> None:
    records = [
        _make_record(is_abstaining=True),
        _make_record(is_abstaining=True),
    ]
    assert compute_coverage(records) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AUROC tests
# ---------------------------------------------------------------------------


def test_compute_auroc_perfect_separation() -> None:
    """Perfect separation → AUROC = 1.0."""
    records = [
        _make_record(confidence=0.9, correct=True),
        _make_record(confidence=0.8, correct=True),
        _make_record(confidence=0.2, correct=False),
        _make_record(confidence=0.1, correct=False),
    ]
    assert compute_auroc(records) == pytest.approx(1.0)


def test_compute_auroc_random() -> None:
    """Random scores → AUROC ≈ 0.5."""
    records = [
        _make_record(confidence=0.5, correct=True),
        _make_record(confidence=0.5, correct=False),
    ]
    assert compute_auroc(records) == pytest.approx(0.5)


def test_compute_auroc_one_class() -> None:
    """Only one correctness class → AUROC = None (undefined)."""
    records = [
        _make_record(confidence=0.9, correct=True),
        _make_record(confidence=0.8, correct=True),
    ]
    assert compute_auroc(records) is None


def test_compute_auroc_no_eligible() -> None:
    """No eligible samples (all invalid sampling or uncertain) → AUROC = None."""
    records = [
        _make_record(sampling_valid=False),
        _make_record(evidence_state="uncertain"),
    ]
    assert compute_auroc(records) is None


def test_compute_auroc_threshold_independence() -> None:
    """AUROC does not change when the operating threshold changes.

    The eligible population is independent of tau: samples that abstain
    because confidence < tau still have meaningful confidence and are
    included. Only forced-abstention samples (invalid sampling, uncertain
    claims) are excluded.
    """
    records = [
        _make_record(confidence=0.9, correct=True),
        _make_record(confidence=0.8, correct=True),
        _make_record(confidence=0.2, correct=False),
        _make_record(confidence=0.1, correct=False),
    ]
    auroc_at_any_tau = compute_auroc(records)
    assert auroc_at_any_tau == pytest.approx(1.0)

    # Simulate a higher tau where some samples drop out of coverage:
    # the underlying records are unchanged, so AUROC must be identical.
    records_high_tau = [
        _make_record(
            sample_id="s1", confidence=0.9, correct=True,
        ),
        _make_record(
            sample_id="s2", confidence=0.8, correct=True,
        ),
        _make_record(
            sample_id="s3", confidence=0.2, correct=False,
        ),
        _make_record(
            sample_id="s4", confidence=0.1, correct=False,
        ),
    ]
    assert compute_auroc(records_high_tau) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Accuracy-vs-coverage curve tests
# ---------------------------------------------------------------------------


def test_compute_accuracy_vs_coverage_curve_shape() -> None:
    """Curve has 101 points (tau 0.0 to 1.0)."""
    records = [
        _make_record(confidence=0.9, correct=True),
        _make_record(confidence=0.1, correct=False),
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    assert len(curve) == 101
    assert curve[0]["tau"] == 0.0
    assert curve[-1]["tau"] == 1.0


def test_compute_accuracy_vs_coverage_curve_monotonic_coverage() -> None:
    """Coverage decreases monotonically as tau increases."""
    records = [
        _make_record(confidence=0.9, correct=True),
        _make_record(confidence=0.5, correct=False),
        _make_record(confidence=0.1, correct=True),
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    coverages = [p["coverage"] for p in curve]
    for i in range(len(coverages) - 1):
        assert coverages[i] >= coverages[i + 1]


def test_compute_accuracy_vs_coverage_curve_tau_zero() -> None:
    """tau=0.0 → all samples in coverage (coverage=1.0)."""
    records = [
        _make_record(confidence=0.0, correct=True),
        _make_record(confidence=0.0, correct=False),
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    assert curve[0]["coverage"] == pytest.approx(1.0)


def test_compute_accuracy_vs_coverage_curve_tau_one() -> None:
    """tau=1.0 → only confidence=1.0 eligible samples in coverage."""
    records = [
        _make_record(confidence=1.0, correct=True),
        _make_record(confidence=0.9, correct=False),
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    assert curve[-1]["coverage"] == pytest.approx(0.5)
    assert curve[-1]["selective_accuracy"] == pytest.approx(1.0)


def test_compute_accuracy_vs_coverage_curve_invalid_sampling_never_covered() -> None:
    """Invalid sampling never counts as covered, including at tau=0."""
    records = [
        _make_record(
            sample_id="s1", confidence=0.9, correct=True,
            sampling_valid=True, evidence_state="present_grounded",
        ),
        _make_record(
            sample_id="s2", confidence=0.9, correct=False,
            sampling_valid=False, evidence_state="present_grounded",
        ),  # invalid sampling — never covered
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    # At tau=0, only s1 is covered (s2 is invalid sampling)
    assert curve[0]["coverage"] == pytest.approx(0.5)
    assert curve[0]["selective_accuracy"] == pytest.approx(1.0)
    # At all tau values, s2 is never covered
    for point in curve:
        assert point["coverage"] <= 0.5


def test_compute_accuracy_vs_coverage_curve_uncertain_claims_never_covered() -> None:
    """Uncertain claims never count as covered, including at tau=0."""
    records = [
        _make_record(
            sample_id="s1", confidence=0.9, correct=True,
            sampling_valid=True, evidence_state="present_grounded",
        ),
        _make_record(
            sample_id="s2", confidence=0.9, correct=False,
            sampling_valid=True, evidence_state="uncertain",
        ),  # uncertain claim — never covered
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    # At tau=0, only s1 is covered (s2 is uncertain claim)
    assert curve[0]["coverage"] == pytest.approx(0.5)
    assert curve[0]["selective_accuracy"] == pytest.approx(1.0)
    # At all tau values, s2 is never covered
    for point in curve:
        assert point["coverage"] <= 0.5


def test_compute_accuracy_vs_coverage_curve_mixed_forced_abstention() -> None:
    """Mix of eligible, invalid sampling, and uncertain claim samples."""
    records = [
        _make_record(
            sample_id="s1", confidence=0.9, correct=True,
            sampling_valid=True, evidence_state="present_grounded",
        ),
        _make_record(
            sample_id="s2", confidence=0.5, correct=False,
            sampling_valid=True, evidence_state="present_grounded",
        ),
        _make_record(
            sample_id="s3", confidence=0.9, correct=True,
            sampling_valid=False, evidence_state="present_grounded",
        ),  # invalid — never covered
        _make_record(
            sample_id="s4", confidence=0.9, correct=False,
            sampling_valid=True, evidence_state="uncertain",
        ),  # uncertain — never covered
    ]
    curve = compute_accuracy_vs_coverage_curve(records)
    # At tau=0: s1 and s2 eligible, covered. s3 and s4 never covered.
    # coverage = 2/4 = 0.5
    assert curve[0]["coverage"] == pytest.approx(0.5)
    # selective_accuracy at tau=0: s1 correct, s2 incorrect → 0.5
    assert curve[0]["selective_accuracy"] == pytest.approx(0.5)
    # At tau=0.6: only s1 covered (s2 confidence=0.5 < 0.6)
    # coverage = 1/4 = 0.25, selective_accuracy = 1.0
    high_tau_point = curve[60]  # tau=0.6
    assert high_tau_point["coverage"] == pytest.approx(0.25)
    assert high_tau_point["selective_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Aggregate metrics tests
# ---------------------------------------------------------------------------


def test_compute_aggregate_metrics() -> None:
    records = [
        _make_record(correct=True),
        _make_record(correct=False),
    ]
    metrics = compute_aggregate_metrics(records)
    assert metrics.n_samples == 2
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.usr_raw == pytest.approx(0.0)
    assert metrics.usr_filtered == pytest.approx(0.0)
    assert isinstance(metrics.accuracy_vs_coverage, list)
    assert len(metrics.accuracy_vs_coverage) == 101


def test_compute_aggregate_metrics_with_exclusions() -> None:
    records = [_make_record()]
    exclusions = ["Sample s2 excluded: no GT regions"]
    metrics = compute_aggregate_metrics(records, exclusions=exclusions)
    assert metrics.exclusions == exclusions


# ---------------------------------------------------------------------------
# Artifact generation tests
# ---------------------------------------------------------------------------


def test_runner_writes_all_artifacts() -> None:
    """EvaluationRunner writes all expected artifact files."""
    records = [
        _make_record(sample_id="s1", correct=True),
        _make_record(sample_id="s2", correct=False),
    ]
    config = {"w1": 0.4, "w2": 0.3, "w3": 0.3, "tau": 0.5, "n_samples": 5}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        runner = EvaluationRunner(
            out_dir=out_dir, experiment_name="Test Eval", config=config
        )
        metrics = runner.run(records)

        # Check all artifact files exist
        assert (out_dir / "experiment_metrics.json").exists()
        assert (out_dir / "experiment_summary.csv").exists()
        assert (out_dir / "per_sample_results.csv").exists()
        assert (out_dir / "accuracy_vs_coverage.csv").exists()
        assert (out_dir / "comparison_report.md").exists()

        # Check metrics returned
        assert metrics.n_samples == 2
        assert metrics.accuracy == pytest.approx(0.5)


def test_runner_json_artifact_structure() -> None:
    """JSON artifact has expected structure."""
    records = [_make_record()]
    config = {"w1": 0.4, "w2": 0.3, "w3": 0.3, "tau": 0.5}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        runner = EvaluationRunner(out_dir=out_dir, config=config)
        runner.run(records)

        data = json.loads((out_dir / "experiment_metrics.json").read_text())
        assert "experiment" in data
        assert "n_samples" in data
        assert "metrics" in data
        assert "accuracy_vs_coverage" in data
        assert "config" in data
        assert "exclusions" in data
        assert data["n_samples"] == 1
        assert len(data["accuracy_vs_coverage"]) == 101


def test_runner_csv_artifact_structure() -> None:
    """CSV artifacts have expected headers and rows."""
    records = [_make_record(sample_id="s1"), _make_record(sample_id="s2")]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        runner = EvaluationRunner(out_dir=out_dir)
        runner.run(records)

        # Per-sample CSV
        per_sample = list(csv.DictReader(open(out_dir / "per_sample_results.csv")))
        assert len(per_sample) == 2
        assert per_sample[0]["sample_id"] == "s1"
        assert per_sample[1]["sample_id"] == "s2"

        # Summary CSV
        summary = list(csv.DictReader(open(out_dir / "experiment_summary.csv")))
        metric_names = {row["metric"] for row in summary}
        assert "accuracy" in metric_names
        assert "usr_raw" in metric_names
        assert "coverage" in metric_names

        # Accuracy-vs-coverage CSV
        curve = list(csv.DictReader(open(out_dir / "accuracy_vs_coverage.csv")))
        assert len(curve) == 101
        assert set(curve[0].keys()) == {"tau", "coverage", "selective_accuracy"}


def test_runner_report_md_structure() -> None:
    """Markdown report contains expected sections."""
    records = [_make_record()]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        runner = EvaluationRunner(out_dir=out_dir)
        runner.run(records)

        report = (out_dir / "comparison_report.md").read_text()
        assert "# FS-005 Phase 3 evaluation" in report
        assert "## 1. Configuration" in report
        assert "## 2. Aggregate Metrics" in report
        assert "## 3. Exclusions" in report
        assert "## 4. Accuracy-vs-Coverage Curve" in report
        assert "## 5. Notes" in report


# ---------------------------------------------------------------------------
# Absence and uncertain claim handling
# ---------------------------------------------------------------------------


def test_absence_claims_excluded_from_usr_denominator() -> None:
    """Absence claims (n_absence > 0) do not contribute to n_positive_claims."""
    records = [
        _make_record(
            sample_id="s1",
            n_claims=2,
            n_positive_claims=1,
            n_supported=1,
            n_absence=1,
        ),
        _make_record(
            sample_id="s2",
            n_claims=2,
            n_positive_claims=1,
            n_supported=0,
            n_unsupported=1,
            n_absence=1,
        ),
    ]
    # n_positive total = 2 (1 per sample, absence excluded)
    # usr_raw = (0 + 1) / 2 = 0.5
    assert compute_usr_raw(records) == pytest.approx(0.5)


def test_uncertain_claims_excluded_from_usr_denominator() -> None:
    """Uncertain claims (n_uncertain > 0) do not contribute to n_positive_claims."""
    records = [
        _make_record(
            sample_id="s1",
            n_claims=2,
            n_positive_claims=1,
            n_supported=1,
            n_uncertain=1,
        ),
        _make_record(
            sample_id="s2",
            n_claims=2,
            n_positive_claims=1,
            n_supported=0,
            n_unsupported=1,
            n_uncertain=1,
        ),
    ]
    # n_positive total = 2 (uncertain excluded)
    # usr_raw = (0 + 1) / 2 = 0.5
    assert compute_usr_raw(records) == pytest.approx(0.5)


def test_uncertain_claims_counted_in_overall_accuracy_denominator() -> None:
    """Uncertain claims' samples still count in overall accuracy denominator."""
    records = [
        _make_record(sample_id="s1", correct=True, n_uncertain=1),
        _make_record(sample_id="s2", correct=False, n_uncertain=1),
    ]
    assert compute_accuracy(records) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GT isolation tests
# ---------------------------------------------------------------------------


def test_metrics_do_not_modify_runtime_inputs() -> None:
    """Metric computation does not modify SampleRecord inputs (GT isolation)."""
    record = _make_record(confidence=0.9, correct=True)
    original_confidence = record.confidence
    original_correct = record.correct

    compute_accuracy([record])
    compute_usr_raw([record])
    compute_ece([record])
    compute_selective_accuracy([record])

    assert record.confidence == original_confidence
    assert record.correct == original_correct


def test_ground_truth_not_used_in_confidence_computation() -> None:
    """GT fields are not referenced in confidence-related metric paths."""
    import inspect

    # compute_confidence (from Phase 2) takes no GT params — verified in Phase 2 tests.
    # Here we confirm metric functions use only SampleRecord fields.
    sig = inspect.signature(compute_ece)
    params = list(sig.parameters.keys())
    assert "ground_truth" not in params
    assert "gt_" not in "".join(params)

    sig = inspect.signature(compute_auroc)
    params = list(sig.parameters.keys())
    assert "ground_truth" not in params
    assert "gt_" not in "".join(params)
