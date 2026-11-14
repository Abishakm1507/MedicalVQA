"""Evaluation metrics harness for FS-005 Phase 3.

Computes all required metrics per the FS-005 technical design Section 6:

- USR_raw (Raw Unsupported Claim Rate)
- USR_filtered (Filtered Unsupported Claim Rate)
- Pointing accuracy (IoU >= 0.5 against ground-truth regions)
- Sampling agreement (semantic agreement a)
- Normalized semantic entropy (H_n)
- Expected Calibration Error (ECE, M=10 bins)
- Selective accuracy
- Coverage
- Accuracy-vs-coverage / risk-coverage curve
- AUROC (diagnostic)

All functions are pure: they take per-sample ``SampleRecord`` objects and
return metric values. Ground-truth data is used only in this evaluation layer;
runtime prediction paths (``verification/``, ``confidence_gate.py``) never
access ground truth.

Traceability:
  - Feature Specification: FS-005 (Phase 3), sections 4.8, 6
  - Research Paper: Section III-D, III-E (Claim Verification, Confidence Gate)
  - ADR 0004: Negative/Absence Claims (Option 3 accepted)
  - ADR 0005: Confidence Gate Formulation
"""

from dataclasses import dataclass, field


@dataclass
class SampleRecord:
    """Per-sample evaluation record.

    Contains all information needed to compute FS-005 metrics. Ground-truth
    fields (``ground_truth_answer``, ``has_gt_regions``, ``pointing_correct``,
    ``correct``) are evaluation-only and must not influence runtime prediction
    or abstention decisions.

    USR denominators use positive claims only (``n_positive_claims`` excludes
    absence and uncertain claims per ADR 0004 Option 3).

    Traceability: FS-005 design Section 4.8 (Evaluation metrics and artifact
    generation).
    """

    sample_id: str
    dataset_name: str
    split: str
    track: str  # EvaluationTrack value (e.g. 'track_b_oracle_slake')
    condition: int  # 1, 2, or 3

    # Answer fields (evaluation-only)
    ground_truth_answer: str
    model_answer: str | None  # None if abstained
    correct: bool  # normalized exact match; False if abstained
    question_type: str  # 'OPEN' or 'CLOSED'

    # Region/grounding fields (evaluation-only)
    has_gt_regions: bool
    pointing_correct: bool  # exists claim with IoU >= 0.5 against GT

    # Claim counts (from Phase 1 claim_filter)
    n_claims: int  # total claims emitted
    n_positive_claims: int  # positive claims only (absence/uncertain excluded)
    n_supported: int
    n_unsupported: int
    n_unverifiable: int
    n_absence: int
    n_uncertain: int

    # Uncertainty fields (from Phase 2 confidence_gate)
    agreement: float  # a in [0, 1]
    normalized_entropy: float  # H_n in [0, 1]
    confidence: float  # c in [0, 1]
    threshold: float  # tau
    is_abstaining: bool
    in_coverage: bool  # not is_abstaining

    # Forced-abstention eligibility (from Phase 2 confidence_gate)
    # evidence_state: 'present_grounded' | 'absent' | 'uncertain' | 'present_ungrounded'
    # Uncertain claims are forced to abstain regardless of confidence threshold.
    evidence_state: str  # Phase 2 EvidenceState value
    # sampling_valid: True if sampling produced a usable uncertainty estimate.
    # Invalid sampling is forced to abstain regardless of confidence threshold.
    sampling_valid: bool  # True if sampling produced valid uncertainty estimate


# ---------------------------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(records: list[SampleRecord]) -> float | None:
    """Overall accuracy across all samples.

    Abstained samples (``correct=False``) are counted in the denominator,
    consistent with the FS-004 convention that failed/missing predictions
    are incorrect and retained in the denominator.

    Returns None if there are no samples.
    """
    if not records:
        return None
    return sum(1 for r in records if r.correct) / len(records)


def compute_open_accuracy(records: list[SampleRecord]) -> float | None:
    """Accuracy on OPEN question-type samples only."""
    open_records = [r for r in records if r.question_type == "OPEN"]
    if not open_records:
        return None
    return sum(1 for r in open_records if r.correct) / len(open_records)


def compute_closed_accuracy(records: list[SampleRecord]) -> float | None:
    """Accuracy on CLOSED question-type samples only."""
    closed_records = [r for r in records if r.question_type == "CLOSED"]
    if not closed_records:
        return None
    return sum(1 for r in closed_records if r.correct) / len(closed_records)


# ---------------------------------------------------------------------------
# USR (Unsupported Claim Rate)
# ---------------------------------------------------------------------------


def compute_usr_raw(records: list[SampleRecord]) -> float | None:
    """Raw Unsupported Claim Rate.

    USR_raw = (N_unsupported + N_unverifiable) / N_positive_claims

    Computed over positive claims only — absence and uncertain claims are
    excluded from the denominator per ADR 0004 Option 3.

    Returns None if there are no positive claims (undefined metric — do not
    represent as 0.0).
    """
    n_unsup = sum(r.n_unsupported for r in records)
    n_unver = sum(r.n_unverifiable for r in records)
    n_positive = sum(r.n_positive_claims for r in records)
    if n_positive == 0:
        return None
    return (n_unsup + n_unver) / n_positive


def compute_usr_filtered(records: list[SampleRecord]) -> float | None:
    """Filtered Unsupported Claim Rate.

    USR_filtered = N_unsupported / (N_positive_claims - N_unverifiable)

    Unverifiable claims (bad region_id or R = empty) are stripped from the
    denominator. Absence and uncertain claims are excluded entirely.

    Returns None if the filtered denominator is 0 (undefined metric — do not
    represent as 0.0).
    """
    n_unsup = sum(r.n_unsupported for r in records)
    n_unver = sum(r.n_unverifiable for r in records)
    n_positive = sum(r.n_positive_claims for r in records)
    denominator = n_positive - n_unver
    if denominator == 0:
        return None
    return n_unsup / denominator


# ---------------------------------------------------------------------------
# Pointing accuracy
# ---------------------------------------------------------------------------


def compute_pointing_accuracy(records: list[SampleRecord]) -> float | None:
    """Pointing accuracy (IoU >= 0.5 against ground-truth regions).

    PointingAcc = sum(1[exists claim with IoU >= 0.5]) / sum(1[has GT regions])

    Samples with no ground-truth regions are excluded from the denominator
    (nothing to point to). If a sample has GT regions but no grounded claim
    meets the IoU threshold, it counts as not pointing correctly.

    Returns None if no samples have ground-truth regions.
    """
    gt_records = [r for r in records if r.has_gt_regions]
    if not gt_records:
        return None
    return sum(1 for r in gt_records if r.pointing_correct) / len(gt_records)


# ---------------------------------------------------------------------------
# Sampling agreement and normalized semantic entropy
# ---------------------------------------------------------------------------


def compute_sampling_agreement(records: list[SampleRecord]) -> float | None:
    """Mean semantic agreement (a) across all samples."""
    if not records:
        return None
    return sum(r.agreement for r in records) / len(records)


def compute_normalized_entropy(records: list[SampleRecord]) -> float | None:
    """Mean normalized semantic entropy (H_n) across all samples."""
    if not records:
        return None
    return sum(r.normalized_entropy for r in records) / len(records)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def compute_ece(
    records: list[SampleRecord], n_bins: int = 10
) -> float | None:
    """Expected Calibration Error with equal-width bins over [0, 1].

    ECE = sum(|B_m| / n * |acc(B_m) - conf(B_m)|), M=10 bins.

    Computed over accepted (in-coverage) samples only. Abstained samples
    received no confidence decision and are excluded.

    Returns None if there are no accepted samples.
    """
    accepted = [r for r in records if r.in_coverage]
    if not accepted:
        return None

    n = len(accepted)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for m in range(n_bins):
        lo = bin_edges[m]
        hi = bin_edges[m + 1]
        bin_indices = [
            i
            for i, r in enumerate(accepted)
            if lo <= r.confidence < hi or (m == n_bins - 1 and r.confidence == hi)
        ]
        if not bin_indices:
            continue
        bin_size = len(bin_indices)
        bin_acc = sum(accepted[i].correct for i in bin_indices) / bin_size
        bin_conf = sum(accepted[i].confidence for i in bin_indices) / bin_size
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


# ---------------------------------------------------------------------------
# Selective accuracy and coverage
# ---------------------------------------------------------------------------


def compute_selective_accuracy(records: list[SampleRecord]) -> float | None:
    """Selective accuracy: accuracy over in-coverage (accepted) samples only.

    selective_accuracy = sum(1[correct] for in_coverage) / count(in_coverage)

    Returns None if there are no in-coverage samples.
    """
    in_coverage = [r for r in records if r.in_coverage]
    if not in_coverage:
        return None
    return sum(1 for r in in_coverage if r.correct) / len(in_coverage)


def compute_coverage(records: list[SampleRecord]) -> float:
    """Coverage: fraction of samples in coverage (not abstained)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.in_coverage) / len(records)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def compute_auroc(records: list[SampleRecord]) -> float | None:
    """ROC-AUC of confidence predicting binary correctness.

    Population: all samples with valid, evaluable confidence predictions —
    i.e., samples that are not forced to abstain by Phase 2 policy:
    - sampling_valid == True (invalid sampling produces no meaningful confidence)
    - evidence_state != "uncertain" (uncertain claims are forced to abstain)

    The population is independent of the operating threshold tau. Samples
    that abstain because confidence < tau are still included (they have
    meaningful confidence). Samples forced to abstain by policy are excluded.

    Uses the Mann-Whitney U statistic. Returns 0.5 if only one correctness
    class is present in the eligible population. Returns None if there are
    no eligible samples.
    """
    eligible = [
        r
        for r in records
        if r.sampling_valid and r.evidence_state != "uncertain"
    ]
    if not eligible:
        return None

    pos_scores = [r.confidence for r in eligible if r.correct]
    neg_scores = [r.confidence for r in eligible if not r.correct]

    if not pos_scores or not neg_scores:
        return None

    concordant = 0.0
    tied = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1.0
            elif ps == ns:
                tied += 1.0

    total = len(pos_scores) * len(neg_scores)
    return (concordant + 0.5 * tied) / total


# ---------------------------------------------------------------------------
# Accuracy-vs-coverage curve
# ---------------------------------------------------------------------------


def _is_eligible_for_coverage(record: SampleRecord) -> bool:
    """Determine if a sample is eligible for coverage at any threshold.

    A sample is eligible if it is NOT forced to abstain by Phase 2 policy:
    - sampling_valid == False: invalid sampling, no meaningful confidence
    - evidence_state == "uncertain": uncertain claims always abstain

    Forced-abstention samples never count as covered, including at tau=0.
    Samples that abstain only because confidence < tau are eligible.
    """
    return record.sampling_valid and record.evidence_state != "uncertain"


def compute_accuracy_vs_coverage_curve(
    records: list[SampleRecord],
) -> list[dict]:
    """Accuracy-vs-coverage (risk-coverage) curve points.

    Sweeps tau in [0, 0.01, ..., 1.0]. For each tau, coverage counts only
    samples eligible for coverage (not forced to abstain by Phase 2 policy)
    with confidence >= tau. Selective accuracy is computed over covered
    samples only.

    Invalid sampling and uncertain claims never count as covered, including
    at tau=0, consistent with the Phase 2 selective-prediction decision.

    Returns list of dicts: [{'tau': float, 'coverage': float,
    'selective_accuracy': float}].
    """
    taus = [round(i / 100, 2) for i in range(101)]
    n = len(records)
    eligible = [r for r in records if _is_eligible_for_coverage(r)]
    points: list[dict] = []

    for tau in taus:
        covered = [r for r in eligible if r.confidence >= tau]
        coverage = len(covered) / n if n > 0 else 0.0
        if covered:
            sel_acc = sum(1 for r in covered if r.correct) / len(covered)
        else:
            sel_acc = 0.0
        points.append(
            {"tau": tau, "coverage": coverage, "selective_accuracy": sel_acc}
        )

    return points


# ---------------------------------------------------------------------------
# Aggregate metric bundle
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """All aggregate metrics for an evaluation run."""

    n_samples: int
    n_positive_claims_total: int

    accuracy: float | None
    open_accuracy: float | None
    closed_accuracy: float | None

    usr_raw: float | None
    usr_filtered: float | None

    pointing_accuracy: float | None

    sampling_agreement: float | None
    normalized_entropy: float | None

    ece: float | None
    selective_accuracy: float | None
    coverage: float
    auroc: float | None

    accuracy_vs_coverage: list[dict]

    exclusions: list[str] = field(default_factory=list)


def compute_aggregate_metrics(
    records: list[SampleRecord],
    exclusions: list[str] | None = None,
) -> AggregateMetrics:
    """Compute all aggregate metrics from per-sample records.

    Args:
        records: Per-sample evaluation records.
        exclusions: Human-readable exclusion notes (e.g. samples excluded
            from pointing-accuracy denominator due to no GT regions).

    Returns:
        AggregateMetrics with all computed metrics.
    """
    exclusions = exclusions or []
    n_samples = len(records)
    n_positive_total = sum(r.n_positive_claims for r in records)

    return AggregateMetrics(
        n_samples=n_samples,
        n_positive_claims_total=n_positive_total,
        accuracy=compute_accuracy(records),
        open_accuracy=compute_open_accuracy(records),
        closed_accuracy=compute_closed_accuracy(records),
        usr_raw=compute_usr_raw(records),
        usr_filtered=compute_usr_filtered(records),
        pointing_accuracy=compute_pointing_accuracy(records),
        sampling_agreement=compute_sampling_agreement(records),
        normalized_entropy=compute_normalized_entropy(records),
        ece=compute_ece(records),
        selective_accuracy=compute_selective_accuracy(records),
        coverage=compute_coverage(records),
        auroc=compute_auroc(records),
        accuracy_vs_coverage=compute_accuracy_vs_coverage_curve(records),
        exclusions=exclusions,
    )
