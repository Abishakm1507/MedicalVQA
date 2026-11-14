"""Confidence estimation, calibration, and abstention gate.

Implements the confidence score fusion and selective-prediction decision from
ADR 0005 and the FS-005 technical design:

    c = w1 * a + w2 * (1 - H_n) + w3 * rho

where a = semantic agreement, H_n = normalized semantic entropy, and rho =
localizer score of the cited region. All weights and the abstention threshold
tau are configurable parameters fitted on held-out validation data -- none are
hardcoded as scientifically validated values.

Traceability:
  - Feature Specification: FS-005 (Phase 2), sections 4.6 and 4.7
  - ADR 0005: Confidence Gate Formulation, Semantic Clustering & Calibration
  - ADR 0004: Negative/Absence Claims (Option 3 accepted)
  - Research Paper: Section III-E (Confidence Gate)
"""

from dataclasses import dataclass
from typing import Literal

EvidenceState = Literal[
    "present_grounded",   # status="present", region_id in R (has rho)
    "absent",             # status="absent", region_id="NONE" (no rho, rescaled)
    "uncertain",          # status="uncertain" (no rho, rescaled, distinct state)
    "present_ungrounded", # status="present", region_id="NONE" (rho=0.0, no rescaling)
]


@dataclass
class ConfidenceConfig:
    """Configuration for confidence estimation and abstention.

    All parameters are configurable. None are hardcoded as scientifically
    validated values. Weights and threshold must be fitted on held-out
    validation data before benchmark use (per ADR 0005).
    """

    w1: float  # weight on semantic agreement a
    w2: float  # weight on (1 - normalized_entropy)
    w3: float  # weight on localizer score rho
    tau: float  # abstention threshold: accept if c >= tau
    n_samples: int = 5  # Standardized N=5 per ADR 0005
    temperature: float = 0.7  # Sampling temperature for N>1
    random_seed: int | None = None  # Recorded for reproducibility

    def __post_init__(self) -> None:
        """Validate weight constraints."""
        if not (abs(self.w1 + self.w2 + self.w3 - 1.0) < 1e-9):
            raise ValueError(
                f"Weights must sum to 1.0, got {self.w1 + self.w2 + self.w3}"
            )
        if self.w1 < 0 or self.w2 < 0 or self.w3 < 0:
            raise ValueError("Weights must be non-negative")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"Threshold tau must be in [0, 1], got {self.tau}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")


@dataclass
class CalibrationResult:
    """Result of confidence-weight and threshold calibration.

    Produced by ConfidenceCalibrator on held-out validation data. Not a
    scientifically validated configuration until fitted and reviewed.
    """

    w1: float
    w2: float
    w3: float
    tau: float
    ece: float  # Expected Calibration Error (diagnostic)
    auroc: float  # AUROC of fused confidence predicting correctness (secondary)
    selective_risk: float  # selective risk at operating tau (primary objective)
    coverage: float  # coverage at operating tau
    coverage_curve: list[tuple[float, float, float]]  # (tau, coverage, accuracy)
    calibration_bins: list[dict]  # for ECE reporting
    n_validation: int


@dataclass
class SelectivePrediction:
    """Result of the abstention gate decision.

    Runtime-only output: contains inference-time information only.
    Ground-truth correctness is NOT part of this output -- it is computed
    separately in the offline evaluation layer.

    ``is_uncertain`` is True when the claim evidence state is "uncertain" --
    such claims always abstain and are explicitly flagged so they never
    silently appear as ordinary confident medical findings.
    """

    sample_id: str
    confidence: float  # fused confidence c in [0, 1]
    threshold: float  # tau used for the decision
    is_abstaining: bool  # True if c < tau
    final_answer: str | None  # None if abstaining
    in_coverage: bool  # True if not abstaining
    is_uncertain: bool = False  # True if claim evidence state is "uncertain"


def compute_confidence(
    agreement: float,
    normalized_entropy: float,
    localizer_score: float | None,
    config: ConfidenceConfig,
    evidence_state: EvidenceState = "present_grounded",
    sampling_valid: bool = False,
) -> float:
    """Compute fused confidence score c.

    Formula (when sampling_valid=True and evidence_state="present_grounded"):
        c = w1 * a + w2 * (1 - H_n) + w3 * rho

    Evidence-state policies (per ADR 0004 Option 3 and FS-005 design):

    - "present_grounded": Full formula with the claim's localizer_score (rho).
    - "absent": ADR 0004 Option 3 -- no artificial localization confidence.
      The rho term is omitted and remaining weights are rescaled:
      c = (w1 * a + w2 * (1 - H_n)) / (w1 + w2)
    - "uncertain": Distinct state -- rho omitted and weights rescaled (same
      formula as absent, but tracked separately as uncertain).
    - "present_ungrounded": Positive claim citing region_id="NONE" -- NOT an
      absence claim. Uses the full formula with rho=0.0 (no rescaling), so
      missing localization evidence does NOT improve confidence by removing
      the localization term and rescaling the other weights.

    When sampling_valid=False (missing/empty/malformed sampling), returns 0.0
    so the abstention gate abstains. No placeholder agreement/entropy values
    are ever converted into a usable confidence score.

    Args:
        agreement: Semantic agreement a in [0, 1].
        normalized_entropy: Normalized semantic entropy H_n in [0, 1].
        localizer_score: Localizer score rho in [0, 1], or None.
        config: ConfidenceConfig with weights and threshold.
        evidence_state: Claim evidence state (default "present_grounded").
        sampling_valid: Whether sampling produced a valid uncertainty estimate
            (default False for safety -- unvalidated sampling forces abstention).

    Returns:
        Fused confidence c in [0, 1]. Returns 0.0 when sampling is invalid.
    """
    # Invalid sampling: cannot compute a meaningful confidence score.
    # Return 0.0 so the abstention gate abstains (c < tau for any tau >= 0,
    # except the degenerate tau=0.0 case which still evaluates c=0.0 >= 0.0).
    if not sampling_valid:
        return 0.0

    agreement_term = config.w1 * agreement
    entropy_term = config.w2 * (1.0 - normalized_entropy)

    if evidence_state in ("absent", "uncertain"):
        # ADR 0004 Option 3: omit rho, rescale remaining weights.
        # No artificial localization confidence for absence/uncertain claims.
        denominator = config.w1 + config.w2
        if denominator <= 0.0:
            return 0.0
        return (agreement_term + entropy_term) / denominator

    if evidence_state == "present_ungrounded":
        # Positive but ungrounded claim: full formula with rho=0.0, no
        # rescaling. Missing localization evidence does not improve confidence.
        rho_term = config.w3 * 0.0
        return agreement_term + entropy_term + rho_term

    # evidence_state == "present_grounded": full formula with actual rho.
    if localizer_score is None:
        # Present grounded claim but no localizer score available -- treat
        # as missing localization evidence (rho=0.0, no rescaling).
        rho_term = config.w3 * 0.0
    else:
        rho_term = config.w3 * localizer_score
    return agreement_term + entropy_term + rho_term


def abstain_decision(
    confidence: float,
    threshold: float,
    sampling_valid: bool = True,
) -> bool:
    """Decide whether to abstain based on confidence and threshold.

    Invalid sampling always forces abstention, regardless of confidence or
    threshold. This prevents the degenerate case where confidence=0.0 (returned
    for invalid sampling) is accepted when tau=0.0 (since 0.0 >= 0.0).

    Args:
        confidence: Fused confidence c in [0, 1].
        threshold: Abstention threshold tau.
        sampling_valid: Whether sampling produced a valid uncertainty estimate
            (default True). If False, always abstain.

    Returns:
        True if abstain (c < tau, or sampling invalid), False if accept.
    """
    if not sampling_valid:
        return True
    return confidence < threshold


def make_selective_prediction(
    sample_id: str,
    confidence: float,
    threshold: float,
    answer: str | None,
    evidence_state: EvidenceState = "present_grounded",
    sampling_valid: bool = True,
) -> SelectivePrediction:
    """Produce a selective prediction (accept or abstain).

    Runtime-only: does NOT accept or return ground-truth correctness.
    Correctness is computed separately in the offline evaluation layer.

    Abstention rules:
    - Invalid sampling (sampling_valid=False): always abstain.
    - Uncertain claims (evidence_state="uncertain"): always abstain, flagged
      with is_uncertain=True so they never silently appear as ordinary
      confident medical findings (ADR 0004 Option 3).
    - Otherwise: accept if confidence >= threshold, abstain otherwise.

    Args:
        sample_id: Identifier for the sample.
        confidence: Fused confidence c.
        threshold: Abstention threshold tau.
        answer: The candidate answer (None if abstaining).
        evidence_state: Claim evidence state (default "present_grounded").
        sampling_valid: Whether sampling produced a valid uncertainty estimate
            (default True). If False, always abstain.

    Returns:
        SelectivePrediction with the abstention decision.
    """
    if not sampling_valid:
        return SelectivePrediction(
            sample_id=sample_id,
            confidence=confidence,
            threshold=threshold,
            is_abstaining=True,
            final_answer=None,
            in_coverage=False,
            is_uncertain=False,
        )

    if evidence_state == "uncertain":
        # Uncertain claims always abstain and are explicitly flagged.
        return SelectivePrediction(
            sample_id=sample_id,
            confidence=confidence,
            threshold=threshold,
            is_abstaining=True,
            final_answer=None,
            in_coverage=False,
            is_uncertain=True,
        )

    is_abstaining = abstain_decision(confidence, threshold, sampling_valid=True)
    return SelectivePrediction(
        sample_id=sample_id,
        confidence=confidence,
        threshold=threshold,
        is_abstaining=is_abstaining,
        final_answer=None if is_abstaining else answer,
        in_coverage=not is_abstaining,
        is_uncertain=False,
    )


class ConfidenceCalibrator:
    """Grid-search calibration of confidence weights and threshold.

    Fits (w1, w2, w3) and tau on held-out validation data per ADR 0005:
    - Grid search over step size 0.05
    - Constraint: w1 + w2 + w3 = 1, w_i >= 0
    - PRIMARY objective: minimize selective risk at operating coverage
    - SECONDARY objective: AUROC (diagnostic only)
    - CALIBRATION: ECE (diagnostic only)
    - Coverage constraint prevents trivial all-abstain solution
    - Selected weights and tau are frozen before touching the test set.

    This class provides the calibration interface; it does NOT ship with
    pre-fitted values. Calibration remains pending until run on held-out
    validation data.
    """

    def __init__(
        self,
        grid_step: float = 0.05,
        min_coverage: float = 0.1,
    ) -> None:
        """Initialize the calibrator.

        Args:
            grid_step: Grid search step size for weight enumeration (0.05 per ADR 0005).
            min_coverage: Minimum acceptable coverage at operating tau.
                Prevents trivial all-abstain solutions (default 0.1 = 10%).
        """
        self.grid_step = grid_step
        self.min_coverage = min_coverage

    def _enumerate_weights(self) -> list[tuple[float, float, float]]:
        """Enumerate all valid weight triples summing to 1.0."""
        weights: list[tuple[float, float, float]] = []
        step = self.grid_step
        w1 = 0.0
        while w1 <= 1.0 + 1e-9:
            w2 = 0.0
            while w2 <= 1.0 - w1 + 1e-9:
                w3 = round(1.0 - w1 - w2, 10)
                if w3 >= -1e-9:
                    weights.append(
                        (round(w1, 10), round(w2, 10), max(0.0, round(w3, 10)))
                    )
                w2 = round(w2 + step, 10)
            w1 = round(w1 + step, 10)
        return weights

    def _auroc(self, scores: list[float], labels: list[bool]) -> float:
        """Compute ROC-AUC using the Mann-Whitney U statistic.

        Args:
            scores: Confidence scores for each sample.
            labels: Binary correctness labels.

        Returns:
            AUROC in [0, 1]. Returns 0.5 if only one class present.
        """
        pos_scores = [s for s, label in zip(scores, labels, strict=False) if label]
        neg_scores = [s for s, label in zip(scores, labels, strict=False) if not label]

        if not pos_scores or not neg_scores:
            return 0.5

        # Mann-Whitney U: P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
        concordant = 0.0
        tied = 0.0
        total = len(pos_scores) * len(neg_scores)
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    concordant += 1.0
                elif ps == ns:
                    tied += 1.0
        return (concordant + 0.5 * tied) / total

    def _ece(
        self,
        scores: list[float],
        labels: list[bool],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error with equal-width bins.

        Args:
            scores: Confidence scores.
            labels: Binary correctness labels.
            n_bins: Number of equal-width bins (10 per design).

        Returns:
            ECE value.
        """
        n = len(scores)
        if n == 0:
            return 0.0

        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        ece = 0.0
        for m in range(n_bins):
            lo = bin_edges[m]
            hi = bin_edges[m + 1]
            bin_indices = [
                i
                for i, s in enumerate(scores)
                if lo <= s < hi or (m == n_bins - 1 and s == hi)
            ]
            if not bin_indices:
                continue
            bin_size = len(bin_indices)
            bin_acc = sum(labels[i] for i in bin_indices) / bin_size
            bin_conf = sum(scores[i] for i in bin_indices) / bin_size
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
        return ece

    def _selective_risk(
        self,
        scores: list[float],
        labels: list[bool],
        tau: float,
    ) -> tuple[float, float]:
        """Compute selective risk and coverage at threshold tau.

        Selective risk = 1 - selective_accuracy = fraction incorrect among
        accepted (in-coverage) samples.

        Args:
            scores: Confidence scores for each sample.
            labels: Binary correctness labels.
            tau: Abstention threshold.

        Returns:
            (selective_risk, coverage) tuple.
        """
        n = len(scores)
        accepted = [i for i, s in enumerate(scores) if s >= tau]
        coverage = len(accepted) / n if n > 0 else 0.0
        if not accepted:
            return 1.0, coverage  # all abstained: maximum risk
        n_incorrect = sum(1 for i in accepted if not labels[i])
        selective_risk = n_incorrect / len(accepted)
        return selective_risk, coverage

    def calibrate(
        self,
        agreements: list[float],
        normalized_entropies: list[float],
        localizer_scores: list[float | None],
        evidence_states: list[EvidenceState],
        correctness: list[bool],
        candidate_taus: list[float] | None = None,
    ) -> CalibrationResult:
        """Calibrate weights and threshold on validation data.

        Grid searches over valid weight triples and candidate thresholds to
        minimize selective risk at operating coverage (primary objective per
        ADR 0005), with AUROC as secondary diagnostic and ECE as calibration
        diagnostic.

        A coverage constraint (min_coverage) prevents the trivial solution of
        abstaining on nearly every sample to achieve zero risk.

        Args:
            agreements: Semantic agreement a per validation sample.
            normalized_entropies: Normalized entropy H_n per sample.
            localizer_scores: Localizer score rho per sample (None for absence).
            evidence_states: Evidence state per validation sample.
            correctness: Binary correctness label per validation sample.
            candidate_taus: Thresholds to sweep (defaults to 101 points in [0,1]).

        Returns:
            CalibrationResult with best weights, tau, and metrics.
        """
        n = len(agreements)
        if n == 0:
            return CalibrationResult(
                w1=0.0, w2=0.0, w3=0.0, tau=0.0, ece=0.0, auroc=0.0,
                selective_risk=1.0, coverage=0.0,
                coverage_curve=[], calibration_bins=[], n_validation=0,
            )

        if candidate_taus is None:
            candidate_taus = [round(i / 100, 2) for i in range(101)]

        best_result: CalibrationResult | None = None
        best_risk = float("inf")

        for w1, w2, w3 in self._enumerate_weights():
            config = ConfidenceConfig(w1=w1, w2=w2, w3=w3, tau=0.5)
            scores = [
                compute_confidence(a, h, rho, config, es, sampling_valid=True)
                for a, h, rho, es in zip(
                    agreements, normalized_entropies, localizer_scores, evidence_states,
                    strict=False,
                )
            ]

            for tau in candidate_taus:
                config.tau = tau
                risk, coverage = self._selective_risk(scores, correctness, tau)

                # Coverage constraint: skip solutions below min_coverage
                if coverage < self.min_coverage:
                    continue

                # Primary objective: minimize selective risk
                if risk < best_risk:
                    best_risk = risk
                    ece = self._ece(scores, correctness)
                    auroc = self._auroc(scores, correctness)
                    # Build coverage curve at the selected tau.
                    coverage_curve: list[tuple[float, float, float]] = []
                    accepted = [i for i, s in enumerate(scores) if s >= tau]
                    cov = len(accepted) / n
                    if accepted:
                        sel_acc = sum(correctness[i] for i in accepted) / len(accepted)
                    else:
                        sel_acc = 0.0
                    coverage_curve.append((tau, cov, sel_acc))
                    best_result = CalibrationResult(
                        w1=w1,
                        w2=w2,
                        w3=w3,
                        tau=tau,
                        ece=ece,
                        auroc=auroc,
                        selective_risk=risk,
                        coverage=coverage,
                        coverage_curve=coverage_curve,
                        calibration_bins=[],  # populated by evaluation layer
                        n_validation=n,
                    )

        if best_result is None:
            # No solution met the coverage constraint -- return a safe
            # default that will abstain on most samples rather than fabricate
            # a high-coverage high-risk configuration.
            best_result = CalibrationResult(
                w1=0.0, w2=0.0, w3=0.0, tau=1.0, ece=0.0, auroc=0.0,
                selective_risk=1.0, coverage=0.0,
                coverage_curve=[], calibration_bins=[], n_validation=n,
            )
        return best_result
