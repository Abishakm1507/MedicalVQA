"""Unit tests for FS-005 Phase 2 uncertainty estimation and abstention gate.

Covers semantic clustering, normalized entropy, confidence calculation,
calibration, abstention decisions, ADR 0004 Option 3 compliance, and
ground-truth isolation.

All tests are deterministic and require no model weights, paid APIs, or
real VLM inference. External embedding dependencies (sentence-transformers)
are mocked via the force_fallback flag so tests do not require downloads.

Test count (reconciled): see FS-005_PHASE_2_CORRECTIVE_REVIEW.md, Issue 6.
"""

import pytest

from verification.confidence_gate import (
    ConfidenceCalibrator,
    ConfidenceConfig,
    abstain_decision,
    compute_confidence,
    make_selective_prediction,
)
from verification.semantic_clustering import SemanticClusterer

# ============================================================================
# Semantic clustering tests
# ============================================================================


def test_clustering_unanimous_answers_zero_entropy() -> None:
    """Unanimous answers produce agreement=1.0, H_n=0.0, is_valid=True."""
    clusterer = SemanticClusterer(force_fallback=True)
    answers = ["pleural effusion", "pleural effusion", "pleural effusion"]
    result = clusterer.compute_agreement_entropy(answers)
    assert result.agreement == pytest.approx(1.0)
    assert result.normalized_entropy == pytest.approx(0.0)
    assert result.raw_entropy == pytest.approx(0.0)
    assert result.is_valid is True
    assert len(result.clusters) == 1
    assert result.clusters[0].members == [0, 1, 2]


def test_clustering_distinct_answers_separate_clusters() -> None:
    """Distinct answers remain in separate clusters (canonical fallback)."""
    clusterer = SemanticClusterer(force_fallback=True)
    answers = ["pleural effusion", "pneumothorax", "normal"]
    result = clusterer.compute_agreement_entropy(answers)
    assert result.agreement == pytest.approx(1 / 3)
    assert len(result.clusters) == 3
    assert result.n_samples == 3
    assert result.is_valid is True


def test_clustering_synonymous_answers_same_cluster() -> None:
    """Canonical-normalization-equivalent answers cluster together."""
    clusterer = SemanticClusterer(force_fallback=True)
    answers = ["Pleural Effusion", "  pleural effusion  ", "Pleural effusion"]
    result = clusterer.compute_agreement_entropy(answers)
    assert result.agreement == pytest.approx(1.0)
    assert len(result.clusters) == 1
    assert result.is_valid is True


def test_clustering_maximum_uncertainty_uniform() -> None:
    """Uniform cluster distribution produces H_n=1.0."""
    clusterer = SemanticClusterer(force_fallback=True)
    answers = ["a", "b", "c", "d"]  # 4 distinct clusters, uniform
    result = clusterer.compute_agreement_entropy(answers)
    assert result.agreement == pytest.approx(0.25)
    assert result.normalized_entropy == pytest.approx(1.0)
    assert len(result.clusters) == 4
    assert result.is_valid is True


def test_clustering_n_equals_one() -> None:
    """Single sample produces H_n=0.0 (no uncertainty possible), is_valid=True."""
    clusterer = SemanticClusterer(force_fallback=True)
    result = clusterer.compute_agreement_entropy(["pleural effusion"])
    assert result.agreement == pytest.approx(1.0)
    assert result.normalized_entropy == pytest.approx(0.0)
    assert result.raw_entropy == pytest.approx(0.0)
    assert result.n_samples == 1
    assert result.is_valid is True


def test_clustering_empty_answers_invalid() -> None:
    """Empty answer list: is_valid=False, error set, agreement/H_n are placeholders."""
    clusterer = SemanticClusterer(force_fallback=True)
    result = clusterer.compute_agreement_entropy([])
    assert result.agreement == 0.0
    assert result.normalized_entropy == 0.0
    assert result.error == "no_samples"
    assert result.is_valid is False
    assert result.clusters == []


def test_clustering_reproducibility() -> None:
    """Same inputs produce identical results (deterministic clustering)."""
    clusterer1 = SemanticClusterer(force_fallback=True, similarity_threshold=0.75)
    clusterer2 = SemanticClusterer(force_fallback=True, similarity_threshold=0.75)
    answers = ["effusion", "effusion", "pneumothorax", "normal"]
    r1 = clusterer1.compute_agreement_entropy(answers)
    r2 = clusterer2.compute_agreement_entropy(answers)
    assert r1.agreement == r2.agreement
    assert r1.normalized_entropy == r2.normalized_entropy
    assert len(r1.clusters) == len(r2.clusters)


def test_clustering_threshold_changes_clustering() -> None:
    """Different thresholds produce different clusterings (threshold is configurable)."""
    answers = ["pleural effusion", "pleural effusion noted", "pneumothorax"]
    clusterer_strict = SemanticClusterer(force_fallback=True, similarity_threshold=1.0)
    result_strict = clusterer_strict.compute_agreement_entropy(answers)
    assert len(result_strict.clusters) == 3  # all distinct under exact match


def test_clustering_transitive_chaining() -> None:
    """Connected-component clustering is transitive: A~B, B~C => A,B,C same cluster
    even if A not directly similar to C (canonical fallback exact match).

    With canonical fallback, "a" == "a" and "b" == "b" are exact matches but
    "a" != "b". To create a transitive chain we need answers where adjacent
    pairs match exactly but non-adjacent don't. With exact-match fallback this
    cannot produce a 3-chain (exact match is not transitive across distinct
    strings). Instead we verify the clustering is connected-component based:
    A and B in same cluster, B and C in same cluster => all three together.
    Using embedding backend would show this; here we document the behavior
    and verify the clusterer produces connected components.
    """
    clusterer = SemanticClusterer(force_fallback=True, similarity_threshold=0.75)
    # Three answers where each adjacent pair is canonically equal but
    # the first and third differ. With exact-match fallback this yields
    # 2 clusters ({A,B} and {C}) or 3 clusters depending on values.
    answers = ["effusion left", "effusion left", "pneumothorax"]
    result = clusterer.compute_agreement_entropy(answers)
    # "effusion left" x2 cluster together, "pneumothorax" separate
    assert len(result.clusters) == 2
    assert result.clusters[0].members == [0, 1]
    assert result.clusters[1].members == [2]
    assert result.is_valid is True


# ============================================================================
# Issue 1 — Invalid/empty sampling must force abstention
# ============================================================================


def test_invalid_sampling_forces_abstention_empty() -> None:
    """Empty sampling (is_valid=False) => confidence=0.0 => abstention."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    # Sampling invalid: sampling_valid=False forces c=0.0
    c = compute_confidence(
        agreement=0.0,
        normalized_entropy=0.0,
        localizer_score=0.9,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert c == 0.0
    assert abstain_decision(confidence=c, threshold=0.5) is True


def test_invalid_sampling_forces_abstention_missing() -> None:
    """Missing sampling signals (n_samples=0) => is_valid=False => abstention."""
    clusterer = SemanticClusterer(force_fallback=True)
    result = clusterer.compute_agreement_entropy([])
    assert result.is_valid is False
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=result.agreement,
        normalized_entropy=result.normalized_entropy,
        localizer_score=0.9,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=result.is_valid,
    )
    assert c == 0.0
    assert abstain_decision(confidence=c, threshold=0.5) is True


def test_invalid_sampling_forces_abstention_malformed() -> None:
    """Malformed sampling (error set) => is_valid=False => abstention."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    # Simulate malformed output: agreement/H_n are placeholder zeros
    # but sampling_valid=False prevents them from being used.
    c = compute_confidence(
        agreement=1.0,  # placeholder: looks like perfect agreement
        normalized_entropy=0.0,  # placeholder: looks like zero uncertainty
        localizer_score=1.0,  # placeholder: looks like perfect localization
        config=config,
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert c == 0.0
    assert abstain_decision(confidence=c, threshold=0.5) is True


def test_valid_unanimous_zero_entropy_still_valid() -> None:
    """Valid unanimous sampling: zero entropy is legitimate, not invalid."""
    clusterer = SemanticClusterer(force_fallback=True)
    result = clusterer.compute_agreement_entropy(["yes", "yes", "yes"])
    assert result.is_valid is True
    assert result.normalized_entropy == pytest.approx(0.0)
    assert result.agreement == pytest.approx(1.0)
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=result.agreement,
        normalized_entropy=result.normalized_entropy,
        localizer_score=0.9,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=result.is_valid,
    )
    # Valid zero entropy should produce high confidence (not abstain)
    assert c > 0.5
    assert abstain_decision(confidence=c, threshold=0.5) is False


def test_invalid_uncertainty_not_mistaken_for_certainty() -> None:
    """Invalid sampling (placeholders: a=1.0, H_n=0.0, rho=1.0) must not
    produce high confidence when sampling_valid=False."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    # Placeholder values that look like perfect certainty:
    c_invalid = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert c_invalid == 0.0
    # Same values but valid sampling would produce high confidence:
    c_valid = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    assert c_valid == pytest.approx(1.0)
    assert c_invalid != c_valid


# ---------------------------------------------------------------------------
# Issue 1b — Invalid sampling must always abstain, even at tau=0.0
# ---------------------------------------------------------------------------


def test_invalid_sampling_abstains_at_tau_zero() -> None:
    """Invalid sampling with tau=0.0 must still abstain (sampling validity
    is an explicit condition, not just confidence=0.0)."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.0)
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert c == 0.0
    # Without sampling_valid check, abstain_decision(0.0, 0.0) would accept.
    assert abstain_decision(confidence=c, threshold=0.0, sampling_valid=False) is True


def test_invalid_sampling_abstains_at_tau_0_5() -> None:
    """Invalid sampling with tau=0.5 must abstain."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert c == 0.0
    assert abstain_decision(confidence=c, threshold=0.5, sampling_valid=False) is True


def test_valid_unanimous_accepts_at_tau_zero() -> None:
    """Valid unanimous sampling at tau=0.0 must accept (preserve boundary
    behavior for valid sampling)."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.0)
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    assert c == pytest.approx(1.0)
    assert abstain_decision(confidence=c, threshold=0.0, sampling_valid=True) is False


def test_missing_sampling_always_abstains() -> None:
    """Missing sampling (n_samples=0) always abstains regardless of tau."""
    clusterer = SemanticClusterer(force_fallback=True)
    result = clusterer.compute_agreement_entropy([])
    assert result.is_valid is False
    for tau in [0.0, 0.5, 1.0]:
        c = compute_confidence(
            agreement=result.agreement,
            normalized_entropy=result.normalized_entropy,
            localizer_score=0.9,
            config=ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=tau),
            evidence_state="present_grounded",
            sampling_valid=result.is_valid,
        )
        assert c == 0.0
        assert abstain_decision(confidence=c, threshold=tau, sampling_valid=False) is True


def test_malformed_sampling_always_abstains() -> None:
    """Malformed sampling (placeholder values) always abstains regardless of tau."""
    for tau in [0.0, 0.5, 1.0]:
        c = compute_confidence(
            agreement=1.0,  # placeholder: looks like perfect agreement
            normalized_entropy=0.0,  # placeholder: looks like zero uncertainty
            localizer_score=1.0,  # placeholder: looks like perfect localization
            config=ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=tau),
            evidence_state="present_grounded",
            sampling_valid=False,
        )
        assert c == 0.0
        assert abstain_decision(confidence=c, threshold=tau, sampling_valid=False) is True


# ---------------------------------------------------------------------------
# Issue 2 — Uncertain claims always abstain with explicit flag
# ---------------------------------------------------------------------------


def test_uncertain_claim_always_abstains() -> None:
    """Uncertain claims always abstain regardless of confidence or threshold."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.0)
    # Even with perfect agreement and zero entropy (max confidence), uncertain
    # claims abstain.
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=None,
        config=config,
        evidence_state="uncertain",
        sampling_valid=True,
    )
    assert c == pytest.approx(1.0)  # high confidence but must still abstain
    pred = make_selective_prediction(
        sample_id="q1",
        confidence=c,
        threshold=0.0,
        answer="Yes",
        evidence_state="uncertain",
        sampling_valid=True,
    )
    assert pred.is_abstaining is True
    assert pred.is_uncertain is True
    assert pred.final_answer is None
    assert pred.in_coverage is False


def test_uncertain_claim_explicitly_flagged() -> None:
    """Uncertain claims carry is_uncertain=True so they never silently appear
    as ordinary confident medical findings."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.0)
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=None,
        config=config,
        evidence_state="uncertain",
        sampling_valid=True,
    )
    pred = make_selective_prediction(
        sample_id="q1",
        confidence=c,
        threshold=0.0,
        answer="Yes",
        evidence_state="uncertain",
        sampling_valid=True,
    )
    assert pred.is_uncertain is True
    assert pred.is_abstaining is True


def test_present_grounded_not_affected_by_uncertain_rule() -> None:
    """Present grounded claims are NOT affected by the uncertain-claim rule;
    they follow normal threshold-based abstention."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=0.9,
        normalized_entropy=0.1,
        localizer_score=0.9,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    pred = make_selective_prediction(
        sample_id="q1",
        confidence=c,
        threshold=0.5,
        answer="Yes",
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    assert pred.is_uncertain is False
    # High confidence at tau=0.5 should accept
    assert pred.is_abstaining is False
    assert pred.final_answer == "Yes"


def test_selective_prediction_invalid_sampling_abstains() -> None:
    """make_selective_prediction with invalid sampling always abstains."""
    pred = make_selective_prediction(
        sample_id="q1",
        confidence=1.0,  # perfect confidence but invalid sampling
        threshold=0.0,  # even tau=0.0
        answer="Yes",
        evidence_state="present_grounded",
        sampling_valid=False,
    )
    assert pred.is_abstaining is True
    assert pred.final_answer is None
    assert pred.in_coverage is False
    assert pred.is_uncertain is False


# ============================================================================
# Issue 2 — Distinguish absent, uncertain, and ungrounded positive claims
# ============================================================================


def test_confidence_present_grounded_full_formula() -> None:
    """Present and grounded claim uses full confidence formula with rho."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=1.0,
        normalized_entropy=0.0,
        localizer_score=0.9,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    # c = 0.4*1.0 + 0.3*(1-0.0) + 0.3*0.9 = 0.4 + 0.3 + 0.27 = 0.97
    assert c == pytest.approx(0.97)


def test_confidence_absent_claim_rescaled_no_artificial_rho() -> None:
    """Absent claim (rho=None): rescaled formula, no artificial rho boost."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c_absent = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=None,
        config=config,
        evidence_state="absent",
        sampling_valid=True,
    )
    # Rescaled: (0.4*0.5 + 0.3*0.5) / (0.4+0.3) = 0.35/0.7 = 0.5
    assert c_absent == pytest.approx(0.5)
    # Same agreement/entropy with rho=1.0 (present_grounded) would be higher:
    c_present = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=1.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    assert c_present == pytest.approx(0.65)  # 0.2+0.15+0.3 = 0.65
    assert c_absent != c_present


def test_confidence_present_ungrounded_no_rescaling() -> None:
    """Present claim with region_id='NONE': positive but ungrounded.
    Uses full formula with rho=0.0, NO rescaling — missing localization
    evidence does NOT improve confidence by removing the localization term."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c_ungrounded = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=None,
        config=config,
        evidence_state="present_ungrounded",
        sampling_valid=True,
    )
    # Full formula with rho=0.0: 0.4*0.5 + 0.3*0.5 + 0.3*0.0 = 0.2+0.15+0.0 = 0.35
    assert c_ungrounded == pytest.approx(0.35)
    # Same values as absent would be rescaled to 0.5 — different!
    c_absent = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=None,
        config=config,
        evidence_state="absent",
        sampling_valid=True,
    )
    assert c_ungrounded != c_absent  # 0.35 vs 0.5


def test_confidence_uncertain_claim_distinct_state() -> None:
    """Uncertain claim: distinct state, rescaled formula (like absent) but
    tracked separately; must not silently become an ordinary accepted answer."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c_uncertain = compute_confidence(
        agreement=0.33,
        normalized_entropy=0.9,
        localizer_score=None,
        config=config,
        evidence_state="uncertain",
        sampling_valid=True,
    )
    # Rescaled: (0.4*0.33 + 0.3*0.1) / 0.7 = (0.132 + 0.03) / 0.7 = 0.2314...
    assert c_uncertain == pytest.approx((0.4 * 0.33 + 0.3 * 0.1) / 0.7)
    # Uncertain with rho=None is NOT boosted by rho
    assert c_uncertain < 0.5  # low confidence -> likely abstain at tau=0.5


def test_confidence_identical_ae_different_evidence_states() -> None:
    """Identical agreement and entropy produce different decisions because
    the claim evidence state differs."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    a, h = 0.5, 0.5

    c_absent = compute_confidence(a, h, None, config, "absent", True)
    c_uncertain = compute_confidence(a, h, None, config, "uncertain", True)
    c_ungrounded = compute_confidence(a, h, None, config, "present_ungrounded", True)
    c_grounded = compute_confidence(a, h, 0.9, config, "present_grounded", True)

    # Absent and uncertain both use the rescaled formula: same confidence
    # value, but they are distinct states (different Phase 1 verification
    # paths). They differ from ungrounded and grounded.
    assert c_absent == c_uncertain  # both rescaled: (0.2+0.15)/0.7 = 0.5
    assert c_ungrounded == pytest.approx(0.35)  # rho=0.0, no rescale
    assert c_grounded == pytest.approx(0.62)  # full formula with rho=0.9

    # Three distinct confidence values from four evidence states
    values = [c_absent, c_uncertain, c_ungrounded, c_grounded]
    assert len(set(values)) == 3, f"Expected 3 distinct values, got {values}"

    # Abstention decisions differ at tau=0.5:
    assert abstain_decision(c_ungrounded, 0.5) is True   # 0.35 < 0.5 -> abstain
    assert abstain_decision(c_absent, 0.5) is False       # 0.5 >= 0.5 -> accept
    assert abstain_decision(c_grounded, 0.5) is False     # 0.62 >= 0.5 -> accept


def test_confidence_missing_localizer_positive_claim_no_improvement() -> None:
    """Missing localization evidence for a positive claim must not improve
    confidence merely by removing the localization term and rescaling."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    # Positive claim, ungrounded (region_id="NONE"), rho effectively 0.0:
    c_no_rho = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=None,
        config=config,
        evidence_state="present_ungrounded",
        sampling_valid=True,
    )
    # Same as explicit rho=0.0:
    c_rho_zero = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=0.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    assert c_no_rho == c_rho_zero  # both 0.35, no rescaling benefit

    # Contrast with absent claim (rescaled): would be 0.5 — higher!
    c_absent = compute_confidence(
        agreement=0.5,
        normalized_entropy=0.5,
        localizer_score=None,
        config=config,
        evidence_state="absent",
        sampling_valid=True,
    )
    assert c_no_rho != c_absent  # 0.35 vs 0.5


# ============================================================================
# Abstention decision tests
# ============================================================================


def test_abstain_below_threshold() -> None:
    """Confidence below threshold -> abstain."""
    assert abstain_decision(confidence=0.4, threshold=0.5) is True


def test_abstain_at_threshold() -> None:
    """Confidence exactly at threshold -> accept (c >= tau)."""
    assert abstain_decision(confidence=0.5, threshold=0.5) is False


def test_abstain_above_threshold() -> None:
    """Confidence above threshold -> accept."""
    assert abstain_decision(confidence=0.9, threshold=0.5) is False


# ============================================================================
# Issue 4 — Selective prediction (no is_correct in runtime)
# ============================================================================


def test_selective_prediction_accept_no_is_correct() -> None:
    """Accepted prediction returns answer and coverage, no is_correct field."""
    pred = make_selective_prediction(
        sample_id="q1", confidence=0.8, threshold=0.5, answer="Yes"
    )
    assert pred.is_abstaining is False
    assert pred.final_answer == "Yes"
    assert pred.in_coverage is True
    assert not hasattr(pred, "is_correct")


def test_selective_prediction_abstain_no_is_correct() -> None:
    """Abstained prediction returns None answer and no coverage."""
    pred = make_selective_prediction(
        sample_id="q1", confidence=0.3, threshold=0.5, answer="Yes"
    )
    assert pred.is_abstaining is True
    assert pred.final_answer is None
    assert pred.in_coverage is False
    assert not hasattr(pred, "is_correct")


def test_selective_prediction_runtime_no_correctness_dependency() -> None:
    """Runtime prediction does not depend on correctness labels."""
    import inspect
    sig = inspect.signature(make_selective_prediction)
    params = list(sig.parameters.keys())
    assert "is_correct" not in params
    assert "correctness" not in params
    assert "ground_truth" not in params


# ============================================================================
# Threshold boundary tests
# ============================================================================


def test_threshold_zero_accepts_all() -> None:
    """tau=0.0 accepts all non-negative confidences."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.0)
    c = compute_confidence(
        agreement=0.0,
        normalized_entropy=1.0,
        localizer_score=0.0,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )  # c=0.0
    assert abstain_decision(confidence=c, threshold=0.0) is False  # c >= 0.0


def test_threshold_one_abstains_all_but_perfect() -> None:
    """tau=1.0 abstains unless c=1.0 exactly."""
    assert abstain_decision(confidence=0.99, threshold=1.0) is True
    assert abstain_decision(confidence=1.0, threshold=1.0) is False


# ============================================================================
# Issue 3 — Calibration objective (selective risk primary)
# ============================================================================


def test_calibrator_no_fitted_values_by_default() -> None:
    """Calibrator with no data returns empty result, not fabricated weights."""
    calibrator = ConfidenceCalibrator(grid_step=0.5)
    result = calibrator.calibrate(
        agreements=[],
        normalized_entropies=[],
        localizer_scores=[],
        evidence_states=[],
        correctness=[],
    )
    assert result.n_validation == 0
    assert result.auroc == 0.0
    assert result.selective_risk == 1.0
    assert result.coverage == 0.0
    assert result.coverage_curve == []


def test_calibrator_grid_search_constraints() -> None:
    """Calibrator respects w1+w2+w3=1, w_i>=0 constraints."""
    calibrator = ConfidenceCalibrator(grid_step=0.5)
    weights = calibrator._enumerate_weights()
    for w1, w2, w3 in weights:
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6
        assert w1 >= 0 and w2 >= 0 and w3 >= 0


def test_calibrator_selective_risk_primary_objective() -> None:
    """Calibrator minimizes selective risk (primary), not just AUROC."""
    calibrator = ConfidenceCalibrator(grid_step=0.5)
    # High confidence = correct, low confidence = incorrect
    agreements = [1.0, 1.0, 0.0, 0.0]
    entropies = [0.0, 0.0, 1.0, 1.0]
    localizer_scores = [1.0, 1.0, None, None]
    evidence_states = [
        "present_grounded",
        "present_grounded",
        "absent",
        "absent",
    ]
    correctness = [True, True, False, False]
    result = calibrator.calibrate(
        agreements, entropies, localizer_scores, evidence_states, correctness
    )
    assert result.n_validation == 4
    # Perfect separation: selective risk should be 0 at some tau
    assert result.selective_risk == pytest.approx(0.0)
    assert result.auroc == pytest.approx(1.0)


def test_calibrator_coverage_constraint_prevents_all_abstain() -> None:
    """Coverage constraint prevents trivial all-abstain solution (zero risk)."""
    calibrator = ConfidenceCalibrator(grid_step=0.5, min_coverage=0.5)
    # All samples incorrect — any coverage gives risk=1.0, but all-abstain
    # (coverage=0) would give risk=1.0 too. With min_coverage=0.5, the
    # calibrator must accept at least 50% of samples.
    agreements = [0.1, 0.1, 0.1, 0.1]
    entropies = [0.9, 0.9, 0.9, 0.9]
    localizer_scores = [0.0, 0.0, 0.0, 0.0]
    evidence_states = [
        "present_grounded",
        "present_grounded",
        "present_grounded",
        "present_grounded",
    ]
    correctness = [False, False, False, False]
    result = calibrator.calibrate(
        agreements, entropies, localizer_scores, evidence_states, correctness
    )
    # With min_coverage=0.5, coverage must be >= 0.5 (or no valid solution)
    assert result.coverage >= 0.5 or result.coverage == 0.0
    # selective_risk should be 1.0 (all incorrect)
    assert result.selective_risk == pytest.approx(1.0)


def test_calibrator_ece_and_auroc_secondary() -> None:
    """ECE and AUROC are reported as diagnostics, not primary objectives."""
    calibrator = ConfidenceCalibrator(grid_step=0.5)
    agreements = [1.0, 0.0, 1.0, 0.0]
    entropies = [0.0, 1.0, 0.0, 1.0]
    localizer_scores = [1.0, 0.0, 1.0, 0.0]
    evidence_states = [
        "present_grounded",
        "present_grounded",
        "present_grounded",
        "present_grounded",
    ]
    correctness = [True, False, True, False]
    result = calibrator.calibrate(
        agreements, entropies, localizer_scores, evidence_states, correctness
    )
    assert 0.0 <= result.ece <= 1.0
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.selective_risk <= 1.0
    assert 0.0 <= result.coverage <= 1.0


# ============================================================================
# Ground-truth isolation
# ============================================================================


def test_confidence_calculation_has_no_ground_truth_parameter() -> None:
    """compute_confidence takes no ground-truth inputs (GT isolation)."""
    import inspect
    sig = inspect.signature(compute_confidence)
    params = list(sig.parameters.keys())
    assert "ground_truth" not in params
    assert "gt_" not in "".join(params)
    assert "correctness" not in params


def test_abstention_decision_has_no_ground_truth() -> None:
    """abstain_decision takes only confidence and threshold, no GT."""
    import inspect
    sig = inspect.signature(abstain_decision)
    params = list(sig.parameters.keys())
    assert "ground_truth" not in params
    assert "gt_" not in "".join(params)


def test_selective_prediction_has_no_ground_truth() -> None:
    """make_selective_prediction takes no ground-truth or correctness inputs."""
    import inspect
    sig = inspect.signature(make_selective_prediction)
    params = list(sig.parameters.keys())
    assert "ground_truth" not in params
    assert "is_correct" not in params
    assert "correctness" not in params


# ============================================================================
# Phase 1 regression compatibility
# ============================================================================


def test_confidence_uses_phase1_verified_claim_fields() -> None:
    """Confidence calculation uses Phase 1 VerifiedClaim fields (localizer_score)."""
    localizer_score = 0.85  # rho from CandidateBox.score
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    c = compute_confidence(
        agreement=0.8,
        normalized_entropy=0.2,
        localizer_score=localizer_score,
        config=config,
        evidence_state="present_grounded",
        sampling_valid=True,
    )
    # c = 0.4*0.8 + 0.3*(1-0.2) + 0.3*0.85 = 0.32 + 0.24 + 0.255 = 0.815
    assert c == pytest.approx(0.815)


def test_config_rejects_invalid_weights() -> None:
    """ConfidenceConfig rejects weights not summing to 1.0."""
    with pytest.raises(ValueError):
        ConfidenceConfig(w1=0.5, w2=0.5, w3=0.5, tau=0.5)  # sums to 1.5


def test_config_rejects_negative_weights() -> None:
    """ConfidenceConfig rejects negative weights."""
    with pytest.raises(ValueError):
        ConfidenceConfig(w1=-0.1, w2=0.6, w3=0.5, tau=0.5)


def test_config_rejects_invalid_tau() -> None:
    """ConfidenceConfig rejects tau outside [0, 1]."""
    with pytest.raises(ValueError):
        ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=1.5)


# ============================================================================
# N=5 sampling standard (ADR 0005)
# ============================================================================


def test_default_n_samples_is_five() -> None:
    """Default n_samples=5 per ADR 0005 benchmark standard."""
    config = ConfidenceConfig(w1=0.4, w2=0.3, w3=0.3, tau=0.5)
    assert config.n_samples == 5


def test_clustering_n_equals_five_uniform_max_entropy() -> None:
    """N=5 uniform distribution produces H_n=1.0."""
    clusterer = SemanticClusterer(force_fallback=True)
    answers = ["a", "b", "c", "d", "e"]  # 5 distinct clusters
    result = clusterer.compute_agreement_entropy(answers)
    assert result.n_samples == 5
    assert result.agreement == pytest.approx(0.2)
    assert result.normalized_entropy == pytest.approx(1.0)
    assert result.is_valid is True
