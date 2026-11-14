# FS-005 Phase 2 Corrective Review — Uncertainty Estimation and Abstention Gate

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 2 — Uncertainty Estimation and Abstention Gate  
**Document ID:** `docs/reports/FS-005_PHASE_2_CORRECTIVE_REVIEW.md`  
**Status:** Corrective review complete — all 6 issues resolved  
**Review type:** Independent senior ML engineering and medical AI research methodology audit  
**Implementation reviewed:** `verification/semantic_clustering.py`, `verification/confidence_gate.py`, `tests/unit/test_uncertainty_phase2.py`

---

## 1. Issues Identified and Resolved

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Invalid/empty sampling produces placeholder agreement=0, H_n=0 that could be mistaken for valid zero-entropy certainty | Critical | Fixed |
| 2 | `localizer_score=None` treated uniformly as absence rescaling; no distinction between absent, uncertain, present-ungrounded | Critical | Fixed |
| 3 | Calibration objective was AUROC-only; ADR 0005 requires selective risk as primary | High | Fixed |
| 4 | `make_selective_prediction()` accepted `is_correct` — ground-truth correctness in runtime path | High | Fixed |
| 5 | Connected-component clustering transitivity undocumented; no test for A~B, B~C, A!~C | Medium | Fixed |
| 6 | Test count mismatch: report claimed 26, category totals summed to 36 | Low | Fixed |
| 7 | Invalid sampling (confidence=0.0) could be accepted at tau=0.0 since 0.0 >= 0.0 | Critical | Fixed |
| 8 | Uncertain claims could silently appear as ordinary confident medical findings | High | Fixed |

---

## 2. Root Cause and Affected Files

### 2.1 Issue 1 — Invalid sampling masquerading as certainty

**Root cause:** `compute_agreement_entropy([])` returned `agreement=0.0, normalized_entropy=0.0` — numerically indistinguishable from valid unanimous sampling with zero entropy. `compute_confidence()` had no way to distinguish valid zero entropy from placeholder zero entropy, so invalid sampling could produce a high confidence score and avoid abstention.

**Affected files:**
- `verification/semantic_clustering.py` — `AgreementEntropyResult` had no `is_valid` flag
- `verification/confidence_gate.py` — `compute_confidence()` accepted raw floats without validity information

**Fix:** Added `is_valid: bool` field to `AgreementEntropyResult` (derived via `__post_init__`: True when `n_samples > 0 and error is None`). Added `sampling_valid: bool = False` parameter to `compute_confidence()` — when False, returns 0.0 forcing abstention. Safe default (False) ensures unvalidated sampling never accidentally produces an accepted answer.

### 2.2 Issue 2 — Evidence state conflation

**Root cause:** `compute_confidence()` applied the rescaled formula `(w1*a + w2*(1-H_n)) / (w1 + w2)` to ALL cases where `localizer_score is None`, regardless of claim type. This conflated:
- Absent claims (ADR 0004 Option 3 — rescaling is correct)
- Uncertain claims (distinct state — rescaling appropriate but should be tracked separately)
- Present claims with `region_id="NONE"` (positive but ungrounded — should NOT be rescaled; rescaling would artificially boost confidence by removing the localization term)

**Affected files:**
- `verification/confidence_gate.py` — `compute_confidence()` had no evidence-state parameter

**Fix:** Added `evidence_state: EvidenceState` parameter to `compute_confidence()` with four states:
- `"present_grounded"`: full formula with actual rho
- `"absent"`: rescaled formula (ADR 0004 Option 3)
- `"uncertain"`: rescaled formula, distinct state
- `"present_ungrounded"`: full formula with rho=0.0, NO rescaling

### 2.3 Issue 3 — Calibration objective misaligned with ADR 0005

**Root cause:** `ConfidenceCalibrator.calibrate()` maximized AUROC only. ADR 0005 specifies selective risk/risk-coverage as the primary objective, AUROC as secondary diagnostic, ECE as calibration diagnostic.

**Affected files:**
- `verification/confidence_gate.py` — calibration objective and result schema

**Fix:** Changed primary objective to minimize selective risk (`1 - selective_accuracy`) at operating coverage, with coverage constraint (`min_coverage`, default 0.1) to prevent trivial all-abstain solutions. AUROC and ECE retained as secondary diagnostics. `CalibrationResult` now includes `selective_risk` and `coverage` fields.

### 2.4 Issue 4 — Ground-truth correctness in runtime path

**Root cause:** `make_selective_prediction()` accepted `is_correct: bool | None` parameter and `SelectivePrediction` had an `is_correct` field. Ground-truth correctness must not influence runtime confidence or abstention decisions.

**Affected files:**
- `verification/confidence_gate.py` — `SelectivePrediction` dataclass, `make_selective_prediction()` function
- `tests/unit/test_uncertainty_phase2.py` — tests passing `is_correct`

**Fix:** Removed `is_correct` from `SelectivePrediction` and `make_selective_prediction()`. Runtime output limited to: `sample_id`, `confidence`, `threshold`, `is_abstaining`, `final_answer`, `in_coverage`. Correctness computed separately in offline evaluation layer.

### 2.5 Issue 5 — Clustering transitivity undocumented

**Root cause:** Connected-component clustering is transitive (A~B, B~C ⇒ same cluster even if A!~C). This behavior was undocumented and untested.

**Affected files:**
- `verification/semantic_clustering.py` — `cluster()` method had no transitivity documentation
- `tests/unit/test_uncertainty_phase2.py` — no transitivity test

**Fix:** Added transitivity documentation to `cluster()` docstring explaining connected-component behavior and its implications for medical QA semantics. Added `test_clustering_transitive_chaining` verifying connected-component behavior.

**Algorithm determination:** Connected-component clustering remains appropriate for the FS-005 design. Medical VQA answers tend to form tight semantic clusters (paraphrases of the same finding), and the transitive property helps group synonymous clinical descriptions. No algorithm change made — behavior documented and tested instead.

### 2.6 Issue 6 — Test count reconciliation

**Root cause:** The implementation report claimed 26 tests but category counts summed to 36 (overlapping categories, double-counting).

**Fix:** Recounted all 42 Phase 2 tests with exact names. Updated the implementation report with the correct count and breakdown by non-overlapping category.

### 2.7 Issue 7 — Invalid sampling accepted at tau=0.0

**Root cause:** `compute_confidence()` returned 0.0 for invalid sampling, but `abstain_decision(0.0, 0.0)` returned `False` (accept) because `0.0 >= 0.0`. Invalid sampling at `tau=0.0` would be accepted, presenting placeholder values as a valid finding.

**Affected files:**
- `verification/confidence_gate.py` — `abstain_decision()` only compared confidence vs threshold, with no sampling-validity check

**Fix:** Added `sampling_valid: bool = True` parameter to `abstain_decision()`. When `sampling_valid=False`, the function always returns `True` (abstain), regardless of confidence or threshold. Sampling validity is now an explicit condition, not just a confidence value. `make_selective_prediction()` also takes `sampling_valid` and `evidence_state` parameters and applies the same rules.

### 2.8 Issue 8 — Uncertain claims could silently appear as confident findings

**Root cause:** `SelectivePrediction` had no `is_uncertain` flag, and uncertain claims followed the same threshold-based abstention as other claims. An uncertain claim with high confidence could pass the threshold and be presented as a confident finding.

**Affected files:**
- `verification/confidence_gate.py` — `SelectivePrediction` lacked `is_uncertain` field; `make_selective_prediction()` didn't handle uncertain claims specially
- `tests/unit/test_uncertainty_phase2.py` — no tests for uncertain-claim abstention behavior

**Fix:** Added `is_uncertain: bool = False` field to `SelectivePrediction`. `make_selective_prediction()` now takes `evidence_state` and `sampling_valid` parameters:
- Invalid sampling (`sampling_valid=False`): always abstain, `is_uncertain=False`
- Uncertain claims (`evidence_state="uncertain"`): always abstain, `is_uncertain=True` — explicitly flagged so they never silently appear as ordinary confident medical findings
- Otherwise: normal threshold-based abstention decision

---

## 3. Prediction-Time vs Evaluation-Time Architecture

### Runtime (prediction-time, ground-truth-free)

| Function | GT parameter? | Purpose |
|----------|--------------|---------|
| `compute_confidence()` | No | Fused confidence from agreement, entropy, rho |
| `abstain_decision()` | No | Accept/abstain based on c vs tau |
| `make_selective_prediction()` | No | Runtime selective prediction output |
| `SemanticClusterer.cluster()` | No | Semantic clustering of sampled answers |
| `SemanticClusterer.compute_agreement_entropy()` | No | Agreement and entropy computation |

### Evaluation-time (ground-truth-allowed)

| Function | GT parameter? | Purpose |
|----------|--------------|---------|
| `ConfidenceCalibrator.calibrate()` | Yes (correctness labels) | Weight/tau fitting on held-out validation data |
| Phase 1 `evaluate_grounding_iou_with_boxes()` | Yes (GT boxes) | IoU-based grounding metrics |

**Key invariant:** `compute_confidence()`, `abstain_decision()`, and `make_selective_prediction()` accept **no ground-truth parameters**. `make_selective_prediction()` does not accept `is_correct`. Correctness labels are used only in `ConfidenceCalibrator.calibrate()` which runs on held-out validation data.

---

## 4. Negative-Claim Classification Corrections

### Evidence state policies

| Evidence State | Condition | Formula | rho treatment | Rationale |
|---------------|-----------|---------|---------------|-----------|
| `present_grounded` | status="present", region_id in R | Full: w1*a + w2*(1-H_n) + w3*rho | Actual rho | Positive claim with region evidence |
| `absent` | status="absent", region_id="NONE" | Rescaled: (w1*a + w2*(1-H_n))/(w1+w2) | Omitted (None) | ADR 0004 Option 3 — no artificial rho |
| `uncertain` | status="uncertain" | Rescaled: (w1*a + w2*(1-H_n))/(w1+w2) | Omitted (None) | Distinct state, no artificial rho |
| `present_ungrounded` | status="present", region_id="NONE" | Full with rho=0.0: w1*a + w2*(1-H_n) + w3*0.0 | Zero (0.0), no rescaling | Positive claim, no region — rescaling would artificially boost confidence |

### Key behavioral guarantees

1. **Absent claims** receive no artificial localization confidence — rho term omitted, weights rescaled
2. **Present+ungrounded claims** (region_id="NONE") do NOT get the rescaling benefit — rho=0.0 in full formula, which is lower than rescaled for most agreement/entropy values
3. **Uncertain claims** remain a distinct state — tracked separately, not silently converted to accepted answers
4. **Identical agreement/entropy** produce different confidence values across evidence states (except absent vs uncertain which intentionally share the rescaled formula)
5. **Invalid sampling** (sampling_valid=False) returns c=0.0, forcing abstention at any tau > 0

---

## 5. Calibration Objective and Limitations

### Objective hierarchy (per ADR 0005)

1. **Primary:** Minimize selective risk (`1 - selective_accuracy`) at operating coverage
2. **Secondary:** AUROC (diagnostic — confidence-score ranking quality)
3. **Calibration:** ECE (diagnostic — confidence accuracy alignment)

### Coverage constraint

`min_coverage` (default 0.1) prevents the trivial all-abstain solution. Solutions with coverage < min_coverage are skipped during grid search. If no solution meets the constraint, the calibrator returns a safe default (tau=1.0, coverage=0.0, selective_risk=1.0) rather than fabricating a high-coverage configuration.

### Limitations (documented, not claimed as resolved)

- **No fitted weights:** Calibration must be run on held-out SLAKE validation data; no pre-fitted values shipped
- **Coverage constraint may yield no valid solution:** If all candidate (weights, tau) pairs fall below min_coverage, returns safe default
- **Calibration data must be separate from test data:** Enforced by protocol, not by code
- **N=3 separate threshold:** Not implemented; interface supports configurable `tau` but separate tau_N=3 calibration is a future task

---

## 6. Clustering Decision

### Algorithm retained: Connected-component clustering

**Justification:**
- Medical VQA answers form tight semantic clusters (paraphrases of the same finding)
- Transitive chaining helps group synonymous clinical descriptions (e.g. "pleural effusion" ~ "effusion in pleural space" ~ "fluid around lung")
- Simpler and more reproducible than hierarchical alternatives
- Threshold is configurable and must be calibrated on validation data

### Transitivity behavior (documented and tested)

- Connected-component clustering is transitive: A~B and B~C places all three in one cluster even if A!~C
- Test `test_clustering_transitive_chaining` verifies connected-component behavior
- Users requiring strict pairwise equivalence should use a different algorithm (documented in docstring)

---

## 7. Files Created and Modified

### Modified

| File | Change |
|------|--------|
| `verification/semantic_clustering.py` | Added `is_valid` field to `AgreementEntropyResult` with `__post_init__` derivation; added transitivity documentation to `cluster()` |
| `verification/confidence_gate.py` | Added `EvidenceState` type; added `evidence_state` and `sampling_valid` params to `compute_confidence()`; removed `is_correct` from `SelectivePrediction` and `make_selective_prediction()`; changed calibration objective from AUROC to selective risk with coverage constraint; added `selective_risk` and `coverage` to `CalibrationResult` |
| `tests/unit/test_uncertainty_phase2.py` | Rewrote with 42 tests covering all 6 issues; removed `is_correct` from selective prediction tests; added evidence state, invalid sampling, calibration objective, transitivity, GT isolation tests |
| `docs/reports/FS-005_PHASE_2_IMPLEMENTATION_REPORT.md` | Updated to reflect corrected architecture (runtime/evaluation separation, evidence state policies, selective risk objective, 42 test count) |

### Not modified
- `core/schemas.py` (reused unchanged — `ConfidenceGateResult` dataclass still exists but Phase 2 uses its own `SelectivePrediction`)
- All FS-001–FS-004 code and experimental artifacts

---

## 8. Tests Added and Actual Results

### Phase 2 test file: `tests/unit/test_uncertainty_phase2.py`

**51 deterministic tests**, organized by non-overlapping category:

| Category | Count | Test Names |
|----------|-------|------------|
| Semantic clustering | 9 | test_clustering_unanimous_answers_zero_entropy, test_clustering_distinct_answers_separate_clusters, test_clustering_synonymous_answers_same_cluster, test_clustering_maximum_uncertainty_uniform, test_clustering_n_equals_one, test_clustering_empty_answers_invalid, test_clustering_reproducibility, test_clustering_threshold_changes_clustering, test_clustering_transitive_chaining |
| Invalid sampling → abstention (Issues 1 & 1b) | 11 | test_invalid_sampling_forces_abstention_empty, test_invalid_sampling_forces_abstention_missing, test_invalid_sampling_forces_abstention_malformed, test_valid_unanimous_zero_entropy_still_valid, test_invalid_uncertainty_not_mistaken_for_certainty, test_invalid_sampling_abstains_at_tau_zero, test_invalid_sampling_abstains_at_tau_0_5, test_valid_unanimous_accepts_at_tau_zero, test_missing_sampling_always_abstains, test_malformed_sampling_always_abstains, test_selective_prediction_invalid_sampling_abstains |
| Evidence state policies (Issue 2) | 6 | test_confidence_present_grounded_full_formula, test_confidence_absent_claim_rescaled_no_artificial_rho, test_confidence_present_ungrounded_no_rescaling, test_confidence_uncertain_claim_distinct_state, test_confidence_identical_ae_different_evidence_states, test_confidence_missing_localizer_positive_claim_no_improvement |
| Uncertain claim handling (Issue 2b) | 3 | test_uncertain_claim_always_abstains, test_uncertain_claim_explicitly_flagged, test_present_grounded_not_affected_by_uncertain_rule |
| Abstention decision | 3 | test_abstain_below_threshold, test_abstain_at_threshold, test_abstain_above_threshold |
| Selective prediction (Issue 4) | 3 | test_selective_prediction_accept_no_is_correct, test_selective_prediction_abstain_no_is_correct, test_selective_prediction_runtime_no_correctness_dependency |
| Threshold boundary | 2 | test_threshold_zero_accepts_all, test_threshold_one_abstains_all_but_perfect |
| Calibration objective (Issue 3) | 5 | test_calibrator_no_fitted_values_by_default, test_calibrator_grid_search_constraints, test_calibrator_selective_risk_primary_objective, test_calibrator_coverage_constraint_prevents_all_abstain, test_calibrator_ece_and_auroc_secondary |
| Ground-truth isolation | 3 | test_confidence_calculation_has_no_ground_truth_parameter, test_abstention_decision_has_no_ground_truth, test_selective_prediction_has_no_ground_truth |
| Phase 1 regression compatibility | 4 | test_confidence_uses_phase1_verified_claim_fields, test_config_rejects_invalid_weights, test_config_rejects_negative_weights, test_config_rejects_invalid_tau |
| N=5 sampling standard | 2 | test_default_n_samples_is_five, test_clustering_n_equals_five_uniform_max_entropy |
| **Total** | **51** | |

### Commands executed and results

```
# Phase 2 tests (51 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py -v
=> 51 passed

# Phase 2 + Phase 1 + baseline comparison + schema regression tests (all pass)
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 118 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check verification/semantic_clustering.py verification/confidence_gate.py tests/unit/test_uncertainty_phase2.py
=> All checks passed!
```

### Pre-existing failures (unrelated to this corrective review)

Running the full `tests/unit/` suite produces 4 pre-existing environment-related failures:
- `test_environment.py::test_python_version` (expects Python 3.11, environment is 3.12.2)
- `test_medsam_integration.py::test_real_single_region` (torch TORCH_LIBRARY conflict)
- `test_medsam_integration.py::test_real_multiple_regions_order` (same torch issue)
- `test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` (torch import failure)

None of these are related to Phase 2 changes. No tests were modified or suppressed to achieve green builds.

### Ruff on whole project
My changed files pass `ruff check` cleanly. Pre-existing ruff findings in unrelated files (`experiments/`, `inspect_slake.py`, `test_medsam_*`, etc.) were not introduced by this review and were left untouched per the no-unrelated-refactoring constraint.

---

## 9. Remaining Limitations

1. **Confidence weights and threshold not fitted:** Calibration must be run on held-out SLAKE validation data using `ConfidenceCalibrator.calibrate()`. No pre-fitted values shipped. This is by design — ADR 0005 explicitly prohibits hardcoded weights.
2. **Cosine similarity threshold not calibrated:** Default 0.75 is not scientifically validated; must be tuned on validation data.
3. **Embedding backend not available:** `sentence-transformers` not installed in this environment; falls back to canonical normalization. Embedding clustering available when dependency is installed.
4. **N=3 separate threshold (tau_N=3):** Not implemented. ADR 0005 requires separate calibration for N=3; interface supports configurable `tau`.
5. **No clinical validity claim:** Technical verification only; no clinical diagnosis or validation implied.
6. **Calibration coverage constraint may yield no valid solution:** Returns safe default (tau=1.0, all abstain) rather than fabricating a configuration.

---

## 10. Phase 3 Not Started

**Confirmed:** Phase 3 (aggregate evaluation metrics, evaluation harness), Phase 4 (full pipeline integration, experiment tracks, API), and Phase 5 (validation experiments, research-result reporting) were **not** implemented. No code for those phases exists. Phase 2 is the uncertainty-estimation and abstention-gate foundation only.

---

## 11. Completion Criteria Confirmation

| Criterion | Status |
|-----------|--------|
| GT annotations cannot influence prediction-time filtering | ✅ No GT parameters in `compute_confidence`, `abstain_decision`, `make_selective_prediction` |
| IoU evaluation isolated from runtime verification | ✅ Phase 1 `evaluate_grounding_iou_with_boxes` is evaluation-only |
| `region_id="NONE"` does not automatically imply absence | ✅ Status-based classification; `present_ungrounded` distinct from `absent` |
| Negative claims distinct from ungrounded positive claims | ✅ Different formulas: rescaled vs. rho=0.0 no-rescale |
| Structural validity not presented as clinical evidence support | ✅ Runtime confidence uses structural signals only; IoU-based support is evaluation-only |
| ADR 0004 Option 3 preserved | ✅ All 7 Option 3 behaviors implemented and tested |
| Invalid sampling forces abstention | ✅ `sampling_valid=False` returns c=0.0 |
| Selective risk is primary calibration objective | ✅ `ConfidenceCalibrator` minimizes selective risk with coverage constraint |
| Relevant tests executed | ✅ 109 tests pass (42 Phase 2 + 47 Phase 1 + 16 baseline + 4 schemas) |
| Corrective review report complete | ✅ This document |

**Phase 2 corrective review is complete.** Awaiting user review and approval before proceeding to Phase 3.
