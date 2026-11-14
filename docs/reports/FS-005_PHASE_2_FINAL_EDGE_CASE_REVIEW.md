# FS-005 Phase 2 Final Edge-Case Review — Abstention Edge Cases

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 2 — Uncertainty Estimation and Abstention Gate  
**Document ID:** `docs/reports/FS-005_PHASE_2_FINAL_EDGE_CASE_REVIEW.md`  
**Status:** Addendum to corrective review — edge-case fixes complete  
**Review type:** Independent senior ML engineering and medical AI research methodology audit  
**Implementation reviewed:** `verification/confidence_gate.py`, `tests/unit/test_uncertainty_phase2.py`

---

## 1. Issues Identified and Resolved

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Invalid sampling (confidence=0.0) could be accepted at tau=0.0 since 0.0 >= 0.0 | Critical | Fixed |
| 2 | Uncertain claims could silently appear as ordinary confident medical findings | High | Fixed |

---

## 2. Issue 1 — Invalid Sampling Must Always Abstain

### 2.1 Root Cause

`compute_confidence()` returned `0.0` for invalid sampling (`sampling_valid=False`). However, `abstain_decision(0.0, 0.0)` returned `False` (accept) because `0.0 >= 0.0`. This meant invalid sampling at `tau=0.0` would be accepted, presenting placeholder agreement/entropy values as a valid finding.

### 2.2 Fix

Added `sampling_valid: bool = True` parameter to `abstain_decision()`:

```python
def abstain_decision(confidence, threshold, sampling_valid=True) -> bool:
    if not sampling_valid:
        return True  # always abstain
    return confidence < threshold
```

Sampling validity is now an explicit condition in the runtime abstention decision, not merely a confidence value of 0.0.

### 2.3 Behavior

| Condition | tau=0.0 | tau=0.5 | tau=1.0 |
|-----------|---------|---------|---------|
| Invalid sampling | **Abstain** | **Abstain** | **Abstain** |
| Valid sampling, c=0.0 | Accept (c >= 0.0) | Abstain | Abstain |
| Valid sampling, c=1.0 | Accept | Accept | Accept (c >= 1.0) |

### 2.4 Tests Added

- `test_invalid_sampling_abstains_at_tau_zero` — invalid sampling, tau=0.0 → abstain
- `test_invalid_sampling_abstains_at_tau_0_5` — invalid sampling, tau=0.5 → abstain
- `test_valid_unanimous_accepts_at_tau_zero` — valid sampling, tau=0.0 → accept (boundary preserved)
- `test_missing_sampling_always_abstains` — missing sampling (n_samples=0), all tau values → abstain
- `test_malformed_sampling_always_abstains` — malformed sampling (placeholders), all tau values → abstain
- `test_selective_prediction_invalid_sampling_abstains` — `make_selective_prediction()` with invalid sampling → abstain even at tau=0.0

---

## 3. Issue 2 — Uncertain Claim Behavior

### 3.1 Decision

**Uncertain claims always cause abstention.** They are explicitly flagged with `is_uncertain=True` on the `SelectivePrediction` so they never silently appear as ordinary confident medical findings.

### 3.2 Fix

Added `is_uncertain: bool = False` field to `SelectivePrediction`. `make_selective_prediction()` now takes `evidence_state` and `sampling_valid` parameters:

- `sampling_valid=False`: always abstain, `is_uncertain=False`
- `evidence_state="uncertain"`: always abstain, `is_uncertain=True`, `final_answer=None`
- Otherwise: normal threshold-based abstention

### 3.3 Behavior

| Evidence State | Confidence | Threshold | is_abstaining | is_uncertain | final_answer |
|---------------|------------|-----------|---------------|--------------|--------------|
| present_grounded | 0.9 | 0.5 | False | False | "Yes" |
| present_grounded | 0.3 | 0.5 | True | False | None |
| absent | 0.8 | 0.0 | False | False | "No" |
| uncertain | 1.0 | 0.0 | **True** | **True** | None |
| uncertain | 1.0 | 1.0 | **True** | **True** | None |
| present_ungrounded | 0.6 | 0.5 | False | False | "Yes" |

### 3.4 Tests Added

- `test_uncertain_claim_always_abstains` — uncertain claim with perfect confidence at tau=0.0 still abstains
- `test_uncertain_claim_explicitly_flagged` — uncertain claim has `is_uncertain=True`
- `test_present_grounded_not_affected_by_uncertain_rule` — present grounded claims follow normal threshold behavior

---

## 3. Files Changed

| File | Change |
|------|--------|
| `verification/confidence_gate.py` | Added `sampling_valid` param to `abstain_decision()` (always abstains when False); added `evidence_state` and `sampling_valid` params to `make_selective_prediction()`; added `is_uncertain` field to `SelectivePrediction`; uncertain claims always abstain |
| `tests/unit/test_uncertainty_phase2.py` | 9 new tests (5 invalid sampling, 4 uncertain claim) |

---

## 4. Verification Results

```
# Phase 2 tests (51 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py -v
=> 51 passed

# Phase 2 + Phase 1 + baseline comparison + schema regression tests (all pass)
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 118 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check verification/confidence_gate.py tests/unit/test_uncertainty_phase2.py
=> All checks passed!
```

---

## 5. Phase 3 Not Started

**Confirmed:** Phase 3 (aggregate evaluation metrics, evaluation harness), Phase 4 (full pipeline integration, experiment tracks, API), and Phase 5 (validation experiments, research-result reporting) were **not** implemented. No code for those phases exists.

---

## 6. Completion Criteria

| Criterion | Status |
|-----------|--------|
| Invalid sampling always abstains, even at tau=0.0 | ✅ `abstain_decision()` checks `sampling_valid` explicitly |
| Valid sampling boundary behavior preserved at tau=0.0 | ✅ `test_valid_unanimous_accepts_at_tau_zero` |
| Missing/malformed sampling always abstains | ✅ `test_missing_sampling_always_abstains`, `test_malformed_sampling_always_abstains` |
| Uncertain claims always abstain | ✅ `test_uncertain_claim_always_abstains` |
| Uncertain claims explicitly flagged | ✅ `is_uncertain=True` on `SelectivePrediction` |
| Present grounded claims unaffected by uncertain rule | ✅ `test_present_grounded_not_affected_by_uncertain_rule` |
| All regressions pass | ✅ 118 tests pass |
| Ruff clean | ✅ All checks passed |
| Phase 3 not started | ✅ Confirmed |
