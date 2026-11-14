# FS-005 Phase 2 Implementation Report — Uncertainty Estimation and Abstention Gate

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 2 — Uncertainty Estimation and Abstention Gate  
**Document ID:** `docs/reports/FS-005_PHASE_2_IMPLEMENTATION_REPORT.md`  
**Status:** Implementation complete — Phase 2 only (after corrective review)  
**Implementation type:** Implementation (code written), not design-only  
**Prerequisite:** FS-005 Phase 1 (claim extraction and region verification) approved  
**Corrective review:** `docs/reports/FS-005_PHASE_2_CORRECTIVE_REVIEW.md`

---

## 1. Scope and Implementation Summary

Implement FS-005 Phase 2: uncertainty estimation and the abstention/selective-answering gate. This phase builds on the Phase 1 claim-verification foundation and implements the confidence gate from ADR 0005:

    c = w1 * a + w2 * (1 - H_n) + w3 * rho

where `a` = semantic agreement, `H_n` = normalized semantic entropy, and `rho` = localizer score of the cited region.

**In scope (this phase):**
- Semantic agreement estimation across repeated model samples (N=5 per ADR 0005)
- Semantic answer clustering (embedding-based primary, canonical fallback)
- Normalized semantic entropy calculation (H_n = H / ln(N))
- Confidence estimation using Phase 1 verification outputs, with evidence-state-aware policies
- Confidence calibration interfaces and configuration (no hardcoded weights; selective risk primary objective)
- Selective prediction and abstention gate (runtime-only, no ground-truth inputs)
- ADR 0004 Option 3 handling (absence claims get no artificial localization confidence)
- Invalid sampling detection forcing abstention

**Out of scope (deferred to Phase 3+):**
- Aggregate evaluation metrics (USR, pointing accuracy, ECE, selective accuracy, coverage curves)
- Evaluation harness and experiment artifact generation
- Full pipeline integration, experiment tracks (A/B/C), or API integration
- Validation experiments or research-result reporting
- New model inference or benchmark experiments
- Changes to FS-001 through FS-004

---

## 2. Design Decisions and Correspondence with Technical Design

### 2.1 Semantic Clustering (Section 4.5 of design)

| Design requirement | Implementation | Status |
|-------------------|----------------|--------|
| Kuhn et al. (2023) semantic clustering | Greedy threshold-based connected-component clustering | Implemented |
| Embedding-based primary (`all-MiniLM-L6-v2`) | `SemanticClusterer` with sentence-transformers backend | Implemented (graceful fallback) |
| Canonical entity normalization fallback | Exact-match on canonical normalization | Implemented |
| Configurable cosine threshold | `similarity_threshold` parameter on `SemanticClusterer` | Implemented |
| Threshold calibrated on validation (not hardcoded) | No default threshold claimed as validated | Documented |
| N=5 standard per ADR 0005 | `n_samples=5` default in `ConfidenceConfig` | Implemented |
| Invalid sampling flagged | `is_valid` field on `AgreementEntropyResult` | Implemented (corrective review) |
| Clustering transitivity documented | Docstring on `cluster()` method | Implemented (corrective review) |

### 2.2 Confidence Gate (Sections 4.6–4.7 of design)

| Design requirement | Implementation | Status |
|-------------------|----------------|--------|
| c = w1*a + w2*(1-H_n) + w3*rho | `compute_confidence()` function | Implemented |
| Weights sum to 1, w_i >= 0 | `ConfidenceConfig.__post_init__` validation | Implemented |
| No hardcoded weights/tau | All parameters configurable via `ConfidenceConfig` | Implemented |
| rho=None rescaling (absence claims) | Rescaled formula for `absent`/`uncertain` states | Implemented |
| Present+ungrounded not rescaled | `present_ungrounded` state: rho=0.0, no rescaling | Implemented (corrective review) |
| Invalid sampling → abstention | `sampling_valid=False` returns c=0.0 AND `abstain_decision()` explicitly checks `sampling_valid` (always abstains when False, even at tau=0.0) | Implemented (corrective review) |
| Abstention: accept if c >= tau | `abstain_decision()` function with `sampling_valid` parameter | Implemented (corrective review) |
| Selective prediction (no is_correct) | `make_selective_prediction()` without correctness; uncertain claims always abstain with `is_uncertain=True` flag | Implemented (corrective review) |
| Calibration: selective risk primary | `ConfidenceCalibrator` minimizes selective risk | Implemented (corrective review) |
| Calibration: AUROC secondary, ECE diagnostic | Both reported in `CalibrationResult` | Implemented (corrective review) |
| Coverage constraint | `min_coverage` prevents all-abstain trivial solution | Implemented (corrective review) |

### 2.3 ADR 0004 Option 3 Compliance

| Requirement | Implementation |
|-------------|----------------|
| Negative claims preserved as valid answer claims | `is_absence_claim` flag on Phase 1 `VerifiedClaim` |
| Absence claims evaluated via answer-level GT, not region grounding | Confidence gate uses rho=None for absence; no IoU-based support |
| Absence claims excluded from positive-region grounding metrics | Confidence calculation uses rescaled formula (rho term omitted) |
| No artificial localization confidence to absence claims | `localizer_score=None` for absence claims; not injected in confidence |
| Empty detections never prove clinical absence | No inference from empty R to absence claim truth; invalid sampling → abstention |
| Present+ungrounded (region_id="NONE") not treated as absence | `present_ungrounded` evidence state: rho=0.0, no rescaling |

---

## 3. Files Created and Modified

### Created

No new files created in corrective review. All Phase 2 modules were created in the initial implementation and corrected in place.

### Modified

| File | Change |
|------|--------|
| `verification/semantic_clustering.py` | Added `is_valid` field to `AgreementEntropyResult` (derived via `__post_init__`); added transitivity documentation to `cluster()` docstring |
| `verification/confidence_gate.py` | Added `EvidenceState` Literal type; added `evidence_state` and `sampling_valid` parameters to `compute_confidence()`; added `sampling_valid` parameter to `abstain_decision()` (always abstains when False); added `evidence_state` and `sampling_valid` parameters to `make_selective_prediction()`; added `is_uncertain` field to `SelectivePrediction`; uncertain claims always abstain; removed `is_correct` from `SelectivePrediction` and `make_selective_prediction()`; changed calibration objective from AUROC to selective risk with coverage constraint; added `selective_risk` and `coverage` fields to `CalibrationResult` |
| `tests/unit/test_uncertainty_phase2.py` | 51 tests covering all 8 corrective issues; removed `is_correct` from selective prediction tests; added evidence state, invalid sampling (incl. tau=0.0), calibration objective, transitivity, uncertain-claim handling, and GT isolation tests |
| `docs/reports/FS-005_PHASE_2_CORRECTIVE_REVIEW.md` | Created — corrective review report documenting all 8 issues and resolutions |
| `docs/reports/FS-005_PHASE_2_FINAL_EDGE_CASE_REVIEW.md` | Created — addendum documenting invalid-sampling-at-tau-zero and uncertain-claim fixes |

### Not modified
- `core/schemas.py` (reused unchanged — `ConfidenceGateResult` dataclass exists but Phase 2 uses its own `SelectivePrediction`)
- All FS-001–FS-004 code and experimental artifacts

---

## 4. Semantic Clustering and Entropy Methodology

### 4.1 Clustering backends

`SemanticClusterer` supports two backends:

1. **Embedding backend (primary):** Uses `sentence-transformers` (recommended: `all-MiniLM-L6-v2`) to compute sentence embeddings, then cosine-similarity threshold clustering. If `sentence-transformers` is not installed or initialization fails, automatically falls back to backend 2.

2. **Canonical fallback (deterministic):** Normalizes answers (lowercase, strip, collapse whitespace) and clusters exact matches. No external dependencies; fully reproducible.

The backend is auto-detected at initialization and exposed via `clusterer.backend_name`.

### 4.2 Clustering algorithm

Greedy threshold-based connected-component clustering:
1. Compute pairwise similarity matrix (cosine for embeddings, exact-match for fallback)
2. For each unvisited answer, BFS/DFS to find all answers connected via similarity >= threshold
3. Each connected component forms one cluster

### 4.3 Transitivity behavior (documented)

Connected-component clustering is transitive: if answer A is similar to B and B is similar to C, then A, B, and C are placed in the same cluster even if A is not directly similar to C. This is documented in the `cluster()` docstring and tested by `test_clustering_transitive_chaining`.

For medical VQA answers this is generally acceptable because clinical paraphrases tend to cluster tightly. Users requiring strict pairwise equivalence should use a different algorithm.

### 4.4 Agreement and entropy computation

Given N sampled answers partitioned into K clusters {C_1 … C_K}:

- Cluster probability: p_k = |C_k| / N
- Agreement: a = max_k p_k (modal cluster proportion)
- Raw entropy: H = -sum p_k * ln(p_k) (0 * ln(0) = 0)
- Normalized entropy: H_n = H / ln(N) in [0, 1]

Edge cases:
- N=0: `is_valid=False`, error="no_samples", agreement=0.0, H_n=0.0 (placeholders only)
- N=1: H_n=0.0 (no uncertainty possible with single sample)
- Unanimous (K=1): H_n=0.0 (legitimate zero entropy)
- Uniform (K=N, equal sizes): H_n=1.0

### 4.5 Invalid sampling flag

`AgreementEntropyResult.is_valid` is True only when `n_samples > 0 and error is None`. Invalid results (empty/missing/malformed sampling) have `is_valid=False` and must NOT be fed into confidence calculation — the runtime gate must abstain instead.

---

## 5. Confidence Calculation and Calibration Status

### 5.1 Confidence calculation

`compute_confidence(agreement, normalized_entropy, localizer_score, config, evidence_state, sampling_valid)` implements:

    c = w1 * a + w2 * (1 - H_n) + w3 * rho

**Evidence-state policies:**

| Evidence State | Condition | Formula | rho treatment |
|---------------|-----------|---------|---------------|
| `present_grounded` | status="present", region_id in R | Full: w1*a + w2*(1-H_n) + w3*rho | Actual rho |
| `absent` | status="absent", region_id="NONE" | Rescaled: (w1*a + w2*(1-H_n))/(w1+w2) | Omitted (None) |
| `uncertain` | status="uncertain" | Rescaled: (w1*a + w2*(1-H_n))/(w1+w2) | Omitted (None) |
| `present_ungrounded` | status="present", region_id="NONE" | Full with rho=0.0: w1*a + w2*(1-H_n) + w3*0.0 | Zero (0.0), no rescaling |

**Invalid sampling:** When `sampling_valid=False`, returns 0.0 so the abstention gate abstains. No placeholder agreement/entropy values are ever converted into a usable confidence score.

### 5.2 Calibration status

**Calibration is pending.** The `ConfidenceCalibrator` class provides the grid-search interface (step 0.05 per ADR 0005) but ships with **no pre-fitted weights or threshold**. Calibration must be run on held-out SLAKE validation data to minimize selective risk (primary objective per ADR 0005). Selected weights and tau are frozen before touching the test set.

**Objective hierarchy:**
1. Primary: Minimize selective risk (`1 - selective_accuracy`) at operating coverage
2. Secondary: AUROC (diagnostic — confidence-score ranking quality)
3. Calibration: ECE (diagnostic — confidence accuracy alignment)

**Coverage constraint:** `min_coverage` (default 0.1) prevents trivial all-abstain solutions. Solutions with coverage < min_coverage are skipped during grid search.

No weights or thresholds are hardcoded as scientifically validated values. All parameters in `ConfidenceConfig` are configurable.

### 5.3 Sampling configuration (ADR 0005)

| Parameter | Value | Source |
|-----------|-------|--------|
| n_samples | 5 (default) | ADR 0005 benchmark standard |
| temperature | 0.7 (default) | FS-005 design section 3.1 |
| random_seed | None (configurable) | Recorded for reproducibility |

---

## 6. Abstention Behavior

### 6.1 Abstention decision

`abstain_decision(confidence, threshold)` returns `True` if c < tau (abstain), `False` if c >= tau (accept). Boundary: confidence exactly at threshold is accepted.

### 6.2 Selective prediction (runtime-only)

`make_selective_prediction()` produces a `SelectivePrediction` with:
- `sample_id`: Identifier for the sample
- `confidence`: Fused confidence c in [0, 1]
- `threshold`: tau used for the decision
- `is_abstaining`: True if c < tau
- `final_answer`: None if abstaining, otherwise the candidate answer
- `in_coverage`: True if not abstaining

**No `is_correct` field:** Ground-truth correctness is NOT part of runtime output. Correctness is computed separately in the offline evaluation layer.

### 6.3 Absent and uncertain claims

| Claim type | localizer_score (rho) | Confidence treatment |
|------------|----------------------|---------------------|
| Present (grounded) | float from CandidateBox.score | Full formula c = w1*a + w2*(1-H_n) + w3*rho |
| Absent (NONE) | None | Rescaled: (w1*a + w2*(1-H_n)) / (w1 + w2) |
| Uncertain | None | Rescaled formula (distinct state) |
| Present+ungrounded | None (treated as 0.0) | Full formula with rho=0.0, no rescaling |

Absence claims receive **no artificial localization confidence** — the rho term is omitted entirely. Present+ungrounded claims do NOT get the rescaling benefit — rho=0.0 in the full formula, which typically produces lower confidence than rescaling.

### 6.4 Abstention when signals unavailable

If required uncertainty signals are unavailable:
- Empty/missing/malformed sampling → `AgreementEntropyResult` with `is_valid=False`
- `compute_confidence` with `sampling_valid=False` → returns 0.0
- `abstain_decision()` with `sampling_valid=False` → **always abstains**, even at tau=0.0 (sampling validity is an explicit condition, not just confidence=0.0)
- `make_selective_prediction()` with `sampling_valid=False` → always abstains, `is_uncertain=False`

### 6.5 Uncertain claim handling

Uncertain claims (`evidence_state="uncertain"`) always abstain and are explicitly flagged:
- `make_selective_prediction()` with `evidence_state="uncertain"` → `is_abstaining=True`, `is_uncertain=True`, `final_answer=None`, `in_coverage=False`
- Uncertain claims never silently appear as ordinary confident medical findings, regardless of confidence or threshold
- Present grounded claims follow normal threshold-based abstention and are unaffected by the uncertain-claim rule

---

## 7. Ground-Truth Isolation

Ground-truth annotations have **zero influence** on runtime uncertainty estimation or abstention decisions:

| Function | GT parameter? | GT influence? |
|----------|--------------|---------------|
| `compute_confidence()` | No | None |
| `abstain_decision()` | No | None |
| `make_selective_prediction()` | No | None |
| `SemanticClusterer.cluster()` | No | None |
| `SemanticClusterer.compute_agreement_entropy()` | No | None |
| `ConfidenceCalibrator.calibrate()` | Yes (correctness labels) | Yes — but ONLY on held-out validation data |

GT bounding boxes are used **only** in the evaluation layer (Phase 1 `evaluate_grounding_iou_with_boxes`), which is explicitly separated from runtime decision-making.

---

## 8. Tests, Exact Commands, and Results

### Test file: `tests/unit/test_uncertainty_phase2.py`

**51 deterministic tests**, organized by non-overlapping category:

| Category | Count |
|----------|-------|
| Semantic clustering (incl. transitivity) | 9 |
| Invalid sampling → abstention (Issues 1 & 1b) | 11 |
| Evidence state policies (Issue 2) | 6 |
| Uncertain claim handling (Issue 2b) | 3 |
| Abstention decision | 3 |
| Selective prediction / no is_correct (Issue 4) | 3 |
| Threshold boundary | 2 |
| Calibration objective (Issue 3) | 5 |
| Ground-truth isolation | 3 |
| Phase 1 regression compatibility | 4 |
| N=5 sampling standard | 2 |
| **Total** | **51** |

### Commands executed and results

```
# Phase 2 tests (51 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py -v
=> 51 passed

# Phase 2 + Phase 1 + baseline comparison + schema regression tests
.venv\Scripts\python.exe -m pytest tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 118 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check verification/semantic_clustering.py verification/confidence_gate.py tests/unit/test_uncertainty_phase2.py
=> All checks passed!
```

### Pre-existing failures (unrelated to this phase)

Running the full `tests/unit/` suite produces 4 pre-existing environment-related failures:
- `test_environment.py::test_python_version` (expects Python 3.11, environment is 3.12.2)
- `test_medsam_integration.py::test_real_single_region` (torch TORCH_LIBRARY conflict)
- `test_medsam_integration.py::test_real_multiple_regions_order` (same torch issue)
- `test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` (torch import failure)

None are related to Phase 2 changes.

### Ruff on whole project
Phase 2 changed files pass `ruff check` cleanly. The 40 ruff findings project-wide are all in pre-existing unrelated files and were not introduced by this phase.

### Type checking
mypy is not installed in this environment; ruff is the project's configured linter.

---

## 9. Known Limitations and Unresolved Decisions

| Limitation / Decision | Status |
|----------------------|--------|
| Confidence weights and threshold not fitted | Calibration pending — must run `ConfidenceCalibrator.calibrate()` on held-out SLAKE validation data; no pre-fitted values shipped |
| Cosine similarity threshold not calibrated | Default 0.75 is not scientifically validated; must be tuned on validation data |
| Embedding backend not available | `sentence-transformers` not installed; falls back to canonical normalization |
| N=3 separate threshold (tau_N=3) | Not implemented; interface supports configurable `tau` |
| Calibration coverage constraint may yield no valid solution | Returns safe default (tau=1.0, all abstain) |
| No clinical validity claim | Technical verification only; no clinical diagnosis or validation implied |

---

## 10. Phase 3 and Later Not Started

**Confirmed:** Phase 3 (aggregate evaluation metrics, evaluation harness), Phase 4 (full pipeline integration, experiment tracks, API), and Phase 5 (validation experiments, research-result reporting) were **not** implemented. No code for those phases exists. Phase 2 is the uncertainty-estimation and abstention-gate foundation only.

---

## 11. Completion Criteria Confirmation

| Criterion | Status |
|-----------|--------|
| Uncertainty and abstention components implemented | ✅ `semantic_clustering.py` + `confidence_gate.py` |
| Runtime decisions independent of ground-truth data | ✅ No GT parameters in any runtime function |
| ADR 0004 Option 3 preserved | ✅ All Option 3 behaviors implemented and tested |
| Deterministic tests cover specified edge cases | ✅ 42 tests covering all 6 corrective issues |
| Regressions pass | ✅ 109 tests pass (Phase 2 + Phase 1 + baseline + schemas) |
| Corrective review completed | ✅ `docs/reports/FS-005_PHASE_2_CORRECTIVE_REVIEW.md` |
| Markdown report created | ✅ This document |

**Phase 2 is complete.** Awaiting review and approval of this implementation report before proceeding to Phase 3.
