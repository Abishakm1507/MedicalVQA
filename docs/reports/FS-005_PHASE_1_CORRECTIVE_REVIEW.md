# FS-005 Phase 1 Corrective Review — Claim Extraction and Region Verification

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 1 — Claim Extraction and Region Verification  
**Document ID:** `docs/reports/FS-005_PHASE_1_CORRECTIVE_REVIEW.md`  
**Status:** Corrective review complete — all issues resolved  
**Review type:** Independent senior ML engineering and medical AI research methodology audit  
**Implementation reviewed:** `verification/claim_extractor.py`, `verification/claim_associator.py`, `verification/evidence_verifier.py`, `verification/claim_filter.py`, `tests/unit/test_claim_verification_phase1.py`

---

## 1. Issues Identified

| # | Issue | Severity | Affected Files |
|---|-------|----------|----------------|
| 1 | **Ground-truth leakage**: `verify_evidence()` took `ground_truth_regions` and used GT IoU to determine `is_supported`, meaning GT annotations directly influenced runtime claim filtering outputs. | Critical | `verification/evidence_verifier.py`, `verification/claim_filter.py` |
| 2 | **Incorrect negative-claim classification**: Absence was classified as `status == "absent" OR region_id == "NONE"`, incorrectly treating `status="present"` + `region_id="NONE"` as absence. | Critical | `verification/evidence_verifier.py`, `verification/claim_filter.py` |
| 3 | **Structural validity conflated with evidence support**: `is_supported = is_grounded` for absence claims represented a structurally valid absence claim as clinically supported. | High | `verification/evidence_verifier.py` |
| 4 | **USR/filtering depended on GT-derived IoU**: `filter_claims` used `is_supported` which was GT-IoU-based, violating the prediction-time/evaluation-time separation. | Critical | `verification/claim_filter.py` |

---

## 2. Root Cause

The original implementation combined prediction-time claim verification with evaluation-time ground-truth IoU comparison in a single `verify_evidence()` function. This meant:

- `verify_associated` (renamed from `verify_evidence`) required `ground_truth_regions` as input
- `is_supported` was computed as `max_iou >= 0.5` against GT boxes
- `filter_claims` consumed this GT-derived `is_supported` to classify claims and compute USR

This violated the fundamental separation required by the research design: prediction-time verification must not access ground-truth annotations, and GT IoU belongs exclusively to the evaluation layer.

Additionally, absence-claim identification used a disjunctive condition (`status == "absent" OR region_id == "NONE"`) that conflated two distinct situations:
- `status="absent"` + `region_id="NONE"` → genuine absence claim
- `status="present"` + `region_id="NONE"` → positive claim claiming presence but citing no region (ungrounded)

These must be classified differently per ADR 0004 Option 3.

---

## 3. Prediction-Time vs Evaluation-Time Architecture

### Separation established

| Layer | Function | Inputs | Ground-truth access? | Purpose |
|-------|----------|--------|---------------------|---------|
| **Prediction-time** | `verify_association(associated_claims)` | Claims + candidate regions only | **No** | Structural validity: does the claim cite a valid region reference? |
| **Evaluation-time** | `evaluate_grounding_iou_with_boxes(associated_claims, ground_truth_regions)` | Claims + GT boxes | **Yes** | IoU ≥ 0.5 comparison against GT boxes for metrics |

### Data flow

```
VLM Structured Output
     │
     ▼
extract_claims() → ExtractedClaim[]
     │
     ▼
associate_claims() → AssociatedClaim[]  (structural association, no GT)
     │
     ├──▶ verify_association() → VerifiedClaim[]   (runtime: iou_with_ground_truth=None)
     │                                  │
     │                                  ▼
     │                              filter_claims() → FilterResult  (structural USR)
     │
     └──▶ [EVALUATION LAYER ONLY]
            evaluate_grounding_iou_with_boxes() → GroundingEvaluation[]
            (IoU ≥ 0.5 vs GT; does NOT modify runtime VerifiedClaim)
```

### Key invariant

`verify_association()` and `filter_claims()` accept **no ground-truth parameters**. GT IoU is computed exclusively by `evaluate_grounding_iou_with_boxes()`, which returns a separate `GroundingEvaluation` type and does not modify runtime `VerifiedClaim.is_supported`.

The IoU ≥ 0.5 threshold is preserved in the evaluation layer, consistent with the approved FS-005 research design.

---

## 4. Negative-Claim Classification Corrections

### Classification rule (status-based, not region_id-based)

| `status` | `region_id` | Classification | `is_absence_claim` | `verification_path` |
|----------|-------------|----------------|--------------------|---------------------|
| `"absent"` | `"NONE"` | Absence/negative claim | `True` | `"absence_answer"` |
| `"present"` | `"NONE"` | Positive but ungrounded claim | `False` | `"positive_grounding"` |
| `"uncertain"` | any | Uncertain claim (distinct) | `False` | `"uncertain"` |
| `"present"` | `"R1"`…`"Rk"` | Positive grounded claim | `False` | `"positive_grounding"` |

**`region_id == "NONE"` does NOT imply absence.** Only `status == "absent"` identifies an absence claim. A present claim citing NONE is a positive claim without a region box — structurally ungrounded as a positive claim, but not an absence claim.

### Implementation

`_classify_claim(status)` in `evidence_verifier.py` returns `(is_absence, is_uncertain, verification_path)` based solely on `status`. `verify_association()` routes each claim accordingly:
- Absence → `verification_path="absence_answer"`, `localizer_score=None`, no positive-region verification
- Uncertain → `verification_path="uncertain"`, distinct meaning preserved
- Positive → structural validation against R; `is_supported=True` only when a cited box exists

---

## 5. Structural Validity vs Evidence Support

### Data model distinctions

| Concept | Field | Meaning | Populated at |
|---------|-------|---------|--------------|
| Structural validity of claim | `is_grounded` | region_id names a valid reference (in R, or NONE) | Runtime (associator) |
| Validity of cited region association | `is_supported` | Positive claim has a cited box; absence/uncertain claims structural only | Runtime (verify_association) |
| Positive-region evidence support | `meets_evaluation_threshold` (in `GroundingEvaluation`) | IoU ≥ 0.5 vs GT boxes | Evaluation only |
| Negative-answer correctness | Deferred to evaluation layer | Answer-level GT comparison, not region grounding | Evaluation layer (Phase 3+) |

### Key guarantees

- `is_supported` at runtime reflects **structural validity only** — never clinical evidence support.
- `iou_with_ground_truth` is `None` at runtime; populated only by `evaluate_grounding_iou_with_boxes()`.
- Absence claims are **not** represented as clinically supported findings. They are structurally valid (NONE is a valid token) but carry no localization confidence and no evidence support.
- Empty detections (`R = ∅`) never produce a verified negative finding. An absence claim with empty R has `localizer_score=None` and `iou_with_ground_truth=None`.

---

## 6. USR / Filtering Decisions

### Runtime filtering (structural, GT-free)

`filter_claims()` classifies based on `VerifiedClaim` fields set by `verify_association()`:
- Absence claims (`status == "absent"`) → `absence_claims`, excluded from USR denominators (ADR 0004 Option 3)
- Uncertain claims (`status == "uncertain"`) → `uncertain_claims`, excluded from USR denominators
- Positive claims → `supported`/`unsupported`/`unverifiable` based on structural validity

### Structural USR metrics

These are **runtime structural USR** — computed from structural validity, not GT IoU:

- `usr_raw = (unsupported + unverifiable) / n_positive`
- `usr_filtered = unsupported / (n_positive − unverifiable)`

Where `n_positive` excludes absence and uncertain claims.

### Evaluation-layer USR (deferred)

The evaluation layer (Phase 3+) will compute IoU-based USR separately using `evaluate_grounding_iou_with_boxes()`, where "supported" means IoU ≥ 0.5 against GT boxes. This is explicitly not part of Phase 1.

### Decision documented

USR calculations remain in `claim_filter.py` as **structural USR** for runtime filtering. They are clearly labeled as structural, not GT-IoU-based. The evaluation layer will produce separate IoU-based USR metrics. No additional evaluation metrics were introduced; Phase 1 scope is preserved.

---

## 7. Files Created and Modified

### Created

| File | Explanation |
|------|-------------|
| `verification/claim_extractor.py` | Claim extraction (unchanged from initial implementation) |
| `verification/claim_associator.py` | Claim-to-region association (unchanged from initial implementation) |
| `verification/evidence_verifier.py` | **Rewritten**: split into `verify_association` (runtime, GT-free) and `evaluate_grounding_iou_with_boxes` (evaluation-only, GT-based); added `GroundingEvaluation` type; corrected absence/uncertain classification |
| `verification/claim_filter.py` | **Rewritten**: structural filtering only, added `uncertain_claims`, USR over positive claims only |
| `tests/unit/test_claim_verification_phase1.py` | **Rewritten**: 47 deterministic tests covering all corrective invariants |

### Modified

| File | Explanation |
|------|-------------|
| `docs/architecture/adr/0004-negative-claim-handling.md` | Status → Accepted (Option 3); decision rationale recorded (pre-existing, from initial implementation) |
| `docs/reports/FS-005_PHASE_1_IMPLEMENTATION_REPORT.md` | Updated to reflect corrected architecture (runtime/evaluation separation, status-based classification, structural USR) |

### Not modified

- `core/schemas.py`, `vlm/base.py`, `localization/detector.py`, `localization/postprocessing.py` (reused unchanged)
- All FS-001–FS-004 code and experimental artifacts
- Original baseline results under `experiments/runs/`

---

## 8. Tests Added and Actual Results

### Test file: `tests/unit/test_claim_verification_phase1.py`

**47 deterministic tests** covering:

| Category | Count | Key invariants verified |
|----------|-------|------------------------|
| Claim extraction | 7 | Valid/multiple, positive, negative, empty, malformed, answer preservation, empty findings |
| Claim-to-region association | 6 | Valid region, multiple regions, unknown region, empty collection, NONE-with-empty-R, provenance |
| Negative-claim classification | 6 | Present+valid region≠absence, present+NONE≠absence, absent+NONE=absence, absent+empty R, uncertain preserved, mixed claims |
| Runtime GT-independence | 4 | verify_association takes no GT, filter_claims takes no GT, runtime iou=None, structural≠clinical |
| Evaluation-time IoU isolation | 4 | Evaluation takes GT, IoU≥0.5 threshold, IoU<0.5 not met, isolation from runtime, absence not evaluated vs GT |
| compute_iou utility | 4 | Overlapping, no overlap, identical, contained |
| Unsupported claim filtering | 6 | Supported retained, unsupported filtered, absence excluded from USR, mixed, uncertain separated, empty |
| ADR 0004 Option 3 | 8 | All Option 3 behaviors including negative≠unsupported-positive, empty detections don't prove absence |

### Commands executed and results

```
# New Phase 1 tests (47 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_claim_verification_phase1.py -v
=> 47 passed

# Phase 1 + baseline comparison + schema regression tests
.venv\Scripts\python.exe -m pytest tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 57 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check verification/ tests/unit/test_claim_verification_phase1.py
=> All checks passed!

# Type check: mypy not installed in this environment; ruff is the configured linter.
```

### Pre-existing failures (unrelated to this corrective review; not regressions)

Running the full `tests/unit/` suite produces 4 pre-existing environment-related failures:
- `test_environment.py::test_python_version` (expects Python 3.11, environment is 3.12.2)
- `test_medsam_integration.py::test_real_single_region` (torch TORCH_LIBRARY conflict)
- `test_medsam_integration.py::test_real_multiple_regions_order` (same torch issue)
- `test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` (torch import failure)

None of these are related to Phase 1 changes. No unrelated tests were modified or suppressed.

---

## 9. Ruff and Type-Check Results

- **Ruff**: All changed Python files (`verification/*.py`, `tests/unit/test_claim_verification_phase1.py`) pass `ruff check` cleanly.
- **Mypy**: Not installed in this environment; ruff is the project's configured linter.
- **Pre-existing ruff findings** in unrelated files (`experiments/`, `inspect_slake.py`, `test_medsam_*`, etc.) were left untouched per the no-unrelated-refactoring constraint.

---

## 10. Remaining Limitations

1. **USR metrics are structural, not IoU-based.** The evaluation layer (Phase 3+) will compute IoU-based USR using `evaluate_grounding_iou_with_boxes()`. Phase 1 produces structural USR only.
2. **Negative-claim clinical truth is deferred.** Absence claims are marked `verification_path="absence_answer"` with `localizer_score=None`; their correctness against dataset ground-truth answers is evaluated in a later phase, consistent with ADR 0004 Option 3.
3. **No clinical validity claim.** This is technical verification only; no clinical diagnosis or validation is implied.
4. **Pre-existing environment test failures** (Python version, torch, MedSAM checkpoint) are unrelated to Phase 1.

---

## 11. Phase 2 Not Started

**Confirmed:** Phase 2 (uncertainty aggregation, confidence calibration, abstention, selective prediction, risk-coverage evaluation, ECE, semantic agreement, entropy) was **not** implemented. No Phase 2 code exists. Phase 1 is the claim-verification foundation only.

---

## 12. Completion Criteria Confirmation

| Criterion | Status |
|-----------|--------|
| GT annotations cannot influence prediction-time filtering | ✅ `verify_association` and `filter_claims` accept no GT parameters |
| IoU evaluation isolated from runtime verification | ✅ `evaluate_grounding_iou_with_boxes` is evaluation-only, separate `GroundingEvaluation` type |
| `region_id="NONE"` does not automatically imply absence | ✅ Classification is status-based; `present+NONE` ≠ absence |
| Negative claims distinct from ungrounded positive claims | ✅ `absent+NONE` → `absence_answer`; `present+NONE` → `positive_grounding`, `is_supported=False` |
| Structural validity not presented as clinical evidence support | ✅ `is_supported` = structural at runtime; IoU ≥ 0.5 = evaluation-only |
| ADR 0004 Option 3 remains intact | ✅ All 7 Option 3 behaviors implemented and tested |
| Relevant tests and regression checks executed | ✅ 47 Phase 1 tests pass; 57 including regressions; ruff clean |
| Corrective review report complete | ✅ This document |

**Phase 1 corrective review is complete.** Awaiting user review and approval before proceeding to Phase 2.
