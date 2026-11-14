# FS-005 Phase 3 Documentation Reconciliation Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 3 — Evaluation Metrics and Artifact Generation  
**Document ID:** `docs/reports/FS-005_PHASE_3_DOCUMENTATION_RECONCILIATION.md`  
**Status:** Documentation reconciliation complete  
**Scope:** Documentation-only corrections — no Python code, tests, or experimental artifacts changed

---

## 1. Files Inspected

| File | Role |
|------|------|
| `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md` | Technical design — AUROC definition (Section 6) |
| `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md` | Phase 3 implementation report |
| `docs/reports/FS-005_PHASE_3_CORRECTIVE_REVIEW.md` | Phase 3 corrective review |
| `docs/reports/FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md` | Phase 3 final closure report |

---

## 2. Inconsistencies Found and Corrected

### Issue 1 — AUROC "Returns 0.5" Contradiction in Implementation Report

**File:** `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md`  
**Location:** Section 3.11 (AUROC), line 123

**Problem:** The implementation report stated:
> "Returns 0.5 if only one correctness class is present in the eligible population. Returns None if there are no eligible samples or only one correctness class."

This is self-contradictory: it says both `0.5` and `None` for the same condition (only one correctness class). The correct behavior, as implemented in `evaluation/metrics.py` and as defined in the technical design, is `None`.

**Correction:** Removed the "Returns 0.5 if only one correctness class is present" clause. The report now states:
> "Returns None if there are no eligible samples or only one correctness class."

This matches the technical design Section 6 and the corrective review.

---

### Issue 2 — Test Count Mismatch in Implementation Report

**File:** `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md`

**Problems found:**

| Location | Stated | Actual | Correction |
|----------|--------|--------|------------|
| Line 48: Test file description | "62 deterministic unit tests" | 63 | Changed to "63 deterministic unit tests" |
| Category table: Accuracy count | 5 | 9 | Changed to 9 |
| Category table: USR count | 9 | 11 | Changed to 11 |
| Category table: Pointing accuracy count | 8 | 8 | Unchanged (correct) |
| Category table: Sampling/entropy count | 3 | 4 | Changed to 4 |
| Category table: ECE count | 5 | 5 | Unchanged (correct) |
| Category table: Sel Acc/Coverage count | 5 | 5 | Unchanged (correct) |
| Category table: AUROC count | 6 | 5 | Changed to 5 |
| Category table: A-v-C curve count | 9 | 7 | Changed to 7 |
| Category table: Aggregate metrics count | 2 | 2 | Unchanged (correct) |
| Category table: Artifact generation count | 4 | 4 | Unchanged (correct) |
| Category table: Absence/uncertain count | 2 | 0 | Removed (folded into USR and Accuracy) |
| Category table: GT isolation count | 2 | 2 | Unchanged (correct) |
| Category table: Total | 63 | 63 | Unchanged (was already correct despite category errors) |
| Line 269: Phase 3 test run | "=> 63 passed" | 63 passed | Unchanged (correct) |
| Line 273: Combined test run | "=> 212 passed" | 181 passed | Changed to "=> 181 passed" |
| Line 331: Completion criteria | "62 Phase 3 tests pass; 211 total" | 63 Phase 3 tests pass; 181 total | Changed accordingly |

---

### Issue 3 — Test Count Mismatch in Corrective Review

**File:** `docs/reports/FS-005_PHASE_3_CORRECTIVE_REVIEW.md`

**Problems found:**

| Location | Stated | Actual | Correction |
|----------|--------|--------|------------|
| Line 182: Test count header | "62 Phase 3 tests" | 63 | Changed to "63 Phase 3 tests" |
| Category table: Accuracy count | 5 | 9 | Changed to 9 |
| Category table: USR count | 9 | 11 | Changed to 11 |
| Category table: Pointing accuracy count | 7 | 8 | Changed to 8 |
| Category table: Sampling/entropy count | 3 | 4 | Changed to 4 |
| Category table: AUROC count | 6 | 5 | Changed to 5 |
| Category table: A-v-C curve count | 9 | 7 | Changed to 7 |
| Category table: Absence/uncertain count | 2 | 0 | Removed (folded into USR/Accuracy) |
| Category table: Total | 62 | 63 | Changed to 63 |
| Line 208: Combined test run | "=> 212 passed" | 181 passed | Changed to "=> 181 passed" |

---

### Issue 4 — Test Count Mismatch in Closure Report

**File:** `docs/reports/FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md`

**Problems found:**

| Location | Stated | Actual | Correction |
|----------|--------|--------|------------|
| Line 65: Combined test count | "Total: 212 passed" | 181 passed | Changed to "Total: 181 passed" |

---

## 3. Final Test Counts

| Test file | Count |
|-----------|-------|
| `tests/unit/test_evaluation_metrics_phase3.py` | 63 |
| `tests/unit/test_uncertainty_phase2.py` | 51 |
| `tests/unit/test_claim_verification_phase1.py` | 47 |
| `tests/unit/test_baseline_comparison.py` | 16 |
| `tests/unit/test_schemas.py` | 4 |
| **Combined** | **181** |

**Verification:** These counts were confirmed by running `pytest --collect-only` on each file individually and running all five files together (`181 passed`).

---

## 4. Final Category Breakdown (63 Phase 3 Tests)

| Category | Count |
|----------|-------|
| Accuracy (overall, OPEN/CLOSED, empty, abstained) | 9 |
| USR_raw / USR_filtered (incl. None denominator cases) | 11 |
| Pointing accuracy (incl. multiple predictions, multiple GT, same-sample multiple claims) | 8 |
| Sampling agreement / normalized entropy | 4 |
| ECE (perfectly calibrated, no accepted, empty, bin boundary, max confidence) | 5 |
| Selective accuracy / coverage | 5 |
| AUROC (incl. threshold independence, None cases) | 5 |
| Accuracy-vs-coverage curve (incl. forced-abstention) | 7 |
| Aggregate metrics | 2 |
| Artifact generation (all files, JSON structure, CSV structure, report MD) | 4 |
| GT isolation (no modification of inputs, no GT in confidence paths) | 2 |
| **Total** | **63** |

---

## 5. Final AUROC Definition (Consistent Across All Documents)

**Population:** All eligible samples — `sampling_valid == True` AND `evidence_state != "uncertain"`. Independent of the abstention threshold $\tau$.

**ROC-AUC:** Mann-Whitney U statistic of confidence $c$ predicting binary correctness.

**Return values:**
- `None` if there are no eligible samples
- `None` if only one correctness class is present in the eligible population
- Mann-Whitney U statistic in [0, 1] otherwise

**Rationale:** AUROC evaluates confidence-score ranking across eligible predictions. Restricting it to accepted samples ($c \ge \tau$) would make AUROC dependent on the abstention threshold, which is not a meaningful property of the confidence score itself.

**ECE definition:** Remains accepted-sample-only conditional calibration over [0,1] with M=10 equal-width bins. Abstained samples excluded.

---

## 6. Pointing Accuracy Limitation (Documented Consistently)

**Aggregation:** Sample-level — `pointing_correct=True` if at least one claim in the sample has max IoU >= 0.5 against any ground-truth region.

**Limitation:** Sample-level success can conceal incorrectly grounded additional claims within the same sample. If a sample has multiple claims where one meets IoU >= 0.5 against a GT region but another does not, `pointing_correct=True` at the sample level regardless of the incorrectly grounded claim. Sample-level pointing accuracy does not capture per-claim grounding errors.

This limitation is documented consistently in:
- `FS-005_PHASE_3_IMPLEMENTATION_REPORT.md` (Section 3.5)
- `FS-005_PHASE_3_CORRECTIVE_REVIEW.md` (Section 5.3)
- `FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md` (Section 3)

---

## 7. Python Code Status

**No Python code files were modified.** No tests were changed. No experimental artifacts were modified. The only changes were to documentation files (`.md` reports).

---

## 8. Phase Status

- **FS-005 Phase 3:** Complete
- **FS-005 Phase 4:** Not started
- **FS-005 Phase 5:** Not started
- **Model inference:** Not executed
- **FS-001–FS-004:** Not modified
