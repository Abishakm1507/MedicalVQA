# FS-005 Phase 3 Final Count Reconciliation

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 3 — Evaluation Metrics and Artifact Generation  
**Document ID:** `docs/reports/FS-005_PHASE_3_FINAL_COUNT_RECONCILIATION.md`  
**Status:** Documentation reconciliation complete  
**Scope:** Documentation-only corrections — no Python code, tests, or experimental artifacts changed

---

## 1. Files Inspected

| File | Role |
|------|------|
| `tests/unit/test_evaluation_metrics_phase3.py` | Phase 3 unit tests (63 test functions) |
| `tests/unit/test_uncertainty_phase2.py` | Phase 2 unit tests (51 test functions) |
| `tests/unit/test_claim_verification_phase1.py` | Phase 1 unit tests (47 test functions) |
| `tests/unit/test_baseline_comparison.py` | Baseline comparison tests (16 test functions) |
| `tests/unit/test_schemas.py` | Schema tests (4 test functions) |
| `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md` | Phase 3 implementation report |
| `docs/reports/FS-005_PHASE_3_CORRECTIVE_REVIEW.md` | Phase 3 corrective review |
| `docs/reports/FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md` | Phase 3 final closure report |

---

## 2. Actual Test Inventory (63 Phase 3 Tests)

All 63 test functions were identified by reading `tests/unit/test_evaluation_metrics_phase3.py` and verified with `pytest --collect-only`.

### Accuracy (9 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_accuracy_all_correct | All correct → accuracy = 1.0 |
| test_compute_accuracy_all_incorrect | All incorrect → accuracy = 0.0 |
| test_compute_accuracy_mixed | 2/3 correct → accuracy = 2/3 |
| test_compute_accuracy_empty | Empty dataset → None |
| test_compute_accuracy_abstained_counted_incorrect | Abstained counted in denominator |
| test_compute_open_accuracy | OPEN subset accuracy |
| test_compute_open_accuracy_empty | Empty OPEN subset → None |
| test_compute_closed_accuracy | CLOSED subset accuracy |
| test_compute_closed_accuracy_empty | Empty CLOSED subset → None |

### USR_raw / USR_filtered (11 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_usr_raw_all_supported | All supported → usr_raw = 0.0 |
| test_compute_usr_raw_all_unsupported | All unsupported → usr_raw = 1.0 |
| test_compute_usr_raw_mixed | Mixed → usr_raw formula |
| test_compute_usr_raw_no_positive_claims | No positive claims → usr_raw = None |
| test_compute_usr_filtered_all_supported | All supported → usr_filtered = 0.0 |
| test_compute_usr_filtered_all_unsupported | All unsupported → usr_filtered = 1.0 |
| test_compute_usr_filtered_with_unverifiable | Unverifiable stripped from denominator |
| test_compute_usr_filtered_denominator_zero | Denominator = 0 → usr_filtered = None |
| test_compute_usr_absence_excluded_from_denominator | Absence excluded from USR denominator |
| test_absence_claims_excluded_from_usr_denominator | Absence claims not in n_positive_claims |
| test_uncertain_claims_excluded_from_usr_denominator | Uncertain claims not in n_positive_claims |

### Pointing accuracy (8 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_pointing_accuracy_all_correct | All pointing correct → 1.0 |
| test_compute_pointing_accuracy_none_correct | None correct → 0.0 |
| test_compute_pointing_accuracy_excludes_no_gt | Samples without GT excluded from denominator |
| test_compute_pointing_accuracy_no_gt_regions | No GT regions → None |
| test_compute_pointing_accuracy_multiple_predictions | Multiple claims, one correct → 0.5 |
| test_compute_pointing_accuracy_multiple_gt_regions | Multiple GT boxes, max IoU used → 1.0 |
| test_compute_pointing_accuracy_no_predictions | No claims → pointing_correct = False |
| test_compute_pointing_accuracy_multiple_claims_one_sample | One sample, one correct claim → sample-level correct |

### Sampling agreement / normalized entropy (4 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_sampling_agreement | Mixed agreement → 0.5 |
| test_compute_sampling_agreement_empty | Empty → None |
| test_compute_normalized_entropy | Mixed entropy → 0.5 |
| test_compute_normalized_entropy_empty | Empty → None |

### ECE (5 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_ece_perfectly_calibrated | Perfect calibration → ECE = 0.1 |
| test_compute_ece_no_accepted_samples | All abstained → None |
| test_compute_ece_empty | Empty → None |
| test_compute_ece_bin_boundary | Boundary confidence → bin assignment |
| test_compute_ece_max_confidence_last_bin | Confidence = 1.0 → last bin |

### Selective accuracy / coverage (5 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_selective_accuracy_all_correct | All correct → 1.0 |
| test_compute_selective_accuracy_none_in_coverage | None in coverage → None |
| test_compute_coverage | Mixed coverage → 2/3 |
| test_compute_coverage_empty | Empty → 0.0 |
| test_compute_coverage_all_abstained | All abstained → 0.0 |

### AUROC (5 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_auroc_perfect_separation | Perfect separation → 1.0 |
| test_compute_auroc_random | Random → 0.5 |
| test_compute_auroc_one_class | Single class → None |
| test_compute_auroc_no_eligible | No eligible → None |
| test_compute_auroc_threshold_independence | Threshold independence |

### Accuracy-vs-coverage curve (7 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_accuracy_vs_coverage_curve_shape | 101 points |
| test_compute_accuracy_vs_coverage_curve_monotonic_coverage | Coverage decreases monotonically |
| test_compute_accuracy_vs_coverage_curve_tau_zero | tau=0 → coverage = 1.0 |
| test_compute_accuracy_vs_coverage_curve_tau_one | tau=1 → only c=1.0 covered |
| test_compute_accuracy_vs_coverage_curve_invalid_sampling_never_covered | Invalid sampling never covered |
| test_compute_accuracy_vs_coverage_curve_uncertain_claims_never_covered | Uncertain claims never covered |
| test_compute_accuracy_vs_coverage_curve_mixed_forced_abstention | Mixed eligibility scenarios |

### Aggregate metrics (2 tests)
| Test | What it verifies |
|------|------------------|
| test_compute_aggregate_metrics | All metrics computed |
| test_compute_aggregate_metrics_with_exclusions | Exclusions preserved |

### Artifact generation (4 tests)
| Test | What it verifies |
|------|------------------|
| test_runner_writes_all_artifacts | All 5 artifact files created |
| test_runner_json_artifact_structure | JSON structure valid |
| test_runner_csv_artifact_structure | CSV headers and rows valid |
| test_runner_report_md_structure | Markdown report sections valid |

### GT isolation (2 tests)
| Test | What it verifies |
|------|------------------|
| test_metrics_do_not_modify_runtime_inputs | Input records not modified |
| test_ground_truth_not_used_in_confidence_computation | No GT params in confidence functions |

**Total: 63 tests** ✓

---

## 3. Category Breakdown (Consistent Across All Reports)

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

**Verification: 9 + 11 + 8 + 4 + 5 + 5 + 5 + 7 + 2 + 4 + 2 = 63** ✓

---

## 4. Five-File Combined Total (Consistent Across All Reports)

| Test file | Count |
|-----------|-------|
| tests/unit/test_evaluation_metrics_phase3.py | 63 |
| tests/unit/test_uncertainty_phase2.py | 51 |
| tests/unit/test_claim_verification_phase1.py | 47 |
| tests/unit/test_baseline_comparison.py | 16 |
| tests/unit/test_schemas.py | 4 |
| **Combined total** | **181** |

**Verification: 63 + 51 + 47 + 16 + 4 = 181** ✓

This was confirmed by running `pytest --collect-only` on each file and running all five files together (`181 passed`).

---

## 5. Corrections Made

### 5.1 AUROC Contradiction Fixed
**File:** `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md`  
**Location:** Section 3.11 (AUROC)  
**Problem:** The report stated both "Returns 0.5 if only one correctness class" AND "Returns None if only one correctness class" — contradictory.  
**Correction:** Removed the "Returns 0.5" clause. Now states only "Returns None if there are no eligible samples or only one correctness class." Matches `evaluation/metrics.py` behavior and the technical design.

### 5.2 Test Count Corrected (62 → 63)
**Files:** All four Phase 3 reports  
**Problem:** Reports stated "62 deterministic tests" but `pytest --collect-only` confirms 63 test functions.  
**Correction:** All reports now state "63 deterministic unit tests" / "63 Phase 3 tests".

### 5.3 Category Breakdown Corrected
**Files:** `FS-005_PHASE_3_IMPLEMENTATION_REPORT.md` and `FS-005_PHASE_3_CORRECTIVE_REVIEW.md`  
**Problem:** Previous category counts summed to 60 (implementation) and 59 (corrective review), not 63. The `test_uncertain_claims_counted_in_overall_accuracy_denominator` test was not accounted for in any category, and the USR count (9) did not include the 3 absence/uncertain USR tests.  
**Correction:** All reports now use the verified category breakdown that sums to exactly 63:
- Accuracy: 9 (includes `test_uncertain_claims_counted_in_overall_accuracy_denominator`)
- USR: 11 (includes `test_compute_usr_absence_excluded_from_denominator`, `test_absence_claims_excluded_from_usr_denominator`, `test_uncertain_claims_excluded_from_usr_denominator`)
- Pointing accuracy: 8
- Sampling agreement / entropy: 4
- ECE: 5
- Selective accuracy / coverage: 5
- AUROC: 5
- Accuracy-vs-coverage curve: 7
- Aggregate metrics: 2
- Artifact generation: 4
- GT isolation: 2

### 5.4 Combined Total Corrected (212 → 181)
**Files:** All four Phase 3 reports  
**Problem:** Reports stated "212 passed" but `pytest` on all five files confirms 181 passed.  
**Correction:** All reports now state "181 passed".

---

## 6. No Python Code, Tests, or Artifacts Changed

**Verification:** Only documentation files (`.md` reports) were modified. No Python implementation files, test files, or experimental artifacts were touched. The actual test functions in `tests/unit/test_evaluation_metrics_phase3.py` remain exactly as they were — 63 tests, all passing.

---

## 7. Phase Status

- **FS-005 Phase 3:** Complete — metrics harness and artifact generation implemented and verified
- **FS-005 Phase 4:** Not started
- **FS-005 Phase 5:** Not started
- **Model inference:** Not executed
- **FS-001–FS-004:** Not modified
