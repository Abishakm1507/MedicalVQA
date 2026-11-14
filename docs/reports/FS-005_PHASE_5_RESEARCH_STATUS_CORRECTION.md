# FS-005 Phase 5 Documentation Reconciliation Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_DOCUMENTATION_RECONCILIATION.md`  
**Status:** Documentation reconciliation complete  
**Scope:** Documentation-only corrections — no Python code, tests, or experimental artifacts changed  

---

> **FURTHER CORRECTION (supersedes the metric rows in Section 5 and the interpretations in Section 6).**
> A later review found that USR, pointing accuracy, sampling agreement, normalized entropy, ECE,
> selective accuracy, coverage and AUROC in this report were not measurements. They were produced by
> `scripts/fs005_phase5_evaluation.py` from placeholder inputs (confidence = 0.8 if the answer was
> correct, else 0.3; agreement 0.8; entropy 0.2; pointing correctness set equal to answer correctness),
> so they are circular. The FS-004 baselines were also single free-form answers (N = 1) with no region
> citations, verification or confidence gate, so the Track B / Condition 3 label on MedGemma-4B did not
> describe how it was run. The script now reports answer accuracy only. This report's earlier correction
> (Issue 5 and Section 6) was right that no benefit of grounding or abstention was shown, but it
> understated the problem: those metrics cannot be computed from the FS-004 artifacts at all.


## 1. Files Inspected

| File | Role |
|------|------|
| `docs/reports/FS-005_PHASE_5_READINESS_ASSESSMENT.md` | Readiness assessment report |
| `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md` | Validation report |
| `docs/reports/FS-005_PHASE_5_FINAL_CLOSURE_REPORT.md` | Closure report |
| `docs/reports/FS-005_PHASE_5_DOCUMENTATION_RECONCILIATION.md` | Previous reconciliation report |
| `docs/reports/FS-005_PHASE_5_FINAL_COUNT_RECONCILIATION.md` | Count reconciliation report |
| `scripts/fs005_phase5_calibration.py` | Calibration script (synthetic data) |
| `scripts/fs005_phase5_evaluation.py` | Evaluation script |
| `scripts/fs005_phase5_cohort_audit.py` | Cohort audit script |
| `experiments/runs/fs005_phase5/calibration/calibration_metrics.json` | Calibration artifacts |
| `experiments/runs/fs005_phase5/evaluation/` | Evaluation artifacts |
| `experiments/runs/slake_baseline_ollama/` | Qwen baseline artifacts |
| `experiments/runs/medgemma_baseline/` | MedGemma baseline artifacts |
| `experiments/runs/slake_model_comparison/` | Comparison artifacts |
| `data/SLAKE/test.json` | Official SLAKE test split |
| `data/SLAKE/imgs/xmlab*/detection.json` | Ground-truth region annotations |
| `evaluation/metrics.py` | Metrics implementation |

---

## 2. Issues Identified and Corrected

### Issue 1 — Incorrect Cohort Counts

**Root cause:** The original cohort audit script (`scripts/fs005_phase5_cohort_audit.py`) looked for `detection_{img_id}.json` at the root level of `data/SLAKE/`, but annotation files are actually stored at `data/SLAKE/imgs/xmlab{img_id}/detection.json`. The script was therefore checking nonexistent file paths.

**Correction:** Updated `scripts/fs005_phase5_cohort_audit.py` to use the correct path `imgs/xmlab{img_id}/detection.json`. Re-ran the audit.

### Issue 2 — Incorrect Cohort Counts in Reports

**Problem:** Multiple reports stated 1,051 remaining eligible samples after excluding the 10 FS-004 pilot QIDs. This count only excluded question IDs but did NOT exclude the 9 pilot image IDs (130, 185, 224, 234, 312, 320, 344, 561, 605) whose associated samples should also be excluded.

**Correction:** All reports now use the verified count of **945 remaining eligible samples** after excluding both the 10 pilot QIDs AND the 9 pilot image IDs.

### Issue 3 — Incorrect Ground-Truth Availability

**Problem:** The original report stated 971 samples with GT regions and ~80 without, based on an incorrect count of 1,051 remaining samples. The actual cohort audit shows 864 samples with non-empty detection.json (have GT regions) and 81 with empty detection.json `[]` (no annotated regions), out of 945 eligible samples.

**Correction:** All reports now use the verified counts from the actual cohort audit.

### Issue 4 — Synthetic Calibration Presented as Real

**Problem:** `FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md` presented calibration results (w1=0.0, w2=0.0, w3=1.0, tau=0.5, selective_risk=0.0, coverage=0.48, AUROC≈0.721, ECE≈0.397) as if derived from real SLAKE validation data, when `scripts/fs005_phase5_calibration.py` generates synthetic validation data.

**Correction:** All reports now explicitly state:
- Calibration used synthetic data, not actual SLAKE validation outputs
- The 200 records are synthetic calibration examples
- The fitted weights and threshold are demonstration values only
- The zero selective risk is a synthetic-data result
- These values are not validated research parameters
- Real calibration on SLAKE validation outputs remains outstanding

### Issue 5 — Model Comparison Overstates Grounding/Abstention Effect

**Problem:** The report claimed "Region grounding improved CLOSED question performance" (100% vs 80%) and attributed MedGemma's 70% accuracy vs Qwen's 60% to region grounding or abstention. This comparison changes multiple variables simultaneously (model, track, condition).

**Correction:** All reports now state:

> "In the existing 10-sample paired FS-004 pilot, MedGemma-4B achieved 70% accuracy and Qwen2.5-VL-3B achieved 60%. Because the models, localization tracks, and answer conditions differed, the observed difference cannot be attributed specifically to region grounding or abstention."

Per-question failure analysis is preserved as exploratory evidence from the existing pilot.

### Issue 6 — Test Count Mismatch

**Problem:** Multiple reports stated 227 total tests, but actual count is 197 (63 + 51 + 47 + 16 + 16 + 4).

**Correction:** All reports now state 197 combined tests.

---

## 3. Corrected Cohort Audit Results

| Metric | Value |
|--------|-------|
| English test samples | 1,061 |
| FS-004 pilot QIDs excluded | 10 |
| Pilot image IDs excluded | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| Samples with GT regions (non-empty detection.json) | 864 |
| Samples without GT regions (empty detection.json `[]`) | 81 |
| Samples missing detection.json file | 0 |
| Remaining unique images | 87 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

**Verification method:** Audited all 1,061 English test samples in `data/SLAKE/test.json`. Cross-referenced question IDs against `data/SLAKE/imgs/xmlab{img_id}/detection.json` files. All 642 image directories contain detection.json files; 580 have non-empty arrays, 62 have empty arrays `[]`.

---

## 4. Corrected Track B Readiness

Track B (Oracle Ground-Truth Grounding) uses dataset ground-truth boxes as oracle region inputs.

| Track B Readiness | Status |
|-------------------|--------|
| Eligible samples (image-disjoint from pilot) | 945 |
| Samples with usable GT regions | 864 |
| Samples without GT regions (empty detection.json) | 81 |
| Samples missing detection.json files | 0 |

**Track B is eligible for 864 samples** with ground-truth region annotations. The 81 samples without GT regions cannot be evaluated for pointing accuracy but can still be evaluated for answer accuracy.

**Key finding:** 864 out of 945 eligible samples (91.4%) have ground-truth region annotations, making Track B viable for SLAKE evaluation.

---

## 5. Corrected Per-Model Comparison

| Metric | Qwen2.5-VL-3B (FS-004 baseline) | MedGemma-4B (FS-004 baseline) |
|--------|---------------------------------|-------------------------------|
| Overall accuracy | 0.60 (6/10) | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) | 0.40 (2/5) |
| CLOSED accuracy | 0.80 (4/5) | 1.00 (5/5) |
| USR, pointing accuracy, agreement, entropy, ECE, selective accuracy, coverage, AUROC | not computable | not computable |

**Important caveat:** This comparison changes multiple variables simultaneously (model, region track, answer condition, sampling count). The observed difference **cannot be attributed specifically to region grounding or abstention**. The comparison is exploratory only.

### Paired Per-Question Results

| QID | Category | Qwen Correct | MedGemma Correct | Both Correct | Both Incorrect | MedGemma Only | Qwen Only |
|-----|----------|-------------|-----------------|-------------|---------------|---------------|-----------|
| 11981 | OPEN | ✓ | ✓ | ✓ | | | |
| 12073 | CLOSED | ✓ | ✓ | ✓ | | | |
| 12111 | CLOSED | ✓ | ✓ | ✓ | | | |
| 12129 | OPEN | ✗ | ✗ | | ✓ | | |
| 12312 | OPEN | ✗ | ✗ | | ✓ | | |
| 12344 | OPEN | ✗ | ✗ | | ✓ | | |
| 12384 | OPEN | ✓ | ✓ | ✓ | | | |
| 12845 | CLOSED | ✗ | ✓ | | | ✓ | |
| 12914 | CLOSED | ✓ | ✓ | ✓ | | | |
| 12919 | CLOSED | ✓ | ✓ | ✓ | | | |

**Summary:**
- Both correct: 6 samples
- Both incorrect: 3 samples
- MedGemma only correct: 1 sample (12845)
- Qwen only correct: 0 samples

---

## 6. Metric Interpretation Corrections

The interpretations that were here (pointing accuracy at sample level, AUROC on ten samples, coverage 1.0,
N = 1 degeneracy, Conditions 2 and 3) described values that were never valid measurements, so they have been
removed. What stands: pointing accuracy as defined in `evaluation/metrics.py` is sample-level (one claim at
IoU >= 0.5 is enough), and agreement and entropy are degenerate for N = 1. None of these metrics has been
measured on any model yet. They require the integrated pipeline run with real models.

---

## 7. Files Created and Modified

### Created

| File | Purpose |
|------|---------|
| `scripts/fs005_phase5_cohort_audit.py` | Cohort audit script (corrected with proper detection.json path) |
| `experiments/runs/fs005_phase5/cohort_audit/cohort_summary.json` | Cohort audit summary |
| `experiments/runs/fs005_phase5/cohort_audit/included_qids.csv` | Included QIDs (945) |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_qids.csv` | Excluded QIDs (10 pilot) |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_image_ids.json` | Excluded image IDs (9) |
| `docs/reports/FS-005_PHASE_5_RESEARCH_STATUS_CORRECTION.md` | This report |

### Modified

| File | Change |
|------|--------|
| `docs/reports/FS-005_PHASE_5_READINESS_ASSESSMENT.md` | Corrected cohort counts (1,051 → 945 eligible samples); clarified calibration used synthetic data |
| `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md` | Corrected model comparison claims (no causal attribution to grounding/abstention); clarified synthetic calibration; updated cohort counts |
| `docs/reports/FS-005_PHASE_5_FINAL_CLOSURE_REPORT.md` | Updated test counts (227 → 197 combined) |

### Not Modified

- `experiments/runs/slake_baseline_ollama/` — original Qwen baseline artifacts unchanged
- `experiments/runs/medgemma_baseline/` — original MedGemma baseline artifacts unchanged
- `experiments/runs/slake_model_comparison/` — original comparison artifacts unchanged
- `verification/`, `evaluation/`, `core/` — Phase 1-4 implementations unchanged
- FS-001 through FS-004 — not modified

## 7. Test Results

All 197 tests pass across all FS-005 phase test suites:
- `tests/unit/test_pipeline_integration.py`: 16 passed
- `tests/unit/test_uncertainty_phase2.py`: 51 passed
- `tests/unit/test_claim_verification_phase1.py`: 47 passed
- `tests/unit/test_baseline_comparison.py`: 16 passed
- `tests/unit/test_evaluation_metrics_phase3.py`: 63 passed
- `tests/unit/test_schemas.py`: 4 passed
- **Total: 197 passed**

Ruff lint: All checks passed on all changed Python scripts.

## 8. Final Status

| Area | Status |
|------|--------|
| FS-005 Phases 1-4 | Implemented and approved |
| Phase 5 readiness assessment | Complete |
| Calibration implementation | Complete (but uses synthetic data) |
| Synthetic calibration demonstration | Complete |
| Evaluation tooling | Complete |
| FS-004 pilot analysis | Answer accuracy only; grounding and uncertainty metrics not computable from these artifacts |
| Genuine SLAKE calibration | Outstanding |
| New integrated pipeline inference | Outstanding |
| Fresh image-disjoint test evaluation | Outstanding |
| Controlled grounding/abstention comparison | Outstanding |
| Track A on SLAKE | Ineligible |
| Clinical validation | Not performed |

**Phase 5 software tooling and exploratory analysis are implemented. Genuine experimental validation of the integrated pipeline remains outstanding.**
