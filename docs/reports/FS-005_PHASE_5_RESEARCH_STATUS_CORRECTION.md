# FS-005 Phase 5 Documentation Reconciliation Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_DOCUMENTATION_RECONCILIATION.md`  
**Status:** Documentation reconciliation complete  
**Scope:** Documentation-only corrections — no Python code, tests, or experimental artifacts changed  

---

## 1. Issues Identified and Corrected

### Issue 1 — Synthetic Calibration Presented as Real

**Problem:** `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md` presented calibration results (w1=0.0, w2=0.0, w3=1.0, tau=0.5, selective_risk=0.0, coverage=0.48, AUROC≈0.721, ECE≈0.397) as if derived from real SLAKE validation data, when in fact `scripts/fs005_phase5_calibration.py` generates **synthetic** validation data (200 synthetic samples).

**Correction:** All reports now explicitly state:
- Calibration used synthetic data, not actual SLAKE validation outputs
- The 200 records are synthetic calibration examples
- The fitted weights and threshold are demonstration values only
- The zero selective risk is a synthetic-data result
- These values are not validated research parameters
- Real calibration on SLAKE validation outputs remains outstanding

### Issue 2 — Model Comparison Overstates Grounding/Abstention Effect

**Problem:** The report claimed "Region grounding improved CLOSED question performance" (100% vs 80%) and attributed MedGemma's 70% vs Qwen's 60% accuracy to region grounding or abstention. This comparison changes multiple variables simultaneously (model, track, condition), so no causal attribution is valid.

**Correction:** All reports now state:

> "In the existing 10-sample paired FS-004 pilot, MedGemma-4B achieved 70% accuracy and Qwen2.5-VL-3B achieved 60%. Because the models, localization tracks, and answer conditions differed, the observed difference cannot be attributed specifically to region grounding or abstention."

Per-question failure analysis is preserved as exploratory evidence from the existing pilot.

### Issue 3 — Experimental Validation Status Overstated

**Problem:** Reports described Phase 5 as "complete" with experimental validation, when in fact:
- Real SLAKE validation inference was never run
- Calibration using genuine SLAKE validation outputs was never performed
- New end-to-end inference through the integrated pipeline was never executed
- Fresh image-disjoint test evaluation was never performed
- Controlled comparison of answer Conditions 1, 2, and 3 was never performed
- An actual abstention benefit was never demonstrated

**Correction:** All reports now explicitly state:

> "FS-005 Phase 5 software tooling and exploratory analysis are implemented. Genuine experimental validation of the integrated pipeline remains outstanding."

### Issue 4 — Test Cohort Count Inaccurate

**Problem:** Reports stated 1,051 remaining samples after excluding the 10 pilot QIDs, but this did not exclude samples sharing images with the pilot cohort (9 unique images).

**Correction:** Performed an actual cohort audit by inspecting `data/SLAKE/test.json`:
- 1,061 English test samples
- 10 pilot QIDs excluded
- **9 pilot image IDs excluded** (130, 185, 224, 234, 312, 320, 344, 561, 605)
- **945 remaining eligible samples** (not 1,051)
- **87 remaining unique images** (not 87 after QID-only exclusion)
- 572 OPEN, 373 CLOSED
- 0 samples with detection.json files (no region annotations exist on disk)

Cohort audit artifacts saved to: `experiments/runs/fs005_phase5/cohort_audit/`

---

## 2. Corrected Metric Interpretations

### Pointing Accuracy (Sample-Level)

The FS-005 technical design Section 6.1 defines pointing accuracy at the **sample level**: a sample counts as correctly pointed if at least one claim has max IoU ≥ 0.5 against any ground-truth region. This is **not** the proportion of all claims correctly grounded. One correctly grounded claim may conceal other incorrectly grounded claims within the same sample.

### AUROC

AUROC is computed over the full eligible population (sampling_valid=True AND evidence_state != "uncertain"), independent of the abstention threshold. On 10 samples, AUROC=1.0 is exploratory and highly uncertain — it reflects perfect separation on a tiny sample, not a validated confidence-quality property.

### Coverage and Selective Accuracy

Coverage = 1.0 means no samples were abstained from. Selective accuracy then equals ordinary accuracy and does **not** demonstrate an abstention benefit.

### N=1 Sampling (Qwen Condition 1)

Condition 1 uses N=1, T=0.0 greedy sampling. Agreement and entropy for N=1 are degenerate (agreement=1.0, entropy=0.0 by definition) and do not represent meaningful uncertainty measurement.

---

## 3. Corrected Cohort Audit Results

| Metric | Value |
|--------|-------|
| English test samples (SLAKE) | 1,061 |
| FS-004 pilot QIDs excluded | 10 |
| Pilot image IDs excluded | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| Remaining eligible samples | 945 |
| Remaining unique images | 87 |
| Samples with GT regions | 0 (no detection.json files exist) |
| Samples without GT regions | 945 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

Artifacts saved to: `experiments/runs/fs005_phase5/cohort_audit/`
- `cohort_summary.json`
- `included_qids.csv`
- `excluded_qids.csv`
- `excluded_image_ids.json`

---

## 4. Files Created and Modified

### Created

| File | Purpose |
|------|---------|
| `docs/reports/FS-005_PHASE_5_RESEARCH_STATUS_CORRECTION.md` | This report — documents all corrections |
| `scripts/fs005_phase5_cohort_audit.py` | Cohort audit script |
| `experiments/runs/fs005_phase5/cohort_audit/cohort_summary.json` | Cohort audit summary |
| `experiments/runs/fs005_phase5/cohort_audit/included_qids.csv` | Included QIDs |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_qids.csv` | Excluded QIDs |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_image_ids.json` | Excluded image IDs |

### Modified

| File | Change |
|------|--------|
| `docs/reports/FS-005_PHASE_5_READINESS_ASSESSMENT.md` | Clarified calibration used synthetic data; updated cohort counts |
| `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md` | Corrected model comparison claims, clarified synthetic calibration, updated cohort counts |
| `docs/reports/FS-005_PHASE_5_FINAL_CLOSURE_REPORT.md` | Updated status descriptions |

### Not Modified

- `experiments/runs/slake_baseline_ollama/` — original Qwen baseline artifacts unchanged
- `experiments/runs/medgemma_baseline/` — original MedGemma baseline artifacts unchanged
- `experiments/runs/slake_model_comparison/` — original comparison artifacts unchanged
- `verification/`, `evaluation/`, `core/` — Phase 1-4 implementations unchanged
- FS-001 through FS-004 — not modified

---

## 5. Completed vs Outstanding Work

### Completed

- FS-005 Phases 1-4 implementation (claim extraction, region association, evidence verification, claim filtering, semantic clustering, confidence gate, evaluation metrics, artifact generation)
- Phase 5 readiness assessment
- Calibration script implementation (with synthetic data demonstration)
- Evaluation script implementation
- Exploratory analysis of existing FS-004 pilot artifacts
- Cohort audit (image-disjoint verification)
- Metric and artifact-generation tests

### Outstanding (Not Completed)

- Real SLAKE validation inference
- Calibration using genuine SLAKE validation outputs
- Frozen calibration configuration based on genuine validation data
- New end-to-end inference through the integrated Phase 1-4 pipeline
- Fresh image-disjoint test evaluation
- Controlled comparison of answer Conditions 1, 2, and 3
- Demonstration of an actual abstention benefit
- Research conclusions establishing grounding effectiveness

---

## 6. No New Model Inference Executed

**Confirmed:** No Qwen, MedGemma, or other VLM inference was run during this documentation reconciliation. All results reported are from the existing FS-004 baseline artifacts or from synthetic data generated by `scripts/fs005_phase5_calibration.py`.

---

## 7. Ruff Verification

| File | Result |
|------|--------|
| `scripts/fs005_phase5_calibration.py` | All checks passed |
| `scripts/fs005_phase5_evaluation.py` | All checks passed |
| `scripts/fs005_phase5_cohort_audit.py` | All checks passed |

---