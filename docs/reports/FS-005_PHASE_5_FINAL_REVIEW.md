# FS-005 Phase 5 — Final Execution Plan Review Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_FINAL_REVIEW.md`  
**Status:** Corrected design complete  
**Scope:** Execution planning only — no model inference executed in this document  

---

## 1. Issues Inspected and Corrected

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | **Workload arithmetic error:** Original plan claimed ~20-30 hours and ~29,829 unique generations. Correct grand total is 40,210 unique generations (19,980 test + 20,230 calibration). The 20-32 hour estimate is NOT verified — runtime must be based on measured pilot throughput. | Critical | Workload recalculated with correct unique generation counts |
| 2 | **Paired cohort not explicit:** Track B has 864 annotated samples, Track C has 945 eligible samples. Paired comparison must use exactly the same 864 QIDs and image IDs | Critical | Paired cohort explicitly defined; 81 Track C-only samples used for answer accuracy only |
| 3 | **Calibration methodology:** Calibration must use only validation split; separate fit/assessment subsets by image ID; Track B calibration limited to 970 annotated validation samples | Critical | Calibration methodology corrected |
| 4 | **Conditions 2 and 3 reuse:** Both conditions must share same N=5 generated responses; only abstention decision differs | Critical | Plan corrected to require shared responses |
| 5 | **Statistical analysis:** Image-clustered paired bootstrap (1,000 resamples); image-level clustering; not naive string matching | Critical | Statistical analysis plan corrected |
| 6 | **Checkpointing/resumption:** Not supported by current implementation | Critical | Documented as blocker |

---

## 2. Corrected Workload Arithmetic

### 2.1 Unique Generation Count

Conditions 2 and 3 share the same N=5 generated responses per sample. Condition 1 uses N=1. The table below counts **unique generations**, not condition evaluations.

| Track | Condition | Model | Samples | N | Unique Generations | Est. Time |
|-------|-----------|-------|---------|---|------------------|-----------|
| B | 2 | MedGemma-4B | 864 | 5 | 4,320 | ~4-6 hours |
| B | 3 | MedGemma-4B | 864 | 5 | *(shared with Cond. 2)* | *(shared)* |
| C | 1 | MedGemma-4B | 945 | 1 | 945 | ~1-2 hours |
| C | 2 | MedGemma-4B | 945 | 5 | 4,725 | ~4-6 hours |
| C | 3 | MedGemma-4B | 945 | 5 | *(shared with Cond. 2)* | *(shared)* |
| B | 2 | Qwen2.5-VL-3B | 864 | 5 | 4,320 | ~4-6 hours |
| B | 3 | Qwen2.5-VL-3B | 864 | 5 | *(shared with Cond. 2)* | *(shared)* |
| C | 1 | Qwen2.5-VL-3B | 945 | 1 | 945 | ~1-2 hours |
| C | 2 | Qwen2.5-VL-3B | 945 | 5 | 4,725 | ~4-6 hours |
| C | 3 | Qwen2.5-VL-3B | 945 | 5 | *(shared with Cond. 2)* | *(shared)* |
| **Test unique generations** | | | | **19,980** | | |
| **Calibration total** | | | | **20,230** | | |
| **Grand total unique generations** | | | | **40,210** | Not verified | |

**Important:** Estimates assume ~1-1.5 seconds per inference on T4. Actual times may vary significantly. Calibration (validation split) adds ~10,530 additional inferences.

**Runtime estimate is NOT verified.** The previous 20-32 hour estimate was not based on measured pilot throughput and must not be presented as verified.

---

## 3. Corrected Calibration Methodology

### 3.1 Calibration Data

**Dataset:** SLAKE English validation split (`data/SLAKE/validate.json`)
- Total English validation samples: 1,053
- Samples with GT regions: 970
- Samples without GT regions: 83

**Important:** Calibration uses **only** validation data. Test set samples are never used for calibration.

### 3.2 Calibration Procedure

Calibration is performed **separately for each model/track combination** that uses confidence gating (Condition 3). Condition 1 (N=1) has degenerate agreement/entropy and is not used for calibration.

For each model/track combination:
1. Run inference on SLAKE validation samples (Track B Condition 3 or Track C Condition 3, N=5, T=0.7)
2. For each sample, compute: agreement (a), normalized entropy (H_n), localizer score (rho), confidence (c), correctness (binary)
3. Partition validation data into fit and assessment subsets **by image ID** (not individual question) to prevent leakage from multiple questions sharing the same image
4. Grid search over (w1, w2, w3) with step 0.05, subject to w1 + w2 + w3 = 1, w_i ≥ 0
5. For each weight combination, grid search over tau in [0, 1] with step 0.01 on the **fit subset**
6. Select (w1, w2, w3, tau) that minimizes selective risk subject to coverage ≥ 0.10 on the **fit subset**
7. Evaluate selected weights/tau on the **assessment subset**
8. Freeze selected weights and threshold — never touch test set

**Track B:** Only samples with GT regions (970 validation samples) are eligible for calibration. Track C: All 1,053 validation samples are eligible (ungrounded, no GT requirement).

### 3.3 Calibration Artifacts

All calibration outputs saved to `experiments/runs/fs005_phase5/calibration/`:
- `calibration_metrics.json` — weights, tau, selective risk, coverage, AUROC, ECE per model/track
- `calibration_curve.json` — accuracy-vs-coverage points for each candidate (w, tau)
- `calibration_samples.json` — per-sample calibration data (answers, confidences, correctness)

---

## 4. Corrected Test Evaluation Protocol

### 4.1 Test Cohort

**945 eligible SLAKE test samples** (image-disjoint from FS-004 pilot):
- 864 samples with GT regions (eligible for Track B pointing accuracy)
- 81 samples without GT regions (Track C only; excluded from pointing accuracy denominator)
- 572 OPEN questions
- 373 CLOSED questions

**Cohort artifacts:** `experiments/runs/fs005_phase5/cohort_audit/`

### 4.2 Metrics Computed

| Metric | Definition | Applies To |
|--------|-----------|------------|
| Overall accuracy | correct / total | Both tracks |
| OPEN accuracy | correct / total (OPEN subset) | Both tracks |
| CLOSED accuracy | correct / total (CLOSED subset) | Both tracks |
| USR_raw | (unsupported + unverifiable) / positive claims | Track B only |
| USR_filtered | unsupported / (positive − unverifiable) | Track B only |
| Pointing accuracy | samples with IoU ≥ 0.5 / samples with GT regions | Track B only |
| Sampling agreement | mean agreement | Both tracks |
| Normalized entropy | mean H_n | Both tracks |
| ECE | Σ\|B_m\|/n × \|acc(B_m) − conf(B_m)\| | Both tracks (accepted samples only) |
| Selective accuracy | correct / accepted | Both tracks |
| Coverage | accepted / total | Both tracks |
| AUROC | Mann-Whitney U on confidence vs correctness | Both tracks (eligible population) |

### 4.3 Metric Edge Cases

| Situation | Handling |
|-----------|----------|
| USR denominator = 0 | Return None (not 0.0) |
| Pointing accuracy denominator = 0 | Return None |
| ECE with no accepted samples | Return None |
| AUROC with one correctness class | Return None |
| Coverage = 0 | Selective accuracy = None |

### 4.5 Statistical Analysis

- **Bootstrap confidence intervals:** 1,000 bootstrap resamples at the **image level** (not per-sample) to account for multiple questions sharing the same image (87 unique images for 945 samples)
- **Paired comparison:** Paired McNemar's test for paired correctness on the same samples
- **Multiple comparison correction:** Bonferroni correction for primary comparisons

### 4.6 Primary Comparisons

| Comparison | Question | Test |
|------------|----------|------|
| Track B vs Track C (MedGemma) | Does grounding improve accuracy? | Paired McNemar's test |
| Condition 2 vs Condition 3 (MedGemma, Track B) | Does abstention improve selective accuracy? | Paired McNemar's test |
| Condition 2 vs Condition 3 (MedGemma, Track C) | Does abstention improve selective accuracy? | Paired McNemar's test |
| Track B vs Track C (Qwen) | Does grounding help Qwen? | Paired McNemar's test |
| MedGemma vs Qwen (Track C) | Does MedGemma outperform Qwen ungrounded? | Paired McNemar's test |
| MedGemma vs Qwen (Track B) | Does MedGemma outperform Qwen grounded? | Paired McNemar's test |

All tests use **image-clustered bootstrap** (1,000 resamples) to account for multiple questions sharing the same image.

---

## 7. Implementation Feasibility

### 9.1 `scripts/fs005_phase5_calibration.py`

- **Current behavior:** Uses synthetic validation data (200 samples), not real SLAKE validation data
- **Calibration:** Grid search over (w1, w2, w3) and tau using `ConfidenceCalibrator`
- **Missing:** Does not separate fit/assessment subsets by image ID; does not support separate model/track calibration; uses synthetic data not real SLAKE validation outputs
- **Status:** Implementation exists but requires real validation data and fit/assessment separation

### 9.2 `scripts/fs005_phase5_evaluation.py`

- **Current behavior:** Loads FS-004 baseline artifacts; computes aggregate metrics; does not run real inference
- **Missing:** Does not run real model inference; uses FS-004 baseline artifacts as evaluation data
- **Status:** Implementation exists but requires real model inference for Phase 5

### 9.3 `verification/pipeline.py`

- **Current behavior:** Runs Qwen/MedGemma inference via `VLMClient`; claim extraction, association, verification, filtering; semantic clustering; confidence gate; selective prediction
- **Missing:** Does not support sharing N=5 responses between Conditions 2 and 3 (runs them separately); does not support separate model/track calibration; does not support image-clustered statistical evaluation
- **Status:** Implementation exists but requires modifications for Phase 5 requirements

### 9.4 `evaluation/runner.py`

- **Current behavior:** Computes aggregate metrics from per-sample records; generates JSON/CSV/Markdown artifacts
- **Missing:** Does not run real inference; uses pre-computed `SampleRecord` objects
- **Status:** Implementation exists but requires real per-sample records from Phase 5 inference

---

## 8. Remaining Blockers

| Blocker | Status | Resolution |
|-----------|--------|------------|
| Real model inference on Colab T4 | Not started | Requires HF_TOKEN for MedGemma; Colab Tesla T4 access |
| Real calibration on SLAKE validation split | Not started | Requires MedGemma and Qwen inference on validation samples |
| Fresh image-disjoint test evaluation | Not started | Requires model inference on 945 test samples |
| Controlled comparison of Conditions 2 and 3 | Not started | Requires shared N=5 responses between conditions |
| Demonstration of actual abstention benefit | Not started | Requires Condition 2 vs Condition 3 comparison with coverage < 1.0 |

---

## 9. Files Created and Modified

### Files Created

| File | Purpose |
|------|---------|
| `docs/reports/FS-005_PHASE_5_EXPERIMENT_EXECUTION_PLAN.md` | This document — corrected experimental plan |

### Files Modified

| File | Change |
|------|--------|
| None | No Python code changes made |

### Files Not Modified

- `experiments/runs/slake_baseline_ollama/` — original Qwen baseline artifacts unchanged
- `experiments/runs/medgemma_baseline/` — original MedGemma baseline artifacts unchanged
- `experiments/runs/slake_model_comparison/` — original comparison artifacts unchanged
- `verification/`, `evaluation/`, `core/` — Phase 1-4 implementations unchanged
- FS-001 through FS-004 — not modified

---

## 11. Final Response

**Experimental Design:** Complete. Three tracks (A inactive, B and C active), three conditions (1, 2, 3), 945 eligible test samples, 864 with GT regions.

**Validation Protocol:** Calibration on SLAKE validation split (1,053 samples, 970 with GT), test on 945 eligible test samples.

**Test Cohort Protocol:** Image-disjoint from FS-004 pilot (excludes 10 QIDs and 9 image IDs), 945 eligible samples, 87 unique images.

**Compute Estimate:** 40,210 unique generations (19,980 test + 20,230 calibration) on Colab Tesla T4, sequential model loading. The 20-32 hour estimate is NOT verified — a runtime estimate must be based on measured pilot throughput.

**Statistical Analysis:** Bootstrap CIs (1,000 resamples, image-clustered), paired McNemar's tests, Bonferroni correction.

**Files Created:**
- `docs/reports/FS-005_PHASE_5_EXPERIMENT_EXECUTION_PLAN.md` (this document)

**Files Modified:** None.

**FS-005 Phases 1-4:** Unmodified. FS-001 through FS-004: Unmodified.

**Model inference executed:** None. **Calibration executed:** None (synthetic only in prior phase).

**Phase 5 is planned but not executed.** Awaiting approval to proceed with Colab inference.
