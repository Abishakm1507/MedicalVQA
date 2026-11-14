# FS-005 Phase 5 — Experimental Execution Plan

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_EXPERIMENT_EXECUTION_PLAN.md`  
**Status:** Corrected implementation-ready design  
**Scope:** Execution planning only — no model inference executed in this document  

---

## 1. Research Objective

The core research question of FS-005 is:

> **The system can only claim what it can point to.** Every visual claim should be associated with evidence from an image region, and unsupported claims should be identified, filtered, or appropriately abstained from.

Phase 5 executes the validated pipeline on genuine SLAKE data to answer:

| Research Question | Status |
|-------------------|--------|
| Does region-grounded answering improve accuracy compared to ungrounded baselines? | To be tested (controlled comparison) |
| Does confidence-gate abstention improve selective accuracy at the cost of coverage? | To be tested (controlled comparison) |
| Can uncertainty estimation (semantic agreement + entropy) effectively identify unreliable predictions? | To be tested (controlled comparison) |
| How does MedGemma-4B compare to Qwen2.5-VL-3B on the SLAKE test split? | To be tested (paired, same track when possible) |

---

## 2. Experimental Design

### 2.1 Core Principle: Controlled Comparison

**Critical constraint:** Comparisons must control for confounding variables. The FS-004 pilot compared MedGemma (Track B, Condition 3) against Qwen (Track C, Condition 1) — three dimensions differed simultaneously (model, track, condition). No causal attribution was possible.

Phase 5 uses a **controlled comparison matrix**: for each model, run both Track B (oracle ground-truth grounding) and Track C (ungrounded), and both Condition 2 (no abstention) and Condition 3 (with abstention), on the **same sample cohort**. This isolates:

- **Track effect:** Track B vs Track C, same model, same condition — isolates grounding effect
- **Condition effect:** Condition 2 vs Condition 3, same model, same track — isolates abstention effect
- **Model effect:** MedGemma vs Qwen, same track, same condition — isolates model effect

**Important:** Track A (predicted CXR detector grounding) is **inactive for SLAKE** — SLAKE contains zero chest X-ray images (all chest-location entries are CT). No CXR detector evaluation is performed on SLAKE.

### 2.2 Experimental Matrix

| Track | Condition | Model | N | Temperature | Sampling | Grounding | Abstention | Samples |
|-------|-----------|-------|---|-------------|----------|-----------|------------|-----------|
| B | 2 | MedGemma-4B | 5 | 0.7 | Yes | Yes | No | 864 |
| B | 3 | MedGemma-4B | 5 | 0.7 | Yes | Yes | Yes | 864 |
| C | 1 | MedGemma-4B | 1 | 0.0 | No | No | No | 945 |
| C | 2 | MedGemma-4B | 5 | 0.7 | No | No | No | 945 |
| C | 3 | MedGemma-4B | 5 | 0.7 | No | No | Yes | 945 |
| B | 2 | Qwen2.5-VL-3B | 5 | 0.7 | Yes | Yes | No | 864 |
| B | 3 | Qwen2.5-VL-3B | 5 | 0.7 | Yes | Yes | Yes | 864 |
| C | 1 | Qwen2.5-VL-3B | 1 | 0.0 | No | No | No | 945 |
| C | 2 | Qwen2.5-VL-3B | 5 | 0.7 | No | No | No | 945 |
| C | 3 | Qwen2.5-VL-3B | 5 | 0.7 | No | No | Yes | 945 |

**Sample cohort:** Same 945 eligible SLAKE test samples (image-disjoint from FS-004 pilot) for all tracks and conditions.

### 2.3 Sample Cohort

| Metric | Value |
|--------|-------|
| English test samples | 1,061 |
| Minus FS-004 pilot QIDs | 10 |
| Minus pilot image IDs | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| Samples with GT regions (eligible for Track B) | 864 |
| Samples without GT regions (Track C only) | 81 |
| Remaining unique images | 87 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

**Cohort artifacts:** `experiments/runs/fs005_phase5/cohort_audit/`

### 2.4 Validation-Only Pilot Cohort (Colab T4 Feasibility Pilot)

The Colab T4 feasibility pilot uses a **validation-only** cohort — it does NOT
consume any sample from the official 945-sample test cohort.

| Metric | Value |
|--------|-------|
| Source dataset | SLAKE English validation split |
| Pilot samples selected | 15 (within the required 10–20 range) |
| Distinct pilot images | 15 (image-disjoint selection) |
| Samples with GT regions (Track B eligible) | 7 |
| Samples without GT regions (Track C only) | 8 |
| FS-004 pilot QIDs excluded | 10 |
| FS-004 pilot image IDs excluded | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |

**Calibration isolation:** The 15 pilot image IDs are excluded from calibration
fitting AND assessment, and the calibration split manifests are regenerated so
that no pilot image ID appears in either calibration subset.

**Test cohort preservation:** The official 945-sample test cohort is preserved
completely untouched. Pilot samples are never used to fit calibration
parameters or assess model performance.

**Pilot cohort artifacts:** `experiments/runs/fs005_phase5/pilot_cohort/`
- `pilot_manifest.json` — selected QIDs, image IDs, GT-region flags
- `pilot_qids.csv` — selected question IDs
- `pilot_image_ids.json` — selected image IDs

---

## 3. Calibration Protocol (Stage 2)

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
4. Grid search over $(w_1, w_2, w_3)$ with step 0.05, subject to $w_1 + w_2 + w_3 = 1, w_i \geq 0$
5. For each weight combination, grid search over $\tau$ in [0, 1] with step 0.01 on the **fit subset**
6. Select $(w_1, w_2, w_3, \tau)$ that minimizes selective risk subject to coverage $\ge$ 0.10 on the **fit subset**
7. Evaluate selected weights/tau on the **assessment subset**
8. Freeze selected weights and threshold — never touch test set

**Track B:** Only samples with GT regions (970 validation samples) are eligible for calibration.  
**Track C:** All 1,053 validation samples are eligible (ungrounded, no GT requirement).

### 3.3 Calibration Artifacts

All calibration outputs saved to `experiments/runs/fs005_phase5/calibration/`:
- `calibration_metrics.json` — weights, tau, selective risk, coverage, AUROC, ECE per model/track
- `calibration_curve.json` — accuracy-vs-coverage points for each candidate (w, tau)
- `calibration_samples.json` — per-sample calibration data (answers, confidences, correctness)

---

## 4. Test Evaluation Protocol

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

## 5. Compute and Runtime Estimate

### 5.1 Target Environment

| Resource | Specification |
|----------|---------------|
| GPU | Google Colab Tesla T4 (15.3 GB VRAM) |
| CPU | 2 vCPUs |
| RAM | ~12.7 GB |
| Storage | ~70 GB available |
| Duration limit | ~12 hours per session |

### 5.2 Model VRAM Requirements (4-bit Quantized)

| Model | VRAM Required | Sequential Loading |
|-------|---------------|-------------------|
| Qwen2.5-VL-3B | ~5-6 GB | Load once, run all samples |
| MedGemma-4B | ~6-7 GB | Load once, run all samples |

**Important:** Models must be loaded sequentially — never resident simultaneously. Total peak VRAM: ~6-7 GB (within T4's 15.3 GB limit).

### 5.3 Inference Workload Estimate

**Important:** Conditions 2 and 3 share the same N=5 generated responses per sample. Condition 1 uses N=1. The table below counts **unique generations**, not condition evaluations.

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
| **Calibration** | 3 | MedGemma-4B | 970 | 5 | 4,850 | ~2-4 hours |
| **Calibration** | 3 | Qwen2.5-VL-3B | 970 | 5 | 4,850 | ~2-4 hours |
| **Calibration** | 3 | MedGemma-4B | 1,053 | 5 | 5,265 | ~2-4 hours |
| **Calibration** | 3 | Qwen2.5-VL-3B | 1,053 | 5 | 5,265 | ~2-4 hours |
| **Calibration total** | | | | **20,230** | | |
| **Grand total unique generations** | | | | **40,210** | Not verified | |

**Important:** Estimates assume ~1-1.5 seconds per inference on T4. Actual times may vary significantly. Calibration (validation split) adds ~10,530 additional inferences.

**Runtime estimate is NOT verified.** The previous 20-32 hour estimate was not based on measured pilot throughput and must not be presented as verified. A reliable runtime estimate can only be produced after measuring per-inference throughput during the Colab T4 feasibility pilot (see `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`). The experiment requires up to 40,210 unique generations and exceeds a single Colab session (~12 hours). Plan for multiple sessions with checkpointing after each model/condition completion.

---

## 7. Implementation and Reproducibility

### 7.1 Output Directory Structure

```
experiments/runs/fs005_phase5/
├── calibration/
│   ├── calibration_metrics.json
│   ├── calibration_curve.json
│   └── calibration_samples.json
├── cohort_audit/
│   ├── cohort_summary.json
│   ├── included_qids.csv
│   ├── excluded_qids.csv
│   └── excluded_image_ids.json
└── experiments/
    ├── medgemma_track_b_condition_2/
    │   ├── per_sample_results.json
    │   ├── metrics.json
    │   └── metadata.json
    ├── medgemma_track_b_condition_3/
    ├── medgemma_track_c_condition_1/
    ├── medgemma_track_c_condition_2/
    ├── medgemma_track_c_condition_3/
    ├── qwen_track_b_condition_2/
    ├── qwen_track_b_condition_3/
    ├── qwen_track_c_condition_1/
    ├── qwen_track_c_condition_2/
    └── qwen_track_c_condition_3/
```

### 7.2 Per-Sample Record Format

Each per-sample record includes:
- sample_id, dataset_name, split, track, condition
- question, ground_truth_answer, model_answer, correct
- question_type, has_gt_regions, pointing_correct
- n_claims, n_positive_claims, n_supported, n_unsupported, n_unverifiable, n_absence, n_uncertain
- agreement, normalized_entropy, confidence, threshold, is_abstaining, in_coverage
- evidence_state, sampling_valid
- raw_response, parsed_answer, raw_inference_time

### 7.3 Reproducibility Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Obtain HF_TOKEN for MedGemma-4B access (Colab Secrets)
3. Run calibration script: `python scripts/fs005_phase5_calibration.py` (on validation split)
4. Run evaluation script: `python scripts/fs005_phase5_evaluation.py` (on test split)
5. Verify tests: `pytest tests/unit/`
6. Verify lint: `ruff check .`

---

## 8. Risks and Unresolved Decisions

| Risk | Mitigation |
|------|-----------|
| Colab session timeout (~12 hours) | Checkpoint after each model/condition; resume from checkpoints |
| HF_TOKEN not available | Cannot run MedGemma; report as blocked |
| VRAM insufficient | Sequential loading; bfloat16 quantization |
| Unexpected model behavior | Log all raw responses for post-hoc analysis |
| Calibration instability | Use multiple random seeds; report variance |

### Unresolved Decisions

| Decision | Status |
|-----------|----------|
| Whether to include Qwen Track C Condition 1 | Confirmed: included (free-form baseline) |
| Whether to include MedGemma Track C Condition 2 | Confirmed: included |
| Whether to evaluate Track A on VQA-RAD | Not available in this environment |

---

## 9. Strict Restrictions

- **No model inference is executed in this document.** All inference will be executed separately.
- **No FS-001 through FS-004 modifications.**
- **No ADR 0004 or ADR 0005 changes.**
- **No Phase 1-4 behavior changes.**
- **No synthetic results.**
- **No paid APIs.**
- **No HF_TOKEN hardcoded or printed.**
- **No Phase 6 started.**

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
