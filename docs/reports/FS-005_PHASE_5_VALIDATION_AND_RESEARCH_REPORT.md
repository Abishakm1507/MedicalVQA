# FS-005 Phase 5 — Validation and Uncertainty-Aware Answering: Final Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md`  
**Status:** Documentation complete — Phase 5 software tooling implemented; genuine experimental validation outstanding  
**Implementation type:** Implementation (code written and executed on existing artifacts), not new experimental validation  
**Prerequisites:** FS-005 Phases 1-4 complete and approved  

---

> **CORRECTION (supersedes the metric values previously shown in this report).**
> A later review found that the grounding and uncertainty metrics originally quoted here
> (USR, pointing accuracy, sampling agreement, normalized entropy, ECE, selective accuracy,
> coverage, AUROC) were not measurements. They came from `scripts/fs005_phase5_evaluation.py`,
> which filled their inputs with placeholders: confidence = 0.8 if the answer was correct and
> 0.3 if wrong, agreement and entropy fixed at 0.8 and 0.2, and pointing correctness set equal to
> answer correctness. Those metrics are therefore circular and carry no information. The script
> has been corrected and now reports answer accuracy only. In addition, the FS-004 baselines were
> single free-form answers (N = 1) without region citations, verification or a confidence gate, so
> the Track B / Condition 2 / Condition 3 labels previously attached to the MedGemma-4B result do
> not describe how it was run. Only the answer-accuracy figures below are valid. Every other metric
> is **not computable** from the FS-004 artifacts and requires the integrated pipeline to be run with
> real models.


## 1. Research Objective

FS-005 Phase 5 documents the grounding and uncertainty-aware answering system developed in Phases 1-4 and performs exploratory analysis of existing FS-004 baseline artifacts on the SLAKE dataset. The system's core research objective is:

> **The system can only claim what it can point to.** Every visual claim should be associated with evidence from an image region, and unsupported claims should be identified, filtered, or appropriately abstained from.

Phase 5 evaluates this objective using existing FS-004 baseline artifacts (10-sample pilot), not new model inference. Real experimental validation of the integrated pipeline remains outstanding.

---

## 2. Research Objectives and Scope

### 2.1 Core Research Questions

The following research questions were identified for FS-005:

1. Does region-grounded answering improve accuracy compared to ungrounded baselines? *(Exploratory only — FS-004 pilot compared different models on different tracks; cannot isolate grounding effect)*
2. Does confidence-gate abstention improve selective accuracy at the cost of coverage? *(Exploratory only — FS-004 pilot had coverage=1.0 for both models, so no abstention benefit was demonstrated)*
3. Can uncertainty estimation (semantic agreement + entropy) effectively identify unreliable predictions? *(Exploratory only — N=1 for Qwen condition makes entropy degenerate)*
4. How does MedGemma-4B compare to Qwen2.5-VL-3B on the SLAKE test split? *(Exploratory only — different tracks and conditions prevent causal attribution)*

### 2.2 Scope — In Scope

- Calibration script implementation (demonstrated on synthetic data)
- Evaluation script implementation (applied to existing FS-004 baseline artifacts)
- Exploratory analysis of existing FS-004 pilot artifacts
- Cohort audit (image-disjoint verification)
- Metric and artifact-generation tests

### 2.3 Scope — Explicitly Out of Scope

- Track A (predicted CXR detector grounding): **Inactive for SLAKE** — SLAKE contains zero chest X-ray images (all chest-location entries are CT)
- Real model inference: **Not executed** — all results computed from existing FS-004 baseline artifacts
- Genuine SLAKE validation calibration: **Outstanding** — synthetic calibration only
- Fresh image-disjoint test evaluation: **Outstanding** — cohort identified but not evaluated
- Controlled comparison of answer Conditions 1, 2, and 3: **Outstanding** — only Conditions 1 and 3 were represented in FS-004 pilot artifacts, with different models
- Clinical validation: **Not performed** — technical verification only
- RSNA, VinDr-CXR datasets: Not available in this environment
- VQA-RAD dataset: Not available in this environment

---

## 3. Proposed Architecture and Data Flow

The FS-005 pipeline (Phase 4) integrates Phases 1-3:

```
Input Image + Question
    │
    ▼
Region Proposal (Track A: detector / Track B: GT boxes / Track C: none)
    │
    ▼
VLM Inference (N=5 samples, T=0.7)
    │
    ▼
Claim Extraction → Claim-to-Region Association → Evidence Verification → Claim Filtering
    │
    ▼
Semantic Clustering → Agreement + Entropy → Confidence Calculation → Abstention Gate
    │
    ▼
Final Answer (or Abstention) + Evaluation Record
    │
    ▼
Metrics Computation + Artifact Generation
```

Ground-truth annotations are used **only** in the evaluation layer (pointing accuracy, answer correctness, ECE, AUROC). They never influence runtime confidence or abstention decisions.

---

## 4. Dataset and Cohort Verification

### 4.1 SLAKE Dataset Verification

| Property | Value |
|----------|-------|
| Local path | `data/SLAKE` (drive `D:`) |
| Total entries (all splits) | 14,028 |
| English entries | 7,033 |
| Train entries | 9,835 |
| Validate entries | 2,099 |
| Test entries | 2,094 |
| English test entries | 1,061 |
| English test unique images | 96 |
| English-only image-level overlap (train/validate/test) | 0 |
| Image-level overlap (any language) | 298 img_ids (46.4%) — all bilingual |
| Chest X-ray images | **0** (all chest-location entries are CT) |
| Track A eligibility | **Excluded** — no CXR images |

### 4.2 Previously Evaluated Samples (FS-004 Pilot)

The FS-004 paired pilot evaluated 10 SLAKE test samples. These samples **must be excluded** from any new confirmatory test cohort:

| QID | Image ID | Modality | Question Type | Answer |
|-----|----------|----------|-------------|--------|
| 11981 | 130 | X-Ray | OPEN | Chest |
| 12073 | 185 | X-Ray | CLOSED | Yes |
| 12111 | 224 | CT | CLOSED | Yes |
| 12129 | 234 | CT | OPEN | 0 |
| 12312 | 312 | X-Ray | OPEN | Lower Left Lung |
| 12344 | 320 | X-Ray | OPEN | Lung |
| 12384 | 344 | X-Ray | OPEN | X-Ray |
| 12845 | 561 | CT | CLOSED | No |
| 12914 | 605 | CT | CLOSED | Yes |
| 12919 | 605 | CT | CLOSED | Yes |

**Images excluded from new cohort:** 130, 185, 224, 234, 312, 320, 344, 561, 605 (9 unique images)

### 4.3 Remaining Eligible Cohort (Verified by Cohort Audit)

| Metric | Value |
|--------|-------|
| English test samples | 1,061 |
| Minus FS-004 pilot QIDs | 10 |
| Minus pilot image IDs | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| Remaining unique images | 87 |
| Samples with GT regions (detection.json non-empty) | 864 |
| Samples without GT regions (empty detection.json) | 81 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

**Cohort audit artifacts:** `experiments/runs/fs005_phase5/cohort_audit/`

---

## 5. Model Configurations

### MedGemma-4B (FS-004 baseline: single free-form answer, no grounding)

| Property | Value |
|----------|-------|
| Model ID | `google/medgemma-4b-it` |
| Inference environment | Kaggle Tesla T4 (14.56 GB VRAM) |
| Quantization | bfloat16 (no 4-bit quantization) |
| Attention implementation | eager |
| Sampling | N=1 (single answer per question) |
| Region source | None (the answer was requested as JSON with an answer field only) |
| Region grounding | No |

### Qwen2.5-VL-3B (FS-004 baseline: single free-form answer, no verification) — From FS-004 Pilot Artifacts

| Property | Value |
|----------|-------|
| Model ID | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Inference backend | Ollama (`qwen2.5vl:3b`) |
| Quantization | Q4_K_M |
| Sampling | N=1, T=0.0 (greedy) |
| Region grounding | None (ungrounded baseline) |

---

## 6. Confidence Calibration

### 6.1 Calibration Protocol (ADR 0005)

| Parameter | Value |
|-----------|-------|
| Grid search step size | 0.05 |
| Weight constraint | w1 + w2 + w3 = 1, w_i ≥ 0 |
| Primary objective | Minimize selective risk at operating coverage |
| Secondary objective | AUROC (diagnostic) |
| Calibration diagnostic | ECE |
| Coverage constraint | min_coverage = 0.1 (10%) |
| Validation split | SLAKE English validation (1,053 samples) |
| Test set | Held-out, never used for calibration |

### 6.2 Calibration Results (Synthetic Validation Data)

**Note:** Real model inference could not be executed (RTX 2050 4GB VRAM insufficient for 4-bit VLMs). Calibration was demonstrated using synthetic validation data generated by `scripts/fs005_phase5_calibration.py`. In production, replace with actual VLM outputs from the SLAKE validation split.

| Parameter | Value |
|-----------|-------|
| w1 (agreement weight) | 0.000 |
| w2 (entropy weight) | 0.000 |
| w3 (localizer weight) | 1.000 |
| tau (abstention threshold) | 0.500 |
| Selective risk | 0.0000 |
| Coverage | 0.4800 |
| AUROC | 0.7210 |
| ECE | 0.3974 |
| Validation samples | 200 (synthetic) |

**Calibration artifacts saved to:** `experiments/runs/fs005_phase5/calibration/calibration_metrics.json`

**Limitation:** Calibration used synthetic validation data. Real calibration requires running VLM inference on the SLAKE validation split (1,053 English samples) on Google Colab T4. These values are demonstration values only and are not validated research parameters.

---

## 7. Experimental Tracks

### Track A — Predicted Localization (CXR Detector)

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Excluded** — SLAKE contains zero chest X-ray images |
| Alternative CXR data | Not available locally |
| Detector checkpoint | Not downloaded; medical fine-tuned checkpoint needed |
| Status | **Not evaluated** — no eligible CXR data in SLAKE |

**Conclusion:** Track A cannot be evaluated on SLAKE. The dataset contains no chest X-ray images (all chest-location entries are CT). Track A would require VQA-RAD CXR or RSNA/VinDr-CXR data, which are not available in this environment.

### Track B — Oracle Ground-Truth Localization

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Eligible** — 864 of the 945 eligible English test samples have GT regions |
| Region source | Ground-truth boxes promoted to CandidateBox |
| Evaluation | IoU ≥ 0.5 against GT boxes (offline) |
| Status | **Not evaluated as a grounded track** — the only MedGemma-4B result is the ungrounded FS-004 answer accuracy below |

**FS-004 results (MedGemma-4B, ungrounded answers):**

| Metric | Value |
|--------|-------|
| Samples | 10 |
| Accuracy | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) |
| CLOSED accuracy | 1.00 (5/5) |
| USR_raw | not computable (placeholder inputs removed) |
| USR_filtered | not computable (placeholder inputs removed) |
| Pointing accuracy | not computable (placeholder inputs removed) |
| Sampling agreement | not computable (placeholder inputs removed) |
| Normalized entropy | not computable (placeholder inputs removed) |
| ECE | not computable (placeholder inputs removed) |
| Selective accuracy | not computable (placeholder inputs removed) |
| Coverage | not computable (placeholder inputs removed) |
| AUROC | not computable (placeholder inputs removed) |

### Track C — Ungrounded Baseline

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Eligible** — all 1,061 English test samples |
| VQA-RAD eligibility | Eligible (no region annotations) |
| Region grounding | None |
| Status | Answer accuracy only, from the FS-004 pilot artifacts |

**FS-004 results (Qwen2.5-VL-3B):**

| Metric | Value |
|--------|-------|
| Samples | 10 |
| Accuracy | 0.60 (6/10) |
| OPEN accuracy | 0.40 (2/5) |
| CLOSED accuracy | 0.80 (4/5) |
| USR_raw | not computable (placeholder inputs removed) |
| USR_filtered | not computable (placeholder inputs removed) |
| Pointing accuracy | not computable (placeholder inputs removed) |
| Sampling agreement | not computable (placeholder inputs removed) |
| Normalized entropy | not computable (placeholder inputs removed) |
| ECE | not computable (placeholder inputs removed) |
| Selective accuracy | not computable (placeholder inputs removed) |
| Coverage | not computable (placeholder inputs removed) |
| AUROC | not computable (placeholder inputs removed) |

---

## 8. Answer Conditions

Conditions 1, 2 and 3 were **not run** in Phase 5. The FS-004 baselines are single free-form answers
(N = 1) with no region citations, no claim verification and no confidence gate, so they correspond to
Condition 1 in form only. The earlier labelling of the MedGemma-4B result as Condition 2 or 3 (N = 5,
grounded, with abstention) was incorrect. A controlled comparison of the three conditions on the same
model and the same cohort remains outstanding.

| Result | Model | How it was actually run | Accuracy |
|--------|-------|-------------------------|----------|
| FS-004 baseline | Qwen2.5-VL-3B | Ollama, N = 1, greedy, no verification | 0.60 (6/10) |
| FS-004 baseline | MedGemma-4B | Kaggle T4, bfloat16, N = 1, answer-only JSON | 0.70 (7/10) |

The 10-point difference is one question (QID 12845), comes from two different models run through
different serving setups, and **cannot be attributed to region grounding or abstention**.

---

## 9. Per-Model Results

### MedGemma-4B (FS-004 baseline)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.70 |
| OPEN accuracy | 0.40 |
| CLOSED accuracy | 1.00 |
| USR_raw | not computable (placeholder inputs removed) |
| USR_filtered | not computable (placeholder inputs removed) |
| Pointing accuracy | not computable (placeholder inputs removed) |
| Sampling agreement | not computable (placeholder inputs removed) |
| Normalized entropy | not computable (placeholder inputs removed) |
| ECE | not computable (placeholder inputs removed) |
| Selective accuracy | not computable (placeholder inputs removed) |
| Coverage | not computable (placeholder inputs removed) |
| AUROC | not computable (placeholder inputs removed) |

### Qwen2.5-VL-3B (FS-004 baseline)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.60 |
| OPEN accuracy | 0.40 |
| CLOSED accuracy | 0.80 |
| USR_raw | not computable (placeholder inputs removed) |
| USR_filtered | not computable (placeholder inputs removed) |
| Pointing accuracy | not computable (placeholder inputs removed) |
| Sampling agreement | not computable (placeholder inputs removed) |
| Normalized entropy | not computable (placeholder inputs removed) |
| ECE | not computable (placeholder inputs removed) |
| Selective accuracy | not computable (placeholder inputs removed) |
| Coverage | not computable (placeholder inputs removed) |
| AUROC | not computable (placeholder inputs removed) |

### Paired Comparison (Same 10 Samples)

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
- Both correct: 6 samples (11981, 12073, 12111, 12384, 12914, 12919)
- Both incorrect: 3 samples (12129, 12312, 12344)
- MedGemma only correct: 1 sample (12845)
- Qwen only correct: 0 samples

**Accuracy difference:** 70% - 60% = 10 percentage points (1 sample difference out of 10)

**Important caveat:** The 10-percentage-point difference comes from a single sample (QID 12845). Because the models, localization tracks, and answer conditions differed, this difference **cannot be attributed specifically to region grounding or abstention**. With only 10 samples, no statistically significant difference can be claimed.

---

## 10. Grounding and Uncertainty Metrics

**Not available.** USR, pointing accuracy, sampling agreement, normalized entropy and ECE cannot be
computed from the FS-004 artifacts: they contain no region citations, no sampled answers and no
confidence scores. The values published in an earlier version of this section (for example USR 0.30,
pointing accuracy 0.70, ECE 0.23, AUROC 1.00) were derived from placeholders and must not be cited.
These metrics require the integrated pipeline (generation harness plus confidence gate) to be run with
real models on the calibration and test cohorts.

---

## 11. Selective Prediction Results

**Not available.** The FS-004 runs have no confidence score and no abstention decision, so coverage,
selective accuracy and the accuracy-versus-coverage curve are undefined. Earlier values (coverage 1.00,
selective accuracy 0.70 / 0.60) merely restated plain accuracy and show nothing about the confidence gate.

---

## 12. Statistical Analysis

### Paired Comparison (McNemar's Test Equivalent)

| Outcome | Count |
|---------|-------|
| Both correct | 6 |
| Both incorrect | 3 |
| MedGemma only correct | 1 |
| Qwen only correct | 0 |

**Accuracy difference:** 70% - 60% = 10 percentage points (1 sample difference out of 10)

**Statistical significance:** With only 10 samples, statistical significance cannot be established. The 10% difference could easily be due to chance. Bootstrap confidence intervals (1,000 resamples) would be needed for reliable inference.

### Confidence Intervals (Illustrative)

With n=10 and observed accuracy p:
- Qwen (60%): 95% CI approximately [26%, 88%]
- MedGemma (70%): 95% CI approximately [35%, 93%]

The wide intervals reflect the small sample size. No statistically significant difference can be claimed.

---

## 13. Failure Analysis

### Qwen2.5-VL-3B Failures (4 samples)

| QID | Category | Ground Truth | Qwen Prediction | Error Type |
|-----|----------|-------------|-----------------|------------|
| 12129 | OPEN | 0 | 2 | Counting error |
| 12312 | OPEN | Lower Left Lung | Right Lung | Wrong laterality |
| 12344 | OPEN | Lung | lungs | Pluralization |
| 12845 | CLOSED | No | yes | Yes/No error |

### MedGemma-4B Failures (3 samples)

| QID | Category | Ground Truth | MedGemma Prediction | Error Type |
|-----|----------|-------------|---------------------|------------|
| 12129 | OPEN | 0 | 1 | Counting error |
| 12312 | OPEN | Lower Left Lung | Left lung | Partial match (missing "Lower") |
| 12344 | OPEN | Lung | Lungs | Pluralization |

**Common Failure Modes:**
1. **Counting errors:** Both models struggled with counting questions (q12129)
2. **Laterality errors:** Both models confused left/right (q12312)
3. **Pluralization:** Both models used plural forms when singular was expected (q12344)
4. **Yes/No errors:** Qwen incorrectly answered "yes" for a "no" question (q12845); MedGemma correctly answered "No"

---

## 14. Limitations and Threats to Validity

### 14.1 Sample Size Limitation

| Limitation | Impact |
|------------|--------|
| 10-sample pilot | Results not statistically conclusive; wide confidence intervals |
| 9 unique images | Limited image diversity; results may not generalize |
| Single hospital (SLAKE) | Results may not generalize to other hospitals or imaging protocols |

### 14.2 Data Limitations

| Limitation | Impact |
|------------|--------|
| No CXR images in SLAKE | Track A (predicted CXR) cannot be evaluated |
| No patient IDs | Patient-level leakage cannot be verified; image-level leakage is zero for English-only |
| 96 unique test images | Limited test set size; results may not generalize |
| English-only | Results may not generalize to non-English queries |

### 14.3 Model Limitations

| Limitation | Impact |
|------------|--------|
| Qwen2.5-VL-3B via Ollama | Quantization (Q4_K_M) may reduce performance vs full precision |
| MedGemma-4B via Kaggle T4 | Different hardware than Qwen; wall times not directly comparable |
| No real-time inference | All results from FS-004 pilot artifacts; no new inference executed |

### 14.4 Methodological Limitations

| Limitation | Impact |
|------------|--------|
| Calibration on synthetic data | Real calibration requires SLAKE validation split inference |
| No bootstrap CIs | Statistical uncertainty not quantified |
| Single run per sample | N=1 for Qwen condition makes entropy degenerate |
| Confounded comparison | Model, track, and condition all differ between Qwen and MedGemma |

---

## 15. Reproducibility

### 15.1 Artifact Locations

| Artifact | Path |
|----------|------|
| FS-004 baseline artifacts | `experiments/runs/slake_baseline_ollama/` (Qwen) |
| FS-004 baseline artifacts | `experiments/runs/medgemma_baseline/` (MedGemma) |
| FS-004 comparison artifacts | `experiments/runs/slake_model_comparison/` |
| FS-005 calibration artifacts | `experiments/runs/fs005_phase5/calibration/` |
| FS-005 evaluation artifacts | `experiments/runs/fs005_phase5/evaluation/` |
| FS-005 cohort audit artifacts | `experiments/runs/fs005_phase5/cohort_audit/` |

### 15.2 Reproduction Steps

1. **Install dependencies:** `pip install -r requirements.txt` (torch, transformers, pydantic, scikit-learn, pandas)
2. **Reproduce FS-004 baseline:** Run Qwen via Ollama (`ollama run qwen2.5vl:3b`) or MedGemma via Kaggle
3. **Run FS-005 calibration:** `python scripts/fs005_phase5_calibration.py` (uses synthetic data; replace with SLAKE validation outputs)
4. **Run FS-005 evaluation:** `python scripts/fs005_phase5_evaluation.py` (uses FS-004 baseline artifacts)
5. **Run FS-005 cohort audit:** `python scripts/fs005_phase5_cohort_audit.py`
6. **Verify tests:** `pytest tests/unit/test_pipeline_integration.py tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_evaluation_metrics_phase3.py tests/unit/test_schemas.py`

### 15.3 Environment Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.23.0 |
| GPU | RTX 2050 4GB (insufficient for 4-bit VLMs) or Colab T4 15.3GB |
| RAM | 16 GB |
| Disk | ~70 GB free (D: drive) |

---

## 16. Conclusions Supported by Actual Evidence

### 16.1 What the Evidence Shows

1. **MedGemma-4B (70% accuracy) achieved higher accuracy than Qwen2.5-VL-3B (60% accuracy)** on the 10-sample FS-004 pilot. The difference came entirely from CLOSED questions (100% vs 80%).

2. **Both models made similar errors:** Both models failed on counting questions (q12129), laterality questions (q12312), and pluralization (q12344). Qwen additionally made a Yes/No error (q12845) that MedGemma got correct.

3. **The 10-percentage-point difference cannot be attributed to region grounding or abstention** because the models, localization tracks, and answer conditions all differed between Qwen and MedGemma.

4. **OPEN question performance was identical (40%):** Both models struggled equally with open-ended questions.

### 16.2 What Cannot Be Concluded

1. **Statistical significance:** With only 10 samples, no statistically significant difference can be claimed.
2. **Clinical superiority:** No evidence of clinical superiority for either model.
3. **Generalization:** Results may not generalize to other datasets, modalities, or patient populations.
4. **Real-world deployment:** Results are from a controlled pilot, not a clinical deployment.
5. **Region grounding effectiveness:** Cannot be determined from this pilot because Qwen (ungrounded) and MedGemma (oracle grounded) differ on multiple dimensions.
6. **Abstention benefit:** Coverage was 1.0 for both models, so no abstention benefit can be demonstrated.

---

## 17. Remaining Work Beyond FS-005

| Work Item | Status | Priority |
|-----------|--------|----------|
| Run real model inference on Colab T4 | Not started | High |
| Calibrate confidence weights on SLAKE validation split | Not started | High |
| Evaluate Track A on VQA-RAD CXR subset | Not started | Medium |
| Bootstrap confidence intervals (1,000 resamples) | Not started | Medium |
| Evaluate on larger SLAKE test cohort (945 remaining samples) | Not started | High |
| Evaluate on VQA-RAD dataset | Not started | Medium |
| Clinical validation study | Not started | High (requires clinical partners) |

---

## 18. Final Conclusion

FS-005 Phase 5 has implemented the grounding and uncertainty-aware answering system (Phases 1-4) and performed exploratory analysis of the existing FS-004 pilot artifacts on the SLAKE dataset.

**Key findings:**
- MedGemma-4B (70% accuracy) achieved higher accuracy than Qwen2.5-VL-3B (60% accuracy) on the 10-sample pilot
- The difference came entirely from CLOSED questions (100% vs 80%)
- Both models shared similar failure modes (counting, laterality, pluralization)
- Region grounding effectiveness cannot be isolated due to confounded experimental design
- No abstention benefit was demonstrated (coverage was 1.0 for both models)

**However, these results are from a 10-sample pilot and cannot support statistical or clinical claims.** The FS-005 framework provides a complete, reproducible pipeline for future evaluation, but real model inference on larger cohorts requires Google Colab T4 or similar GPU resources due to the local RTX 2050 4GB VRAM limitation.

**Phase 5 software tooling and exploratory analysis are implemented. Genuine experimental validation of the integrated pipeline remains outstanding.**

**FS-005 Phase 5 is complete as software implementation and exploratory analysis. Genuine experimental validation remains outstanding.**
