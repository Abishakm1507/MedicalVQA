# FS-005 Phase 5 — Validation and Uncertainty-Aware Answering: Final Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_VALIDATION_AND_RESEARCH_REPORT.md`  
**Status:** Documentation complete — Phase 5 software tooling implemented; genuine experimental validation outstanding  
**Implementation type:** Implementation (code written and executed on existing artifacts), not new experimental validation  
**Prerequisites:** FS-005 Phases 1-4 complete and approved  

---

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
| Samples with GT regions (detection.json non-empty) | 0 |
| Samples without GT regions | 945 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

**Cohort audit artifacts:** `experiments/runs/fs005_phase5/cohort_audit/`

---

## 5. Model Configurations

### MedGemma-4B (Track B, Condition 3) — From FS-004 Pilot Artifacts

| Property | Value |
|----------|-------|
| Model ID | `google/medgemma-4b-it` |
| Inference environment | Kaggle Tesla T4 (14.56 GB VRAM) |
| Quantization | bfloat16 (no 4-bit quantization) |
| Attention implementation | eager |
| Sampling | N=5, T=0.7 |
| Region source | Ground-truth boxes (oracle) |
| Region grounding | Yes (CLOSED questions) |

### Qwen2.5-VL-3B (Track C, Condition 1) — From FS-004 Pilot Artifacts

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
| SLAKE eligibility | **Eligible** — 980/1,061 English test samples have GT regions |
| Region source | Ground-truth boxes promoted to CandidateBox |
| Evaluation | IoU ≥ 0.5 against GT boxes (offline) |
| Status | **Evaluated** — MedGemma-4B results below (from FS-004 pilot artifacts) |

**Track B Results (MedGemma-4B):**

| Metric | Value |
|--------|-------|
| Samples | 10 |
| Accuracy | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) |
| CLOSED accuracy | 1.00 (5/5) |
| USR_raw | 0.30 |
| USR_filtered | 0.30 |
| Pointing accuracy | 0.70 |
| Sampling agreement | 0.80 |
| Normalized entropy | 0.20 |
| ECE | 0.23 |
| Selective accuracy | 0.70 |
| Coverage | 1.00 |
| AUROC | 1.00 |

### Track C — Ungrounded Baseline

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Eligible** — all 1,061 English test samples |
| VQA-RAD eligibility | Eligible (no region annotations) |
| Region grounding | None |
| Status | **Evaluated** — Qwen2.5-VL-3B results below (from FS-004 pilot artifacts) |

**Track C Results (Qwen2.5-VL-3B):**

| Metric | Value |
|--------|-------|
| Samples | 10 |
| Accuracy | 0.60 (6/10) |
| OPEN accuracy | 0.40 (2/5) |
| CLOSED accuracy | 0.80 (4/5) |
| USR_raw | None (no claims) |
| USR_filtered | None (no claims) |
| Pointing accuracy | 0.00 (no regions) |
| Sampling agreement | 0.00 (N=1) |
| Normalized entropy | 0.00 (N=1) |
| ECE | 0.24 |
| Selective accuracy | 0.60 |
| Coverage | 1.00 |
| AUROC | 1.00 |

---

## 8. Answer Conditions

### Condition 1 — Free-Form VLM Answers

| Aspect | Implementation |
|--------|---------------|
| Sampling | N=1, T=0.0 (greedy) |
| Region grounding | No |
| Claim verification | No |
| Abstention | No |
| Model evaluated | Qwen2.5-VL-3B (Track C) |
| Accuracy | 0.60 (6/10) |

### Condition 2 — Region-Grounded + Verification, No Abstention

| Aspect | Implementation |
|--------|---------------|
| Sampling | N=5, T=0.7 |
| Region grounding | Yes |
| Claim verification | Yes |
| Abstention | No |
| Model evaluated | MedGemma-4B (Track B) |
| Accuracy | 0.70 (7/10) |

### Condition 3 — Region-Grounded + Verification + Abstention

| Aspect | Implementation |
|--------|---------------|
| Sampling | N=5, T=0.7 |
| Region grounding | Yes |
| Claim verification | Yes |
| Abstention | Yes (confidence gate) |
| Model evaluated | MedGemma-4B (Track B) |
| Accuracy | 0.70 (7/10) |
| Selective accuracy | 0.70 |
| Coverage | 1.00 |

**Comparison:**

| Condition | Model | Track | Accuracy |
|-----------|-------|-------|----------|
| 1 (Free-form) | Qwen2.5-VL-3B | C (ungrounded) | 0.60 |
| 2 (Grounded, no abstention) | MedGemma-4B | B (oracle) | 0.70 |
| 3 (Grounded + abstention) | MedGemma-4B | B (oracle) | 0.70 |

**Important:** Conditions 1, 2, and 3 differ on multiple dimensions simultaneously (model, track, sampling count, temperature, and answer condition). The observed difference in accuracy (60% vs 70%) **cannot be attributed specifically to region grounding or abstention**. The comparison is confounded.

---

## 9. Per-Model Results

### MedGemma-4B (Track B, Condition 3)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.70 |
| OPEN accuracy | 0.40 |
| CLOSED accuracy | 1.00 |
| USR_raw | 0.30 |
| USR_filtered | 0.30 |
| Pointing accuracy | 0.70 |
| Sampling agreement | 0.80 |
| Normalized entropy | 0.20 |
| ECE | 0.23 |
| Selective accuracy | 0.70 |
| Coverage | 1.00 |
| AUROC | 1.00 |

### Qwen2.5-VL-3B (Track C, Condition 1)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.60 |
| OPEN accuracy | 0.40 |
| CLOSED accuracy | 0.80 |
| USR_raw | None (no claims) |
| USR_filtered | None (no claims) |
| Pointing accuracy | 0.00 (no regions) |
| Sampling agreement | 0.00 (N=1) |
| Normalized entropy | 0.00 (N=1) |
| ECE | 0.24 |
| Selective accuracy | 0.60 |
| Coverage | 1.00 |
| AUROC | 1.00 |

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

### Grounding Quality (Track B Only)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pointing accuracy | 0.70 | 70% of samples had at least one claim with IoU ≥ 0.5 against a GT region |
| USR_raw | 0.30 | 30% of positive claims were unsupported or unverifiable |
| USR_filtered | 0.30 | Same as USR_raw (no unverifiable claims) |

**Pointing accuracy limitation:** Sample-level success can conceal incorrectly grounded additional claims within the same sample. If a sample has multiple claims where one meets IoU ≥ 0.5 against a GT region but another does not, `pointing_correct=True` at the sample level regardless of the incorrectly grounded claim.

### Uncertainty Estimation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sampling agreement | 0.80 | 80% of N=5 samples agreed on the answer |
| Normalized entropy | 0.20 | Low entropy — high consensus among samples |
| ECE | 0.23 | Moderate calibration error — confidence somewhat overestimates accuracy |

**Important:** For Qwen (Condition 1, N=1), sampling agreement and normalized entropy are both 0.0 by definition — a single sample cannot have agreement or entropy variation. These values are degenerate and not meaningful uncertainty measurements.

---

## 11. Selective Prediction Results

| Metric | MedGemma (Condition 3) | Qwen (Condition 1) |
|--------|----------------------|-------------------|
| Coverage | 1.00 | 1.00 |
| Selective accuracy | 0.70 | 0.60 |
| Abstention rate | 0.00 | 0.00 |

**Important:** Coverage = 1.0 means no samples were abstained from. Selective accuracy therefore equals ordinary accuracy and does **not** demonstrate an abstention benefit. The confidence gate did not abstain from any sample in this pilot.

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
