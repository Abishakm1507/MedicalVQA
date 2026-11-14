# FS-005 Phase 5 — Experiment Readiness Assessment

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Stage:** Stage 1 — Readiness and Protocol  
**Document ID:** `docs/reports/FS-005_PHASE_5_READINESS_ASSESSMENT.md`  
**Status:** Documentation-only — no inference executed  

---

## 1. Existing Project State

### FS-004 Pilot Summary (Previously Evaluated)

The FS-004 paired pilot evaluated 10 SLAKE test samples on both Qwen2.5-VL-3B and MedGemma-4B:

| Metric | Qwen2.5-VL-3B | MedGemma-4B |
|--------|---------------|-------------|
| Overall accuracy | 0.60 (6/10) | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) | 0.40 (2/5) |
| CLOSED accuracy | 0.80 (4/5) | 1.00 (5/5) |
| Correct predictions | 6 | 7 |
| Incorrect predictions | 4 | 3 |
| Failed predictions | 0 | 0 |
| Avg inference time | 15.91 s | 5.31 s |

**Pilot QIDs (10):** 11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919  
**Pilot images (9 unique):** 130, 185, 224, 234, 312, 320, 344, 561, 605  
**Inference hardware:** Qwen via local Ollama; MedGemma via Kaggle Tesla T4  

---

## 2. Dataset Audit

### SLAKE (Primary Dataset)

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
| Image-level overlap (any language) | 298 img_ids (46.4%) — all bilingual, not English-only |
| Patient IDs published | None |
| Chest X-ray images | 0 (all chest-location entries are CT) |
| Total bounding boxes | 1,622 |
| Patients with empty detection.json | 62 |

### Track Eligibility by Dataset

| Track | Requirement | SLAKE | VQA-RAD | RSNA | VinDr-CXR |
|-------|-------------|-------|---------|------|-----------|
| Track A (predicted CXR detector) | CXR images | **Excluded** — no CXR images | TBD — requires CXR subset | TBD — requires CXR subset | TBD — requires CXR subset |
| Track B (oracle GT boxes) | Usable GT region annotations | **Eligible** — 980/1,061 English test samples have regions | TBD | TBD | TBD |
| Track C (ungrounded baseline) | No region requirement | **Eligible** — all 1,061 English test samples | TBD | TBD | TBD |

### Previously Evaluated Samples (FS-004 Pilot)

The following 10 SLAKE test samples were already evaluated in FS-004 and **must be excluded** from any new confirmatory test cohort:

| QID | Image ID | Modality | Location | Answer Type | Question |
|-----|----------|----------|----------|-------------|----------|
| 11981 | 130 | X-Ray | Lung | OPEN | Which part of the body does this image belong to? |
| 12073 | 185 | X-Ray | Lung | CLOSED | Are there abnormalities in this image? |
| 12111 | 224 | CT | Abdomen | CLOSED | Does the picture contain spleen? |
| 12129 | 234 | CT | Abdomen | OPEN | How many kidneys in this image? |
| 12312 | 312 | X-Ray | Lung | OPEN | Where is the abnormality in this image? |
| 12344 | 320 | X-Ray | Lung | OPEN | Which organs appear in pairs? |
| 12384 | 344 | X-Ray | Lung | OPEN | What imaging modality was used? |
| 12845 | 561 | CT | Abdomen | CLOSED | Are there any organs used for breathing in this image? |
| 12914 | 605 | CT | Pelvic Cavity | CLOSED | Does the picture contain rectum? |
| 12919 | 605 | CT | Pelvic Cavity | CLOSED | Does the picture contain the organ which has the effect of excreting feces? |

**Images excluded from new cohort:** 130, 185, 224, 234, 312, 320, 344, 561, 605 (9 unique images)

### Remaining Eligible SLAKE Test Samples

| Metric | Value |
|--------|-------|
| English test samples | 1,061 |
| Minus FS-004 pilot QIDs | 10 |
| Minus pilot image IDs | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| English test samples with GT regions | 0 (no detection.json files exist) |
| Samples without GT regions | 945 |
| Remaining unique images | 87 |

---

## 3. Model and Checkpoint Inventory

### Available VLM Clients (Code-Implemented)

| Model | Client Class | Model ID | Gated? | Quantization | Status |
|-------|-------------|----------|--------|--------------|--------|
| Qwen2.5-VL-3B | `QwenVLMClient` | `Qwen/Qwen2.5-VL-3B-Instruct` | No | 4-bit optional | Code ready |
| MedGemma-4B | `MedGemmaVLMClient` | `google/medgemma-4b-it` | Yes (HF gate) | 4-bit optional | Code ready |
| Mock (testing) | `MockVLMClient` | N/A | No | N/A | Code ready |
| Ollama Qwen | `OllamaVLMClient` | `qwen2.5vl:3b` | No | Q4_K_M | Code ready |

### Checkpoint Status

| Model | Local Checkpoint | Remote Access | Inference Hardware |
|-------|-----------------|---------------|-------------------|
| Qwen2.5-VL-3B | Not downloaded locally | Hugging Face (free, no gate) | RTX 2050 4GB VRAM **insufficient** for 4-bit; Colab T4 required |
| MedGemma-4B | Not downloaded locally | Hugging Face (gated, requires HF token + license acceptance) | RTX 2050 4GB VRAM **insufficient**; Colab T4 required |
| MedSAM | Not downloaded locally | Hugging Face (free) | RTX 2050 4GB VRAM marginal; CPU fallback available |

### Inference Backend Status

| Backend | Status | Notes |
|---------|--------|-------|
| Local PyTorch (RTX 2050) | Available but VRAM-constrained | 4GB VRAM insufficient for 4-bit VLMs; CPU fallback too slow |
| Google Colab (T4 15.3GB) | Available as remote worker | Primary environment for model experiments |
| Ollama (local) | Available for Qwen | `qwen2.5vl:3b` tag used in FS-004 |

---

## 4. Detector and Segmentation Status

### Region Detector

| Component | Status | Notes |
|-----------|--------|-------|
| `BaseRegionDetector` (abstract) | Implemented | Abstract interface |
| `MockRegionDetector` | Implemented | Deterministic mock for testing only |
| `FasterRCNNDetector` | Referenced in code | Medical fine-tuned checkpoint needed; not yet available locally |
| RT-DETR / Torchvision Faster R-CNN | Not downloaded | Apache 2.0 / BSD-3-Clause compliant options available |

**Constraint:** AGPL-3.0 (Ultralytics YOLOv8) is **excluded** per AGENTS.md licensing constraint.

### MedSAM Segmenter

| Component | Status | Notes |
|-----------|--------|-------|
| `MedSAMRegionSegmenter` | Implemented | Wraps MedSAM promptable segmentation |
| MedSAM checkpoint | Not downloaded locally | Apache 2.0; available on Hugging Face |
| CPU fallback | Available | ~1.5s per image on CPU |

**Segmenter role:** MedSAM is a prompt-based segmenter, **not** an object detector. Boxes come from the detector (Track A) or GT (Track B); MedSAM provides segmentation masks for visual overlay only.

---

## 5. Compute Resource Assessment

### Local Environment (Laptop)

| Resource | Specification | Adequacy for Phase 5 |
|----------|---------------|---------------------|
| GPU | NVIDIA RTX 2050, 4 GB VRAM | **Insufficient** for 4-bit VLMs (~5-6 GB needed each) |
| System RAM | 16 GB | Adequate for CPU mock tests, data processing |
| Disk (D: drive) | 71.42 GB free | Adequate for experiment artifacts |
| Disk (C: drive) | 18.62 GB free | Limited; avoid large downloads |

### Remote Environment (Google Colab)

| Resource | Specification | Notes |
|----------|---------------|-------|
| GPU | Tesla T4, 15.3 GB VRAM | Sufficient for 4-bit VLMs (one at a time) |
| Primary use | Larger model experiments | Sequential model loading required |

### Sequential Model Loading Requirement

Per ADR 0005 and FS-005 design: Qwen, MedGemma, and MedSAM must **never** be resident in GPU VRAM simultaneously. Sequential on-demand loading is required.

---

## 6. Track-by-Track Feasibility

### Track A — End-to-End Predicted Grounding (CXR Detector)

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Excluded** — SLAKE contains zero chest X-ray images |
| Alternative CXR data | Not available locally; would require VQA-RAD CXR or RSNA/VinDr-CXR |
| Detector checkpoint | Not downloaded locally; medical fine-tuned checkpoint needed |
| Recommendation | **Track A inactive for SLAKE.** Consider VQA-RAD CXR subset if data available and accessible. |

### Track B — Oracle Ground-Truth Grounding

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Eligible** — 980/1,061 English test samples have GT regions |
| Grounding source | Dataset GT boxes (promoted to CandidateBox) |
| Runtime grounding | No IoU comparison at runtime; structural validity only |
| Evaluation pointing accuracy | Computed offline via IoU ≥ 0.5 against GT boxes |
| Recommendation | **Track B viable for SLAKE** — primary recommended track |

### Track C — Ungrounded Baseline

| Aspect | Status |
|--------|--------|
| SLAKE eligibility | **Eligible** — all 1,061 English test samples |
| VQA-RAD eligibility | **Eligible** — all samples (no region annotations) |
| Region requirement | None |
| Recommendation | **Track C viable for SLAKE and VQA-RAD** |

---

## 7. Experiment-Readiness Assessment

### Readiness Summary

| Component | Status | Blockers |
|-----------|--------|----------|
| SLAKE dataset (English test split) | **Ready** | None |
| SLAKE validation split (for calibration) | **Ready** | 1,053 English entries available |
| Qwen2.5-VL-3B client | **Code ready** | Requires model download + Colab T4 |
| MedGemma-4B client | **Code ready** | Requires HF token + model download + Colab T4 |
| Region detector (medical) | **Blocked** | No trained checkpoint available locally; YOLOv8 excluded by licensing |
| MedSAM segmenter | **Code ready** | Checkpoint not downloaded; optional for visual overlay |
| Sequential model loading | **Implemented** | `RemoteModelWorker` loads one model at a time |
| Semantic clustering | **Code ready** | Falls back to canonical normalization (sentence-transformers not installed) |
| Confidence calibration | **Code ready** | `ConfidenceCalibrator` grid-search implementation available |
| Evaluation metrics | **Code ready** | All 9 metrics implemented in `evaluation/metrics.py` |
| Artifact generation | **Code ready** | `EvaluationRunner` writes JSON, CSV, Markdown |

### Recommended Experimental Plan

#### Track A (Predicted CXR Grounding)
- **SLAKE:** Not applicable — no CXR images in dataset
- **Recommendation:** Skip Track A for SLAKE; document as ineligible
- **Alternative:** Consider VQA-RAD CXR subset if data becomes available

#### Track B (Oracle GT Grounding)
- **SLAKE test split:** Eligible — 1,061 English samples, 980 with GT regions
- **SLAKE validation split:** Eligible — 1,053 English samples, 970 with GT regions (for calibration)
- **Recommendation:** Primary track for SLAKE evaluation

#### Track C (Ungrounded Baseline)
- **SLAKE test split:** Eligible — all 1,061 English samples
- **VQA-RAD:** Eligible if dataset accessible
- **Recommendation:** Secondary baseline track

#### Answer Conditions
| Condition | N | Temperature | Use Case |
|-----------|-----|-------------|----------|
| Condition 1 (free-form) | 1 | 0.0 | Baseline accuracy |
| Condition 2 (grounded, no abstention) | 5 | 0.7 | Grounded answer accuracy |
| Condition 3 (grounded + abstention) | 5 | 0.7 | Selective prediction with confidence gate |

---

## 8. Previously Evaluated Samples — Exclusion List

The following 10 SLAKE test samples were evaluated in FS-004 and **must be excluded** from any new confirmatory test cohort:

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

**Images excluded from new cohort:** 130, 185, 224, 234, 312, 320, 344, 561, 605

---

## 9. Proposed Experimental Cohort

### Recommended Cohort (Track B + Track C on SLAKE Test Split)

| Component | Specification |
|-----------|---------------|
| Dataset | SLAKE English test split |
| Total samples | 1,061 |
| Minus FS-004 pilot QIDs | 10 |
| Minus pilot image IDs | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| English test samples with GT regions | 0 (no detection.json files exist) |
| Samples without GT regions | 945 |
| Remaining unique images | 87 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

### Calibration Cohort (SLAKE Validation Split)

| Component | Specification |
|-----------|---------------|
| Dataset | SLAKE English validation split |
| Total samples | 1,053 |
| Samples with GT regions | 970 |
| Use | Weight (w1, w2, w3) and threshold (tau) fitting |
| **Constraint:** Validation data never touches test set |

---

## 10. Readiness Summary

| Readiness Area | Status | Notes |
|---------------|--------|-------|
| Dataset (SLAKE English test) | **Ready** | 1,051 eligible samples after pilot exclusion |
| Dataset (SLAKE validation) | **Ready** | 1,053 samples for calibration |
| Qwen2.5-VL-3B inference | **Conditionally ready** | Requires Colab T4 + model download |
| MedGemma-4B inference | **Conditionally ready** | Requires HF token + Colab T4 + model download |
| Region detector (medical) | **Blocked** | No trained checkpoint; YOLOv8 excluded by licensing |
| MedSAM segmenter | **Code ready** | Optional for visual overlay; checkpoint not downloaded |
| Local GPU (RTX 2050 4GB) | **Insufficient** | Cannot run 4-bit VLMs locally |
| Remote GPU (Colab T4) | **Available** | Primary environment for model inference |
| Calibration code | **Code ready** | `ConfidenceCalibrator` implemented |
| Evaluation metrics | **Code ready** | All 9 metrics implemented |
| Artifact generation | **Code ready** | `EvaluationRunner` implemented |

### Recommended Next Step

Proceed to **Stage 2 (Calibration)** using the SLAKE validation split (1,053 English samples). Fit confidence weights $(w_1, w_2, w_3)$ and abstention threshold $\tau$ using the selective-risk objective with minimum coverage constraint. Freeze the selected configuration before evaluating on the held-out test split.

**Blocker:** MedGemma-4B requires Hugging Face license acceptance and HF token. Qwen2.5-VL-3B requires model download (~7 GB for 4-bit). Both require Colab T4 access due to local RTX 2050 4GB VRAM limitation.

---
