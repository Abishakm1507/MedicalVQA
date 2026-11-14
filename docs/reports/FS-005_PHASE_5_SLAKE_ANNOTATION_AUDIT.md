# FS-005 Phase 5 — SLAKE Annotation Audit and Track B Readiness

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_SLAKE_ANNOTATION_AUDIT.md`  
**Status:** Annotation audit complete  

---

## 1. Files Inspected

| File | Role |
|------|------|
| `data/SLAKE/test.json` | Official SLAKE test split (2,094 entries, 1,061 English) |
| `data/SLAKE/validate.json` | Official SLAKE validation split |
| `data/SLAKE/imgs/xmlab{img_id}/detection.json` | Ground-truth region annotations |
| `scripts/fs005_phase5_cohort_audit.py` | Cohort audit script (fixed and re-run) |
| `experiments/runs/fs005_phase5/cohort_audit/` | Cohort audit artifacts |

---

## 2. SLAKE Dataset Structure and Annotation Schema

### 2.1 Directory Structure

The SLAKE dataset organizes images by ID under `data/SLAKE/imgs/xmlab{img_id}/`. Each image directory contains:

| File | Description |
|------|-------------|
| `source.jpg` | The source medical image |
| `question.json` | Question-answer pair metadata |
| `detection.json` | Ground-truth region bounding boxes |
| `mask.png` | Segmentation mask |

### 2.2 Detection JSON Schema

Each `detection.json` file contains a JSON array of objects. Each object maps an anatomical region name to its bounding box coordinates `[ymin, xmin, ymax, xmax]`.

**Example — xmlab130/detection.json:**
```json
[{"Pneumonia": [703.0, 616.0, 218.0, 249.0]}]
```

**Example — xmlab1/detection.json:**
```json
[{"Liver": [54.0, 106.0, 30.0, 31.0]}]
```

**Example — xmlab25/detection.json (empty):**
```json
[]
```

All detection.json files are valid JSON arrays. There are no malformed or missing detection.json files in the SLAKE dataset.

---

## 3. Cohort Audit Results

### 3.1 SLAKE English Test Split

| Metric | Count |
|--------|-------|
| Total English test samples | 1,061 |
| FS-004 pilot QIDs excluded | 10 |
| Pilot image IDs excluded | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |
| **Remaining eligible samples** | **945** |
| Samples with GT regions (non-empty detection.json) | 864 |
| Samples without GT regions (empty detection.json `[]`) | 81 |
| Samples missing detection.json file | 0 |
| Samples with malformed detection.json | 0 |
| Remaining unique images | 87 |
| OPEN questions | 572 |
| CLOSED questions | 373 |

### 3.2 SLAKE English Validation Split

| Metric | Count |
|--------|-------|
| Total English validation samples | 1,053 |
| Samples with GT regions | 970 |
| Samples without GT regions (empty detection.json) | 83 |
| Samples missing detection.json file | 0 |
| Samples with malformed detection.json | 0 |
| Remaining unique images | 93 |

### 3.3 Pilot Exclusion Verification

The FS-004 pilot evaluated 10 samples with these QIDs: 11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919.

All 10 pilot QIDs were correctly excluded from the eligible cohort. Additionally, all samples sharing the 9 pilot image IDs (130, 185, 224, 234, 312, 320, 344, 561, 605) were excluded, ensuring no image-level overlap between the pilot cohort and the evaluation cohort.

### 3.4 Annotation Quality

All detection.json files are valid JSON arrays. There are:
- **0 samples** with missing detection.json files
- **0 samples** with malformed detection.json files
- **81 eligible test samples** with explicitly empty detection.json `[]` (no annotated regions)
- **83 validation samples** with explicitly empty detection.json `[]` (no annotated regions)

An empty detection.json `[]` means the image has no annotated regions — this is an explicit absence annotation, not missing data.

---

## 4. Track B Readiness Determination

### Track B Eligibility

Track B (Oracle Ground-Truth Grounding) uses dataset ground-truth boxes as oracle localization inputs. For a sample to be eligible for Track B, it must have non-empty ground-truth region annotations.

| Split | Eligible Samples | Samples with GT Regions | Samples Without GT Regions | Coverage |
|-------|-----------------|------------------------|---------------------------|----------|
| English Test | 945 | 864 | 81 | 91.4% |
| English Validation | 1,053 | 970 | 83 | 92.1% |

### Track B Readiness: READY

Track B is **ready** for genuine validation experiments on the SLAKE test split:
- 864 eligible samples have ground-truth region annotations
- 81 samples (8.6%) have no annotated regions and are excluded from pointing-accuracy evaluation
- The 864-sample cohort provides substantial coverage for meaningful Track B evaluation

### Track C Eligibility

Track C (Ungrounded Baseline) requires no region grounding. All 945 eligible test samples are eligible for Track C evaluation.

---

## 5. Verification of Pilot Exclusion

The cohort audit correctly excludes:

- **All 10 pilot QIDs:** 11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919
- **All questions sharing pilot image IDs:** Any sample whose `img_id` is in {130, 185, 224, 234, 312, 320, 344, 561, 605}

This ensures no image-level leakage between the FS-004 pilot cohort and the new evaluation cohort.

---

## 6. Recommendations for Next Experimental Step

### Immediate Next Step

**Track B (Oracle Ground-Truth Grounding) on SLAKE Test Split** is ready for execution:

1. **945 eligible samples** available (864 with GT regions, 81 without)
2. **MedGemma-4B** can be run on Google Colab Tesla T4 (15.3 GB VRAM) using sequential model loading
3. **Qwen2.5-VL-3B** can also be run via Ollama for Track C comparison

### Recommended Experiment Sequence

1. **Track C baseline** (Qwen2.5-VL-3B, ungrounded): Run on all 945 eligible SLAKE test samples
2. **Track B evaluation** (MedGemma-4B, oracle grounding): Run on 864 eligible samples with GT regions
3. **Compare results** between Track C and Track B on the same cohort
4. **Confidence calibration**: Use the validation split (970 samples with GT regions) to fit confidence weights and abstention threshold

---

## 7. Files Created and Modified

### Created

| File | Purpose |
|------|---------|
| `scripts/fs005_phase5_cohort_audit.py` | Cohort audit script (corrected with proper `imgs/xmlab{img_id}/detection.json` path) |
| `experiments/runs/fs005_phase5/cohort_audit/cohort_summary.json` | Cohort audit summary |
| `experiments/runs/fs005_phase5/cohort_audit/included_qids.csv` | Included QIDs (945) |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_qids.csv` | Excluded QIDs (10 pilot) |
| `experiments/runs/fs005_phase5/cohort_audit/excluded_image_ids.json` | Excluded image IDs (9) |
| `docs/reports/FS-005_PHASE_5_SLAKE_ANNOTATION_AUDIT.md` | This report |

### Not Modified

- `experiments/runs/slake_baseline_ollama/` — original Qwen baseline artifacts unchanged
- `experiments/runs/medgemma_baseline/` — original MedGemma baseline artifacts unchanged
- `data/SLAKE/` — original dataset unchanged
- FS-001 through FS-004 — not modified
- ADR 0004, ADR 0005 — not modified

---

## 8. Test Results

Ruff lint check on `scripts/fs005_phase5_cohort_audit.py`: **All checks passed.**

All existing FS-005 phase tests pass: **197 tests passed** across test suites for Phase 1-4.

---

## 9. Final Decision

**Track B is READY for genuine validation experiments on the SLAKE test split.**

- 864 eligible samples with ground-truth region annotations are available
- The cohort is image-disjoint from the FS-004 pilot (no image-level leakage)
- All annotation files are valid JSON arrays (no malformed files)
- No detection.json files are missing

The next step is to run real MedGemma-4B inference on the 864 eligible Track B samples using Google Colab Tesla T4, followed by Qwen2.5-VL-3B inference on all 945 eligible samples for Track C.
