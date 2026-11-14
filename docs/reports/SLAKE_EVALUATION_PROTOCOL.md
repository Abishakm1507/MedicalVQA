# SLAKE Dataset Evaluation Protocol — Leakage Investigation

**Dataset:** SLAKE (`data/SLAKE`, drive `D:`)  
**Inspection date:** 2026-11-13  
**Original dataset files modified:** None  
**Production code modified:** None  

---

## 1. Split Overlap Investigation

### 1.1 Overlap counts

| Metric | Value |
|--------|-------|
| Total unique images (`img_id`) | 642 |
| Train entries | 9,835 |
| Validate entries | 2,099 |
| Test entries | 2,094 |
| Total entries | 14,028 |
| **img_ids appearing in multiple splits (any language)** | **298** (46.4% of all images) |
| Train ∩ Validate | 130 |
| Train ∩ Test | 142 |
| Validate ∩ Test | 26 |
| Train ∩ Validate ∩ Test | 0 |

### 1.2 Root cause of the overlap: multilingual entries, not data leakage

The overlap is **entirely explained by SLAKE's bilingual (English/Chinese) structure**. Every overlapping image has its English QA pairs in exactly one split and its Chinese QA pairs in exactly one (potentially different) split. No `(img_id, language)` pair appears in more than one split.

| Language | img_ids in multiple splits | Train∩Val | Train∩Test | Val∩Test | Triple |
|----------|---------------------------|-----------|------------|----------|--------|
| English | **0** | 0 | 0 | 0 | 0 |
| Chinese | **0** | 0 | 0 | 0 | 0 |
| Any language | 298 | 130 | 142 | 26 | 0 |

**Representative examples:**

| img_id | Train | Validate | Test |
|--------|-------|----------|------|
| 0 | ZH (10 entries) | EN (10 entries) | — |
| 1 | EN (7 entries) | — | ZH (7 entries) |
| 2 | EN (9 entries) | ZH (9 entries) | — |
| 4 | — | EN (9 entries) | ZH (6 entries) |
| 13 | ZH (8 entries) | — | EN (8 entries) |

**Conclusion:** The 298 overlapping img_ids are **not traditional data leakage**. They are the same image appearing with English questions in one split and Chinese questions in another. When evaluating English-only (which is this project's configuration — `language="en"`), there is **zero image-level overlap** between any pair of official splits.

### 1.3 Same image, different questions/answers across splits

For overlapping img_ids, the questions in different splits are **different QA pairs** (different qids), typically translations of the same underlying question template into the other language. Examples (img_id=18):

- Train (ZH): "How many kidneys are there in this image?" → answer: "1"
- Test (EN): "How many kidneys are there in this image?" → answer: "1"

The same question template appears in both languages with consistent answers. There are no cases where the same `(img_id, language)` pair has contradictory questions or answers across splits — because each `(img_id, language)` pair exists in exactly one split.

---

## 2. Recommended Evaluation Protocol

### 2.1 Core principle

**Preserve the official SLAKE splits unchanged.** Do not create custom splits, do not remove samples, and do not alter train/validate/test assignments. The official splits are already leakage-free when evaluated per-language.

### 2.2 Official-split evaluation (recommended for Qwen baseline)

Use the official `test.json` split with `language="en"` filter:

| Split | English entries | Unique English images | Use case |
|-------|----------------|----------------------|----------|
| Train | 4,919 | 450 | Few-shot prompting / calibration (if needed) |
| Validate | 1,053 | 96 | Hyperparameter / gate tuning (per experiment protocol Section 4) |
| Test | 1,061 | 96 | **Final Qwen baseline evaluation** |

**Recommendation:** Use the **official test split (English, 1,061 entries, 96 unique images)** for the first Qwen smoke test and baseline evaluation. This is the standard held-out split and has zero image-level leakage when filtered to English.

### 2.3 Leakage-controlled evaluation (if ever needed)

If a future experiment requires absolute image-level isolation across all languages combined (not just English), construct a leakage-controlled subset:

1. Identify the 298 overlapping img_ids (listed in appendix `_overlap_ids_appendix.txt`).
2. For each overlapping image, assign all of its entries (both languages) to a single split based on where its English entries reside (or use a deterministic rule).
3. Remove the duplicate-language entries from the other split(s).
4. Report the reduced split sizes and clearly label results as "leakage-controlled subset."

**This is NOT recommended for the Qwen baseline** because it discards valid data and departs from the official SLAKE protocol, making results incomparable with published SLAKE benchmarks.

### 2.4 Distinction: official vs leakage-controlled

| Aspect | Official-split evaluation | Leakage-controlled evaluation |
|--------|---------------------------|-------------------------------|
| Splits | Unchanged official train/validate/test | Custom, modified splits |
| Data used | All 1,061 English test entries | Subset excluding duplicate-language entries |
| Leakage (English) | None (verified: 0 overlap) | None |
| Leakage (all languages) | 298 img_ids repeat across splits | Eliminated by construction |
| Comparability | Comparable to published SLAKE results | Not comparable to published results |
| Recommended for Qwen baseline | **Yes** | No |

### 2.5 First Qwen baseline recommendation

**Use the official test split, English only, all 1,061 entries.**

**Limitations to disclose:**
1. SLAKE publishes no clinical patient IDs; `img_id` is the only leakage proxy. While English-level leakage is zero, we cannot verify patient-level isolation (the same patient could theoretically appear with different `img_id`s).
2. The test set contains only 96 unique English images. Results have high variance; report confidence intervals (per experiment protocol Section 5: 1,000 bootstrap resamples).
3. No Chest X-ray images exist in this dataset (all chest-location images are CT). Track A (Predicted CXR) cannot be evaluated on SLAKE.
4. `q_type` is `None` for all entries; question-type classification falls back to `answer_type` (OPEN/CLOSED).

---

## 3. Other Dataset Issues

### 3.1 Empty answer

- **train `qid=1622`** (`xmlab281/source.jpg`): question "Does the picture contain liver?", `answer: ""` (empty string), `answer_type: OPEN`, `content_type: Organ`.
- All other 14,027 entries have non-empty answers.
- **Handling:** The loader preserves the empty string. Downstream metrics (accuracy, F1) must handle empty answers gracefully (e.g., treat as incorrect or exclude). Do not remove the sample.

### 3.2 `q_type` is `None`; `answer_type` contains OPEN/CLOSED

- `q_type` is `None` for all 14,028 entries across all splits.
- `answer_type` is populated: OPEN 8,467 / CLOSED 5,561.
- The loader (`slake_loader.py:79`) coerces `str(None)` → `"None"` → defaults to `"open"`, silently treating every question as open.
- **Handling:** Document this limitation. If open/closed distinction is needed for metrics, derive it from `answer_type` instead of `q_type`. No loader change made in this stage (per constraints).

### 3.3 No Chest X-ray images

- All 503 chest-location entries are CT modality (train: 503 CT, validate: 80 CT, test: 42 CT).
- There are **zero Chest X-ray images** in the entire SLAKE dataset.
- `location` values include `Chest_heart`, `Chest_lung`, `Chest_mediastinal` but all are CT, not X-Ray.
- **Handling:** Track A (Predicted CXR) never activates for SLAKE. Only Tracks B (Oracle SLAKE) and C (Answer-only) are compatible. Document this as a data characteristic, not a defect.

### 3.4 `docs/research/dataset_catalog.md` inaccuracies

The task referenced `docs/dataset_catalog.md`, but the actual file is at **`docs/research/dataset_catalog.md`**. The following fields are inaccurate:

| Field | Catalog claim | Actual value |
|-------|--------------|--------------|
| Local path | `D:/GitHub/honors-project/MedicalVQA/data/raw/slake` | `data/SLAKE` (repo-relative, on drive `D:`) |
| Layout | `masks/` and `json/` subdirectories | Per-patient sidecars: `imgs/xmlabN/{source.jpg,mask.png,detection.json,question.json}` |
| Storage footprint | ~1.2 GB uncompressed | ~260.27 MB total (`imgs/` ~255.91 MB) |
| Grounding annotations | `mask/`, `json/` | `detection.json` per patient (XYWH boxes) + `mask.png` per patient |

**Required correction:** Update `docs/research/dataset_catalog.md` Section 2.1 (SLAKE) and Section 3 (Storage Hierarchy) to reflect the actual `data/SLAKE/` layout, per-patient sidecar format, and ~260 MB footprint.

---

## 4. Documentation files changed

| File | Action |
|------|--------|
| `docs/reports/SLAKE_DATASET_VERIFICATION.md` | Created earlier (dataset structure, integrity, loader compatibility) |
| `docs/reports/_overlap_analysis.txt` | Created (appendix: overlap analysis details) |
| `docs/reports/_overlap_ids_appendix.txt` | Created (appendix: full list of 298 overlapping img_ids) |
| `docs/reports/SLAKE_EVALUATION_PROTOCOL.md` | **This report** (created now) |
| `docs/research/dataset_catalog.md` | **Not yet modified** — corrections identified but not applied (pending review) |
| `scripts/investigate_overlap.py` | Temporary analysis script (not part of deliverables) |
| `scripts/analyze_lang_overlap.py` | Temporary analysis script (not part of deliverables) |

---

## 5. Outstanding questions and risks

1. **Patient-level isolation cannot be verified.** SLAKE publishes no clinical patient IDs. Two different `img_id`s could theoretically belong to the same patient. The loader correctly reports `patient_ids_available=False`.
2. **`dataset_catalog.md` needs updating** — path, layout, and footprint are inaccurate. Correction is documentation-only; no production code impact.
3. **`q_type` absence** — if closed-vs-open accuracy is reported separately, derive question type from `answer_type` rather than `q_type`. This is a loader enhancement, not a blocker for the baseline.
4. **Small English test set (96 unique images)** — results will have high variance; use bootstrap confidence intervals per experiment protocol Section 5.
5. **Track A (CXR) is unevaluable on SLAKE** — no chest X-ray images exist. CXR baseline must use a different dataset (e.g., VQA-RAD or VinDr-CXR).

---

## 6. Readiness for Qwen smoke test

**The dataset is ready for the Qwen smoke test.**

- Official splits are preserved unchanged.
- English-only evaluation has **zero image-level leakage** across train/validate/test (verified: EN Train∩Val = 0, EN Train∩Test = 0, EN Val∩Test = 0).
- The `SlakeDatasetLoader` is compatible with the actual `data/SLAKE/` layout and correctly loads 1,061 English test samples with normalized bounding boxes and track assignments.
- All integrity checks pass: no missing images, no broken references, no invalid/out-of-bounds boxes, no duplicate qids.

**Recommended configuration for the first Qwen baseline:**
- Split: `test`, language `"en"` (1,061 entries, 96 unique images)
- Model: Qwen2.5-VL-3B via existing VLMClient interface
- Metrics: accuracy (closed), BLEU/Token F1 (open), with bootstrap confidence intervals
- Report limitations: no CXR images, no patient IDs, small test set

**Do not proceed to the Qwen smoke test.** Wait for review of this report.
