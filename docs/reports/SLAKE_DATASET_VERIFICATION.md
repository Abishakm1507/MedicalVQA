# SLAKE Dataset Verification Report

**Dataset:** SLAKE (Semantically-Labeled Knowledge-Enhanced Dataset for Medical VQA)  
**Local path:** `data/SLAKE` (drive `D:`)  
**Citation:** Liu et al., *"SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering"*, ISBI 2021  
**Inspection date:** 2026-11-13  
**Original dataset files modified:** None

---

## 1. Actual dataset structure

### 1.1 Directory layout

```
data/SLAKE/
├── imgs/                         # 642 patient folders (xmlab0 … xmlab641)
│   ├── xmlabN/
│   │   ├── source.jpg            # JPEG RGB, square (240–1024 px)
│   │   ├── mask.png              # PNG segmentation mask
│   │   ├── detection.json        # per-image bounding boxes [x, y, w, h]
│   │   └── question.json         # per-image QA pairs (duplicate of split JSON)
├── KG/                           # knowledge graphs (CSV, contain Chinese text)
│   ├── disease.csv
│   ├── en_disease.csv
│   ├── en_organ.csv
│   ├── en_organ_rel.csv
│   ├── organ.csv
│   └── organ_rel.csv
├── train.json    (9,835 entries)
├── validate.json (2,099 entries)
├── test.json     (2,094 entries)
├── mask.txt      (26 organ/lesion labels, Chinese text)
└── .DS_Store
```

### 1.2 Counts

| Item | Count |
|------|-------|
| Patient folders | 642 |
| Total split QA entries (train+validate+test) | 14,028 |
| English QA pairs | 7,033 |
| Total bounding boxes (across all detection.json) | 1,622 |
| Patients with non-empty detection.json | 580 |
| Patients with empty detection.json `[]` | 62 |

### 1.3 Image formats and dimensions

- All 642 `source.jpg`: JPEG, RGB mode
- All masks `mask.png`: PNG
- Dimensions per patient (all square):

| Dimension | Patient count |
|-----------|---------------|
| 256×256 | 30 |
| 240×240 | 140 |
| 320×320 | 10 |
| 512×512 | 282 |
| 1024×1024 | 179 |
| 288×288 | 1 |

### 1.4 Dataset size on disk

- Total `data/SLAKE/`: ~260.27 MB (272,915,956 bytes)
- `imgs/`: ~255.91 MB (268,341,374 bytes)

*(Note: `docs/research/dataset_catalog.md` claims ~1.2 GB and a `data/raw/slake` path with `masks/`/`json/` subdirectories; neither matches the actual `data/SLAKE/` layout with per-patient sidecars.)*

---

## 2. Split statistics

| Split | Entries | English | Chinese | Modality (EN) |
|-------|---------|---------|---------|---------------|
| train | 9,835 | 4,919 | 4,916 | MRI 2,208 / CT 4,819 / X-Ray 2,808 |
| validate | 2,099 | 1,053 | 1,046 | MRI 478 / CT 955 / X-Ray 666 |
| test | 2,094 | 1,061 | 1,033 | CT 865 / X-Ray 740 / MRI 489 |
| **Total** | **14,028** | **7,033** | **6,995** | — |

Split ratio: 70.1% / 15.0% / 14.9% (matches the claimed 70/15/15 protocol).

### 2.1 Field distributions (all splits combined)

- `q_lang`: en 7,033 / zh 6,995
- `modality`: MRI 3,175 / CT 6,633 / X-Ray 4,220
- `answer_type`: OPEN 8,467 / CLOSED 5,561
- `base_type`: vqa 12,288 / kvqa 1,740
- `q_type`: `None` for all 14,028 entries (not populated in source files)
- `content_type`: Organ 3,560 / Position 2,426 / Abnormality 1,995 / Modality 1,522 / Size 892 / Quantity 635 / Plane 740 / Color 397 / Shape 121 / KG 1,740
- `location` (top): Lung 5,001 / Abdomen 4,247 / Brain_Tissue 2,059 / Chest_heart 301 / Brain_Face 407 / Pelvic Cavity 591 / Brain 682 / Neck 415 / Chest_lung 362 / Chest_mediastinal 64

---

## 3. Annotation examples

### 3.1 Split JSON entry (test.json, qid 11934)

```json
{
  "img_id": 102,
  "img_name": "xmlab102/source.jpg",
  "question": "What modality is used to take this image?",
  "answer": "CT",
  "q_lang": "en",
  "location": "Lung",
  "modality": "CT",
  "answer_type": "OPEN",
  "base_type": "vqa",
  "content_type": "Modality",
  "triple": ["vhead", "_", "_"],
  "qid": 11934
}
```

### 3.2 detection.json sidebar (xmlab102)

```json
[{"Lung Cancer": [289.0, 301.0, 17.0, 22.0]}]
```

### 3.3 question.json sidecar (xmlab102, first entry)

Same fields as split JSON; 22 entries for this patient.

### 3.4 Bounding-box convention

- **Format:** `[x, y, w, h]` **XYWH**, floating-point pixels
- **Origin:** top-left
- **Example:** `{"Lung Cancer": [289.0, 301.0, 17.0, 22.0]}` → x=289, y=301, w=17, h=22
- **Project convention:** `(ymin, xmin, ymax, xmax)` = `(y, x, y+h, x+w)` = `(301, 289, 323, 306)`
- **Normalized:** `(0.5645, 0.5879, 0.5977, 0.6309)` on a 512×512 image

**Not all samples contain bounding boxes.** Only abnormality/organ/content-type entries have grounding; modality/position/plane/etc. questions have `detection.json: []` and produce zero ground-truth regions.

---

## 4. Dataset integrity

| Check | Result | Details |
|-------|--------|---------|
| Missing `source.jpg` | 0 | All 642 present and readable |
| Missing `mask.png` | 0 | All 642 present |
| Missing `detection.json` | 0 | All 642 present (62 are empty `[]`) |
| Missing `question.json` | 0 | All 642 present |
| Broken image references | 0 | All split `img_name` values resolve to existing files |
| Invalid box annotations | 0 | All 1,622 boxes are 4-element numeric lists |
| Out-of-bounds boxes | 0 | All boxes within their image dimensions |
| Missing questions | 0 | All 14,028 entries have non-empty question |
| Missing answers | 1 | train `qid=1622` (`xmlab281`, "Does the picture contain liver?") has `answer: ""` |
| Duplicate qids | 0 | All 14,028 qids unique |
| Duplicate img_ids across splits | 298 | Same image appears in multiple splits with different questions (image-level leakage) |
| No patient IDs published | — | SLAKE does not publish clinical patient IDs; `img_id` is the only leakage proxy and it is leaked across splits |

**Leakage detail:** 298 `img_id` values appear in more than one split (e.g. img_id=1 in both train and test; img_id=116 in both train and validate). Since SLAKE publishes no patient identifiers, patient-level split isolation cannot be verified from this data. The loader correctly sets `patient_id=None` and `base.py` reports `patient_ids_available=False`.

**Modality note:** All 503 chest-location entries are CT modality — there are **no Chest X-ray images** in this SLAKE subset. Consequently, Track A (Predicted CXR) never activates; only Tracks B and C are compatible.

---

## 5. Existing loader compatibility

### 5.1 `datasets/slake_loader.py` vs actual dataset

The loader is **compatible** with the actual layout:

| Loader behavior | Actual dataset | Status |
|-----------------|----------------|--------|
| Reads `dataset_dir/{split}.json` | `data/SLAKE/test.json` etc. | ✓ |
| Resolves image via `images_dir/<img_name>` | `data/SLAKE/imgs/xmlabN/source.jpg` | ✓ |
| Reads `detection.json` sidecar | `data/SLAKE/imgs/xmlabN/detection.json` | ✓ |
| Box format `[x, y, w, h]` XYWH | Confirmed in detection.json | ✓ |
| Converts to `(ymin, xmin, ymax, xmax)` | Matches loader `slake_loader.py:216` | ✓ |
| Normalizes to [0,1], rejects OOB | `normalize_bounding_box(..., clip_out_of_bounds=False)` | ✓ |
| Language filter `language="en"` | 7,033 EN / 6,995 ZH entries | ✓ |
| `patient_id=None` | SLAKE publishes no patient IDs | ✓ |
| Track B & C for all; Track A for chest+X-RAY | No chest X-ray images → Track A never activates | ✓ (data characteristic) |

### 5.2 Loader run against actual dataset (English)

| Split | Samples loaded | With regions | Total regions | Modalities |
|-------|---------------|--------------|---------------|------------|
| train | 4,919 | 4,487 | 12,023 | MRI 1,129 / CT 2,367 / X-RAY 1,423 |
| validate | 1,053 | 970 | 2,708 | MRI 233 / CT 482 / X-RAY 338 |
| test | 1,061 | 980 | 2,549 | CT 472 / X-RAY 361 / MRI 228 |

Empty `detection.json` patients correctly yield 0 ground-truth regions (e.g. `xmlab100`).

### 5.3 Compatibility problems & required corrections

1. **`dataset_catalog.md` is inaccurate** — lists local path `data/raw/slake` with `masks/`/`json/` subdirectories and ~1.2 GB footprint. Actual path is `data/SLAKE/` with per-patient `detection.json`/`mask.png`/`question.json` sidecars and ~260 MB. Update the catalog to reflect the real layout.

2. **`q_type` is `None` for all entries** — the loader coerces `str(None)` → `"None"` → defaults to `"open"` (`slake_loader.py:79`). The dataset's actual open/closed signal is `answer_type` (`OPEN`/`CLOSED`), not `q_type`. The loader silently treats every question as open. Consider mapping `answer_type` to question type as a secondary signal, or documenting that `q_type` is absent.

3. **No Chest X-ray images** — Track A is unreachable for this dataset. Not a defect, but worth documenting so Track-A-dependent evaluation code isn't expected to produce CXR results.

4. **Image-level leakage across splits** — 298 img_ids repeat across train/validate/test. The loader cannot prevent this (no patient IDs). Document as a known limitation; evaluation metrics computed per-split may be optimistic due to repeated images.

5. **One empty answer** — train `qid=1622` has `answer: ""`. The loader preserves it as an empty string; downstream metrics should handle empty answers gracefully.

---

## 6. Exact dataset paths

- Root: `D:\Abi\New folder\MedicalVQA\data\SLAKE`
- Split files: `data/SLAKE/{train,validate,test}.json`
- Images: `data/SLAKE/imgs/xmlabN/source.jpg`
- Masks: `data/SLAKE/imgs/xmlabN/mask.png`
- Boxes: `data/SLAKE/imgs/xmlabN/detection.json`
- Per-patient QA: `data/SLAKE/imgs/xmlabN/question.json`
- Knowledge graphs: `data/SLAKE/KG/{disease,en_disease,en_organ,en_organ_rel,organ,organ_rel}.csv`
- Label list: `data/SLAKE/mask.txt`

---

## 7. Summary

The actual SLAKE dataset matches the catalog's volume claims (642 images, 14,028 QA pairs, 7,033 English, 70/15/15 split) but differs in directory layout (per-patient sidecars, not separate `masks/`/`json/` dirs) and footprint (~260 MB, not ~1.2 GB). The existing `SlakeDatasetLoader` is compatible with the actual layout and correctly loads boxes, normalizes them, and assigns tracks. No modifications to the original dataset were made. The primary data-quality observations are: (a) `q_type` is absent (all `None`), (b) one empty answer, (c) image-level leakage across splits (298 img_ids), and (d) no Chest X-ray images (Track A inactive).

**No code changes were made in this session.** Corrections to `docs/research/dataset_catalog.md` and optional loader handling of `q_type`/`answer_type` are proposed above and can be implemented on request.
