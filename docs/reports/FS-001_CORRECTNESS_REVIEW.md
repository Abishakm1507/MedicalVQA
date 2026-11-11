# FS-001 Independent Correctness Review & Scientific Audit

**Audit Date**: November 11, 2026  
**Auditor Roles**: Senior ML Engineer, Multimodal Architect, & Research Methodology Auditor  
**Feature Target**: FS-001 (Dataset Ingestion, Standardized Loaders & Split Isolation)  
**Branch**: `feat/phase-1-data-localization`  
**Prior Commit Audited**: `6f71fcf`  
**Review Decision**: **Defects Corrected, Audited, and Re-Verified**

---

## 1. Issues Identified in Initial Implementation

### Issue 1.1: Scientifically Invalid Patient ID Proxying in SLAKE and VinDr
* **Problem**: In the initial commit `6f71fcf`, `SlakeDatasetLoader` assigned `patient_id = f"slake_case_{img_id}"`, and `VinDrCXRDatasetLoader` assigned `patient_id = f"vindr_patient_{image_id[:12]}"`.
* **Scientific Violation**:
  - In SLAKE, `img_id` is merely a sequential image index, not a clinical patient identifier. Treating image IDs as patient IDs creates a false guarantee of patient-level split isolation.
  - In VinDr-CXR, `image_id` is an MD5 hash of the DICOM file. Taking the first 12 characters of a hash produces arbitrary pseudorandom string fragments, not a true clinical patient identity.
  - Claiming "patient-level split leakage prevention" on datasets that do not publish clinical patient IDs compromises research integrity.

### Issue 1.2: Coordinate Clamping Hazard (Silent Annotation Distortion)
* **Problem**: The original `normalize_bounding_box` function clamped out-of-bounds coordinates to `[0.0, 1.0]` using `min/max` without logging or rejection options.
* **Impact**: If an annotation has negative coordinates or extends substantially beyond image boundaries, silently clamping it distorts the true center, aspect ratio, and area of the bounding box without alerting the researcher.

### Issue 1.3: Insufficient Round-Trip & Aspect Ratio Verification
* **Problem**: Bounding box coordinate transformations lacked mathematical round-trip consistency guarantees (`pixels -> normalized -> pixels`) across non-square images (e.g. 1920x1080 or 800x600).

### Issue 1.4: Track Compatibility Enforcement Gap
* **Problem**: Although `EvaluationTrack` was defined in the schema, samples lacked programmatic assertion methods to prevent non-CXR modalities (e.g. Brain MRI or Abdominal CT in SLAKE) or unannotated datasets (VQA-RAD) from accidentally entering Track A (predicted CXR localization).

---

## 2. Corrections Implemented

1. **Explicit Patient Identity Decoupling**:
   - `core/schemas.py`: Separated `image_id: str` (mandatory image-level identifier) from `patient_id: str | None` (strictly clinical patient identity).
   - `datasets/slake_loader.py`: Strictly sets `patient_id = None`. SLAKE image IDs are recorded exclusively in `image_id`.
   - `datasets/vindr_loader.py`: Strictly sets `patient_id = None`. Removed the arbitrary `image_id[:12]` hash slicing heuristic.
   - `datasets/vqarad_loader.py`: Strictly sets `patient_id = None`.
   - `datasets/rsna_loader.py`: Correctly preserves official `patientId` as both `patient_id` and `image_id` (RSNA is the only dataset among the four with official clinical patient IDs).
2. **Patient Isolation Reporting (`check_patient_leakage`)**:
   - Updated `check_patient_leakage` to return a structured `PatientIsolationReport`.
   - If samples have `patient_id is None`, the report explicitly outputs `patient_ids_available: False`, accurately documenting that patient-level isolation cannot be evaluated rather than falsely claiming absence of leakage.
3. **Strict Coordinate Normalization & Rejection**:
   - Added `clip_out_of_bounds: bool = False` (default strict). Out-of-bounds annotations exceeding a 1-pixel rounding tolerance are **rejected** with an explicit warning log rather than silently mutated.
   - Added degenerate box checking ($xmax \le xmin$ or $ymax \le ymin$).
   - Added `denormalize_bounding_box` utility to support exact round-trip pixel reconstruction.
4. **Programmatic Track Compatibility Enforcers**:
   - Added `is_compatible_with(track)` and `assert_track_compatibility(track)` to `MedicalVQASample`.
   - Incompatible track evaluation raises an immediate `ValueError` with clear clinical context.
5. **Real Dataset Smoke-Test Tooling**:
   - Implemented `datasets/validate_dataset.py`: A standalone CLI tool to validate directory structure, image formats, annotation schemas, and bounding box distributions on real external datasets.

---

## 3. Files Created and Modified

```
MedicalVQA/
├── core/
│   └── schemas.py                            # Added image_id, patient_id=None, track assertion methods
├── datasets/
│   ├── base.py                               # Strict coordinate normalization, denormalize, PatientIsolationReport
│   ├── slake_loader.py                       # Corrected patient_id=None, separated image_id, Track B/C isolation
│   ├── vqarad_loader.py                      # Corrected patient_id=None, explicit has_ground_truth_regions=False
│   ├── rsna_loader.py                        # Official patientId preservation, non-square coordinate handling
│   ├── vindr_loader.py                       # Corrected patient_id=None, removed hash slicing heuristic
│   ├── validate_dataset.py                   # Real dataset smoke-test and directory verification CLI
│   └── __init__.py                           # Exported denormalize_bounding_box, validate_dataset_sample
├── tests/
│   ├── unit/
│   │   ├── test_patient_isolation.py         # Added full-image, boundary, non-square, degenerate, & round-trip tests
│   │   ├── test_slake_loader.py              # Added patient_id is None, track assertion, & malformed JSON tests
│   │   ├── test_vqarad_loader.py             # Added track rejection, missing regions, & malformed JSON tests
│   │   ├── test_rsna_loader.py               # Pre-split JSON vs CSV tests, patientId validation
│   │   ├── test_vindr_loader.py              # Patient_id is None validation, xyxy coordinate parsing tests
│   │   └── test_validate_dataset.py          # Unit tests for the validate_dataset CLI tool
└── docs/
    └── reports/
        └── FS-001_CORRECTNESS_REVIEW.md      # This independent audit report
```

---

## 4. Tests Added in this Review

| Test Function | Target Scope | Verification Criteria |
| :--- | :--- | :--- |
| `test_full_image_bounding_box` | `base.py` | Full-image $[0, 0, W, H]$ on non-square image normalizes to $[0, 0, 1, 1]$ and round-trips to exact pixels. |
| `test_top_left_and_bottom_right_boundaries` | `base.py` | Corner boundary boxes on $1000 \times 500$ image correctly snap to boundaries without numerical drift. |
| `test_non_square_aspect_ratios` | `base.py` | $1920 \times 1080$ radiograph `xyxy` box normalizes and round-trips within $1e-4$ pixel tolerance. |
| `test_invalid_image_dimensions` | `base.py` | Image dimensions with $W \le 0$ or $H \le 0$ are rejected (`None`). |
| `test_reversed_and_degenerate_coordinates` | `base.py` | Inverted coordinates ($xmax \le xmin$ or $ymax \le ymin$) and zero-area boxes are rejected (`None`). |
| `test_out_of_bounds_rejection_vs_clipping` | `base.py` | Boxes exceeding boundary are rejected in strict mode, clamped only when explicitly flagged. |
| `test_patient_leakage_when_patient_id_is_none` | `base.py` | Verifies `PatientIsolationReport` correctly outputs `patient_ids_available=False`. |
| `test_patient_leakage_detected_with_multiple_views` | `base.py` | Successfully catches patient ID appearing across train and test splits. |
| `test_slake_loader_modality_track_separation` | `slake_loader.py` | Verifies Brain MRI sample raises `ValueError` if assigned to Track A (CXR predicted). |
| `test_slake_loader_malformed_json` | `slake_loader.py` | Gracefully catches and rejects malformed non-list JSON split files. |
| `test_vqarad_loader_missing_regions` | `vqarad_loader.py` | Confirms VQA-RAD explicitly has `has_ground_truth_regions=False` and rejects Track A/B. |
| `test_validate_dataset_sample` (5 tests) | `validate_dataset.py` | End-to-end directory structure and sample streaming audit. |

---

## 5. Actual Test & Quality Verification Results

```text
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\GitHub\honors-project\MedicalVQA
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.16.0, platformdirs-4.14.5, asyncio-1.4.1.post1, cov-7.1.0
collected 34 items

tests\unit\test_environment.py ..                                        [  5%]
tests\unit\test_patient_isolation.py .........                           [ 32%]
tests\unit\test_rsna_loader.py ...                                       [ 41%]
tests\unit\test_schemas.py ....                                          [ 52%]
tests\unit\test_slake_loader.py .....                                    [ 67%]
tests\unit\test_validate_dataset.py .....                                [ 82%]
tests\unit\test_vindr_loader.py ...                                      [ 91%]
tests\unit\test_vqarad_loader.py ...                                     [100%]

=============================== tests coverage ================================
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
core\__init__.py                   0      0   100%
core\schemas.py                  108      0   100%
datasets\__init__.py               6      0   100%
datasets\base.py                  93      7    92%   61, 99-100, 138, 209, 221-222
datasets\cxr_loader.py             3      0   100%
datasets\rsna_loader.py           69      3    96%   79-80, 109
datasets\slake_loader.py          71      6    92%   52-53, 138, 153-154, 173
datasets\validate_dataset.py      78     19    76%   90, 100-101, 121, 124-127, 151-161, 165
datasets\vindr_loader.py          88      5    94%   81, 122-123, 131, 158
datasets\vqarad_loader.py         40      2    95%   49-50
------------------------------------------------------------
TOTAL                            556     42    92%
============================= 34 passed in 0.34s ==============================
```

* **Pytest**: 34 unit tests passed in 0.34s.
* **Line Coverage**: **92%** across all dataset and core modules.
* **Linter & Formatter**: `ruff check .` and `ruff format --check .` passed with 0 errors and 0 warnings.
* **Type Checker**: `mypy core datasets` passed with `Success: no issues found in 10 source files`.
* **Static Security**: `bandit -r core datasets -ll` passed with 0 issues identified across 1,119 lines.
* **Vulnerability Audit**: `pip-audit` verified 0 known CVEs across all installed packages.

---

## 6. Real Dataset Validation Protocol

When real dataset samples are downloaded into `data/raw/`, validate them using the following commands:

### 1. SLAKE Validation
```bash
python -m datasets.validate_dataset --dataset slake --dir data/raw/slake --split test --max-samples 50
```
* **Required Files**: `data/raw/slake/test.json`, `data/raw/slake/train.json`, `data/raw/slake/validate.json`.
* **Expected Directories**: `data/raw/slake/imgs/`.
* **Image Formats**: `.jpg`, `.png`.

### 2. VQA-RAD Validation
```bash
python -m datasets.validate_dataset --dataset vqarad --dir data/raw/vqarad --split Dataset --max-samples 50
```
* **Required Files**: `data/raw/vqarad/VQA_RAD_Dataset.json`.
* **Expected Directories**: `data/raw/vqarad/images/`.

### 3. RSNA Pneumonia Validation
```bash
python -m datasets.validate_dataset --dataset rsna --dir data/raw/rsna --split train --max-samples 50
```
* **Required Files**: `data/raw/rsna/stage_2_train_labels.csv`.
* **Expected Directories**: `data/raw/rsna/stage_2_train_images/`.
* **Patient Identity Support**: Confirmed available (`patientId`).

### 4. VinDr-CXR Validation
```bash
python -m datasets.validate_dataset --dataset vindr --dir data/raw/vindr --split train --max-samples 50
```
* **Required Files**: `data/raw/vindr/annotations_train.csv`.
* **Expected Directories**: `data/raw/vindr/images/`.

---

## 7. Remaining Limitations

1. **Patient Identifier Scarcity**: Only RSNA provides official clinical patient identifiers. In SLAKE, VQA-RAD, and VinDr-CXR, `patient_id` is set to `None`. Any multi-view split claims on these three datasets are explicitly disclaimed.
2. **DICOM Extraction**: Full multi-frame DICOM extraction will utilize `pydicom` upon ingesting full raw datasets; our current test fixtures validate PNG representations.

---

## 8. FS-001 Acceptance Status

* **Status**: **ACCEPTED AND VERIFIED**.
* **Integrity Sign-Off**: The implementation now strictly reflects research ground truth without proxy illusions or silent coordinate mutations.
