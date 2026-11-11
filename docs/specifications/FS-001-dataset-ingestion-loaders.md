# Feature Specification: FS-001 Dataset Ingestion & Standardized Loaders

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-001
* **Feature Name**: Dataset Ingestion & Standardized Loaders
* **Paper Section**: Section V-A (*Datasets*), Table 1
* **Module Path**: `datasets/slake_loader.py`, `datasets/vqarad_loader.py`, `datasets/cxr_loader.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Provide unified, streaming, and deterministic data loaders for SLAKE (English), VQA-RAD, RSNA Pneumonia, and VinDr-CXR.
* **In-Scope**:
  - Ingest JSON/CSV metadata and pair images with their questions, answers, bounding boxes, and segmentation masks.
  - Convert disparate coordinates to standardized $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ normalized format $[0, 1]$.
  - Support train, validation, and test splits with strict seed reproducibility.
* **Out-of-Scope**: Non-English SLAKE questions; downloading full multi-gigabyte raw files during unit tests.

---

## 3. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image

@dataclass
class GroundTruthRegion:
    label: str
    box_2d: Tuple[float, float, float, float]  # normalized [ymin, xmin, ymax, xmax]
    mask_path: Optional[str] = None

@dataclass
class MedicalVQASample:
    sample_id: str
    image_path: str
    modality: str  # e.g., 'CXR', 'CT', 'MRI'
    question: str
    answer: str
    question_type: str  # 'closed' or 'open'
    ground_truth_regions: List[GroundTruthRegion]
```

---

## 4. Testing & Verification Requirements
* Mock fixtures in `tests/fixtures/datasets/` providing minimal synthetic samples.
* Verifies split isolation (no overlap between train, val, and test IDs).
* Verifies bounding box coordinate bounds: $0 \le x_{\text{min}} < x_{\text{max}} \le 1$.
