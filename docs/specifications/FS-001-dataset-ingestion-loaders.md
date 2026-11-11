# Feature Specification: FS-001 Dataset Ingestion & Standardized Loaders

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-001
* **Feature Name**: Dataset Ingestion & Standardized Loaders
* **Paper Section**: Section V-A (*Datasets*), Table 1
* **Module Path**: `datasets/slake_loader.py`, `datasets/vqarad_loader.py`, `datasets/cxr_loader.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved (Revised Post-Audit)

---

## 2. Objectives & Scope
* **Objective**: Provide unified, streaming, and deterministic data loaders for SLAKE (English), VQA-RAD, RSNA Pneumonia, and VinDr-CXR.

### 2.1 The Three Experimental Evaluation Tracks
To prevent modality mismatches (e.g. running a thoracic CXR detector on abdominal CT or brain MRI), the data loaders must support three distinct evaluation tracks:
1. **Track A (Predicted Localization on CXR)**: Evaluates the complete end-to-end pipeline (Detector + MedSAM + VLM + Gate) on Chest X-ray images from SLAKE, RSNA, and VinDr-CXR.
2. **Track B (Oracle Ground-Truth Bounding Box Grounding)**: Evaluates VLM visual grounding across all SLAKE modalities (Chest, Abdomen, Brain) by directly providing the dataset's ground-truth bounding boxes as candidate regions $R$. This isolates VLM reasoning from detector proposal errors.
3. **Track C (Ungrounded Answer Baseline)**: Evaluates free-form answering on VQA-RAD and ungrounded SLAKE.

### 2.2 Patient-Level Split Leakage Prevention
For RSNA and VinDr-CXR, split grouping must be enforced by `patient_id` so that multiple radiographic views of the same patient never appear in both training and evaluation sets.

---

## 3. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class GroundTruthRegion:
    label: str
    box_2d: Tuple[float, float, float, float]  # normalized [ymin, xmin, ymax, xmax] in [0, 1]
    mask_path: Optional[str] = None

@dataclass
class MedicalVQASample:
    sample_id: str
    patient_id: Optional[str]
    image_path: str
    modality: str  # e.g., 'CXR', 'CT', 'MRI'
    organ: str     # e.g., 'chest', 'abdomen', 'brain'
    question: str
    answer: str
    question_type: str  # 'closed' or 'open'
    ground_truth_regions: List[GroundTruthRegion]
    metadata: Optional[Dict[str, Any]] = None
```
