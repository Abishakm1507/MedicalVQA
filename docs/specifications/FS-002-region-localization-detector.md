# Feature Specification: FS-002 Region Localization Detector

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-002
* **Feature Name**: Region Localization & Candidate Box Proposer
* **Paper Section**: Section III-B (*Region Proposal*)
* **Module Path**: `localization/detector.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Generate $k$ candidate bounding boxes $b_i$ and objectness scores $\rho_i \in [0, 1]$ from an input radiograph.
* **In-Scope**:
  - Ultralytics YOLOv8 inference wrapper.
  - Non-maximum suppression (NMS) and top-$k$ filtering ($k \in [3, 7]$).
  - Normalization of coordinates to image dimensions.
  - Fallback grid/salience proposer when detector finds zero lesions.
* **Out-of-Scope**: Dense pixel segmentation (delegated to MedSAM in FS-003).

---

## 3. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

@dataclass
class CandidateBox:
    region_id: str  # 'R1', 'R2', ...
    box_2d: Tuple[int, int, int, int]  # [ymin, xmin, ymax, xmax] in absolute pixels
    normalized_box: Tuple[float, float, float, float]
    score: float  # detector confidence rho in [0, 1]
    label: str

class RegionDetector:
    def __init__(self, model_path: Optional[str] = None, max_k: int = 5, score_threshold: float = 0.25):
        pass

    def detect(self, image: Image.Image) -> List[CandidateBox]:
        pass
```

---

## 4. Edge Cases
* Zero detections above threshold: return fallback anatomical quadrant regions so VLM still has valid spatial anchors.
* Excessive overlapping boxes: NMS IoU threshold $= 0.45$.
