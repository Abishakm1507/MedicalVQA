# Feature Specification: FS-002 Region Localization Detector

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-002
* **Feature Name**: Region Localization & Candidate Box Proposer
* **Paper Section**: Section III-B (*Region Proposal*)
* **Module Path**: `localization/detector.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved (Revised Post-Audit)

---

## 2. Objectives & Scope
* **Objective**: Generate $k$ candidate bounding boxes $b_i$ and objectness scores $\rho_i \in [0, 1]$ from an input radiograph.
* **In-Scope**:
  - Detection model inference (RT-DETR or Faster R-CNN under Apache 2.0 / BSD license).
  - Non-maximum suppression (NMS) and top-$k$ filtering ($k \le 7$).
  - Normalization of coordinates to image dimensions $[0, 1]$.
  - Clean handling of empty detections ($R = \emptyset$).
* **Out-of-Scope**: Dense pixel segmentation (delegated to MedSAM in FS-003).

---

## 3. Strict Scientific Rule: No Artificial Box Generation
Under no circumstances may the detector generate artificial anatomical quadrant boxes when no pathological lesions exceed the detection threshold.
- If zero boxes exceed the score threshold, the detector returns an empty list: $R = \emptyset$.
- This adheres strictly to Section VI-B: *"A lesion the localizer misses makes a true claim impossible to ground."* Inventing boxes would falsely attribute visual support to hallucinated claims.

---

## 4. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image

@dataclass
class CandidateBox:
    region_id: str  # 'R1', 'R2', ...
    box_2d: Tuple[int, int, int, int]  # [ymin, xmin, ymax, xmax] in absolute pixels
    normalized_box: Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] in [0, 1]
    score: float  # detector confidence rho in [0, 1]
    label: str

class RegionDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_k: int = 5,
        score_threshold: float = 0.25
    ):
        self.max_k = max_k
        self.score_threshold = score_threshold

    def detect(self, image: Image.Image) -> List[CandidateBox]:
        """Runs candidate bounding box detection. Returns up to max_k boxes.
        Returns empty list if no candidate exceeds score_threshold.
        """
        pass
```
