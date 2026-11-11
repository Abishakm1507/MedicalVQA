# Feature Specification: FS-004 Visual Marker Overlay Engine

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-004
* **Feature Name**: Set-of-Mark Visual Overlay Engine
* **Paper Section**: Section III-C (*Region Labelling and Structured Answers*)
* **Module Path**: `localization/visual_marker.py`
* **Assigned Agent**: Backend Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Render high-contrast, visually unambiguous bounding boxes and numbered labels ($R_1 \dots R_k$) directly onto a copy of the input medical image before passing it to the VLM.
* **In-Scope**:
  - Drawing semi-transparent colored bounding boxes.
  - Adding high-contrast pill labels ($R_1, R_2, \dots$) at the top-left corner of each box with dark text on bright background.
  - Ensuring labels and contours do not obscure critical pathology inside the bounding box.
  - Returning both the marked PIL image and JSON metadata.

---

## 3. Technical Requirements & Interfaces
```python
from typing import List, Tuple
from PIL import Image
from localization.detector import CandidateBox

class VisualMarkerEngine:
    def __init__(self, line_width: int = 3, font_size: int = 16):
        pass

    def annotate(
        self,
        image: Image.Image,
        candidate_boxes: List[CandidateBox]
    ) -> Tuple[Image.Image, List[CandidateBox]]:
        """Renders Set-of-Mark visual overlays onto a fresh copy of the image."""
        pass
```
