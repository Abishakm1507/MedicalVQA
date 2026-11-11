# Feature Specification: FS-003 MedSAM Segmentation Integration

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-003
* **Feature Name**: MedSAM Promptable Segmentation Integration
* **Paper Section**: Section III-B (*Region Proposal*), Reference [6]
* **Module Path**: `localization/segmenter.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Ingest candidate bounding boxes $b_i$ from the detector and produce binary segmentation masks $m_i \in \{0, 1\}^{H \times W}$ for each region using MedSAM.
* **In-Scope**:
  - MedSAM model loader (`medsam_vit_b.pth` checkpoint).
  - Image embedding generation and box prompt inference.
  - Run-Length Encoding (RLE) mask serialization for lightweight API JSON transmission.
  - Fallback rectangular mask generation when MedSAM weights are not downloaded locally.

---

## 3. Technical Requirements & Interfaces
```python
from typing import List, Tuple
import numpy as np
from PIL import Image
from localization.detector import CandidateBox


class MedSAMSegmenter:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        pass

    def segment_boxes(self, image: Image.Image, boxes: List[CandidateBox]) -> List[np.ndarray]:
        """Generates boolean masks for each candidate bounding box."""
        pass
```
