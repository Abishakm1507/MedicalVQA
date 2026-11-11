# ADR 0003: Candidate Region Detector Selection and Licensing Review

## Status
Under Review (License Alignment in Progress)

## Context
Section III-B of the paper specifies:
> *"A localizer returns k candidate regions R = {r1, ..., rk}... YOLO- or DETR-style detectors propose the boxes and MedSAM turns each box into a mask."*

We need a detection model that is lightweight, capable of running efficiently on CPU for local development, and easily fine-tuned on the RSNA Pneumonia and VinDr-CXR datasets.

### Licensing & Copyleft Audit
Ultralytics YOLOv8 is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- AGPL-3.0 requires that any networked application (e.g. FastAPI backend or interactive web service) that uses the library must release its complete source code under AGPL-3.0.
- Because MedicalVQA is set to MIT license, direct bundling of Ultralytics YOLOv8 creates a copyleft licensing conflict.

### Visual Evidence Integrity Rule
Under no circumstances may the detector generate artificial, synthetic, or arbitrary anatomical quadrant boxes when the detector finds no lesions above its confidence threshold. If no lesions are detected, the candidate set is empty ($R = \emptyset$). Any positive finding asserted by a VLM on an image with $R = \emptyset$ is ungrounded and must be dropped.

## Evaluated Detector Alternatives

| Detector Candidate | Architecture Style | License | CPU Feasibility | Free GPU Training |
| :--- | :--- | :--- | :--- | :--- |
| **RT-DETR (Real-Time DETR)** | Transformer / DETR | Apache 2.0 | High (<40ms on CPU) | Yes (PaddlePaddle/MMDetection/Torch) |
| **Torchvision Faster R-CNN / RetinaNet** | Two-stage / Focal Loss | BSD-3-Clause | Moderate (~60ms on CPU) | Native PyTorch |
| **YOLOv10 / YOLOv9** | One-stage anchor-free | Apache 2.0 / GPL Dual | High (<25ms on CPU) | Standard PyTorch |
| **Ultralytics YOLOv8** | One-stage anchor-free | AGPL-3.0 (Viral Copyleft) | High (<20ms on CPU) | Ultralytics Hub |

## Decision
1. **License Alignment**: To maintain MIT compatibility, we prioritize **RT-DETR** or **Torchvision Faster R-CNN/RetinaNet** (or an Apache 2.0 YOLO variant) over AGPL-3.0 YOLOv8.
2. **MedSAM Prompting**: The chosen detector provides normalized bounding boxes $[ymin, xmin, ymax, xmax]$ directly into MedSAM's prompt encoder.
3. **Strict Rejection of Artificial Anchor Boxes**: If zero candidate regions exceed threshold, $R = \emptyset$.
