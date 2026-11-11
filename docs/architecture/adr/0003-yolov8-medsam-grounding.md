# ADR 0003: Selection of YOLOv8 and MedSAM for Region Grounding

## Status
Accepted

## Context
Section III-B of the paper specifies:
> *"A localizer returns k candidate regions R = {r1, ..., rk}... YOLO- or DETR-style detectors propose the boxes and MedSAM turns each box into a mask."*

We need a detection model that is lightweight, capable of running efficiently on CPU for local development, and easily fine-tuned on the RSNA Pneumonia and VinDr-CXR datasets.

## Decision
We select **Ultralytics YOLOv8 (nano/small variants)** as the candidate bounding box detector, coupled with **MedSAM (Segment Anything in Medical Images)** for promptable mask generation.

## Consequences
- **Positive**: YOLOv8 provides state-of-the-art mAP, highly optimized CPU inference (<25 ms per image on Intel i5), standard Python API, and direct box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ with confidence scores $\rho_i$.
- **Positive**: MedSAM takes prompt bounding boxes directly and generates clean boundary masks without requiring full retraining.
- **Negative**: Transformers-based DETR is deferred as an optional baseline due to higher VRAM and CPU compute demands.
