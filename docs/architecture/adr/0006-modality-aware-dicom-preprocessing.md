# ADR 0006: Modality-Aware DICOM Preprocessing (CXR vs. CT Decoupling)

* **Status**: Accepted
* **Date**: November 11, 2026
* **Deciders**: Lead Architect, Senior ML Research Engineer

---

## 1. Context and Problem Statement

In previous preliminary documentation, DICOM preprocessing was described using Hounsfield Units (HU) via `RescaleSlope` and `RescaleIntercept`.
However, **Hounsfield Units are physically defined only for Computed Tomography (CT)**, where quantitative attenuation coefficients relative to water and air are calibrated.

Chest X-rays (CXR) are 2D projection radiographs, where pixel values represent transmitted photon counts or optical density. Applying Hounsfield Unit formulas or CT windowing presets (e.g. lung window $W=1500, L=-600$) to chest radiographs is physically incorrect and causes catastrophic contrast degradation.

A modality-aware DICOM preprocessing architecture is required that strictly decouples projection radiography (CXR) from cross-sectional volumetric imaging (CT).

---

## 2. Decision

We mandate **Modality-Aware DICOM Preprocessing**:

### 2.1 Chest Radiograph (CXR) Pipeline
1. **Pixel Array Decoding**: Extract raw pixel data via `pydicom` with decompressed pixel handlers (`pylibjpeg` or `imagecodecs` when required).
2. **Photometric Interpretation**:
   - `MONOCHROME2`: Standard visual presentation where low values are dark (air) and high values are bright (bone/metal). Retain orientation.
   - `MONOCHROME1`: Inverted presentation where low values are bright and high values are dark. **Must be inverted**:
     $$\text{pixel}_{\text{corrected}} = \max(\text{pixel}) - \text{pixel}$$
3. **Modality LUT**: Apply Modality LUT sequence if present in DICOM header to transform manufacturer-specific sensor values into optical density or linear presentation values.
4. **VOI LUT / Windowing**:
   - If `VOILUTSequence` is present, apply non-linear LUT transformation.
   - Else if `WindowCenter` ($WC$) and `WindowWidth` ($WW$) tags are present:
     Apply linear windowing mapping $[WC - WW/2, WC + WW/2]$ to $[0, 255]$.
   - Else, apply robust percentile normalization (e.g., 0.5th to 99.5th percentile stretching) to avoid outlier pixel saturation.
5. **Dynamic Range & Bit Depth**: Convert high-bit-depth (12-bit, 14-bit, 16-bit) to standardized 8-bit unsigned integer (`uint8`, $[0, 255]$) in RGB channel replication format for vision backbones.

### 2.2 Computed Tomography (CT) Pipeline (Isolated)
1. **Hounsfield Unit Conversion**: Strictly for CT modality:
   $$\text{HU} = \text{PixelValue} \times \text{RescaleSlope} + \text{RescaleIntercept}$$
2. **Clinical Windowing Presets**:
   - Lung Window: Center $L = -600$, Width $W = 1500$ (HU $[-1350, 150]$)
   - Mediastinal Window: Center $L = 40$, Width $W = 400$ (HU $[-160, 240]$)
   - Bone Window: Center $L = 400$, Width $W = 1800$ (HU $[-500, 1300]$)
3. Convert windowed range to $[0, 255]$ `uint8`.

---

## 3. Implementation Plan
- Implemented in `localization/preprocessing.py`:
  - `normalize_cxr_pixels()`: Dedicated to CXR projection radiographs.
  - `normalize_ct_pixels()`: Dedicated to volumetric CT slices with calibrated HU.
  - `prepare_image_for_detection()`: Converts PIL image to standardized PyTorch tensor $[3, H, W] \in [0.0, 1.0]$.

---

## 4. Consequences
* **Positive**: Clinically and physically accurate pixel intensities for both CXR and CT. Eliminates contrast distortion.
* **Integrity**: Clear architectural separation between 2D projection imaging and 3D cross-sectional imaging.
