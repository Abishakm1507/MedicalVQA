# Feature Specification: FS-012 React Interactive Web App

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-012
* **Feature Name**: React Interactive Web Application
* **Paper Section**: Section IV-A (*Architecture*), Figure 2
* **Module Path**: `web/src/`
* **Assigned Agent**: Frontend Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Build an interactive, responsive medical web interface using React 18, Vite, and Tailwind CSS.
* **Core Views & Components**:
  1. `ImageUploader`: Drag-and-drop chest X-ray uploader with instant client-side preview.
  2. `CanvasOverlay`: Interactive HTML5 Canvas component that plots colored candidate region boxes ($R_1 \dots R_k$) and MedSAM segmentation contours directly over the medical radiograph with hover tooltips.
  3. `SideBySideVLM`: Split-screen comparison card displaying Qwen2.5-VL-3B versus MedGemma-4B outputs under identical visual prompts.
  4. `ConfidenceBadge`: Color-coded badge (`High [Green]`, `Moderate [Yellow]`, `Low [Orange]`, `Abstained / Uncertain [Red/Gray]`).
  5. `UncertaintyAlert`: Prominent clinician banner when the confidence gate abstains ($c < \tau$), displaying deferral rationale.
