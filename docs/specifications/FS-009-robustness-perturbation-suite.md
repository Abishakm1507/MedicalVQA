# Feature Specification: FS-009 Robustness Perturbation & Region Ablation

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-009
* **Feature Name**: Robustness Perturbation & Masked-Region Ablation Suite
* **Paper Section**: Section V-D (*Robustness Tests*)
* **Module Path**: `evaluation/robustness.py`
* **Assigned Agent**: QA & Evaluation Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Evaluate model stability under image corruption, lateral inversion, and visual evidence masking.
* **Perturbation Transforms**:
  1. Gaussian noise ($\sigma \in \{0.02, 0.05, 0.1\}$).
  2. Contrast / brightness jitter ($\pm 20\%$).
  3. JPEG compression artifacting ($Q \in \{85, 60, 40\}$).
  4. Subtle rotation ($\pm 5^\circ, \pm 10^\circ$).
  5. Horizontal flip: Inverts image and transforms lateral query labels ("left" $\leftrightarrow$ "right").
* **Masked-Region Ablation**: Erases the cited candidate region pixels (infills with neutral gray) and re-evaluates VLM to detect spurious language prior correlations.
