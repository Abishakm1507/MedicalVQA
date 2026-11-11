# Feature Specification: FS-010 Evaluation Metrics Harness

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-010
* **Feature Name**: Evaluation Metrics & Statistical Benchmark Harness
* **Paper Section**: Section V-B, Section V-C (*Metrics*), Table 2, Equation 1, Equation 3
* **Module Path**: `evaluation/metrics.py`, `evaluation/runner.py`
* **Assigned Agent**: QA & Evaluation Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Evaluate models across the 3 experimental conditions (Free-form, Grounded JSON contract, Grounded with Gate).
* **Metrics Calculated**:
  1. Closed-Question Accuracy & Macro F1.
  2. Unsupported-Claim Rate (USR): $\text{USR} = N_{\text{unsup}} / N_{\text{claims}}$ (before & after filtering).
  3. Pointing Accuracy ($\text{IoU} \ge 0.5$).
  4. Expected Calibration Error (ECE) across $M$ bins:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  5. AUROC of confidence score $c$ predicting answer correctness.
  6. Accuracy vs. Coverage curve across threshold sweep $\tau \in [0, 1]$.
  7. 1,000-sample bootstrap confidence intervals ($95\%$ CIs) and paired SciPy significance testing.
