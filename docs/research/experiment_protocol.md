# Experimental Protocol and Benchmarking Standard

This document establishes the experimental protocol required to reproduce the paper's findings, isolate the individual effects of visual grounding versus confidence abstention, and evaluate model robustness.

---

## 1. Experimental Conditions Matrix

To measure and isolate the individual contributions of visual grounding and selective abstention, every evaluation run is conducted across **three orthogonal conditions** for each model:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Experimental Conditions                         │
├───────────────────┬────────────────────────────┬───────────────────────┤
│ Condition ID      │ Configuration              │ Research Purpose      │
├───────────────────┼────────────────────────────┼───────────────────────┤
│ C1: Free-Form     │ Standard prompting, no     │ Establishes baseline  │
│     Baseline      │ candidate regions, free-   │ hallucination and raw │
│                   │ text generation.           │ answering accuracy.   │
├───────────────────┼────────────────────────────┼───────────────────────┤
│ C2: Grounded      │ Candidate regions marked   │ Isolates the impact   │
│     Contract      │ (R1..Rk), strict JSON      │ of the visual contract│
│                   │ schema, claim filtering,   │ and region enforcement│
│                   │ NO abstention gate.        │ on unsupported claims.│
├───────────────────┼────────────────────────────┼───────────────────────┤
│ C3: Grounded +    │ Full pipeline: visual      │ Evaluates selective   │
│     Confidence    │ contract + claim filtering │ prediction, risk-     │
│     Gate          │ + confidence thresholding  │ coverage trade-off,   │
│                   │ c ≥ τ (abstain if c < τ).  │ and calibration.      │
└───────────────────┴────────────────────────────┴───────────────────────┘
```

Both **Qwen2.5-VL-3B** and **MedGemma-4B** are tested across all three conditions under identical prompt phrasing and temperature settings.

---

## 2. In-Depth Robustness Perturbation Suite

To assess whether the VLM's grounding is brittle or invariant to acquisition variations, images are subjected to transformations using OpenCV and Albumentations:

1. **Gaussian Noise Injection**: Simulates low-dose sensor detector noise ($\sigma \in \{0.02, 0.05, 0.1\}$).
2. **Contrast & Brightness Jitter**: Simulates suboptimal window-leveling / display rendering ($\pm 20\%$).
3. **JPEG Compression**: Simulates lossy telemetry/PACS transfer artifacts ($Q \in \{85, 60, 40\}$).
4. **Subtle Rotations**: Simulates patient positioning tilt ($\pm 5^\circ, \pm 10^\circ$).
5. **Horizontal Flips with Lateral Query Inversion**:
   - The radiograph is horizontally reflected ($x \mapsto W - x$).
   - Questions containing lateral references ("left" vs. "right") are automatically inverted in the ground truth.
   - *Failure check*: If the model fails to invert its lateral finding (e.g. still answers "right lower zone"), it demonstrates language prior bias rather than visual comprehension.

---

## 3. Masked-Region Visual Ablation Test

To prove that the model legitimately reasoned about the cited region rather than generating a lucky guess from global image context:

```
[Original Query] ──► Model cites R2 (Right lower zone effusion)
                            │
                            ▼
[Ablation Step]  ──► Infill / mask region R2 pixels with neutral gray
                            │
                            ▼
[Re-Query VLM]   ──► If model still answers "effusion at R2":
                     ==> SPURIOUS REASONING / LANGUAGE PRIOR
                     If model changes answer to "no effusion" or abstains:
                     ==> VALID VISUAL DEPENDENCE
```

* Metric: **Visual Dependence Rate (VDR)**: Proportion of cases where ablaing the cited region correctly causes the model to retract the claim.

---

## 4. Confidence Gate Calibration Protocol

1. **Held-Out Calibration Split**:
   - The validation splits of SLAKE ($15\%$) are reserved strictly for tuning gate hyperparameters:
     $$\mathbf{w}^* = \arg\max_{w_1+w_2+w_3=1} \text{AUROC}(c(\mathbf{w}), \mathbf{y}_{\text{correct}})$$
2. **Threshold Sweep**:
   - Sweep $\tau$ from $0.00$ to $1.00$ in increments of $0.01$.
   - Record coverage $\Phi(\tau)$ and selective accuracy $A(\Phi)$.
   - Generate publication-quality Accuracy-vs-Coverage curves comparing Qwen2.5-VL-3B vs. MedGemma-4B.

---

## 5. Statistical Protocol
* **Replication**: All generation sampling uses fixed pseudo-random seeds ($42, 1337, 2026$).
* **Confidence Intervals**: 1,000 non-parametric bootstrap resamples to construct two-sided $95\%$ empirical confidence intervals.
* **Significance Testing**:
  - Closed-question binary accuracy: Paired McNemar's test.
  - Continuous calibration (ECE) & USR: Two-tailed Wilcoxon signed-rank test ($\alpha = 0.05$).
