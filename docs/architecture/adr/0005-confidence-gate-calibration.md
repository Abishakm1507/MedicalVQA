# ADR 0005: Confidence Gate Formulation, Semantic Clustering & Calibration Protocol

## Status
Accepted (Revised Following Independent Audit)

## Context
Section III-E of the research paper defines the confidence gate:
$$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
with decision rule: accept if $c \ge \tau$, else report "uncertain".

### Flaws Identified in Prior Draft:
1. Hardcoded weights ($w_1=0.4, w_2=0.3, w_3=0.3$) and threshold ($\tau=0.65$) were ungrounded inventions. The paper explicitly states that weights and $\tau$ are determined on a held-out split.
2. Naive string matching was used for agreement $a$ and entropy $H_n$, violating Kuhn et al. (2023) [Ref 13], which demonstrated that semantic paraphrasing falsely inflates uncertainty unless answers are grouped into semantic equivalence classes.
3. Adaptive sampling ($N=3$ vs. $N=5$) shifts the entropy scale $H_n = \frac{H}{\ln N}$, making a single threshold $\tau$ non-transferable between interactive UI and benchmark evaluation.

---

## Technical Specifications

### 1. Semantic Equivalence Clustering (Kuhn et al., 2023)
Given $N$ sampled responses $\{y_1, \dots, y_N\}$ from the VLM at temperature $T > 0$:
1. Answers are partitioned into discrete semantic clusters $\{C_1, \dots, C_K\}$ where answers within the same cluster share clinical meaning (e.g. *"effusion at right base"*, *"right pleural effusion"* $\in C_1$).
2. Agreement $a$ is the proportion belonging to the modal cluster:
   $$a = \frac{\max_k |C_k|}{N}$$
3. Normalized Semantic Entropy is computed over cluster probabilities $p_k = \frac{|C_k|}{N}$:
   $$H_n = \frac{-\sum_{k=1}^K p_k \ln p_k}{\ln N}, \quad H_n \in [0, 1]$$

### 2. Parameter Fitting Protocol (No Hardcoded Values)
1. Weights $(w_1, w_2, w_3)$ satisfying $\sum w_i = 1$ and $w_i \ge 0$ are fitted on the **held-out validation split** of SLAKE via grid search (step size 0.05) to maximize selective classification AUROC.
2. The abstention threshold $\tau$ is not a fixed constant; the system sweeps $\tau \in [0, 1]$ to generate and report the full **Accuracy-versus-Coverage curve**.

### 3. Sampling Count Standard
- **Benchmark Evaluation**: Strictly standardized at **$N = 5$** across all reported experimental tables and baseline comparisons.
- **Interactive UI**: If low latency requires $N = 3$, a separate threshold operating point $\tau_{N=3}$ calibrated specifically on $N=3$ validation outputs must be utilized.
