# ADR 0005: Confidence Gate Formulation and Calibration Protocol

## Status
Accepted

## Context
Section III-E defines the confidence gate:
$$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
with decision rule: accept if $c \ge \tau$, else report "uncertain".
We must specify how the weights $(w_1, w_2, w_3)$ and threshold $\tau$ are fitted, and how sample count $N$ is set across execution modes.

## Decision
1. **Adaptive Sampling ($N$)**:
   - Interactive Web UI: $N = 3$ samples for responsive real-time inference.
   - Batch Evaluation Benchmark: $N = 5$ samples for maximal statistical power.
2. **Calibration Protocol**:
   - Weights $(w_1, w_2, w_3)$ where $\sum w_i = 1$ are fitted via 2D simplex grid search (step size 0.05) on the validation split of SLAKE to maximize AUROC.
   - Initial baseline default weights: $w_1 = 0.4$ (agreement), $w_2 = 0.3$ (entropy), $w_3 = 0.3$ (detector score).
   - $\tau$ is treated as an operational operating point swept from $0.00$ to $1.00$ to produce the Accuracy vs. Coverage curve.

## Consequences
- **Positive**: Balances real-time web latency with rigorous benchmark reproducibility.
- **Positive**: Guarantees zero data leakage by tuning exclusively on the validation split.
