# Feature Specification: FS-008 Confidence Gating & Selective Abstention

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-008
* **Feature Name**: Confidence Gate & Selective Abstention Engine
* **Paper Section**: Section III-E (*Confidence Gate*), Equation 2
* **Module Path**: `verification/confidence_gate.py`
* **Assigned Agent**: Backend Engineer
* **Status**: Approved (Revised Post-Audit)

---

## 2. Objectives & Scope
* **Objective**: Compute fused scalar confidence $c$ and make the binary decision to either accept the answer or abstain ("uncertain"):
  $$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
  Decision rule: Return $\hat{y}$ if $c \ge \tau$, otherwise return `"uncertain"`.

---

## 3. Mathematical Specifications

### 3.1 Semantic Equivalence Clustering (Kuhn et al., 2023)
Raw text matching overestimates uncertainty because clinically identical answers can be phrased differently.
- Responses $\{y_1, \dots, y_N\}$ are partitioned into semantic equivalence clusters $\{C_1, \dots, C_K\}$.
- $a = \frac{\max_k |C_k|}{N} \in [\frac{1}{N}, 1]$.
- $H_n = \frac{-\sum p_k \ln p_k}{\ln N} \in [0, 1]$.

### 3.2 Dynamic Weight & Threshold Calibration
Weights $(w_1, w_2, w_3)$ are **not hardcoded**; they are fitted on the SLAKE validation split to maximize AUROC.
The gate sweeps $\tau \in [0, 1]$ to generate the full Accuracy vs. Coverage curve.

---

## 4. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from core.schemas import VLMStructuredOutput


@dataclass
class ConfidenceBreakdown:
    majority_answer: str
    agreement: float  # a in [0, 1]
    raw_entropy: float  # Shannon entropy H
    normalized_entropy: float  # H_n = H / ln(N) in [0, 1]
    localizer_score: float  # rho in [0, 1]
    fused_confidence: float  # c in [0, 1]
    is_abstaining: bool  # True if c < tau
    decision_threshold: float  # tau


class ConfidenceGate:
    def __init__(self, w1: float, w2: float, w3: float, tau: float):
        assert abs((w1 + w2 + w3) - 1.0) < 1e-4, "Weights must sum to 1.0"
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.tau = tau

    def cluster_semantic_answers(self, answers: List[str]) -> List[List[str]]:
        """Groups answers into semantic equivalence clusters conforming to Kuhn et al."""
        pass

    def compute_agreement_and_entropy(
        self, sampled_outputs: List[VLMStructuredOutput]
    ) -> Tuple[str, float, float, float]:
        """Calculates majority answer, agreement a, and normalized semantic entropy H_n."""
        pass

    def evaluate_gate(
        self,
        sampled_outputs: List[VLMStructuredOutput],
        cited_localizer_score: Optional[float] = None,
    ) -> ConfidenceBreakdown:
        """Calculates c and returns full breakdown including abstention status."""
        pass
```
