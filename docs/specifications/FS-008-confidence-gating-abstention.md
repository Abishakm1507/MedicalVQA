# Feature Specification: FS-008 Confidence Gating & Selective Abstention

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-008
* **Feature Name**: Confidence Gate & Selective Abstention Engine
* **Paper Section**: Section III-E (*Confidence Gate*), Equation 2
* **Module Path**: `verification/confidence_gate.py`
* **Assigned Agent**: Backend Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Compute fused scalar confidence $c$ and make the binary decision to either accept the answer or abstain ("uncertain"):
  $$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
  Decision rule: Return $\hat{y}$ if $c \ge \tau$, otherwise return `"uncertain"`.

---

## 3. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Tuple
from core.schemas import VLMStructuredOutput

@dataclass
class ConfidenceBreakdown:
    agreement: float         # a in [0, 1]
    raw_entropy: float       # Shannon entropy H
    normalized_entropy: float # H_n = H / ln(N) in [0, 1]
    localizer_score: float   # rho in [0, 1]
    fused_confidence: float  # c in [0, 1]
    is_abstaining: bool      # True if c < tau
    decision_threshold: float # tau

class ConfidenceGate:
    def __init__(
        self,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
        tau: float = 0.65
    ):
        assert abs((w1 + w2 + w3) - 1.0) < 1e-4, "Weights must sum to 1.0"
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.tau = tau

    def compute_agreement_and_entropy(
        self,
        sampled_outputs: List[VLMStructuredOutput]
    ) -> Tuple[str, float, float, float]:
        """Clusters sampled answers and computes majority answer, agreement a, and normalized entropy H_n."""
        pass

    def evaluate_gate(
        self,
        sampled_outputs: List[VLMStructuredOutput],
        cited_localizer_score: float
    ) -> ConfidenceBreakdown:
        """Calculates c and returns full breakdown including abstention status."""
        pass
```
