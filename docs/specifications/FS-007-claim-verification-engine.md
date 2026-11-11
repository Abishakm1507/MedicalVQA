# Feature Specification: FS-007 Claim Verification Engine

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-007
* **Feature Name**: Claim Grounding & Verification Engine
* **Paper Section**: Section III-D (*Claim Verification*)
* **Module Path**: `verification/claim_verifier.py`
* **Assigned Agent**: Backend Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Check and enforce the contract between candidate regions $R$ and the claims emitted by the VLM.
* **Paper Rule**:
  > *"A finding is kept only if its region ID names a region in R, and the rest are dropped before anything is shown to the user. During evaluation we go one step further and check the cited region against the ground-truth mask or box. A finding is said to be unsupported when the cited region has an IoU below 0.5 with the annotated structure."*

---

## 3. Technical Requirements & Interfaces
```python
from dataclasses import dataclass
from typing import List, Optional, Set
from core.schemas import VLMClaim, VLMStructuredOutput
from localization.detector import CandidateBox


@dataclass
class VerifiedClaim:
    claim: str
    region_id: str
    is_grounded: bool
    localizer_score: float
    iou_with_ground_truth: Optional[float] = None
    is_supported: bool = True


class ClaimVerifier:
    def verify_runtime_claims(
        self, structured_output: VLMStructuredOutput, candidate_boxes: List[CandidateBox]
    ) -> List[VerifiedClaim]:
        """Filters out claims that fail to cite a valid region in R (or valid NONE token)."""
        pass

    def evaluate_grounding_iou(
        self,
        verified_claims: List[VerifiedClaim],
        candidate_boxes: List[CandidateBox],
        ground_truth_boxes: List[Tuple[float, float, float, float]],
        iou_threshold: float = 0.5,
    ) -> List[VerifiedClaim]:
        """Computes IoU against ground truth and flags claims with IoU < 0.5 as unsupported."""
        pass
```
