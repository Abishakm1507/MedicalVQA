# Feature Specification: FS-006 Structured JSON Validation

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-006
* **Feature Name**: Structured JSON Contract & Pydantic Validation
* **Paper Section**: Section III-C (*Region Labelling and Structured Answers*)
* **Module Path**: `core/schemas.py`, `verification/json_validator.py`
* **Assigned Agent**: Backend Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Parse, validate, and shape raw string responses from generative VLMs into strongly typed Pydantic models. Rejects non-conforming responses.
* **Paper Example Schema**:
  ```json
  {
    "answer": "Yes, right lower zone",
    "findings": [
      {
        "claim": "effusion",
        "region_id": "R2",
        "confidence": "high"
      }
    ]
  }
  ```

---

## 3. Technical Requirements & Interfaces
```python
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class VLMClaim(BaseModel):
    claim: str = Field(..., description="The clinical finding asserted by the model.")
    region_id: str = Field(..., description="The region ID cited (e.g. 'R1', 'R2', or 'NONE' for absence).")
    confidence: Literal["high", "medium", "low"] = Field("high", description="Self-reported confidence level.")

class VLMStructuredOutput(BaseModel):
    answer: str = Field(..., description="Natural language answer to the clinical question.")
    findings: List[VLMClaim] = Field(default_factory=list, description="List of grounded claims.")

class JSONValidator:
    @staticmethod
    def parse_and_validate(raw_text: str) -> Optional[VLMStructuredOutput]:
        """Extracts JSON block (handling markdown code fences if present) and validates with Pydantic."""
        pass
```
