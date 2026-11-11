# ADR 0004: Negative and Absence Claims Policy Formulation

## Status
Proposed / Pending User Decision (Unresolved Research Question from Paper Section VI-B)

## Context
Section VI-B (*Limitations*) of the research paper explicitly highlights:
> *"Negative answers, such as 'no effusion,' are a particular problem for the approach. They claim an absence, so there is no region to point to, and the schema needs a rule for them; we have not settled on one."*

### Scientific Risks Identified During Independent Audit:
1. **The Overconfidence Hazard**: If a negative claim citing `NONE` is assigned an arbitrary localizer score of $\rho = 1.0$, the confidence gate equation $c = w_1 a + w_2 (1 - H_n) + w_3(1.0)$ artificially inflates the confidence of absence claims. A model that misses a subtle pneumothorax and claims "no pneumothorax" would receive a bonus score of 1.0 on localization, producing an overconfident and uncalibrated false negative.
2. **Text Negation vs. Medical Truth**: A natural language regex or keyword match for "no" / "negative" is strictly an NLP signal of syntactic negation; it is **not** clinical proof that the abnormality is genuinely absent from the radiograph.

---

## Proposed Options for User Approval

### Option 1: Multi-Tier Finding Schema with Rescaled Confidence (Recommended)
* **Schema Contract**:
  Findings are structured with explicit finding status:
  ```json
  {
    "answer": "There is no pleural effusion visible in either lung.",
    "findings": [
      {
        "claim": "pleural effusion",
        "status": "absent",
        "evidence_type": "image_level",
        "region_id": "NONE"
      }
    ]
  }
  ```
* **Confidence Gate Policy for Negative Claims**:
  Because there is no lesion bounding box to score with a detector, $\rho$ is omitted, and confidence is computed purely over consistency and normalized entropy:
  $$c_{\text{negative}} = \frac{w_1 a + w_2 (1 - H_n)}{w_1 + w_2}$$
  This prevents artificial inflation while remaining grounded in model certainty.
* **Evaluation Grounding Rule**:
  An "absent" claim is scored as **supported** if and only if the ground truth annotation for that image has zero overlapping bounding boxes for that pathology.

### Option 2: Anatomical Compartment Grounding
* For negative claims, the model must point to the *anatomical region* where the finding would typically reside (e.g. citing lung base region $R_2$ when ruling out effusion).
* *Limitation*: Requires the detector to propose anatomical organ landmarks in addition to pathological lesions.

### Option 3: Two-Phase Gated Answering
* Separate questions asking about existence (Closed Yes/No) from spatial location queries (Open). Apply visual grounding strictly to positive findings; evaluate negative findings purely on answer accuracy.

---

## Current Working Policy
Until user approval is granted, this decision remains **OPEN**. Any benchmark results for negative claims will be reported separately from positive localized claims.
