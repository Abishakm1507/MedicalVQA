# ADR 0004: Handling Negative and Absence Claims via `NONE` Region Token

## Status
Accepted

## Context
Section VI-B (*Limitations*) of the research paper explicitly states:
> *"Negative answers, such as 'no effusion,' are a particular problem for the approach. They claim an absence, so there is no region to point to, and the schema needs a rule for them; we have not settled on one."*

If the model is forced to cite a candidate region $R_i$ when stating an abnormality is absent, it either points to healthy tissue or causes a schema validation failure.

## Decision
We introduce a formal rule into the structured JSON schema and claim verifier:
1. The schema permits `region_id: "NONE"` (or `"GLOBAL"`) specifically for negative findings asserting the absence of a lesion (e.g. *"no cardiomegaly"*, *"clear lung fields"*, *"no pneumothorax"*).
2. The claim verifier checks for clinical negation keywords (*"no"*, *"not"*, *"negative for"*, *"free of"*, *"without"*).
3. If a negative assertion cites `NONE`, it is marked as **validly grounded** and is not flagged as an unsupported claim.
4. For negative assertions citing `NONE`, the localizer score contribution is set to $\rho = 1.0$ (reflecting that no ungrounded lesion was hallucinated).

## Consequences
- **Positive**: Directly resolves the open question identified in Section VI-B without violating the grounding philosophy.
- **Positive**: Prevents artificial penalties on accurate negative radiology assessments.
- **Negative**: Requires lightweight clinical negation parsing in `verification/claim_verifier.py`.
