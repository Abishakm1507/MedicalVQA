# FS-005 Design Consistency Review

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Document ID:** `docs/reports/FS-005_DESIGN_CONSISTENCY_REVIEW.md`  
**Status:** Review complete — design corrected; one decision blocks implementation  
**Review type:** Documentation only — no implementation or model inference performed

---

## 1. Issue 1 — ADR 0004 Negative/Absence Claims (resolved: no option approved)

### 1.1 Source documents and evidence

| Document | Path | Relevant finding |
|----------|------|------------------|
| ADR 0004 | `docs/architecture/adr/0004-negative-claim-handling.md` | Status: **"Proposed / Pending User Decision"**. Current Working Policy: "Until user approval is granted, this decision remains **OPEN**." Lists Options 1, 2, and 3 as proposals — none approved. |
| Phase 0 Independent Audit | `docs/reports/PHASE_0_INDEPENDENT_AUDIT.md` §6.2 | Confirms ADR 0004 is "Marked as PROPOSED / PENDING USER DECISION." Identifies the $\rho=1.0$ defect but does not approve any replacement option. |

### 1.2 Existing design status (before correction)

The FS-005 design (`docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md`) section 4.6 and 11.1 described ADR 0004 Option 1 (rescaled confidence, omit $\rho$) as the "**current working policy**" and said the design "**adopts** ADR 0004 Option 1 ... as the working policy but marks it as pending user decision." This overstated adoption: ADR 0004 has approved no option.

The task premise claimed "an earlier explicit project decision approved Option 3." This is **not supported** by the actual ADR 0004 file, which lists Option 3 as one of three proposals, none approved.

### 1.3 Corrections made to FS-005

- Section 4.6: Changed "current working policy" → "candidate implementation approach only." Explicitly states "No negative-claims policy is adopted; ADR 0004 remains 'Proposed / Pending User Decision' with Options 1, 2, and 3 all under consideration."
- Section 11.1: Changed to state "None of Options 1, 2, or 3 is adopted as authoritative policy." The rescaled-confidence formula (Option 1) is presented as a candidate only, pending user confirmation.
- Summary section: Updated unresolved questions to state "no option approved."

### 1.4 Decision still requiring user approval

**Negative/absence claim confidence policy (ADR 0004).** User must select one of:
- **Option 1:** Multi-tier finding schema with rescaled confidence (omit $\rho$, rescale $w_1, w_2$).
- **Option 2:** Anatomical compartment grounding (model points to anatomical region for absence claims).
- **Option 3:** Two-phase gated answering (existence questions separated from spatial queries; negative claims evaluated on answer accuracy only).

This decision blocks Phase 1 claim-verification implementation of the NONE-claim confidence path. No implementation should proceed on this path until user approval.

---

## 2. Issue 2 — Sampling Consistency (verified consistent; minor addition)

### 2.1 Source documents and evidence

| Document | Path | Relevant finding |
|----------|------|------------------|
| ADR 0005 | `docs/architecture/adr/0005-confidence-gate-calibration.md` §3 | "Benchmark Evaluation: Strictly standardized at **N = 5** across all reported experimental tables and baseline comparisons." "Interactive UI: If low latency requires N = 3, a separate threshold operating point $\tau_{N=3}$ calibrated specifically on $N=3$ validation outputs must be utilized." |
| Phase 0 Independent Audit | `docs/reports/PHASE_0_INDEPENDENT_AUDIT.md` §6.3 | Confirms entropy normalization $H_n = H / \ln N$ depends on sample count; calibrating on N=5 and applying to N=3 causes distribution shift. |

### 2.2 Existing design status (before correction)

FS-005 section 4.6 already specified N=5 for benchmark calibration and separate $\tau_{N=3}$ for UI, consistent with ADR 0005. Section 7.2 specified Condition 1 uses N=1 greedy (T=0.0); Conditions 2/3 use N=5 (T=0.7). **No inconsistency found.**

### 2.3 Corrections made to FS-005

Added explicit **sampling parameter recording requirement** to section 4.6: sample count $N$, temperature $T$, and random seed must be logged per experiment for reproducibility and to prevent cross-contamination of entropy scales between N=5 benchmark and N=3 UI modes.

### 2.4 Verification result

Sampling configuration is consistent with ADR 0005:
- N=5 for primary benchmark experiments ✓
- Identical sampling configurations when comparing models ✓ (N=5, T=0.7 for Conditions 2/3)
- Sampling parameters recorded ✓ (added)
- Separate calibration for different sample counts ✓ ($\tau_{N=3}$ on N=3 validation outputs)

---

## 3. Issue 3 — Semantic Clustering (resolved: recommendation added)

### 3.1 Options evaluated

| Approach | Medical semantic correctness | Negation handling | Contradictory findings | Paraphrase equivalence | Computational cost | Free/local availability | Reproducibility |
|----------|------------------------------|-------------------|------------------------|------------------------|--------------------|-------------------------|-----------------|
| **Bidirectional NLI entailment** | High (trained on entailment) | Good (NLI models capture negation) | Good (models distinguish contradiction) | High | Heavy (cross-encoder per pair) | Moderate (HF NLI models) | Moderate (model-dependent) |
| **Embedding similarity + threshold** | Moderate (captures topical similarity) | Weak (embeddings may not distinguish negation well) | Moderate | Moderate-High | Light (single forward pass) | High (sentence-transformers local) | High (deterministic given threshold) |
| **Canonical entity normalization** | High for entities (UMLS/SNOMED) | Poor (may lose negation scope) | Poor | Low for entities | Very light (lookup) | Low (UMLS/SNOMED access/licensing) | High (deterministic) |

### 3.2 Recommendation

**Primary:** sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2`, free, local, ~80 MB) with cosine-similarity threshold; threshold calibrated on SLAKE validation split. Best balance of cost, reproducibility, and paraphrase handling for free/local deployment.

**Fallback:** canonical medical entity normalization (UMLS/SNOMED exact-match on canonical forms) as deterministic lightweight fallback when no embedding model is available.

**Not default:** bidirectional NLI — heavier, less reproducible, not the default given zero-cost and hardware constraints.

### 3.3 Validation plan (added to FS-005 section 4.5)

1. Calibrate cosine threshold on SLAKE validation split using synonymous/paraphrase clinical answer pairs.
2. Measure cluster purity against held-out human annotations where available.
3. Report cluster count distribution and $H_n$ sensitivity to threshold on validation.
4. Do not claim clinical validation of any clustering method — report as a technical configuration choice evaluated on benchmark accuracy/entropy stability only.

### 3.4 Corrections made to FS-005

Added recommendation and validation plan to section 4.5. Updated Summary unresolved questions to reflect the recommended approach (embeddings recommended, threshold calibrated).

---

## 4. Issue 4 — Confidence-Weight Fitting Objective (resolved: objective changed)

### 4.1 Objectives compared

| Objective | Appropriateness for selective prediction | Rationale |
|-----------|------------------------------------------|-----------|
| **AUROC** | Moderate | Measures confidence-score ranking quality (separates correct from incorrect) but does not directly optimize the abstention operating point ($\tau$). A high-AUROC scorer can still have poor selective accuracy at any given coverage level. |
| **AUPRC** | Moderate | Relevant under class imbalance, but still a ranking metric, not an operating-point objective. |
| **Selective risk / risk-coverage** | **High** | Directly optimizes the accuracy-vs-coverage trade-off that abstention is designed to control. Selects weights maximizing accuracy at target coverage levels. |
| **ECE minimization** | Moderate | Improves calibration (confidence matches accuracy) but does not directly optimize selective prediction utility. |

### 4.2 Recommendation (implemented in FS-005)

**Primary objective:** selective risk / risk-coverage optimization — select $(w_1, w_2, w_3)$ maximizing accuracy at target coverage across the tau sweep on the SLAKE validation split.

**Secondary:** AUROC reported as ranking-quality metric.

**Calibration:** ECE reported for calibration quality (not the fitting objective).

All fitting restricted to validation data; test set untouched until final reporting.

### 4.3 Corrections made to FS-005

Changed section 4.6 weight fitting objective from "maximize AUROC" to "maximize selective risk / risk-coverage trade-off," with AUROC secondary and ECE for calibration. Justification documented.

---

## 5. Issue 5 — Experimental Track Eligibility (verified correct; exclusions made explicit)

### 5.1 Source documents and evidence

| Document | Path | Relevant finding |
|----------|------|------------------|
| SLAKE Dataset Verification | `docs/reports/SLAKE_DATASET_VERIFICATION.md` §3.3 | "All 503 chest-location entries are CT modality — there are **zero Chest X-ray images** in this SLAKE dataset. Track A (Predicted CXR) never activates for SLAKE." |
| SLAKE Dataset Verification | `docs/reports/SLAKE_DATASET_VERIFICATION.md` §5.2 | "SLAKE test: 1,061 samples loaded, 980 with regions" (English test split). |
| Phase 0 Independent Audit | `docs/reports/PHASE_0_INDEPENDENT_AUDIT.md` §4.1 | Mandates Track A (CXR only), Track B (oracle GT boxes across all SLAKE modalities), Track C (ungrounded). |
| Phase 0 Independent Audit | `docs/reports/PHASE_0_INDEPENDENT_AUDIT.md` ISSUE-07 | Modality mismatch: CXR detector applied to brain/abdomen invalidates grounding tests; separation into tracks required. |

### 5.2 Existing design status (before correction)

FS-005 section 7.1 correctly stated Track A requires CXR data and is inactive on SLAKE (no CXR images), Track B covers all SLAKE modalities with GT boxes, Track C is ungrounded baseline. Section 7.2 clearly separated tracks (region source) from conditions (answer processing). **No inconsistency found** — but track exclusions were not presented in an explicit table.

### 5.3 Corrections made to FS-005

Added an explicit **Exclusions** column to the track table in section 7.1, documenting:
- Track A: excluded from SLAKE (zero CXR images); requires VQA-RAD CXR or RSNA/VinDr-CXR.
- Track B: samples with empty `detection.json: []` excluded from pointing-accuracy denominator (but remain in answer-accuracy and USR denominators).
- Track C: no exclusions; only applicable track for VQA-RAD (no bounding box annotations).

Added explicit statement: "Because SLAKE has no CXR images, any Track A results on SLAKE would be empty/invalid. Track A evaluation must use VQA-RAD CXR or RSNA/VinDr-CXR data only, never SLAKE."

### 5.4 Verification result

Tracks and conditions kept separate (Section 7.2 table confirms). Dataset eligibility and exclusions explicitly documented. No Track A claims on SLAKE CXR (none exist).

---

## 6. Summary of corrections made to FS-005

| Section | Change | Reason |
|---------|--------|--------|
| 4.5 (Semantic clustering) | Added sentence-transformer embedding recommendation + validation plan | Task required a practical recommendation |
| 4.6 (Confidence calibration) | Changed weight fitting objective from AUROC to selective risk/risk-coverage (primary), AUROC secondary, ECE calibration | Better aligned with abstention objective |
| 4.6 (Negative claims) | Changed "current working policy" → "candidate implementation approach only"; stated no option approved | ADR 0004 has approved no option; avoids overstating adoption |
| 4.6 (Sampling) | Added recording of N, T, random seed per experiment | Task required recorded sampling parameters |
| 7.1 (Track eligibility) | Added explicit Exclusions column; Track A never-on-SLAKE statement | Task required explicit dataset eligibility/exclusions |
| 11.1 (Open decisions) | Clarified no negative-claims option approved; user decision required | ADR 0004 status is "Proposed / Pending User Decision" |
| Summary | Updated unresolved questions and blocking decisions | Reflects corrections above |

---

## 7. Decisions still requiring user approval

| Decision | Status | Blocks |
|----------|--------|--------|
| Negative/absence claim confidence policy (ADR 0004: Option 1 / 2 / 3) | **OPEN — no option approved** | Phase 1 claim-verification NONE-claim confidence path |
| Semantic clustering threshold (embedding cosine threshold) | Configuration choice (recommended: calibrated on validation) | Phase 2 — does not block; can be tuned during implementation |
| Weight fitting objective (selective risk primary) | Design recommendation (validated against audit) | Phase 2 — does not block |

**Only the ADR 0004 negative-claims decision blocks implementation.** All other decisions are configuration choices that can be finalized during implementation.

---

## 8. Final recommendation on implementation readiness

**FS-005 is conditionally ready for implementation.**

- **Design is internally consistent** with ADR 0004, ADR 0005, Phase 0 audit findings, and the FS-001–FS-004 codebase.
- **All track eligibility and exclusion rules verified** against `docs/reports/SLAKE_DATASET_VERIFICATION.md` — Track A correctly excluded from SLAKE (no CXR images).
- **Sampling (N=5) consistent** with ADR 0005; recording requirement added.
- **Weight fitting objective** changed to selective risk/risk-coverage (better aligned with abstention).
- **Semantic clustering recommendation** added (sentence-transformer embeddings, calibrated on validation).

**One blocking decision:** The negative/absence claim confidence policy (ADR 0004) remains "Proposed / Pending User Decision" with no option approved. User must select Option 1, 2, or 3 before Phase 1 claim-verification code finalizes the NONE-claim confidence path.

**Recommended path:** Proceed with Phases 1–5, keeping the NONE-claim confidence path parameterized so it can be switched to the user-approved option once ADR 0004 is decided. Do not hardcode any negative-claims policy.

**No implementation, model inference, or FS-001–FS-004 modifications were performed in this review.**
