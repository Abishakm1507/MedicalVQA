# FS-005 Phase 1 Implementation Report — Claim Extraction and Region Verification

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 1 — Claim Extraction and Region Verification  
**Document ID:** `docs/reports/FS-005_PHASE_1_IMPLEMENTATION_REPORT.md`  
**Status:** Implementation complete — Phase 1 only  
**Implementation type:** Implementation (code written), not design-only

---

## 1. Objective and Scope

Implement FS-005 Phase 1: the claim-level evidence verification foundation for the
Region-Grounded and Uncertainty-Aware Medical VQA system. This phase establishes
how atomic claims emitted by a VLM are extracted, associated with candidate
regions, verified against region evidence, and filtered — including the
ADR 0004 Option 3 handling of negative/absence claims.

**In scope (this phase):**
- Claim extraction from structured VLM responses (`VLMStructuredOutput`)
- Claim-to-region association against candidate regions (`CandidateBox[]`)
- **Runtime region-association verification (ground-truth-free)**: structural validity of claim-region association, no GT access
- **Evaluation-time grounding IoU (evaluation-only)**: IoU ≥ 0.5 against GT boxes, explicitly isolated from runtime filtering
- Unsupported claim detection and filtering (structural USR, not GT-IoU-based)
- Negative/absence claim handling per ADR 0004 Option 3

**Out of scope (deferred to Phase 2+):**
- Semantic agreement / entropy / uncertainty aggregation
- Confidence calibration, weight fitting, abstention, selective prediction
- Evaluation metric calculations (USR, pointing accuracy, ECE, etc.)
- Track A/B/C execution, API or frontend integration
- New model inference or dataset experiments
- Changes to FS-001 through FS-004 behavior or artifacts

---

## 2. Design Documents and Existing Components Reviewed

| Document | Path | Role |
|----------|------|------|
| FS-005 Technical Design | `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md` | Phase 1 component specifications (sections 4.1–4.4) |
| FS-005 Design Consistency Review | `docs/reports/FS-005_DESIGN_CONSISTENCY_REVIEW.md` | Review corrections; ADR 0004 Option 3 approval context |
| ADR 0004 | `docs/architecture/adr/0004-negative-claim-handling.md` | Negative/absence claims policy (updated to Accepted, Option 3) |
| Core schemas | `core/schemas.py` | `VLMClaim`, `VLMStructuredOutput`, `CandidateBox`, `GroundTruthRegion` (reused, not modified) |
| VLM client interface | `vlm/base.py` | `VLMClient.generate_structured()` produces `VLMStructuredOutput` (reused) |
| Region detector | `localization/detector.py` | `CandidateBox`, `MockRegionDetector` (reused for tests) |
| Postprocessing | `localization/postprocessing.py` | `CandidateBox` construction (reused) |
| Testing strategy | `docs/architecture/testing_strategy.md` | Test conventions (≥85% coverage, deterministic mocks, no skipped tests) |

**Existing interfaces reused without modification:** `core/schemas.py`, `vlm/base.py`, `localization/detector.py`, `localization/postprocessing.py`. No FS-001–FS-004 files were modified.

---

## 3. ADR 0004 Decision Recorded

**File:** `docs/architecture/adr/0004-negative-claim-handling.md`

The ADR was updated to record the user's explicit approval:

| Field | Value |
|-------|-------|
| Status | Accepted |
| Selected policy | Option 3 — Two-Phase Gated Answering |

**Option 3 behavior implemented:**
1. Negative/absence claims are preserved as valid answer claims in structured outputs.
2. Their correctness is evaluated through answer-level ground-truth comparison in the evaluation layer, not through positive-region grounding.
3. Negative/absence claims are excluded from positive-region grounding metrics and denominators.
4. No artificial localization confidence (`ρ`) is assigned to negative/absence claims.
5. An empty detector output (`R = ∅`) is never interpreted as proof of clinical absence.
6. An empty region collection is never converted into a verified negative finding.
7. Negative claims are preserved in structured outputs even when no candidate regions are available.

**Rationale preserved:** An empty detector output does not establish that a medical finding is absent; the system must not interpret missing detections as proof of clinical absence.

The ADR's existing structure and decision history (Context, Scientific Risks, Proposed Options) were preserved; only the Status and Current Working Policy sections were updated.

---

## 4. Architecture and Implementation Summary

Four new modules were added under `verification/`. Each is self-contained, imports only existing project types, and introduces no external dependencies beyond the project's existing stack (`pydantic`, `typing`, `dataclasses`).

### 4.1 Claim Extraction — `verification/claim_extractor.py`

**Responsibilities:** Parse a Pydantic-validated `VLMStructuredOutput` into canonical `ExtractedClaim` records, preserving the answer text and source information. Detects missing/malformed structured output safely (no exception raised; failure recorded in result).

**Data structures:**
- `ExtractedClaim`: `claim`, `region_id`, `status`, `confidence`, `raw_claim_text`
- `ExtractionResult`: `answer_text`, `claims`, `parse_success`, `error`

**Key function:** `extract_claims(vlm_output, raw_response=None) -> ExtractionResult`

**Behavior:**
- `vlm_output=None` → `parse_success=False`, error recorded (`empty_raw_response` or `parse_failure`), no exception.
- Valid output → maps each `VLMClaim` to an `ExtractedClaim`, preserves all fields, normalizes answer text (whitespace collapse, strip).
- Empty findings list → valid result with zero claims (not a failure).
- Deterministic parsing only; no external LLM/API for extraction.

### 4.2 Claim-to-Region Association — `verification/claim_associator.py`

**Responsibilities:** Associate each extracted claim with a candidate region from the detector-proposed set `R`. Validates `region_id` names an existing region; handles `NONE` (absence claims). Never creates synthetic boxes or region IDs.

**Data structure:** `AssociatedClaim`: `claim`, `region_id`, `status`, `confidence`, `is_grounded`, `cited_box`, `rejection_reason`

**Key function:** `associate_claims(claims, candidate_boxes) -> list[AssociatedClaim]`

**Behavior:**
- `region_id == "NONE"` → structurally grounded, `cited_box=None`, no rejection reason.
- `region_id` in `R` → grounded, `cited_box` points to the matching `CandidateBox` (provenance preserved).
- Unknown `region_id` → ungrounded, `rejection_reason` names the missing ID and `R` contents.
- `R = ∅` with a region claim → ungrounded, `rejection_reason="R is empty..."`.
- `R = ∅` with a `NONE` claim → structurally grounded.

### 4.3 Region Evidence Verification — `verification/evidence_verifier.py`

**Responsibilities:** Two separated functions:

- **`verify_associated`** (runtime, ground-truth-free): Verifies claim-to-region association structurally. Operates ONLY on VLM claims and detector-proposed candidate regions. Does NOT access ground-truth annotations. Does NOT use GT IoU to decide support.
- **`evaluate_grounding_iou_with_boxes`** (evaluation-only): Compares cited boxes against ground-truth boxes using IoU ≥ 0.5. Explicitly evaluation-only; must not influence runtime filtering.

**Data structures:**
- `VerifiedClaim`: `claim`, `region_id`, `status`, `is_grounded`, `localizer_score`, `is_supported` (structural validity at runtime), `is_absence_claim`, `is_uncertain_claim`, `verification_path`, `iou_with_ground_truth` (None at runtime, populated only by evaluation)
- `GroundingEvaluation`: `claim`, `region_id`, `status`, `iou_with_ground_truth`, `meets_evaluation_threshold`, `iou_threshold` (evaluation-only output)

**Key functions:**
- `compute_iou(box_a, box_b) -> float`: Pure IoU utility on normalized boxes; no GT dependency.
- `verify_association(associated_claims) -> list[VerifiedClaim]`: Runtime structural verification.
- `evaluate_grounding_iou_with_boxes(associated_claims, ground_truth_regions, iou_threshold=0.5) -> list[GroundingEvaluation]`: Evaluation-time IoU comparison.

**Runtime behavior (positive claims):**
- Grounded + cited box → `is_supported=True` (structural validity only), `localizer_score` from cited box.
- Ungrounded (bad region_id or R = ∅) → `is_supported=False`, `localizer_score=None`.
- Present claim citing `NONE` → `is_grounded=True` (NONE valid token) but `is_supported=False` (no box for positive claim).
- No GT is accessed at any point.

**Runtime behavior (absence claims, ADR 0004 Option 3):**
- `status == "absent"` → `verification_path="absence_answer"`, `localizer_score=None`, `is_absence_claim=True`. Structural support only; clinical truth deferred to evaluation layer.
- `status == "uncertain"` → `verification_path="uncertain"`, `is_uncertain_claim=True`. Distinct meaning preserved.

**Evaluation behavior:**
- Positive claims with cited boxes: `max_iou` over GT boxes; `meets_evaluation_threshold = max_iou ≥ 0.5`.
- Absence/uncertain claims: `iou_with_ground_truth=0.0`, `meets_evaluation_threshold=False`.

### 4.4 Unsupported Claim Filtering — `verification/claim_filter.py`

**Responsibilities:** Partition verified claims into supported / unsupported / unverifiable / absence / uncertain using **structural validity only** (no GT IoU). Computes structural USR metrics over positive claims only. Preserves original claims and verification records for auditing.

**Data structure:** `FilterResult`: `supported_claims`, `unsupported_claims`, `unverifiable_claims`, `absence_claims`, `uncertain_claims`, `usr_raw`, `usr_filtered`, `n_claims`, `n_positive_claims`

**Key function:** `filter_claims(verified_claims) -> FilterResult`

**Classification (structural, GT-free):**
- Absence claims (`status == "absent"`) → `absence_claims`, excluded from USR denominators (ADR 0004 Option 3).
- Uncertain claims (`status == "uncertain"`) → `uncertain_claims`, excluded from USR denominators.
- Positive claims:
  - structurally supported (`is_supported=True`) → `supported_claims`
  - structurally unsupported (e.g. present + NONE, no box) → `unsupported_claims`
  - ungrounded (bad region_id or R = ∅) → `unverifiable_claims`

**USR definitions (structural, positive claims only):**
- `usr_raw = (unsupported + unverifiable) / n_positive` (0.0 if no positive claims)
- `usr_filtered = unsupported / (n_positive − unverifiable)` (0.0 if denominator 0)

**Note:** These are structural USR metrics from runtime filtering. The evaluation layer (Phase 3+) will compute IoU-based USR separately using `evaluate_grounding_iou_with_boxes`.

Absence and uncertain claims go to their respective lists and are excluded from both USR denominators.

---

## 5. Files Created and Modified

### Created

| File | Explanation |
|------|-------------|
| `verification/claim_extractor.py` | Claim extraction from `VLMStructuredOutput` (Phase 1, section 4.1) |
| `verification/claim_associator.py` | Claim-to-region association (Phase 1, section 4.2) |
| `verification/evidence_verifier.py` | Region evidence verification via IoU (Phase 1, section 4.3) |
| `verification/claim_filter.py` | Unsupported claim filtering + USR metrics (Phase 1, section 4.4) |
| `tests/unit/test_claim_verification_phase1.py` | 32 deterministic unit tests for all Phase 1 components |
| `docs/reports/FS-005_PHASE_1_IMPLEMENTATION_REPORT.md` | This report |

### Modified

| File | Explanation |
|------|-------------|
| `docs/architecture/adr/0004-negative-claim-handling.md` | Status → Accepted; Option 3 selected; decision rationale recorded (existing structure/history preserved) |

### Not modified
- `core/schemas.py`, `vlm/base.py`, `localization/detector.py`, `localization/postprocessing.py` (reused unchanged)
- All FS-001–FS-004 code and experimental artifacts
- Original baseline results under `experiments/runs/`

---

## 6. Claim Data Structures and Verification/Filtering Behavior

| Structure | Purpose | Key fields |
|-----------|---------|------------|
| `ExtractedClaim` | Atomic claim from VLM output | `claim`, `region_id`, `status`, `confidence`, `raw_claim_text` |
| `AssociatedClaim` | Claim + region association | `is_grounded`, `cited_box`, `rejection_reason` |
| `VerifiedClaim` | Claim + structural verification | `is_grounded`, `is_supported` (structural only), `localizer_score`, `is_absence_claim`, `is_uncertain_claim`, `verification_path`, `iou_with_ground_truth` (None at runtime) |
| `GroundingEvaluation` | Evaluation-time IoU result | `iou_with_ground_truth`, `meets_evaluation_threshold` (IoU ≥ 0.5), `iou_threshold` |
| `FilterResult` | Partitioned claims + structural USR | `supported_claims`, `unsupported_claims`, `unverifiable_claims`, `absence_claims`, `uncertain_claims`, `usr_raw`, `usr_filtered` |

**Verification/filtering behavior:**
- Runtime `verify_association` is **ground-truth-free**: `iou_with_ground_truth` is always None; `is_supported` reflects structural validity only.
- Evaluation-time `evaluate_grounding_iou_with_boxes` computes IoU ≥ 0.5 against GT boxes and is explicitly isolated from runtime filtering.
- Positive claims verified via structural validity at runtime; IoU-based support is evaluation-only.
- Absence claims (`status == "absent"`) use `verification_path="absence_answer"`: no `localizer_score`, no IoU, excluded from USR denominators.
- `region_id == "NONE"` with `status == "present"` is a **positive but ungrounded** claim, NOT an absence claim.
- Uncertain claims (`status == "uncertain"`) use `verification_path="uncertain"`, distinct meaning preserved.
- `is_supported` for absence/uncertain claims is structural only; clinical truth is determined in the evaluation layer.
- Empty `R` with a region claim → ungrounded (not an error); empty `R` with a `NONE` claim → structurally grounded.

---

## 7. Negative/Absence Claim Handling (ADR 0004 Option 3)

The following Option 3 behaviors are implemented and verified by tests:

| Requirement | Implementation | Test |
|-------------|---------------|------|
| Negative claims preserved as valid answer claims | `ExtractionResult.claims` retains `status="absent"` / `region_id="NONE"` claims | `test_extract_claims_negative_absence` |
| `region_id="NONE"` ≠ absence (status-based) | `status="present"` + `region_id="NONE"` → positive, ungrounded, NOT absence | `test_classify_positive_claim_with_none_is_not_absence` |
| `status="absent"` + `region_id="NONE"` → absence claim | Absence identified by status, not region_id | `test_classify_negative_claim_with_none_is_absence` |
| Correctness via answer-level GT (evaluation layer), not positive-region grounding | `verification_path="absence_answer"`; no IoU-based support | `test_option3_negative_claim_no_localization_confidence` |
| Excluded from positive-region grounding metrics/denominators | `absence_claims` separate; USR computed over `n_positive_claims` only | `test_option3_negative_claim_excluded_from_grounding_metrics` |
| No artificial localization confidence | `localizer_score=None` for absence claims | `test_option3_negative_claim_no_localization_confidence` |
| Empty detector output ≠ proof of absence | `R=∅` + NONE claim → `localizer_score=None`, `iou_with_ground_truth=None`, not clinically verified | `test_option3_empty_detector_output_not_verified_negative` |
| Empty region collection ≠ verified negative finding | Absence claim `is_supported` structural only; `verification_path="absence_answer"` | `test_option3_empty_detector_output_not_verified_negative` |
| Negative claims preserved even with no candidate regions | `associate_claims` grounds NONE claims when `R=∅`; filter retains them in `absence_claims` | `test_associate_claims_none_claim_grounded_when_empty` |
| Positive and negative claims follow distinct paths | `verification_path` distinguishes `positive_grounding` vs `absence_answer` | `test_option3_positive_and_negative_distinct_paths` |
| Negative claim distinct from unsupported positive | `absent+NONE` ≠ `present+NONE`; different verification paths | `test_negative_claim_distinct_from_unsupported_positive` |
| Uncertain claims preserved with distinct meaning | `status="uncertain"` → `verification_path="uncertain"`, `is_uncertain_claim=True` | `test_classify_uncertain_claim_preserved` |
| Mixed positive/negative/uncertain claims classified distinctly | All three statuses produce correct paths and flags | `test_classify_mixed_claims` |
| GT cannot influence runtime filtering | `verify_association` and `filter_claims` take no GT parameter | `test_runtime_verification_does_not_access_ground_truth`, `test_runtime_filtering_cannot_use_gt_iou` |
| Evaluation IoU isolated from runtime | `evaluate_grounding_iou_with_boxes` takes GT; does not modify runtime `is_supported` | `test_evaluation_iou_isolation_from_runtime` |
| Structural validity ≠ clinical evidence support | Runtime `is_supported=True` is structural; IoU ≥ 0.5 is evaluation-only | `test_supported_is_structural_not_clinical` |

---

## 8. Tests Added and Commands Executed

### Test file
`tests/unit/test_claim_verification_phase1.py` — **47 deterministic tests**, no model weights, no paid APIs, no real VLM inference.

**Coverage by component:**
- Claim extraction: 7 tests (valid multiple, positive, negative, empty, malformed, answer preservation, empty findings)
- Claim-to-region association: 6 tests (valid region, multiple regions, unknown region, empty collection, NONE-with-empty-R, provenance preservation)
- Negative-claim classification: 6 tests (positive+valid region, present+NONE≠absence, absent+NONE=absence, absent+empty R, uncertain preserved, mixed claims)
- Runtime GT-independence: 4 tests (no GT param in verify/filter, no GT iou at runtime, filtering can't use GT, structural≠clinical)
- Evaluation-time IoU isolation: 4 tests (evaluation takes GT, IoU≥0.5 threshold, IoU<0.5 not met, isolation from runtime, absence not evaluated vs GT, evaluate_grounding_iou takes GT)
- compute_iou utility: 4 tests (overlapping, no overlap, identical, contained)
- Unsupported claim filtering: 6 tests (supported retained, unsupported filtered, absence excluded from USR, mixed, uncertain separated, empty, original records preserved)
- ADR 0004 Option 3: 8 tests (preserved empty regions, no localization confidence, excluded from metrics, empty output not verified negative, distinct paths, negative≠unsupported positive, empty detections don't prove absence)

### Commands executed and results

```
# New Phase 1 tests (47 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_claim_verification_phase1.py -v
=> 47 passed

# Phase 1 + baseline comparison + schema regression tests (all pass)
.venv\Scripts\python.exe -m pytest tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 57 passed

# Lint on new files only
.venv\Scripts\ruff.exe check verification/ tests/unit/test_claim_verification_phase1.py
=> All checks passed!

# Type check: mypy not installed in this environment; ruff is the configured linter.
```

### Pre-existing failures (unrelated to this phase; not regressions)

Running the full `tests/unit/` suite produces 4 failures that pre-date this work and are environment-related, not caused by Phase 1 changes:

| Test | Failure cause | Related to Phase 1? |
|------|---------------|---------------------|
| `tests/unit/test_environment.py::test_python_version` | Asserts Python 3.11; environment runs Python 3.12.2 | No |
| `tests/unit/test_medsam_integration.py::test_real_single_region` | Torch `TORCH_LIBRARY` registration conflict + MedSAM checkpoint not loadable | No |
| `tests/unit/test_medsam_integration.py::test_real_multiple_regions_order` | Same torch/checkpoint issue | No |
| `tests/unit/test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` | Torch import fails in this environment | No |

These tests exercise MedSAM/torch/environment setup, none of which is touched by Phase 1 (my changes are confined to new `verification/*.py` modules, one new test file, and an ADR markdown edit). They were failing before this implementation and remain unrelated.

### Ruff lint on whole project
My new files pass `ruff check` cleanly. The 40 ruff findings reported project-wide are all in pre-existing unrelated files (`experiments/`, `inspect_slake.py`, `test_medsam_*`, `test_ollama_*`, `test_slake_*`) and were not introduced by this phase. Per the constraint to avoid unrelated refactoring, those were left untouched.

---

## 9. Compatibility and Regression Assessment

- **Existing public interfaces preserved:** `core/schemas.py`, `vlm/base.py`, `localization/detector.py`, `localization/postprocessing.py` unchanged; new modules import and reuse them.
- **No unrelated refactoring:** Only ADR 0004 markdown was modified (status/decision), plus four new verification modules and one new test file.
- **No changes to localization/segmentation semantics:** coordinate conventions (`[ymin, xmin, ymax, xmax]` normalized `[0,1]` for GT; pixel + normalized for `CandidateBox`) and RLE format unchanged.
- **Modular imports:** verification modules import lazily from `core.schemas` and sibling verification modules; no eager model imports.
- **Typing/linting:** new modules pass `ruff check`; type hints used throughout (dataclasses, `Literal`, `Optional`/`float | None`).
- **Error handling:** extraction and association never raise on malformed/empty input; failures are recorded in result objects. Verification and filtering handle empty `R`, empty GT, unknown region IDs gracefully.
- **No experimental outputs or baseline results modified.**

---

## 10. Known Limitations and Unresolved Issues

1. **USR metrics are computed but not yet consumed by an evaluation harness.** USR_raw/USR_filtered are produced by `FilterResult`; aggregate reporting (FS-010 metrics harness) is Phase 3 work.
2. **Negative-claim clinical truth is deferred.** Absence claims are marked `verification_path="absence_answer"` with `localizer_score=None`; their correctness against dataset ground truth is evaluated in a later phase (evaluation layer), consistent with ADR 0004 Option 3.
3. **IoU threshold (0.5) is a module default**, not yet fitted on validation data. Weight/threshold fitting is Phase 2 (confidence calibration).
4. **No clinical validity claim.** This is technical verification only; no clinical diagnosis or validation is implied or supported.
5. **Pre-existing environment test failures** (Python version, torch, MedSAM checkpoint) are unrelated to Phase 1 and were not introduced here.

---

## 11. Phase 2+ Work Not Implemented

Explicitly confirmed: the following were **not** implemented in this phase:
- Semantic agreement or sampling-based agreement (N=5, entropy, Kuhn et al. clustering)
- Confidence calibration / weight fitting $(w_1, w_2, w_3, \tau)$
- Abstention / selective prediction
- Risk-coverage evaluation, ECE, USR aggregate reporting, pointing accuracy
- Track A/B/C execution
- API (`POST /analyze`) or frontend integration
- New model inference or dataset experiments
- Any FS-001–FS-004 changes

Phase 1 is the claim-verification foundation only. Phase 2 (uncertainty and confidence gate) and beyond await review and approval of this implementation report.

---

## 12. Completion Criteria Confirmation

| Criterion | Status |
|-----------|--------|
| ADR 0004 records Option 3 as Accepted | ✅ Done |
| Claim extraction implemented and tested | ✅ Done (7 tests) |
| Claim-to-region association implemented and tested | ✅ Done (6 tests) |
| Region evidence verification implemented and tested | ✅ Done (6 tests) |
| Unsupported claim filtering implemented and tested | ✅ Done (5 tests) |
| Negative/absence claim handling follows Option 3 | ✅ Done (5 tests) |
| Empty detector output never interpreted as proof of absence | ✅ Verified by test |
| No synthetic localization evidence or confidence introduced | ✅ Verified by test |
| Relevant regression tests run and results reported | ✅ 42 passed (baseline + phase1 + schemas); 4 pre-existing env failures unrelated |
| Implementation report exists and accurately documents the work | ✅ This document |

**Phase 1 is complete.** Awaiting review and approval of this implementation report before proceeding to Phase 2.
