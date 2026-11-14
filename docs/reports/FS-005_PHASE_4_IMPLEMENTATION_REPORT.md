# FS-005 Phase 4 Implementation Report — Full Pipeline Integration and Experiment Track Orchestration

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 4 — Full Pipeline Integration and Experiment Track Orchestration  
**Document ID:** `docs/reports/FS-005_PHASE_4_IMPLEMENTATION_REPORT.md`  
**Status:** Implementation complete — Phase 4 only  
**Implementation type:** Implementation (code written), not design-only  
**Prerequisites:** FS-005 Phase 1 (claim extraction and region verification), Phase 2 (uncertainty estimation and abstention gate), and Phase 3 (evaluation metrics and artifact generation) complete and approved

---

## 1. Objective and Implementation Scope

Implement FS-005 Phase 4: full pipeline integration integrating Phases 1-3 into a coherent pipeline, plus experiment track orchestration and FastAPI API integration.

**In scope (this phase):**
- `verification/pipeline.py` — `PipelineEngine` orchestrating Phase 1-3 components
- `api/pipeline_service.py` — `PipelineService` + FastAPI app with `POST /analyze` endpoint
- `tests/unit/test_pipeline_integration.py` — 16 deterministic integration tests

**Out of scope (deferred to Phase 5):**
- Validation experiments or research-result reporting
- Real model inference or benchmark experiments
- New model inference training or fine-tuning
- Clinical validation or deployment

---

## 2. Architecture Inspected Before Implementation

| Component | Path | Role |
|-----------|------|------|
| Claim extraction | `verification/claim_extractor.py` | Phase 1 — extract `ExtractedClaim` from `VLMStructuredOutput` |
| Claim association | `verification/claim_associator.py` | Phase 1 — associate claims with candidate regions |
| Evidence verification | `verification/evidence_verifier.py` | Phase 1 — IoU-based region verification |
| Claim filtering | `verification/claim_filter.py` | Phase 1 — filter claims (structural USR) |
| Semantic clustering | `verification/semantic_clustering.py` | Phase 2 — Kuhn et al. clustering + entropy |
| Confidence gate | `verification/confidence_gate.py` | Phase 2 — confidence calculation + abstention |
| Metrics | `evaluation/metrics.py` | Phase 3 — all 9 evaluation metrics |
| Runner | `evaluation/runner.py` | Phase 3 — artifact generation |
| Schemas | `core/schemas.py` | `MedicalVQASample`, `CandidateBox`, `GroundTruthRegion`, `VLMStructuredOutput` |
| Remote worker | `api/remote_worker.py` | Existing sequential model loading |

---

## 3. Integration Design and Execution Flow

The pipeline executes in the correct logical order:

1. **Region proposal** — detector proposed boxes (Track A), GT boxes promoted (Track B), or empty set (Track C)
2. **VLM inference** — N sampled answers via `VLMClient.sample_n()`
3. **Claim extraction** — `claim_extractor.extract_claims()` → `ExtractedClaim[]`
4. **Claim-to-region association** — `claim_associator.associate_claims()` → `AssociatedClaim[]`
5. **Runtime evidence verification** — `evidence_verifier.verify_association()` → `VerifiedClaim[]` (ground-truth-free)
6. **Claim filtering** — `claim_filter.filter_claims()` → `FilterResult` (structural USR)
7. **Semantic clustering** — `semantic_clustering.SemanticClusterer` → agreement + entropy
8. **Confidence calculation** — `confidence_gate.compute_confidence()` with evidence-state policy
9. **Selective prediction** — `confidence_gate.make_selective_prediction()` with abstention gate
10. **Evaluation record generation** — `SampleRecord` for Phase 3 `compute_aggregate_metrics()`

---

## 4. Files Created and Modified

### Created

| File | Explanation |
|------|-------------|
| `verification/pipeline.py` | `PipelineEngine` — end-to-end pipeline orchestration; `PipelineConfig`, `PipelineResult`, `AnswerCondition` types |
| `api/pipeline_service.py` | `PipelineService` — service layer; FastAPI app with `POST /analyze`, `GET /health`, `GET /tracks`; `AnalyzeRequest`, `AnalyzeResponse` Pydantic models |
| `tests/unit/test_pipeline_integration.py` | 16 deterministic unit tests for Phase 4 pipeline integration |

### Modified

None — all FS-001 through FS-004 code and experimental artifacts untouched. Phase 1-3 modules reused unchanged.

---

## 5. Component Integration Details

| Phase | Component | Integration Point |
|-------|-----------|-------------------|
| FS-005 Phase 1 | `claim_extractor.extract_claims()` | Called on each of N VLM structured outputs |
| FS-005 Phase 1 | `claim_associator.associate_claims()` | Associates extracted claims with candidate boxes |
| FS-005 Phase 1 | `evidence_verifier.verify_association()` | Structural verification (no GT IoU) |
| FS-005 Phase 1 | `claim_filter.filter_claims()` | Structural USR computation |
| FS-005 Phase 2 | `semantic_clustering.SemanticClusterer` | Agreement + normalized entropy from N answers |
| FS-005 Phase 2 | `confidence_gate.compute_confidence()` | Confidence with evidence-state-aware policy |
| FS-005 Phase 2 | `confidence_gate.make_selective_prediction()` | Abstention gate decision |
| FS-005 Phase 3 | `evaluation.metrics.SampleRecord` | Per-sample record for aggregate metrics |
| FS-005 Phase 3 | `evaluation.runner.EvaluationRunner` | Artifact generation (JSON, CSV, Markdown) |

---

## 6. Experiment Track Implementation

### Track A — Predicted Localization (CXR)

| Aspect | Implementation |
|--------|---------------|
| Region source | Detector-proposed `CandidateBox[]` via `MockRegionDetector` |
| Eligible data | VQA-RAD CXR or RSNA/VinDr-CXR images only |
| SLAKE eligibility | **Excluded** — SLAKE contains zero chest X-ray images |
| Region IDs | `R1`..`Rk` from detector |

**Implementation:** `PipelineEngine._propose_regions()` calls `region_detector.detect(image)` for Track A. If detector returns empty set, `R = ∅` (valid state, no synthetic boxes).

### Track B — Oracle Ground-Truth Localization

| Aspect | Implementation |
|--------|---------------|
| Region source | Dataset ground-truth boxes promoted to `CandidateBox` |
| Eligible data | All SLAKE modalities (chest CT, abdomen CT, brain MRI) with GT regions |
| Empty GT samples | Excluded from pointing-accuracy denominator |

**Implementation:** `PipelineEngine._propose_regions()` promotes `GroundTruthRegion[]` to `CandidateBox[]` with score=1.0. GT boxes are used as oracle region proposals only — runtime verification still uses structural validity, not IoU against GT.

### Track C — Ungrounded Baseline

| Aspect | Implementation |
|--------|---------------|
| Region source | None (`R = ∅`) |
| Eligible data | VQA-RAD and SLAKE (all modalities) |
| Region IDs | None — all claims cite `NONE` or are ungrounded |

**Implementation:** `PipelineEngine._propose_regions()` returns empty list for Track C.

---

## 7. Answer Condition Implementation

| Condition | Implementation |
|-----------|---------------|
| **Condition 1 — Free-form** | N=1 greedy sample (T=0.0), no JSON contract, no region grounding, no verification, no abstention |
| **Condition 2 — Grounded, no abstention** | N=5 samples, claim verification, no confidence-gate abstention |
| **Condition 3 — Grounded + abstention** | N=5 samples, claim verification, confidence-gate abstention |

**Per-condition outputs:** Each `PipelineResult` carries `raw_answer` (Condition 1), `filtered_answer` (Condition 2), and `selective_answer` (Condition 3). Condition 3 may return `None` for `selective_answer` when abstained.

---

## 8. API Integration Details

**FastAPI app** (`api/pipeline_service.py`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /analyze` | POST | Full FS-005 pipeline execution with structured JSON response |
| `GET /health` | GET | Health check |
| `GET /tracks` | GET | List available tracks and answer conditions |

**Request model** (`AnalyzeRequest`): sample_id, dataset_name, split, image_id, image_path, modality, organ, question, answer, question_type, ground_truth_regions, has_ground_truth_regions, track, condition, model_name, n_samples, temperature, w1, w2, w3, tau, random_seed.

**Response model** (`AnalyzeResponse`): request_id, sample_id, track, condition, question, model, status (ACCEPTED/ABSTAINED), final_answer, is_abstaining, confidence_score, confidence_breakdown, regions, findings, raw_answer, filtered_answer, selective_answer, n_* counts, agreement, normalized_entropy.

**Runtime/evaluation separation:** The API does not pass ground-truth to runtime confidence or abstention. GT fields (`ground_truth_regions`, `has_ground_truth_regions`) are carried in the sample for evaluation-layer use only.

---

## 9. Runtime vs Evaluation Ground-Truth Separation

**Runtime (ground-truth-free):**
- `verify_association()` — no GT parameter
- `filter_claims()` — no GT parameter
- `compute_confidence()` — no GT parameter
- `abstain_decision()` — no GT parameter
- `make_selective_prediction()` — no GT parameter

**Evaluation (ground-truth-allowed):**
- `evaluate_grounding_iou_with_boxes()` — GT boxes for pointing accuracy
- `compute_aggregate_metrics()` — GT answers for accuracy calculation
- `compute_ece()` — GT correctness for calibration

**ADR 0004 Option 3 compliance:**
- Absence claims (`status="absent"`, `region_id="NONE"`) → `verification_path="absence_answer"`, `localizer_score=None`, excluded from USR denominators
- Present+ungrounded (`status="present"`, `region_id="NONE"`) → `rho=0.0`, no rescaling
- Uncertain claims (`status="uncertain"`) → `verification_path="uncertain"`, distinct state
- Empty detector output → never interpreted as proof of absence

---

## 10. Metric Integration

| Metric | Implementation | Ground-Truth Dependency |
|--------|---------------|------------------------|
| Overall accuracy | `compute_accuracy()` | Evaluation-only (correctness) |
| OPEN/CLOSED accuracy | `compute_open_accuracy()`, `compute_closed_accuracy()` | Evaluation-only |
| USR_raw | `compute_usr_raw()` | Structural (no GT) |
| USR_filtered | `compute_usr_filtered()` | Structural (no GT) |
| Pointing accuracy | `compute_pointing_accuracy()` | Evaluation-only (GT IoU) |
| Sampling agreement | `compute_sampling_agreement()` | Runtime (no GT) |
| Normalized entropy | `compute_normalized_entropy()` | Runtime (no GT) |
| ECE | `compute_ece()` | Evaluation-only (accepted samples) |
| Selective accuracy | `compute_selective_accuracy()` | Evaluation-only (accepted samples) |
| Coverage | `compute_coverage()` | Runtime (no GT) |
| AUROC | `compute_auroc()` | Evaluation-only (eligibility-based, not threshold-based) |
| Accuracy-vs-coverage | `compute_accuracy_vs_coverage_curve()` | Runtime evaluation (eligible samples) |

**AUROC population:** All eligible samples (`sampling_valid=True` AND `evidence_state!="uncertain"`), independent of abstention threshold. Returns `None` for no eligible samples or single correctness class.

**USR denominators:** Positive claims only — absence and uncertain claims excluded per ADR 0004 Option 3. Returns `None` when denominator is 0.

---

## 11. Tests Added and Exact Results

### Test file: `tests/unit/test_pipeline_integration.py`

**16 deterministic tests**, no model weights, no paid APIs, no real VLM inference. Uses `MockVLMClient` and `MockRegionDetector`.

| Category | Count | Test Names |
|----------|-------|------------|
| Component integration | 5 | test_pipeline_end_to_end_present_grounded, test_pipeline_absence_claim_no_rho, test_pipeline_uncertain_claim_abstains, test_pipeline_present_ungrounded_no_rescaling, test_pipeline_invalid_sampling_abstains_tau_zero |
| Track implementation | 4 | test_track_a_uses_predicted_regions, test_track_b_uses_oracle_gt_boxes, test_track_c_ungrounded_no_regions, test_track_a_ineligible_for_slake |
| Answer conditions | 3 | test_condition_1_free_form, test_condition_2_grounded_no_abstention, test_condition_3_grounded_with_abstention |
| Metadata preservation | 1 | test_metadata_preserved_per_sample |
| Evaluation record generation | 1 | test_evaluation_record_from_pipeline |
| Artifact generation | 1 | test_artifact_generation_from_pipeline_results |
| **Total** | **16** | |

### Commands executed and results

```
# Phase 4 pipeline integration tests (16 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline_integration.py -v
=> 16 passed

# All test suites together (227 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline_integration.py tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_evaluation_metrics_phase3.py tests/unit/test_schemas.py -q
=> 227 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check verification/pipeline.py api/pipeline_service.py tests/unit/test_pipeline_integration.py
=> All checks passed!
```

### Pre-existing failures (unrelated to Phase 4)

Running the full `tests/unit/` suite produces 4 pre-existing environment-related failures:
- `test_environment.py::test_python_version` (expects Python 3.11, environment is 3.12.2)
- `test_medsam_integration.py::test_real_single_region` (torch TORCH_LIBRARY conflict)
- `test_medsam_integration.py::test_real_multiple_regions_order` (same torch issue)
- `test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` (torch import failure)

None are related to Phase 4 changes. No tests were modified or suppressed.

### Type checking

mypy is not installed in this environment; ruff is the project's configured linter.

---

## 12. Known Limitations and Unresolved Issues

| Limitation | Explanation |
|------------|-------------|
| No fitted calibration weights | `ConfidenceCalibrator` (Phase 2) provides the interface but ships with no pre-fitted weights. Calibration must be run on held-out SLAKE validation data. |
| No real model inference | All pipeline tests use `MockVLMClient` and `MockRegionDetector`. No Qwen, MedGemma, or other VLM inference executed. |
| MedSAM segmenter unused in pipeline | `segmenter` parameter accepted but not used in verification pipeline — available for visual overlay generation in future phases |
| Pointing accuracy limitation | Sample-level success can conceal incorrectly grounded additional claims within the same sample |
| Negative-claim clinical truth deferred | Absence claims marked `verification_path="absence_answer"`; correctness against dataset GT evaluated in Phase 3 evaluation layer |

---

## 13. Confirmation Statements

| Statement | Status |
|-----------|--------|
| No real model inference executed | ✅ All tests use `MockVLMClient` and `MockRegionDetector` |
| No Phase 5 started | ✅ No Phase 5 code written |
| FS-001 through FS-004 unmodified | ✅ No changes to FS-001–FS-004 code or artifacts |
| ADR 0004 Option 3 preserved | ✅ Absence claims: no artificial rho, excluded from USR denominators, answer-level correctness deferred |
| ADR 0005 preserved | ✅ N=5 default, configurable weights/threshold, semantic clustering |
| Runtime/evaluation separation | ✅ No GT parameters in runtime functions; GT used only in evaluation layer |
| No paid APIs or services | ✅ All mocks are deterministic and free |
| Ruff passes | ✅ All Phase 4 files pass `ruff check` |

---

## 14. Recommendation

FS-005 Phase 4 is complete. The pipeline integrates Phases 1-3 cleanly, with:

- Ground-truth-free runtime verification
- ADR 0004 Option 3 absence-claim handling
- ADR 0005 N=5 sampling standard
- Three experimental tracks (A/B/C) × three answer conditions (1/2/3)
- FastAPI `POST /analyze` endpoint following existing conventions
- Phase 3 evaluation metrics integration
- Deterministic artifact generation

**Phase 4 is ready for review.** Phase 5 (validation experiments) has not been started.
