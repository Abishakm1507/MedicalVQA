# FS-005 Phase 3 Implementation Report — Evaluation Metrics and Artifact Generation

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 3 — Evaluation Metrics and Artifact Generation  
**Document ID:** `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md`  
**Status:** Implementation complete — Phase 3 only  
**Implementation type:** Implementation (code written), not design-only  
**Prerequisites:** FS-005 Phase 1 (claim extraction and region verification) and Phase 2 (uncertainty estimation and abstention gate) approved

---

## 1. Scope and Implementation Summary

Implement FS-005 Phase 3: the evaluation metrics harness and experiment artifact generation layer. This phase computes aggregate metrics from per-sample records produced by Phases 1-2 and writes deterministic experiment artifacts (JSON, CSV, Markdown).

**In scope (this phase):**
- All 9 required metrics per FS-005 technical design Section 6:
  - USR_raw (Raw Unsupported Claim Rate)
  - USR_filtered (Filtered Unsupported Claim Rate)
  - Pointing accuracy (IoU >= 0.5 against ground-truth regions)
  - Sampling agreement (semantic agreement a)
  - Normalized semantic entropy (H_n)
  - Expected Calibration Error (ECE, M=10 bins)
  - Selective accuracy
  - Coverage
  - Accuracy-vs-coverage / risk-coverage curve
  - AUROC (diagnostic)
- Experiment artifact generation (experiment_metrics.json, experiment_summary.csv, per_sample_results.csv, accuracy_vs_coverage.csv, comparison_report.md)
- Runtime/evaluation separation: ground-truth data used only in evaluation layer

**Out of scope (deferred to Phase 4+):**
- Full pipeline integration, experiment tracks (A/B/C), API integration
- New model inference or benchmark experiments
- Phase 5 validation experiments or research-result reporting
- Changes to FS-001 through FS-004

---

## 2. Files Created and Modified

### Created

| File | Explanation |
|------|-------------|
| `evaluation/metrics.py` | All metric functions + `SampleRecord` dataclass + `AggregateMetrics` dataclass (Phase 3, sections 4.8, 6). `SampleRecord` includes `evidence_state` and `sampling_valid` fields for forced-abstention eligibility. |
| `evaluation/runner.py` | `EvaluationRunner` class — metric computation orchestration + artifact writing (JSON, CSV, Markdown) |
| `tests/unit/test_evaluation_metrics_phase3.py` | 63 deterministic unit tests for all metrics and artifact generation |

### Modified

None — no existing files were modified. Phase 1 (`verification/`) and Phase 2 (`verification/semantic_clustering.py`, `verification/confidence_gate.py`) files are reused unchanged.

### Not modified
- `core/schemas.py` (reused — `MedicalVQASample`, `CandidateBox`, `GroundTruthRegion`, `EvaluationTrack` unchanged)
- `vlm/base.py` (reused unchanged)
- All FS-001–FS-004 code and experimental artifacts

---

## 3. Metric Definitions and Denominator Choices

All metrics follow the formal definitions in FS-005 technical design Section 6.

### 3.1 Overall accuracy
$$\text{Accuracy} = \frac{\sum_s \text{correct}_s}{|S|}$$

Abstained samples (correct=False) counted in denominator per FS-004 convention.

### 3.2 OPEN / CLOSED accuracy
Computed on subsets where `question_type == "OPEN"` / `"CLOSED"` respectively. Denominator = subset size. Returns None if subset is empty.

### 3.3 USR_raw
$$\text{USR}_{\text{raw}} = \frac{N_{\text{unsupported}} + N_{\text{unverifiable}}}{N_{\text{positive\_claims}}}$$

Computed over positive claims only — absence and uncertain claims excluded from denominator per ADR 0004 Option 3. Returns **None** (not 0.0) if there are no positive claims — the metric is undefined, not zero.

### 3.4 USR_filtered
$$\text{USR}_{\text{filtered}} = \frac{N_{\text{unsupported}}}{N_{\text{positive\_claims}} - N_{\text{unverifiable}}}$$

Unverifiable claims stripped from denominator. Returns **None** (not 0.0) if the filtered denominator is 0.

### 3.5 Pointing accuracy
$$\text{PointingAcc} = \frac{\sum_s \mathbf{1}[\exists \text{ claim with IoU} \ge 0.5]}{\sum_s \mathbf{1}[\text{sample has GT regions}]}$$

**Denominator: eligible samples** — samples with at least one ground-truth region (`has_gt_regions == True`). Samples with no GT regions are excluded (nothing to point to).

**Matching rules (Phase 1 `evaluate_grounding_iou_with_boxes`):**
1. For each predicted claim with a cited box, compute IoU against **every** ground-truth box
2. Take the **maximum** IoU across all GT boxes
3. A claim is correctly grounded if `max IoU >= 0.5`
4. `pointing_correct=True` means **at least one** claim met the threshold (sample-level aggregation)

Returns None if no samples have ground-truth regions.

**Limitation:** Sample-level success can conceal incorrectly grounded additional claims within the same sample. If a sample has multiple claims where one meets IoU >= 0.5 against a GT region but another does not, `pointing_correct=True` at the sample level regardless of the incorrectly grounded claim. This reflects the design's sample-level aggregation (at least one correctly grounded claim suffices) but means sample-level pointing accuracy does not capture per-claim grounding errors.

### 3.6 Sampling agreement
Mean agreement ($a$) across all samples. Returns None if no samples.

### 3.7 Normalized semantic entropy
Mean $H_n$ across all samples. Returns None if no samples.

### 3.8 ECE (Expected Calibration Error)
$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Equal-width bins over [0, 1]. Computed over accepted (in-coverage) samples only. Abstained samples excluded. Returns None if no accepted samples.

### 3.9 Selective accuracy
$$\text{SelectiveAcc} = \frac{\sum_{s \in \text{coverage}} \mathbf{1}[\text{correct}_s]}{|\text{coverage}|}$$

Computed over in-coverage samples only. Returns None if no in-coverage samples.

### 3.10 Coverage
$$\text{Coverage} = \frac{|\{s: c_s \ge \tau \text{ AND eligible}\}|}{|S|}$$

Where eligible means not forced to abstain by Phase 2 policy (`sampling_valid == True` AND `evidence_state != "uncertain"`). Invalid sampling and uncertain claims never count as covered, including at tau=0.

### 3.11 AUROC

**Population: all eligible samples**, independent of the operating threshold tau. Eligible = `sampling_valid == True` AND `evidence_state != "uncertain"`. Samples that abstain because `confidence < tau` are **included** (they have meaningful confidence). Only forced-abstention samples are excluded.

ROC-AUC of confidence predicting correctness using the Mann-Whitney U statistic. Returns None if there are no eligible samples or only one correctness class.

**Rationale:** AUROC evaluates confidence-score ranking across eligible predictions. Restricting it to accepted samples (those passing `confidence >= tau`) would make AUROC dependent on the abstention threshold, which is not a meaningful property of the confidence score itself. ECE remains accepted-sample-only conditional calibration.

**Definition:** Matches `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md` Section 6 (AUROC).

### 3.12 Accuracy-vs-coverage curve

Points $(\text{coverage}(\tau), \text{selective\_accuracy}(\tau))$ for $\tau \in \{0, 0.01, \dots, 1.0\}$.

Coverage at each tau counts only **eligible** samples (not forced to abstain) with `confidence >= tau`. Invalid sampling and uncertain claims never count as covered, including at tau=0, consistent with Phase 2's forced-abstention rules.

Selective accuracy at each tau is computed over covered eligible samples only.

---

## 4. Runtime/Evaluation Separation

This is a mandatory methodological constraint, fully enforced:

| Layer | Function | Ground-truth access? |
|-------|----------|---------------------|
| Runtime (prediction) | `verification/claim_extractor.py` | No |
| Runtime (prediction) | `verification/claim_associator.py` | No |
| Runtime (prediction) | `verification/evidence_verifier.py` (runtime `verify_association`) | No |
| Runtime (prediction) | `verification/claim_filter.py` | No |
| Runtime (prediction) | `verification/semantic_clustering.py` | No |
| Runtime (prediction) | `verification/confidence_gate.py` (`compute_confidence`, `abstain_decision`, `make_selective_prediction`) | No |
| Evaluation (offline) | `evaluation/metrics.py` (all metric functions) | Yes — takes `SampleRecord` with GT fields |
| Evaluation (offline) | `evaluation/runner.py` | Yes — writes artifacts from GT-informed records |

**Key invariants:**
- `compute_confidence()`, `abstain_decision()`, `make_selective_prediction()` accept **no ground-truth parameters**
- `make_selective_prediction()` does not accept `is_correct` — correctness is computed separately in the offline evaluation layer
- `SampleRecord` carries both runtime fields (confidence, agreement, entropy, evidence state, sampling_valid) and evaluation-only fields (ground_truth_answer, has_gt_regions, pointing_correct, correct)
- Metric functions are pure: they read `SampleRecord` fields but never modify them (verified by test `test_metrics_do_not_modify_runtime_inputs`)
- Evaluation results never feed back into runtime filtering or abstention decisions

**SampleRecord eligibility fields (from Phase 2):**
- `evidence_state`: `'present_grounded' | 'absent' | 'uncertain' | 'present_ungrounded'` — uncertain claims are forced to abstain
- `sampling_valid`: `True` if sampling produced a usable uncertainty estimate — invalid sampling is forced to abstain

These fields drive AUROC population and coverage curve eligibility, ensuring forced-abstention samples are excluded from both regardless of tau.

**ADR 0004 Option 3 compliance:**
- Absence claims excluded from USR denominators (n_positive_claims excludes absence and uncertain)
- Absence claims not treated as positive-region grounding evidence
- `region_id="NONE"` does not by itself imply absence (status-based classification from Phase 1 preserved)
- Uncertain claims remain a distinct state and always abstain (Phase 2 behavior preserved)

---

## 5. Artifact Formats and Example Output Structure

Five artifacts are written to the output directory:

### 5.1 `experiment_metrics.json`
Machine-readable metrics and configuration:
```json
{
  "experiment": "FS-005 Phase 3 evaluation",
  "n_samples": 10,
  "n_positive_claims_total": 8,
  "metrics": {
    "accuracy": 0.6,
    "open_accuracy": 0.4,
    "closed_accuracy": 0.8,
    "usr_raw": 0.25,
    "usr_filtered": 0.2,
    "pointing_accuracy": 0.75,
    "sampling_agreement": 0.7,
    "normalized_entropy": 0.3,
    "ece": 0.05,
    "selective_accuracy": 0.75,
    "coverage": 0.8,
    "auroc": 0.85
  },
  "accuracy_vs_coverage": [
    {"tau": 0.0, "coverage": 1.0, "selective_accuracy": 0.6},
    ...
  ],
  "config": {"w1": 0.4, "w2": 0.3, "w3": 0.3, "tau": 0.5, "n_samples": 5},
  "exclusions": []
}
```

### 5.2 `experiment_summary.csv`
One row per metric:
```
metric,value
n_samples,10
accuracy,0.6
usr_raw,0.25
coverage,0.8
...
```

### 5.3 `per_sample_results.csv`
One row per sample with all `SampleRecord` fields:
```
sample_id,dataset_name,split,track,condition,ground_truth_answer,model_answer,correct,question_type,has_gt_regions,pointing_correct,n_claims,n_positive_claims,n_supported,n_unsupported,n_unverifiable,n_absence,n_uncertain,agreement,normalized_entropy,confidence,threshold,is_abstaining,in_coverage
```

### 5.4 `accuracy_vs_coverage.csv`
101 rows (tau 0.0 to 1.0):
```
tau,coverage,selective_accuracy
0.0,1.0,0.6
0.01,1.0,0.6
...
1.0,0.2,1.0
```

### 5.5 `comparison_report.md`
Human-readable report with configuration, aggregate metrics, exclusions, accuracy-vs-coverage curve summary, and methodology notes.

---

## 6. Tests, Exact Commands, and Results

### Test file
`tests/unit/test_evaluation_metrics_phase3.py` — **62 deterministic tests**, no model weights, no paid APIs, no real VLM inference. Uses synthetic fixtures for all metric tests.

**Coverage by category:**

| Category | Count |
|----------|-------|
| Accuracy (overall, OPEN/CLOSED, empty, abstained) | 9 |
| USR_raw / USR_filtered (incl. None denominator cases) | 11 |
| Pointing accuracy (incl. multiple predictions, multiple GT, same-sample multiple claims) | 8 |
| Sampling agreement / normalized entropy | 4 |
| ECE (perfectly calibrated, no accepted, empty, bin boundary, max confidence) | 5 |
| Selective accuracy / coverage | 5 |
| AUROC (incl. threshold independence, None cases) | 5 |
| Accuracy-vs-coverage curve (incl. forced-abstention) | 7 |
| Aggregate metrics | 2 |
| Artifact generation (all files, JSON structure, CSV structure, report MD) | 4 |
| GT isolation (no modification of inputs, no GT in confidence paths) | 2 |
| **Total** | **63** |

### Commands executed and results

```
# Phase 3 tests (63 tests)
.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_metrics_phase3.py -q
=> 63 passed

# Phase 3 + Phase 2 + Phase 1 + baseline comparison + schema regression tests
.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_metrics_phase3.py tests/unit/test_uncertainty_phase2.py tests/unit/test_claim_verification_phase1.py tests/unit/test_baseline_comparison.py tests/unit/test_schemas.py -q
=> 181 passed

# Lint on changed Python files
.venv\Scripts\ruff.exe check evaluation/metrics.py evaluation/runner.py tests/unit/test_evaluation_metrics_phase3.py
=> All checks passed!

# Ruff on full evaluation directory
.venv\Scripts\ruff.exe check evaluation/
=> All checks passed!
```

### Pre-existing failures (unrelated to Phase 3)

Running the full `tests/unit/` suite produces 4 pre-existing environment-related failures:
- `test_environment.py::test_python_version` (expects Python 3.11, environment is 3.12.2)
- `test_medsam_integration.py::test_real_single_region` (torch TORCH_LIBRARY conflict)
- `test_medsam_integration.py::test_real_multiple_regions_order` (same torch issue)
- `test_medsam_segmenter.py::test_b02_import_error_when_segment_anything_missing` (torch import failure)

None of these are related to Phase 3 changes. No tests were modified or suppressed.

### Type checking
mypy is not installed in this environment; ruff is the project's configured linter.

---

## 7. Known Limitations, Exclusions, and Outstanding Issues

| Limitation | Explanation |
|------------|-------------|
| No fitted calibration weights | `ConfidenceCalibrator` (Phase 2) provides the interface but ships with no pre-fitted weights. Calibration must be run on held-out SLAKE validation data. |
| Cosine similarity threshold not calibrated | Default 0.75 in `SemanticClusterer` (Phase 2) is not scientifically validated; must be tuned on validation data. |
| Embedding backend not available | `sentence-transformers` not installed in this environment; `SemanticClusterer` falls back to canonical normalization. |
| No clinical validity claim | Technical verification only; no clinical diagnosis or validation implied. |
| Phase 4 not started | Full pipeline integration, experiment tracks (A/B/C), API integration not implemented. |
| Model inference not run | No Qwen, MedGemma, or other VLM inference executed in this phase. All metrics computed from synthetic per-sample fixtures. |
| N=3 separate threshold | Not implemented; interface supports configurable `tau` but separate `tau_N=3` calibration is a future task. |

---

## 8. Phase 4 and Model Inference Not Started

**Confirmed:** Phase 4 (full pipeline integration, experiment tracks, API), Phase 5 (validation experiments, research-result reporting), and all model inference (Qwen, MedGemma, MedSAM) were **not** implemented or executed. No code for those phases exists. No model weights were downloaded. No datasets were accessed beyond synthetic test fixtures.

---

## 9. Completion Criteria Confirmation

| Criterion | Status |
|-----------|--------|
| All metrics implemented per formal definitions | ✅ USR_raw, USR_filtered, pointing accuracy, sampling agreement, normalized entropy, ECE, selective accuracy, coverage, accuracy-vs-coverage, AUROC |
| USR metrics return None (not 0.0) when denominator is 0 | ✅ Both `compute_usr_raw` and `compute_usr_filtered` return `None` |
| AUROC uses eligible population independent of tau | ✅ Eligible = `sampling_valid AND evidence_state != "uncertain"` |
| Accuracy-vs-coverage curve respects forced-abstention | ✅ Invalid sampling and uncertain claims never covered, even at tau=0 |
| Pointing accuracy denominator and matching rules documented | ✅ Eligible samples = samples with GT regions; max IoU over GT boxes |
| Missing/empty/malformed/unverifiable handling matches design §4.8 table | ✅ Empty datasets return None/0.0 appropriately; absence/uncertain excluded from USR denominators; samples without GT regions excluded from pointing denominator |
| Artifacts generated (JSON, CSV, Markdown) | ✅ experiment_metrics.json, experiment_summary.csv, per_sample_results.csv, accuracy_vs_coverage.csv, comparison_report.md |
| Ruff lint clean | ✅ All Phase 3 files pass `ruff check` |
| Tests pass | ✅ 63 Phase 3 tests pass; 181 total including regressions |
| Runtime/evaluation separation preserved | ✅ No GT parameters in runtime functions; metric functions are pure and don't modify inputs |
| ADR 0004 Option 3 preserved | ✅ Absence claims excluded from USR denominators; `region_id="NONE"` not treated as absence |
| Phase 4 not started | ✅ Confirmed |
| Model inference not run | ✅ Confirmed |

**FS-005 Phase 3 is complete.** Awaiting review and approval before proceeding to Phase 4.
