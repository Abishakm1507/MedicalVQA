# FS-005 Phase 3 Corrective Review — Metric Semantics and Edge-Case Correction

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 3 — Evaluation Metrics and Artifact Generation  
**Document ID:** `docs/reports/FS-005_PHASE_3_CORRECTIVE_REVIEW.md`  
**Status:** Corrective review complete — all 4 issues resolved  
**Review type:** Independent senior ML engineering and medical AI research methodology audit  
**Implementation reviewed:** `evaluation/metrics.py`, `evaluation/runner.py`, `tests/unit/test_evaluation_metrics_phase3.py`

---

## 1. Issues Identified and Resolved

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | USR_raw and USR_filtered returned 0.0 when denominator is 0, misrepresenting undefined metrics | Critical | Fixed |
| 2 | AUROC restricted to accepted samples (in_coverage) instead of using full eligible population | Critical | Fixed (documented discrepancy with design below) |
| 3 | Accuracy-vs-coverage curve didn't respect forced-abstention rules (invalid sampling, uncertain claims could be covered at tau=0) | Critical | Fixed |
| 4 | Pointing accuracy aggregation and matching rules undocumented; missing edge-case tests | Medium | Fixed |

---

## 2. Issue 1 — Undefined USR Metrics

### 2.1 Root Cause

`compute_usr_raw()` returned `0.0` when `n_positive_claims == 0`, and `compute_usr_filtered()` returned `0.0` when the filtered denominator was 0. This misrepresented undefined metrics as genuinely zero-valued.

### 2.2 Fix

Both functions now return `None` when their denominator is 0:

| Function | Condition | Return value |
|----------|-----------|-------------|
| `compute_usr_raw` | `n_positive_claims == 0` | `None` |
| `compute_usr_filtered` | `n_positive_claims - N_unverifiable == 0` | `None` |

### 2.3 Artifact handling

`None` values are correctly serialized:
- JSON: `null`
- CSV: empty string
- Markdown: `None` (Python str representation)

### 2.4 Tests added

- `test_compute_usr_raw_no_positive_claims` — asserts `None` (was `0.0`)
- `test_compute_usr_filtered_denominator_zero` — asserts `None` (was `0.0`)

---

## 3. Issue 2 — AUROC Population

### 3.1 Root Cause

AUROC was computed only over `in_coverage` samples (those passing the operating threshold `confidence >= tau`). This meant AUROC changed when tau changed, even though the underlying prediction records were identical.

### 3.2 Fix

AUROC now uses the **full eligible population**, independent of tau:

| Population criterion | Included? |
|---------------------|-----------|
| `sampling_valid == True` | Yes |
| `sampling_valid == False` (invalid sampling) | No |
| `evidence_state == "uncertain"` (forced abstention) | No |
| `evidence_state != "uncertain"` AND `sampling_valid == True` | Yes (regardless of `in_coverage`) |

Samples that abstain because `confidence < tau` are **included** — they have meaningful confidence. Only forced-abstention samples are excluded.

### 2.3 Return values

| Condition | Return value |
|-----------|-------------|
| No eligible samples | `None` |
| Only one correctness class in eligible population | `None` |
| Normal case | Mann-Whitney U statistic in [0, 1] |

### 2.4 Tests added/modified

- `test_compute_auroc_threshold_independence` — AUROC identical across different tau values with same underlying records
- `test_compute_auroc_no_eligible` — `None` when all samples are forced-abstention
- `test_compute_auroc_one_class` — changed from `0.5` to `None`

---

## 4. Issue 3 — Accuracy-vs-Coverage Curve

### 3.1 Root Cause

The curve used `confidence >= tau` as the sole coverage criterion, without considering forced-abstention rules. Invalid sampling and uncertain claims could be counted as covered at `tau=0` (since `0.0 >= 0.0`).

### 3.2 Fix

Added `_is_eligible_for_coverage()` helper:

```python
def _is_eligible_for_coverage(record: SampleRecord) -> bool:
    return record.sampling_valid and record.evidence_state != "uncertain"
```

The curve now:
1. Filters to eligible samples first (not forced to abstain)
2. At each tau, counts `eligible AND confidence >= tau` as covered
3. Coverage denominator = total samples (including forced-abstention)
4. Selective accuracy = correct / covered among eligible samples

### 3.3 Tests added

- `test_compute_accuracy_vs_coverage_curve_invalid_sampling_never_covered`
- `test_compute_accuracy_vs_coverage_curve_uncertain_claims_never_covered`
- `test_compute_accuracy_vs_coverage_curve_mixed_forced_abstention`

---

## 5. Issue 4 — Pointing Accuracy Aggregation

### 5.1 Denominator rule

**Denominator: eligible samples** — samples with at least one ground-truth region (`has_gt_regions == True`). Samples without GT regions are excluded (nothing to point to).

### 5.2 Matching rules

Pointing correctness is determined by Phase 1's `evaluate_grounding_iou_with_boxes`:
1. For each predicted claim with a cited box, compute IoU against **every** ground-truth box
2. Take the **maximum** IoU across all GT boxes
3. A claim is correctly grounded if `max IoU >= 0.5`
4. `pointing_correct=True` means **at least one** claim met the threshold

This is a sample-level aggregation: "exists a claim with IoU >= 0.5" — not claim-level.

### 5.3 Tests added

- `test_compute_pointing_accuracy_multiple_predictions` — multiple claims, one grounded correctly
- `test_compute_pointing_accuracy_multiple_gt_regions` — multiple GT boxes, max IoU used
- `test_compute_pointing_accuracy_no_predictions` — no claims, pointing_correct=False
- `test_compute_pointing_accuracy_multiple_claims_one_sample` — one sample with one correctly grounded claim and one incorrectly grounded claim → `pointing_correct=True` at sample level

**Limitation:** Sample-level success can conceal incorrectly grounded additional claims within the same sample. If a sample has multiple claims where one meets IoU >= 0.5 against a GT region but another does not, `pointing_correct=True` at the sample level regardless of the incorrectly grounded claim. This reflects the design's sample-level aggregation (at least one correctly grounded claim suffices) but means sample-level pointing accuracy does not capture per-claim grounding errors.

---

## 6. SampleRecord changes

Added two new fields to `SampleRecord`:

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `evidence_state` | `str` | Phase 2 `EvidenceState` | Identifies uncertain claims for forced-abstention exclusion |
| `sampling_valid` | `bool` | Phase 2 `compute_confidence` | Identifies invalid sampling for forced-abstention exclusion |

Both are needed for correct AUROC population and coverage curve computation.

### AggregateMetrics type changes

| Field | Old type | New type |
|-------|----------|----------|
| `usr_raw` | `float` | `float \| None` |
| `usr_filtered` | `float` | `float \| None` |

---

## 7. Files Created and Modified

### Modified

| File | Change |
|------|--------|
| `evaluation/metrics.py` | Added `evidence_state`/`sampling_valid` to `SampleRecord`; USR metrics return `None` on zero denominator; AUROC uses eligible population; coverage curve respects forced-abstention; `AggregateMetrics` types updated |
| `evaluation/runner.py` | No code changes needed (None values handled natively by json/csv) |
| `tests/unit/test_evaluation_metrics_phase3.py` | Updated USR expectations (None vs 0.0), AUROC tests (threshold independence, None for single class), added forced-abstention curve tests, added pointing accuracy matching tests |

### Not modified

- `core/schemas.py`, `vlm/base.py`, `verification/` (Phase 1/2), FS-001–FS-004

---

## 8. Tests and Results

### Test count: 63 Phase 3 tests

| Category | Count |
|----------|-------|
| Accuracy (overall, OPEN/CLOSED, empty, abstained) | 9 |
| USR_raw / USR_filtered (incl. None denominator cases) | 11 |
| Pointing accuracy (incl. multiple predictions, multiple GT) | 8 |
| Sampling agreement / normalized entropy | 4 |
| ECE | 5 |
| Selective accuracy / coverage | 5 |
| AUROC (incl. threshold independence, None cases) | 5 |
| Accuracy-vs-coverage curve (incl. forced-abstention) | 7 |
| Aggregate metrics | 2 |
| Artifact generation | 4 |
| GT isolation | 2 |
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

None are related to Phase 3 changes.

---

## 9. Remaining Discrepancy with Technical Design

No discrepancies remain. The technical design Section 6 (AUROC definition) was updated to match the implementation:

**AUROC population (now aligned):** The technical design Section 6 now defines AUROC over the full eligible population — all samples with `sampling_valid == True` AND `evidence_state != "uncertain"` — independent of the abstention threshold. Single correctness class returns None. ECE remains accepted-sample-only conditional calibration, as specified in the design.

**Rationale:** AUROC evaluates confidence-score ranking across eligible predictions. Restricting AUROC to accepted samples ($c \ge \tau$) would make it dependent on the abstention threshold, which is not a meaningful property of the confidence score itself.

**All aspects match the technical design:**
- USR_raw and USR_filtered return `None` (not 0.0) when denominator is 0
- AUROC computed over full eligible population, independent of tau
- Accuracy-vs-coverage curve respects forced-abstention rules
- Pointing accuracy denominator and matching rules documented per design

---

## 10. Phase 4 Not Started

**Confirmed:** Phase 4 (full pipeline integration, experiment tracks, API), Phase 5 (validation experiments), and all model inference were **not** implemented or executed.
