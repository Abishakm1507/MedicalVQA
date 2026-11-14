# FS-005 Phase 5 — Mock Resume Diagnostic Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_MOCK_RESUME_DIAGNOSTIC.md`  
**Status:** Diagnostic complete — root cause identified and fixed  
**Scope:** Mock-mode resume/skip behavior diagnosis and fix. No real inference executed.

---

## 1. Symptom

The first mock dry run completed successfully:
- Track B: 15/15 completed
- Track C: 15/15 completed
- 0 failures or corrupt artifacts

The second mock dry run against the same output directory reported:
- `n_completed` equal to the first run (samples regenerated, not skipped)
- `n_skipped=0` for both tracks (expected: 15 skipped for each track)

This violates the resume requirement: completed artifacts must be recognized and skipped on restart, preventing duplicate generation.

---

## 2. Root Cause

Two issues were identified in `scripts/fs005_phase5_generation.py`, `GenerationRunner.run_batch()`:

### Issue 1: Progress counter accumulation (primary)

`_load_progress()` loads the progress file from the previous run, which contains cumulative counters (`n_completed`, `n_skipped`, etc.). On the second `run_batch()` call, these counters were **not reset**, so:

- `n_skipped` accumulated across runs (Run 3 showed `n_skipped=30` instead of 15)
- `n_completed` was preserved from the prior run even when all samples were skipped

This made it appear that samples were being regenerated rather than skipped, because the summary reported non-zero `n_completed` even when every sample was skipped.

### Issue 2: Dead code (secondary)

Lines in `run_batch()` that handled `action == "skipped"` returned from `generate_or_load()` were unreachable dead code. `run_batch()` performs its own `is_completed_valid()` check **before** calling `generate_or_load()`, so `generate_or_load()` is only invoked when the artifact is missing, corrupt, or failed — never when it's valid-completed. The `generate_or_load()` internal skip check can never trigger in this call path.

---

## 3. Fix

### Change 1: Reset per-call counters at the start of `run_batch()`

Added counter reset after `_load_progress()` in `run_batch()` (`scripts/fs005_phase5_generation.py:560-569`):

```python
progress = self._load_progress(len(samples))
# Reset per-call summary counters. The progress file retains
# completed_question_ids / failed_question_ids for resumption, but the
# counters reported in this run's summary must start from zero so that
# re-running against an already-complete directory reports skips, not
# accumulated counts from prior runs.
progress.n_completed = 0
progress.n_skipped = 0
progress.n_failed = 0
progress.n_retried = 0
progress.n_corrupt = 0
```

The progress file still tracks `completed_question_ids`, `failed_question_ids`, `corrupt_question_ids`, and `failures` for resumption (knowing which samples are done). Only the summary counters are per-call.

### Change 2: Removed dead code

Removed the unreachable `if action == "skipped": progress.n_skipped += 1` and `elif action == "retried": progress.n_retried += 1` block inside the `else` branch of `run_batch()`. Kept the `n_retried` increment (reachable when `generate_or_load()` retries a FAILED record).

---

## 4. Artifact Evidence

Verified with a clean reproduction (mock mode, no model weights):

### Before fix

| Run | Track B | Track C |
|-----|---------|---------|
| 1 (fresh) | completed=15, skipped=0 | completed=15, skipped=0 |
| 2 (resume) | completed=15, skipped=15 | completed=15, skipped=15 |
| 3 (new runner) | completed=15, skipped=30 | completed=15, skipped=30 |

### After fix

| Run | Track B | Track C |
|-----|---------|---------|
| 1 (fresh) | completed=15, skipped=0 | completed=15, skipped=0 |
| 2 (resume) | completed=0, skipped=15 | completed=0, skipped=15 |
| 3 (new runner) | completed=0, skipped=15 | completed=0, skipped=15 |

Artifact count remains exactly 15 per track across all runs (no duplicates, no overwrites). Artifact bytes are byte-identical across runs.

---

## 5. Test Results

### New tests added (`tests/unit/test_pilot_cohort.py`)

| Test | Purpose |
|------|---------|
| `test_resume_from_persistent_artifact_directory` | Skip on second run, counter reset, new runner instance (Colab restart), artifact count stability, artifact validity |
| `test_resume_retries_failed_without_overwriting_valid` | FAILED records retried and replaced; COMPLETED records never overwritten |
| `test_resume_detects_corrupt_artifact` | Corrupt artifacts detected, regenerated on re-run |
| `test_duplicate_prevention_no_new_artifacts_on_rerun` | No new artifact files created on skip; existing artifacts byte-identical |

### Updated tests (`tests/unit/test_resume_execution.py`)

| Test | Change |
|------|--------|
| `test_second_run_skips_all_duplicate_prevention` | `n_completed` changed from 3 to 0 on second run (per-call counter reset) |
| `test_interruption_resume_progress_persisted` | Progress file `n_completed` changed from 3 to 0 on second run |

### Test execution

```
.venv\Scripts\python -m pytest tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py
```

**Result: 48 passed in 7.30s**

### Ruff

```
.venv\Scripts\python -m ruff check core/generation_artifact.py scripts/fs005_phase5_generation.py scripts/fs005_phase5_calibration_split.py scripts/fs005_phase5_pilot_cohort.py colab/fs005_phase5_colab_pilot.py evaluation/bootstrap.py tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py
```

**Result: All checks passed!** (0 errors on all changed Python files.)

---

## 6. Verification that no real inference was executed

- `RUN_REAL_INFERENCE = False` in `colab/fs005_phase5_colab_pilot.py` (asserted by `test_real_inference_flag_disabled`)
- All tests use `MockVLMClient` — no model weights downloaded, no inference performed
- No calibration fitting, no test cohort evaluation
- FS-001 through FS-004 unmodified
- ADR 0004 and ADR 0005 unmodified

---

## 7. Changed Files

| File | Change |
|------|--------|
| `scripts/fs005_phase5_generation.py` | Reset per-call counters in `run_batch()`; removed dead code; `n_retried` increment preserved |
| `tests/unit/test_pilot_cohort.py` | Expanded resume tests: counter reset, new runner instance, failed retry, corrupt detection, duplicate prevention |
| `tests/unit/test_resume_execution.py` | Updated expectations for per-call counter reset |

---

## 8. Remaining Limitations

- `n_skipped` in the progress file now reflects only the current run (not cumulative). Callers that need cumulative skip counts across multiple `run_batch()` invocations must aggregate summaries themselves.
- `completed_question_ids` in the progress file may contain duplicates across runs (a QID is appended each time it's encountered, even when skipped). This is harmless for resumption logic (membership check) but may be noisy for auditing.
- Real inference has not been executed; `RUN_REAL_INFERENCE = False`. The mock resume behavior has been verified but real-model resume behavior on Colab T4 remains untested.
