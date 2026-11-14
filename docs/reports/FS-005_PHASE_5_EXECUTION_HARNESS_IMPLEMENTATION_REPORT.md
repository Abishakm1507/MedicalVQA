# FS-005 Phase 5 — Execution Harness Implementation Report

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_EXECUTION_HARNESS_IMPLEMENTATION_REPORT.md`  
**Status:** Implementation complete — pilot-ready; inference not executed  
**Scope:** FS-005 Phase 5 execution infrastructure only. No model inference, calibration, or test evaluation executed.

---

## 1. Objective

Implement and verify the missing FS-005 Phase 5 experiment execution infrastructure so that a small Colab T4 feasibility pilot can be run safely. This task is implementation and verification only. No real model inference, calibration, or test evaluation was executed, and the 40,210-generation experiment was not started.

---

## 2. Files Created

| File | Purpose |
|------|---------|
| `core/generation_artifact.py` | Condition-independent generation-artifact system: `GenerationRecord`, `GenerationSample`, deterministic `compute_generation_id`, `validate_generation_record`, and `GenerationArtifactStore` (persist/load/exists/list/corrupt-detection). |
| `scripts/fs005_phase5_generation.py` | Resumable execution harness: `GenerationRunner` — N=5 generation once per (model, track, sample), Conditions 2 & 3 from the same persisted artifact, resume/skip/retry/corrupt handling, incremental progress + failure summaries. |
| `scripts/fs005_phase5_calibration_split.py` | Calibration fit/assessment split by image ID for Track B (970 GT samples) and Track C (1,053 samples), per model/track, with manifests and GT-region treatment documentation. |
| `evaluation/bootstrap.py` | Image-clustered bootstrap (`bootstrap_metric`, `bootstrap_paired_difference`) and sample-level McNemar (`mcnemar_paired`, explicitly NOT image-cluster-adjusted). Records seed + replicate count; >= 1,000 replicates. |
| `scripts/fs005_phase5_validate_cohort.py` | Paired cohort consistency validation: Track B = 864 GT-annotated, Track C = 945 eligible, paired = exactly the same 864 QIDs/image IDs, 81 Track C-only kept separate; rejects mismatched paired cohorts. |
| `tests/unit/test_generation_artifact.py` | Artifact serialization/validation, deterministic IDs, store skip/retry/corrupt detection. |
| `tests/unit/test_condition_reuse.py` | Conditions 2 & 3 consume identical source artifacts. |
| `tests/unit/test_resume_execution.py` | Interruption/resume, duplicate prevention, corrupt-artifact detection and regeneration. |
| `tests/unit/test_calibration_split.py` | Image-level disjointness, all-questions-per-image preservation, manifest consistency, pilot exclusion. |
| `tests/unit/test_paired_cohort.py` | Paired Track B/C cohort consistency (864/945/81/864). |
| `tests/unit/test_cluster_bootstrap.py` | Cluster-bootstrap behavior: cluster preservation, seed/replicate recording, >= 1,000 replicates, paired same-clusters, McNemar not cluster-adjusted. |
| `scripts/fs005_phase5_pilot_cohort.py` | Validation-only pilot cohort selection (10–20 image-disjoint validation samples); excludes pilot image IDs from calibration fit/assessment; regenerates calibration split manifests; preserves test cohort. |
| `colab/fs005_phase5_colab_pilot.py` | Colab T4 execution entry point: Drive mount, persistent output under MyDrive/MedicalVQA/outputs/fs005_phase5/, HF auth via Colab Secrets, sequential MedGemma/Qwen loading, GenerationRunner with real VLMClient, Tracks B & C, N=5 response reuse across Conditions 2 & 3, resume/skip-completed, mock-mode dry run by default. |
| `tests/unit/test_pilot_cohort.py` | Pilot/calibration image-ID disjointness, pilot/test separation, persistent output path config, resume from persistent artifact directory. |
| `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md` | Colab T4 feasibility-pilot execution guide (rewritten: validation-only cohort, Drive persistence, Colab commands, safety criteria). |
| `docs/reports/FS-005_PHASE_5_PILOT_SAFETY_REVIEW.md` | Pilot safety review (see separate document). |
| `docs/reports/FS-005_PHASE_5_FINAL_PILOT_EXECUTION_AUDIT.md` | Final pilot execution consistency audit (this report's sibling): calibration reconciliation, Colab invocation, Secrets mechanism, model/track matrix, verification results. |

---

## 3. Files Modified

| File | Change |
|------|--------|
| `docs/reports/FS-005_PHASE_5_EXPERIMENT_EXECUTION_PLAN.md` | Corrected grand total 30,210 → 40,210 unique generations; added Section 2.4 validation-only pilot cohort (15 image-disjoint validation samples, calibration exclusion, test-cohort preservation); removed unverified 20–32 hour estimate. |
| `docs/reports/FS-005_PHASE_5_FINAL_REVIEW.md` | Same workload corrections (30,210 → 40,210; 29,829 → 40,210; unverified estimate removed). |
| `core/generation_artifact.py` | Added verify-after-write (re-read + re-validate persisted artifact) so truncated/unreadable writes are detected before resume trusts them. Root already configurable via constructor (supports Google Drive path). |
| `scripts/fs005_phase5_calibration_split.py` | Added `--exclude-image-ids` and `--output-dir` CLI options so pilot image IDs can be excluded from calibration fit/assessment and manifests written to a configurable directory. |

### Files NOT Modified (per restrictions)
- FS-001 through FS-004 artifacts and behavior: untouched.
- ADR 0004 and ADR 0005: untouched.
- Phase 1–4 behavior: untouched (`verification/`, `evaluation/`, `core/` except the new additive `core/generation_artifact.py` and `evaluation/bootstrap.py`).
- Existing Phase 5 scripts (`fs005_phase5_calibration.py`, `fs005_phase5_evaluation.py`, `fs005_phase5_cohort_audit.py`): left as-is; new additive scripts used instead to avoid rewriting working components.
- VLM clients (`vlm/`): untouched; harness uses them via the existing `VLMClient` interface.

---

## 4. Architecture and Data Flow

```
Cohort audit (included_qids.csv, 945 eligible test samples)
        │
        ▼
Generation runner ─────────────────────────────────────┐
  ├─ For each (model, track, sample):                  │
  │     compute deterministic generation_id            │
  │     (condition-independent: condition NOT in hash) │
  │     ├─ valid COMPLETED record exists? → SKIP       │
  │     │     (no duplicate generation; resume-safe)   │
  │     ├─ FAILED record? → RETRY (no overwrite)       │
  │     ├─ corrupt/unparseable? → DETECT + REGENERATE  │
  │     └─ generate N=5 once via VLMClient → persist   │
  │         GenerationRecord (JSON, atomic write)      │
  │                                                    │
  │     Conditions 2 & 3 from the SAME record:         │
  │     ├─ agreement, entropy, confidence (shared)     │
  │     ├─ Condition 2: no abstention gate             │
  │     └─ Condition 3: abstention gate (tau)          │
  │         ← only the abstention decision differs     │
  │                                                    │
  └─ Incremental progress + failure summaries persisted│
         after every sample (Colab-disconnect safe)    │
        │                                              │
        ▼                                              ▼
experiments/runs/fs005_phase5/generations/<model>/<track>/<id>.json
experiments/runs/fs005_phase5/conditions/progress_<model>_track_<track>.json

Calibration split (separate, by image ID):
  SLAKE validate.json (English, pilot-excluded)
    ├─ Track B: GT-region samples only (970) → fit/assessment by image ID
    └─ Track C: all eligible (1,053) → fit/assessment by image ID
      → manifests: fit_qids.csv, fit_image_ids.json, assessment_qids.csv,
         assessment_image_ids.json, split_manifest.json (per model/track)

Pilot cohort (validation-only; does not touch the test cohort):
  SLAKE validate.json (English, FS-004 pilot-excluded)
    ├─ select 10–20 image-disjoint samples (deterministic, seed)
    ├─ exclude pilot image IDs from calibration fit/assessment
    ├─ regenerate calibration split manifests
    └─ manifests: pilot_manifest.json, pilot_qids.csv, pilot_image_ids.json

Persistent storage (Colab):
  All artifacts written to a configurable root (Google Drive:
  MyDrive/MedicalVQA/outputs/fs005_phase5/). `GenerationArtifactStore` verifies
  each write (re-read + re-validate); progress + failure summaries are saved
  after every sample; resume skips valid completed artifacts.

Statistical analysis:
  Image-clustered bootstrap: resample image IDs, preserve all questions per
  image, preserve paired Track B/C + model observations within clusters,
  percentile CIs, seed + replicate count (>= 1,000).
  McNemar: secondary sample-level, NOT image-cluster-adjusted.

Paired cohort validation:
  Rejects mismatched Track B/C paired sets (question IDs and image IDs).
```

### Key design decisions
- **Condition-independent generation identifier:** `compute_generation_id` hashes `(model_name, model_revision, track, question_id, image_id, n_samples, temperature, random_seed)`. The answer condition is deliberately excluded, so Conditions 2 and 3 (and Condition 1 when N matches) share one generation artifact. Condition is never part of generation.
- **Deterministic identifiers:** SHA-256 hex digest over canonical sorted JSON; completed records are recognized reliably across restarts and Colab disconnects.
- **Validation before evaluation:** `validate_generation_record` enforces schema + business rules (sample count = n_samples, contiguous indices, completed⇔error consistency, parseable timestamps, non-negative durations). `GenerationArtifactStore.save` validates before persisting and marks `validation_status = VALID`.
- **Resumability:** progress persisted after every sample; on restart, valid COMPLETED records are skipped (no duplicate generation), FAILED/corrupt records are retried/regenerated without overwriting valid results. No in-memory-only counter.
- **No real inference by default:** the harness runs with `MockVLMClient` unless a real `VLMClient` is supplied; `main()` uses the mock (fixtures only). Real MedGemma/Qwen inference must be launched separately on Colab T4 (see pilot guide).

---

## 5. Implemented versus Deferred Features

| Feature | Status | Notes |
|---------|--------|-------|
| Generation-artifact system (record, ID, store, validation) | Implemented | `core/generation_artifact.py` + tests |
| Deterministic, condition-independent generation IDs | Implemented | SHA-256; condition excluded from hash |
| Persisted, incrementally saved generation records | Implemented | Atomic JSON writes; one file per record |
| Skip valid completed records on restart | Implemented | `is_completed_valid`; tested |
| Retry failed records without overwriting valid results | Implemented | Tested |
| Detect corrupt/incompatible artifacts | Implemented | `load()` raises ValueError; tested |
| Prevent duplicate generation | Implemented | Skip-then-generate logic; tested |
| Interruption/resume (Colab disconnect safe) | Implemented | Per-sample progress persistence; tested |
| Conditions 2 & 3 share identical N=5 responses | Implemented | Same record; only abstention differs; tested |
| Agreement/entropy/confidence computed from shared responses | Implemented | Semantic clustering + confidence gate |
| Calibration fit/assessment split by image ID | Implemented | `fs005_phase5_calibration_split.py` + manifests |
| Track B = 970 GT samples; Track C = 1,053 (no-GT documented) | Implemented | Verified against real SLAKE validate.json |
| Image-level calibration disjointness + manifest consistency | Implemented | Tested |
| Paired cohort consistency validation (864/945/81/864) | Implemented | `fs005_phase5_validate_cohort.py` + tests; rejects mismatches |
| Image-clustered bootstrap (>= 1,000 replicates, seed recorded) | Implemented | `evaluation/bootstrap.py` + tests |
| Paired McNemar (secondary, NOT image-cluster-adjusted) | Implemented | `mcnemar_paired` with explicit note |
| Validation-only pilot cohort selection (10–20 image-disjoint validation samples) | Implemented | `fs005_phase5_pilot_cohort.py` + tests; excludes pilot image IDs from calibration |
| Colab T4 execution entry point (Drive mount, HF Secrets, sequential loading) | Implemented | `colab/fs005_phase5_colab_pilot.py`; all 4 model/track combos; mock dry run by default, real cells gated |
| Persistent Drive storage + verify-after-write | Implemented | Configurable store root; artifacts verified after each write; resume from Drive |
| Real model inference (MedGemma/Qwen on Colab T4) | Deferred | Not executed; requires HF_TOKEN + Colab T4; see pilot guide |
| Real calibration fitting (weights/tau on validation fit subset) | Deferred | Placeholder weights in harness; must be fitted on real validation outputs |
| Fresh test evaluation on 945 samples | Deferred | Not executed |
| Full 40,210-generation experiment | Deferred | Not started; pilot first |

---

## 6. Test Commands and Exact Results

Run from repository root (`D:\Abi\New folder\MedicalVQA`):

### New FS-005 Phase 5 tests
```
.venv\Scripts\python -m pytest tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py -v
```
**Result:** `41 passed in 1.79s`

### Full unit suite
```
.venv\Scripts\python -m pytest tests/ -v
```
**Result:** `450 passed, 4 failed, 1 skipped in 40.35s`

The 4 failures are pre-existing environment issues unrelated to this change:
- `tests/unit/test_environment.py::test_python_version` (python version assertion)
- `tests/unit/test_medsam_integration.py::TestMedSAMRealInference::test_real_single_region` (MedSAM/torch triton namespace)
- `tests/unit/test_medsam_integration.py::TestMedSAMRealInference::test_real_multiple_regions_order` (MedSAM/torch triton namespace)
- `tests/unit/test_medsam_segmenter.py::TestMedSAMMissingDependencies::test_b02_import_error_when_segment_anything_missing` (MedSAM/torch triton namespace)

None touch FS-005 Phase 5 execution code. All 45 new Phase 5 tests pass.

### Pilot cohort selection (read-only on data; writes pilot + calibration manifests)
```
python scripts/fs005_phase5_pilot_cohort.py
```
**Result:** 15 pilot samples selected (15 distinct images; 7 with GT regions, 8 without); FS-004 pilot material excluded; pilot/test cohort separation verified; pilot/calibration image-ID disjointness verified; calibration split manifests regenerated excluding the 15 pilot image IDs (Track B/C: fit=719 samples/65 images, assessment=172 samples/16 images, `image_disjoint=True` for all 4 model/track combinations). Test cohort untouched.

### Calibration split script (read-only on data; writes split manifests)
```
.venv\Scripts\python scripts/fs005_phase5_calibration_split.py
```
**Result:** Track B: 970 → fit 788 (70 images) / assessment 182 (18 images); Track C: 1,053 → fit 850 (77 images) / assessment 203 (19 images); `image_disjoint=True` for all 4 model/track combinations. Manifests written under `experiments/runs/fs005_phase5/calibration/splits/`.

### Paired cohort validation script (read-only)
```
.venv\Scripts\python scripts/fs005_phase5_validate_cohort.py
```
**Result:** `COHORT VALIDATION PASSED: paired Track B/C cohort is consistent.` — eligible_total=945, track_b_gt_annotated=864, track_c_total=945, track_c_only_no_gt=81, paired_track_b_c=864.

### Ruff lint
```
.venv\Scripts\python -m ruff check core/generation_artifact.py scripts/fs005_phase5_generation.py scripts/fs005_phase5_calibration_split.py scripts/fs005_phase5_pilot_cohort.py colab/fs005_phase5_colab_pilot.py evaluation/bootstrap.py tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py
```
**Result:** `All checks passed!` (0 errors on all changed Python files.)

---

## 7. Corrected Workload Arithmetic (40,210 unique generations)

The previous grand total of 30,210 was incorrect arithmetic (19,980 + 20,230 = 40,210, not 30,210). Corrected everywhere.

### Test generations (unique, Conditions 2 & 3 share N=5 responses)

| Configuration | Calculation | Unique generations |
|---|---|---:|
| MedGemma Track B, N=5 | 864 × 5 | 4,320 |
| MedGemma Track C, N=5 | 945 × 5 | 4,725 |
| MedGemma Track C, N=1 | 945 × 1 | 945 |
| Qwen Track B, N=5 | 864 × 5 | 4,320 |
| Qwen Track C, N=5 | 945 × 5 | 4,725 |
| Qwen Track C, N=1 | 945 × 1 | 945 |
| **Test total** | | **19,980** |

Conditions 2 and 3 reuse the same N=5 generated responses; not counted twice.

### Calibration generations

| Configuration | Calculation | Generations |
|---|---|---:|
| MedGemma Track B | 970 × 5 | 4,850 |
| Qwen Track B | 970 × 5 | 4,850 |
| MedGemma Track C | 1,053 × 5 | 5,265 |
| Qwen Track C | 1,053 × 5 | 5,265 |
| **Calibration total** | | **20,230** |

**Correct grand total: 40,210 unique generations.**

### Runtime estimate
The previous 20–32 hour / ~22–40 hour estimate is **NOT verified** and was removed from the plan and final-review reports. A runtime estimate must be based on measured pilot throughput (Colab T4 feasibility pilot, see `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`). The experiment requires up to 40,210 unique generations and exceeds a single Colab session (~12 hours); multiple sessions with checkpointing are required.

---

## 8. Remaining Risks and Blockers

| Risk | Status / Mitigation |
|------|---------------------|
| Real model inference not executed | Deferred — requires HF_TOKEN (MedGemma is gated) + Colab T4; see pilot guide. Not started per task restriction. |
| Real calibration weights/tau not fitted | Placeholder weights (w1=w2=w3=1/3, tau=0.5) in harness; must be fitted on SLAKE validation fit subset and assessed on held-out assessment subset before the full experiment. |
| MedGemma-4B is a gated HF model | Requires license acceptance + HF token via Colab Secrets; never hardcoded or printed. |
| Local RTX 2050 4GB VRAM insufficient | Inference isolated to Colab T4; sequential model loading; never resident simultaneously. |
| 40,210 generations exceed one Colab session | Checkpoint after each model/condition; resume from persisted artifacts. |
| Full experiment not started | Awaiting approval — per task restriction, pilot only. |

---

## 9. Pilot-Readiness Decision

**PILOT-READY: YES (for a 10–20 sample Colab T4 feasibility pilot), with the following conditions:**

1. The execution harness, generation-artifact system (with verify-after-write), resume/duplicate prevention, Conditions 2/3 reuse, calibration split (with pilot image-ID exclusion), paired cohort validation, image-clustered bootstrap, validation-only pilot cohort selection, Colab T4 entry point, and persistent Drive storage are implemented and tested (41 new tests passing; ruff clean).
2. Workload arithmetic corrected to 40,210 unique generations everywhere; the 20–32 hour estimate is explicitly NOT verified.
3. A real pilot requires: HF_TOKEN for MedGemma-4B (Colab Secrets), Colab Tesla T4 access, model downloads (~7–15 GB), sequential model loading, and persistent Google Drive storage (MyDrive/MedicalVQA/outputs/fs005_phase5/). Credentials must never be printed, exposed, hardcoded, or saved in project artifacts.
4. The pilot is a feasibility test only — not statistically meaningful; no performance or clinical-superiority claims.
5. Real calibration fitting, fresh test evaluation, and the full 40,210-generation experiment remain deferred and require separate approval.

**Awaiting approval to run the Colab T4 feasibility pilot. Do not run the pilot or full experiment without approval.**

---

## 10. Traceability

- Feature Specification: FS-005 (Phase 5 execution harness)
- ADR 0005: Confidence gate (`c = w1·a + w2·(1−Hn) + w3·ρ`), N=5 sampling, configurable weights/threshold
- ADR 0004: Negative/Absence claims (Option 3)
- FS-001 through FS-004: unmodified
