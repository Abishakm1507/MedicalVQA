# FS-005 Phase 5 — Final Pilot Execution Consistency Audit

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_FINAL_PILOT_EXECUTION_AUDIT.md`  
**Status:** Audit complete — implementation consistent; pilot cleared for approval (not executed)  
**Scope:** Read-only-first audit of the Colab T4 feasibility-pilot implementation; inconsistencies corrected. No inference, calibration, or test evaluation executed.

---

## 1. Audit summary

The Phase 5 pilot implementation was audited for consistency across the pilot cohort,
calibration split, Colab entry point, Colab guide, model/track matrix, secret handling,
and test-cohort isolation. Inconsistencies found and corrected:

| # | Inconsistency | Correction |
|---|---------------|------------|
| 1 | `configure_hf_auth()` read `HF_TOKEN` from `os.environ`, but Colab Secrets do NOT auto-populate `os.environ`. | Now reads via `google.colab.userdata.get("HF_TOKEN")`, falling back to `os.environ` only for local/non-Colab testing. Token never printed or persisted. |
| 2 | Real-inference path ran only MedGemma×B and Qwen×C (2 of 4 combinations). | Real path now runs all 4 combinations sequentially: MedGemma×B, MedGemma×C, Qwen×B, Qwen×C; each model loaded once, both tracks run, then unloaded. |
| 3 | Track B `region_ids` hardcoded to `[]`; GT region identifiers not loaded. | `build_input_samples(track=...)` now loads GT region IDs from `detection.json` for Track B. |
| 4 | Mock dry run exercised only Track C. | Mock dry run now exercises both Track B and Track C. |

Real inference remains disabled by default (`RUN_REAL_INFERENCE = False`). No real inference,
calibration fitting, or test evaluation was executed.

---

## 2. Calibration cohort reconciliation

Calibration is computed read-only on the SLAKE English **validation** split (1,053 samples).
The official test cohort is never used for calibration.

The 15 validation-only pilot image IDs are excluded from calibration fitting AND
assessment, and split manifests are regenerated so no pilot image ID appears in either
subset.

### Reconciliation table (all four model/track combinations)

| Metric | Value |
|--------|-------|
| Original validation cohort (English) | 1,053 samples |
| Excluded because FS-004 pilot material | 0 (FS-004 pilot lives on test split) |
| Remaining after FS-004 exclusion | 1,053 samples |
| Excluded because image ID in pilot set | 162 samples (on 15 pilot images) |
| **Remaining for calibration** | **891 samples** |
| Track B (with GT regions) | 891 samples |
| Track B excluded (no GT regions) | 0 samples |
| Track C (all, ungrounded) | 891 samples |
| Fit samples (per model/track) | 719 |
| Fit images (per model/track) | 65 |
| Assessment samples (per model/track) | 172 |
| Assessment images (per model/track) | 16 |
| Fit ∩ Assessment images | 0 (image-disjoint) |
| Pilot image overlap with any split | 0 (empty) |

### Why Track B and Track C have these counts

- **Track C = 891:** Track C is ungrounded, so every remaining validation sample
  (after FS-004 and pilot-image exclusion) is eligible. Track C = remaining = 891.
- **Track B = 891 with GT, 0 without GT:** Track B requires non-empty GT regions
  (`detection.json`). The data property is that **all 83 without-GT validation samples
  happen to reside on the 15 pilot images**, which are excluded from calibration.
  Therefore, after excluding the 15 pilot images, every remaining sample has GT
  regions: Track B = 891 with GT, 0 excluded for lacking GT. This is a
  data-driven property, not a forced count; it was verified against the actual
  SLAKE validation split and detection files.
- **Identical across models:** The split is model-independent (no inference), so all
  four model/track combinations report identical counts. Fit/assessment use an
  80/20 image-level split (seed 42, fit_fraction 0.8).

Pilot image IDs excluded from calibration: `{0, 34, 84, 92, 207, 213, 245, 257,
291, 348, 391, 476, 482, 510, 609}` (15 distinct images; 7 with GT regions,
8 without).

### Manifests

Written under `experiments/runs/fs005_phase5/calibration/splits/<model>/<track>/`:
- `fit_qids.csv`, `fit_image_ids.json`, `fit_manifest.json`
- `assessment_qids.csv`, `assessment_image_ids.json`, `assessment_manifest.json`
- `split_manifest.json` (records `image_disjoint=True`, `test_data_used=False`)

---

## 3. Colab real-inference invocation

The executable entry point is `colab/fs005_phase5_colab_pilot.py` (Colab-compatible,
`# %%` cell markers, no Windows path separators).

### Default: mock-mode dry run (safe to execute now)

`RUN_REAL_INFERENCE = False`. `main()` calls `install_dependencies()`, then instantiates
`MockVLMClient` and runs `GenerationRunner` for **both Track B and Track C** on the
15-sample validation-only pilot cohort, persisting artifacts to the configured output
directory. No model weights are downloaded; no inference is performed.

### Enabling real inference (after explicit approval)

1. Set `RUN_REAL_INFERENCE = True` in `colab/fs005_phase5_colab_pilot.py`.
2. Add `HF_TOKEN` to Colab Secrets (key: `HF_TOKEN`).
3. Re-run the notebook; `main()` executes the real path.

### Real path (all four combinations, sequential loading)

`main()` authenticates via `configure_hf_auth()`, then iterates the model/track
matrix:

| Order | Model | Track | Loading |
|-------|-------|-------|---------|
| 1 | MedGemma-4B | B | MedGemma loaded once on CUDA (4-bit) |
| 2 | MedGemma-4B | C | (same MedGemma load) |
| — | MedGemma-4B | — | **unloaded**, GPU cache cleared |
| 3 | Qwen2.5-VL-3B | B | Qwen loaded once on CUDA (4-bit) |
| 4 | Qwen2.5-VL-3B | C | (same Qwen load) |
| — | Qwen2.5-VL-3B | — | **unloaded**, GPU cache cleared |

Models are never resident in GPU VRAM simultaneously.

### Conditions 2 and 3 response reuse

`generation_id` = SHA-256 over `(model_name, model_revision, track, question_id,
image_id, n_samples, temperature, random_seed)`. The answer condition is excluded
from the hash, so Conditions 2 and 3 resolve to the same `GenerationRecord`; only
the abstention decision differs. Verified by `tests/unit/test_condition_reuse.py`.

### Resume from Google Drive after runtime restart

- Artifacts written to `MyDrive/MedicalQA/outputs/fs005_phase5/generations/...`
  and `.../conditions/progress_<model>_track_<track>.json`.
- Each write is verified after write (re-read + re-validated).
- On restart: re-mount Drive, re-run with same inputs; valid COMPLETED artifacts
  are skipped (no duplicate generation); FAILED/corrupt artifacts retried/regenerated.

### Exact cells (per `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`)

- **Cell A** — Repository preparation (git clone, `sys.path`).
- **Cell B** — Dependency verification (`pip install -r requirements-dev.txt`).
- **Cell C** — Dataset preparation (verify `data/SLAKE/` present).
- **Cell D** — Colab Secrets authentication (`google.colab.userdata.get("HF_TOKEN")`).
- **Cell E** — Mock-mode dry run (`raise SystemExit(main())`; `RUN_REAL_INFERENCE=False`).
- **Cell F** — Real-model initialization note (gated; not executed).
- **Cell G** — Pilot execution (flip `RUN_REAL_INFERENCE=True` after approval).
- **Cell H** — Resume after Colab disconnect.

---

## 4. Colab Secrets token-loading mechanism

Colab Secrets do NOT automatically populate `os.environ`. The entry point's
`configure_hf_auth()`:

1. Tries `from google.colab import userdata; userdata.get("HF_TOKEN")`.
2. Falls back to `os.environ.get("HF_TOKEN")` only for non-Colab/local testing.
3. Raises `RuntimeError` if neither source has the token.
4. Calls `huggingface_hub.login(token=..., add_to_git_credential=False)` — the
   token is **not** written to the Git credential store or printed.

The guide (Section 4, Cell D) documents this mechanism explicitly. No credentials
are printed, persisted, hardcoded, or committed.

---

## 5. Model/track matrix (explicit)

The pilot covers **all four** combinations, matching the full experiment plan
(MedGemma and Qwen each run Track B and Track C):

| Combination | Model | Track | Included? |
|-------------|-------|-------|-----------|
| 1 | MedGemma-4B | B | Yes |
| 2 | MedGemma-4B | C | Yes |
| 3 | Qwen2.5-VL-3B | B | Yes |
| 4 | Qwen2.5-VL-3B | C | Yes |

No combination is intentionally excluded. The guide, entry point, calibration
configuration, and reports all agree on this 4-combo matrix. Calibration is
performed per model/track (fit/assessment split by image ID) for all four
combinations.

---

## 6. Test-cohort wording

The official test cohort (`data/SLAKE/test.json`, 1,061 English samples, 945
eligible) is **read solely for a read-only separation assertion**:
`pilot_qids ∩ test_qids = ∅`. No test samples are selected, used for inference,
used for calibration, or modified during the pilot. This is asserted in
`tests/unit/test_pilot_cohort.py::test_test_cohort_read_only_separation_assertion`.

The pilot uses the SLAKE English **validation** split only.

---

## 7. Verification results

### Tests executed

Command:
```
.venv\Scripts\python -m pytest tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py
```

Result: **45 passed in 2.24s**

New Phase 5 tests (45 total across the suite):
- `test_generation_artifact.py` — artifact serialization/validation, deterministic IDs, store skip/retry/corrupt detection.
- `test_condition_reuse.py` — Conditions 2 & 3 consume identical source artifacts.
- `test_resume_execution.py` — interruption/resume, duplicate prevention, corrupt-artifact detection and regeneration.
- `test_calibration_split.py` — image-level disjointness, all-questions-per-image preservation, manifest consistency, pilot exclusion.
- `test_paired_cohort.py` — paired Track B/C cohort consistency (864/945/81/864).
- `test_cluster_bootstrap.py` — cluster-bootstrap behavior, seed/replicate recording, ≥ 1,000 replicates, McNemar not cluster-adjusted.
- `test_pilot_cohort.py` — pilot/calibration image-ID disjointness, pilot/test separation, persistent output path config, resume from persistent artifact directory, calibration reconciliation relationships, manifest pilot overlap, read-only test-cohort assertion, `RUN_REAL_INFERENCE=False` flag.

### Ruff lint

Command:
```
.venv\Scripts\python -m ruff check core/generation_artifact.py scripts/fs005_phase5_generation.py scripts/fs005_phase5_calibration_split.py scripts/fs005_phase5_pilot_cohort.py colab/fs005_phase5_colab_pilot.py evaluation/bootstrap.py tests/unit/test_generation_artifact.py tests/unit/test_condition_reuse.py tests/unit/test_resume_execution.py tests/unit/test_calibration_split.py tests/unit/test_paired_cohort.py tests/unit/test_cluster_bootstrap.py tests/unit/test_pilot_cohort.py
```

Result: **All checks passed!** (0 errors on all changed Python files.)

### Full unit suite (context)

` .venv\Scripts\python -m pytest tests/ -v` → **450 passed, 4 failed, 1 skipped**.

The 4 failures are pre-existing, environment-related, and unrelated to this change:
- `tests/unit/test_environment.py::test_python_version` (python version assertion)
- `tests/unit/test_medsam_integration.py::TestMedSAMRealInference::test_real_single_region` (MedSAM/torch triton namespace)
- `tests/unit/test_medsam_integration.py::TestMedSAMRealInference::test_real_multiple_regions_order` (MedSAM/torch triton namespace)
- `tests/unit/test_medsam_segmenter.py::TestMedSAMMissingDependencies::test_b02_import_error_when_segment_anything_missing` (MedSAM/torch triton namespace)

None touch FS-005 Phase 5 execution code. They are preserved (not excluded).

### Defaults verified

- `RUN_REAL_INFERENCE = False` in `colab/fs005_phase5_colab_pilot.py` (asserted by
  `test_real_inference_flag_disabled`).
- Mock-mode dry run is the default execution path.
- No real inference, calibration fitting, or test evaluation was executed during
  this audit.

---

## 8. Changed files

### Created
- `colab/fs005_phase5_colab_pilot.py` — Colab T4 execution entry point (Drive mount, HF Secrets via `userdata.get`, sequential MedGemma/Qwen loading, all 4 model/track combos, Conditions 2/3 reuse, Drive resume).
- `scripts/fs005_phase5_pilot_cohort.py` — validation-only pilot cohort selection + calibration manifest regeneration excluding pilot image IDs.
- `tests/unit/test_pilot_cohort.py` — pilot safety tests (10 tests).
- `docs/reports/FS-005_PHASE_5_FINAL_PILOT_EXECUTION_AUDIT.md` (this report).

### Modified
- `core/generation_artifact.py` — verify-after-write (re-read + re-validate persisted artifact).
- `scripts/fs005_phase5_calibration_split.py` — `--exclude-image-ids` and `--output-dir` CLI options.
- `scripts/fs005_phase5_generation.py` — (unchanged in this audit; prior Phase 5 work).
- `colab/fs005_phase5_colab_pilot.py` — Secrets mechanism, 4-combo matrix, Track B GT region IDs, mock both tracks.
- `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md` — updated cells, Secrets mechanism, 4-combo matrix, success criteria.
- `docs/reports/FS-005_PHASE_5_PILOT_SAFETY_REVIEW.md` — updated findings.
- `docs/reports/FS-005_PHASE_5_EXECUTION_HARNESS_IMPLEMENTATION_REPORT.md` — updated test counts (45 passed), file lists.

### Generated artifacts (read-only on data; not committed to experiment)
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_manifest.json` (15 samples, 15 images)
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_qids.csv`
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_image_ids.json`
- `experiments/runs/fs005_phase5/calibration/splits/<model>/<track>/` (regenerated excluding pilot image IDs)

---

## 9. Unresolved limitations

- Real model inference (MedGemma-4B / Qwen2.5-VL-3B on Colab T4) has **not** been
  executed; `RUN_REAL_INFERENCE = False`. Awaiting explicit approval to run the pilot.
- Real calibration fitting (weights/tau on the validation fit subset) is **deferred**;
  placeholder weights are used in the harness.
- Set-of-Mark image marking (drawing GT boxes on `source.jpg` for Track B) is not
  performed by the entry point; region IDs are loaded from `detection.json` and made
  available in the sample dict, but actual pixel-level marking is orthogonal to this
  infrastructure audit.
- The Track B = 891 / excluded-no-GT = 0 property is data-driven (all without-GT
  validation samples happen to be on the 15 pilot images); if the validation split
  or pilot selection changes, this relationship must be re-verified.
- HF_TOKEN for MedGemma-4B (gated model) must be provided via Colab Secrets at pilot
  runtime; never stored in the repository.

---

## 10. Pilot-readiness decision

**CLEARED FOR APPROVAL (provisional).**

The Colab T4 feasibility pilot is safe to run subject to explicit approval:

1. Flip `RUN_REAL_INFERENCE = True` in `colab/fs005_phase5_colab_pilot.py` only after
   approval is granted.
2. Provide `HF_TOKEN` via Colab Secrets (MedGemma-4B is gated).
3. Use a Tesla T4 runtime; mount Google Drive; write artifacts to
   `MyDrive/MedicalVQA/outputs/fs005_phase5/`.
4. Run the mock dry run first (`RUN_REAL_INFERENCE = False`) to validate the
   end-to-end harness on Drive before any real inference.
5. Confirm all 10 success criteria (see pilot guide); none are performance claims.

**Do not run the pilot or the full 40,210-generation experiment without explicit
approval.**
