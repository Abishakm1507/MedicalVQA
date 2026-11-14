# FS-005 Phase 5 — Final Pilot Readiness Fix

**Status:** Implemented and locally verified with mocks  
**Scope:** Track B oracle overlays, Track C isolation, and batch progress/resume  
**Real inference:** Not run

## Files inspected

- `colab/fs005_phase5_colab_pilot.py`
- `scripts/fs005_phase5_generation.py`
- `core/generation_artifact.py`
- `vlm/base.py`
- `core/schemas.py`
- `localization/rle.py`
- `requirements-dev.txt`
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_manifest.json`
- `data/SLAKE/imgs/xmlab6/detection.json` and `source.jpg`
- `data/SLAKE/imgs/xmlab130/detection.json` and `source.jpg`
- `docs/reports/FS-005_PHASE_5_SLAKE_ANNOTATION_AUDIT.md`
- `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`
- `verification/claim_filter.py` and `verification/evidence_verifier.py`
- `tests/unit/test_condition_reuse.py`, `tests/unit/test_resume_execution.py`,
  and `tests/unit/test_pilot_cohort.py`

## Files changed or created

- Changed `colab/fs005_phase5_colab_pilot.py` to prepare separate Track B and
  Track C inputs and protect Colab's installed PyTorch/CUDA packages during
  dependency setup.
- Changed `core/generation_artifact.py` to persist `grounding_mode` and include
  the Track B overlay version in its generation identity. Existing old
  Track B artifacts are left in place but are not reused for the corrected
  input.
- Changed `scripts/fs005_phase5_generation.py` to validate response region IDs
  against the current sample, fingerprint/reset per-batch progress, and keep
  the batch counters and question-ID lists consistent.
- Created `scripts/fs005_phase5_ground_truth.py` for strict annotation parsing
  and PIL overlays.
- Created `tests/unit/test_fs005_phase5_ground_truth.py` for coordinate,
  overlay, track-isolation, artifact, and cohort-progress regressions.
- Updated `tests/unit/test_resume_execution.py` with corruption-retry counter
  assertions.
- Corrected the coordinate schema in
  `docs/reports/FS-005_PHASE_5_SLAKE_ANNOTATION_AUDIT.md`.
- Updated generation identity and grounding-mode notes in
  `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`.
- Created this report.

The worktree already contained edits in the generation runner and
`tests/unit/test_resume_execution.py`, plus changes in
`tests/unit/test_pilot_cohort.py` and untracked mock/run outputs. Existing
changes were preserved; generated outputs and unrelated files were not
deleted or cleaned up.

## Annotation convention and overlay

SLAKE `detection.json` boxes use source-pixel **XYWH** coordinates:
`[x, y, width, height]`. The dataset image `xmlab6/source.jpg` is 256×256.
Its supplied annotation
`[42, 105, 21, 41]` converts to XYXY `[42, 105, 63, 146]`.

The independent `xmlab130` example is also consistent: its 1024×1024 image
has `[703, 616, 218, 249]`, which converts to `[703, 616, 921, 865]`.
That annotation cannot represent XYXY because its third coordinate is less
than its first. The prior audit's `[ymin, xmin, ymax, xmax]` description was
incorrect and has been corrected.

`build_ground_truth_overlay` reads the annotation JSON, validates its shape,
finite numeric coordinates, positive extents, and image bounds, then converts
each XYWH box to XYXY. It assigns `R1`, `R2`, etc. in annotation order and
draws red box outlines plus visible yellow-backed, black-text labels onto an
RGB copy. Invalid or missing annotations raise an explicit error; boxes are
not clipped and no missing regions are fabricated.

## Track and artifact separation

- **Track B:** Empty/missing region lists are excluded. The real pilot manifest
  produced 7 inputs (QIDs `9841, 9957, 10262, 10415, 10603, 10622, 10819`),
  each with its marked image and generated `R#` IDs. The runner passes the
  marked image to the existing `sample_n(marked_image, question, n,
  temperature)` interface. Returned claim IDs outside that sample's visible
  region-ID set become `NONE`; no claim is marked grounded simply because the
  sample is Track B.
- **Track C:** All 15 pilot inputs retain their original unmarked image, have
  an empty region-ID list, and receive no ground-truth box metadata.
- Generation records persist `track` and `grounding_mode`
  (`oracle_gt_xywh_overlay_v1` or `original_unmarked_image`). The Track B
  input-version component changes its generation ID, so valid artifacts made
  before overlays are not mistaken for generations from the corrected input.
  Those older files are preserved. Conditions 2 and 3 still evaluate the same
  record; they do not call generation separately.

## Progress, cohort fingerprint, and resume

The runner computes a SHA-256 cohort fingerprint from the sorted question IDs,
model name, and track. Progress with a different fingerprint/run identity is
warned about and safely reinitialized without deleting generation artifacts.
Each batch resets its counters and batch-specific IDs, sets `n_expected` to
that batch's sample count, and rebuilds completed IDs from both new completions
and valid artifacts skipped during resume. `n_retried` includes repaired
failures/corruption; `n_corrupt` records artifacts found corrupt during the
batch. Progress is written atomically after each sample.

The regression tests cover a 15-QID Track B batch followed by a 7-QID batch:
the second progress file contains only those 7 QIDs, reports 7 expected and 7
skipped, and keeps all 15 generation artifacts. Synthetic pilot execution also
asserts exactly 7 Track B and 15 Track C progress QIDs. Resume skips valid
existing artifacts without regenerating them.

## Colab pilot path

- `RUN_REAL_INFERENCE = False` remains the default.
- Mock mode loops over both tracks; synthetic tests exercise both input paths
  and the generation runner.
- The real path uses the same corrected `build_input_samples` function.
- MedGemma and Qwen are loaded in separate sequential loops, with the first
  client unloaded before the second is loaded.
- The runner receives the output directory under
  `MyDrive/MedicalVQA/outputs/fs005_phase5` in Colab, so generations and
  progress are persistent and resumable there.
- HF authentication continues to read the token from Colab Secrets and does
  not print or save it.
- Dependency installation rejects PyTorch, Triton, NVIDIA, CUDA, or nested
  requirements directives in `requirements-dev.txt` before invoking pip; the
  current manifest does not install or downgrade Colab's PyTorch/CUDA stack.

## Verification

Focused command:

```text
python -m pytest tests/unit/test_fs005_phase5_ground_truth.py tests/unit/test_resume_execution.py tests/unit/test_condition_reuse.py tests/unit/test_pilot_cohort.py tests/unit/test_generation_artifact.py tests/unit/test_calibration_split.py
55 passed
```

The output included the existing `pytest-asyncio` deprecation warning because
`asyncio_default_fixture_loop_scope` is unset; it was not a test failure.

Ruff on all changed Python files: **passed**.

Repository-wide `ruff check . --output-format concise`: **42 findings in
untouched files** (39 marked fixable; one unsafe fix available). Findings:

| File | Existing findings |
|---|---|
| `experiments/runs/medsam_verification/verify_medsam.py` | `I001` at 24, 69; `F541` at 222, 276, 412 |
| `inspect_slake.py` | `F541` at 19; `C401` at 23 |
| `scripts/fs005_phase5_cohort_detail.py` | `UP015` at 38; `W292` at 79 |
| `test_medsam_inference.py` | `I001` at 2, 5; `F401` at 10, 11; `W293` at 64, 69; `F541` at 72 |
| `test_medsam_load.py` | `I001` at 1, 3 |
| `test_ollama_api.py` | `I001` at 2, 5, 43; `E402` at 43; `W293` at 73, 82, 109, 113; `F541` at 74, 110 |
| `test_ollama_client.py` | `I001` at 2, 5; `F541` at 25, 39 |
| `test_slake_actual.py` | `I001` at 2, 5, 53; `E402` at 53 |
| `tests/unit/test_ollama_client.py` | `I001` at 3; `F401` at 4 |
| `tests/unit/test_slake_loader_actual.py` | `F401` at 6, 8, 68; `I001` at 68 |

These unrelated lint findings were not changed. `git diff --check` reported no
whitespace errors.

## Remaining limitations

This verifies the pilot input and resume mechanics with synthetic/mock calls
and actual local dataset image preparation only. It does not establish model
performance, grounding quality, or clinical validity. Confidence calibration
and the full experiment remain separate work.

**No real model inference, calibration fitting, or official test-set
evaluation was run.** The 40,210-generation experiment was not run, and real
inference remains disabled.
