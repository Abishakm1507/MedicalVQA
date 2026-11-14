# FS-005 Phase 5 — Pilot Safety Review

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_PILOT_SAFETY_REVIEW.md`  
**Status:** Safety review complete — pilot provisionally cleared for approval; **not executed**  
**Scope:** Pre-execution safety review of the Colab T4 feasibility pilot: data isolation,
Colab execution instructions, persistent checkpoint storage, credential handling, and
success criteria. No inference, calibration, or test evaluation performed.

---

## 1. Review objective

Confirm that the Colab feasibility-pilot workflow is safe to authorize:

- Pilot data comes **only** from the SLAKE validation split; the official test cohort is
  untouched.
- Pilot image IDs are excluded from calibration fit AND assessment; manifests regenerated.
- Colab execution uses persistent Google Drive storage (not ephemeral `/content`).
- Credentials (HF_TOKEN) are never printed, exposed, hardcoded, or saved.
- Real-model execution cells are present but **gated off** (`RUN_REAL_INFERENCE = False`);
  they cannot run without explicitly flipping the flag.
- Success criteria are feasibility-only; no accuracy or clinical claims.

No model inference, calibration fitting, or test evaluation was performed as part of this
review.

---

## 2. Data isolation findings

### 2.1 Validation-only pilot cohort

- Source: SLAKE English **validation** split (1,053 samples).
- Selection: 15 image-disjoint samples (within the required 10–20 range), 15 distinct
  images, deterministic (seed 20261114).
- Track B-eligible (with GT regions): 7; Track C-only (without GT regions): 8.
- FS-004 pilot QIDs (10) and image IDs (9) explicitly excluded from selection.
- Artifacts: `experiments/runs/fs005_phase5/pilot_cohort/{pilot_manifest.json,
  pilot_qids.csv, pilot_image_ids.json}`.

### 2.2 Calibration isolation

- The 15 pilot image IDs are excluded from calibration fitting AND assessment.
- Calibration split manifests regenerated via
  `scripts/fs005_phase5_calibration_split.py --exclude-image-ids <pilot IDs>`.
- Verified: no pilot image ID appears in any fit or assessment manifest for any of the
  4 model/track combinations (`image_disjoint=True`).
- Fit/assessment after exclusion: Track B/C fit=719 samples (65 images),
  assessment=172 samples (16 images).

### 2.3 Test cohort preservation

- Official test cohort (945 eligible samples) is **never read** by the pilot cohort
  script except for a separation assertion.
- Verified: pilot QIDs ∩ test QIDs = ∅.
- Test cohort files (`data/SLAKE/test.json`) are unmodified.

**Safety finding:** PASS — pilot data is validation-only, image-disjoint, calibration-
isolated, and test-cohort-preserving.

---

## 3. Colab execution instructions findings

### 3.1 Entry point

- Executable entry point: `colab/fs005_phase5_colab_pilot.py` (Colab-compatible,
  `# %%` cell markers, no Windows path separators).
- Real-model loading code is present and runs **all four model/track combinations**
  (MedGemma×B, MedGemma×C, Qwen×B, Qwen×C) sequentially, but is gated behind
  `RUN_REAL_INFERENCE = False`.
- Default mode is a mock dry run (`MockVLMClient`) exercising both Track B and
  Track C — no model weights downloaded, no inference performed.

### 3.2 Model loading

- Models loaded sequentially; `unload_client` clears GPU cache between models.
- Each model loaded once, both tracks run, then unloaded; models never resident in
  VRAM simultaneously (peak ~6–7 GB per model, within T4's 15.3 GB).
- MedGemma-4B uses `torch.bfloat16` + 4-bit on CUDA; Qwen uses `torch.float16` + 4-bit.

### 3.3 Colab Secrets token mechanism

- `configure_hf_auth()` reads `HF_TOKEN` via `google.colab.userdata.get("HF_TOKEN")`
  (Colab Secrets do **not** auto-populate `os.environ`), falling back to
  `os.environ.get("HF_TOKEN")` only for non-Colab/local testing.
- Token is never printed, echoed, persisted, or committed;
  `huggingface_hub.login(token=..., add_to_git_credential=False)`.

### 3.3 Conditions 2 & 3 response reuse

- `generation_id` = SHA-256 over `(model_name, model_revision, track, question_id,
  image_id, n_samples, temperature, random_seed)` — condition is **excluded** from the
  hash.
- Conditions 2 and 3 resolve to the same `GenerationRecord`; only the abstention
  decision differs. Verified by `tests/unit/test_condition_reuse.py`.

### 3.4 Resume / duplicate prevention

- Progress + failure summaries saved after **every sample** to Drive.
- Valid COMPLETED artifacts skipped on restart (`is_completed_valid`); FAILED/corrupt
  artifacts retried/regenerated without overwriting valid results.
- Verified by `tests/unit/test_resume_execution.py` and `tests/unit/test_pilot_cohort.py`.

**Safety finding:** PASS — Colab entry point is Colab-compatible, real inference gated
off by default, models loaded sequentially, response reuse enforced, resume tested.

---

## 4. Persistent checkpoint storage findings

### 4.1 Storage location

- All artifacts written to a configurable `GenerationArtifactStore` root.
- Colab target: Google Drive `MyDrive/MedicalVQA/outputs/fs005_phase5/`.
- `/content` is explicitly **not** treated as persistent.

### 4.2 Write verification

- `GenerationArtifactStore.save()` performs verify-after-write: re-reads the persisted
  JSON and re-validates it (`validate_generation_record`), raising `ValueError` on any
  corruption or incompatibility.
- Protects against truncated/unreadable writes on Drive before resume trusts a write.

### 4.3 Resume after runtime restart

- Re-mount Drive, re-run with same inputs; completed artifacts intact and valid;
  skipped (no duplicate generation).
- Verified by `tests/unit/test_pilot_cohort.py::test_resume_from_persistent_artifact_directory`.

**Safety finding:** PASS — persistent Drive storage with verify-after-write and tested
resume; no reliance on ephemeral `/content`.

---

## 5. Credential handling findings

- `HF_TOKEN` read via `google.colab.userdata.get("HF_TOKEN")` (Colab Secrets),
  falling back to `os.environ.get("HF_TOKEN")` only for non-Colab/local testing.
- Entry point never prints, echoes, or writes the token.
- `huggingface_hub.login(token=..., add_to_git_credential=False)` — no persistence to
  Git credential store.
- Guide explicitly warns: never print/save/hardcode credentials.

**Safety finding:** PASS — no credentials exposed.

---

## 6. Success criteria (feasibility only)

A successful pilot must demonstrate:
1. Validation-only pilot cohort isolation.
2. No pilot/calibration image-ID overlap.
3. All four model/track combinations run sequentially on T4 (MedGemma×B,
   MedGemma×C, Qwen×B, Qwen×C); models never resident simultaneously.
4. Both tracks execute for each model.
5. N=5 outputs persist correctly (verified after write).
6. Conditions 2 and 3 reuse identical generation artifacts.
7. Drive artifacts survive a simulated runtime restart.
8. Resume skips completed samples.
9. No credentials exposed.
10. No test-set samples consumed.

No accuracy or clinical-performance claims are permitted.

---

## 7. Restrictions honored

- No model inference executed.
- No calibration fitting executed.
- No test evaluation executed.
- Full 40,210-generation experiment not started.
- No secrets accessed or printed.
- FS-001–FS-004 unmodified.
- ADR 0004 and ADR 0005 unmodified.
- Phase 1–4 behavior unmodified.
- Stage 6 not begun.

---

## 8. Pilot-readiness decision

**CLEARED FOR APPROVAL (provisional).**

The Colab T4 feasibility pilot is safe to run subject to explicit approval:

1. Flip `RUN_REAL_INFERENCE = True` in `colab/fs005_phase5_colab_pilot.py` only after
   approval is granted.
2. Provide `HF_TOKEN` via Colab Secrets (MedGemma-4B is gated).
3. Use a Tesla T4 runtime; mount Google Drive; write artifacts to
   `MyDrive/MedicalVQA/outputs/fs005_phase5/`.
4. Run the mock dry run first (`RUN_REAL_INFERENCE = False`) to validate the end-to-end
   harness on Drive before any real inference.
5. Confirm all 10 success criteria; none are performance claims.

**Do not run the pilot or the full 40,210-generation experiment without explicit
approval.**
