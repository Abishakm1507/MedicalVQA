# FS-005 Phase 5 — Colab T4 Feasibility-Pilot Guide

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 5 — Validation Experiments and Research Reporting  
**Document ID:** `docs/reports/FS-005_PHASE_5_COLAB_PILOT_GUIDE.md`  
**Status:** Pilot-readiness guide (implementation complete; pilot not executed)  
**Scope:** Practical execution guide for a future 10–20-sample validation pilot on Google Colab Tesla T4.

---

## ⚠️ Purpose and scope (read first)

- This is a **feasibility pilot on 10–20 samples**. It is **not** a statistically
  meaningful experiment and must **not** be used to support performance,
  accuracy, grounding, or clinical-superiority claims.
- **Do not execute real model inference, calibration, or test evaluation as
  part of this implementation task.** The pilot is run separately, after
  explicit approval.
- **Do not print, expose, hardcode, or save credentials** (HF_TOKEN, etc.) in
  project artifacts, notebooks, cells, or Git. Use Colab Secrets only.
- The full experiment requires up to **40,210 unique generations** and exceeds
  a single Colab session (~12 hours). The pilot validates the harness
  end-to-end on a tiny subset before any full run.

---

## 1. Pilot data isolation (mandatory)

The pilot uses a **validation-only** cohort — it never touches the official
945-sample test cohort.

| Item | Specification |
|------|---------------|
| Source dataset | SLAKE English **validation** split |
| Pilot samples | 15 (within the required 10–20 range) |
| Distinct pilot images | 15 (image-disjoint: no two samples share an image) |
| Track B-eligible (with GT regions) | 7 |
| Track C-only (without GT regions) | 8 |
| FS-004 pilot QIDs excluded | 10 |
| FS-004 pilot image IDs excluded | 9 (130, 185, 224, 234, 312, 320, 344, 561, 605) |

**Calibration isolation:** The 15 pilot image IDs are excluded from calibration
fitting **and** assessment. Calibration split manifests are regenerated so that
no pilot image ID appears in either calibration subset.

**Test cohort preservation:** The official test cohort is preserved completely
untouched. Pilot samples are never used to fit calibration parameters or
assess model performance.

Generate the pilot cohort (read-only on data; writes manifests) before running
the Colab pilot:

```
python scripts/fs005_phase5_pilot_cohort.py
```

This writes:
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_manifest.json`
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_qids.csv`
- `experiments/runs/fs005_phase5/pilot_cohort/pilot_image_ids.json`

and regenerates calibration splits under
`experiments/runs/fs005_phase5/calibration/splits/` excluding the pilot image
IDs.

---

## 2. Colab T4 setup

1. Open a new Google Colab notebook (`File → New notebook`).
2. Set the runtime to GPU: `Runtime → Change runtime type → T4 GPU`. Save.
3. Confirm the runtime reports a Tesla T4 (~15.3 GB VRAM). If a T4 is not
   assigned, disconnect and reconnect until one is assigned.
4. Clone this repository into Colab's ephemeral `/content` (do **not** commit
   credentials):
   ```
   !git clone <this-repo-url> /content/MedicalVQA
   %cd /content/MedicalVQA
   ```

### Project/data preparation

The SLAKE dataset (heavy, on drive `D:` locally) must be available in Colab.
Copy `data/SLAKE/test.json`, `data/SLAKE/validate.json`, and
`data/SLAKE/imgs/` to `/content/MedicalVQA/data/SLAKE/`, or download SLAKE
from its official public source inside Colab (no credentials needed).

Verify counts before generating:
- English validation: 1,053 samples (970 with GT regions, 83 without).
- English test: 1,061 samples → 945 eligible after FS-004 exclusion
  (864 with GT, 81 without). The test cohort is **never consumed** by the pilot.

---

## 3. Persistent Google Drive storage (mandatory)

`/content` is ephemeral — artifacts there are lost on runtime restart. All
pilot artifacts must be written to Google Drive.

### Mount and path configuration

Run this cell at the start of every session:

```python
from google.colab import drive  # type: ignore
drive.mount("/content/drive", force_remount=True)

PERSISTENT_OUTPUT_DIR = "/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5"
import os
os.makedirs(PERSISTENT_OUTPUT_DIR + "/generations", exist_ok=True)
os.makedirs(PERSISTENT_OUTPUT_DIR + "/conditions", exist_ok=True)
os.makedirs(PERSISTENT_OUTPUT_DIR + "/calibration", exist_ok=True)
print("Persistent output dir:", PERSISTENT_OUTPUT_DIR)
```

- Generation JSON records are saved directly to
  `PERSISTENT_OUTPUT_DIR/generations/<model>/<track>/<generation_id>.json`.
- Progress and failure summaries are saved after **every sample** to
  `PERSISTENT_OUTPUT_DIR/conditions/progress_<model>_track_<track>.json`.
- Artifacts are **verified after every write** (re-read + re-validated) so a
  truncated write is detected before resume relies on it.
- On runtime restart: re-mount Drive, re-run the harness with the same inputs,
  and completed artifacts are skipped (no duplicate generation). Corrupt or
  failed artifacts are retried/regenerated.
- Never treat `/content` as persistent storage.

---

## 4. Secure Hugging Face authentication (Colab Secrets)

MedGemma-4B (`google/medgemma-4b-it`) is a **gated** model. Qwen2.5-VL-3B is public.

1. In Colab, open the left sidebar → **Secrets** (key icon).
2. Add a secret named `HF_TOKEN` with your Hugging Face user access token
   (read access; you must have accepted the MedGemma license on huggingface.co).
3. **Never** print `HF_TOKEN`, never `!echo $HF_TOKEN`, never save it to a file
   or to Git.
4. **Colab Secrets do NOT automatically populate `os.environ`.** The entry point
   reads the token via `google.colab.userdata.get("HF_TOKEN")`, falling back to
   `os.environ.get("HF_TOKEN")` only for non-Colab/local testing, then calls
   `huggingface_hub.login(token=..., add_to_git_credential=False)`.

The entry-point script (`colab/fs005_phase5_colab_pilot.py`,
`configure_hf_auth()`) retrieves `HF_TOKEN` from Colab Secrets and never prints
or persists it.

---

## 5. Models and sequential loading

| Model | Model ID | Gated? | Quantization | Track |
|-------|----------|--------|--------------|-------|
| MedGemma-4B | `google/medgemma-4b-it` | Yes (HF gate) | 4-bit (nf4) on GPU | B and C |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | No | 4-bit (nf4) on GPU | B and C |

**Pilot model/track matrix (all four combinations run in real mode):**

| # | Model | Track | Sequential loading |
|---|-------|-------|--------------------|
| 1 | MedGemma-4B | B | MedGemma loaded once; Track B then Track C; then unloaded |
| 2 | MedGemma-4B | C | (same MedGemma load) |
| 3 | Qwen2.5-VL-3B | B | Qwen loaded once after MedGemma unloaded; Track B then Track C; then unloaded |
| 4 | Qwen2.5-VL-3B | C | (same Qwen load) |

- Load models **sequentially** — never resident in GPU VRAM at the same time.
  Peak VRAM ~6–7 GB per model (within T4's 15.3 GB).
- Unload a model between models (delete model/processor +
  `torch.cuda.empty_cache()`).
- Gemma 3 (MedGemma) requires `torch.bfloat16` on CUDA (float16 yields empty
  output — known transformers limitation).

---

## 6. Tracks, conditions, and shared response artifacts

| Track | Grounding | Pilot samples | Purpose |
|-------|-----------|---------------|---------|
| Track B | Oracle GT boxes (detection.json non-empty) | 7 (from pilot) | Pointing accuracy + grounded answering |
| Track C | Ungrounded (no regions) | 15 (all pilot) | Answer-accuracy baseline |

| Condition | N | Temperature | Abstention | Purpose |
|-----------|---|-------------|------------|---------|
| Condition 1 | 1 | 0.0 (greedy) | No | Free-form baseline (where applicable) |
| Condition 2 | 5 | 0.7 | No | Grounded + verification, no abstention |
| Condition 3 | 5 | 0.7 | Yes (confidence gate) | Grounded + verification + abstention |

**Mandatory:** Conditions 2 and 3 share the **same N=5 generated responses**
per sample. Generate once, persist, evaluate both conditions from the same
artifact. Only the abstention decision differs. Do **not** regenerate for
Condition 3.

### Generation artifact format

Each generation is persisted as a `GenerationRecord` (JSON, one file per
sample) under:

```
<PERSISTENT_OUTPUT_DIR>/generations/<model_name>/<track>/<generation_id>.json
```

`generation_id` = SHA-256 over
`(model_name, model_revision, track, question_id, image_id, n_samples,
temperature, random_seed)`. The condition is **excluded** from the hash, so
Conditions 2 and 3 resolve to the same file.

Each record contains:
- model name + checkpoint/revision identifier
- track (B or C)
- question ID + image ID
- condition-independent generation identifier
- sampling count (N), temperature, random seed
- individual raw responses for each of the N samples
- parsed answers + extracted claims
- generation status and validation status
- inference duration + timestamps (UTC)
- error information when generation fails

Records are validated before admission to evaluation
(`validate_generation_record`).

---

## 7. Execution entry point (Colab-compatible)

The executable entry point is `colab/fs005_phase5_colab_pilot.py`. It is
structured with `# %%` cell markers so it can be run cell-by-cell in Colab,
or copied into a Colab notebook.

### Default: mock-mode dry run (no inference)

The script runs a mock-mode dry run by default
(`RUN_REAL_INFERENCE = False`). This exercises:
- Drive mount + persistent output path
- Pilot cohort loading
- Image/sample preparation
- GenerationRunner with `MockVLMClient`
- Track C execution, artifact persistence, progress summaries
- Resume / skip-completed behavior

No model weights are downloaded and no inference is performed.

### Real-model cells (present but NOT executed in this task)

The script also contains real-model execution cells (MedGemma-4B load, Qwen2.5-VL-3B load, sequential loading, Track B and Track C execution) gated behind
`RUN_REAL_INFERENCE = True`. **Do not flip this flag until you have explicit
approval to run the pilot.** These cells are present for completeness and
auditability but must not be executed now.

Key demonstrations in code:
1. Loading the MedGemma-4B client (`load_medgemma_client`)
2. Loading the Qwen2.5-VL-3B client (`load_qwen_client`)
3. Sequential loading (`unload_client` between models)
4. Preparing images and input samples (`build_input_samples`)
5. Initializing `GenerationRunner` with the real `VLMClient`
6. Executing Track B and Track C (`run_tracks`)
7. Persisting generation artifacts (via `GenerationArtifactStore`)
8. Reusing the same N=5 responses for Conditions 2 and 3 (same
   `generation_id`, only abstention differs)
9. Resuming completed work without duplicate generation (`is_completed_valid`
   skip)

---

## 8. Colab commands (cell-by-cell)

Replace all Windows-only commands (`.venv\Scripts\python`, `&&`) with these
Colab-compatible commands.

### Cell A — Repository preparation
```python
# %% [markdown]
# ## A. Repository preparation
# %%
from google.colab import drive  # type: ignore
drive.mount("/content/drive", force_remount=True)
import os
os.makedirs("/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5/generations", exist_ok=True)
os.makedirs("/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5/conditions", exist_ok=True)
os.makedirs("/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5/calibration", exist_ok=True)
print("Drive mounted and persistent output dirs created.")

# %%
import subprocess, sys
repo = "/content/MedicalVQA"
if not os.path.exists(repo):
    subprocess.run(["git", "clone", "<this-repo-url>", repo], check=True)
if repo not in sys.path:
    sys.path.insert(0, repo)
os.chdir(repo)
print("Working directory:", os.getcwd())
```

### Cell B — Dependency verification
```python
# %% [markdown]
# ## B. Dependency verification
# %%
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-dev.txt"], check=True)
print("Dependencies verified.")
```

### Cell C — Dataset preparation
```python
# %% [markdown]
# ## C. Dataset preparation
# %%
# Ensure SLAKE is present under /content/MedicalVQA/data/SLAKE/
# (copy test.json, validate.json, imgs/ or download from the official source).
import os
for p in ["data/SLAKE/test.json", "data/SLAKE/validate.json", "data/SLAKE/imgs"]:
    print(p, "exists:", os.path.exists(p))
```

### Cell D — Colab Secrets authentication
```python
# %% [markdown]
# ## D. Hugging Face authentication (Colab Secrets)
# %%
# Colab Secrets do NOT auto-populate os.environ; read via google.colab.userdata.
try:
    from google.colab import userdata  # type: ignore
    token = userdata.get("HF_TOKEN")
except Exception:
    token = None
import os
if not token:
    token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError(
        "HF_TOKEN secret not set. Add it in Colab Secrets (key: HF_TOKEN) "
        "and restart the runtime."
    )
from huggingface_hub import login  # type: ignore
login(token=token, add_to_git_credential=False)
print("HF authentication configured (token not printed).")
```

### Cell E — Mock-mode dry run
```python
# %% [markdown]
# ## E. Mock-mode dry run (no inference) — safe to execute now
# %%
# Generates pilot cohort locally first (run once before uploading):
#   python scripts/fs005_phase5_pilot_cohort.py
from colab.fs005_phase5_colab_pilot import main
# Dry run: RUN_REAL_INFERENCE is False by default. Exercises BOTH Track B and
# Track C with MockVLMClient (no model weights downloaded, no inference).
raise SystemExit(main())
```

### Cell F — Real-model initialization (present, NOT executed now)
```python
# %% [markdown]
# ## F. Real-model initialization (DO NOT RUN without approval)
# %%
# This cell is present for completeness. RUN_REAL_INFERENCE remains False.
# To run real inference after approval, set RUN_REAL_INFERENCE=True in
# colab/fs005_phase5_colab_pilot.py and re-run this notebook.
print("Real-model cells are present but gated off (RUN_REAL_INFERENCE=False).")
```

### Cell G — Pilot execution (real model, gated)
```python
# %% [markdown]
# ## G. Pilot execution (real model — gated; NOT executed in this task)
# %%
# The real execution path lives in colab/fs005_phase5_colab_pilot.py:main().
# It is gated behind RUN_REAL_INFERENCE (currently False). After explicit
# approval, set RUN_REAL_INFERENCE=True in that file and re-run this notebook
# (Cell E will then execute real inference).
#
# Real path: loads MedGemma-4B once, runs Track B then Track C, unloads;
# loads Qwen2.5-VL-3B once, runs Track B then Track C, unloads. All four
# model/track combinations run sequentially on the 15-sample validation-only
# pilot cohort. Conditions 2 and 3 reuse the same N=5 artifacts; artifacts are
# persisted to Google Drive and verified after every write; resume skips
# completed samples.
print("Cell G is gated off via RUN_REAL_INFERENCE=False. Do not execute without approval.")
```

### Cell H — Resume procedure (after Colab disconnect)
```python
# %% [markdown]
# ## H. Resume after Colab disconnect
# %%
# 1. Reconnect to a T4 runtime (or create a new one).
# 2. Re-run Cell A (Drive mount) and Cell B (dependencies).
# 3. Re-add HF_TOKEN to Colab Secrets if the runtime was recycled.
# 4. Re-run Cell E (mock dry run) or Cell G (real model, after approval).
#    The harness skips valid completed artifacts and retries only
#    failed/corrupt ones — no duplicate generation.
print("Resume procedure documented. Artifacts persist on Google Drive.")
```

---

## 9. GPU memory and inference-time measurements

Record (pilot telemetry only, **not** a verified runtime estimate for the full
experiment):
- Per-model peak VRAM during load and generation
  (`torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()`).
- Per-sample inference duration (wall-clock seconds) — captured in each
  `GenerationRecord` (`inference_duration_s`, `total_duration_s`).
- Total pilot wall time.
- Confirm models are never resident simultaneously (sequential loading).

These measurements inform whether the full 40,210-generation run fits within
Colab session limits. They are **not** a verified runtime estimate.

---

## 10. Output parsing and artifact validation

- Each VLM raw response is parsed into `VLMStructuredOutput` (answer +
  findings with region_id/status/confidence). Malformed responses are recorded
  as failed samples with error info, not silently dropped.
- `validate_generation_record` checks: schema version, sample count = n_samples,
  contiguous sample indices, completed⇔error consistency, parseable timestamps,
  non-negative durations.
- Only valid COMPLETED records are admitted to condition evaluation.
- Condition 2 output: answer (majority), no abstention.
- Condition 3 output: answer or None (abstained), confidence vs tau.

---

## 11. Expected files (pilot)

```
/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5/
├── generations/
│   └── <model>/<track>/<generation_id>.json   # one per sample (N=5 each)
├── conditions/
│   └── progress_<model>_track_<track>.json    # incremental progress + failures
└── calibration/                               # split manifests (precomputed locally)
    ├── <model>/B/{fit,assessment}_qids.csv, ..._image_ids.json, split_manifest.json
    └── <model>/C/{fit,assessment}_qids.csv, ..._image_ids.json, split_manifest.json
```

Calibration splits are computed locally (read-only on SLAKE validate.json) via
`scripts/fs005_phase5_calibration_split.py --exclude-image-ids <pilot image IDs>`
and copied to Colab; no inference needed for the split.

---

## 12. Pilot success criteria

A successful feasibility pilot must demonstrate ALL of the following
(none are performance claims):

1. **Validation-only pilot cohort isolation:** 10–20 samples from SLAKE
   validation, image-disjoint, no overlap with the official test cohort.
2. **No calibration overlap:** No pilot image ID appears in calibration fit
   or assessment subsets; manifests regenerated excluding pilot image IDs.
3. **All four model/track combinations run sequentially on T4:** MedGemma-4B
   × Track B, MedGemma-4B × Track C, Qwen2.5-VL-3B × Track B, Qwen2.5-VL-3B
   × Track C; each model loads once in 4-bit without OOM; models never resident
   simultaneously.
4. **Both tracks execute:** Track B and Track C run on the pilot cohort for
   each model.
5. **N=5 outputs persist correctly:** Generation records written, verified
   after write, and readable on Drive.
6. **Conditions 2 and 3 reuse identical generation artifacts:** Same
   `generation_id`, responses, claims, agreement, entropy, confidence; only
   the abstention decision differs.
7. **Drive artifacts survive a simulated runtime restart:** Re-mount Drive,
   re-run with same inputs, completed artifacts intact and valid.
8. **Resume skips completed samples:** Second run reports all samples as
   skipped; no duplicate generation.
9. **No credentials are exposed:** HF_TOKEN never printed, saved, or committed.
10. **No test-set samples are consumed:** Pilot uses validation only; test
    cohort untouched.

Failure of any item blocks the full experiment until resolved.

---

## 13. Recovery after Colab disconnects

1. Reconnect to the Colab runtime (T4), or create a new T4 runtime.
2. Re-run Cell A (Drive mount) — artifacts are on Drive, not `/content`.
3. Re-run Cell B (dependencies).
4. Re-add `HF_TOKEN` to Colab Secrets if the runtime was recycled.
5. Re-run with the **same inputs** (same model, track, samples, N, temperature,
   seed). Valid COMPLETED records are skipped; FAILED/corrupt records are
   retried/regenerated. Confirm `n_completed` matches the pre-disconnect count
   before generating new samples.

---

## 14. Restrictions (pilot)

- No model inference as part of this implementation task; pilot execution
  requires separate approval.
- No calibration fitting or test evaluation during the pilot beyond feasibility
  telemetry.
- No HF_TOKEN printed, exposed, hardcoded, or saved in artifacts.
- No FS-001–FS-004 modifications; no ADR 0004/0005 changes; no Phase 1–4
  behavior changes.
- No performance or clinical-superiority claims from the pilot.
- Do not begin the 40,210-generation full experiment or Stage 6.
