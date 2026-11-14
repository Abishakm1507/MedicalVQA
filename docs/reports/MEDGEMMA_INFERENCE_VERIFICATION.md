# MedGemma-4B Inference Verification Report

**Objective:** Verify MedGemma-4B (`google/medgemma-4b-it`) can perform genuine image-based Medical VQA on SLAKE images.

**Result:** **BLOCKED** — Model weights could not be loaded. No inference was performed.

---

## 1. Implementation

### 1.1 New file: `vlm/medgemma_client.py`

Implemented `MedGemmaVLMClient` following the existing `VLMClient` interface and the `QwenVLMClient` pattern:

- Lazy model loading (torch/transformers imported only on first `_load()` call)
- Supports `device="cpu"` or `device="cuda"` with optional `quantization="4bit"`
- Uses `AutoProcessor` + `Gemma3ForConditionalGeneration` (MedGemma-4B is based on Gemma 3 architecture)
- Constructs Gemma-3 chat messages with image + text content
- Returns raw text responses conforming to the structured JSON output schema
- `generate_structured()` inherited from `VLMClient` base class
- Proper error handling for loading and inference failures

### 1.2 Updated file: `vlm/__init__.py`

Added `MedGemmaVLMClient` to `__all__` and the lazy `__getattr__` export pattern. Existing clients (`QwenVLMClient`, `MockVLMClient`, `OllamaVLMClient`) unchanged and verified to still import correctly.

### 1.3 Interface compatibility

`MedGemmaVLMClient` implements the full `VLMClient` interface:
- `model_name` property → `"medgemma-4b"`
- `device` property
- `generate(marked_image, question, temperature, max_tokens)` → `str`
- `sample_n(marked_image, question, n, temperature, max_tokens)` → `list[str]`
- `generate_structured(...)` → `VLMStructuredOutput` (inherited)

Compatible with future grounding and uncertainty integration (FS-007).

---

## 2. Environment and Access Check

### 2.1 Hardware

| Resource | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 2050, 4.00 GB VRAM |
| CUDA | Available (1 device) |
| RAM total | 15.71 GB |
| RAM available | 2.39 GB (13.33 GB used) |
| Disk free (D:) | 71.62 GB |

### 2.2 Dependencies (all installed)

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| transformers | 5.23.0 |
| huggingface_hub | 2.4.0 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |

### 2.3 Model access

| Field | Value |
|-------|-------|
| Model ID | `google/medgemma-4b-it` |
| Gated | Yes (`gated=auto`) |
| HF token configured | No |
| HF hub cache | Empty |
| Accessible | No — 401 Client Error |

`google/medgemma-4b-it` is a **gated model** on Hugging Face. Access requires:
1. Accepting the MedGemma license on huggingface.co
2. Being added to the allowlist (Google reviews access requests)
3. Configuring an authenticated `huggingface_hub` token via `huggingface-cli login`

None of these preconditions are met in this environment.

---

## 3. Smoke Test Attempt

| Field | Value |
|-------|-------|
| Samples selected | 3 (from Qwen baseline manifest: qids 11981, 12073, 12111) |
| Device | cpu |
| Quantization | None |
| Load time | 19.45 s |
| Load result | **FAILED** |

### 3.1 Error

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/medgemma-4b-it.
401 Client Error.
Cannot access gated repo for url https://huggingface.co/google/medgemma-4b-it/resolve/.../config.json.
Access to model google/medgemma-4b-it is restricted. You must have access to it and be authenticated to access it. Please log in.
```

### 3.2 Blocker report

Saved to: `experiments/runs/medgemma_baseline/blocker_report.json`

Root causes:
1. **No HF authentication** — MedGemma-4B-it is gated; no token configured, no license accepted
2. **Hardware constraints** — RTX 2050 (4 GB VRAM) and only 2.39 GB RAM available make local 4B model inference impractical even if access were granted

---

## 4. Inference Performed

**None.** The model could not be loaded, so no images were submitted and no inferences were run. No synthetic images were used, no paid APIs were contacted, and no substitute model was loaded.

---

## 5. Required Next Steps (Unblocking)

To execute MedGemma-4B inference, one of the following is required:

1. **HF access:** Configure `huggingface-cli login` with a token that has been granted access to `google/medgemma-4b-it` (license accepted on HF web + allowlist approval from Google).
2. **Hardware:** Local inference on RTX 2050 (4 GB VRAM) with 4-bit quantization may be marginal; Colab/Kaggle free GPU tier (per AGENTS.md) is the recommended path for heavy VLM inference.
3. **RAM:** Free up system RAM (currently only 2.39 GB available) before attempting local model loading.

---

## 6. Files

| File | Status |
|------|--------|
| `vlm/medgemma_client.py` | Created (implementation, weights not loaded) |
| `vlm/__init__.py` | Modified (added MedGemmaVLMClient export) |
| `experiments/runs/medgemma_baseline/blocker_report.json` | Created |
| `docs/reports/MEDGEMMA_INFERENCE_VERIFICATION.md` | This report |
| `docs/reports/FS-004_MEDGEMMA_BASELINE_REPORT.md` | Separate report |

---

## 7. Conclusion

MedGemma-4B integration code is complete and imports correctly, but **inference verification could not be performed** because the model weights are gated and inaccessible without HF authentication, and local hardware is marginal for a 4B model. Per the task constraints, no substitute model was used and no paid API was contacted. Waiting for HF access or alternative hardware before proceeding to the controlled 10-sample baseline.
