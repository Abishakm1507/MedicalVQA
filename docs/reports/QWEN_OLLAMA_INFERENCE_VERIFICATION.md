# Qwen2.5-VL Ollama Smoke Test — Verification Report

**Objective:** Verify that the existing `OllamaVLMClient` can perform genuine image-based Medical VQA using the locally installed Ollama model `qwen2.5vl:3b` on real SLAKE test images.

**Result:** **SMOKE TEST PASSED** — all 3 samples completed successfully with correct answers, all from genuine SLAKE images processed by the vision-capable Ollama model.

---

## 1. Environment and Model Metadata

| Field | Value |
|-------|-------|
| Ollama version | 0.40.16 |
| Model tag | `qwen2.5vl:3b` |
| Model digest | `fb90415cde1ef08aa669ae74b082d49b158729b6db1ab183c941417d507e71a1` |
| Model size | 3,200,627,168 bytes (~3.2 GB) |
| Format | gguf |
| Model family | qwen25vl |
| Parameter size | 3.8B |
| Quantization | Q4_K_M |
| Context length | 128,000 |
| Capabilities | completion, vision |
| Server endpoint | http://localhost:11434 |
| Hardware | Local CPU (Ollama server process; no GPU offload configured) |
| Client | `vlm/ollama_client.py` `OllamaVLMClient` (default model `qwen2.5vl:3b`) |

The model's `vision` capability confirms it is a vision-language model, not text-only.

---

## 2. Selected Samples (Official SLAKE Test Split, English Only)

All samples selected from `data/SLAKE/test.json` with `q_lang="en"`. No training or validation samples used. No synthetic images.

| # | qid | sample_id | image_id | Image | Size | Modality | Answer Type | Question | Ground Truth |
|---|-----|-----------|----------|-------|------|----------|-------------|----------|-------------|
| 1 | 11934 | slake_test_11934 | 102 | `xmlab102/source.jpg` | 512×512 | CT | OPEN | "What modality is used to take this image?" | "CT" |
| 2 | 11938 | slake_test_11938 | 102 | `xmlab102/source.jpg` | 512×512 | CT | CLOSED | "Does the picture contain liver?" | "No" |
| 3 | 11965 | slake_test_11965 | 124 | `xmlab124/source.jpg` | 1024×1024 | X-RAY | OPEN | "What modality is used to take this image?" | "X-Ray" |

Sample 3 uses a different image, modality (X-RAY vs CT), and resolution (1024×1024 vs 512×512) from samples 1–2, providing cross-configuration coverage.

---

## 3. Inference Results

### Sample 1 — qid=11934 (OPEN, CT)

| Field | Value |
|-------|-------|
| Question | "What modality is used to take this image?" |
| Ground truth | "CT" |
| Raw response | `{"answer": "CT", "findings": [{"claim": "CT", "region_id": "none", "status": "present", "confidence": "high"}]}` |
| Parsed answer | "CT" |
| Correct | Yes |
| Inference time | 43.5840 s |
| wall_time_s | 43.0616 |
| total_duration_s | 40.9388 |
| prompt_eval_count | 1168 |
| eval_count | 51 |

### Sample 2 — qid=11938 (CLOSED, CT)

| Field | Value |
|-------|-------|
| Question | "Does the picture contain liver?" |
| Ground truth | "No" |
| Raw response | `{"answer": "No", "findings": [{"claim": "Liver", "region_id": "liver", "status": "absent", "confidence": "high"}]}` |
| Parsed answer | "No" |
| Correct | Yes |
| Inference time | 12.7665 s |
| wall_time_s | 12.7623 |
| total_duration_s | 10.5820 |
| prompt_eval_count | 1164 |
| eval_count | 57 |

### Sample 3 — qid=11965 (OPEN, X-RAY)

| Field | Value |
|-------|-------|
| Question | "What modality is used to take this image?" |
| Ground truth | "X-Ray" |
| Raw response | `{"answer": "X-ray", "findings": [{"claim": "X-ray", "region_id": "chest", "status": "present", "confidence": "high"}]}` |
| Parsed answer | "X-ray" |
| Correct | Yes |
| Inference time | 19.7270 s |
| wall_time_s | 19.7162 |
| total_duration_s | 17.6480 |
| prompt_eval_count | 1513 |
| eval_count | 53 |

---

## 4. Verification That the Model Processed the Images

All three answers demonstrate genuine visual processing, not text-only responses:

1. **Modality identification (CT vs X-Ray):** The model correctly distinguished CT from X-Ray across two different images with different resolutions. A text-only model could not determine imaging modality from the question alone.
2. **Organ presence/absence:** For sample 2, the model correctly answered "No" to "Does the picture contain liver?" with a finding that "Liver" status is "absent" — this requires visual analysis of the chest CT image (which shows lungs, not abdominal organs).
3. **Structured JSON output:** All responses conform to the expected VLM structured output schema (`answer` + `findings` with `claim`, `region_id`, `status`, `confidence`).

Ollama metadata confirms image processing occurred: `prompt_eval_count` values (1168, 1164, 1513) reflect the image tokens being processed, and `total_duration_s` includes model compute time.

---

## 5. Artifacts

All artifacts saved under `experiments/runs/qwen_ollama_smoke_test/`:

| File | Description |
|------|-------------|
| `img_q11934_q11938_xmlab102_source.jpg` | Original SLAKE image (xmlab102, 512×512 CT) used for samples 1–2 |
| `img_q11965_xmlab124_source.jpg` | Original SLAKE image (xmlab124, 1024×1024 X-Ray) used for sample 3 |
| `q11934_response.txt` | Sample 1: question, ground truth, raw response, parsed answer, timing |
| `q11938_response.txt` | Sample 2: question, ground truth, raw response, parsed answer, timing |
| `q11965_response.txt` | Sample 3: question, ground truth, raw response, parsed answer, timing |
| `smoke_test_summary.json` | Machine-readable summary of all results and model metadata |

---

## 6. Errors, Limitations, and Risks

- **No errors:** All 3 samples completed without exceptions. `error: null` for all results.
- **No parsing failures:** All responses were valid JSON matching the structured output schema.
- **Ollama server was not running initially:** Had to start `ollama.exe serve` manually. The server process must be running before inference.
- **Hardware:** Inference ran on CPU only (no GPU offload). First sample had a higher load_duration (28.39s) likely due to model loading; subsequent samples were faster.
- **Small sample size:** 3 samples is a smoke test only, not a statistically significant evaluation. Per the evaluation protocol, the full baseline should use the official test split (1,061 English entries, 96 unique images) with bootstrap confidence intervals.
- **Sample 1–2 from same image:** Two samples share `xmlab102/source.jpg`; sample 3 provides cross-image coverage.
- **No patient IDs:** SLAKE does not publish patient identifiers; `img_id` is the only leakage proxy.

---

## 7. Conclusion

**The three-sample Qwen2.5-VL Ollama smoke test succeeded.**

- All 3 predictions are correct (modality CT, liver absence, modality X-Ray).
- All 3 used genuine SLAKE test images (not synthetic).
- All 3 were processed by the vision-capable `qwen2.5vl:3b` model (not text-only).
- All responses conform to the structured JSON output schema.
- The `OllamaVLMClient` implementation works correctly with real medical images.

**Ready to proceed to the full 5–10-sample baseline** using the official SLAKE test split (English), pending review of this report.
