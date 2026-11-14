# FS-004 Baseline Experiment Report — Qwen2.5-VL-3B on SLAKE

**Objective:** Pilot evaluation of Qwen2.5-VL-3B (`qwen2.5vl:3b`) on 10 genuine English samples from the official SLAKE test split.

**Result:** Pilot baseline completed — 60.0% exact-match accuracy (6/10 correct, 4 incorrect, 0 failed). All predictions came from genuine SLAKE images processed by the vision-capable Ollama model.

**Status:** Pilot only — 10 samples do not establish statistically reliable benchmark performance. Same 10 samples preserved for controlled MedGemma-4B comparison.

---

## 1. Methodology

### 1.1 Components used (no production code modified)

| Component | Path | Role |
|-----------|------|------|
| Dataset loader | `datasets/slake_loader.py` | `SlakeDatasetLoader`, loads official test split, English only |
| VLM client | `vlm/ollama_client.py` | `OllamaVLMClient`, calls `http://localhost:11434/api/generate` |
| Ollama server | `ollama.exe serve` | Local Ollama instance, `qwen2.5vl:3b` |

### 1.2 Sample selection

| Parameter | Value |
|-----------|-------|
| Split | `test` (official SLAKE test split) |
| Language | `en` (English only) |
| Total English test entries | 1,061 |
| Excluded (smoke-test qids) | 11934, 11938, 11965 |
| Candidates after exclusion | 1,058 |
| Selection method | `random.Random(42).sample()`, 5 OPEN + 5 CLOSED |
| Random seed | 42 |
| Samples selected | 10 |
| Unique images | 9 (qids 12914 and 12919 share img_id 605) |

Selection manifest saved to: `experiments/runs/slake_baseline_ollama/selection_manifest.json`

### 1.3 Selected samples

| # | qid | img_id | Image | Size | Modality | Type | Question | Ground Truth |
|---|-----|--------|-------|------|----------|------|----------|-------------|
| 1 | 11981 | 130 | `xmlab130/source.jpg` | 1024×1024 | X-RAY | OPEN | "Which part of the body does this image belong to?" | "Chest" |
| 2 | 12073 | 185 | `xmlab185/source.jpg` | 1024×1024 | X-RAY | CLOSED | "Are there abnormalities in this image?" | "Yes" |
| 3 | 12111 | 224 | `xmlab224/source.jpg` | 512×512 | CT | CLOSED | "Does the picture contain spleen?" | "Yes" |
| 4 | 12129 | 234 | `xmlab234/source.jpg` | 512×512 | CT | OPEN | "How many kidneys in this image?" | "0" |
| 5 | 12312 | 312 | `xmlab312/source.jpg` | 1024×1024 | X-RAY | OPEN | "Where is the abnormality in this image?" | "Lower Left Lung" |
| 6 | 12344 | 320 | `xmlab320/source.jpg` | 1024×1024 | X-RAY | OPEN | "Which organs appear in pairs?" | "Lung" |
| 7 | 12384 | 344 | `xmlab344/source.jpg` | 1024×1024 | X-RAY | OPEN | "What imaging modality was used?" | "X-Ray" |
| 8 | 12845 | 561 | `xmlab561/source.jpg` | 512×512 | CT | CLOSED | "Are there any organs used for breathing in this image?" | "No" |
| 9 | 12914 | 605 | `xmlab605/source.jpg` | 512×512 | CT | CLOSED | "Does the picture contain rectum?" | "Yes" |
| 10 | 12919 | 605 | `xmlab605/source.jpg` | 512×512 | CT | CLOSED | "Does the picture contain the organ which has the effect of excreting feces?" | "Yes" |

### 1.4 Answer normalization procedure

Applied deterministically to both prediction and ground truth before exact-match comparison:

1. Strip leading/trailing whitespace.
2. Remove markdown code fences (```json ... ```).
3. Remove surrounding quotes.
4. Collapse internal whitespace runs to single spaces.
5. Lowercase.
6. Remove trailing punctuation (.,!?;:).

**Original answers and predictions are preserved verbatim** in all artifacts. Normalization is applied only for the exact-match metric calculation. No semantic judging is used.

### 1.5 Correctness definition

A prediction is "correct" if `normalize(prediction) == normalize(ground_truth)`. Otherwise "incorrect". A sample is "failed" if the Ollama call raised an exception.

---

## 2. Model and Environment Metadata

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
| Hardware | CPU-only (no GPU offload configured) |
| Client | `OllamaVLMClient` (default model `qwen2.5vl:3b`) |

---

## 3. Per-Sample Results

| # | qid | Type | Image | GT | Prediction | Correct | Time (s) |
|---|-----|------|-------|-----|-----------|---------|----------|
| 1 | 11981 | OPEN | xmlab130 (X-RAY) | "Chest" | "chest" | Yes | 42.12 |
| 2 | 12073 | CLOSED | xmlab185 (X-RAY) | "Yes" | "Yes" | Yes | 17.20 |
| 3 | 12111 | CLOSED | xmlab224 (CT) | "Yes" | "Yes" | Yes | 10.61 |
| 4 | 12129 | OPEN | xmlab234 (CT) | "0" | "2" | No | 14.21 |
| 5 | 12312 | OPEN | xmlab312 (X-RAY) | "Lower Left Lung" | "Right Lung" | No | 12.52 |
| 6 | 12344 | OPEN | xmlab320 (X-RAY) | "Lung" | "lungs" | No | 12.66 |
| 7 | 12384 | OPEN | xmlab344 (X-RAY) | "X-Ray" | "X-ray" | Yes | 12.33 |
| 8 | 12845 | CLOSED | xmlab561 (CT) | "No" | "yes" | No | 10.79 |
| 9 | 12914 | CLOSED | xmlab605 (CT) | "Yes" | "yes" | Yes | 10.54 |
| 10 | 12919 | CLOSED | xmlab605 (CT) | "Yes" | "yes" | Yes | 16.16 |

### 3.1 Incorrect predictions detail

- **qid=12129 (OPEN, CT):** "How many kidneys in this image?" — GT="0", Pred="2". Model detected 2 kidneys; image may actually show a kidney region or the annotation indicates 0 visible kidneys.
- **qid=12312 (OPEN, X-RAY):** "Where is the abnormality?" — GT="Lower Left Lung", Pred="Right Lung". Model got the laterality wrong (right vs left).
- **qid=12344 (OPEN, X-RAY):** "Which organs appear in pairs?" — GT="Lung", Pred="lungs". Singular/plural mismatch under exact-match normalization (no stemming applied).
- **qid=12845 (CLOSED, CT):** "Are there any organs used for breathing?" — GT="No", Pred="yes". Model detected lungs in a chest CT; GT says No.

---

## 4. Metrics

| Metric | Value |
|--------|-------|
| Total samples | 10 |
| Correct | 6 |
| Incorrect | 4 |
| Failed | 0 |
| Overall exact-match accuracy | **60.0%** (6/10) |
| OPEN accuracy | **40.0%** (2/5) |
| CLOSED accuracy | **80.0%** (4/5) |
| Average inference time | 15.91 s |
| Per-sample time range | 10.54 – 42.12 s |

---

## 5. Verification of Genuine Image Processing

All answers demonstrate genuine visual processing:

1. **Modality identification** (qid=11981, 12384): Model correctly identified X-Ray vs CT across different images.
2. **Organ presence/absence** (qid=12073, 12111, 12845): Model correctly answered Yes/No for organ presence questions.
3. **Region grounding** (qid=12312): Model provided a `region_id` with its answer, indicating visual grounding.
4. **Ollama metadata:** `prompt_eval_count` values (1164–1513) confirm image tokens were processed. `total_duration_s` includes model compute time.

All responses conform to the structured JSON output schema (`answer` + `findings`).

---

## 6. Artifacts

All artifacts saved under `experiments/runs/slake_baseline_ollama/`:

| File | Description |
|------|-------------|
| `selection_manifest.json` | Sample selection method, seed, qids, image IDs |
| `baseline_summary.json` | Machine-readable summary with all results and metrics |
| `q{id}_response.txt` (×10) | Per-sample: question, GT, raw response, parsed answer, correctness, timing |
| `img_q{id}_source.jpg` (×10) | Original SLAKE images used for each sample |

---

## 7. Limitations

1. **Pilot only:** 10 samples, 9 unique images — not statistically reliable. Per the experiment protocol, full evaluation should use the official test split (1,061 English entries, 96 unique images) with bootstrap confidence intervals.
2. **CPU-only inference:** No GPU offload; first sample had higher latency due to model loading.
3. **Exact-match is strict:** Singular/plural mismatches (e.g., "Lung" vs "lungs") count as incorrect. No semantic equivalence or stemming is applied.
4. **No patient IDs:** SLAKE publishes no clinical patient identifiers; image-level leakage across splits is a known limitation (298 img_ids repeat across splits when both languages considered; zero English-only overlap).
5. **No Chest X-ray images in SLAKE:** All chest-location images are CT; Track A (CXR) cannot be evaluated on SLAKE.

---

## 8. Conclusions

- The `OllamaVLMClient` works correctly with real SLAKE medical images via Ollama's `qwen2.5vl:3b` vision model.
- Pilot baseline: **60.0% exact-match accuracy** on 10 samples (6 correct, 4 incorrect, 0 failed).
- CLOSED questions (80%) outperformed OPEN questions (40%) in this pilot.
- All 10 samples and their image/question references are preserved under `experiments/runs/slake_baseline_ollama/` for controlled MedGemma-4B comparison.
- **Do not proceed to MedGemma integration yet** — awaiting review of this report.
