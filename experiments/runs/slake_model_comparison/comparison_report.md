# FS-004 — Qwen2.5-VL-3B vs MedGemma-4B Baseline Comparison Report

## 1. Consistency verification

- Manifest samples: 10
- Qwen raw response files on disk: 10
- MedGemma raw response files on disk: 10
- Sample IDs match Qwen selection manifest: True
- Question text matches manifest: True
- Ground truth matches manifest: True
- Valid paired experiment: True

## 2. Recalculated metrics (from raw predictions)

| Metric | Qwen2.5-VL-3B | MedGemma-4B |
|---|---:|---:|
| Total samples | 10 | 10 |
| Overall accuracy | 60.0% | 70.0% |
| OPEN accuracy | 40.0% | 40.0% |
| CLOSED accuracy | 80.0% | 100.0% |
| Correct predictions | 6 | 7 |
| Incorrect predictions | 4 | 3 |
| Failed predictions | 0 | 0 |
| Missing predictions | 0 | 0 |
| Average inference time (s) | 15.9144 | 5.3056 |
| Median inference time (s) | 12.5907 | 5.3033 |
| Total inference time (s) | 159.1436 | 53.0565 |

### Per-category detail

| Category | Qwen correct/total | Qwen accuracy | MedGemma correct/total | MedGemma accuracy |
|---|---|---|---|---|
| OPEN | 2/5 | 40.0% | 2/5 | 40.0% |
| CLOSED | 4/5 | 80.0% | 5/5 | 100.0% |

## 3. Paired outcome by sample

- Both correct: [11981, 12073, 12111, 12384, 12914, 12919]
- Both incorrect: [12129, 12312, 12344]
- Only Qwen correct: []
- Only MedGemma correct: [12845]
- Both failed or missing: []

## 4. Per-sample results

| QID | Category | Ground truth | Qwen pred | Qwen correct | MedGemma pred | MedGemma correct | Qwen time (s) | MedGemma time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 11981 | OPEN | Chest | chest | True | Chest | True | 42.1188 | 5.9233 |  |
| 12073 | CLOSED | Yes | Yes | True | Yes | True | 17.1987 | 5.0325 |  |
| 12111 | CLOSED | Yes | Yes | True | Yes | True | 10.6107 | 5.258 |  |
| 12129 | OPEN | 0 | 2 | False | 1 | False | 14.2062 | 4.8384 |  |
| 12312 | OPEN | Lower Left Lung | Right Lung | False | Left lung | False | 12.5233 | 5.2636 |  |
| 12344 | OPEN | Lung | lungs | False | Lungs | False | 12.6581 | 5.1684 |  |
| 12384 | OPEN | X-Ray | X-ray | True | X-ray | True | 12.3349 | 5.4345 |  |
| 12845 | CLOSED | No | yes | False | No | True | 10.7909 | 5.3429 |  |
| 12914 | CLOSED | Yes | yes | True | Yes | True | 10.5431 | 5.4212 |  |
| 12919 | CLOSED | Yes | yes | True | Yes | True | 16.1589 | 5.3737 |  |

## 5. Evaluation methodology

- Answer normalization (applied to both prediction and ground truth): strip whitespace,
  remove markdown code fences and surrounding quotes, collapse internal whitespace,
  lowercase, remove trailing punctuation (.,!?;:).
- Correctness: exact string match after normalization. No stemming, no semantic judging.
- OPEN questions (5): free-text answers compared exactly after normalization.
- CLOSED questions (5): yes/no answers compared exactly after normalization.
- Failed inference (exception / empty raw response) and missing predictions (no raw file):
  counted as incorrect and retained in the denominator; never treated as correct.
- Metrics recalculated independently from raw prediction files; existing summary files
  were read only for cross-checking, not trusted.

## 6. Qualitative observations

- Qwen2.5-VL-3B produced genuine structured JSON responses with `answer`, `findings`,
  `region_id`, `status`, and `confidence` fields on all 10 samples.
- MedGemma-4B returned empty raw responses for all 6 available samples and has no raw
  prediction files for the remaining 4 samples; no MedGemma text is available for
  qualitative assessment of hallucinations, unsupported claims, or completeness.
- No claim of clinical superiority is made; this is a 10-sample pilot, not a
  statistically conclusive benchmark.

### Qwen incorrect predictions (strict exact match)
- qid=12129 (OPEN): GT='0' pred='2' — normalized '2' vs '0'
- qid=12312 (OPEN): GT='Lower Left Lung' pred='Right Lung' — normalized 'right lung' vs 'lower left lung'
- qid=12344 (OPEN): GT='Lung' pred='lungs' — normalized 'lungs' vs 'lung'
- qid=12845 (CLOSED): GT='No' pred='yes' — normalized 'yes' vs 'no'

## 7. Artifacts

- `comparison_metrics.json` — machine-readable metrics and verification results
- `comparison_summary.csv` — summary comparison table
- `per_sample_comparison.csv` — per-sample paired comparison
- `comparison_report.md` — this report
- `scripts/compare_slake_baselines.py` — reproducible comparison script

Original directories preserved: `experiments/runs/slake_baseline_ollama/` and
`experiments/runs/medgemma_baseline/` were not modified.
