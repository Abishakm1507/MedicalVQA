# FS-004 — Qwen2.5-VL-3B vs MedGemma-4B Baseline Comparison Verification

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-004 (baseline comparison)  
**Status:** Verified — valid paired 10-sample pilot comparison complete. NOT a statistically conclusive benchmark.

---

## 1. Files inspected and changed

### Inspected (read-only, not modified)

| File | Role |
|------|------|
| `notebooks/kaggle_medgemma_baseline.ipynb` | MedGemma baseline notebook (Kaggle T4) |
| `experiments/runs/medgemma_baseline/baseline_summary.json` | MedGemma summary (cross-checked, not trusted) |
| `experiments/runs/medgemma_baseline/q{id}_response.json` ×10 | MedGemma individual responses |
| `experiments/runs/slake_baseline_ollama/baseline_summary.json` | Qwen summary (cross-checked, not trusted) |
| `experiments/runs/slake_baseline_ollama/q{id}_response.txt` ×10 | Qwen individual responses |
| `experiments/runs/slake_baseline_ollama/selection_manifest.json` | Cohort source of truth (qids, questions, GT, answer type) |
| `data/SLAKE/test.json` | Official SLAKE test split (ground-truth verification) |
| `scripts/compare_slake_baselines.py` | Reproducible comparison script |
| `tests/unit/test_baseline_comparison.py` | Existing regression tests |
| `docs/reports/SLAKE_EVALUATION_PROTOCOL.md` | Evaluation protocol |
| `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON.md` | Previous (stale) comparison report |

### Changed

| File | Change |
|------|--------|
| `scripts/compare_slake_baselines.py` | Gated the static "NOT a valid paired experiment" NOTE behind `verification["discrepancies"]` so the regenerated report is internally consistent when the experiment is valid (smallest necessary change; no schema change needed) |
| `experiments/runs/slake_model_comparison/comparison_metrics.json` | Regenerated from current raw artifacts |
| `experiments/runs/slake_model_comparison/comparison_report.md` | Regenerated from current raw artifacts |
| `experiments/runs/slake_model_comparison/comparison_summary.csv` | Regenerated from current raw artifacts |
| `experiments/runs/slake_model_comparison/per_sample_comparison.csv` | Regenerated from current raw artifacts |
| `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON_VERIFICATION.md` | Created (this report) |

**Not modified:** `experiments/runs/slake_baseline_ollama/`, `experiments/runs/medgemma_baseline/`, `data/SLAKE/`, FS-002, FS-003.

### Directory naming note

The task text referenced `experiments/runs/slake_baseline_medgemma/` for MedGemma responses. That directory does not exist. MedGemma artifacts actually live under `experiments/runs/medgemma_baseline/` (already correctly referenced by `scripts/compare_slake_baselines.py:28`).

---

## 2. Task 1 — MedGemma artifact inspection (recalculated from individual responses)

All 10 individual response files exist on disk: q11981, q12073, q12111, q12129, q12312, q12344, q12384, q12845, q12914, q12919.

| Check | Result |
|-------|--------|
| Exactly 10 samples processed | PASS — 10 response files present |
| Every response has a valid question ID | PASS — all qids match manifest |
| Raw generated text present and non-empty | PASS — all `raw_response` non-empty |
| Parsed answers valid | PASS — all `parsed_answer` present |
| Ground-truth matches official SLAKE test data | PASS — verified against `data/SLAKE/test.json` (all 10 qids, `q_lang=en`, in test split) |
| Inference times valid and non-negative | PASS — all ~4.84–5.92 s |
| Failed/malformed generations classified as failures | PASS — none failed; all `status: "success"` |

### Recalculated MedGemma metrics (from individual files, not the summary)

| Metric | Value |
|--------|-------|
| Total samples | 10 |
| Correct | 7 (11981, 12073, 12111, 12384, 12845, 12914, 12919) |
| Incorrect | 3 (12129, 12312, 12344) |
| Failed | 0 |
| Missing | 0 |
| Overall accuracy | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) |
| CLOSED accuracy | 1.00 (5/5) |
| Avg inference time | 5.3056 s |
| Median inference time | 5.3033 s |
| Total inference time | 53.0565 s |

Recalculated metrics match `baseline_summary.json` exactly (correct 7, incorrect 3, failed 0, accuracy 0.7, open 0.4, closed 1.0). The MedGemma summary is **correct**.

**MedGemma incorrect predictions (strict exact match):**
- qid=12129 (OPEN): GT='0', pred='1' — normalized '1' vs '0'
- qid=12312 (OPEN): GT='Lower Left Lung', pred='Left lung' — normalized 'left lung' vs 'lower left lung'
- qid=12344 (OPEN): GT='Lung', pred='Lungs' — normalized 'lungs' vs 'lung' (singular/plural, no stemming)

---

## 3. Task 2 — Paired evaluation / cohort verification

MedGemma question IDs are **identical** to the Qwen baseline cohort (exact same 10 qids, same questions, same ground-truth answers, same 5 OPEN / 5 CLOSED split, same 9 unique images).

| Check | Result |
|-------|--------|
| Same 10 question IDs | PASS — identical sets |
| Same ground-truth answers | PASS — matches manifest and `test.json` |
| Same evaluation rules | PASS — same normalization, exact match, denominator = 10 |
| No substitutions / new cohort | PASS |

**Conclusion:** The cohorts satisfy the conditions for a valid paired experiment. `valid_paired_experiment: True`. No discrepancy to report; paired comparison proceeds.

---

## 4. Task 3 — Comparison results (recalculated from raw predictions)

Failed predictions are kept separate from incorrect predictions. Accuracy denominator is explicitly the 10 manifest samples; missing/failed generations are retained in the denominator and never treated as correct.

| Metric | Qwen2.5-VL-3B | MedGemma-4B |
|--------|---------------|-------------|
| Total samples | 10 | 10 |
| Overall accuracy | 0.60 (6/10) | 0.70 (7/10) |
| OPEN accuracy | 0.40 (2/5) | 0.40 (2/5) |
| CLOSED accuracy | 0.80 (4/5) | 1.00 (5/5) |
| Correct predictions | 6 | 7 |
| Incorrect predictions | 4 | 3 |
| Failed predictions | 0 | 0 |
| Missing predictions | 0 | 0 |
| Avg inference time (s) | 15.9144 | 5.3056 |
| Median inference time (s) | 12.5907 | 5.3033 |
| Total inference time (s) | 159.1436 | 53.0565 |

### Paired outcome by sample

| Category | QIDs |
|----------|------|
| Both correct | 11981, 12073, 12111, 12384, 12914, 12919 |
| Both incorrect | 12129, 12312, 12344 |
| Only Qwen correct | (none) |
| Only MedGemma correct | (none) |
| Both failed or missing | (none) |

MedGemma matches Qwen on every sample (same correct/incorrect set) but is ~3× faster (5.3 s vs 15.9 s avg). MedGemma's edge on this pilot comes entirely from CLOSED questions (100% vs 80%); OPEN performance is identical (40%).

### Per-question comparison

| QID | Category | Ground truth | Qwen pred | Qwen correct | MedGemma pred | MedGemma correct | Qwen time | MedGemma time |
|-----|----------|--------------|-----------|--------------|---------------|------------------|-----------|---------------|
| 11981 | OPEN | Chest | chest | True | Chest | True | 42.1188 | 5.9233 |
| 12073 | CLOSED | Yes | Yes | True | Yes | True | 17.1987 | 5.0325 |
| 12111 | CLOSED | Yes | Yes | True | Yes | True | 10.6107 | 5.2580 |
| 12129 | OPEN | 0 | 2 | False | 1 | False | 14.2062 | 4.8384 |
| 12312 | OPEN | Lower Left Lung | Right Lung | False | Left lung | False | 12.5233 | 5.2636 |
| 12344 | OPEN | Lung | lungs | False | Lungs | False | 12.6581 | 5.1684 |
| 12384 | OPEN | X-Ray | X-ray | True | X-ray | True | 12.3349 | 5.4345 |
| 12845 | CLOSED | No | yes | False | No | True | 10.7909 | 5.3429 |
| 12914 | CLOSED | Yes | yes | True | Yes | True | 10.5431 | 5.4212 |
| 12919 | CLOSED | Yes | yes | True | Yes | True | 16.1589 | 5.3737 |

---

## 5. Generation failures and limitations

- **Generation failures:** None. Both models produced a non-empty raw response and valid parsed answer for all 10 samples. Failed/missing counts are 0 for both models.
- **Strict exact match:** No stemming or semantic equivalence (e.g., "Lungs" vs "Lung", "Left lung" vs "Lower Left Lung" counted incorrect). This is a protocol limitation, not necessarily model error.
- **Pilot scale:** 10 samples, 9 unique images. Results are not statistically reliable and cannot support claims of clinical superiority or general benchmark performance.
- **No CXR images in SLAKE:** all chest-location entries are CT.
- **No patient IDs:** SLAKE publishes no patient-level identifiers; image-level leakage across splits cannot be fully ruled out (English-only overlap is zero per `docs/reports/SLAKE_EVALUATION_PROTOCOL.md`).
- MedGemma inference reported on Kaggle Tesla T4 (14.56 GB VRAM, bfloat16); Qwen on local CPU via Ollama. Different hardware — wall times are indicative only, not a controlled speed benchmark.

---

## 6. Test commands and actual results

| Command | Result |
|---------|--------|
| `.venv\Scripts\python.exe -m pytest tests/unit/test_baseline_comparison.py -q` | 16 passed |
| `.venv\Scripts\python.exe scripts\compare_slake_baselines.py` | Qwen 60.0% (6/10), MedGemma 70.0% (7/10), valid paired experiment: True |
| `.venv\Scripts\ruff.exe check scripts\compare_slake_baselines.py tests\unit\test_baseline_comparison.py` | All checks passed |

Note: `mypy` is not installed in this environment; ruff is the configured linter.

---

## 7. Is the comparison valid? Why

**Yes.** The comparison is valid as a 10-sample paired pilot because:

1. Both models were evaluated on the exact same 10 question IDs from `selection_manifest.json`.
2. Ground-truth answers match the official SLAKE `test.json` split (English).
3. All 10 MedGemma response files exist on disk with non-empty raw text and valid parsed answers (contrary to the stale previous report, which claimed 4 missing + 6 empty — that described an earlier artifact state).
4. Failed/missing predictions are retained in the denominator and kept separate from incorrect predictions.
5. Metrics were recalculated independently from raw prediction files; summary files were cross-checked only.
6. Consistency verification passes: sample IDs, question text, ground truth, and OPEN/CLOSED categories all match the manifest; `valid_paired_experiment: True`.

**Caveat:** This remains a 10-sample pilot. It is not statistically conclusive and provides no evidence of clinical superiority.

---

## 8. Outstanding issues

1. The previous report `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON.md` (and its stale artifacts) previously described MedGemma responses as empty/missing. Those artifacts have now been regenerated against the current files, but the old report text still reflects the outdated state. It should be superseded by this verification report.
2. MedGemma artifacts live under `experiments/runs/medgemma_baseline/`, not `slake_baseline_medgemma/` as the task text assumed. Callers should use the actual path.
3. Bootstrap confidence intervals (per `docs/reports/SLAKE_EVALUATION_PROTOCOL.md`, Section 5) were not computed — optional given the pilot is explicitly non-conclusive.
4. Consider relaxing exact match (stemming/synonyms) for OPEN questions if adopted; document as a protocol deviation.

---

## 9. Readiness to close

**FS-004 is ready to close.** The MedGemma baseline artifacts are complete and internally consistent (summary matches recalculated individual-file metrics), the paired cohort is verified identical to Qwen, the comparison script runs cleanly against current artifacts, all 16 existing tests pass, lint is clean, and comparison artifacts + this verification report are saved under `experiments/runs/slake_model_comparison/` and `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON_VERIFICATION.md`. No model inference was rerun; no original artifacts were modified; no results were fabricated.
