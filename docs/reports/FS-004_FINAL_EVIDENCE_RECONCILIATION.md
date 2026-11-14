# FS-004 Final Evidence Reconciliation — Qwen2.5-VL vs MedGemma Paired Pilot

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-004 (baseline comparison)  
**Status:** Reconciliation complete — paired outcome bug fixed, checkpoint verified, reports corrected.

---

## 1. Files inspected

| File | Role |
|------|------|
| `experiments/runs/slake_model_comparison/comparison_metrics.json` | Machine-readable paired metrics (regenerated after fix) |
| `experiments/runs/slake_model_comparison/comparison_report.md` | Human-readable comparison report (regenerated after fix) |
| `experiments/runs/slake_model_comparison/comparison_summary.csv` | Summary comparison table (regenerated after fix) |
| `experiments/runs/slake_model_comparison/per_sample_comparison.csv` | Per-sample paired comparison (regenerated after fix) |
| `experiments/runs/slake_baseline_ollama/baseline_summary.json` | Qwen baseline summary (10 samples, source of truth for Qwen) |
| `experiments/runs/slake_baseline_ollama/q{id}_response.txt` ×10 | Qwen individual raw responses |
| `experiments/runs/slake_baseline_ollama/selection_manifest.json` | Cohort manifest: qids, questions, GT, answer type |
| `experiments/runs/medgemma_baseline/baseline_summary.json` | MedGemma baseline summary (10 samples, source of truth for MedGemma) |
| `experiments/runs/medgemma_baseline/q{id}_response.json` ×10 | MedGemma individual raw responses |
| `notebooks/kaggle_medgemma_baseline.ipynb` | MedGemma baseline notebook — checkpoint configuration |
| `vlm/medgemma_client.py` | MedGemma client — default model_id |
| `artifacts/kaggle_upload/MedicalVQA_kaggle_bundle/vlm/medgemma_client.py` | Kaggle bundle MedGemma client — default model_id |
| `scripts/compare_slake_baselines.py` | Comparison script — `_paired()` function fixed |
| `tests/unit/test_baseline_comparison.py` | Regression tests — 16 passed |
| `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON_VERIFICATION.md` | Prior verification report — corrected |

**Not modified:** `experiments/runs/slake_baseline_ollama/`, `experiments/runs/medgemma_baseline/`, `data/SLAKE/`, FS-001, FS-002, FS-003.

---

## 2. Issue 1 — Paired outcome inconsistency (resolved)

### Root cause

The `_paired()` function in `scripts/compare_slake_baselines.py` had a coverage gap: when one model was correct and the other incorrect on the same sample, and neither prediction was failed/missing, none of the `if/elif` branches matched. The sample fell through all conditions and was omitted from every paired list.

QID 12845 was the affected sample:
- Qwen: GT="No", pred="yes", correct=false, status="ok" → Qwen INCORRECT
- MedGemma: GT="No", pred="No", correct=true, status="ok" → MedGemma CORRECT

The previous `comparison_metrics.json` listed only 9 of 10 samples across the four paired categories (6 both_correct + 3 both_incorrect = 9), omitting QID 12845 entirely. The prior verification report also claimed "Only MedGemma correct: (none)" and "MedGemma matches Qwen on every sample" — both incorrect.

### Fix

Added two `elif` branches to `_paired()` in `scripts/compare_slake_baselines.py:495-498`:
- `q_ok and not m_ok` → `qwen_only_correct`
- `m_ok and not q_ok` → `medgemma_only_correct`

These branches are reached only when the earlier failed/missing conditions do not match, so they correctly capture the "one correct, one incorrect, neither failed" case.

### Corrected per-question paired outcomes

Recalculated directly from the underlying per-sample prediction artifacts (baseline_summary.json files and individual response files), verified against `data/SLAKE/test.json`:

| QID | Category | Ground truth | Qwen pred | Qwen correct | MedGemma pred | MedGemma correct | Paired outcome |
|-----|----------|--------------|-----------|--------------|---------------|------------------|----------------|
| 11981 | OPEN | Chest | chest | True | Chest | True | Both correct |
| 12073 | CLOSED | Yes | Yes | True | Yes | True | Both correct |
| 12111 | CLOSED | Yes | Yes | True | Yes | True | Both correct |
| 12129 | OPEN | 0 | 2 | False | 1 | False | Both incorrect |
| 12312 | OPEN | Lower Left Lung | Right Lung | False | Left lung | False | Both incorrect |
| 12344 | OPEN | Lung | lungs | False | Lungs | False | Both incorrect |
| 12384 | OPEN | X-Ray | X-ray | True | X-ray | True | Both correct |
| 12845 | CLOSED | No | yes | False | No | True | **MedGemma only correct** |
| 12914 | CLOSED | Yes | yes | True | Yes | True | Both correct |
| 12919 | CLOSED | Yes | yes | True | Yes | True | Both correct |

### Recalculated metrics

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

### Corrected paired outcome counts

| Category | Count | QIDs |
|----------|-------|------|
| Both correct | 6 | 11981, 12073, 12111, 12384, 12914, 12919 |
| Both incorrect | 3 | 12129, 12312, 12344 |
| Only Qwen correct | 0 | (none) |
| Only MedGemma correct | 1 | 12845 |
| Both failed or missing | 0 | (none) |

Check: 6 + 3 + 0 + 1 = 10 samples accounted for. Qwen correct = 6 + 0 = 6 ✓. MedGemma correct = 6 + 1 = 7 ✓.

### Cohort verification

- Identical 10 question IDs: PASS — both models evaluated on the same 10 qids from `selection_manifest.json`
- Identical questions: PASS — question text matches manifest for all 10 qids
- Identical ground-truth answers: PASS — matches manifest and `data/SLAKE/test.json` (all 10 qids, `q_lang=en`, in test split)
- Same evaluation rules: PASS — same normalization, exact-match correctness, denominator = 10
- No substitutions / new cohort: PASS
- `valid_paired_experiment`: True

---

## 3. Issue 2 — Exact checkpoint identification (resolved)

### Determination

The experiment used **`google/medgemma-4b-it`** (not `google/medgemma-1.5-4b-it`).

### Evidence

| Source | Location | Evidence |
|--------|----------|----------|
| `baseline_summary.json` | `experiments/runs/medgemma_baseline/baseline_summary.json:3` | `"model_id": "google/medgemma-4b-it"` |
| Notebook cell | `notebooks/kaggle_medgemma_baseline.ipynb` | `MODEL_ID = "google/medgemma-4b-it"` |
| Notebook comment | `notebooks/kaggle_medgemma_baseline.ipynb` | `# NOTE: must be the model you were granted access to (NOT medgemma-1.5-4b-it)` |
| `medgemma_client.py` | `vlm/medgemma_client.py:37` | Default `model_id: str = "google/medgemma-4b-it"` |
| `medgemma_client.py` (kaggle bundle) | `artifacts/kaggle_upload/.../vlm/medgemma_client.py:37` | Default `model_id: str = "google/medgemma-4b-it"` |
| Comparison script | `scripts/compare_slake_baselines.py:818` | `"MedGemma-4B (google/medgemma-4b-it)"` |

### Unresolved status

None. The checkpoint is confirmed as `google/medgemma-4b-it`. No evidence of `google/medgemma-1.5-4b-it` was found in any experiment artifact related to the 10-sample baseline. The notebook explicitly rules out the 1.5 variant with a comment.

**Scope note:** This determination is based only on the FS-004 baseline experiment artifacts (notebook, baseline_summary.json, individual response files, comparison script). It does not cover any separate debugging or smoke-test experiments.

---

## 4. Remaining limitations

1. **Pilot scale:** 10 samples, 9 unique images. Results are not statistically reliable and cannot support claims of clinical superiority or general benchmark performance.
2. **Strict exact match:** No stemming or semantic equivalence (e.g., "Lungs" vs "Lung", "Left lung" vs "Lower Left Lung" counted incorrect). This is a protocol limitation, not necessarily model error.
3. **Different hardware:** MedGemma ran on Kaggle Tesla T4 (14.56 GB VRAM, bfloat16); Qwen ran on local CPU via Ollama. Wall times are indicative only, not a controlled speed benchmark.
4. **No CXR images in SLAKE:** all chest-location entries are CT.
5. **No patient IDs:** SLAKE publishes no patient-level identifiers; image-level leakage across splits cannot be fully ruled out (English-only overlap is zero per `docs/reports/SLAKE_EVALUATION_PROTOCOL.md`).
6. **Bootstrap confidence intervals:** Not computed (optional per `docs/reports/SLAKE_EVALUATION_PROTOCOL.md`, Section 5, given the pilot is explicitly non-conclusive).
7. **Consider relaxing exact match:** Stemming/synonyms for OPEN questions could be adopted as a documented protocol deviation.

---

## 5. Final FS-004 closure recommendation

**FS-004 is ready to close.**

Evidence summary:
- MedGemma baseline artifacts are complete: all 10 response files present, non-empty, valid parsed answers, ground truth verified against `data/SLAKE/test.json`.
- Recalculated MedGemma metrics match `baseline_summary.json` exactly (7 correct / 3 incorrect / 0 failed, accuracy 0.70, OPEN 0.40, CLOSED 1.00).
- Paired cohort verified identical to Qwen (same 10 qids, questions, ground truth, 5 OPEN / 5 CLOSED).
- `_paired()` bug fixed: QID 12845 correctly classified as MedGemma-only correct; all 10 samples now accounted for.
- Checkpoint confirmed as `google/medgemma-4b-it` from multiple experiment artifacts.
- Comparison script runs cleanly; ruff lint passes; all 16 unit tests pass.
- Original prediction artifacts (Qwen and MedGemma) were not modified.
- No model inference was rerun; no results were fabricated.
- No FS-001, FS-002, or FS-003 files were modified.
- FS-005 was not implemented.

Corrected deliverables:
1. `docs/reports/FS-004_QWEN_MEDGEMMA_COMPARISON_VERIFICATION.md` — corrected paired outcomes, resolved checkpoint status
2. `docs/reports/FS-004_FINAL_EVIDENCE_RECONCILIATION.md` — this report
3. `experiments/runs/slake_model_comparison/` — regenerated comparison_metrics.json, comparison_report.md, comparison_summary.csv, per_sample_comparison.csv
4. `scripts/compare_slake_baselines.py` — `_paired()` function fixed
