# FS-005 Phase 5: Measured Results (Paper Subset)

Date: 2026-11-18. Supersedes the "not computable" tables in the 10-question pilot reports.
All numbers come from `experiments/runs/fs005_phase5_full/evaluation/**` and `analysis.json`,
produced by `scripts/fs005_phase5_full_run.py` (`generate` on a Kaggle T4; `calibrate`,
`evaluate`, `analyze` on a laptop CPU) with `--subset paper`.

## What was run

- Models: Qwen2.5-VL-3B-Instruct and MedGemma-4B-IT, both 4-bit (bf16/fp16 compute), 256 new-token cap.
- Cohort: seeded random subset of the SLAKE English splits: 200 validation-fit, 80 validation-assess,
  250 test questions (77 images) per arm. Tracks B and C use the same questions. The full 945-question
  test cohort was **not** run (free GPU budget).
- Sampling: N = 5 at T = 0.7 (Conditions 2 and 3); Condition 1 is greedy, N = 1, Track C only.
- Gate weights and tau are fitted on validation-fit, assessed on validation-assess, frozen, then applied to test.
  Every arm selected w = (0, 0.05, 0.95); tau = 0.99 (Track B) and 0.04 (Track C).
- Scoring: strict normalized exact match; abstentions count as wrong in overall accuracy.
- Offline salvage (`scripts/fs005_phase5_salvage.py`): one fixed rule for all arms; no question excluded.
  Test questions needing salvage: MedGemma B 19, MedGemma C 15, Qwen B 86, Qwen C 92 (of 250).

## Test results (250 questions, 95% image-clustered bootstrap CI, 2,000 replicates)

| Arm / condition | Accuracy | Coverage | Selective acc. | AUROC | ECE |
|---|---|---|---|---|---|
| MedGemma B, Cond 2 | 0.392 (0.332-0.452) | 1.00 | 0.392 | 0.828 | 0.591 |
| MedGemma B, Cond 3 | 0.332 (0.275-0.389) | 0.492 | 0.675 (0.586-0.766) | n/a (see note) | 0.325 |
| MedGemma C, Cond 1 | 0.412 (0.357-0.468) | 1.00 | 0.412 | n/a | 0.412 |
| MedGemma C, Cond 2 | 0.424 (0.368-0.480) | 1.00 | 0.424 | 0.830 | 0.391 |
| MedGemma C, Cond 3 | 0.356 (0.298-0.415) | 0.496 | 0.718 (0.638-0.793) | n/a (see note) | 0.668 |
| Qwen B, Cond 2 | 0.404 (0.338-0.472) | 1.00 | 0.404 | 0.801 | 0.571 |
| Qwen B, Cond 3 | 0.164 (0.121-0.209) | 0.220 | 0.745 (0.646-0.841) | n/a (see note) | 0.255 |
| Qwen C, Cond 1 | 0.384 (0.313-0.450) | 1.00 | 0.384 | n/a | 0.384 |
| Qwen C, Cond 2 | 0.380 (0.311-0.451) | 1.00 | 0.380 | 0.793 | 0.356 |
| Qwen C, Cond 3 | 0.172 (0.128-0.216) | 0.216 | 0.796 (0.674-0.900) | n/a (see note) | 0.746 |

Grounding (Track B, Cond 2): USR_raw 0.090 (MedGemma) vs 0.433 (Qwen); pointing accuracy 0.88 vs 0.70;
sampling agreement 0.73 vs 0.62.

## Paired comparisons (accuracy difference, paired bootstrap CI, McNemar p)

| Comparison | Diff | 95% CI | McNemar p |
|---|---|---|---|
| MedGemma C: Cond 2 - Cond 1 | +0.012 | -0.004 to 0.030 | 0.37 |
| Qwen C: Cond 2 - Cond 1 | -0.004 | -0.043 to 0.038 | 1.00 |
| MedGemma C: Cond 3 - Cond 2 (all questions) | -0.068 | -0.098 to -0.041 | 0.0001 |
| Qwen C: Cond 3 - Cond 2 (all questions) | -0.208 | -0.272 to -0.150 | 1.5e-12 |
| MedGemma: Track B - Track C (Cond 2) | -0.032 | -0.071 to 0.004 | 0.099 |
| Qwen: Track B - Track C (Cond 2) | +0.024 | -0.020 to 0.065 | 0.40 |
| Track C Cond 2: MedGemma - Qwen | +0.044 | -0.012 to 0.104 | 0.14 |

McNemar is question-level and not adjusted for image clusters.

## Findings

1. Grounding and verification did not change accuracy measurably on this subset.
2. The fused confidence ranks correct answers well (Cond 2 AUROC 0.79-0.83) but is not calibrated as a probability.
3. Abstention raises accuracy on answered questions to 0.67-0.80 at the cost of coverage (about 0.5 for MedGemma, 0.22 for Qwen).
4. MedGemma-4B follows the JSON contract and cites valid regions much more reliably than Qwen2.5-VL-3B.

## Caveats and unresolved edge cases (documented, not hidden)

- Subset only (250 test questions on 77 images); intervals are wide; full cohort not run.
- Track C USR = 1.0 and pointing = 0.0 hold by construction and are not findings.
- Track B pointing is scored against the same oracle boxes the model was shown; it is not localizer quality.
- In Track B the localizer score rho is constant, and in Track C it is absent, so the fitted gate acts as an
  entropy threshold; the localizer term is untested. No separate signal ablation was run.
- `usr_filtered` equals `usr_raw` in the evaluator output; the cause was not investigated, so only `usr_raw` is reported.
- Condition 3 AUROC is NOT valid: the Condition 3 records label abstained questions as wrong, so that AUROC is circular and inflated (0.88-0.97). Use the Condition 2 AUROC (0.79-0.83). Condition 3 ECE is computed over answered questions only.
- Strict exact match understates correct free-text answers; a looser measure and a strict-only (no-salvage) sensitivity analysis are future work.
- Qwen2.5-VL-3B's much higher salvage rate (34-37% vs 6-8%) confounds the model comparison.
- Track A (detector on real CXR) is ineligible on SLAKE. No clinical validation was performed.

## Gate details verified from the code (added 2026-11-18)

- Semantic clustering in the runs: canonical fallback (lower-case + whitespace normalization, exact match); the embedding clusterer was not used (`force_fallback=True`). Paraphrases are therefore different clusters.
- rho: Track B assigns every oracle box score 1.0 (constant). Track C claims are `present_ungrounded`, so rho contributes 0 with no rescaling (rescaling applies only to absent/uncertain claims).
- Calibration: weights on a 0.05 simplex grid, tau 0.00-1.00 step 0.01, objective = minimize selective risk on val_fit subject to coverage >= 0.10; AUROC and ECE are diagnostics. ECE: 10 equal-width bins over answered questions.
- Pointing accuracy: at least one cited present-claim box with IoU >= 0.5 against an annotated box. USR/pointing are evaluation-only; the verifier is structural (checks that the cited id exists) and does not test semantic support.

## Sensitivity analyses (`scripts/fs005_phase5_sensitivity.py`, CPU only)

Strict-only (questions whose generation passed the JSON contract with no salvage; same frozen gate):

| Arm | Strict n | Cond 2 acc (all / strict) | Cond 3 sel. acc (all / strict) | Cond 3 coverage (all / strict) |
|---|---|---|---|---|
| MedGemma B | 231 | 0.392 / 0.403 | 0.675 / 0.678 | 0.492 / 0.498 |
| MedGemma C | 235 | 0.424 / 0.447 | 0.718 / 0.727 | 0.496 / 0.515 |
| Qwen B | 164 | 0.404 / 0.451 | 0.745 / 0.838 | 0.220 / 0.226 |
| Qwen C | 158 | 0.380 / 0.430 | 0.796 / 0.750 | 0.216 / 0.253 |

Condition 1 (Track C) strict n: MedGemma 246 (0.415 vs 0.412), Qwen 247 (0.389 vs 0.384).

Signal ablation (AUROC with Condition 2 correctness): agreement-only 0.827/0.830/0.795/0.799, entropy-only 0.828/0.830/0.801/0.793, fused 0.828/0.830/0.801/0.793 (MG B, MG C, Q B, Q C). Accuracy at the gate's coverage is identical for all three. The fused gate adds nothing over a single signal; the localizer term is constant, so it contributes nothing.
