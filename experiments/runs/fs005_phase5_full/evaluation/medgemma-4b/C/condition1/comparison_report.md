# FS-005 medgemma-4b track C cond 1

## 1. Configuration

- model: medgemma-4b
- track: C
- condition: 1
- w: [0.0, 0.05, 0.95]
- tau: 0.04
- n_samples: 5
- temperature: 0.7
- seed: 42

## 2. Aggregate Metrics

| Metric | Value |
|---|---:|
| n_samples | 250 |
| n_positive_claims_total | 0 |
| accuracy | 0.412 |
| open_accuracy | 0.23809523809523808 |
| closed_accuracy | 0.6601941747572816 |
| usr_raw | None |
| usr_filtered | None |
| pointing_accuracy | 0.0 |
| sampling_agreement | 0.0 |
| normalized_entropy | 0.0 |
| ece | 0.412 |
| selective_accuracy | 0.412 |
| coverage | 1.0 |
| auroc | None |

## 3. Exclusions

- Weights and tau frozen from validation FIT split only.
- Track B pointing accuracy is computed against the same boxes the model was shown (oracle regions); it is not a localization-quality measure.
- Condition 1 is a greedy single sample with the same prompt contract.
- Test questions with a usable generation: 250 of 250. Questions with unparseable samples were salvaged offline with one fixed rule (answer kept if present; only complete schema-valid findings kept; question kept if >=3 of 5 samples usable): {'records_with_failures': 15, 'records_salvaged': 15, 'samples_salvaged': 23, 'samples_dropped': 0} so far in this run.

## 4. Accuracy-vs-Coverage Curve (first 5 and last 5 points)

| tau | coverage | selective_accuracy |
|---|---:|---:|
| 0.0 | 0.0000 | 0.0000 |
| 0.01 | 0.0000 | 0.0000 |
| 0.02 | 0.0000 | 0.0000 |
| 0.03 | 0.0000 | 0.0000 |
| 0.04 | 0.0000 | 0.0000 |
| 0.96 | 0.0000 | 0.0000 |
| 0.97 | 0.0000 | 0.0000 |
| 0.98 | 0.0000 | 0.0000 |
| 0.99 | 0.0000 | 0.0000 |
| 1.0 | 0.0000 | 0.0000 |
| ... (91 more points) | | |

## 5. Notes

- USR metrics computed over positive claims only (absence and uncertain claims excluded per ADR 0004 Option 3).
- Pointing accuracy denominator excludes samples with no ground-truth regions.
- ECE, selective accuracy, coverage, and AUROC computed over accepted (in-coverage) samples only.
- Abstained samples counted as incorrect in overall accuracy but excluded from coverage/selective accuracy.
- No model inference was run in this phase; metrics computed from per-sample records produced by Phases 1-2.
