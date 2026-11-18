# FS-005 medgemma-4b track B cond 3

## 1. Configuration

- model: medgemma-4b
- track: B
- condition: 3
- w: [0.0, 0.05, 0.95]
- tau: 0.99
- n_samples: 5
- temperature: 0.7
- seed: 42

## 2. Aggregate Metrics

| Metric | Value |
|---|---:|
| n_samples | 250 |
| n_positive_claims_total | 1819 |
| accuracy | 0.332 |
| open_accuracy | 0.1836734693877551 |
| closed_accuracy | 0.5436893203883495 |
| usr_raw | 0.08960967564595931 |
| usr_filtered | 0.08960967564595931 |
| pointing_accuracy | 0.88 |
| sampling_agreement | 0.7304000000000002 |
| normalized_entropy | 0.3493794247730148 |
| ece | 0.3252032520325203 |
| selective_accuracy | 0.6747967479674797 |
| coverage | 0.492 |
| auroc | 0.8802395209580839 |

## 3. Exclusions

- Weights and tau frozen from validation FIT split only.
- Track B pointing accuracy is computed against the same boxes the model was shown (oracle regions); it is not a localization-quality measure.
- Condition 1 is a greedy single sample with the same prompt contract.
- Test questions with a usable generation: 250 of 250. Questions with unparseable samples were salvaged offline with one fixed rule (answer kept if present; only complete schema-valid findings kept; question kept if >=3 of 5 samples usable): {'records_with_failures': 19, 'records_salvaged': 19, 'samples_salvaged': 44, 'samples_dropped': 0} so far in this run.

## 4. Accuracy-vs-Coverage Curve (first 5 and last 5 points)

| tau | coverage | selective_accuracy |
|---|---:|---:|
| 0.0 | 1.0000 | 0.3320 |
| 0.01 | 1.0000 | 0.3320 |
| 0.02 | 1.0000 | 0.3320 |
| 0.03 | 1.0000 | 0.3320 |
| 0.04 | 1.0000 | 0.3320 |
| 0.96 | 0.7360 | 0.4511 |
| 0.97 | 0.7080 | 0.4689 |
| 0.98 | 0.6000 | 0.5533 |
| 0.99 | 0.4920 | 0.6748 |
| 1.0 | 0.4920 | 0.6748 |
| ... (91 more points) | | |

## 5. Notes

- USR metrics computed over positive claims only (absence and uncertain claims excluded per ADR 0004 Option 3).
- Pointing accuracy denominator excludes samples with no ground-truth regions.
- ECE, selective accuracy, coverage, and AUROC computed over accepted (in-coverage) samples only.
- Abstained samples counted as incorrect in overall accuracy but excluded from coverage/selective accuracy.
- No model inference was run in this phase; metrics computed from per-sample records produced by Phases 1-2.
