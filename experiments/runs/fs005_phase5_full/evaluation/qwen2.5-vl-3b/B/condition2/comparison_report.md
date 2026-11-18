# FS-005 qwen2.5-vl-3b track B cond 2

## 1. Configuration

- model: qwen2.5-vl-3b
- track: B
- condition: 2
- w: [0.0, 0.05, 0.95]
- tau: 0.99
- n_samples: 5
- temperature: 0.7
- seed: 42

## 2. Aggregate Metrics

| Metric | Value |
|---|---:|
| n_samples | 250 |
| n_positive_claims_total | 1433 |
| accuracy | 0.404 |
| open_accuracy | 0.2857142857142857 |
| closed_accuracy | 0.5728155339805825 |
| usr_raw | 0.4333565945568737 |
| usr_filtered | 0.4333565945568737 |
| pointing_accuracy | 0.7 |
| sampling_agreement | 0.6146000000000003 |
| normalized_entropy | 0.5052353457946783 |
| ece | 0.570738232710264 |
| selective_accuracy | 0.404 |
| coverage | 1.0 |
| auroc | 0.8011495780450528 |

## 3. Exclusions

- Weights and tau frozen from validation FIT split only.
- Track B pointing accuracy is computed against the same boxes the model was shown (oracle regions); it is not a localization-quality measure.
- Condition 1 is a greedy single sample with the same prompt contract.
- Test questions with a usable generation: 250 of 250. Questions with unparseable samples were salvaged offline with one fixed rule (answer kept if present; only complete schema-valid findings kept; question kept if >=3 of 5 samples usable): {'records_with_failures': 86, 'records_salvaged': 86, 'samples_salvaged': 117, 'samples_dropped': 1} so far in this run.

## 4. Accuracy-vs-Coverage Curve (first 5 and last 5 points)

| tau | coverage | selective_accuracy |
|---|---:|---:|
| 0.0 | 1.0000 | 0.4040 |
| 0.01 | 1.0000 | 0.4040 |
| 0.02 | 1.0000 | 0.4040 |
| 0.03 | 1.0000 | 0.4040 |
| 0.04 | 1.0000 | 0.4040 |
| 0.96 | 0.6800 | 0.5471 |
| 0.97 | 0.6000 | 0.5733 |
| 0.98 | 0.4120 | 0.6699 |
| 0.99 | 0.2200 | 0.7455 |
| 1.0 | 0.2200 | 0.7455 |
| ... (91 more points) | | |

## 5. Notes

- USR metrics computed over positive claims only (absence and uncertain claims excluded per ADR 0004 Option 3).
- Pointing accuracy denominator excludes samples with no ground-truth regions.
- ECE, selective accuracy, coverage, and AUROC computed over accepted (in-coverage) samples only.
- Abstained samples counted as incorrect in overall accuracy but excluded from coverage/selective accuracy.
- No model inference was run in this phase; metrics computed from per-sample records produced by Phases 1-2.
