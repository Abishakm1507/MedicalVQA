# FS-005 qwen2.5-vl-3b track C cond 3

## 1. Configuration

- model: qwen2.5-vl-3b
- track: C
- condition: 3
- w: [0.0, 0.05, 0.95]
- tau: 0.04
- n_samples: 5
- temperature: 0.7
- seed: 42

## 2. Aggregate Metrics

| Metric | Value |
|---|---:|
| n_samples | 250 |
| n_positive_claims_total | 1348 |
| accuracy | 0.172 |
| open_accuracy | 0.11564625850340136 |
| closed_accuracy | 0.2524271844660194 |
| usr_raw | 1.0 |
| usr_filtered | 1.0 |
| pointing_accuracy | 0.0 |
| sampling_agreement | 0.6090000000000003 |
| normalized_entropy | 0.5158526092742213 |
| ece | 0.7462962962962963 |
| selective_accuracy | 0.7962962962962963 |
| coverage | 0.216 |
| auroc | 0.9734299516908212 |

## 3. Exclusions

- Weights and tau frozen from validation FIT split only.
- Track B pointing accuracy is computed against the same boxes the model was shown (oracle regions); it is not a localization-quality measure.
- Condition 1 is a greedy single sample with the same prompt contract.
- Test questions with a usable generation: 250 of 250. Questions with unparseable samples were salvaged offline with one fixed rule (answer kept if present; only complete schema-valid findings kept; question kept if >=3 of 5 samples usable): {'records_with_failures': 92, 'records_salvaged': 92, 'samples_salvaged': 117, 'samples_dropped': 1} so far in this run.

## 4. Accuracy-vs-Coverage Curve (first 5 and last 5 points)

| tau | coverage | selective_accuracy |
|---|---:|---:|
| 0.0 | 0.8560 | 0.2009 |
| 0.01 | 0.6840 | 0.2515 |
| 0.02 | 0.6000 | 0.2867 |
| 0.03 | 0.3760 | 0.4574 |
| 0.04 | 0.2160 | 0.7963 |
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
