# FS-005 Phase 3 Final Closure Report — Evaluation Metrics and Artifact Generation

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005 (Grounding and Uncertainty-Aware Answering)  
**Phase:** Phase 3 — Evaluation Metrics and Artifact Generation  
**Document ID:** `docs/reports/FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md`  
**Status:** Closure report — Phase 3 complete, ready for Phase 4  
**Scope:** Documentation-only corrections to technical design and Phase 3 reports  
**Implementation status:** No Python code changes in this closure phase

---

## 1. Files Changed

| File | Change |
|------|--------|
| `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md` | Updated AUROC definition (Section 6) to full eligible population, independent of abstention threshold |
| `docs/reports/FS-005_PHASE_3_IMPLEMENTATION_REPORT.md` | Updated AUROC section (3.11) to reference updated design; added pointing accuracy limitation note; updated test counts and results |
| `docs/reports/FS-005_PHASE_3_CORRECTIVE_REVIEW.md` | Updated Section 9 (no discrepancy remains — technical design now matches implementation); added pointing accuracy limitation; updated test counts and results |
| `docs/reports/FS-005_PHASE_3_FINAL_CLOSURE_REPORT.md` | **Created** — this report |

**No Python implementation files were modified.** No tests were changed.

---

## 2. Final AUROC Definition

**Population:** All eligible samples — `sampling_valid == True` AND `evidence_state != "uncertain"`. Independent of the abstention threshold $\tau$.

**ROC-AUC:** Mann-Whitney U statistic of confidence $c$ predicting binary correctness.

**Return values:**
- `None` if no eligible samples
- `None` if only one correctness class present in eligible population
- Mann-Whitney U statistic in [0, 1] otherwise

**Rationale:** AUROC evaluates confidence-score ranking across eligible predictions. Restricting it to accepted samples ($c \ge \tau$) would make AUROC dependent on the abstention threshold, which is not a meaningful property of the confidence score itself.

**ECE definition:** Remains accepted-sample-only conditional calibration over [0,1] with M=10 equal-width bins. Abstained samples excluded.

---

## 3. Pointing Accuracy Limitation

**Aggregation:** Sample-level — `pointing_correct=True` if at least one claim in the sample has max IoU >= 0.5 against any ground-truth region.

**Limitation:** Sample-level success can conceal incorrectly grounded additional claims within the same sample. If a sample has multiple claims where one meets IoU >= 0.5 against a GT region but another does not, `pointing_correct=True` at the sample level regardless of the incorrectly grounded claim. Sample-level pointing accuracy does not capture per-claim grounding errors.

---

## 4. Verification Results

**Technical design consistency:** All three documentation files now consistently define AUROC over the full eligible population, independent of the abstention threshold. The technical design Section 6, the Phase 3 implementation report Section 3.11, and the Phase 3 corrective review Section 9 all agree.

**ECE consistency:** ECE remains accepted-sample-only conditional calibration across all three documents.

**Pointing accuracy:** Sample-level aggregation with max-IoU-over-GT-boxes matching is consistent across all three documents, with the limitation (sample-level success can conceal incorrectly grounded additional claims) documented.

**Tests:** No Python test changes were made in this closure phase. All previously verified tests remain passing:
- Phase 3 metrics tests: 63 passed
- Phase 2 tests: 51 passed
- Phase 1 tests: 47 passed
- Baseline comparison tests: 16 passed
- Schema tests: 4 passed
- **Total: 181 passed**

**Ruff lint:** All checks passed on `evaluation/` directory.

---

## 5. Phase Status

**Phase 3 — Complete.** Evaluation metrics harness and artifact generation implemented and verified:
- `evaluation/metrics.py` — all 9 metrics + `SampleRecord` + `AggregateMetrics`
- `evaluation/runner.py` — `EvaluationRunner` + artifact writers (JSON, CSV, Markdown)
- `tests/unit/test_evaluation_metrics_phase3.py` — 63 deterministic unit tests

**Phase 4 — Not started.** Full pipeline integration, experiment tracks (A/B/C), API integration — no code written.

**Phase 5 — Not started.** Validation experiments, research-result reporting — no code written.

**Model inference — Not run.** No Qwen, MedGemma, MedSAM, or other VLM inference executed. No model weights downloaded. No datasets accessed beyond synthetic test fixtures.

**FS-001 through FS-004 — Not modified.**

---

## 6. Recommendation

FS-005 Phase 3 is ready to close. The technical design, implementation report, and corrective review are mutually consistent on AUROC definition, ECE definition, pointing accuracy aggregation, and all other metric semantics. Phase 4 may proceed upon review approval.
