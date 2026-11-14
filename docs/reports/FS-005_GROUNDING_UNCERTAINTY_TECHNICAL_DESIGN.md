# FS-005 — Grounding and Uncertainty-Aware Answering: Technical Design

**Project:** Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images  
**Feature specification:** FS-005  
**Document ID:** `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md`  
**Status:** Design only — no implementation code written  
**Research paper:** *"Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models"* (Abisha K M, Anusiya S, Bhavya Sree P R, Dirisala Kavya)

---

## 1. Existing Architecture Assessment

### 1.1 Current implementation state (verified facts)

The following components already exist and are verified in prior phases:

| Component | Path | Status |
|-----------|------|--------|
| Dataset loaders (SLAKE, RSNA, VinDr-CXR, VQA-RAD) | `datasets/` | FS-001 approved; `SlakeDatasetLoader` verified compatible with actual `data/SLAKE/` layout |
| Region detector (base + mock) | `localization/detector.py` | `BaseRegionDetector`, `MockRegionDetector`; FS-002 design approved |
| Region segmenter (base + mock + MedSAM) | `localization/segmenter.py` | `MedSAMRegionSegmenter` validated checkpoint + RLE encoding; FS-003 approved |
| Postprocessing (score filter, NMS, CandidateBox build) | `localization/postprocessing.py` | Strict anti-fabrication rule: `R = ∅` when no detections |
| Detector config | `localization/config.py` | `DetectorConfig` with `score_threshold`, `nms_iou_threshold`, `max_k` (≤7) |
| VLM client interface | `vlm/base.py` | `VLMClient` ABC with `generate()`, `sample_n()`, `generate_structured()` |
| Qwen2.5-VL-3B client | `vlm/qwen_client.py` | `QwenVLMClient` (HF Transformers, 4-bit optional) |
| MedGemma-4B client | `vlm/medgemma_client.py` | `MedGemmaVLMClient` (confirmed `google/medgemma-4b-it` per FS-004 reconciliation) |
| Ollama Qwen client | `vlm/ollama_client.py` | `OllamaVLMClient` (used in FS-004 baseline) |
| Core schemas | `core/schemas.py` | `MedicalVQASample`, `CandidateBox`, `VLMClaim`, `VLMStructuredOutput`, `EvaluationTrack` (A/B/C), `ConfidenceGateResult` |
| Remote GPU worker | `api/remote_worker.py` | `RemoteModelWorker` — sequential model loading, VRAM-safe |
| Claim verification spec | `docs/specifications/FS-007-claim-verification-engine.md` | Approved; `verification/claim_verifier.py` not yet implemented |
| Confidence gate spec | `docs/specifications/FS-008-confidence-gating-abstention.md` | Approved post-audit; `verification/confidence_gate.py` not yet implemented |
| Metrics spec | `docs/specifications/FS-010-evaluation-metrics-harness.md` | Approved; `evaluation/metrics.py` not yet implemented |
| API service spec | `docs/specifications/FS-011-fastapi-backend-service.md` | Approved; `api/` FastAPI app not yet implemented |
| Testing strategy | `docs/architecture/testing_strategy.md` | ≥85% coverage, deterministic mocks, no skipped tests |

### 1.2 Audit-corrected design constraints (from `docs/reports/PHASE_0_INDEPENDENT_AUDIT.md`)

The following audit findings are binding constraints on this design:

1. **No synthetic boxes.** When the detector returns zero candidates, `R = ∅`. No quadrant/salience fallback boxes (ISSUE-01).
2. **Semantic equivalence clustering.** Agreement $a$ and entropy $H_n$ use Kuhn et al. (2023) semantic clustering, never naive string matching (ISSUE-04).
3. **No hardcoded weights/threshold.** $(w_1, w_2, w_3, \tau)$ are empirical parameters fitted on held-out SLAKE validation data (ISSUE-05).
4. **Sequential model loading.** Qwen, MedGemma, and MedSAM are never resident in GPU VRAM simultaneously (ISSUE-06).
5. **Track separation.** Track A (predicted CXR detector) is CXR-only; Track B (oracle GT boxes) covers all SLAKE modalities; Track C (ungrounded) is the answer-only baseline (ISSUE-07).
6. **Patient-level grouping.** RSNA and VinDr-CXR splits group by `patient_id` (ISSUE-08).
7. **N=5 benchmark standard.** Confidence-gate benchmarking uses $N=5$; $N=3$ requires a separate calibrated operating point (ISSUE-09).
8. **Negative claim policy is OPEN.** Absence-claim handling is an unresolved research question (ADR 0004, ISSUE-02). This design does not invent a negative-claim strategy.
9. **Licensing.** The root LICENSE is MIT; Ultralytics YOLOv8 (AGPL-3.0) is excluded. Detector must use Apache 2.0 / BSD-3-Clause weights (RT-DETR or Torchvision Faster R-CNN) (ISSUE-03).

### 1.3 Gap analysis — what FS-005 must add

The following capabilities exist only as specifications or partial skeletons and require full design (this document):

- Claim extraction from raw VLM structured output (builds on `FS-006` Pydantic validation)
- Claim-to-region association with IoU verification against ground truth
- Unsupported-claim detection and filtering (raw vs. filtered USR)
- Semantic agreement sampling and normalized entropy computation
- Confidence calibration (weight fitting, threshold sweep)
- Abstention / selective prediction decision logic
- Evaluation metrics harness (USR, pointing accuracy, ECE, selective accuracy, coverage, accuracy-vs-coverage curves)
- Experiment artifact generation (per-sample records, summary JSON, CSV, Markdown)

---

## 2. Research Objectives and Scope

### 2.1 Core research objective

**The system can only claim what it can point to.** Every clinical visual claim emitted by a VLM must be associated with evidence from a specific image region ($R_1 \dots R_k$). Unsupported claims must be identified, filtered, or abstained from rather than presented to the user.

This implements the paper's grounding axiom (Section VI-B): *"A finding is kept only if its region ID names a region in R, and the rest are dropped before anything is shown to the user."*

### 2.2 Scope — in scope

- Claim extraction from structured VLM JSON responses (`VLMStructuredOutput`)
- Claim-to-region association against detector-proposed candidate boxes ($R$)
- Region evidence verification via IoU against ground-truth annotations (≥0.5 threshold)
- Unsupported claim detection, filtering, and reporting
- Semantic equivalence clustering for self-consistency agreement and entropy
- Confidence score fusion: $c = w_1 a + w_2 (1 - H_n) + w_3 \rho$
- Confidence-gate abstention (selective prediction)
- Evaluation metrics: USR_raw, USR_filtered, pointing accuracy, sampling agreement, normalized semantic entropy, ECE, selective accuracy, coverage, accuracy-vs-coverage curves
- Experiment artifact generation (JSON, CSV, Markdown)
- Three experimental tracks (A, B, C) and three answer conditions (free-form, grounded-no-gate, grounded-with-gate)
- Dataset eligibility, leakage controls, split handling
- Hardware-aware execution (RTX 2050 4 GB, Colab T4 15.3 GB)

### 2.3 Scope — explicitly out of scope

- Training new detection or segmentation models (detector uses fine-tuned medical checkpoints or mock)
- Clinical validation or deployment
- Modifying FS-001 through FS-004 code or artifacts
- Implementing FS-005 code (this document is design only)
- Paid APIs or credit-card services
- Synthetic region generation ($R = ∅$ is preserved)
- Fixed/hardcoded confidence weights or thresholds

### 2.4 Separation of concerns: tracks vs. conditions

These two taxonomies are orthogonal and must never be conflated:

| Taxonomy | Dimension | Values |
|----------|-----------|--------|
| Experimental track | Source of candidate regions / evaluation data eligibility | Track A (predicted CXR), Track B (oracle GT boxes), Track C (ungrounded) |
| Answer condition | How the VLM answer is produced and post-processed | Condition 1 (free-form), Condition 2 (grounded + verification, no abstention), Condition 3 (grounded + verification + confidence-gate abstention) |

Each track × condition combination produces a separate result matrix.

---

## 3. Proposed Architecture and Data Flow

### 3.1 High-level data flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: Medical image + question + dataset sample metadata               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1 — Region proposal (Track A/B only)                               │
│  localization/detector.py → CandidateBox[] (R1..Rk, or ∅)               │
│  Track B: ground-truth boxes promoted to CandidateBox[]                 │
│  Track C: skipped (no regions)                                          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2 — Segmentation (optional, mask for visual overlay only)          │
│  localization/segmenter.py → SegmentationResult[] (one per box)         │
│  MedSAM is a segmenter, NOT a detector — boxes come from STEP 1         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3 — Visual marking (Set-of-Mark overlay R1..Rk on image copy)      │
│  localization/visual_marker.py (FS-004)                                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4 — VLM inference (Condition 1/2/3)                                │
│  vlm/base.py → VLMStructuredOutput{answer, findings[VLMClaim]}          │
│  N=5 samples at T=0.7 for Conditions 2/3 (entropy/agreement)            │
│  N=1 sample at T=0.0 for Condition 1 (free-form accuracy)               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5 — Claim verification (FS-007)                                    │
│  verification/claim_verifier.py                                         │
│  - region_id ∈ {R1..Rk, NONE} check → is_grounded                       │
│  - IoU(box_cited, box_gt) ≥ 0.5 → is_supported                          │
│  - Unsupported claims filtered → USR_raw, USR_filtered                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6 — Confidence gate (FS-008) — Condition 3 only                    │
│  verification/confidence_gate.py                                        │
│  - Semantic clustering of N answers → agreement a                       │
│  - Normalized entropy H_n over cluster distribution                     │
│  - ρ from localizer score of cited region (None for NONE claims)        │
│  - c = w1·a + w2·(1-Hn) + w3·ρ                                          │
│  - Decision: accept if c ≥ τ, else abstain ("uncertain")                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 7 — Metrics & artifact generation (FS-010)                         │
│  evaluation/metrics.py → per-sample + aggregate metrics JSON/CSV/MD     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component boundaries and non-modification

FS-005 designs new modules under `verification/` and `evaluation/` without modifying existing FS-001–FS-004 files. The existing `core/schemas.py` dataclasses and Pydantic models are extended with new verification/evaluation dataclasses (additive only). `vlm/base.py` interface is reused unchanged.

---

## 4. Component Specifications

### 4.1 Claim extraction (from structured VLM responses)

**Responsibilities:** Parse a `VLMStructuredOutput` (already Pydantic-validated by FS-006) into a canonical internal claim record, extracting answer text, claim strings, region IDs, statuses, and confidences. Detect malformed/empty outputs.

**Inputs:** `VLMStructuredOutput` (from `vlm/base.py`), raw response string, sample metadata.

**Outputs:** `ExtractedClaim` list + `ExtractionResult`.

**Data schema:**
```python
@dataclass
class ExtractedClaim:
    claim: str
    region_id: str              # 'R1'..'Rk' or 'NONE'
    status: Literal["present","absent","uncertain"]
    confidence: Literal["high","medium","low"]
    raw_claim_text: str | None  # verbatim from model, pre-normalization

@dataclass
class ExtractionResult:
    answer_text: str
    claims: list[ExtractedClaim]
    parse_success: bool
    error: str | None           # JSON parse failure, missing fields, etc.
```

**Algorithm:**
1. If `raw_response` is empty/None → `parse_success=False`, error="empty_raw_response".
2. Run Pydantic validation via FS-006 `JSONValidator.parse_and_validate`.
3. If validation fails → `parse_success=False`, error=validation message.
4. Map each `VLMClaim` to `ExtractedClaim` preserving all fields.
5. Normalize `answer_text` for downstream exact-match metrics (strip, lowercase, remove code fences, collapse whitespace, remove trailing punctuation — same as FS-004 `scripts/compare_slake_baselines.py` normalization).

**Error handling:** Empty raw response, invalid JSON, missing `answer` field, missing `findings` list → recorded as extraction failure, not as an incorrect answer. Failure propagated to metrics (handled as missing/failed prediction per FS-004 convention).

**Dependencies:** `core/schemas.py` (`VLMClaim`, `VLMStructuredOutput`), FS-006 `JSONValidator`.

**Testing:** Unit tests for empty input, malformed JSON, missing fields, valid structured output, code-fence stripping.

---

### 4.2 Claim-to-region association

**Responsibilities:** Associate each extracted claim with a candidate region from the detector-proposed set $R = \{R_1 \dots R_k\}$. Validate that `region_id` names an existing region. Handle `NONE` (negative/absence claims).

**Inputs:** `ExtractedClaim[]`, `CandidateBox[]` (the active region set $R$), track identifier.

**Outputs:** `AssociatedClaim[]` with `is_grounded` flag.

**Data schema:**
```python
@dataclass
class AssociatedClaim:
    claim: str
    region_id: str
    status: str
    confidence: str
    is_grounded: bool           # True iff region_id ∈ {R1..Rk} or region_id == "NONE"
    cited_box: CandidateBox | None   # the CandidateBox matching region_id, else None
    rejection_reason: str | None     # e.g. "region_id R99 not in R", "empty R"
```

**Algorithm:**
1. Build lookup `region_id → CandidateBox` from $R$.
2. For each claim:
   - If `region_id == "NONE"`: `is_grounded=True` (NONE is a valid token for absence), `cited_box=None`, `rejection_reason=None`.
   - If `region_id` in lookup: `is_grounded=True`, `cited_box=lookup[region_id]`.
   - Else: `is_grounded=False`, `cited_box=None`, `rejection_reason=f"region_id {region_id} not in R={sorted(R_ids)}"`.
3. If $R = ∅$ and claim cites any region: `is_grounded=False`, rejection_reason="R is empty (no candidate regions)".
4. If $R = ∅$ and claim cites `NONE`: `is_grounded=True` (negative claim with no regions to ground).

**Error handling:** Empty $R$ is a valid state (not an error). Unknown region IDs are rejected as ungrounded (not an error).

**Dependencies:** `localization/detector.py` (`CandidateBox`), `core/schemas.py`.

**Testing:** Known region IDs pass; unknown IDs rejected; $R=∅$ with NONE claim grounded; $R=∅$ with region claim ungrounded; duplicate region IDs impossible (schema guarantees uniqueness).

---

### 4.3 Region evidence verification

**Responsibilities:** Compute IoU between each grounded claim's cited box and ground-truth boxes. Determine `is_supported` per the paper's rule: IoU ≥ 0.5 against any ground-truth box for the cited pathology.

**Inputs:** `AssociatedClaim[]`, `GroundTruthRegion[]` (from `MedicalVQASample.ground_truth_regions`), IoU threshold (0.5).

**Outputs:** `VerifiedClaim[]` with `is_supported`, `iou_with_ground_truth`, `localizer_score`.

**Data schema:**
```python
@dataclass
class VerifiedClaim:
    claim: str
    region_id: str
    status: str
    is_grounded: bool
    localizer_score: float | None     # rho from CandidateBox.score, None if NONE or ungrounded
    iou_with_ground_truth: float | None
    is_supported: bool                # is_grounded AND (NONE claim OR IoU ≥ 0.5)
```

**Algorithm (IoU with multiple ground-truth boxes):**
1. For each associated claim with `cited_box`:
   - Convert `cited_box.box_2d` (ymin,xmin,ymax,xmax) to the same convention as GT boxes.
   - For each `gt` in `ground_truth_regions`: compute IoU(cited_box.box_2d, gt.box_2d).
   - `iou_with_ground_truth = max_iou` over all GT boxes (handles multiple lesions).
   - `is_supported = is_grounded AND (region_id == "NONE" OR max_iou ≥ 0.5)`.
   - `localizer_score = cited_box.score` (rho), else None.
2. For ungrounded claims: `is_supported=False`, `iou_with_ground_truth=None`, `localizer_score=None`.
3. For NONE claims: `iou_with_ground_truth=None`, `localizer_score=None`, `is_supported = is_grounded` (absence claims supported iff grounded-to-NONE; clinical truth of absence determined against dataset label at metrics stage).

**IoU formula** (normalized [ymin,xmin,ymax,xmax]):
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
Intersection computed on the overlapping rectangle; if no overlap, IoU = 0.

**Error handling:** Empty GT region list → max IoU undefined → treat as IoU=0 → `is_supported=False` for positive claims (nothing to point to). For NONE claims with empty GT, `is_supported=True` (absence consistent with no annotated pathology).

**Dependencies:** `core/schemas.py` (`GroundTruthRegion`, `CandidateBox`), `localization/detector.py`.

**Testing:** IoU on overlapping/non-overlapping/contained boxes; multiple GT boxes (max overlap); empty GT; NONE claims; ungrounded claims.

---

### 4.4 Unsupported claim detection and filtering

**Responsibilities:** Classify claims as supported/unsupported/unverifiable. Compute USR_raw and USR_filtered. Produce the filtered claim set used for answer construction.

**Inputs:** `VerifiedClaim[]`, sample ground-truth answer (for absence-claim truth determination).

**Outputs:** `FilterResult` with supported claims, unsupported claims, unverifiable claims, USR_raw, USR_filtered.

**Data schema:**
```python
@dataclass
class FilterResult:
    supported_claims: list[VerifiedClaim]
    unsupported_claims: list[VerifiedClaim]     # grounded but IoU < 0.5
    unverifiable_claims: list[VerifiedClaim]    # ungrounded (bad region_id or R=∅)
    usr_raw: float                              # N_unsup_raw / N_claims
    usr_filtered: float                         # N_unsup_filtered / N_claims_after_strip
    n_claims: int
```

**Definitions (per audit ISSUE-01 / ISSUE-03):**
- $N_{\text{claims}}$ = total claims emitted (before any filtering).
- $\text{USR}_{\text{raw}}$ = fraction of raw claims that are unsupported OR unverifiable:
  $$\text{USR}_{\text{raw}} = \frac{N_{\text{unsup}} + N_{\text{unverifiable}}}{N_{\text{claims}}}$$
  where unsupported = grounded but IoU < 0.5; unverifiable = ungrounded (bad region_id, R=∅).
- $\text{USR}_{\text{filtered}}$ = computed after stripping unverifiable claims (invalid region IDs dropped):
  $$\text{USR}_{\text{filtered}} = \frac{N_{\text{unsup}}}{N_{\text{claims}} - N_{\text{unverifiable}}}$$
  (denominator = claims citing valid regions; if denominator = 0, USR_filtered = 0.0 by convention).

**Claim classification:**
| Claim type | USR_raw contribution | USR_filtered contribution |
|------------|---------------------|---------------------------|
| Supported (grounded, IoU ≥ 0.5) | supported | supported |
| Unsupported (grounded, IoU < 0.5) | unsupported | unsupported |
| Unverifiable (bad region_id or R=∅) | unsupported | stripped (removed from denominator) |
| NONE (absence, grounded) | supported* | supported* |

\* NONE claims are "supported" at the structural level (they cite a valid token); clinical correctness of absence is evaluated separately against the dataset ground-truth answer in the metrics layer.

**Answer construction from filtered claims:** The final grounded answer text is the VLM's `answer_text` with unsupported claims removed conceptually; at the metrics layer, the answer is scored as correct/incorrect against ground truth regardless (answer-level accuracy is separate from claim-level USR).

**Error handling:** Zero claims → USR_raw = 0.0 (no claims to be unsupported), USR_filtered = 0.0, all lists empty.

**Dependencies:** `verification/claim_verifier.py` (4.3 output), `core/schemas.py`.

**Testing:** USR_raw vs USR_filtered on mixed claim sets; all-supported → 0.0; all-unsupported → 1.0; empty claims → 0.0; denominator-zero guard.

---

### 4.5 Semantic agreement and answer uncertainty estimation

**Responsibilities:** Cluster N sampled answers into semantic equivalence classes, compute agreement $a$ (modal cluster proportion) and normalized semantic entropy $H_n$.

**Inputs:** N `VLMStructuredOutput` answers (or just answer strings), sample ID.

**Outputs:** `AgreementEntropyResult` with majority answer, $a$, raw entropy $H$, $H_n$, cluster mapping.

**Data schema:**
```python
@dataclass
class SemanticCluster:
    cluster_id: int
    members: list[int]          # indices of sampled answers in this cluster
    representative: str         # canonical answer text of cluster

@dataclass
class AgreementEntropyResult:
    majority_answer: str
    agreement: float            # a = max_k |C_k| / N
    raw_entropy: float          # H = -sum p_k ln p_k
    normalized_entropy: float   # H_n = H / ln(N)
    clusters: list[SemanticCluster]
    n_samples: int
```

**Algorithm (Kuhn et al. 2023 semantic clustering):**
1. Collect N answer strings $\{y_1 \dots y_N\}$.
2. Partition into semantic equivalence clusters $\{C_1 \dots C_K\}$ using one of:
   - **Bidirectional NLI entailment** (preferred): answers $y_i, y_j$ are semantically equivalent iff $y_i \Rightarrow y_j$ AND $y_j \Rightarrow y_i$ under a medical NLI model (e.g. a fine-tuned BERT/NLI on medical entailment).
   - **Open embedding distance threshold**: sentence-transformer embeddings, cosine similarity ≥ threshold → same cluster. Threshold to be calibrated on validation data.
   - **Canonical medical entity normalization**: map answers to UMLS/SNOMED canonical forms; exact match on canonical form → same cluster. (Fallback when no NLI/embedding model available.)
3. $p_k = |C_k| / N$.
4. $a = \max_k p_k = |C_{\text{modal}}| / N$.
5. $H = -\sum_{k=1}^K p_k \ln p_k$ (natural log; $0 \ln 0 = 0$).
6. $H_n = H / \ln N \in [0, 1]$. Edge cases: $N=1$ → $H_n = 0$ (single sample, no uncertainty); unanimous answers ($K=1$) → $H_n = 0$; uniform distribution → $H_n = 1$.

**Why not string matching:** Naive `y_i == y_j` overestimates uncertainty because synonymous clinical phrasings ("pleural effusion", "fluid in pleural space", "effusion noted") are clinically identical but lexically distinct. Semantic clustering is mandatory per audit ISSUE-04.

**Error handling:** N=0 → return $a=0, H_n=0$, empty clusters, error="no_samples". Empty answer strings → treated as a distinct cluster value (empty string).

**Dependencies:** None from project code (external NLI/embedding model is a configuration choice); `core/schemas.py` (`ConfidenceGateResult` reuses these fields).

**Recommended initial approach:** sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2`, free/local, ~80 MB, reproducible) with cosine-similarity threshold for clustering; threshold calibrated on the SLAKE validation split. Canonical medical entity normalization (UMLS/SNOMED exact-match on canonical forms) as a deterministic lightweight fallback when no embedding model is available. Bidirectional NLI is a heavier, more accurate alternative but not the default due to model size and reproducibility constraints.

**Validation plan:**
1. Calibrate the cosine threshold on the SLAKE validation split using synonymous/paraphrase clinical answer pairs.
2. Measure cluster purity against held-out human annotations where available.
3. Report cluster count distribution and $H_n$ sensitivity to threshold on validation.
4. Do not claim clinical validation of any clustering method — report it as a technical configuration choice evaluated on benchmark accuracy/entropy stability only.

**Testing:** Unanimous → $a=1, H_n=0$; uniform → $a=1/N, H_n=1$; synonymous answers clustered together; N=1 edge case.

---

### 4.6 Confidence calibration

**Responsibilities:** Learn weights $(w_1, w_2, w_3)$ and select abstention threshold $\tau$ on held-out validation data; generate accuracy-vs-coverage curves; evaluate ECE.

**Inputs:** Validation set samples with (answer_correctness, $a$, $H_n$, $\rho$) per sample.

**Outputs:** `CalibrationResult` with fitted weights, $\tau$, ECE, AUROC, coverage curve points.

**Data schema:**
```python
@dataclass
class CalibrationResult:
    w1: float
    w2: float
    w3: float
    tau: float                  # operating point (one value selected)
    ece: float
    auroc: float                # AUROC of c predicting correctness
    coverage_curve: list[tuple[float, float, float]]  # (tau, coverage, accuracy)
    calibration_bins: list[dict]  # for ECE reporting
    n_validation: int
```

**Weight fitting protocol (no hardcoding):**
1. Constraint: $w_1 + w_2 + w_3 = 1$, $w_i \ge 0$.
2. Grid search over step size 0.05 on the SLAKE validation split.
3. Objective: maximize **selective risk / risk-coverage trade-off** — i.e., select weights that maximize accuracy at target coverage levels across the tau sweep. AUROC is reported as a secondary ranking metric (it measures confidence-score ranking quality but does not directly optimize the abstention operating point). ECE is reported for calibration quality.
4. Selected weights and $\tau$ are frozen before touching the test set.

**Threshold selection:**
- $\tau$ is not a fixed constant. Sweep $\tau \in [0, 1]$ to generate the full accuracy-vs-coverage curve.
- Operating $\tau$ selected per deployment need (e.g., maximize selective accuracy at target coverage).
- $N=5$ standard for benchmark calibration; if $N=3$ used in UI, separate $\tau_{N=3}$ calibrated on $N=3$ validation outputs.
- **Sampling parameters recorded per experiment:** sample count $N$, temperature $T$, and random seed must be logged in experiment metadata to ensure reproducibility and to prevent cross-contamination of entropy scales between $N=5$ benchmark and $N=3$ UI modes.

**Confidence score computation:**
$$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
- $a \in [0,1]$: agreement (modal cluster proportion).
- $H_n \in [0,1]$: normalized semantic entropy; $(1 - H_n)$ rewards consensus.
- $\rho \in [0,1]$: localizer score of cited region; `None` for NONE/unverifiable claims.
- When $\rho$ is `None` (negative/NONE claims or ungrounded): the $\rho$ term is omitted and weights rescaled:
  $$c_{\text{negative}} = \frac{w_1 a + w_2 (1 - H_n)}{w_1 + w_2}$$
  (candidate approach per ADR 0004 Option 1 — prevents artificial inflation of absence claims). **No negative-claims policy is adopted:** ADR 0004 remains "Proposed / Pending User Decision" with Options 1, 2, and 3 all under consideration. This formula is presented as an implementation candidate only, not as approved policy, pending user confirmation (see Section 11.1).

**Effect of empty regions / unavailable evidence:**
- If $R = ∅$ and claim cites a region → ungrounded → $\rho = \text{None}$ → rescaled confidence (negative formula).
- If claim cites `NONE` → $\rho = \text{None}$ → rescaled confidence.
- Empty regions reduce the information term, lowering $c$ relative to a fully-grounded positive claim with the same $a, H_n$ — appropriate, since there is less visual evidence.

**Abstention decision:**
- Accept answer if $c \ge \tau$.
- Abstain (return "uncertain") if $c < \tau$.
- Abstention is separate from correctness: an abstained sample is removed from coverage (not counted as correct or incorrect).

**Dependencies:** `verification/confidence_gate.py` (design), scikit-learn for AUROC, `core/schemas.py` (`ConfidenceGateResult`).

**Testing:** Weight sum constraint; rescaled confidence for None rho; ECE on perfectly-calibrated toy bins; AUROC on separable vs. random scores; tau sweep monotonicity of coverage.

---

### 4.7 Abstention and selective prediction

**Responsibilities:** Apply the confidence gate decision to produce a final answer or abstention. Track coverage and selective accuracy.

**Inputs:** `ConfidenceGateResult` (c, tau), answer text, ground truth answer, correctness flag.

**Outputs:** `SelectivePrediction` with final status (ACCEPTED / ABSTAINED), final answer, coverage flag.

**Data schema:**
```python
@dataclass
class SelectivePrediction:
    sample_id: str
    confidence: float           # c
    threshold: float            # tau
    is_abstaining: bool         # c < tau
    final_answer: str | None    # None if abstaining
    is_correct: bool | None     # None if abstaining
    in_coverage: bool           # not is_abstaining
```

**Selective accuracy:** accuracy computed only over in-coverage samples:
$$\text{selective\_accuracy} = \frac{\sum_{\text{in coverage}} \mathbf{1}[\text{correct}]}{\text{coverage count}}$$

**Coverage:**
$$\text{coverage} = \frac{\text{in-coverage count}}{\text{total samples}} \in [0, 1]$$

**Dependencies:** `verification/confidence_gate.py` (4.6), `core/schemas.py`.

**Testing:** c ≥ tau → accepted; c < tau → abstained; selective accuracy ≥ overall accuracy on well-calibrated gate; coverage decreases monotonically as tau increases.

---

### 4.8 Evaluation metrics and experiment artifact generation

**Responsibilities:** Compute all required metrics per sample and aggregate; generate experiment artifacts (JSON, CSV, Markdown); handle missing/empty/malformed/unverifiable responses consistently.

**Metrics definitions (formal):**

Let $S$ be the set of samples in the evaluated split. For each sample $s \in S$:
- $\text{ans}_s$: model answer (normalized)
- $\text{gt}_s$: ground-truth answer (normalized)
- $\text{correct}_s = \mathbf{1}[\text{normalize(ans}_s) == \text{normalize(gt}_s)]$
- $\text{claims}_s$: list of verified claims
- $\text{unsup}_s$: number of unsupported claims
- $\text{unver}_s$: number of unverifiable claims
- $\text{nclaims}_s$: total claims emitted

**Raw Unsupported Claim Rate:**
$$\text{USR}_{\text{raw}} = \frac{\sum_s (\text{unsup}_s + \text{unver}_s)}{\sum_s \text{nclaims}_s}$$
(denominator = 0 → USR_raw = 0.0)

**Filtered Unsupported Claim Rate:**
$$\text{USR}_{\text{filtered}} = \frac{\sum_s \text{unsup}_s}{\sum_s (\text{nclaims}_s - \text{unver}_s)}$$
(denominator = 0 → USR_filtered = 0.0)

**Overall accuracy (closed + open):**
$$\text{Accuracy} = \frac{\sum_s \text{correct}_s}{|S|}$$

**OPEN / CLOSED accuracy:** computed on subsets where `answer_type == "OPEN"` / `"CLOSED"` respectively, denominator = subset size.

**Pointing accuracy** (IoU ≥ 0.5 against GT regions):
$$\text{PointingAcc} = \frac{\sum_s \mathbf{1}[\exists \text{ claim with } \text{IoU} \ge 0.5]}{\sum_s \mathbf{1}[\text{sample has GT regions}]}$$
Samples with no GT regions are excluded from the denominator (nothing to point to). If a sample has GT regions but the model makes no claims or no grounded claims, it counts as not pointing correctly.

**Sampling agreement:** mean $a$ across samples (from 4.5).

**Normalized semantic entropy:** mean $H_n$ across samples.

**ECE** (Equation 3, $M=10$ equal-width bins over $[0,1]$):
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $B_m$ = samples whose confidence $c$ falls in bin $m$, $\text{acc}(B_m)$ = fraction correct in bin, $\text{conf}(B_m)$ = mean confidence in bin. Abstained samples excluded (no confidence decision applied) or placed in a separate "abstained" bin — design decision: excluded from ECE (they received no accepted confidence).

**Selective accuracy & coverage:** from 4.7, computed across tau sweep.

**Accuracy-vs-coverage curve:** for $\tau \in \{0, 0.01, \dots, 1.0\}$, compute (coverage, selective accuracy). Report as list of points.

**AUROC:** ROC-AUC of confidence $c$ predicting binary correctness over the full eligible population — all samples with valid confidence predictions (i.e., not forced to abstain by Phase 2 policy: `sampling_valid == True` and `evidence_state != "uncertain"`). The population is independent of the abstention threshold $\tau$; samples that abstain only because $c < \tau$ still have meaningful confidence and are included. Uses the Mann-Whitney U statistic. Returns None if there are no eligible samples or only one correctness class.

**Missing / empty / malformed / unverifiable handling in every metric:**
| Situation | Accuracy | USR_raw | USR_filtered | PointingAcc | ECE / Selective |
|-----------|----------|---------|--------------|-------------|-----------------|
| Missing generation (no response file) | incorrect (counted in denominator) | N/A (no claims) | N/A | excluded (no regions claimed) | excluded |
| Empty raw response | incorrect | 0.0 | 0.0 | excluded | excluded |
| Malformed JSON (parse failure) | incorrect | 0.0 | 0.0 | excluded | excluded |
| Unverifiable claim (bad region_id) | based on answer text | counts as unsupported | stripped from denominator | excluded for that claim | excluded |
| Empty GT answer (SLAKE qid=1622) | incorrect (empty vs non-empty) | normal | normal | normal | normal |
| Sample with no GT regions | normal accuracy | normal | normal | excluded from denominator | normal |

**Artifacts generated:**
- `experiment_metrics.json` — per-sample records + aggregate metrics
- `experiment_summary.csv` — metric table
- `per_sample_results.csv` — per-sample correctness, USR, pointing, confidence, abstention
- `accuracy_vs_coverage.csv` — (tau, coverage, accuracy) points
- `comparison_report.md` — human-readable report

**Dependencies:** `verification/` (4.1–4.7), `core/schemas.py`, `evaluation/metrics.py` (new), `evaluation/runner.py` (new), `json`, `csv`.

**Testing:** USR edge cases; pointing accuracy with/without GT regions; ECE on toy bins; accuracy-vs-coverage monotonicity; missing/empty/malformed handling per table above; artifact file formats valid.

---

## 5. Data Schemas and Interfaces

New dataclasses to be added (additive to `core/schemas.py`, no modification of existing fields):

```python
@dataclass
class ExtractedClaim: ...        # 4.1
@dataclass
class ExtractionResult: ...      # 4.1
@dataclass
class AssociatedClaim: ...       # 4.2
@dataclass
class VerifiedClaim: ...         # 4.3 (extends existing with is_supported)
@dataclass
class FilterResult: ...          # 4.4
@dataclass
class SemanticCluster: ...       # 4.5
@dataclass
class AgreementEntropyResult: ...# 4.5
@dataclass
class CalibrationResult: ...     # 4.6
@dataclass
class SelectivePrediction: ...   # 4.7
```

New modules (no existing files modified):
- `verification/claim_extractor.py` — 4.1
- `verification/claim_associator.py` — 4.2
- `verification/evidence_verifier.py` — 4.3
- `verification/claim_filter.py` — 4.4
- `verification/semantic_clustering.py` — 4.5
- `verification/confidence_gate.py` — 4.6, 4.7 (per FS-008 spec)
- `evaluation/metrics.py` — 4.8 metric functions
- `evaluation/runner.py` — 4.8 experiment orchestration + artifact writing

Interfaces reuse `core/schemas.py` types: `MedicalVQASample`, `CandidateBox`, `VLMClaim`, `VLMStructuredOutput`, `GroundTruthRegion`, `EvaluationTrack`.

---

## 6. Mathematical Definitions and Metrics

### 6.1 IoU (pointing accuracy)
$$\text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)}$$
Multiple GT boxes: $\max_{b^* \in \mathcal{B}_{\text{gt}}} \text{IoU}(b_{\text{cited}}, b^*)$. Threshold ≥ 0.5.

### 6.2 USR (raw and filtered)
$$\text{USR}_{\text{raw}} = \frac{N_{\text{unsup}} + N_{\text{unverifiable}}}{N_{\text{claims}}}$$
$$\text{USR}_{\text{filtered}} = \frac{N_{\text{unsup}}}{N_{\text{claims}} - N_{\text{unverifiable}}}$$

### 6.3 Semantic agreement and entropy
$$a = \frac{|C_{\text{modal}}|}{N}, \quad H_n = \frac{-\sum_k p_k \ln p_k}{\ln N}, \quad p_k = \frac{|C_k|}{N}$$

### 6.4 Confidence gate
$$c = w_1 a + w_2 (1 - H_n) + w_3 \rho, \quad \sum w_i = 1, w_i \ge 0$$
Negative/NONE claims ($\rho = \text{None}$):
$$c_{\text{negative}} = \frac{w_1 a + w_2 (1 - H_n)}{w_1 + w_2}$$
Decision: accept if $c \ge \tau$, else abstain.

### 6.5 ECE
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \quad M=10$$

### 6.6 Selective accuracy and coverage
$$\text{coverage} = \frac{|\{s: c_s \ge \tau\}|}{|S|}, \quad \text{selective\_accuracy} = \frac{\sum_{s: c_s \ge \tau} \mathbf{1}[\text{correct}_s]}{|\{s: c_s \ge \tau\}|}$$

### 6.7 Accuracy-vs-coverage curve
Points $(\text{coverage}(\tau), \text{selective\_accuracy}(\tau))$ for $\tau \in [0, 1]$ sweep.

---

## 7. Experimental Tracks and Answer Conditions

### 7.1 Three experimental tracks (region source / data eligibility)

| Track | Description | Region source | Eligible data | Detector used | Exclusions |
|-------|-------------|---------------|---------------|---------------|------------|
| **Track A** | End-to-end predicted grounding | CXR detector proposed boxes | Chest X-ray subsets of SLAKE and VQA-RAD only | Fine-tuned medical CXR detector (RT-DETR / Torchvision Faster R-CNN) | **Excluded from SLAKE:** SLAKE contains zero chest X-ray images (all chest-location entries are CT, per `docs/reports/SLAKE_DATASET_VERIFICATION.md` §3.3). Track A never activates on SLAKE. Requires VQA-RAD CXR or RSNA/VinDr-CXR images. |
| **Track B** | Oracle / GT-box grounding | Dataset ground-truth boxes | All SLAKE modalities (chest CT, abdomen CT, brain MRI) with usable GT regions | None (GT boxes promoted to CandidateBox) | **Excluded:** samples with empty `detection.json: []` (no GT regions) — excluded from pointing-accuracy denominator but still evaluated for answer accuracy. SLAKE test: 980 of 1,061 English test samples have regions. |
| **Track C** | Ungrounded answer baseline | None ($R = ∅$) | VQA-RAD and SLAKE (all modalities) | None | None — applies to all samples. VQA-RAD has no bounding box annotations (`has_ground_truth_regions=False`), so Track C is the only applicable track for VQA-RAD. |

**Track B pointing-accuracy denominator:** only samples with $\geq 1$ ground-truth region. Samples with zero GT regions are excluded from the pointing-accuracy denominator (nothing to point to) but remain in answer-accuracy and USR denominators.

**Track A exclusion consequence:** Because SLAKE has no CXR images, any Track A results on SLAKE would be empty/invalid. Track A evaluation must use VQA-RAD CXR or RSNA/VinDr-CXR data only, never SLAKE.

**Track × Condition separation:** Tracks (A/B/C) answer "where do regions come from"; Conditions (1/2/3) answer "how is the answer processed." A full evaluation reports a 3×3 matrix (track × condition) where applicable, never conflating the two taxonomies.

**Split handling:** Official splits preserved unchanged (train/validate/test). No custom splits. English-only for SLAKE (`language="en"`). Validation split used for weight/threshold fitting; test split used for final reporting.

### 7.2 Three answer conditions (answer processing)

| Condition | Description | Region requirement | Verification | Abstention |
|-----------|-------------|-------------------|--------------|------------|
| **Condition 1** | Free-form VLM answers (natural language, no JSON contract) | No | No | No |
| **Condition 2** | Region-grounded answers + claim verification, no confidence-gate abstention | Yes ($R$ from track) | Yes (USR, pointing) | No |
| **Condition 3** | Region-grounded answers + claim verification + confidence-gate abstention | Yes ($R$ from track) | Yes (USR, pointing) | Yes ($c \ge \tau$) |

**Fair comparison:** Each condition is evaluated on the identical sample set within a track. Condition 1 uses N=1 greedy ($T=0.0$); Conditions 2/3 use N=5 ($T=0.7$) for agreement/entropy. Answer-level accuracy uses the same normalization (FS-004 convention) across all conditions. Condition 3 reports selective accuracy + coverage in addition to overall accuracy.

**Do not confuse:** Tracks (A/B/C) answer "where do regions come from"; Conditions (1/2/3) answer "how is the answer processed." A full evaluation reports a 3×3 matrix (track × condition) where applicable.

---

## 8. Dataset and Leakage Controls

### 8.1 Official split preservation

- SLAKE: official `train.json`/`validate.json`/`test.json` unchanged. English-only filter (`language="en"`).
- Validation split (1,053 EN entries, 96 unique images) used exclusively for weight/threshold fitting.
- Test split (1,061 EN entries, 96 unique images) used exclusively for final reporting.
- No samples removed, no custom splits, no leakage-controlled subsets for primary results.

### 8.2 Patient-level leakage

- **SLAKE:** No clinical patient IDs published. `patient_id = None`, `patient_ids_available = False`. Image-level leakage across splits exists (298 img_ids) but zero English-only overlap (verified in `docs/reports/SLAKE_EVALUATION_PROTOCOL.md` §1.2). Limitation documented, not mitigated by split modification.
- **RSNA & VinDr-CXR:** Multiple views per patient exist. All split parsers in `datasets/` must enforce grouping by `patient_id` before random split. No patient's images appear in more than one split.
- **VQA-RAD:** No patient IDs; relies on official split.

### 8.3 Duplicate and cross-split samples

- SLAKE qids are unique (14,028 unique across all splits).
- Cross-split img_id overlap (298) is bilingual (EN in one split, ZH in another). English-only evaluation has zero overlap — verified.
- No deduplication performed; duplicates are a documented limitation.

### 8.4 Annotation availability

- Not every SLAKE sample has usable region annotations (62 patients have empty `detection.json: []`).
- Track B pointing-accuracy denominator excludes samples with no GT regions.
- Track C has no region requirement — unaffected.
- Missing answers (SLAKE train qid=1622, empty string): preserved by loader; metrics treat empty answer as incorrect against non-empty GT.

### 8.5 Split construction rule

No new split strategy introduced without documentation and justification. Official splits are the default; any deviation (e.g., leakage-controlled subset) must be explicitly labeled and justified.

---

## 9. Hardware and Execution Strategy

### 9.1 Constraints

| Resource | Specification |
|----------|---------------|
| Local GPU | NVIDIA RTX 2050, 4 GB VRAM |
| System RAM | 16 GB |
| Remote GPU | Google Colab Tesla T4, 15.3 GB VRAM (free tier) |
| Local CPU | Intel i5 (MedSAM ~1.5s/image on CPU) |

### 9.2 Model VRAM profiles (4-bit)

| Model | Approx VRAM |
|-------|-------------|
| Qwen2.5-VL-3B | ~5.0 GB |
| MedGemma-4B | ~6.0 GB |
| MedSAM ViT-B | ~3.0 GB |
| PyTorch CUDA context | ~1.5 GB |

Concurrent load of all three ≈ 15.5 GB — exceeds Colab T4 ceiling (15.3 GB) and far exceeds RTX 2050 (4 GB).

### 9.3 Execution strategy

1. **Sequential on-demand loading:** Only one VLM resident in GPU VRAM at a time. After use: unload model, set references to None, `torch.cuda.empty_cache()`, then load the next. Implemented by `api/remote_worker.py` `RemoteModelWorker.load_model()`/`unload_model()` (already sequential).

2. **MedSAM on CPU locally:** MedSAM runs on laptop CPU (no GPU required) for Track B oracle grounding and visual overlay generation. Avoids GPU contention.

3. **RTX 2050 (4 GB):** Too small for 4-bit VLMs. Use CPU inference with `MockVLMClient` for local TDD; run real VLM inference only on Colab/Kaggle remote worker.

4. **No paid APIs:** All models are open-weight (Qwen2.5-VL-3B, MedGemma-4B gated but free with HF token, MedSAM Apache-2.0).

5. **Lazy loading:** Models load on first inference call, not at startup.

---

## 10. Testing and Validation Strategy

### 10.1 Test levels (per `docs/architecture/testing_strategy.py`)

| Level | Tool | Scope |
|-------|------|-------|
| Unit | pytest | Schemas, IoU, USR, entropy, ECE, confidence gate, claim filtering |
| Integration | FastAPI TestClient | `POST /analyze` end-to-end, claim gate, abstention |
| E2E | Playwright | Web UI upload, overlay, answer display |

### 10.2 Mock strategy

- All VLM generation mocked via `MockVLMClient` returning pre-recorded fixture `VLMStructuredOutput` with controlled claims, region IDs, and correctness.
- `MockRegionDetector` and `MockRegionSegmenter` for deterministic box/mask generation without model weights.
- Mocks never used as silent fallbacks for real models.

### 10.3 Coverage floor

≥ 85% line coverage via `pytest-cov`; no skipped tests without ADR justification.

### 10.4 Mathematical verification tests

- IoU: overlapping, non-overlapping, contained, identical, zero-area boxes.
- USR_raw / USR_filtered: all-supported (0.0), all-unsupported (1.0), mixed, empty claims, denominator-zero guard.
- Entropy: unanimous ($H_n=0$), uniform ($H_n=1$), N=1 edge case.
- Confidence gate: edge conditions ($\rho=0$, $a=1$, $\rho=\text{None}$ rescaled).
- ECE: perfectly calibrated and miscalibrated toy bins.
- Accuracy-vs-coverage: monotonic coverage decrease as tau increases.

### 10.5 Execution commands

```
pytest --cov=core --cov=verification --cov=evaluation --cov=api --cov-report=term-missing
pytest tests/unit/test_metrics.py -v
ruff check .
```

---

## 11. Risks and Unresolved Research Decisions

### 11.1 Open research decisions (do not invent)

1. **Negative/absence claim handling (ADR 0004 — OPEN, no option approved):** ADR 0004 status is "Proposed / Pending User Decision." None of Options 1, 2, or 3 is adopted as authoritative policy. The design presents the rescaled-confidence formula (ADR 0004 Option 1) as a *candidate implementation approach only*, not as approved policy, pending user confirmation. No $\rho=1.0$ for NONE claims (that was the audit defect). **User decision required before implementing negative-claim confidence handling.**

2. **Semantic clustering method:** Three options (bidirectional NLI, embedding threshold, canonical entity normalization) listed in 4.5; selection depends on available medical NLI/embedding models and validation performance. No default hardcoded.

3. **Weight fitting objective:** AUROC maximization proposed; alternative objectives (selective accuracy at target coverage, ECE minimization) not ruled out.

4. **VQA-RAD pointing accuracy:** VQA-RAD has no bounding box annotations → pointing accuracy not applicable (Track C only). No synthetic GT boxes invented.

### 11.2 Known limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| SLAKE has no CXR images | Track A inactive on SLAKE | Track A requires VQA-RAD CXR / RSNA / VinDr-CXR |
| SLAKE no patient IDs | Patient-level leakage unverifiable | Documented limitation; `patient_id=None` |
| SLAKE image-level leakage (298 img_ids) | Possible optimistic metrics | English-only overlap = 0; documented |
| 10-sample FS-004 pilot | Not statistically conclusive | FS-004 explicitly labeled pilot; no superiority claims |
| RTX 2050 4 GB VRAM | Cannot run 4-bit VLMs locally | Remote Colab/Kaggle worker; CPU mocks locally |
| MedGemma-4B is gated | Requires HF license acceptance + token | Documented in FS-004; worker accepts HF_TOKEN |
| Negative claim clinical truth unverified | Absence correctness depends on dataset label | Evaluated against dataset ground-truth answer |

### 11.3 Technical verification vs. clinical validation

This system performs **technical verification** (answer correctness, grounding IoU, calibration metrics) on public datasets. It does **not** constitute clinical validation. No claims of clinical utility, safety, or diagnostic performance are made or supported by this design.

---

## 12. Phased Implementation Plan with Dependencies and Acceptance Criteria

### Phase 1 — Claim verification core (depends on FS-006, FS-007 specs)

**Modules:** `verification/claim_extractor.py`, `verification/claim_associator.py`, `verification/evidence_verifier.py`, `verification/claim_filter.py`

**Dependencies:** `core/schemas.py` (existing), `localization/detector.py` (existing), FS-006 JSON validator.

**Acceptance criteria:**
- Unit tests pass for extraction, association (incl. $R=∅$), evidence verification (IoU ≥ 0.5, multiple GT boxes), USR_raw/filtered.
- Ruff lint clean, ≥85% coverage on new modules.
- No modification to FS-001–FS-004 files.

### Phase 2 — Uncertainty and confidence gate (depends on Phase 1, FS-008 spec)

**Modules:** `verification/semantic_clustering.py`, `verification/confidence_gate.py`

**Dependencies:** Phase 1 outputs (`VerifiedClaim`), scikit-learn (AUROC), `core/schemas.py`.

**Acceptance criteria:**
- Semantic clustering produces correct agreement/entropy on unanimous/uniform/synonymous fixtures.
- Confidence gate computes $c$ correctly, including $\rho=\text{None}$ rescaling.
- Weight fitting grid search runs on validation split without hardcoding.
- Tau sweep generates accuracy-vs-coverage curve.
- Ruff lint clean, ≥85% coverage.

### Phase 3 — Evaluation metrics harness (depends on Phase 1, Phase 2, FS-010 spec)

**Modules:** `evaluation/metrics.py`, `evaluation/runner.py`

**Dependencies:** Phase 1 & 2 outputs, `core/schemas.py`, scikit-learn.

**Acceptance criteria:**
- All 9 metrics computed correctly per formal definitions (USR_raw/filtered, pointing accuracy, agreement, entropy, ECE, selective accuracy, coverage, accuracy-vs-coverage, AUROC).
- Missing/empty/malformed/unverifiable handling matches §4.8 table.
- Artifacts (JSON, CSV, Markdown) generated and valid.
- Ruff lint clean, ≥85% coverage.

### Phase 4 — Integration and tracks (depends on Phase 3)

**Modules:** `evaluation/runner.py` track orchestration; `api/` FastAPI `POST /analyze` (FS-011)

**Dependencies:** Phase 3, `vlm/` clients (existing), `api/remote_worker.py` (existing sequential loading).

**Acceptance criteria:**
- Track A/B/C × Condition 1/2/3 matrix executable.
- API endpoint returns `AnalyzeResponse` with confidence breakdown and abstention status.
- Integration tests pass with `MockVLMClient`.
- No GPU OOM from concurrent model loading (verified via sequential loading).

### Phase 5 — Validation and reporting (depends on Phase 4)

**Activities:** Run full benchmark on SLAKE test split (Track B/C), VQA-RAD (Track C), RSNA/VinDr-CXR (Track A if CXR data available). Generate comparison reports. Fit weights on validation, report on test only.

**Acceptance criteria:**
- Weights and $\tau$ fitted on validation; test results reported with no refitting.
- Bootstrap 95% CIs (1,000 resamples) reported for accuracy metrics.
- Report explicitly states 10-sample FS-004 pilot limitations do not apply to full evaluation.
- No clinical-superiority claims made.

---

## Summary

This design specifies an implementation-ready FS-005 grounding and uncertainty-aware answering system. It is **design and documentation only** — no code is written, no FS-001–FS-004 files are modified, and no experiments are executed.

**Files created:** `docs/reports/FS-005_GROUNDING_UNCERTAINTY_TECHNICAL_DESIGN.md` (this document)

**Unresolved questions:**
1. Negative/absence claim confidence policy (ADR 0004 — no option approved; all three proposals pending user decision)
2. Semantic clustering method (sentence-transformer embeddings recommended; threshold calibrated on validation)
3. Weight fitting objective (selective risk/risk-coverage primary; AUROC secondary; ECE reported)

**Decisions blocking implementation:** Negative/absence claim confidence policy (ADR 0004) requires user approval before Phase 1 claim-verification code can finalize the NONE-claim confidence path.

**Recommended implementation sequence:** Phase 1 (claim verification) → Phase 2 (uncertainty/gate) → Phase 3 (metrics harness) → Phase 4 (integration/tracks/API) → Phase 5 (validation/reporting).
