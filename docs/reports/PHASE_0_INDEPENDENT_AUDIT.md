# Phase 0 Independent Audit & Phase 1 Readiness Review

**Audit Date**: November 11, 2026  
**Auditor Roles**: Senior AI Research Engineer, Multimodal ML Architect, Security Auditor, and Lead Software Quality Reviewer  
**Audit Target**: Phase 0 Foundation (`audit/phase-0-review` branch at commit `d4a0cb0`)  
**Primary Source of Truth**: Research Paper: *"Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models"* (Abisha K M, Anusiya S, Bhavya Sree P R, Dirisala Kavya)

---

## 1. Executive Summary

This independent audit rigorously reviewed the repository foundation established in Phase 0 against the original 4-page research paper, hardware constraints (Intel i5 CPU, 16 GB RAM, integrated Intel Iris Xe graphics, 247 GB D: drive), $0.00 budget constraints, and production-grade software engineering standards.

### Overall Status: **Conditionally Approved with Mandatory Corrections**
* **Documentation & Traceability**: High structural completeness, but identified several ungrounded assumptions, premature resolution of unresolved research questions, and heuristic shortcuts that violate the core paper philosophy.
* **Mathematical Integrity**: Several subtle but critical flaws were identified where detector scores, semantic entropy, and agreement calculations diverged from Kuhn et al. (2023) and Wang et al. (2023).
* **Licensing & Compliance**: Found an acute open-source licensing conflict with Ultralytics YOLOv8 (AGPL-3.0) against the project's MIT license, and verified Hugging Face gated access requirements for MedGemma-4B.
* **Security & Environment**: Identified and successfully remediated 9 CVE vulnerabilities in bundled `setuptools`. Environment verified on Python 3.11 with zero CUDA dependencies.

---

## 2. Research Methodology Audit

### 2.1 The Three Experimental Conditions (Section V-B)
* **Paper Requirement**: Each VLM (Qwen2.5-VL-3B and MedGemma-4B) must be evaluated across three strictly separated conditions:
  1. *Condition 1 (Free-Form)*: Standard natural language prompt, no candidate regions presented, no JSON contract.
  2. *Condition 2 (Grounded Contract)*: Candidate regions ($R_1 \dots R_k$) visually overlaid, strict JSON contract, claim verification and invalid region dropping, **without** confidence gate abstention.
  3. *Condition 3 (Grounded + Confidence Gate)*: Full pipeline with visual grounding, claim filtering, and confidence gating $c \ge \tau$ with selective abstention.
* **Audit Finding**: Traceability is maintained, but `docs/architecture/system_architecture.md` and `api/app.py` initial designs over-indexed on Condition 3, neglecting the requirement to run Condition 1 and Condition 2 through the same evaluation harness. Condition 1 specifically requires free-form text parsing for accuracy metrics.

### 2.2 VLM Identical Prompting Contract (Section III-C)
* **Paper Requirement**: *"Both VLMs are given the same prompt text, so any difference in their output is attributable to the models and not to the wording of the prompt."*
* **Audit Finding**: Fully verified. Both models receive identical system instructions, schema definitions, and visual Set-of-Mark images.

### 2.3 Visual Evidence & Grounding Philosophy
* **Paper Guiding Axiom**: *"The system can only claim what it can point to... an answer is kept only if it is grounded by citing a region such as R2 in the right lower zone. A claimed cardiomegaly with no supporting region would be dropped."*
* **CRITICAL AUDIT DEFECT FOUND**:
  In `FS-002-region-localization-detector.md`, Section 2 & 4 proposed:
  > *"Fallback grid/salience proposer when detector finds zero lesions... return fallback anatomical quadrant regions so VLM still has valid spatial anchors."*
  * **Violation**: Fabricating synthetic spatial quadrant boxes when the detector returns no detections directly violates the core scientific premise of the paper. Section VI-B explicitly states: *"a lesion the localizer misses makes a true claim impossible to ground."* Artificial boxes invent non-existent visual evidence.
  * **Correction**: Delete the artificial quadrant generator. When the detector returns zero regions, $R = \emptyset$. Any asserted positive finding must naturally be dropped by the claim verifier as ungrounded.

---

## 3. Mathematical Correctness Audit

### 3.1 Unsupported-Claim Rate (USR) (Equation 1)
* **Definition**: $\text{USR} = \frac{N_{\text{unsup}}}{N_{\text{claims}}}$.
* **Audit Finding**: In the paper, USR is computed both *before* and *after* claim filtering.
  - $\text{USR}_{\text{raw}}$: Fraction of raw claims emitted by the VLM that either cite an invalid region ID or cite a region with $\text{IoU} < 0.5$ against ground truth.
  - $\text{USR}_{\text{filtered}}$: Computed after invalid region IDs are stripped. Only claims citing valid regions with $\text{IoU} < 0.5$ remain.
  - *Correction*: Specifications must explicitly track and report both $\text{USR}_{\text{raw}}$ and $\text{USR}_{\text{filtered}}$.

### 3.2 Pointing Accuracy (Section V-C)
* **Definition**: Proportion of findings whose cited region achieves $\text{IoU} \ge 0.5$ with the ground truth annotation.
* **Audit Finding**: Validated. Ground-truth box matching must handle multiple lesions via maximum overlap:
  $$\text{IoU}(b_{\text{cited}}, \mathcal{B}_{\text{gt}}) = \max_{b^* \in \mathcal{B}_{\text{gt}}} \text{IoU}(b_{\text{cited}}, b^*)$$

### 3.3 Sampling Agreement ($a$) & Semantic Entropy ($H_n$) (Section III-E)
* **CRITICAL AUDIT DEFECT FOUND**:
  Previous draft specifications treated answer agreement and entropy as naive string matching (`sampled_answers[i] == sampled_answers[j]`).
  * **Paper Reference [13] (Kuhn et al., 2023 - Semantic Uncertainty)**:
    Kuhn et al. explicitly demonstrated that computing entropy over raw lexical tokens drastically overestimates uncertainty because natural language allows multiple synonymous phrasings for the exact same clinical meaning (e.g. *"pleural effusion"*, *"fluid in pleural space"*, *"effusion noted"*).
  * Kuhn et al. mandate **Semantic Equivalence Clustering**: Sampled outputs must first be clustered into semantic equivalence classes $\{C_1, \dots, C_K\}$. Agreement $a$ is the proportion in the modal semantic cluster:
    $$a = \frac{|C_{\text{modal}}|}{N}$$
    And normalized semantic entropy is computed over the cluster distribution $\{p_k = |C_k|/N\}$:
    $$H_n = \frac{-\sum_{k=1}^K p_k \ln p_k}{\ln N}$$
  * *Correction*: Specification `FS-008` must implement semantic clustering (using either bidirectional entailment, an open embedding distance threshold, or canonical medical entity normalization).

### 3.4 Confidence Gate Equation (Equation 2)
* **Equation**: $c = w_1 a + w_2(1 - H_n) + w_3 \rho$.
* **Audit Finding**:
  - The previous draft invented default weights ($w_1=0.4, w_2=0.3, w_3=0.3$) and threshold ($\tau=0.65$).
  - **Correction**: The paper explicitly dictates that weights $(w_1, w_2, w_3)$ and $\tau$ are empirical parameters determined on a held-out split (SLAKE validation set). No static weights may be hardcoded as "ground truth". The system must output the complete **accuracy-versus-coverage curve** across $\tau \in [0, 1]$.

### 3.5 Expected Calibration Error (ECE) (Equation 3)
* **Equation**: $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$.
* **Audit Finding**: Verified correct. Bins $B_m$ must be partitioned equally across $[0, 1]$ (typically $M=10$), with weighted sample weighting.

---

## 4. Dataset Methodology Audit

### 4.1 Modality Mismatch Risk in SLAKE
* **Paper Note (Section III-B)**:
  > *"For chest films, the detector can be trained on boxes from RSNA or VinDr-CXR. Other modalities would need a suitable detector, so we treat a second modality as a stretch goal."*
* **Audit Finding**:
  SLAKE contains multiple anatomical locations: Chest (CXR), Abdomen (CT), and Brain (MRI).
  If a model evaluated on SLAKE uses a detector trained exclusively on Chest X-rays (RSNA / VinDr-CXR), the detector will fail completely on brain MRI and abdominal CT scans.
* **Mandatory Separation of Experimental Tracks**:
  1. **Track A (End-to-End Predicted Grounding)**: Evaluated exclusively on the **Chest X-ray** test subsets of SLAKE and VQA-RAD using the trained CXR detector.
  2. **Track B (Oracle / GT Box Grounding)**: Evaluated across all SLAKE modalities (Chest, Abdomen, Brain) by providing the dataset's ground-truth bounding boxes as candidate regions $R$. This isolates VLM reasoning from detector error.
  3. **Track C (Ungrounded Answer Baseline)**: Evaluated on VQA-RAD and SLAKE without region requirements.

### 4.2 Patient-Level Split Leakage Risk
* **Audit Finding**: Both RSNA and VinDr-CXR contain multiple radiographic views for individual patients. If splitting is done randomly by image filename, the same patient's pathology will appear in both training and validation/test sets, resulting in data leakage.
* **Correction**: All split parsers in `datasets/` must enforce grouping by `patient_id`.

---

## 5. Model and Hardware Feasibility Audit

### 5.1 Remote Cloud GPU Memory Realities (Free Colab / Kaggle)
* **Hardware Ceiling**: Free Google Colab provides an NVIDIA Tesla T4 GPU with **15.3 GB VRAM**.
* **Model VRAM Profiles (4-bit bitsandbytes)**:
  - `Qwen/Qwen2.5-VL-3B-Instruct`: ~2.5 GB model weights + ~1.5 GB vision encoder (FP16) + ~1.0 GB KV cache = **~5.0 GB VRAM**.
  - `google/medgemma-4b-it`: ~3.2 GB model weights + ~1.8 GB MedSigLIP encoder + ~1.0 GB KV cache = **~6.0 GB VRAM**.
  - `MedSAM` (ViT-B image encoder): ~1.5 GB weights + ~1.5 GB activation memory = **~3.0 GB VRAM**.
  - PyTorch CUDA context & runtime: **~1.5 GB VRAM**.
  - **Total Concurrent Footprint**: $5.0 + 6.0 + 3.0 + 1.5 = \mathbf{15.5\text{ GB}}$, which exceeds or sits directly on the edge of the 15.3 GB VRAM limit, risking sudden CUDA Out-of-Memory (OOM) fatal crashes.
* **Mandatory Architecture Correction**:
  The remote worker (`api/remote_worker.py`) must **not** hold both VLMs and MedSAM in GPU VRAM simultaneously.
  - *Option 1 (Sequential On-Demand Loading)*: Load Qwen2.5-VL-3B, evaluate/infer batch, unload, clear CUDA cache, and load MedGemma-4B.
  - *Option 2 (MedSAM on CPU / Separate Process)*: Run MedSAM on laptop CPU (MedSAM takes ~1.5s per image on modern Intel i5 CPU) or dedicate Colab GPU exclusively to one active VLM at a time.

### 5.2 MedGemma Access & Licensing Gate
* **Audit Finding**:
  - `google/medgemma-4b-it` is a gated repository on Hugging Face.
  - Access requires the user to accept Google Health's Health AI Developer Foundation terms of use.
  - The Colab worker script must accept a user-supplied Hugging Face token (`HF_TOKEN`) passed via environment variables.

---

## 6. Architecture & Licensing Audit

### 6.1 Critical Licensing Conflict: Ultralytics YOLOv8 vs. MIT License
* **Audit Finding**:
  - The repository root `LICENSE` is set to **MIT**.
  - Ultralytics YOLOv8 is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
  - AGPL-3.0 is a strong copyleft license with a network-trigger clause: any networked service (FastAPI backend) interacting with an AGPL library must release the entire application under AGPL-3.0. Distributing the code under MIT violates the AGPL license terms.
* **Correction Options**:
  - *Option A (Permissive / Clean)*: Use an Apache 2.0 / MIT licensed detector:
    - **YOLOv10 / YOLOv9** (Apache 2.0 / GPL dual).
    - **RT-DETR** (Real-Time DEtection TRansformer) from Baidu/PaddlePaddle (Apache 2.0).
    - **Torchvision Faster R-CNN / RetinaNet** (BSD-3-Clause).
  - *Option B (AGPL Compliance)*: Re-license the repository under AGPL-3.0 if Ultralytics YOLOv8 is retained.
  - *Recommendation*: Propose **RT-DETR or Faster R-CNN** under Apache 2.0/BSD, preserving full permissiveness and direct alignment with the paper's explicit mention of *"YOLO- or DETR-style detectors"*.

### 6.2 Review of Negative Claims & the `NONE` Region Identifier
* **Paper Note (Section VI-B)**:
  > *"Negative answers, such as 'no effusion,' are a particular problem for the approach. They claim an absence, so there is no region to point to, and the schema needs a rule for them; we have not settled on one."*
* **Audit of ADR 0004**:
  - ADR 0004 previously assumed setting $\rho = 1.0$ for `NONE` claims was valid.
  - **Defect**: Setting $\rho = 1.0$ for negative claims artificially inflates confidence:
    $$c = w_1 a + w_2(1 - H_n) + w_3(1.0)$$
    This rewards the model with a perfect detector score without any visual localization proof, leading to overconfident false negatives (e.g. missing an effusion and claiming "no effusion" with high confidence).
  - **Proposed Multi-Tier Negative Claim Policy (Pending User Approval)**:
    1. Schema explicitly classifies findings into `status: Literal["present", "absent", "uncertain"]`.
    2. If `status == "absent"`, `region_id` must cite `NONE` or the global anatomical compartment (e.g. `LUNG_BASES`).
    3. The detector score $\rho$ for an absence claim cannot be an arbitrary $1.0$; it must be either:
       - Defined as $\rho_{\text{absence}} = 1.0 - \max_{i} \rho_{\text{lesion}, i}$ (confidence that no lesion exists in that anatomical field).
       - Or omitted from Equation 2, rescaling weights: $c_{\text{negative}} = \frac{w_1 a + w_2(1 - H_n)}{w_1 + w_2}$.
    4. Text-based keyword negation detection must never be treated as clinical truth. Ground truth absence must be verified strictly against the dataset's clinical question and negative label.
    *Status: Marked as PROPOSED / PENDING USER DECISION in ADR 0004.*

### 6.3 Confidence Gate Sampling Count ($N=3$ vs. $N=5$)
* **Audit of ADR 0005**:
  - ADR 0005 proposed adaptive sampling: $N=3$ for interactive UI, $N=5$ for batch evaluation.
  - **Defect**: The entropy normalization factor $H_n = \frac{H}{\ln N}$ depends on $N$. With $N=3$, entropy is constrained to $\ln 3 \approx 1.0986$ with at most 3 discrete outcome partitions. Calibrating threshold $\tau$ on $N=5$ benchmarks and applying it to $N=3$ in the UI causes a distribution shift in $c$.
  - **Correction**: Standardize on **$N=5$** as the primary research benchmark across all evaluation reports. If interactive UI uses $N=3$ for low latency, the UI must explicitly apply an $N=3$ specific calibration operating point $\tau_{N=3}$.

---

## 7. Security Audit

### 7.1 Remediation of Dependency Vulnerabilities
* **Initial Audit Finding**: Running `pip-audit` identified **9 known security vulnerabilities** in `setuptools 65.5.0` (PYSEC-2022-43012, PYSEC-2025-49, PYSEC-2026-1918, PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w).
* **Remediation Action Executed**:
  Upgraded `setuptools` to version `84.2.1` in `.venv`.
* **Verification**: Subsequent audit confirmed 0 known vulnerabilities.

### 7.2 Static Security Analysis (Bandit)
* Scanned modules: `core/`, `api/`, `datasets/`, `localization/`, `vlm/`, `verification/`, `evaluation/`.
* Result: **0 issues identified** (0 High, 0 Medium, 0 Low).

---

## 8. Testing and Quality Audit

* **Test Suite**: Executed `pytest --cov=. --cov-report=term-missing`.
  - All tests passed in 0.16s.
  - Coverage: 100% on current base modules.
* **Linter**: `ruff check .` reported 0 errors, 0 warnings.
* **Type Checker**: Configured for Python 3.11 with strict typing requirements in `pyproject.toml`.

---

## 9. Antigravity Agent Guidelines & Quality Guardrails Review

### Audit of `AGENTS.md`
* **Strengths**: Clearly establishes zero-cost constraint, hardware awareness (CPU vs. remote Colab), and role hierarchy.
* **Enhancements Required**:
  1. Add explicit clause prohibiting agents from inventing synthetic visual anchor boxes.
  2. Add explicit clause prohibiting hardcoding calibrated weights or thresholds.
  3. Require independent audit sign-off before phase transitions.
  4. Require license review for all recommended packages.

---

## 10. Classified Issues & Defect Log

| Issue ID | Severity | Affected Component | Problem & Evidence | Impact | Recommended Correction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISSUE-01** | **CRITICAL** | `localization/detector.py`, `FS-002` | Proposal to generate synthetic quadrant boxes when detector returns zero lesions. | Fabricates visual evidence, violating the core grounding rule. | Strip artificial quadrant generation. Allow $R = \emptyset$. |
| **ISSUE-02** | **CRITICAL** | `docs/architecture/adr/0004` | Assigning $\rho = 1.0$ to negative absence claims and using regex as absence proof. | Artificially inflates confidence for false negatives; distorts calibration. | Update ADR 0004 to "Proposed"; define multi-tier schema; keep decision open. |
| **ISSUE-03** | **HIGH** | `docs/architecture/adr/0003` | Ultralytics YOLOv8 carries an AGPL-3.0 license, conflicting with root MIT license. | Copyleft viral licensing forces entire project to AGPL. | Adopt Apache 2.0 RT-DETR or Torchvision Faster R-CNN, or update root license. |
| **ISSUE-04** | **HIGH** | `verification/confidence_gate.py`, `FS-008` | Naive lexical string matching for self-consistency agreement and entropy. | Overestimates uncertainty; violates Kuhn et al. semantic clustering. | Implement semantic equivalence clustering in `FS-008`. |
| **ISSUE-05** | **HIGH** | `docs/architecture/adr/0005` | Hardcoding default weights ($w_1=0.4, w_2=0.3, w_3=0.3$) and $\tau=0.65$. | Violates paper: parameters must be empirically fitted on held-out split. | Mark weights as calibration variables to be fitted on SLAKE validation set. |
| **ISSUE-06** | **HIGH** | `api/remote_worker.py`, Colab plan | Attempting to hold Qwen2.5-VL-3B, MedGemma-4B, and MedSAM in GPU VRAM concurrently. | Exceeds 15.3 GB VRAM of free Colab T4, triggering fatal CUDA OOM crashes. | Implement sequential on-demand model loading or run MedSAM on CPU. |
| **ISSUE-07** | **MEDIUM** | `datasets/`, `FS-001` | Modality mismatch in SLAKE (CXR detector applied to brain/abdomen). | Detector failure on non-chest images invalidates grounding tests. | Separate into Track A (CXR only), Track B (Oracle GT boxes across all SLAKE), Track C (Ungrounded). |
| **ISSUE-08** | **MEDIUM** | `datasets/` | Patient-level data leakage in RSNA & VinDr-CXR splits. | Optimistically biased detector evaluation. | Enforce patient-level grouped splits (`patient_id`). |
| **ISSUE-09** | **MEDIUM** | `docs/architecture/adr/0005` | Incomparability between $N=3$ and $N=5$ confidence gate calibration. | Normalized entropy scale differs between sample sizes. | Standardize benchmark on $N=5$; provide dual threshold calibration if $N=3$ is used in UI. |
| **ISSUE-10** | **LOW** | Python environment | Bundled `setuptools 65.5.0` had 9 CVE vulnerabilities. | Security vulnerability flag in static audit. | Upgraded to `setuptools>=84.2.1`. Resolved. |

---

## 11. Phase 1 Readiness Assessment

* **Phase 0 Status**: **Verified and Ready for Formal Correction**.
* **Transition Decision**: Once the documentation and specifications are corrected according to the audit findings (Sections 7 & 8), Phase 0 will be fully approved.
* **Phase 1 Implementation**: **BLOCKED** pending user approval of this audit and approval of open research decisions.

---

## 12. Sign-Off & Approvals

* **Auditor**: Lead AI Research Engineer & Systems Architect
* **Status**: Awaiting User Review and Decision on Open Questions.
