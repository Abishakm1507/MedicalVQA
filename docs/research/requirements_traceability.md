# Research Paper Requirements Traceability Matrix

This document maps every requirement, equation, model, and experimental condition from the research paper directly to its corresponding software module and Feature Specification (`FS-XXX`).

| Paper Section | Explicit Paper Requirement | Target Software Module | Feature Spec ID | Implementation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Sec III-B** | Localize candidate regions $R = \{r_1, \dots, r_k\}$ with box $b_i$, mask $m_i$, and score $\rho_i$ ($k$ kept small) | `localization/detector.py`, `localization/segmenter.py` | `FS-002`, `FS-003` | Planned |
| **Sec III-B** | Detector trained on RSNA Pneumonia or VinDr-CXR | `datasets/cxr_loader.py`, `localization/detector.py` | `FS-001`, `FS-002` | Planned |
| **Sec III-B** | MedSAM promptable segmentation converting boxes to masks | `localization/segmenter.py` | `FS-003` | Planned |
| **Sec III-C** | Set-of-Mark style region visual labeling ($R_1 \dots R_k$) drawn on image copy | `localization/visual_marker.py` | `FS-004` | Planned |
| **Sec III-C** | Strict JSON schema contract for VLM output: `{"answer": str, "findings": [{"claim": str, "region_id": str, "confidence": str}]}` | `core/schemas.py`, `verification/json_validator.py` | `FS-006` | Planned |
| **Sec III-C** | Identical prompt text provided to both VLMs (Qwen2.5-VL-3B & MedGemma-4B) | `vlm/prompts.py`, `vlm/base.py` | `FS-005` | Planned |
| **Sec III-D** | Claim Verification: Parser rejects non-conforming replies; strips findings citing invalid region IDs | `verification/claim_verifier.py` | `FS-007` | Planned |
| **Sec III-D** | Grounding verification against GT masks/boxes: $\text{IoU} < 0.5$ flags unsupported claims | `verification/claim_verifier.py`, `evaluation/metrics.py` | `FS-007`, `FS-010` | Planned |
| **Sec III-D** | Unsupported-Claim Rate (USR): $\text{USR} = N_{\text{unsup}} / N_{\text{claims}}$ (computed before & after filtering) | `evaluation/metrics.py` | `FS-010` | Planned |
| **Sec III-E** | Self-consistency agreement $a$ across $N$ sampled answers | `verification/confidence_gate.py` | `FS-008` | Planned |
| **Sec III-E** | Normalized semantic entropy $H_n = H(p) / \log N$ over answer clusters | `verification/confidence_gate.py` | `FS-008` | Planned |
| **Sec III-E** | Confidence Gate equation: $c = w_1 a + w_2 (1 - H_n) + w_3 \rho$ | `verification/confidence_gate.py` | `FS-008` | Planned |
| **Sec III-E** | Selective Abstention: Output answer if $c \ge \tau$, otherwise output "uncertain" | `verification/confidence_gate.py`, `api/endpoints.py` | `FS-008`, `FS-011` | Planned |
| **Sec III-E** | Parameter calibration: $(w_1, w_2, w_3, \tau)$ determined on held-out split; report accuracy vs. coverage | `evaluation/calibration.py` | `FS-008`, `FS-010` | Planned |
| **Sec III-F** | Optional case retrieval via BiomedCLIP embeddings and FAISS index | `retrieval/faiss_index.py` | `FS-015` (Optional) | Out of core scope |
| **Sec IV-A** | React + Vite + Tailwind CSS frontend with interactive canvas overlays and confidence badges | `web/src/components/*` | `FS-012` | Planned |
| **Sec IV-A** | Side-by-side comparison view for both VLMs | `web/src/components/SideBySideVLM.jsx` | `FS-012` | Planned |
| **Sec IV-A** | FastAPI backend with single unified endpoint `POST /analyze` validated with Pydantic | `api/app.py`, `api/endpoints.py` | `FS-011` | Planned |
| **Sec IV-B** | Qwen2.5-VL-3B & MedGemma-4B loaded via HF in 4-bit precision (`bitsandbytes`) | `vlm/qwen_client.py`, `vlm/medgemma_client.py`, `api/remote_worker.py` | `FS-005`, `FS-013` | Planned |
| **Sec IV-B** | Docker Compose packaging for frontend, API, and model service | `docker-compose.yml`, `Dockerfile` | `FS-011`, `FS-012` | Planned |
| **Sec V-A** | Ingestion of SLAKE (questions, masks, boxes) and VQA-RAD (questions, answers) | `datasets/slake_loader.py`, `datasets/vqarad_loader.py` | `FS-001` | Planned |
| **Sec V-B** | Evaluation across 3 conditions: Free-form, Grounded JSON contract, Grounded + Confidence Gate | `evaluation/runner.py` | `FS-010` | Planned |
| **Sec V-C** | Pointing accuracy ($\text{IoU} \ge 0.5$) and Closed-question Accuracy & F1 | `evaluation/metrics.py` | `FS-010` | Planned |
| **Sec V-C** | Expected Calibration Error (ECE) across $M$ bins and continuous AUROC | `evaluation/metrics.py` | `FS-010` | Planned |
| **Sec V-D** | Robustness suite: Noise, contrast, JPEG compression, rotation, horizontal flip with lateral query swap | `evaluation/robustness.py` | `FS-009` | Planned |
| **Sec V-D** | Masked-region ablation: Mask out cited region and test if answer changes | `evaluation/robustness.py` | `FS-009` | Planned |
| **Sec V-E** | Clinical rubric evaluation & Cohen's $\kappa$ (with LLM judge fallback) | `evaluation/expert_review.py` | `FS-014` | Planned |
| **Sec V-E** | Bootstrap resampling (1000 iterations) for 95% CIs and paired SciPy hypothesis testing | `evaluation/metrics.py` | `FS-010` | Planned |
