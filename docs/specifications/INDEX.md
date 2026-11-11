# Feature Specifications Index

This catalog indexes all feature specifications for the MedicalVQA system. Every implementation task must trace back to a specification in this index.

| Spec ID | Feature Title | Research Paper Section | Target Module | Assigned Role | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FS-001** | Dataset Ingestion & Standardized Loaders | Section V-A, Table 1 | `datasets/` | ML Research Engineer | Approved |
| **FS-002** | Region Localization & Detector Pipeline | Section III-B | `localization/detector.py` | ML Research Engineer | Approved |
| **FS-003** | MedSAM Segmentation Integration | Section III-B | `localization/segmenter.py` | ML Research Engineer | Approved |
| **FS-004** | Set-of-Mark Visual Overlay Engine | Section III-C | `localization/visual_marker.py` | Backend Engineer | Approved |
| **FS-005** | Unified VLM Client & Adapter Interface | Section III-C, IV-B | `vlm/base.py`, `vlm/` | ML Research Engineer | Approved |
| **FS-006** | Structured JSON Contract & Pydantic Validation | Section III-C | `core/schemas.py` | Backend Engineer | Approved |
| **FS-007** | Claim Verification & Spatial IoU Engine | Section III-D | `verification/claim_verifier.py` | Backend Engineer | Approved |
| **FS-008** | Confidence Gate & Selective Abstention | Section III-E, Eq. 2 | `verification/confidence_gate.py`| Backend Engineer | Approved |
| **FS-009** | Robustness Perturbation & Region Ablation Suite | Section V-D | `evaluation/robustness.py` | QA & Evaluation Engineer | Approved |
| **FS-010** | Evaluation Metrics & Statistical Benchmark Harness | Section V-B, V-C, Table 2 | `evaluation/metrics.py`, `evaluation/runner.py` | QA & Evaluation Engineer | Approved |
| **FS-011** | FastAPI Backend Service (`POST /analyze`) | Section IV-A | `api/` | Backend Engineer | Approved |
| **FS-012** | React Interactive Web Application | Section IV-A | `web/` | Frontend Engineer | Approved |
| **FS-013** | Remote Cloud GPU Inference Worker | Section IV-B | `api/remote_worker.py` | ML Research Engineer | Approved |
| **FS-014** | Expert Review Rubric & LLM-as-a-Judge Harness | Section V-E | `evaluation/expert_review.py` | QA & Evaluation Engineer | Approved |
