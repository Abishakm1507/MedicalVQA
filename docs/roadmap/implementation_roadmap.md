# MedicalVQA Development Roadmap & Milestones

This roadmap divides the full research engineering lifecycle into clearly defined phases. Each phase is independently testable, decoupled, and directly traceable to specific sections of the research paper.

---

## 🗺️ Roadmap Overview

```
Phase 0: Project Foundation (Current Phase)
   │
   ▼
Phase 1: Data Pipeline & Localizer Subsystem
   │  ├── Ingestion of SLAKE, VQA-RAD, RSNA
   │  ├── YOLOv8 Candidate Region Detector
   │  └── MedSAM Promptable Segmentation
   │
   ▼
Phase 2: Dual-VLM Adapters & Verification Core
   │  ├── Unified VLMClient (Mock & Remote)
   │  ├── Structured JSON Schema Contract
   │  ├── Claim Verification & Pointing IoU
   │  └── Confidence Gate & Abstention Engine
   │
   ▼
Phase 3: Backend API & Remote Cloud GPU Server
   │  ├── FastAPI POST /analyze Orchestration
   │  ├── Colab/Kaggle 4-bit Worker & Free Tunnels
   │  └── End-to-End Local Mock Validation
   │
   ▼
Phase 4: Interactive React Web Application
   │  ├── HTML5 Canvas Mask & Region Overlays
   │  ├── Side-by-Side Dual VLM View
   │  └── Dynamic Confidence Badges & Alerts
   │
   ▼
Phase 5: Evaluation, Robustness & Paper Results
   │  ├── 3-Condition Benchmarking (C1, C2, C3)
   │  ├── Perturbation Suite & Region Ablation
   │  ├── ECE, AUROC, Accuracy-vs-Coverage Curves
   │  └── LaTeX Table & Matplotlib Plot Export
```

---

## 📌 Detailed Phase Specifications

### Phase 0: Project Foundation (Executed)
* **Paper Sections Addressed**: Sections I, II, III, IV, V, VI (Global alignment).
* **Objectives**: Repository setup, research paper audit, open research question resolution, technical stack selection, ADRs, AGENTS.md, feature specification backlog, and Python 3.11 environment setup.
* **Deliverables**: Complete documentation tree, `.venv`, `AGENTS.md`, `pyproject.toml`, `requirements-dev.txt`, ADRs 0001–0005, FS-001–FS-014.
* **Acceptance Criteria**: All foundation files in place; Python 3.11 verified; zero test dependencies broken.

---

### Phase 1: Data Pipeline & Localizer Subsystem
* **Paper Sections Addressed**: Section III-B (*Region Proposal*), Section V-A (*Datasets*).
* **Feature Specs**: `FS-001` (Datasets), `FS-002` (Detector), `FS-003` (MedSAM), `FS-004` (Visual Marker).
* **Key Tasks**:
  1. Build streaming dataset loaders for SLAKE and VQA-RAD with synthetic unit test fixtures.
  2. Implement `RegionDetector` wrapping YOLOv8 for chest radiograph candidate box proposal.
  3. Integrate `MedSAM` promptable segmentation converting bounding boxes to binary masks.
  4. Implement `VisualMarkerEngine` rendering Set-of-Mark visual overlays ($R_1 \dots R_k$) onto images.
* **Required Tests**: Unit tests in `tests/unit/test_dataset_loaders.py`, `test_detector.py`, `test_visual_marker.py`.
* **Deliverables**: Working localization and marking pipeline running locally on CPU.
* **Duration**: Milestone 1.

---

### Phase 2: Dual-VLM Adapters & Verification Core
* **Paper Sections Addressed**: Section III-C (*Structured Answers*), Section III-D (*Claim Verification*), Section III-E (*Confidence Gate*).
* **Feature Specs**: `FS-005` (VLM Client), `FS-006` (JSON Contract), `FS-007` (Claim Verifier), `FS-008` (Confidence Gate).
* **Key Tasks**:
  1. Implement `VLMClient` abstract interface with `MockVLMClient` and `RemoteVLMClient`.
  2. Implement strict Pydantic v2 schemas (`VLMStructuredOutput`, `VLMClaim`) and parsing logic.
  3. Implement `ClaimVerifier` enforcing region ID validity and negation `NONE` handling.
  4. Implement `ConfidenceGate` calculating agreement $a$, normalized entropy $H_n$, localizer score $\rho$, and fused confidence $c = w_1 a + w_2 (1 - H_n) + w_3 \rho$.
* **Required Tests**: Mathematical property tests for $H_n \in [0, 1]$, $c \in [0, 1]$, and synthetic claim dropping.
* **Deliverables**: Verification and gating core with $>90\%$ test coverage.
* **Duration**: Milestone 2.

---

### Phase 3: Backend API & Remote Cloud GPU Server
* **Paper Sections Addressed**: Section IV-A (*Architecture*), Section IV-B (*Models and Deployment*).
* **Feature Specs**: `FS-011` (FastAPI), `FS-013` (Remote GPU Worker).
* **Key Tasks**:
  1. Build FastAPI application with endpoint `POST /api/v1/analyze`.
  2. Implement `api/remote_worker.py` and `notebooks/colab_inference_server.ipynb` loading Qwen2.5-VL-3B and MedGemma-4B in 4-bit `bitsandbytes` precision on free Colab/Kaggle GPUs.
  3. Configure secure reverse tunneling (Cloudflare Tunnel or ngrok) with shared token auth.
  4. Create `Dockerfile` and `docker-compose.yml`.
* **Required Tests**: FastAPI `TestClient` integration tests for `/analyze` and `/health`.
* **Deliverables**: Fully functional backend API operable locally via mock or connected to free remote GPU.
* **Duration**: Milestone 3.

---

### Phase 4: Interactive React Web Application
* **Paper Sections Addressed**: Section IV-A (*Architecture*), Figure 2.
* **Feature Specs**: `FS-012` (React Web App).
* **Key Tasks**:
  1. Scaffold React 18 + Vite + Tailwind CSS project in `web/`.
  2. Implement `ImageUploader` with client-side preview.
  3. Implement HTML5 `CanvasOverlay` rendering bounding boxes, MedSAM masks, and region tags.
  4. Implement `SideBySideVLM` comparison layout and dynamic `ConfidenceBadge`.
  5. Implement abstention warning banners when $c < \tau$.
* **Required Tests**: Component unit tests and Playwright UI integration tests.
* **Deliverables**: Working web UI served on `localhost:5173`.
* **Duration**: Milestone 4.

---

### Phase 5: Evaluation, Robustness & Paper Results
* **Paper Sections Addressed**: Section V (*Evaluation Design*), Section VI (*Discussion*).
* **Feature Specs**: `FS-009` (Robustness), `FS-010` (Metrics), `FS-014` (Expert / Judge Review).
* **Key Tasks**:
  1. Execute benchmark runner across the 3 conditions (Free-form, Grounded JSON, Grounded + Gate) on SLAKE and VQA-RAD test splits.
  2. Run Albumentations perturbation suite (noise, contrast, rotation, horizontal flip with lateral inversion).
  3. Run masked-region visual ablation test.
  4. Compute ECE across $M=10$ bins, AUROC, and plot Accuracy-vs-Coverage curves.
  5. Run 1,000 bootstrap iterations for 95% CIs and paired SciPy hypothesis tests.
  6. Generate publication-ready LaTeX tables (`Table 1`, `Table 2`) and high-resolution figures.
* **Deliverables**: Complete reproducible experimental results and paper-ready tables.
* **Duration**: Milestone 5.
