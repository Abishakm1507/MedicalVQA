# Multi-Agent Collaboration Framework (AGENTS.md)

## 1. Project Overview & Role Hierarchy
This repository implements the research paper:
> **"Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models"**  
> *Abisha K M, Anusiya S, Bhavya Sree P R, Dirisala Kavya (RMK Engineering College)*

To maintain research integrity, reproducibility, and code quality under strict zero-cost and hardware-constrained conditions, AI agents collaborate under specialized roles.

```
                    ┌─────────────────────────────────────────┐
                    │          Lead Coordinator / PM          │
                    │   (Architecture, Roadmap, Integrity)    │
                    └────────────────────┬────────────────────┘
                                         │
        ┌───────────────────┬────────────┴──────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
┌──────────────┐    ┌──────────────┐        ┌──────────────┐    ┌──────────────┐
│  ML Research │    │   Backend    │        │   Frontend   │    │  QA & Audit  │
│   Engineer   │    │   Engineer   │        │   Engineer   │    │   Engineer   │
│ (VLM, Local- │    │  (FastAPI,   │        │ (React, Vite,│    │ (pytest, ECE,│
│izer, MedSAM) │    │ Verification,│        │Canvas, Badge,│    │   Security,  │
│              │    │ Docker, Hub) │        │ Side-by-Side)│    │ Code Review) │
└──────────────┘    └──────────────┘        └──────────────┘    └──────────────┘
```

---

## 2. Specialized Agent Roles & Responsibilities

### 2.1 Lead Coordinator / Software Architect
* **Primary Scope**: Global system design, Feature Specification tracking, Architecture Decision Records (ADRs), interface consistency, and cross-agent coordination.
* **Responsibilities**:
  - Validates that every code change is traceable to a research requirement.
  - Ensures adherence to hardware boundaries (laptop CPU vs. remote free GPU).
  - Approves module handoffs between phases only after independent audits.

### 2.2 ML Research Engineer
* **Primary Scope**: Computer vision and multimodal models (`localization/`, `vlm/`, `datasets/`).
* **Responsibilities**:
  - Implements dataset parsers for SLAKE, VQA-RAD, RSNA, and VinDr-CXR with patient-level leakage prevention.
  - Wraps region detector and MedSAM promptable segmentation without inventing synthetic detections.
  - Implements unified `VLMClient` interfaces for Qwen2.5-VL-3B and MedGemma-4B in 4-bit precision.
  - Authors memory-aware Google Colab / Kaggle inference scripts (`api/remote_worker.py`) that respect the 15.3 GB VRAM ceiling.

### 2.3 Backend Engineer
* **Primary Scope**: Core logic, API routing, verification, and confidence gating (`core/`, `verification/`, `api/`).
* **Responsibilities**:
  - Implements FastAPI server (`POST /analyze`).
  - Implements Pydantic v2 schemas enforcing the JSON response contract.
  - Implements Claim Verification (region ID matching and IoU thresholding).
  - Implements Confidence Gate calculation using semantic clustering:
    $$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
  - Generates Set-of-Mark visual overlays using OpenCV and Pillow.

### 2.4 Frontend Engineer
* **Primary Scope**: Interactive user interface (`web/`).
* **Responsibilities**:
  - Implements React 18 + Vite + Tailwind CSS web application.
  - Creates responsive HTML5 Canvas component for drawing bounding boxes, segmentation masks, and region IDs ($R_1 \dots R_k$).
  - Implements side-by-side comparison views for Qwen2.5-VL-3B vs. MedGemma-4B.
  - Displays real-time confidence badges and visual "Uncertain / Abstention" alerts.

### 2.5 QA, Security & Evaluation Engineer
* **Primary Scope**: Testing, metrics verification, security audits, and statistical evaluation (`evaluation/`, `tests/`).
* **Responsibilities**:
  - Enforces Test-Driven Development (TDD) with pytest and mocked VLM clients.
  - Implements and verifies mathematical metrics: Accuracy, F1, USR, Pointing Accuracy, ECE, AUROC, and Risk-Coverage curves.
  - Implements perturbation robustness suites (Albumentations) and masked-region ablation tests.
  - Executes static security scans (`bandit`, `pip-audit`, secret leak prevention).

---

## 3. Strict Operating Guidelines for All Agents

1. **Zero-Cost Constraint**:
   - Never introduce paid APIs (OpenAI, Anthropic paid keys, etc.).
   - Never use services requiring credit cards.
   - All models, datasets, and tools must be open-source and free.
2. **Hardware Awareness**:
   - Development, testing, backend, frontend, and evaluation run locally on Intel i5 CPU / 16 GB RAM without CUDA.
   - Heavy VLM/MedSAM inference is isolated behind `VLMClient` and executed on free Google Colab/Kaggle GPU tiers or deterministic CPU mocks.
   - Heavy data (`data/raw`, `models/checkpoints`) must reside strictly on drive `D:` and never be committed to Git.
3. **Research Integrity & Anti-Fabrication Rules**:
   - Never fabricate test scores, calibration curves, or benchmark results.
   - **Never invent synthetic visual anchor boxes**: If the localizer finds zero candidate regions, candidate set $R = \emptyset$. Never generate arbitrary quadrant boxes as detections.
   - **Never hardcode calibrated weights or thresholds**: $(w_1, w_2, w_3, \tau)$ must be empirically calibrated on the held-out validation split.
   - Separate hypotheses and expectations from measured empirical results.
   - If an edge case is unresolved, document it explicitly instead of making silent assumptions.
4. **Code Quality, Testing & Licensing**:
   - Every feature must have a corresponding Feature Specification (`docs/specifications/FS-XXX.md`).
   - Every backend module must include unit tests with $\ge 85\%$ coverage before merge.
   - Never suppress or delete failing tests to achieve green builds.
   - Review package licenses before proposing dependencies (avoid viral AGPL-3.0 in MIT components).

---

## 4. Git Branching & Collaboration Workflow

* **Main Branch**: `main` (always working, passing tests, deployable).
* **Development Branch**: `develop` (integration branch for completed phases).
* **Feature Branches**: `feat/FS-XXX-short-name` (branched from `develop`).
* **Audit Branches**: `audit/phase-X-review`.
* **Fix Branches**: `fix/issue-description`.

### Pull Request & Review Checklist
Every agent PR must confirm:
- [ ] Linked Feature Specification ID (`FS-XXX`).
- [ ] Unit tests added and passing locally (`pytest`).
- [ ] Linting and type checking clean (`ruff check .`, `mypy`).
- [ ] Security checks clean (`bandit`, `pip-audit`).
- [ ] No hardcoded paths or API secrets.
- [ ] Memory footprint verified on 16 GB RAM.
