# Region-Grounded and Uncertainty-Aware Medical Visual Question Answering

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)

A machine-checkable, uncertainty-aware Visual Question Answering system for medical images that enforces visual grounding and enables selective prediction (abstention).

Based on the research paper:
> **"Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models"**  
> *Abisha K M, Anusiya S, Bhavya Sree P R, Dirisala Kavya (RMK Engineering College)*  
> Mentor: Mr. Sathesh Abraham Leo E, Assistant Professor

---

## 📌 Core Research Concept: "Only Claim What Can Be Pointed To"

Vision-language models (VLMs) can generate fluent radiology reports, but frequently hallucinate abnormalities, fail to link assertions to spatial image evidence, and exhibit overconfidence without expressing doubt.

This system introduces a 6-stage pipeline:
1. **Region Localization**: A detector (YOLO/DETR) proposes candidate bounding boxes; MedSAM extracts pixel-level segmentation masks ($R_1 \dots R_k$).
2. **Visual Markings**: Bounding IDs ($R_1 \dots R_k$) are overlaid onto the image.
3. **Structured JSON Contract**: VLMs (**Qwen2.5-VL-3B** and **MedGemma-4B**) answer strictly in JSON citing valid region IDs.
4. **Claim Verification**: Findings without valid region IDs or failing spatial IoU thresholds ($\ge 0.5$) are filtered.
5. **Confidence Gating & Abstention**: Combines majority agreement ($a$), semantic entropy ($H_n$), and detector score ($\rho$):
   $$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
   If $c \ge \tau$, the answer is returned; otherwise, the system safely abstains ("uncertain").
6. **Side-by-Side Evaluation**: Direct benchmarking across SLAKE and VQA-RAD measuring accuracy, unsupported-claim rate (USR), pointing accuracy, and calibration (ECE).

---

## 🏗️ System Architecture

```
[Medical Image + Question]
            │
            ▼
┌─────────────────────────┐
│   Region Localizer      │ ──► [YOLO Detection + MedSAM Segmentation]
│   (Candidate Regions R) │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Visual Marker Engine  │ ──► [Set-of-Mark Overlay R1..Rk]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Dual-VLM Reasoning    │ ──► [Qwen2.5-VL-3B & MedGemma-4B via HF / Remote Worker]
│   (JSON Output Schema)  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Claim Verifier        │ ──► [Filters ungrounded claims; checks IoU ≥ 0.5]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Confidence Gate       │ ──► [c = w1*a + w2*(1-Hn) + w3*ρ]
│   (Selective Prediction)│
└───────────┬─────────────┘
            ├── c ≥ τ ──► Return Grounded Answer + Overlays
            └── c < τ ──► Abstain ("Uncertain - Clinical Review Recommended")
```

---

## 💻 Environment & Hardware Constraints

This project is engineered to run **100% free of charge** under realistic hardware limitations:
* **Local Laptop (Intel Core i5-1235U, 16 GB RAM, Intel Iris Xe)**:
  - Handles API orchestration, React frontend, dataset preprocessing, claim verification, confidence scoring, metrics calculation, and unit testing.
* **Remote Free GPU (Google Colab / Kaggle T4 or L4)**:
  - Runs heavy VLM inference (Qwen2.5-VL-3B and MedGemma-4B in 4-bit `bitsandbytes` precision) via a decoupled remote worker script (`api/remote_worker.py`).
  - Core application interacts with the model via a unified `VLMClient` interface with deterministic mock support for offline local development.

---

## 🚀 Quick Start (Development Setup)

### 1. Prerequisites
* Python 3.11 (`python --version`)
* Node.js v18+ (`node --version`)
* Git

### 2. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/honors-project/MedicalVQA.git
cd MedicalVQA

# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate   # Windows

# Install development & verification dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 3. Verify Local Installation
```bash
# Run test suite
pytest

# Run static quality check
ruff check .
```

---

## 📚 Documentation Index

Detailed documentation is available in `docs/`:
* [Research Requirements & Traceability](docs/research/requirements_traceability.md)
* [Mathematical Formulations](docs/research/mathematical_formulations.md)
* [Dataset Catalog & Licenses](docs/research/dataset_catalog.md)
* [System Architecture](docs/architecture/system_architecture.md)
* [Data Flow Specification](docs/architecture/data_flow.md)
* [Architecture Decision Records (ADRs)](docs/architecture/adr/)
* [Feature Specifications Index](docs/specifications/INDEX.md)
* [Implementation Roadmap](docs/roadmap/implementation_roadmap.md)
* [Agent Collaboration Guidelines](AGENTS.md)

---

## 📄 Citation

```bibtex
@inproceedings{abisha2026regiongrounded,
  title={Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models},
  author={Abisha, K M and Anusiya, S and Bhavya Sree, P R and Dirisala, Kavya},
  booktitle={Department of Artificial Intelligence and Data Science, RMK Engineering College},
  year={2026}
}
```

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
