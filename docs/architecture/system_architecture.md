# System Architecture Specification

## 1. High-Level Architectural Pattern: Decoupled Multimodal Microservices

The application is engineered as a decoupled, asynchronous system designed to bridge resource-constrained edge/laptop development with high-throughput cloud GPU inference.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Presentation Layer                            │
│           React 18 + Vite + Tailwind CSS + HTML5 Canvas                │
│       (Interactive Image Upload, Visual Overlay, Dual Comparison)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           API Gateway                                  │
│             FastAPI + Pydantic v2 + Uvicorn (Port 8000)                │
│                 Endpoint: POST /api/v1/analyze                         │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │     Localization Engine       │   │    Verification & Gate Engine │
    │ ┌───────────────────────────┐ │   │ ┌───────────────────────────┐ │
    │ │ YOLOv8 Bounding Detector  │ │   │ │ Pydantic JSON Validator   │ │
    │ └─────────────┬─────────────┘ │   │ └─────────────┬─────────────┘ │
    │               ▼               │   │               ▼               │
    │ ┌───────────────────────────┐ │   │ │ Claim Grounding & IoU     │ │
    │ │ MedSAM Promptable Masks   │ │   │ └─────────────┬─────────────┘ │
    │ └─────────────┬─────────────┘ │   │               ▼               │
    │               ▼               │   │ │ Confidence Gating (c ≥ τ) │ │
    │ ┌───────────────────────────┐ │   │ └───────────────────────────┘ │
    │ │ Set-of-Mark Visual Overlay│ │   └───────────────────────────────┘
    │ └───────────────────────────┘ │
    └───────────────┬───────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     VLM Inference Abstraction Layer                   │
│                        (Base Class: VLMClient)                         │
├───────────────────────────────┬────────────────────────────────────────┤
│ Local Mock Engine             │ Remote GPU Worker (Colab / Kaggle)     │
│ - Deterministic offline testing│ - Qwen2.5-VL-3B (4-bit bitsandbytes)  │
│ - Zero GPU / CPU requirement  │ - MedGemma-4B (4-bit bitsandbytes)     │
│ - Instant pytest execution    │ - Secure reverse tunnel (Cloudflare)   │
└───────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### 2.1 Localization & Visual Marking Subsystem (`localization/`)
- **Detector (`detector.py`)**: Accepts an RGB medical image and generates top-$k$ candidate bounding boxes $b_i = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ with detector scores $\rho_i \in [0, 1]$. Limits $k \le 7$ to avoid visual saturation.
- **Segmenter (`segmenter.py`)**: Feeds bounding boxes $b_i$ as prompts to MedSAM to extract high-resolution binary segmentation masks $m_i \in \{0, 1\}^{H \times W}$.
- **Marker (`visual_marker.py`)**: Renders semi-transparent color-coded bounding boxes and badge pills labeled `R1`, `R2`, ..., `Rk` directly onto a copy of the input image using Pillow/OpenCV for VLM ingestion.

### 2.2 VLM Ingestion & Remote Adapter Subsystem (`vlm/`)
- Implements an abstract `VLMClient` base interface with methods:
  - `generate(image: Image, prompt: str, temperature: float) -> str`
  - `sample_n(image: Image, prompt: str, n: int, temperature: float) -> List[str]`
- Implementations:
  - `MockVLMClient`: Reads test fixtures for lightning-fast, reproducible local unit tests.
  - `RemoteVLMClient`: Makes authenticated asynchronous REST calls to `remote_worker.py` running on a free Google Colab/Kaggle GPU.

### 2.3 Verification & Confidence Gating Subsystem (`verification/`)
- **JSON Validator (`json_validator.py`)**: Enforces strict schema conformity using Pydantic. Rejects non-JSON or improperly structured responses.
- **Claim Grounding (`claim_verifier.py`)**:
  - Matches each finding's `region_id` against the active candidate set $R = \{R_1, \dots, R_k\}$.
  - Drops claims that cite non-existent regions or fail the negation rule.
  - Computes spatial IoU against ground truth during benchmark runs.
- **Confidence Gate (`confidence_gate.py`)**:
  - Computes consensus agreement $a$ and normalized entropy $H_n$ across the $N$ sampled responses.
  - Retrieves localizer score $\rho$ for the cited region.
  - Computes fused score $c = w_1 a + w_2 (1 - H_n) + w_3 \rho$.
  - Applies selective abstention decision ($\hat{y}$ vs. `"uncertain"`).

### 2.4 Presentation Layer (`web/`)
- **Canvas Overlay (`CanvasOverlay.jsx`)**: HTML5 Canvas rendering original image, bounding box contours, MedSAM masks, and region tags with interactive hover states.
- **Side-by-Side Comparator (`SideBySideVLM.jsx`)**: Displays Qwen2.5-VL-3B and MedGemma-4B outputs side-by-side with color-coded confidence badges (`High`, `Moderate`, `Low`, `Abstained`).
