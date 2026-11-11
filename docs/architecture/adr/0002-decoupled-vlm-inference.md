# ADR 0002: Decoupled VLM Inference Architecture

## Status
Accepted

## Context
The primary development machine is equipped with an Intel Core i5-1235U processor, 16 GB of DDR4 system RAM, and integrated Intel Iris Xe graphics (no dedicated NVIDIA CUDA hardware). Loading Qwen2.5-VL-3B (~3.8 GB) and MedGemma-4B (~4.5 GB) locally under PyTorch CPU execution consumes severe RAM and results in unacceptably high token latency (>45 seconds per query). Furthermore, the project operates under a strict $0.00 budget constraint.

## Decision
We decouple VLM model execution entirely behind an abstract client interface (`VLMClient` in `vlm/base.py`):
1. **Local Mode (`MockVLMClient`)**: Employs deterministic test fixtures for fast unit tests, API routing, verification logic, and frontend development with zero GPU/RAM overhead.
2. **Remote Mode (`RemoteVLMClient`)**: Sends asynchronous authenticated requests over HTTPS to a free Google Colab or Kaggle instance running `api/remote_worker.py` with 4-bit `bitsandbytes` quantization.

## Consequences
- **Positive**: Enables full-stack development, automated testing, and UI polishing locally on the laptop without lag or memory exhaustion.
- **Positive**: Leverages free high-end cloud GPUs (Tesla T4 or L4) for heavy model inference with $0.00 spent.
- **Negative**: Remote inference requires an active Google Colab/Kaggle notebook session and an active tunnel during end-to-end user testing.
