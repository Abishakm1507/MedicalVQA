# Feature Specification: FS-013 Remote Cloud GPU Inference Worker

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-013
* **Feature Name**: Remote Cloud GPU Inference Worker
* **Paper Section**: Section IV-B (*Models and Deployment*)
* **Module Path**: `api/remote_worker.py`, `notebooks/colab_inference_server.ipynb`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Provide an automated, standalone Python server script engineered to run inside free Google Colab (T4/L4 GPU) or Kaggle (P100/T4) notebooks, hosting **Qwen2.5-VL-3B** and **MedGemma-4B** in 4-bit precision via Hugging Face Transformers and `bitsandbytes`.
* **Zero-Cost & Tunneling**:
  - Exposes endpoints:
    - `POST /v1/models/generate`
    - `POST /v1/models/sample_n`
    - `GET /health`
  - Tunnels securely using free Cloudflare Tunnels (`cloudflared`) or free `ngrok` token.
  - Authenticates local API requests with a shared bearer token (`X-Worker-Auth-Key`).
