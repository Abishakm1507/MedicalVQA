# Feature Specification: FS-013 Remote Cloud GPU Inference Worker

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-013
* **Feature Name**: Remote Cloud GPU Inference Worker
* **Paper Section**: Section IV-B (*Models and Deployment*)
* **Module Path**: `api/remote_worker.py`, `notebooks/colab_inference_server.ipynb`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved (Revised Post-Audit)

---

## 2. Objectives & Scope
* **Objective**: Standalone server script designed for free Google Colab (Tesla T4) or Kaggle (P100) runtimes, serving **Qwen2.5-VL-3B** and **MedGemma-4B** in 4-bit precision via Hugging Face Transformers and `bitsandbytes`.

---

## 3. Strict GPU Memory Management (15.3 GB VRAM Ceiling)
Because free Google Colab T4 instances provide exactly 15.3 GB of VRAM:
- Attempting to keep Qwen2.5-VL-3B (~5 GB), MedGemma-4B (~6 GB), MedSAM (~3 GB), and PyTorch runtime (~1.5 GB) in VRAM simultaneously consumes ~15.5 GB and risks immediate CUDA OOM crashes.
- **Mandatory Memory Policy**:
  1. The remote worker loads models sequentially on demand. When switching between Qwen2.5-VL-3B and MedGemma-4B, the prior model is unallocated and `torch.cuda.empty_cache()` is called.
  2. MedSAM promptable segmentation can execute locally on the laptop CPU or run on Colab before the VLM is loaded.
  3. Gated model authentication: Accepts Hugging Face token `HF_TOKEN` from environment variables for accessing `google/medgemma-4b-it`.

---

## 4. Technical Requirements & Interfaces
```python
class RemoteModelWorker:
    def __init__(self, auth_token: str, hf_token: Optional[str] = None):
        self.auth_token = auth_token
        self.hf_token = hf_token
        self.active_model_name: Optional[str] = None
        self.model = None
        self.processor = None

    def load_model(self, model_name: str) -> None:
        """Loads requested model in 4-bit, unloading prior model if needed to preserve VRAM."""
        pass

    def generate(self, model_name: str, image_b64: str, prompt: str, temperature: float) -> str:
        pass

    def sample_n(self, model_name: str, image_b64: str, prompt: str, n: int, temperature: float) -> List[str]:
        pass
```
