# Feature Specification: FS-005 Unified VLM Interface & Adapters

## 1. Overview & Research Paper Traceability
* **Feature ID**: FS-005
* **Feature Name**: Unified Vision-Language Model Client Interface
* **Paper Section**: Section III-C, Section IV-B (*Models and Deployment*)
* **Module Path**: `vlm/base.py`, `vlm/qwen_client.py`, `vlm/medgemma_client.py`, `vlm/mock_client.py`
* **Assigned Agent**: ML Research Engineer
* **Status**: Approved

---

## 2. Objectives & Scope
* **Objective**: Provide a common, model-agnostic Python interface for prompting both **Qwen2.5-VL-3B** and **MedGemma-4B** with identical prompts, images, sampling parameters, and schema constraints.
* **In-Scope**:
  - `VLMClient` Abstract Base Class.
  - Identical system prompt formatting.
  - Multi-sample generation ($N$ samples at temperature $T=0.7$) for confidence gate calculation.
  - `MockVLMClient` for lightning-fast, CPU-based local testing.
  - `RemoteVLMClient` for connecting over HTTPS to the free Colab/Kaggle worker.

---

## 3. Technical Requirements & Interfaces
```python
from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class VLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model (e.g. 'qwen2.5-vl-3b', 'medgemma-4b')."""
        pass

    @abstractmethod
    def generate(
        self,
        marked_image: Image.Image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        """Generates a single greedy response string adhering to JSON schema."""
        pass

    @abstractmethod
    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> List[str]:
        """Generates N sampled response strings for self-consistency and entropy computation."""
        pass
```
