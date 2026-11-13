"""Qwen2.5-VL-3B VLM client.

Loads Qwen2.5-VL-3B-Instruct via Hugging Face Transformers with optional
4-bit quantization for GPU inference. CPU fallback supported (no quantization).

Traceability:
  - Feature Specification: FS-005 (Unified VLM Interface & Adapters)
  - Research Paper: Section IV-B (Models and Deployment)
"""


from PIL import Image

from vlm.base import VLMClient


class QwenVLMClient(VLMClient):
    """Qwen2.5-VL-3B vision-language model client.

    Loads the model lazily on first inference call. Supports GPU with 4-bit
    quantization and CPU fallback.

    Args:
        model_id: Hugging Face model ID or local path (default: 'Qwen/Qwen2.5-VL-3B-Instruct').
        device: Inference device ('cpu' or 'cuda').
        quantization: '4bit' for GPU 4-bit quantization, or None for full precision.
        max_memory: Optional max memory mapping per device.
        trust_remote_code: Whether to trust remote code (required for Qwen2.5-VL).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cpu",
        quantization: str | None = None,
        max_memory: dict | None = None,
        trust_remote_code: bool = True,
        revision: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._quantization = quantization
        self._max_memory = max_memory
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return "qwen2.5-vl-3b"

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the model and processor lazily."""
        if self._model is not None:
            return

        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        # Configure quantization for GPU
        quantization_config = None
        if self._device.startswith("cuda") and self._quantization == "4bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Determine torch dtype
        if self._device.startswith("cuda"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        load_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self._trust_remote_code,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif self._max_memory is not None:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = self._max_memory
        else:
            load_kwargs["device_map"] = {"": self._device}

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=self._trust_remote_code,
            revision=self._revision,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id,
            revision=self._revision,
            **load_kwargs,
        )
        assert self._model is not None

        if not self._device.startswith("cuda") and self._quantization is None:
            self._model.to(self._device)

        self._model.eval()

    # ------------------------------------------------------------------
    # VLMClient interface
    # ------------------------------------------------------------------

    def generate(
        self,
        marked_image: Image.Image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate a single response from Qwen2.5-VL-3B."""
        self._load()
        assert self._model is not None
        assert self._processor is not None

        import torch

        # Build Qwen2.5-VL chat messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical visual question answering assistant. "
                            "Analyze the medical image and answer the question accurately. "
                            "Respond ONLY with a valid JSON object matching this schema: "
                            '{"answer": "string", "findings": [{"claim": "string", '
                            '"region_id": "string", "status": "present|absent|uncertain", '
                            '"confidence": "high|medium|low"}]}'.replace(
                                "'", '"'
                            )
                            + ". "
                            "Ground findings to candidate region IDs visible in the image. "
                            "Use 'NONE' for region_id if no region is relevant. "
                            "Do not include any text outside the JSON object."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": marked_image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[marked_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (exclude input)
        input_length = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_length:]
        response = self._processor.tokenizer.decode(new_ids, skip_special_tokens=True)

        return response.strip()

    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Generate N sampled responses for self-consistency."""
        self._load()
        assert self._model is not None
        assert self._processor is not None

        import torch

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical visual question answering assistant. "
                            "Analyze the medical image and answer the question accurately. "
                            "Respond ONLY with a valid JSON object matching this schema: "
                            '{"answer": "string", "findings": [{"claim": "string", '
                            '"region_id": "string", "status": "present|absent|uncertain", '
                            '"confidence": "high|medium|low"}]}'.replace(
                                "'", '"'
                            )
                            + ". "
                            "Ground findings to candidate region IDs visible in the image. "
                            "Use 'NONE' for region_id if no region is relevant. "
                            "Do not include any text outside the JSON object."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": marked_image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[marked_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        results: list[str] = []
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        # output_ids shape: (n * num_sequences, seq_len) — with num_return_sequences=n,
        # output_ids has n rows
        for i in range(n):
            new_ids = output_ids[i][input_length:]
            response = self._processor.tokenizer.decode(
                new_ids, skip_special_tokens=True
            )
            results.append(response.strip())

        return results
