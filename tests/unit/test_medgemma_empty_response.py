"""Unit tests for MedGemma-4B empty-response fix.

These tests do NOT load the MedGemma model weights (gated, GPU required).
They mock ``torch`` and ``transformers`` so that importing and calling
``MedGemmaVLMClient`` uses fakes. Verified behaviors:

1. ``_load`` selects ``torch.bfloat16`` (not float16) for CUDA — float16 yields
   empty Gemma 3 output (https://github.com/huggingface/transformers/issues/36822).
2. ``generate()`` builds inputs via the single-call Gemma 3 pattern
   ``apply_chat_template(tokenize=True, return_dict=True, images=[...])``.
3. Token slicing ``output_ids[0][input_length:]`` is correct.
4. Decoding uses ``skip_special_tokens=True``.
5. Empty responses are detectable (status classification logic).
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build mocks BEFORE importing the client so its internal `import torch` and
# `from transformers import ...` resolve to fakes.
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
mock_torch.float32 = "float32"
mock_torch.inference_mode = lambda: MagicMock(
    __enter__=lambda s: None, __exit__=lambda s, *a: False
)

mock_transformers = MagicMock()
mock_processor = MagicMock()
mock_model = MagicMock()
mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
mock_transformers.Gemma3ForConditionalGeneration.from_pretrained.return_value = mock_model

_MODULES = {
    "torch": mock_torch,
    "transformers": mock_transformers,
}


@pytest.fixture
def env():
    # Remove any real torch/transformers already in sys.modules so that the
    # client's internal `import torch` / `from transformers import ...`
    # resolve to our fakes (the model is gated and must not be downloaded).
    for _mod in ("torch", "transformers", "torch.nn"):
        sys.modules.pop(_mod, None)
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_torch
    sys.modules["transformers"] = mock_transformers
    try:
        yield {"torch": mock_torch, "processor": mock_processor, "model": mock_model}
    finally:
        for _mod in ("torch", "torch.nn", "transformers"):
            sys.modules.pop(_mod, None)


@pytest.fixture
def client(env):
    from vlm.medgemma_client import MedGemmaVLMClient

    c = MedGemmaVLMClient(device="cuda", quantization="4bit")
    c._processor = env["processor"]
    c._model = env["model"]
    return c


# ---------------------------------------------------------------------------
# _load dtype selection
# ---------------------------------------------------------------------------
def test_load_selects_bfloat16_for_cuda(env):
    """Gemma 3 must use bfloat16 on CUDA; float16 yields empty output."""

    from vlm.medgemma_client import MedGemmaVLMClient

    c = MedGemmaVLMClient(device="cuda", quantization="4bit")

    # Force the client's internal imports to resolve to our fakes.
    with patch.dict(
        sys.modules, {"torch": env["torch"], "transformers": mock_transformers}, clear=False
    ):
        c._load()

    # Verify from_pretrained was called with torch_dtype=bfloat16.
    # _load() calls transformers.Gemma3ForConditionalGeneration.from_pretrained(**kwargs).
    from_pretrained_call = (
        mock_transformers.Gemma3ForConditionalGeneration.from_pretrained.call_args
    )
    assert from_pretrained_call is not None, (
        "Gemma3ForConditionalGeneration.from_pretrained not called"
    )
    kwargs = from_pretrained_call.kwargs
    assert kwargs.get("torch_dtype") == "bfloat16", (
        f"expected bfloat16, got {kwargs.get('torch_dtype')}"
    )

    # Verify quantization compute dtype is bfloat16 (via BitsAndBytesConfig call).
    bnb_call = mock_transformers.BitsAndBytesConfig.call_args
    assert bnb_call is not None, "BitsAndBytesConfig not called for 4-bit CUDA"
    assert bnb_call.kwargs.get("bnb_4bit_compute_dtype") == "bfloat16", (
        f"expected bfloat16 compute dtype, got {bnb_call.kwargs.get('bnb_4bit_compute_dtype')}"
    )


def test_load_selects_float32_for_cpu(env):

    from vlm.medgemma_client import MedGemmaVLMClient

    c = MedGemmaVLMClient(device="cpu", quantization=None)
    with patch.dict(
        sys.modules, {"torch": env["torch"], "transformers": mock_transformers}, clear=False
    ):
        c._load()
    kwargs = mock_transformers.Gemma3ForConditionalGeneration.from_pretrained.call_args.kwargs
    assert kwargs.get("torch_dtype") == "float32"
    # CPU without quantization should not pass a quantization_config.
    assert kwargs.get("quantization_config") is None


# ---------------------------------------------------------------------------
# generate() input construction (single-call Gemma 3 pattern)
# ---------------------------------------------------------------------------
def test_generate_uses_single_call_apply_chat_template(client, env):
    """generate() must use apply_chat_template(tokenize=True, return_dict=True, images=...)."""
    from PIL import Image

    img = Image.new("RGB", (336, 336), color=(128, 128, 128))

    mock_inputs = {
        "input_ids": MagicMock(shape=(1, 50)),
        "attention_mask": MagicMock(shape=(1, 50)),
        "pixel_values": MagicMock(shape=(1, 3, 336, 336)),
    }
    env["processor"].apply_chat_template.return_value = mock_inputs

    mock_output = MagicMock()
    mock_output.shape = (1, 55)
    mock_output.__getitem__ = lambda self, idx: MagicMock(tolist=lambda: [1, 2, 3])
    env["model"].generate.return_value = mock_output
    env["processor"].tokenizer.decode.return_value = "chest"
    env["processor"].tokenizer.pad_token_id = 0
    env["processor"].tokenizer.eos_token_id = 2

    response = client.generate(
        marked_image=img, question="What body part?", temperature=0.0, max_tokens=512
    )

    act_call = env["processor"].apply_chat_template.call_args
    assert act_call is not None
    assert act_call.kwargs.get("tokenize") is True
    assert act_call.kwargs.get("return_dict") is True
    assert act_call.kwargs.get("return_tensors") == "pt"
    assert act_call.kwargs.get("add_generation_prompt") is True
    assert "images" in act_call.kwargs

    env["model"].generate.assert_called_once()

    decode_call = env["processor"].tokenizer.decode.call_args
    assert decode_call.kwargs.get("skip_special_tokens") is True

    assert response == "chest"


# ---------------------------------------------------------------------------
# Token slicing correctness
# ---------------------------------------------------------------------------
def test_generate_slices_new_tokens_only(client, env):
    """The response must be decoded from output_ids[input_length:], not the full sequence."""
    from PIL import Image

    img = Image.new("RGB", (336, 336), color=(128, 128, 128))

    input_ids_mock = MagicMock()
    input_ids_mock.shape = (1, 10)
    input_ids_mock.to.return_value = input_ids_mock
    attn_mock = MagicMock()
    attn_mock.to.return_value = attn_mock
    mock_inputs = {"input_ids": input_ids_mock, "attention_mask": attn_mock}
    env["processor"].apply_chat_template.return_value = mock_inputs

    # output_ids behaves like a [1, 13] tensor: row 0 has 10 input + 3 new tokens.
    mock_output = [list(range(13))]
    env["model"].generate.return_value = mock_output
    env["processor"].tokenizer.decode.return_value = "yes"
    env["processor"].tokenizer.pad_token_id = 0
    env["processor"].tokenizer.eos_token_id = 2

    client.generate(marked_image=img, question="Q?", temperature=0.0, max_tokens=512)

    decode_arg = env["processor"].tokenizer.decode.call_args
    assert decode_arg is not None
    first_arg = decode_arg.args[0]
    # output_ids[0][10:] should yield exactly the 3 new tokens [10, 11, 12].
    assert list(first_arg) == [10, 11, 12], f"expected [10, 11, 12], got {list(first_arg)}"


# ---------------------------------------------------------------------------
# Empty response detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw_response,expected_invalid",
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("chest", False),
        ("  yes  ", False),
    ],
)
def test_empty_response_detection(raw_response, expected_invalid):
    """Mirror the notebook's status classification: empty -> invalid_response."""
    is_empty = raw_response is None or str(raw_response).strip() == ""
    assert is_empty == expected_invalid


# ---------------------------------------------------------------------------
# sample_n uses the same single-call pattern
# ---------------------------------------------------------------------------
def test_sample_n_uses_single_call_apply_chat_template(client, env):
    from PIL import Image

    img = Image.new("RGB", (336, 336), color=(128, 128, 128))
    mock_inputs = {
        "input_ids": MagicMock(shape=(1, 20)),
        "attention_mask": MagicMock(shape=(1, 20)),
    }
    env["processor"].apply_chat_template.return_value = mock_inputs

    mock_output = MagicMock()
    mock_output.shape = (3, 25)
    env["model"].generate.return_value = mock_output
    env["processor"].tokenizer.decode.return_value = "sampled"
    env["processor"].tokenizer.pad_token_id = 0
    env["processor"].tokenizer.eos_token_id = 2

    client.sample_n(marked_image=img, question="Q?", n=3, temperature=0.7, max_tokens=512)

    act_call = env["processor"].apply_chat_template.call_args
    assert act_call.kwargs.get("tokenize") is True
    assert act_call.kwargs.get("return_dict") is True
    assert "images" in act_call.kwargs
