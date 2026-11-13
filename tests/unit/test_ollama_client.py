"""Unit tests for OllamaVLMClient (mocked HTTP, no live Ollama required)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from PIL import Image

from vlm.ollama_client import OllamaVLMClient


class _FakeResponse:
    """Context-manager-style fake urllib response."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _mock_urlopen(payload: dict):
    """Return a mock urlopen that yields the given payload."""

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(payload)

    return fake_urlopen


def test_ollama_client_properties():
    client = OllamaVLMClient()
    assert client.model_name == "ollama-qwen2.5vl:3b"
    assert client.device == "ollama"
    assert client.last_metadata == {}


def test_ollama_client_custom_model():
    client = OllamaVLMClient(model="medgemma:4b")
    assert client.model_name == "ollama-medgemma:4b"


def test_generate_returns_response_text():
    payload = {"response": '{"answer": "MRI", "findings": []}', "done": True}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)):
        client = OllamaVLMClient()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = client.generate(marked_image=image, question="What modality?")
        assert result == '{"answer": "MRI", "findings": []}'
        assert client.last_metadata["model"] == "qwen2.5vl:3b"
        assert client.last_metadata["done"] is True


def test_generate_structured_parses_json():
    payload = {
        "response": '{"answer": "Brain MRI", "findings": [{"claim": "tumor", "region_id": "R1", "status": "present", "confidence": "high"}]}',
        "done": True,
    }
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)):
        client = OllamaVLMClient()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        structured = client.generate_structured(
            marked_image=image,
            question="What do you see?",
            region_ids=["R1", "R2", "NONE"],
        )
        assert structured.answer == "Brain MRI"
        assert len(structured.findings) == 1
        assert structured.findings[0].claim == "tumor"
        assert structured.findings[0].region_id == "R1"
        assert structured.findings[0].status == "present"
        assert structured.findings[0].confidence == "high"


def test_generate_structured_invalid_region_mapped_to_none():
    payload = {
        "response": '{"answer": "Unclear", "findings": [{"claim": "something", "region_id": "INVALID_REGION", "status": "uncertain", "confidence": "low"}]}',
        "done": True,
    }
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)):
        client = OllamaVLMClient()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        structured = client.generate_structured(
            marked_image=image,
            question="What do you see?",
            region_ids=["R1", "NONE"],
        )
        assert structured.findings[0].region_id == "NONE"


def test_sample_n_returns_n_responses():
    payload = {"response": '{"answer": "sample", "findings": []}', "done": True}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)):
        client = OllamaVLMClient()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        responses = client.sample_n(marked_image=image, question="Q", n=3, temperature=0.7)
        assert len(responses) == 3
        for r in responses:
            assert r == '{"answer": "sample", "findings": []}'


def test_generate_type_error_on_non_image():
    client = OllamaVLMClient()
    with pytest.raises(TypeError, match="marked_image must be a PIL.Image.Image"):
        client.generate(marked_image="not an image", question="Q")


def test_sample_n_raises_on_zero_n():
    client = OllamaVLMClient()
    image = Image.new("RGB", (100, 100), color=(128, 128, 128))
    with pytest.raises(ValueError, match="n must be >= 1"):
        client.sample_n(marked_image=image, question="Q", n=0)


def test_image_converted_to_rgb():
    payload = {"response": '{"answer": "ok", "findings": []}', "done": True}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)):
        client = OllamaVLMClient()
        image = Image.new("L", (100, 100), color=128)  # Grayscale
        result = client.generate(marked_image=image, question="Q")
        assert result == '{"answer": "ok", "findings": []}'


def test_metadata_recorded_after_generate():
    payload = {
        "response": '{"answer": "x", "findings": []}',
        "done": True,
        "total_duration": 12_000_000_000,
        "load_duration": 2_000_000_000,
        "prompt_eval_count": 200,
        "eval_count": 30,
        "eval_duration": 8_000_000_000,
    }
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payload)), patch(
        "vlm.ollama_client.time.perf_counter", side_effect=[1000.0, 1002.5]
    ):
        client = OllamaVLMClient()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        client.generate(marked_image=image, question="Q")
        meta = client.last_metadata
        assert meta["model"] == "qwen2.5vl:3b"
        assert meta["host"] == "http://localhost:11434"
        assert meta["wall_time_s"] == 2.5
        assert meta["total_duration_s"] == 12.0
        assert meta["load_duration_s"] == 2.0
        assert meta["prompt_eval_count"] == 200
        assert meta["eval_count"] == 30
        assert meta["eval_duration_s"] == 8.0
        assert meta["done"] is True
