"""Unit tests for Pydantic and dataclass schemas."""

import pytest
from pydantic import ValidationError

from core.schemas import (
    GroundTruthRegion,
    VLMClaim,
    VLMStructuredOutput,
)


def test_ground_truth_region_valid() -> None:
    region = GroundTruthRegion(label="pneumonia", box_2d=(0.1, 0.2, 0.6, 0.7))
    assert region.label == "pneumonia"
    assert region.box_2d == (0.1, 0.2, 0.6, 0.7)


def test_ground_truth_region_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        # ymax <= ymin
        GroundTruthRegion(label="invalid", box_2d=(0.8, 0.1, 0.2, 0.5))

    with pytest.raises(ValueError):
        # coordinate > 1.0
        GroundTruthRegion(label="out_of_bounds", box_2d=(0.1, 0.2, 1.2, 0.5))


def test_vlm_structured_output_valid() -> None:
    raw_payload = {
        "answer": "Yes, right lower zone opacity.",
        "findings": [
            {"claim": "opacity", "region_id": "R2", "status": "present", "confidence": "high"},
            {
                "claim": "pneumothorax",
                "region_id": "NONE",
                "status": "absent",
                "confidence": "high",
            },
        ],
    }
    parsed = VLMStructuredOutput.model_validate(raw_payload)
    assert parsed.answer == "Yes, right lower zone opacity."
    assert len(parsed.findings) == 2
    assert parsed.findings[0].region_id == "R2"
    assert parsed.findings[0].status == "present"
    assert parsed.findings[1].region_id == "NONE"
    assert parsed.findings[1].status == "absent"


def test_vlm_structured_output_invalid() -> None:
    with pytest.raises(ValidationError):
        # Missing required 'answer'
        VLMStructuredOutput.model_validate({"findings": []})

    with pytest.raises(ValidationError):
        # Invalid confidence string
        VLMClaim.model_validate(
            {"claim": "mass", "region_id": "R1", "status": "present", "confidence": "super_certain"}
        )
