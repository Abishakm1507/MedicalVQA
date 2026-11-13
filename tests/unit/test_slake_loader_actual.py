"""Unit tests for SLAKE loader with actual dataset format (detection.json sidecar)."""

import json
from pathlib import Path

import pytest

from core.schemas import EvaluationTrack, GroundTruthRegion
from datasets.slake_loader import SlakeDatasetLoader


def _make_mini_slake(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal SLAKE-like dataset structure with detection.json sidecars."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    # Create a test split file
    test_entries = [
        {
            "qid": 1001,
            "img_id": 10,
            "img_name": "xmlab10/source.jpg",
            "question": "What modality is this?",
            "answer": "MRI",
            "q_lang": "en",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "content_type": "Modality",
        },
        {
            "qid": 1002,
            "img_id": 10,
            "img_name": "xmlab10/source.jpg",
            "question": "Is there a tumor?",
            "answer": "Yes",
            "q_lang": "en",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Closed",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "content_type": "Abnormality",
        },
        {
            "qid": 1003,
            "img_id": 11,
            "img_name": "xmlab11/source.jpg",
            "question": "What is shown?",
            "answer": "Brain",
            "q_lang": "zh",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "content_type": "Organ",
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    # Create image folders with source.jpg, mask.png, detection.json
    import io
    from PIL import Image

    for patient_id in [10, 11]:
        pdir = imgs_dir / f"xmlab{patient_id}"
        pdir.mkdir()
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(pdir / "source.jpg", format="JPEG")
        img.save(pdir / "mask.png", format="PNG")

        # detection.json: [{"DiseaseName": [x, y, w, h]}, ...]
        if patient_id == 10:
            det = [
                {"Brain Tumor": [30, 40, 20, 30]},
                {"Brain Edema": [50, 60, 10, 15]},
            ]
        else:
            det = [{"Brain Tumor": [10, 10, 5, 5]}]
        (pdir / "detection.json").write_text(
            json.dumps(det), encoding="utf-8"
        )
        (pdir / "question.json").write_text(
            json.dumps(test_entries), encoding="utf-8"
        )

    return dataset_dir, imgs_dir


def test_slake_loader_with_detection_json_sidecar(tmp_path: Path) -> None:
    """Loader reads ground-truth boxes from per-image detection.json sidecar files."""
    dataset_dir, imgs_dir = _make_mini_slake(tmp_path)
    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    # Only English entries (qid 1001, 1002); zh entry (1003) filtered out
    assert len(samples) == 2

    # First sample: image xmlab10, should have 2 GT regions from detection.json
    s1 = samples[0]
    assert s1.sample_id == "slake_test_1001"
    assert s1.image_id == "10"
    assert s1.patient_id is None
    assert s1.modality == "MRI"
    assert s1.organ == "brain"

    # Should have 2 regions from detection.json
    assert len(s1.ground_truth_regions) == 2
    labels = sorted([r.label for r in s1.ground_truth_regions])
    assert labels == ["Brain Edema", "Brain Tumor"]

    # Verify normalized coordinates are in [0, 1]
    for r in s1.ground_truth_regions:
        ymin, xmin, ymax, xmax = r.box_2d
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0

    # Track compatibility: MRI Brain → Track B and Track C, NOT Track A
    assert s1.is_compatible_with(EvaluationTrack.TRACK_B_ORACLE_SLAKE)
    assert s1.is_compatible_with(EvaluationTrack.TRACK_C_ANSWER_ONLY)
    assert not s1.is_compatible_with(EvaluationTrack.TRACK_A_PREDICTED_CXR)


def test_slake_loader_inline_boxes_still_work(tmp_path: Path) -> None:
    """Loader still supports inline boxes/box_labels format (fixture compatibility)."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    test_entries = [
        {
            "id": 1,
            "img_id": 1,
            "img_name": "patient_001.png",
            "question": "Where is the consolidation?",
            "answer": "mid-upper zone",
            "q_lang": "en",
            "location": "chest",
            "modality": "CXR",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "boxes": [[10, 10, 40, 40]],
            "box_labels": ["consolidation"],
        },
        {
            "id": 2,
            "img_id": 2,
            "img_name": "patient_002.png",
            "question": "Any pneumothorax?",
            "answer": "no",
            "q_lang": "en",
            "location": "chest",
            "modality": "CXR",
            "q_type": "Closed",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "boxes": [],
            "box_labels": [],
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    # Create image files
    from PIL import Image
    for name in ["patient_001.png", "patient_002.png"]:
        Image.new("RGB", (100, 100), color=(128, 128, 128)).save(
            imgs_dir / name, format="PNG"
        )

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 2

    # First sample should have 1 GT region from inline boxes
    s1 = samples[0]
    assert s1.sample_id == "slake_test_1"
    assert len(s1.ground_truth_regions) == 1
    assert s1.ground_truth_regions[0].label == "consolidation"

    # Second sample should have 0 GT regions (empty boxes)
    s2 = samples[1]
    assert s2.sample_id == "slake_test_2"
    assert len(s2.ground_truth_regions) == 0


def test_slake_loader_out_of_bounds_inline_boxes_are_rejected(tmp_path: Path) -> None:
    """Genuinely out-of-bounds inline boxes are rejected, not silently loaded."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    test_entries = [
        {
            "id": 1,
            "img_id": 1,
            "img_name": "patient_001.png",
            "question": "Where is the consolidation?",
            "answer": "mid-upper zone",
            "q_lang": "en",
            "location": "chest",
            "modality": "CXR",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            # [x, y, w, h] = [50, 50, 100, 100] on a 100x100 image -> xmax=150, ymax=150 (out of bounds)
            "boxes": [[50, 50, 100, 100]],
            "box_labels": ["consolidation"],
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    from PIL import Image
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(
        imgs_dir / "patient_001.png", format="PNG"
    )

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 1
    # Out-of-bounds box must be rejected by production bounds validation
    assert len(samples[0].ground_truth_regions) == 0


def test_slake_loader_qid_field_used_for_sample_id(tmp_path: Path) -> None:
    """sample_id uses qid field when present, falling back to id then img_id."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    # Entry with qid but no id
    test_entries = [
        {
            "qid": 9999,
            "img_id": 42,
            "img_name": "xmlab42/source.jpg",
            "question": "Test question?",
            "answer": "Test answer",
            "q_lang": "en",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    from PIL import Image
    pdir = imgs_dir / "xmlab42"
    pdir.mkdir()
    Image.new("RGB", (100, 100)).save(pdir / "source.jpg", format="JPEG")
    (pdir / "detection.json").write_text("[]", encoding="utf-8")

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 1
    assert samples[0].sample_id == "slake_test_9999"


def test_slake_loader_missing_detection_json_returns_empty(tmp_path: Path) -> None:
    """Loader returns empty GT regions when detection.json is missing (no crash)."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    test_entries = [
        {
            "qid": 5000,
            "img_id": 99,
            "img_name": "xmlab99/source.jpg",
            "question": "Test?",
            "answer": "Answer",
            "q_lang": "en",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    from PIL import Image
    pdir = imgs_dir / "xmlab99"
    pdir.mkdir()
    Image.new("RGB", (100, 100)).save(pdir / "source.jpg", format="JPEG")
    # No detection.json file

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 1
    assert len(samples[0].ground_truth_regions) == 0


def test_slake_loader_malformed_detection_json_returns_empty(tmp_path: Path) -> None:
    """Loader handles malformed detection.json gracefully (logs warning, returns empty)."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    test_entries = [
        {
            "qid": 6000,
            "img_id": 88,
            "img_name": "xmlab88/source.jpg",
            "question": "Test?",
            "answer": "Answer",
            "q_lang": "en",
            "location": "Brain",
            "modality": "MRI",
            "q_type": "Open",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    from PIL import Image
    pdir = imgs_dir / "xmlab88"
    pdir.mkdir()
    Image.new("RGB", (100, 100)).save(pdir / "source.jpg", format="JPEG")
    (pdir / "detection.json").write_text("not valid json", encoding="utf-8")

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 1
    assert len(samples[0].ground_truth_regions) == 0


def test_slake_loader_actual_dataset_format(tmp_path: Path) -> None:
    """Loader works with actual SLAKE test.json format (qid, content_type, no boxes field)."""
    dataset_dir = tmp_path / "slake"
    imgs_dir = dataset_dir / "imgs"
    imgs_dir.mkdir(parents=True)

    # Mimic actual SLAKE test.json entry format (no boxes/box_labels, has qid/content_type)
    test_entries = [
        {
            "qid": 11934,
            "img_id": 102,
            "img_name": "xmlab102/source.jpg",
            "question": "What modality is used to take this image?",
            "answer": "CT",
            "q_lang": "en",
            "location": "Lung",
            "modality": "CT",
            "answer_type": "OPEN",
            "base_type": "vqa",
            "triple": ["vhead", "_", "_"],
            "content_type": "Modality",
        },
    ]
    (dataset_dir / "test.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    from PIL import Image
    pdir = imgs_dir / "xmlab102"
    pdir.mkdir()
    Image.new("RGB", (100, 100)).save(pdir / "source.jpg", format="JPEG")
    (pdir / "detection.json").write_text(
        json.dumps([{"Lung Cancer": [30, 40, 20, 30]}]), encoding="utf-8"
    )

    loader = SlakeDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=imgs_dir,
        language="en",
    )
    samples = loader.load_split("test")

    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "slake_test_11934"
    assert s.image_id == "102"
    assert s.modality == "CT"
    assert s.organ == "lung"
    assert len(s.ground_truth_regions) == 1
    assert s.ground_truth_regions[0].label == "Lung Cancer"
    # metadata should contain original_id=None (actual SLAKE uses qid, not id)
    assert s.metadata["original_id"] is None
    assert s.metadata["img_id"] == 102
