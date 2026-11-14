"""Focused tests for oracle-box pilot overlays and isolated progress."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from colab.fs005_phase5_colab_pilot import build_input_samples
from core.generation_artifact import (
    GenerationRecord,
    GenerationSample,
    GenerationStatus,
    compute_generation_id,
)
from scripts.fs005_phase5_generation import GenerationRunner
from scripts.fs005_phase5_ground_truth import build_ground_truth_overlay
from vlm.mock_client import MockVLMClient

THREE_REGION_ANNOTATION = [
    {"Liver": [42.0, 105.0, 21.0, 41.0]},
    {"Left Kidney": [164.0, 147.0, 20.0, 20.0]},
    {"Right Kidney": [59.0, 132.0, 34.0, 32.0]},
]


def _write_annotation(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _synthetic_pilot(tmp_path: Path) -> tuple[Path, dict]:
    repo_root = tmp_path / "repo"
    entries = []
    for image_id in range(15):
        image_dir = repo_root / "data" / "SLAKE" / "imgs" / f"xmlab{image_id}"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), (10, 20, 30)).save(image_dir / "source.jpg")
        annotations = [{"Liver": [5.0, 6.0, 20.0, 20.0]}] if image_id < 7 else []
        _write_annotation(image_dir / "detection.json", annotations)
        entries.append(
            {
                "qid": 1000 + image_id,
                "img_id": image_id,
                "question": f"Question {image_id}?",
            }
        )
    return repo_root, {
        "pilot_size": 15,
        "pilot_with_gt_regions": 7,
        "samples": entries,
    }


def test_annotation_xywh_conversion_and_stable_region_ids(tmp_path: Path) -> None:
    annotations = tmp_path / "detection.json"
    _write_annotation(annotations, THREE_REGION_ANNOTATION)
    original = Image.new("RGB", (256, 256), "white")
    original_bytes = original.tobytes()

    marked, region_ids, regions = build_ground_truth_overlay(original, annotations)
    marked_again, region_ids_again, regions_again = build_ground_truth_overlay(
        original, annotations
    )

    assert region_ids == ["R1", "R2", "R3"]
    assert region_ids_again == region_ids
    assert regions == regions_again
    assert [region["label"] for region in regions] == [
        "Liver",
        "Left Kidney",
        "Right Kidney",
    ]
    assert [region["box_xyxy"] for region in regions] == [
        (42.0, 105.0, 63.0, 146.0),
        (164.0, 147.0, 184.0, 167.0),
        (59.0, 132.0, 93.0, 164.0),
    ]
    assert marked is not original
    assert marked.tobytes() == marked_again.tobytes()
    assert marked.getpixel((42, 145)) == (255, 0, 0)
    assert marked.getpixel((42, 94)) == (255, 255, 0)
    assert original.tobytes() == original_bytes


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ([{"Liver": [1, 1, 0, 5]}], "positive width"),
        ([{"Liver": [1, 1, 5, 0]}], "positive width"),
        ([{"Liver": [1, 1, -2, 5]}], "positive width"),
        ([{"Liver": [-1, 1, 5, 5]}], "exceeds image bounds"),
        ([{"Liver": [60, 1, 5, 5]}], "exceeds image bounds"),
        ([{"Liver": [1, 1, 5]}], "must contain"),
        ([{"Liver": [1, 1, "5", 5]}], "non-numeric"),
        ([{"Liver": [1, 1, float("inf"), 5]}], "non-finite"),
    ],
)
def test_invalid_annotation_boxes_fail_safely(
    tmp_path: Path, annotation: object, message: str
) -> None:
    path = tmp_path / "detection.json"
    _write_annotation(path, annotation)

    with pytest.raises(ValueError, match=message):
        build_ground_truth_overlay(Image.new("RGB", (64, 64)), path)


def test_missing_or_malformed_annotation_fails_clearly(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64))
    with pytest.raises(FileNotFoundError, match="not found"):
        build_ground_truth_overlay(image, tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read"):
        build_ground_truth_overlay(image, malformed)


def test_track_b_filters_missing_regions_and_track_c_is_unmarked(
    tmp_path: Path,
) -> None:
    repo_root, manifest = _synthetic_pilot(tmp_path)
    track_b_samples, track_b_images = build_input_samples(manifest, repo_root, "B")
    track_c_samples, track_c_images = build_input_samples(manifest, repo_root, "C")

    assert len(track_b_samples) == 7
    assert len(track_b_images) == 7
    assert {sample["question_id"] for sample in track_b_samples} == set(range(1000, 1007))
    assert all(sample["region_ids"] == ["R1"] for sample in track_b_samples)
    assert all(sample["region_metadata"][0]["label"] == "Liver" for sample in track_b_samples)

    assert len(track_c_samples) == 15
    assert len(track_c_images) == 15
    assert all(sample["region_ids"] == [] for sample in track_c_samples)
    assert all("region_metadata" not in sample for sample in track_c_samples)
    assert track_b_images["0"].getpixel((5, 6)) == (255, 0, 0)
    assert track_c_images["0"].getpixel((5, 6)) != (255, 0, 0)

    for track, samples, images in (
        ("B", track_b_samples, track_b_images),
        ("C", track_c_samples, track_c_images),
    ):
        runner = GenerationRunner(
            vlm_client=MockVLMClient(),
            model_name="mock-vlm",
            model_revision=None,
            track=track,
            output_dir=tmp_path / f"progress_{track}",
            n_samples=1,
        )
        result = runner.run_batch(samples, images)
        progress = json.loads(runner._progress_path().read_text(encoding="utf-8"))
        assert result["n_expected"] == len(samples)
        assert progress["n_expected"] == len(samples)
        assert progress["completed_question_ids"] == [
            sample["question_id"] for sample in samples
        ]


class RecordingMockClient(MockVLMClient):
    def __init__(self) -> None:
        super().__init__(
            findings=[
                {"claim": "visible region", "region_id": "R1"},
                {"claim": "unknown region", "region_id": "R99"},
            ]
        )
        self.images: list[Image.Image] = []

    def sample_n(
        self,
        marked_image: Image.Image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        self.images.append(marked_image)
        return super().sample_n(marked_image, question, n, temperature, max_tokens)


def test_runner_uses_marked_track_b_image_and_validates_region_ids(
    tmp_path: Path,
) -> None:
    repo_root, manifest = _synthetic_pilot(tmp_path)
    samples, images = build_input_samples(manifest, repo_root, "B")
    sample = samples[0]
    client = RecordingMockClient()
    runner = GenerationRunner(
        vlm_client=client,
        model_name="mock-vlm",
        model_revision=None,
        track="B",
        output_dir=tmp_path / "track_b_generation",
        n_samples=1,
        random_seed=1,
    )

    result = runner.run_batch([sample], images)
    generation_id = compute_generation_id(
        "mock-vlm", None, "B", sample["question_id"], sample["image_id"], 1, 0.7, 1
    )
    record = runner.store.load("mock-vlm", "B", generation_id)

    assert result["n_completed"] == 1
    assert client.images == [images[str(sample["image_id"])]]
    assert client.images[0].getpixel((5, 6)) == (255, 0, 0)
    assert len(runner.store.list_records("mock-vlm", "B")) == 1
    assert result["condition_results"][0]["generation_id"] == generation_id
    assert result["condition_results"][0]["source_artifact_identical"] is True
    assert record is not None
    assert record.track == "B"
    assert record.grounding_mode == "oracle_gt_xywh_overlay_v1"
    assert [claim["region_id"] for claim in record.samples[0].claims] == ["R1", "NONE"]


def test_legacy_unmarked_track_b_artifact_is_preserved_but_not_reused(
    tmp_path: Path,
) -> None:
    repo_root, manifest = _synthetic_pilot(tmp_path)
    samples, images = build_input_samples(manifest, repo_root, "B")
    sample = samples[0]
    runner = GenerationRunner(
        vlm_client=RecordingMockClient(),
        model_name="mock-vlm",
        model_revision=None,
        track="B",
        output_dir=tmp_path / "legacy_track_b",
        n_samples=1,
        random_seed=1,
    )
    legacy_identity = {
        "model_name": "mock-vlm",
        "model_revision": None,
        "track": "B",
        "question_id": str(sample["question_id"]),
        "image_id": str(sample["image_id"]),
        "n_samples": 1,
        "temperature": 0.7,
        "random_seed": 1,
    }
    legacy_id = hashlib.sha256(
        json.dumps(legacy_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    legacy_record = GenerationRecord(
        generation_id=legacy_id,
        model_name="mock-vlm",
        model_revision=None,
        track="B",
        question_id=sample["question_id"],
        image_id=sample["image_id"],
        n_samples=1,
        temperature=0.7,
        random_seed=1,
        samples=[
            GenerationSample(
                sample_index=0,
                raw_response='{"answer":"old","findings":[]}',
                parsed_answer="old",
            )
        ],
        status=GenerationStatus.COMPLETED,
    )
    legacy_path = runner.store.save(legacy_record)

    result = runner.run_batch([sample], images)
    current_id = compute_generation_id(
        "mock-vlm", None, "B", sample["question_id"], sample["image_id"], 1, 0.7, 1
    )

    assert current_id != legacy_id
    assert result["n_completed"] == 1
    assert result["n_skipped"] == 0
    assert legacy_path.exists()
    assert runner.store.load("mock-vlm", "B", legacy_id) is not None
    assert len(runner.store.list_records("mock-vlm", "B")) == 2


def test_runner_track_c_receives_original_image_and_no_valid_region_ids(
    tmp_path: Path,
) -> None:
    repo_root, manifest = _synthetic_pilot(tmp_path)
    samples, images = build_input_samples(manifest, repo_root, "C")
    sample = samples[0]
    client = RecordingMockClient()
    runner = GenerationRunner(
        vlm_client=client,
        model_name="mock-vlm",
        model_revision=None,
        track="C",
        output_dir=tmp_path / "track_c_generation",
        n_samples=1,
        random_seed=1,
    )

    runner.run_batch([sample], images)
    generation_id = compute_generation_id(
        "mock-vlm", None, "C", sample["question_id"], sample["image_id"], 1, 0.7, 1
    )
    record = runner.store.load("mock-vlm", "C", generation_id)

    assert client.images == [images[str(sample["image_id"])]]
    assert client.images[0].getpixel((5, 6)) != (255, 0, 0)
    assert sample["region_ids"] == []
    assert "region_metadata" not in sample
    assert record is not None
    assert record.track == "C"
    assert record.grounding_mode == "original_unmarked_image"
    assert [claim["region_id"] for claim in record.samples[0].claims] == ["NONE", "NONE"]


def test_progress_fingerprint_isolates_cohorts_and_resume_counts(
    tmp_path: Path,
) -> None:
    client = MockVLMClient()
    runner = GenerationRunner(
        vlm_client=client,
        model_name="mock-vlm",
        model_revision=None,
        track="B",
        output_dir=tmp_path,
        n_samples=1,
        random_seed=2,
    )
    full_samples = [
        {
            "question_id": f"q{index}",
            "image_id": f"i{index}",
            "question": f"Q{index}?",
            "region_ids": ["R1"],
        }
        for index in range(15)
    ]
    images = {sample["image_id"]: None for sample in full_samples}

    assert runner._cohort_fingerprint(full_samples) == runner._cohort_fingerprint(
        list(reversed(full_samples))
    )
    first = runner.run_batch(full_samples, images)
    first_progress = json.loads(runner._progress_path().read_text(encoding="utf-8"))
    assert first["n_expected"] == 15
    assert first_progress["completed_question_ids"] == [
        sample["question_id"] for sample in full_samples
    ]

    track_b_samples = full_samples[:7]
    with pytest.warns(RuntimeWarning, match="cohort fingerprint"):
        second = runner.run_batch(track_b_samples, images)
    second_progress = json.loads(runner._progress_path().read_text(encoding="utf-8"))

    assert second["n_expected"] == 7
    assert second["n_completed"] == 0
    assert second["n_skipped"] == 7
    assert second["n_failed"] == second["n_retried"] == second["n_corrupt"] == 0
    assert second_progress["n_expected"] == 7
    assert second_progress["cohort_fingerprint"] != first_progress["cohort_fingerprint"]
    assert second_progress["completed_question_ids"] == [
        sample["question_id"] for sample in track_b_samples
    ]
    assert second_progress["failed_question_ids"] == []
    assert second_progress["corrupt_question_ids"] == []
    assert len(runner.store.list_records("mock-vlm", "B")) == 15
