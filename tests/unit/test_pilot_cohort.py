"""Unit tests for FS-005 Phase 5 pilot safety: validation-only cohort,
calibration/pilot image-ID disjointness, pilot/test separation, persistent
output path configuration, and resume from persistent artifacts."""

import json
from pathlib import Path

from core.generation_artifact import (
    GenerationArtifactStore,
    GenerationRecord,
    GenerationSample,
    GenerationStatus,
    ValidationStatus,
    compute_generation_id,
)
from scripts.fs005_phase5_calibration_split import (
    FIT_FRACTION,
    SEED,
    _has_gt_regions,
    _split_by_image,
)
from scripts.fs005_phase5_calibration_split import (
    main as calibration_main,
)
from scripts.fs005_phase5_generation import GenerationRunner
from scripts.fs005_phase5_pilot_cohort import (
    PILOT_IMAGE_IDS,
    PILOT_QIDS,
    _select_pilot_samples,
)
from vlm.mock_client import MockVLMClient

COLAB_ENTRY = Path("colab/fs005_phase5_colab_pilot.py")
VALIDATION_PATH = Path("data/SLAKE/validate.json")
TEST_PATH = Path("data/SLAKE/test.json")
CALIB_ROOT = Path("experiments/runs/fs005_phase5/calibration/splits")
PILOT_MANIFEST = Path("experiments/runs/fs005_phase5/pilot_cohort/pilot_manifest.json")


def _pilot_image_ids() -> set[int]:
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    return {int(i) for i in manifest["pilot_image_ids"]}


def _load_validation() -> list[dict]:
    with open("data/SLAKE/validate.json", encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data if e.get("q_lang") == "en"]


def _load_test_qids() -> set[int]:
    with open("data/SLAKE/test.json", encoding="utf-8") as f:
        data = json.load(f)
    return {int(e["qid"]) for e in data if e.get("q_lang") == "en"}


def test_pilot_selection_is_image_disjoint_and_in_range() -> None:
    validation = _load_validation()
    pilot = _select_pilot_samples(validation, 15, 20261114)
    image_ids = {int(e["img_id"]) for e in pilot}
    assert 10 <= len(pilot) <= 20
    assert len(image_ids) == len(pilot)  # every sample has a distinct image
    # FS-004 pilot material never reused.
    assert not ({int(e["qid"]) for e in pilot} & PILOT_QIDS)
    assert not (image_ids & PILOT_IMAGE_IDS)


def test_pilot_excludes_fs004_pilot_images() -> None:
    validation = _load_validation()
    pilot = _select_pilot_samples(validation, 15, 20261114)
    pilot_images = {int(e["img_id"]) for e in pilot}
    assert not (pilot_images & PILOT_IMAGE_IDS)


def test_pilot_test_cohort_separation() -> None:
    validation = _load_validation()
    pilot = _select_pilot_samples(validation, 15, 20261114)
    pilot_qids = {int(e["qid"]) for e in pilot}
    test_qids = _load_test_qids()
    assert not (pilot_qids & test_qids)


def test_calibration_excludes_pilot_image_ids(tmp_path) -> None:
    validation = _load_validation()
    pilot = _select_pilot_samples(validation, 15, 20261114)
    pilot_image_ids = {int(e["img_id"]) for e in pilot}

    out_dir = tmp_path / "calibration_splits"
    calibration_main(
        exclude_image_ids=pilot_image_ids, output_dir=str(out_dir)
    )

    # No pilot image ID may appear in any fit or assessment manifest.
    for model in ("medgemma-4b", "qwen2.5-vl-3b"):
        for track in ("B", "C"):
            base = out_dir / model / track
            for subset in ("fit", "assessment"):
                image_ids = json.loads(
                    (base / f"{subset}_image_ids.json").read_text(encoding="utf-8")
                )
                assert not (set(image_ids) & pilot_image_ids), (
                    f"pilot image leaked into {model} Track {track} {subset}"
                )


def test_persistent_output_path_configuration(tmp_path) -> None:
    """GenerationArtifactStore must work on an arbitrary persistent root
    (e.g. a Google Drive path), not only the default repo-relative one."""
    drive_root = tmp_path / "MyDrive" / "MedicalVQA" / "outputs" / "fs005_phase5"
    store = GenerationArtifactStore(root=drive_root / "generations")
    record = GenerationRecord(
        generation_id="gen_drive_1",
        model_name="mock-vlm",
        model_revision=None,
        track="C",
        question_id="q1",
        image_id="img1",
        n_samples=2,
        temperature=0.7,
        random_seed=42,
        samples=[
            GenerationSample(sample_index=0, raw_response='{"answer":"Yes"}', parsed_answer="Yes", status="ok"),
            GenerationSample(sample_index=1, raw_response='{"answer":"Yes"}', parsed_answer="Yes", status="ok"),
        ],
        status=GenerationStatus.COMPLETED,
        validation_status=ValidationStatus.NOT_VALIDATED,
    )
    path = store.save(record)
    assert path.exists()
    assert str(path).startswith(str(drive_root / "generations"))
    loaded = store.load("mock-vlm", "C", "gen_drive_1")
    assert loaded is not None
    assert loaded.validation_status == ValidationStatus.VALID


def test_resume_from_persistent_artifact_directory(tmp_path) -> None:
    """Re-running GenerationRunner against an existing persistent artifact
    directory skips completed samples (no duplicate generation)."""
    output_dir = tmp_path / "persistent_runner"
    runner = GenerationRunner(
        vlm_client=MockVLMClient(model_name="mock-vlm"),
        model_name="mock-vlm",
        model_revision=None,
        track="C",
        output_dir=output_dir,
        n_samples=5,
        temperature=0.7,
        random_seed=3,
    )
    samples = [
        {"question_id": "q_persist_1", "image_id": "img_persist_1", "question": "Q1?", "region_ids": []},
        {"question_id": "q_persist_2", "image_id": "img_persist_2", "question": "Q2?", "region_ids": []},
    ]
    images = {s["image_id"]: None for s in samples}

    first = runner.run_batch(samples, images)
    assert first["n_completed"] == 2
    assert first["n_skipped"] == 0

    # Second run against the same persistent directory: all skipped.
    second = runner.run_batch(samples, images)
    assert second["n_completed"] == 2
    assert second["n_skipped"] == 2
    assert second["n_failed"] == 0
    assert second["n_retried"] == 0

    # Artifacts still present and valid on the persistent root.
    for s in samples:
        gen_id = compute_generation_id(
            "mock-vlm", None, "C", s["question_id"], s["image_id"], 5, 0.7, 3
        )
        assert runner.store.is_completed_valid("mock-vlm", "C", gen_id)


def _validation_samples() -> list[dict]:
    data = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    return [e for e in data if e.get("q_lang") == "en"]


def _has_gt(img_id: int) -> bool:
    return _has_gt_regions(img_id)


def test_calibration_reconciliation_relationships() -> None:
    """Recompute the calibration cohort reconciliation from the data and the
    split logic, asserting the expected relationships (counts are data-driven,
    not forced)."""
    val = _validation_samples()
    pilot_image_ids = _pilot_image_ids()
    # FS-004 exclusion (read-only on validation): FS-004 pilot material lives on
    # the test split, so nothing is excluded from validation.
    fs004_excluded = [
        e for e in val
        if e["qid"] in PILOT_QIDS or e["img_id"] in PILOT_IMAGE_IDS
    ]
    after_fs004 = [e for e in val if e not in fs004_excluded]
    assert len(val) == 1053
    assert len(fs004_excluded) == 0
    assert len(after_fs004) == 1053

    # Exclude the 15 validation-only pilot image IDs from calibration.
    remaining = [e for e in after_fs004 if int(e["img_id"]) not in pilot_image_ids]
    excluded_by_pilot = len(after_fs004) - len(remaining)
    assert excluded_by_pilot == 162
    assert len(remaining) == 891

    track_b = [e for e in remaining if _has_gt(int(e["img_id"]))]
    track_b_without_gt = len(remaining) - len(track_b)
    track_c = remaining
    # Data property: all 891 remaining validation samples have GT regions, so
    # Track B has 0 excluded-no-GT. This is because all without-GT validation
    # samples happen to reside on the 15 pilot images excluded from calibration.
    assert len(track_b) == 891
    assert track_b_without_gt == 0
    assert len(track_c) == 891

    # Split: fit + assessment must reconstruct the remaining cohort, with
    # image-level disjointness and no pilot overlap.
    fit_b, assess_b, fit_img_b, assess_img_b = _split_by_image(track_b, FIT_FRACTION, SEED)
    fit_c, assess_c, fit_img_c, assess_img_c = _split_by_image(track_c, FIT_FRACTION, SEED)
    assert len(fit_b) + len(assess_b) == len(track_b)
    assert len(fit_c) + len(assess_c) == len(track_c)
    assert not (fit_img_b & assess_img_b)
    assert not (fit_img_c & assess_img_c)
    assert not (fit_img_b & pilot_image_ids)
    assert not (assess_img_b & pilot_image_ids)
    assert not (fit_img_c & pilot_image_ids)
    assert not (assess_img_c & pilot_image_ids)


def test_calibration_manifests_recorded_no_pilot_overlap() -> None:
    """Read the generated calibration split manifests and assert no pilot image
    ID appears in any fit or assessment subset for any model/track."""
    for model in ("medgemma-4b", "qwen2.5-vl-3b"):
        for track in ("B", "C"):
            for subset in ("fit", "assessment"):
                path = CALIB_ROOT / model / track / f"{subset}_image_ids.json"
                assert path.exists(), f"missing manifest {path}"
                image_ids = {
                    int(x) for x in json.loads(path.read_text(encoding="utf-8"))
                }
                assert not (image_ids & _pilot_image_ids()), (
                    f"pilot image leaked into {model} Track {track} {subset}"
                )
            split = json.loads(
                (CALIB_ROOT / model / track / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            assert split["image_disjoint"] is True
            assert split["test_data_used"] is False


def test_test_cohort_read_only_separation_assertion() -> None:
    """The official test cohort is read solely for a read-only separation
    assertion: no test samples are selected, used for inference, used for
    calibration, or modified during the pilot."""
    test_data = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    test_qids = {int(e["qid"]) for e in test_data if e.get("q_lang") == "en"}
    # Pilot cohort source is validation; assert no QID overlap with test.
    val = _validation_samples()
    pilot = _select_pilot_samples(val, 15, 20261114)
    pilot_qids = {int(e["qid"]) for e in pilot}
    assert not (pilot_qids & test_qids)
    assert len(test_qids) == 1061


def test_real_inference_flag_disabled() -> None:
    """Real inference must remain disabled by default (no accidental execution)."""
    src = COLAB_ENTRY.read_text(encoding="utf-8")
    assert "RUN_REAL_INFERENCE = False" in src
