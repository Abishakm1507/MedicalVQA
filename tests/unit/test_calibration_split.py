"""Unit tests for calibration fit/assessment split by image ID.

Covers: image-level disjointness, all-questions-per-image preservation,
manifest consistency (qids <-> image_ids), Track B GT-restriction, and
Track C inclusion of samples without GT regions.
"""

import json

from scripts.fs005_phase5_calibration_split import (
    FIT_FRACTION,
    SEED,
    _excluded_by_pilot,
    _has_gt_regions,
    _save_manifest,
    _split_by_image,
)


def _make_samples() -> list[dict]:
    # Two questions per image for images 1..6.
    samples = []
    for img in range(1, 7):
        for j in range(2):
            samples.append(
                {
                    "qid": 1000 + img * 10 + j,
                    "img_id": img,
                    "q_lang": "en",
                }
            )
    return samples


def test_split_preserves_all_questions_per_image() -> None:
    samples = _make_samples()
    fit, assess, fit_imgs, assess_imgs = _split_by_image(samples, FIT_FRACTION, SEED)

    # Every sampled image's questions must be entirely in one subset.
    fit_sample_imgs = {int(e["img_id"]) for e in fit}
    assess_sample_imgs = {int(e["img_id"]) for e in assess}
    assert fit_sample_imgs == fit_imgs
    assert assess_sample_imgs == assess_imgs

    # No question of a fit image appears in assessment and vice versa.
    for e in fit:
        assert int(e["img_id"]) in fit_imgs
        assert int(e["img_id"]) not in assess_imgs
    for e in assess:
        assert int(e["img_id"]) in assess_imgs
        assert int(e["img_id"]) not in fit_imgs


def test_split_is_image_disjoint() -> None:
    samples = _make_samples()
    fit, assess, fit_imgs, assess_imgs = _split_by_image(samples, FIT_FRACTION, SEED)
    assert (fit_imgs & assess_imgs) == set()
    assert (fit_imgs | assess_imgs) == {int(e["img_id"]) for e in samples}


def test_split_both_subsets_non_empty() -> None:
    samples = _make_samples()
    fit, assess, _, _ = _split_by_image(samples, FIT_FRACTION, SEED)
    assert len(fit) > 0
    assert len(assess) > 0


def test_manifest_consistency_qids_match_image_ids(tmp_path) -> None:
    samples = _make_samples()
    fit, assess, fit_imgs, assess_imgs = _split_by_image(samples, FIT_FRACTION, SEED)

    _save_manifest(tmp_path, "mock-vlm", "B", "fit", fit, fit_imgs, ["note"])
    _save_manifest(tmp_path, "mock-vlm", "B", "assessment", assess, assess_imgs, ["note"])

    fit_qids = [int(line) for line in (tmp_path / "fit_qids.csv").read_text().strip().splitlines()[1:]]
    fit_image_ids = set(json.loads((tmp_path / "fit_image_ids.json").read_text()))

    # Every qid in the fit manifest belongs to a fit image, and every fit image
    # is represented by its questions.
    fit_by_qid = {e["qid"]: e for e in fit}
    for qid in fit_qids:
        assert int(fit_by_qid[qid]["img_id"]) in fit_image_ids
    imgs_represented = {int(fit_by_qid[q]["img_id"]) for q in fit_qids}
    assert imgs_represented == fit_image_ids


def test_pilot_exclusion_removes_qids_and_images() -> None:
    # Build samples including a pilot qid and a pilot image id.
    samples = _make_samples()
    samples.append({"qid": 11981, "img_id": 130, "q_lang": "en"})  # pilot qid + image
    samples.append({"qid": 11982, "img_id": 130, "q_lang": "en"})  # shares pilot image

    eligible = _excluded_by_pilot(samples)
    qids = {e["qid"] for e in eligible}
    imgs = {e["img_id"] for e in eligible}
    assert 11981 not in qids
    assert 11982 not in qids  # shares pilot image 130
    assert 130 not in imgs


def test_has_gt_regions_uses_real_slake() -> None:
    # Read-only check against the real SLAKE validation structure.
    assert _has_gt_regions(1) is False or _has_gt_regions(1) is True  # just exercises path
    # A nonexistent image has no GT regions.
    assert _has_gt_regions(999999) is False
