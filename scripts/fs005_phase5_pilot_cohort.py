"""FS-005 Phase 5 validation-only pilot cohort selection.

Selects 10-20 image-disjoint samples from the SLAKE English **validation**
dataset for the Colab T4 feasibility pilot. The official test cohort (945
eligible samples) is preserved completely untouched.

The selected pilot image IDs are excluded from calibration fitting AND
assessment, and the calibration split manifests are regenerated so that no
pilot image ID appears in either calibration subset. Pilot samples are never
used to fit calibration parameters or assess model performance.

Outputs:
  experiments/runs/fs005_phase5/pilot_cohort/
    - pilot_manifest.json   selected QIDs, image IDs, GT-region flags
    - pilot_qids.csv        selected question IDs
    - pilot_image_ids.json  selected image IDs

No model inference is performed by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fs005_phase5_calibration_split import (
    FIT_FRACTION,
    _excluded_by_pilot,
    _has_gt_regions,
    _split_by_image,
)
from scripts.fs005_phase5_calibration_split import (
    main as calibration_main,
)

SLAKE_DIR = Path("data/SLAKE")
VALIDATION_PATH = SLAKE_DIR / "validate.json"

PILOT_QIDS = {
    11981,
    12073,
    12111,
    12129,
    12312,
    12344,
    12384,
    12845,
    12914,
    12919,
}
PILOT_IMAGE_IDS = {130, 185, 224, 234, 312, 320, 344, 561, 605}

# Deterministic pilot size (within the required 10-20 range).
PILOT_SIZE = 15
SEED = 20261114

OUTPUT_DIR = Path("experiments/runs/fs005_phase5/pilot_cohort")


def _select_pilot_samples(
    validation_samples: list[dict], size: int, seed: int
) -> list[dict]:
    """Select ``size`` image-disjoint validation samples.

    Prefers a mix of samples with and without GT regions so both Track B and
    Track C are exercised. Selection is deterministic given ``seed``.
    """
    import random

    rng = random.Random(seed)

    # Work on image-disjoint pools: one candidate sample per image.
    by_image: dict[int, list[dict]] = {}
    for e in validation_samples:
        if e["qid"] in PILOT_QIDS or e["img_id"] in PILOT_IMAGE_IDS:
            continue  # never reuse FS-004 pilot material
        by_image.setdefault(int(e["img_id"]), []).append(e)

    with_gt_imgs = sorted(
        img for img, samples in by_image.items() if _has_gt_regions(img)
    )
    without_gt_imgs = sorted(
        img for img, samples in by_image.items() if not _has_gt_regions(img)
    )

    rng.shuffle(with_gt_imgs)
    rng.shuffle(without_gt_imgs)

    n_with = min(max(size // 2, 1), len(with_gt_imgs))
    n_without = min(size - n_with, len(without_gt_imgs))

    chosen_images = with_gt_imgs[:n_with] + without_gt_imgs[:n_without]
    rng.shuffle(chosen_images)

    chosen: list[dict] = []
    for img in chosen_images:
        samples = by_image[img]
        rng.shuffle(samples)
        chosen.append(samples[0])
        if len(chosen) >= size:
            break

    chosen.sort(key=lambda e: e["qid"])
    return chosen


def main() -> int:
    with open(VALIDATION_PATH, encoding="utf-8") as f:
        data = json.load(f)
    validation_samples = [e for e in data if e.get("q_lang") == "en"]
    print(f"English validation samples: {len(validation_samples)}")

    pilot = _select_pilot_samples(validation_samples, PILOT_SIZE, SEED)
    pilot_qids = {int(e["qid"]) for e in pilot}
    pilot_image_ids = {int(e["img_id"]) for e in pilot}
    pilot_with_gt = sum(1 for e in pilot if _has_gt_regions(int(e["img_id"])))
    pilot_without_gt = len(pilot) - pilot_with_gt

    print(f"Pilot samples selected: {len(pilot)}")
    print(f"  distinct images: {len(pilot_image_ids)}")
    print(f"  with GT regions (Track B eligible): {pilot_with_gt}")
    print(f"  without GT regions (Track C only): {pilot_without_gt}")

    # --- Write pilot manifest ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "SLAKE English validation split",
        "pilot_size": len(pilot),
        "seed": SEED,
        "pilot_qids": sorted(pilot_qids),
        "pilot_image_ids": sorted(pilot_image_ids),
        "pilot_with_gt_regions": pilot_with_gt,
        "pilot_without_gt_regions": pilot_without_gt,
        "fs004_pilot_qids_excluded": sorted(PILOT_QIDS),
        "fs004_pilot_image_ids_excluded": sorted(PILOT_IMAGE_IDS),
        "samples": [
            {
                "qid": int(e["qid"]),
                "img_id": int(e["img_id"]),
                "question": e["question"],
                "answer_type": e.get("answer_type"),
                "has_gt_regions": _has_gt_regions(int(e["img_id"])),
            }
            for e in pilot
        ],
    }
    (OUTPUT_DIR / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "pilot_qids.csv").write_text(
        "qid\n" + "\n".join(str(q) for q in sorted(pilot_qids)) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "pilot_image_ids.json").write_text(
        json.dumps(sorted(pilot_image_ids), indent=2), encoding="utf-8"
    )
    print(f"Pilot manifest written to: {OUTPUT_DIR}")

    # --- Regenerate calibration splits excluding pilot image IDs ---
    print("\nRegenerating calibration split manifests excluding pilot image IDs...")
    calibration_main(
        exclude_image_ids=pilot_image_ids,
        output_dir="experiments/runs/fs005_phase5/calibration/splits",
    )

    # --- Disjointness assertions ---
    eligible = _excluded_by_pilot(validation_samples)
    eligible = [e for e in eligible if int(e["img_id"]) not in pilot_image_ids]
    track_b = [e for e in eligible if _has_gt_regions(int(e["img_id"]))]
    for _model in ("medgemma-4b", "qwen2.5-vl-3b"):
        fit_b, _, fit_img_b, _ = _split_by_image(track_b, FIT_FRACTION, SEED)
        fit_c, _, fit_img_c, _ = _split_by_image(eligible, FIT_FRACTION, SEED)
        assert not (fit_img_b & pilot_image_ids), "pilot image leaked into Track B fit"
        assert not (fit_img_c & pilot_image_ids), "pilot image leaked into Track C fit"
        assert not (pilot_qids & {int(e["qid"]) for e in fit_b})
        assert not (pilot_qids & {int(e["qid"]) for e in fit_c})
    print("Pilot/calibration image-ID disjointness verified.")

    # --- Test cohort separation ---
    with open(SLAKE_DIR / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)
    test_qids = {int(e["qid"]) for e in test_data if e.get("q_lang") == "en"}
    overlap = pilot_qids & test_qids
    assert not overlap, f"pilot QIDs overlap with test cohort: {sorted(overlap)}"
    print("Pilot/test cohort separation verified (no QID overlap).")

    print("\nPilot cohort selection complete. Test cohort untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
