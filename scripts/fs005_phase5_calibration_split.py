"""FS-005 Phase 5 calibration fit/assessment split by image ID.

Splits the SLAKE English validation split into fitting and assessment subsets
**by image ID** (not per-question) so that multiple questions sharing the same
image cannot leak across the boundary. The split is kept model/track-specific:

- Track B: only samples with ground-truth regions (970 validation samples).
  Samples without GT regions are excluded from Track B calibration and the
  treatment is documented in the manifest.
- Track C: all 1,053 validation samples (ungrounded, no GT requirement).
  Samples without GT regions are included; their lack of regions is documented.

Fit thresholds/weights are selected on the fit subset only and assessed on the
held-out assessment subset. Test data is never used for calibration.

Outputs (per model/track) under
  experiments/runs/fs005_phase5/calibration/splits/<model>/<track>/:
  - fit_qids.csv          question IDs in the fitting subset
  - fit_image_ids.json    image IDs in the fitting subset
  - assessment_qids.csv   question IDs in the assessment subset
  - assessment_image_ids.json  image IDs in the assessment subset
  - split_manifest.json   split parameters, counts, GT-region treatment notes

No model inference is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path for package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SLAKE_DIR = Path("data/SLAKE")
VALIDATION_PATH = SLAKE_DIR / "validate.json"
TEST_PATH = SLAKE_DIR / "test.json"

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

IMG_DIR = SLAKE_DIR / "imgs"

# Deterministic fit fraction (image level). Calibration parameters must be
# empirically fitted; this fraction is a protocol choice, not a validated value.
FIT_FRACTION = 0.8
SEED = 42


def _has_gt_regions(img_id: int) -> bool:
    det_path = IMG_DIR / f"xmlab{img_id}" / "detection.json"
    if not det_path.exists():
        return False
    try:
        data = json.loads(det_path.read_text(encoding="utf-8"))
        return isinstance(data, list) and len(data) > 0
    except (json.JSONDecodeError, OSError):
        return False


def _english_samples(split: str) -> list[dict]:
    path = VALIDATION_PATH if split == "validate" else TEST_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data if e.get("q_lang") == "en"]


def _excluded_by_pilot(samples: list[dict]) -> list[dict]:
    return [
        e
        for e in samples
        if e["qid"] not in PILOT_QIDS and e["img_id"] not in PILOT_IMAGE_IDS
    ]


def _split_by_image(samples: list[dict], fit_fraction: float, seed: int):
    """Split samples into fit/assessment by unique image ID.

    Returns (fit_samples, assessment_samples, fit_image_ids, assessment_image_ids).
    Image IDs are sorted for determinism before the split.
    """
    import random

    image_ids = sorted({e["img_id"] for e in samples})
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)

    n_fit = max(1, int(round(len(shuffled) * fit_fraction)))
    # Guarantee both subsets are non-empty when possible.
    if len(shuffled) > 1:
        n_fit = min(n_fit, len(shuffled) - 1)
    fit_image_ids = set(shuffled[:n_fit])
    assessment_image_ids = set(shuffled[n_fit:])

    fit_samples = [e for e in samples if e["img_id"] in fit_image_ids]
    assessment_samples = [e for e in samples if e["img_id"] in assessment_image_ids]
    return fit_samples, assessment_samples, fit_image_ids, assessment_image_ids


def _save_manifest(out_dir: Path, model: str, track: str, split_name: str, samples: list[dict], image_ids: set[int], notes: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = sorted(e["qid"] for e in samples)
    (out_dir / f"{split_name}_qids.csv").write_text(
        "qid\n" + "\n".join(str(q) for q in qids) + "\n", encoding="utf-8"
    )
    (out_dir / f"{split_name}_image_ids.json").write_text(
        json.dumps(sorted(int(i) for i in image_ids), indent=2), encoding="utf-8"
    )
    manifest = {
        "model": model,
        "track": track,
        "split": split_name,
        "n_samples": len(samples),
        "n_images": len(image_ids),
        "image_ids": sorted(int(i) for i in image_ids),
        "question_ids": qids,
        "notes": notes,
    }
    (out_dir / f"{split_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_split_manifest(
    out_dir: Path,
    model: str,
    track: str,
    fit_samples: list[dict],
    assessment_samples: list[dict],
    fit_image_ids: set[int],
    assessment_image_ids: set[int],
    gt_treatment_note: str,
) -> None:
    overlap = fit_image_ids & assessment_image_ids
    manifest = {
        "model": model,
        "track": track,
        "fit_fraction": FIT_FRACTION,
        "seed": SEED,
        "n_fit_samples": len(fit_samples),
        "n_fit_images": len(fit_image_ids),
        "n_assessment_samples": len(assessment_samples),
        "n_assessment_images": len(assessment_image_ids),
        "fit_image_ids": sorted(int(i) for i in fit_image_ids),
        "assessment_image_ids": sorted(int(i) for i in assessment_image_ids),
        "image_overlap": sorted(int(i) for i in overlap),
        "image_disjoint": len(overlap) == 0,
        "gt_treatment": gt_treatment_note,
        "test_data_used": False,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main(
    exclude_image_ids: set[int] | None = None,
    output_dir: str | Path = "experiments/runs/fs005_phase5/calibration/splits",
) -> int:
    validation_samples = _english_samples("validate")
    print(f"English validation samples: {len(validation_samples)}")

    eligible = _excluded_by_pilot(validation_samples)
    print(f"Eligible after FS-004 pilot exclusion: {len(eligible)}")

    exclude_image_ids = exclude_image_ids or set()
    if exclude_image_ids:
        eligible = [e for e in eligible if int(e["img_id"]) not in exclude_image_ids]
        print(
            f"Eligible after excluding {len(exclude_image_ids)} pilot image IDs: {len(eligible)}"
        )

    # --- Track B: only samples with GT regions (970 validation samples) ---
    track_b = [e for e in eligible if _has_gt_regions(int(e["img_id"]))]
    track_b_without_gt = [
        e for e in eligible if not _has_gt_regions(int(e["img_id"]))
    ]
    print(f"Track B eligible (with GT regions): {len(track_b)}")
    print(f"Track B excluded (no GT regions): {len(track_b_without_gt)}")

    # --- Track C: all eligible validation samples (ungrounded) ---
    track_c = eligible
    print(f"Track C eligible (all, ungrounded): {len(track_c)}")

    output_dir = Path(output_dir)
    results: list[dict] = []

    for model in ("medgemma-4b", "qwen2.5-vl-3b"):
        # Track B split
        fit_b, assess_b, fit_img_b, assess_img_b = _split_by_image(
            track_b, FIT_FRACTION, SEED
        )
        out_b = output_dir / model / "B"
        _save_manifest(
            out_b,
            model,
            "B",
            "fit",
            fit_b,
            fit_img_b,
            ["Track B fitting subset: samples with GT regions only."],
        )
        _save_manifest(
            out_b,
            model,
            "B",
            "assessment",
            assess_b,
            assess_img_b,
            ["Track B assessment subset: samples with GT regions only."],
        )
        _write_split_manifest(
            out_b,
            model,
            "B",
            fit_b,
            assess_b,
            fit_img_b,
            assess_img_b,
            "Only validation samples with non-empty detection.json (GT regions) are "
            "eligible for Track B calibration. "
            f"{len(track_b_without_gt)} eligible validation samples without GT regions "
            "are excluded from Track B and documented here.",
        )
        results.append(
            {
                "model": model,
                "track": "B",
                "n_fit": len(fit_b),
                "n_assessment": len(assess_b),
                "n_fit_images": len(fit_img_b),
                "n_assessment_images": len(assess_img_b),
                "image_disjoint": len(fit_img_b & assess_img_b) == 0,
            }
        )

        # Track C split
        fit_c, assess_c, fit_img_c, assess_img_c = _split_by_image(
            track_c, FIT_FRACTION, SEED
        )
        out_c = output_dir / model / "C"
        n_no_gt_c = sum(1 for e in track_c if not _has_gt_regions(int(e["img_id"])))
        _save_manifest(
            out_c,
            model,
            "C",
            "fit",
            fit_c,
            fit_img_c,
            [
                "Track C fitting subset: all eligible validation samples (ungrounded).",
                "Samples without GT regions are included (no GT requirement for Track C).",
            ],
        )
        _save_manifest(
            out_c,
            model,
            "C",
            "assessment",
            assess_c,
            assess_img_c,
            [
                "Track C assessment subset: all eligible validation samples (ungrounded).",
                "Samples without GT regions are included (no GT requirement for Track C).",
            ],
        )
        _write_split_manifest(
            out_c,
            model,
            "C",
            fit_c,
            assess_c,
            fit_img_c,
            assess_img_c,
            "Track C is ungrounded: all eligible validation samples are used regardless "
            f"of GT-region presence. {n_no_gt_c} samples without GT regions are included "
            "and documented; their lack of regions does not exclude them from Track C "
            "calibration.",
        )
        results.append(
            {
                "model": model,
                "track": "C",
                "n_fit": len(fit_c),
                "n_assessment": len(assess_c),
                "n_fit_images": len(fit_img_c),
                "n_assessment_images": len(assess_img_c),
                "image_disjoint": len(fit_img_c & assess_img_c) == 0,
            }
        )

    print("\nCalibration split summary:")
    for r in results:
        print(
            f"  {r['model']} Track {r['track']}: "
            f"fit={r['n_fit']} samples ({r['n_fit_images']} images), "
            f"assessment={r['n_assessment']} samples ({r['n_assessment_images']} images), "
            f"image_disjoint={r['image_disjoint']}"
        )
    print("\nSplit manifests written under:", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FS-005 Phase 5 calibration fit/assessment split by image ID."
    )
    parser.add_argument(
        "--exclude-image-ids",
        default="",
        help="Comma-separated image IDs to exclude from calibration (e.g. pilot images).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/fs005_phase5/calibration/splits",
        help="Output directory for split manifests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exclude: set[int] = set()
    if args.exclude_image_ids.strip():
        exclude = {int(x) for x in args.exclude_image_ids.split(",") if x.strip()}
    raise SystemExit(main(exclude_image_ids=exclude, output_dir=args.output_dir))
