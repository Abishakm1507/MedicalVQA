"""FS-005 Phase 5 paired cohort consistency validation.

Validates the audited SLAKE test cohort against the FS-005 Phase 5 protocol:

- Track B: 864 GT-annotated samples (eligible for pointing accuracy).
- Track C: 945 eligible samples (all samples, including those without GT regions).
- Paired Track B/C comparison: exactly the same 864 question IDs AND image IDs.
- Additional Track C-only samples: 81 (kept separate from paired grounding
  comparisons; may contribute to answer-accuracy reporting).
- Rejects mismatched paired cohorts (different question IDs or image IDs
  between Track B and Track C paired sets).

Reads cohort audit artifacts from
  experiments/runs/fs005_phase5/cohort_audit/
and the SLAKE detection.json files to determine GT-region presence.

No model inference is performed by this script.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure project root is on the path for package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SLAKE_DIR = Path("data/SLAKE")
IMG_DIR = SLAKE_DIR / "imgs"
COHORT_DIR = Path("experiments/runs/fs005_phase5/cohort_audit")

EXPECTED_TRACK_B = 864
EXPECTED_TRACK_C = 945
EXPECTED_PAIRED = 864
EXPECTED_TRACK_C_ONLY = 81

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


def _load_included_qids() -> list[int]:
    path = COHORT_DIR / "included_qids.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cohort audit artifact missing: {path}")
    qids: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qids.append(int(row["qid"]))
    return sorted(qids)


def _load_excluded_image_ids() -> set[int]:
    path = COHORT_DIR / "excluded_image_ids.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def _load_test_samples() -> list[dict]:
    with open(SLAKE_DIR / "test.json", encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data if e.get("q_lang") == "en"]


def _has_gt_regions(img_id: int) -> bool:
    det_path = IMG_DIR / f"xmlab{img_id}" / "detection.json"
    if not det_path.exists():
        return False
    try:
        data = json.loads(det_path.read_text(encoding="utf-8"))
        return isinstance(data, list) and len(data) > 0
    except (json.JSONDecodeError, OSError):
        return False


def validate_cohort() -> dict:
    included_qids = _load_included_qids()
    excluded_image_ids = _load_excluded_image_ids()
    test_samples = _load_test_samples()

    # Build the eligible cohort: English test samples minus pilot QIDs and
    # samples sharing pilot image IDs (image-disjoint from FS-004 pilot).
    eligible = [
        e
        for e in test_samples
        if e["qid"] in included_qids
        and e["img_id"] not in excluded_image_ids
        and e["qid"] not in PILOT_QIDS
    ]

    eligible_by_qid = {int(e["qid"]): e for e in eligible}
    eligible_qids = set(eligible_by_qid.keys())

    # Sanity: included_qids.csv must match the image-disjoint eligible set.
    included_set = set(included_qids)
    if included_set != eligible_qids:
        raise ValueError(
            "included_qids.csv does not match the image-disjoint eligible cohort. "
            f"Mismatch: only-in-csv={sorted(included_set - eligible_qids)}, "
            f"only-in-eligible={sorted(eligible_qids - included_set)}"
        )

    # Split by GT-region presence.
    track_b_samples = [e for e in eligible if _has_gt_regions(int(e["img_id"]))]
    track_c_only_samples = [e for e in eligible if not _has_gt_regions(int(e["img_id"]))]

    track_b_qids = {int(e["qid"]) for e in track_b_samples}
    track_c_qids = eligible_qids  # all 945 eligible samples are Track C eligible
    track_c_only_qids = {int(e["qid"]) for e in track_c_only_samples}

    # Paired Track B/C = exactly the 864 Track B qids (subset of Track C's 945).
    paired_qids = track_b_qids & track_c_qids

    errors: list[str] = []
    warnings: list[str] = []

    if len(eligible) != EXPECTED_TRACK_C:
        errors.append(
            f"Expected {EXPECTED_TRACK_C} eligible Track C samples, got {len(eligible)}"
        )
    if len(track_b_samples) != EXPECTED_TRACK_B:
        errors.append(
            f"Expected {EXPECTED_TRACK_B} Track B GT-annotated samples, got {len(track_b_samples)}"
        )
    if len(track_c_only_samples) != EXPECTED_TRACK_C_ONLY:
        errors.append(
            f"Expected {EXPECTED_TRACK_C_ONLY} Track C-only samples (no GT regions), got {len(track_c_only_samples)}"
        )
    if len(paired_qids) != EXPECTED_PAIRED:
        errors.append(
            f"Expected {EXPECTED_PAIRED} paired Track B/C question IDs, got {len(paired_qids)}"
        )
    if track_b_qids != paired_qids:
        errors.append(
            "Track B question IDs do not equal the paired Track B/C set "
            f"(only-in-B: {sorted(track_b_qids - paired_qids)[:10]}, "
            f"only-in-paired: {sorted(paired_qids - track_b_qids)[:10]})"
        )
    if not track_b_qids.issubset(track_c_qids):
        errors.append("Track B question IDs are not a subset of Track C question IDs")
    if track_c_only_qids & track_b_qids:
        errors.append(
            "Track C-only samples overlap with Track B paired samples; "
            "the 81 Track C-only samples must be kept separate"
        )
    if len(track_c_only_qids) + len(track_b_qids) != len(track_c_qids):
        errors.append(
            "Track C-only (81) + Track B paired (864) must equal Track C total (945)"
        )

    # Image-level disjointness between paired set and Track C-only set.
    paired_image_ids = {int(e["img_id"]) for e in track_b_samples}
    track_c_only_image_ids = {int(e["img_id"]) for e in track_c_only_samples}
    image_overlap = paired_image_ids & track_c_only_image_ids
    if image_overlap:
        warnings.append(
            f"{len(image_overlap)} image IDs appear in both paired Track B and "
            f"Track C-only sets (paired grounding comparisons use only the 864 "
            f"paired samples; the 81 Track C-only samples contribute to "
            f"answer-accuracy reporting only). Overlap: {sorted(image_overlap)[:10]}"
        )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "eligible_total": len(eligible),
            "track_b_gt_annotated": len(track_b_samples),
            "track_c_total": len(track_c_qids),
            "track_c_only_no_gt": len(track_c_only_samples),
            "paired_track_b_c": len(paired_qids),
        },
        "expected": {
            "eligible_total": EXPECTED_TRACK_C,
            "track_b_gt_annotated": EXPECTED_TRACK_B,
            "track_c_total": EXPECTED_TRACK_C,
            "track_c_only_no_gt": EXPECTED_TRACK_C_ONLY,
            "paired_track_b_c": EXPECTED_PAIRED,
        },
        "track_b_question_ids": sorted(track_b_qids),
        "track_c_only_question_ids": sorted(track_c_only_qids),
        "paired_question_ids": sorted(paired_qids),
        "paired_image_ids": sorted(paired_image_ids),
        "track_c_only_image_ids": sorted(track_c_only_image_ids),
        "image_overlap_with_track_c_only": sorted(image_overlap),
        "pilot_excluded_qids": sorted(PILOT_QIDS),
        "pilot_excluded_image_ids": sorted(excluded_image_ids),
    }
    return result


def main() -> int:
    result = validate_cohort()
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        print("\nCOHORT VALIDATION FAILED: mismatched paired cohort.", file=sys.stderr)
        return 1
    print("\nCOHORT VALIDATION PASSED: paired Track B/C cohort is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
