import json
from pathlib import Path

slaake_dir = Path(__file__).resolve().parents[1] / "data" / "SLAKE"
imgs_dir = slaake_dir / "imgs"

# Load English test split
with open(slaake_dir / "test.json", encoding="utf-8") as f:
    test_data = json.load(f)

# Load English validation split
with open(slaake_dir / "validate.json", encoding="utf-8") as f:
    val_data = json.load(f)

en_test = [e for e in test_data if e.get("q_lang") == "en"]
en_val = [e for e in val_data if e.get("q_lang") == "en"]

print(f"English test samples: {len(en_test)}")
print(f"English validation samples: {len(en_val)}")

# FS-004 pilot QIDs
pilot_qids = {11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919}
pilot_image_ids = {130, 185, 224, 234, 312, 320, 344, 561, 605}

print(f"\nPilot QIDs: {sorted(pilot_qids)}")
print(f"Pilot image IDs: {sorted(pilot_image_ids)}")

def audit_samples(samples, name):
    """Audit a set of samples for annotation completeness."""
    excluded_qids = []
    eligible = []

    for e in samples:
        if e["qid"] in pilot_qids:
            excluded_qids.append(e["qid"])
        elif e["img_id"] in pilot_image_ids:
            excluded_qids.append(e["qid"])
        else:
            eligible.append(e)

    print(f"\n=== {name} ===")
    print(f"  Total samples: {len(samples)}")
    print(f"  Excluded (pilot QID or image ID): {len(excluded_qids)}")
    print(f"  Eligible samples: {len(eligible)}")

    # Check detection.json for each eligible sample
    with_gt = 0
    without_gt_empty = 0
    missing_det = 0
    malformed_det = 0
    total_boxes = 0

    for e in eligible:
        img_id = e["img_id"]
        img_folder = f"xmlab{img_id}"
        det_path = imgs_dir / img_folder / "detection.json"

        if not det_path.exists():
            missing_det += 1
            print(f"  MISSING detection.json: qid={e['qid']}, img_id={img_id}, dir={img_folder}")
        else:
            try:
                with open(det_path) as f:
                    det = json.load(f)
                if isinstance(det, list):
                    if len(det) == 0:
                        without_gt_empty += 1
                    else:
                        with_gt += 1
                        total_boxes += len(det)
                else:
                    malformed_det += 1
                    print(f"  MALFORMED (not list): qid={e['qid']}, img_id={img_id}, type={type(det)}")
            except json.JSONDecodeError as e:
                malformed_det += 1
                print(f"  MALFORMED (invalid JSON): qid={e['qid']}, img_id={img_id}, error={e}")

    print(f"  Samples with GT regions: {with_gt}")
    print(f"  Samples without GT regions (empty detection.json): {without_gt_empty}")
    print(f"  Samples missing detection.json: {missing_det}")
    print(f"  Samples with malformed detection.json: {malformed_det}")
    print(f"  Total bounding boxes: {total_boxes}")

    return {
        "total": len(samples),
        "eligible": len(eligible),
        "with_gt": with_gt,
        "without_gt_empty": without_gt_empty,
        "missing_det": missing_det,
        "malformed_det": malformed_det,
        "total_boxes": total_boxes,
        "eligible_qids": sorted([e["qid"] for e in eligible]),
        "excluded_qids": sorted(excluded_qids),
    }

# Run audit on test set
test_audit = audit_samples(en_test, "SLAKE English Test Split")

# Also check validation split
en_val = [e for e in val_data if e.get("q_lang") == "en"]
print(f"\nEnglish validation samples: {len(en_val)}")
val_audit = audit_samples(en_val, "SLAKE English Validation Split")

# Save audit report
out_dir = Path("experiments/runs/fs005_phase5/cohort_audit")
out_dir.mkdir(parents=True, exist_ok=True)

audit_report = {
    "test_split": test_audit,
    "validation_split": val_audit,
}
with open(out_dir / "cohort_audit_summary.json", "w") as f:
    json.dump(audit_report, f, indent=2)

print(f"\nCohort audit artifacts saved to: {out_dir}")
