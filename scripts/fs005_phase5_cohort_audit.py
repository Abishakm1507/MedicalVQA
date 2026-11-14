import csv
import json
from pathlib import Path

# Load SLAKE test split
with open('D:\\Abi\\New folder\\MedicalVQA\\data\\SLAKE\\test.json', encoding='utf-8') as f:
    data = json.load(f)

# Filter to English only
en_test = [e for e in data if e.get('q_lang') == 'en']
print(f'English test samples: {len(en_test)}')

# FS-004 pilot QIDs
pilot_qids = {11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919}

# Get pilot image IDs from the baseline artifacts
pilot_image_ids = {130, 185, 224, 234, 312, 320, 344, 561, 605}
print(f'Pilot QIDs: {sorted(pilot_qids)}')
print(f'Pilot image IDs: {sorted(pilot_image_ids)}')

# Filter out pilot QIDs
remaining = [e for e in en_test if e['qid'] not in pilot_qids]
print(f'Remaining after QID exclusion: {len(remaining)}')

# Further exclude samples whose image_id is in the pilot image set
eligible = [e for e in remaining if e['img_id'] not in pilot_image_ids]
print(f'Remaining after image exclusion: {len(eligible)}')
print(f'Remaining unique images: {len({e["img_id"] for e in eligible})}')

# Answer type distribution
ats = {}
for e in eligible:
    at = e.get('answer_type', 'unknown')
    ats[at] = ats.get(at, 0) + 1
print(f'Answer type distribution: {ats}')

# Check detection.json existence for GT regions
slaake_dir = 'D:\\Abi\\New folder\\MedicalVQA\\data\\SLAKE'
samples_with_gt = 0
samples_without_gt = 0
for e in eligible:
    img_id = e['img_id']
    det_path = Path(slaake_dir) / f'detection_{img_id}.json'
    if det_path.exists():
        try:
            with open(det_path) as f:
                det = json.load(f)
                if det:  # has regions
                    samples_with_gt += 1
                else:
                    samples_without_gt += 1
        except Exception:
            samples_without_gt += 1
    else:
        samples_without_gt += 1
print(f'Samples with GT regions (detection.json non-empty): {samples_with_gt}')
print(f'Samples without GT regions: {samples_without_gt}')

# Save cohort artifacts
out_dir = Path('experiments/runs/fs005_phase5/cohort_audit')
out_dir.mkdir(parents=True, exist_ok=True)

# Save summary
summary = {
    "total_english_test_samples": len(en_test),
    "pilot_qids": sorted(pilot_qids),
    "pilot_image_ids": sorted(pilot_image_ids),
    "remaining_after_qid_exclusion": len(remaining),
    "remaining_after_image_exclusion": len(eligible),
    "remaining_unique_images": len({e["img_id"] for e in eligible}),
    "samples_with_gt_regions": samples_with_gt,
    "samples_without_gt_regions": samples_without_gt,
    "answer_type_distribution": ats,
    "included_qids": sorted(e["qid"] for e in eligible),
    "excluded_qids": sorted(pilot_qids),
    "excluded_image_ids": sorted(pilot_image_ids),
}
with open(out_dir / "cohort_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save included QIDs CSV
with open(out_dir / "included_qids.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["qid", "image_id", "modality", "question_type", "ground_truth_answer"])
    for e in sorted(eligible, key=lambda x: x["qid"]):
        writer.writerow([e["qid"], e["img_id"], e.get("modality", ""), e.get("answer_type", ""), e.get("answer", "")])

# Save excluded QIDs CSV
with open(out_dir / "excluded_qids.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["qid", "image_id", "modality", "question_type", "ground_truth_answer", "reason"])
    for e in sorted([e for e in en_test if e["qid"] in pilot_qids], key=lambda x: x["qid"]):
        writer.writerow([e["qid"], e["img_id"], e.get("modality", ""), e.get("answer_type", ""), e.get("answer", ""), "pilot_sample"])

# Save excluded image IDs
with open(out_dir / "excluded_image_ids.json", "w") as f:
    json.dump(sorted(pilot_image_ids), f, indent=2)

print(f'\nCohort audit artifacts saved to: {out_dir}')
