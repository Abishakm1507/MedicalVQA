import json
from pathlib import Path

slaake_dir = 'D:\\Abi\\New folder\\MedicalVQA\\data\\SLAKE'
imgs_dir = Path(slaake_dir) / 'imgs'

empty_det = 0
non_empty_det = 0
empty_examples = []

for d in sorted(imgs_dir.iterdir()):
    if d.is_dir():
        det_path = d / 'detection.json'
        if det_path.exists():
            try:
                with open(det_path) as f:
                    det = json.load(f)
                if det:  # non-empty list
                    non_empty_det += 1
                else:  # empty list []
                    empty_det += 1
                    if len(empty_det) <= 10:
                        empty_examples.append(d.name)
            except Exception as e:
                print(f"Error reading {det_path}: {e}")
        else:
            print(f"Empty file: {det_path}")
        # Shouldn't happen but just in case
    else:
        print(f"No detection.json: {d}")

print(f"\nImage dirs with non-empty detection.json: {non_empty_det}")
print(f"Image dirs with empty detection.json []: {empty_det}")
if empty_examples:
    print(f"Examples of empty detection.json: {empty_examples[:10]}")

# Now check eligible samples specifically
with open('D:\\Abi\\New folder\\MedicalVQA\\data\\SLAKE\\test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

en_test = [e for e in data if e.get('q_lang') == 'en']
pilot_qids = {11981, 12073, 12111, 12129, 12312, 12344, 12384, 12845, 12914, 12919}
pilot_image_ids = {130, 185, 224, 234, 312, 320, 344, 561, 605}

remaining = [e for e in en_test if e['qid'] not in pilot_qids]
eligible = [e for e in remaining if e['img_id'] not in pilot_image_ids]

# Check each eligible sample's detection.json
eligible_with_gt = 0
eligible_without_gt = 0
eligible_empty_det = 0

for e in eligible:
    img_id = e['img_id']
    img_folder = f'xmlab{img_id}'
    det_path = imgs_dir / img_folder / 'detection.json'
    if det_path.exists():
        try:
            with open(det_path) as f:
                det = json.load(f)
                if det:
                    eligible_with_gt += 1
                else:
                    eligible_without_gt += 1
                    eligible_empty_det += 1
                    if eligible_empty_det <= 10:
                        print(f"Eligible sample with empty detection.json: qid={e['qid']}, img_id={img_id}, dir={img_folder}")
        except Exception as ex:
            eligible_without_gt += 1
            print(f"Error reading detection.json for qid={e['qid']}, img_id={img_id}: {ex}")
    else:
        eligible_without_gt += 1
        print(f"Missing detection.json for qid={e['qid']}, img_id={img_id}, expected dir={img_folder}")

print(f"\nEligible samples with GT regions (non-empty detection.json): {eligible_with_gt}")
print(f"Eligible samples without GT regions (empty or missing detection.json): {eligible_without_gt}")
print(f"  - Empty detection.json []: {eligible_empty_det}")
print(f"  - Missing detection.json file: {eligible_without_gt - eligible_empty_det}")
print(f"Total eligible: {eligible_with_gt + eligible_without_gt}")