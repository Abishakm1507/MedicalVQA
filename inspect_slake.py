import json
from pathlib import Path

slake_dir = Path("data/SLAKE")
test_path = slake_dir / "test.json"

with open(test_path, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
en = [e for e in data if e.get("q_lang") == "en"]
print(f"English entries: {len(en)}")
zh = [e for e in data if e.get("q_lang") == "zh"]
print(f"Chinese entries: {len(zh)}")

print(f"\nFirst English entry keys: {list(en[0].keys())}")
print(f"Has boxes field: {'boxes' in en[0]}")
print(f"Has box_labels field: {'box_labels' in en[0]}")
print(f"\nFirst English entry:")
print(json.dumps(en[0], indent=2, ensure_ascii=False))

# Count unique image IDs in English entries
en_img_ids = set(e.get("img_id") for e in en)
print(f"\nUnique image IDs in English test entries: {len(en_img_ids)}")

# Check a detection.json for one of these images
img_id = en[0].get("img_id")
det_path = slake_dir / "imgs" / f"xmlab{img_id}" / "detection.json"
if det_path.exists():
    with open(det_path, encoding="utf-8") as f:
        det_data = json.load(f)
    print(f"\nDetection.json for xmlab{img_id}:")
    print(json.dumps(det_data, indent=2, ensure_ascii=False))

# Count English entries per split file
for split_name in ["test.json", "train.json", "validate.json"]:
    p = slake_dir / split_name
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        e = [x for x in d if x.get("q_lang") == "en"]
        print(f"\n{split_name}: {len(d)} total, {len(e)} English")
