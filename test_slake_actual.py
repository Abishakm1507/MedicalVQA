"""Test SlakeDatasetLoader with actual SLAKE dataset at data/SLAKE."""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from datasets.slake_loader import SlakeDatasetLoader

slake_dir = Path("data/SLAKE")
images_dir = slake_dir / "imgs"

loader = SlakeDatasetLoader(
    dataset_dir=slake_dir,
    images_dir=images_dir,
    language="en",
)

# Load test split
samples = loader.load_split("test")
print(f"Total English test samples: {len(samples)}")

# Show first 5 samples
for s in samples[:5]:
    print(f"\nSample: {s.sample_id}")
    print(f"  image_id: {s.image_id}")
    print(f"  image_path: {s.image_path}")
    print(f"  question: {s.question[:80]}...")
    print(f"  answer: {s.answer}")
    print(f"  modality: {s.modality}")
    print(f"  organ: {s.organ}")
    print(f"  question_type: {s.question_type}")
    print(f"  patient_id: {s.patient_id}")
    print(f"  gt_regions: {len(s.ground_truth_regions)}")
    for r in s.ground_truth_regions:
        print(f"    - {r.label}: {r.box_2d}")
    print(f"  compatible_tracks: {s.compatible_tracks}")
    print(f"  metadata: {s.metadata}")

# Verify a sample with detection boxes
brain_samples = [s for s in samples if s.organ == "brain" and len(s.ground_truth_regions) > 0]
if brain_samples:
    s = brain_samples[0]
    print(f"\n=== Brain sample with GT regions: {s.sample_id} ===")
    print(f"  Question: {s.question}")
    print(f"  Answer: {s.answer}")
    for r in s.ground_truth_regions:
        print(f"  Region: {r.label} at {r.box_2d}")

# Count samples with GT regions
with_gt = [s for s in samples if len(s.ground_truth_regions) > 0]
print(f"\nSamples with GT regions: {len(with_gt)} / {len(samples)}")

# Count by modality
from collections import Counter
modalities = Counter(s.modality for s in samples)
print(f"Modalities: {dict(modalities)}")

# Count by organ
organs = Counter(s.organ for s in samples)
print(f"Organs: {dict(organs)}")
