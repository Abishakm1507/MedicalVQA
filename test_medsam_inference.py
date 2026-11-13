"""Test MedSAM inference with real SLAKE image and bounding box."""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from PIL import Image
import numpy as np

from localization.segmenter import MedSAMRegionSegmenter, SegmenterConfig, SegmentationResult
from localization.rle import rle_encode, rle_decode
from localization.detector import CandidateBox

slake_dir = Path("data/SLAKE")
ckpt = Path("data/models/medsam_vit_b.pth")

# 1. Load MedSAM
seg = MedSAMRegionSegmenter(SegmenterConfig(checkpoint_path=str(ckpt), device="cpu"))
print("MedSAM loaded successfully on CPU")

# 2. Load a real SLAKE image with detection boxes
img_id = 57
img_path = slake_dir / "imgs" / f"xmlab{img_id}" / "source.jpg"
det_path = slake_dir / "imgs" / f"xmlab{img_id}" / "detection.json"

image = Image.open(img_path).convert("RGB")
print(f"Image: {img_path}, size: {image.size}, mode: {image.mode}")

with open(det_path, encoding="utf-8") as f:
    det_data = json.load(f)
print(f"Detection data: {json.dumps(det_data, indent=2)}")

# 3. Convert detection.json format to CandidateBox objects
# detection.json: [{"Disease Name": [x, y, w, h]}, ...]
# SLAKE format: [x, y, w, h] where (x,y) is top-left, w=width, h=height
# Project convention: (ymin, xmin, ymax, xmax)
boxes = []
for entry in det_data:
    for disease_name, bbox in entry.items():
        x, y, w, h = bbox
        ymin, xmin, ymax, xmax = y, x, y + h, x + w
        box = CandidateBox(
            region_id=f"R{len(boxes)+1}",
            box_2d=(ymin, xmin, ymax, xmax),
            normalized_box=(ymin/image.size[1], xmin/image.size[0], ymax/image.size[1], xmax/image.size[0]),
            score=1.0,
            label=disease_name,
        )
        boxes.append(box)
        print(f"  Box {box.region_id}: {disease_name} -> (ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax})")

# 4. Run MedSAM segmentation
results = seg.segment_boxes(image, boxes)
print(f"\nSegmentation results: {len(results)} masks")

# 5. Verify each mask
for result in results:
    mask = result.mask
    rle = result.mask_rle
    print(f"\nRegion {result.region_id}:")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Mask sum (pixels): {int(mask.sum())}")
    print(f"  RLE: {rle[:80]}..." if len(str(rle)) > 80 else f"  RLE: {rle}")
    
    # Verify RLE round-trip
    decoded = rle_decode(rle)
    match = np.array_equal(mask, decoded)
    print(f"  RLE round-trip match: {match}")
    
    # Verify mask is within image bounds
    assert mask.shape == image.size[::-1], f"Mask shape {mask.shape} != image size {image.size[::-1]}"
    print(f"  Mask matches image dimensions: True")

print("\nAll MedSAM inference tests PASSED")
