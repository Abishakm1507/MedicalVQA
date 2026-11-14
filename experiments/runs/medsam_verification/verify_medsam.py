"""MedSAM checkpoint verification: real inference on a genuine SLAKE image.

Uses the existing MedSAMRegionSegmenter (localization/segmenter.py) without
modifying FS-003 implementation. Runs on CPU per hardware constraints.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import CandidateBox  # noqa: E402
from localization.segmenter import (  # noqa: E402
    MedSAMRegionSegmenter,
    SegmenterConfig,
    MEDSAM_CHECKPOINT_EXPECTED_SIZE,
    MEDSAM_CHECKPOINT_EXPECTED_MD5,
)
from localization.rle import rle_encode, rle_decode  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "medsam_vit_b.pth"
IMAGE_PATH = PROJECT_ROOT / "data" / "SLAKE" / "imgs" / "xmlab102" / "source.jpg"
OUT_DIR = PROJECT_ROOT / "experiments" / "runs" / "medsam_verification"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    import hashlib as _hl

    m = _hl.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            m.update(chunk)
    return m.hexdigest()


def inspect_state_dict(path: Path) -> dict:
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd


def compare_with_registry(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Compare checkpoint keys/shapes against sam_model_registry vit_b.

    Builds the ViT-B model directly (no checkpoint loaded) so expected keys and
    shapes are obtained without allocating/loading any weights file.
    """
    import torch
    from functools import partial
    from segment_anything.modeling import (
        ImageEncoderViT,
        MaskDecoder,
        PromptEncoder,
        Sam,
        TwoWayTransformer,
    )

    image_size = 1024
    vit_patch_size = 16
    prompt_embed_dim = 256
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    expected = dict(sam.state_dict())

    ckpt_keys = set(state.keys())
    exp_keys = set(expected.keys())

    missing = sorted(exp_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - exp_keys)

    shape_mismatches = []
    for k in sorted(ckpt_keys & exp_keys):
        ckpt_shape = tuple(state[k].shape)
        exp_shape = tuple(expected[k].shape)
        if ckpt_shape != exp_shape:
            shape_mismatches.append(
                f"{k}: checkpoint={ckpt_shape}, expected={exp_shape}"
            )

    return missing, unexpected, shape_mismatches


def save_artifacts(
    image: Image.Image,
    box_2d: tuple[int, int, int, int],
    mask: np.ndarray,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ymin, xmin, ymax, xmax = box_2d

    # 1. Original image
    orig_path = out_dir / "01_original_image.png"
    image.save(orig_path)

    # 2. Image with input bounding box
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    # draw rectangle in (xmin, ymin, xmax, ymax)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin + 4, ymin + 4), "R1", fill="red")
    boxed_path = out_dir / "02_image_with_bbox.png"
    boxed.save(boxed_path)

    # 3. Segmentation mask (boolean -> uint8 0/255)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_path = out_dir / "03_segmentation_mask.png"
    mask_img.save(mask_path)

    # 4. Original image with segmentation overlay (red tint on mask pixels)
    overlay = image.convert("RGBA")
    overlay_np = np.array(overlay).copy()
    # red overlay where mask True
    overlay_np[mask, 0] = 255  # R
    overlay_np[mask, 1] = 0  # G
    overlay_np[mask, 2] = 0  # B
    overlay_np[mask, 3] = 120  # alpha
    overlay_img = Image.fromarray(overlay_np, mode="RGBA")
    # also draw the box on overlay
    odraw = ImageDraw.Draw(overlay_img)
    odraw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    overlay_path = out_dir / "04_image_with_overlay.png"
    overlay_img.convert("RGB").save(overlay_path)

    return {
        "original": orig_path,
        "with_bbox": boxed_path,
        "mask": mask_path,
        "overlay": overlay_path,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Check the checkpoint file
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Checkpoint file verification")
    print("=" * 70)

    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"
    size_bytes = CHECKPOINT_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    sha256 = sha256_file(CHECKPOINT_PATH)
    md5 = md5_file(CHECKPOINT_PATH)

    report["checkpoint"] = {
        "path": str(CHECKPOINT_PATH),
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 4),
        "expected_size_bytes": MEDSAM_CHECKPOINT_EXPECTED_SIZE,
        "size_matches": size_bytes == MEDSAM_CHECKPOINT_EXPECTED_SIZE,
        "sha256": sha256,
        "md5": md5,
        "expected_md5": MEDSAM_CHECKPOINT_EXPECTED_MD5,
        "md5_matches": md5 == MEDSAM_CHECKPOINT_EXPECTED_MD5,
    }

    print(f"Path: {CHECKPOINT_PATH}")
    print(f"Exists: True")
    print(f"Size: {size_bytes} bytes ({size_mb:.4f} MB)")
    print(f"Expected size: {MEDSAM_CHECKPOINT_EXPECTED_SIZE} bytes -> match: {size_bytes == MEDSAM_CHECKPOINT_EXPECTED_SIZE}")
    print(f"SHA-256: {sha256}")
    print(f"MD5: {md5}")
    print(f"Expected MD5: {MEDSAM_CHECKPOINT_EXPECTED_MD5} -> match: {md5 == MEDSAM_CHECKPOINT_EXPECTED_MD5}")

    # Inspect state dict
    state = inspect_state_dict(CHECKPOINT_PATH)
    report["checkpoint"]["state_dict_type"] = type(state).__name__
    report["checkpoint"]["num_state_keys"] = len(state)
    print(f"State dict type: {type(state).__name__}, num keys: {len(state)}")

    # ------------------------------------------------------------------
    # Step 2: Architecture compatibility
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 2: Architecture compatibility vs sam_model_registry vit_b")
    print("=" * 70)

    missing, unexpected, shape_mismatches = compare_with_registry(state)
    report["architecture"] = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "compatible": (len(missing) == 0 and len(unexpected) == 0 and len(shape_mismatches) == 0),
    }
    print(f"Missing keys: {len(missing)} -> {missing[:5]}{'...' if len(missing)>5 else ''}")
    print(f"Unexpected keys: {len(unexpected)} -> {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"Shape mismatches: {len(shape_mismatches)} -> {shape_mismatches[:5]}{'...' if len(shape_mismatches)>5 else ''}")
    print(f"Architecture compatible: {report['architecture']['compatible']}")

    # ------------------------------------------------------------------
    # Step 3: Load checkpoint via existing MedSAMRegionSegmenter (CPU)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 3: Load checkpoint via MedSAMRegionSegmenter (CPU)")
    print("=" * 70)

    config = SegmenterConfig(device="cpu", checkpoint_path=str(CHECKPOINT_PATH), image_size=1024)

    t_init0 = time.perf_counter()
    segmenter = MedSAMRegionSegmenter(config)
    t_init1 = time.perf_counter()
    init_time_s = t_init1 - t_init0

    report["loading"] = {
        "device": "cpu",
        "image_size": 1024,
        "init_time_s": round(init_time_s, 4),
        "is_mock": segmenter.IS_MOCK,
    }
    print(f"Device: cpu")
    print(f"Is mock: {segmenter.IS_MOCK}")
    print(f"Init time: {init_time_s:.4f} s")

    # ------------------------------------------------------------------
    # Step 4: Genuine segmentation inference on SLAKE image
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 4: Segmentation inference on genuine SLAKE image")
    print("=" * 70)

    image = Image.open(IMAGE_PATH).convert("RGB")
    img_w, img_h = image.size
    print(f"Image: {IMAGE_PATH.name}, size={image.size} (w,h), mode={image.mode}")

    # detection.json sidecar: {"Lung Cancer": [x, y, w, h]} in pixels
    det_path = IMAGE_PATH.parent / "detection.json"
    with open(det_path, encoding="utf-8") as f:
        det = json.load(f)
    # SLAKE format: [x, y, w, h]
    x, y, w, h = det[0]["Lung Cancer"]
    xmin, ymin = int(x), int(y)
    xmax, ymax = int(x + w), int(y + h)
    # Project convention: (ymin, xmin, ymax, xmax)
    box_2d = (ymin, xmin, ymax, xmax)
    print(f"SLAKE detection box [x,y,w,h] = [{x}, {y}, {w}, {h}]")
    print(f"Project convention box_2d (ymin,xmin,ymax,xmax) = {box_2d}")
    # Validate box within image
    assert 0 <= ymin < ymax <= img_h, f"ymin/ymax out of bounds: {box_2d}"
    assert 0 <= xmin < xmax <= img_w, f"xmin/xmax out of bounds: {box_2d}"

    box = CandidateBox(region_id="R1", box_2d=box_2d, normalized_box=(0.0, 0.0, 0.0, 0.0), score=1.0, label="Lung Cancer")

    t_inf0 = time.perf_counter()
    results = segmenter.segment_boxes(image, [box])
    t_inf1 = time.perf_counter()
    inf_time_s = t_inf1 - t_inf0

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    result = results[0]

    mask = result.mask
    report["inference"] = {
        "image_path": str(IMAGE_PATH),
        "image_size": [img_w, img_h],
        "input_box_2d": list(box_2d),
        "box_label": det[0].get("Lung Cancer"),
        "region_id": result.region_id,
        "num_results": len(results),
        "inference_time_s": round(inf_time_s, 4),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "mask_expected_shape": [img_h, img_w],
        "mask_shape_matches": mask.shape == (img_h, img_w),
        "mask_true_pixels": int(mask.sum()),
        "mask_total_pixels": int(mask.size),
        "mask_foreground_fraction": round(float(mask.mean()), 6),
    }

    print(f"Region ID: {result.region_id}")
    print(f"Num results: {len(results)}")
    print(f"Inference time: {inf_time_s:.4f} s")
    print(f"Mask shape: {mask.shape} (expected {(img_h, img_w)}) -> match: {mask.shape == (img_h, img_w)}")
    print(f"Mask dtype: {mask.dtype} -> is bool: {mask.dtype == np.bool_}")
    print(f"Mask true pixels: {int(mask.sum())} / {int(mask.size)} ({float(mask.mean()):.6f})")
    assert mask.dtype == np.bool_, f"Mask dtype is {mask.dtype}, expected bool"
    assert mask.shape == (img_h, img_w), f"Mask shape {mask.shape} != {(img_h, img_w)}"
    assert int(mask.sum()) > 0, "Mask contains no foreground pixels"
    assert result.region_id == "R1", f"Region ID not preserved: {result.region_id}"

    # ------------------------------------------------------------------
    # RLE round-trip
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 5: RLE round-trip verification")
    print("=" * 70)

    rle = rle_encode(mask)
    decoded = rle_decode(rle)
    round_trip_exact = np.array_equal(decoded, mask)
    report["rle"] = {
        "rle_string": rle,
        "round_trip_exact": bool(round_trip_exact),
    }
    print(f"RLE: {rle[:120]}{'...' if len(rle)>120 else ''}")
    print(f"RLE round-trip exact: {round_trip_exact}")
    assert round_trip_exact, "RLE round-trip did not reproduce the original mask"

    # ------------------------------------------------------------------
    # Step 6: Save visual artifacts
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 6: Save visual verification artifacts")
    print("=" * 70)

    artifact_paths = save_artifacts(image, box_2d, mask, OUT_DIR)
    report["artifacts"] = {k: str(v) for k, v in artifact_paths.items()}
    for k, v in artifact_paths.items():
        print(f"Saved {k}: {v}")

    # ------------------------------------------------------------------
    # Performance measurements (RAM via psutil if available)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 7: Performance measurements")
    print("=" * 70)

    ram_mb = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        psutil = None

    report["performance"] = {
        "init_time_s": round(init_time_s, 4),
        "inference_time_s": round(inf_time_s, 4),
        "device": "cpu",
        "cuda_available": False,
        "ram_rss_mb": round(ram_mb, 2) if ram_mb is not None else None,
        "torch_threads": None,
    }
    try:
        import torch

        report["performance"]["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    print(f"Init time: {init_time_s:.4f} s")
    print(f"Inference time: {inf_time_s:.4f} s")
    print(f"Device: cpu (CUDA not used per hardware constraints)")
    print(f"RAM RSS: {ram_mb:.2f} MB" if ram_mb is not None else "RAM RSS: n/a (psutil not installed)")

    # Write report JSON
    report_path = OUT_DIR / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote machine-readable report: {report_path}")

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE — SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
