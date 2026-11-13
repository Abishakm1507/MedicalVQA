# MedSAM Checkpoint Verification — Real Inference Test

**Feature Specification:** FS-003 (MedSAM Segmentation Integration)  
**Module:** `localization/segmenter.py` (`MedSAMRegionSegmenter`), `localization/rle.py`  
**Checkpoint:** `data/models/medsam_vit_b.pth`  
**Device:** CPU (per AGENTS.md hardware constraints; CUDA available on host but not used)  
**Result:** SUCCESS — checkpoint loads into the existing MedSAM ViT-B implementation and produces a valid segmentation mask from a genuine SLAKE image and valid bounding-box prompt.

---

## 1. Checkpoint Source and File Details

| Field | Value |
|-------|-------|
| Source | `https://huggingface.co/SansuiHan/medical_models` (user-provided) |
| Local path | `data/models/medsam_vit_b.pth` |
| Exists | Yes |
| Size (bytes) | 375,049,145 |
| Size (MB) | 357.6747 |
| Expected size (bytes) | 375,049,145 |
| Size match | Yes |
| SHA-256 | `34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74` |
| MD5 | `3bb6db55bd0c9ca30b61248bca72f8d6` |
| Expected MD5 | `3bb6db55bd0c9ca30b61248bca72f8d6` |
| MD5 match | Yes |
| State-dict type | `dict` (raw `state_dict`, not wrapped) |
| Number of state-dict keys | 314 |

The checkpoint's size and MD5 both match the official MedSAM ViT-B reference values hardcoded in `localization/segmenter.py` (`MEDSAM_CHECKPOINT_EXPECTED_SIZE`, `MEDSAM_CHECKPOINT_EXPECTED_MD5`), confirming this HuggingFace checkpoint is byte-identical to the official Zenodo release.

---

## 2. Architecture Compatibility

The checkpoint state dictionary was compared key-by-key and shape-by-shape against a freshly constructed MedSAM/SAM ViT-B model (built directly from `segment_anything.modeling` with the standard ViT-B configuration: depth=12, embed_dim=768, num_heads=12, img_size=1024, patch_size=16, global_attn_indexes=[2,5,8,11]), without loading any weights.

| Check | Result |
|-------|--------|
| Missing keys | 0 |
| Unexpected keys | 0 |
| Shape mismatches | 0 |
| Architecture compatible | Yes |

All 314 checkpoint keys (`image_encoder.*`, `prompt_encoder.*`, `mask_decoder.*`) match the ViT-B architecture exactly, with identical tensor shapes and `torch.float32` dtype.

---

## 3. Checkpoint Loading

Loaded via the existing `MedSAMRegionSegmenter` (no FS-003 implementation changes):

- Config: `SegmenterConfig(device="cpu", checkpoint_path="data/models/medsam_vit_b.pth", image_size=1024)`
- `weights_only=True` load succeeded (validated by `_validate_checkpoint`)
- `is_mock`: False (real MedSAM, not `MockRegionSegmenter`)
- Initialization time: 3.5475 s
- Device: cpu

No missing or unexpected state-dict keys were reported by the loader; `_validate_checkpoint` passed size, MD5, and `weights_only=True` loadability checks.

---

## 4. Genuine Segmentation Inference

**Input image (genuine SLAKE, not synthetic):**
- Path: `data/SLAKE/imgs/xmlab102/source.jpg`
- Modality: CT, Location: Lung
- Size: 512 × 512 (w × h), mode: RGB
- Source box (SLAKE `detection.json` `[x, y, w, h]` pixels): `[289.0, 301.0, 17.0, 22.0]` (label: "Lung Cancer")
- Project convention `box_2d` `(ymin, xmin, ymax, xmax)`: `(301, 289, 323, 306)` — validated within image bounds

**Inference:**
- Model initialization succeeds: Yes
- Inference completes without exceptions: Yes
- Inference time: 27.2642 s (CPU)
- Output mask shape: (512, 512) — matches image dimensions
- Mask dtype: `np.bool_`
- Mask foreground pixels: 139 / 262,144 (0.053%)
- Mask contains valid (non-empty) segmentation output: Yes
- Region ID preserved: `R1`
- Number of results: 1 (one per input box, in order)

---

## 5. RLE Round-Trip

- RLE string: `512,512:154916 2 508 6 505 7 504 8 504 10 501 12 500 11 7 1 493 10 7 3 492 8 9 3 493 5 1028 1 508 7 505 8 503 9 503 9 503 8 505 7 506 4 98005`
- `rle_decode(rle_encode(mask))` reproduces the original mask exactly: **Yes**

---

## 6. Saved Visual Verification Artifacts

All artifacts saved under `experiments/runs/medsam_verification/`:

| Artifact | Path |
|----------|------|
| 1. Original SLAKE image | `experiments/runs/medsam_verification/01_original_image.png` |
| 2. Image with input bounding box | `experiments/runs/medsam_verification/02_image_with_bbox.png` |
| 3. Segmentation mask (boolean, 0/255) | `experiments/runs/medsam_verification/03_segmentation_mask.png` |
| 4. Original image with segmentation overlay | `experiments/runs/medsam_verification/04_image_with_overlay.png` |
| Machine-readable report | `experiments/runs/medsam_verification/verification_report.json` |
| Verification script | `experiments/runs/medsam_verification/verify_medsam.py` |

---

## 7. Performance Measurements

| Metric | Value |
|--------|-------|
| Model initialization time | 3.5475 s |
| Inference time (single box) | 27.2642 s |
| Device | CPU |
| CUDA available on host | Yes (not used — CPU per hardware constraints) |
| Torch CPU threads | 8 |
| Process RAM (RSS) | 193.87 MB |

---

## 8. Limitations

- Inference executed on CPU; a CUDA-capable GPU is present on the host but was deliberately not used to honor the project's CPU hardware constraint. GPU execution would reduce inference time.
- A single bounding box / single region was verified; multi-box sequential inference follows the same code path (one `predict` call per box) and is covered by the existing unit tests.
- Mask foreground fraction is small (0.053%) because the SLAKE annotation box `[289, 301, 17, 22]` tightly bounds a small lung lesion; this is expected behavior for a small region prompt, not a defect.
- Checkpoint SHA-256 is recorded for traceability but is not part of the implementation's validation (which uses size + MD5).

---

## 9. Success Criterion

The checkpoint loads successfully into the existing MedSAM ViT-B implementation (`MedSAMRegionSegmenter`, unmodified) and produces a valid segmentation mask (shape 512×512, dtype `bool`, non-empty foreground, RLE round-trip exact, region ID `R1` preserved) from a genuine SLAKE image (`data/SLAKE/imgs/xmlab102/source.jpg`) and a valid bounding-box prompt. **Criterion met.**
