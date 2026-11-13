"""Integration test for FS-003 Stage 4: real MedSAM inference (gated).

This test ONLY runs when:
  1. torch and segment_anything are installed and importable, AND
  2. The MedSAM checkpoint (medsam_vit_b.pth) exists at data/models/medsam_vit_b.pth,
     AND
  3. The checkpoint passes size + MD5 checksum verification.

If any of these conditions fail, the test is SKIPPED — it is NOT reported as
proof that real MedSAM inference works.

Mocked unit tests (tests/unit/test_medsam_segmenter.py) are the authoritative
proof of adapter behavior. This integration test merely confirms the real
model end-to-end when a verified checkpoint is available.

Traceability:
  - Feature Specification: FS-003 (MedSAM Segmentation Integration)
"""

import hashlib
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.schemas import CandidateBox
from localization.rle import rle_decode
from localization.segmenter import (
    MEDSAM_CHECKPOINT_EXPECTED_MD5,
    MEDSAM_CHECKPOINT_EXPECTED_SIZE,
    MEDSAM_CHECKPOINT_FILENAME,
    MedSAMRegionSegmenter,
    SegmenterConfig,
)

CHECKPOINT_PATH = Path("data/models") / MEDSAM_CHECKPOINT_FILENAME


def _checkpoint_verified() -> bool:
    """Return True iff the checkpoint exists and passes size+MD5 verification."""
    if not CHECKPOINT_PATH.exists():
        return False
    actual_size = CHECKPOINT_PATH.stat().st_size
    if actual_size != MEDSAM_CHECKPOINT_EXPECTED_SIZE:
        return False
    md5 = hashlib.md5()
    with open(CHECKPOINT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == MEDSAM_CHECKPOINT_EXPECTED_MD5


def _deps_available() -> bool:
    """Return True iff torch and segment_anything are importable."""
    try:
        import segment_anything  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_RUN_INTEGRATION = _deps_available() and _checkpoint_verified()


@unittest.skipUnless(
    _RUN_INTEGRATION,
    "Real MedSAM integration test requires torch, segment_anything, and a "
    "size+MD5-verified medsam_vit_b.pth checkpoint at data/models/. "
    "Checkpoint is currently incomplete/corrupt — skipping.",
)
class TestMedSAMRealInference(unittest.TestCase):
    """End-to-end MedSAM inference against the real verified checkpoint."""

    def setUp(self) -> None:
        self.image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        self.cfg = SegmenterConfig(
            device="cpu",
            checkpoint_path=str(CHECKPOINT_PATH),
            image_size=1024,
        )
        self.seg = MedSAMRegionSegmenter(self.cfg)

    def test_real_single_region(self) -> None:
        """Segment a single region and verify mask shape/dtype/RLE."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(100, 100, 400, 400),
            normalized_box=(100 / 512, 100 / 512, 400 / 512, 400 / 512),
            score=0.9,
            label="lesion",
        )
        results = self.seg.segment_boxes(self.image, [box])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].region_id, "R1")
        self.assertEqual(results[0].mask.shape, (512, 512))
        self.assertEqual(results[0].mask.dtype, np.bool_)
        # RLE round-trip
        decoded = rle_decode(results[0].mask_rle)
        np.testing.assert_array_equal(decoded, results[0].mask)

    def test_real_multiple_regions_order(self) -> None:
        """Multiple regions preserve input order and region IDs."""
        boxes = [
            CandidateBox(
                region_id="R1",
                box_2d=(50, 50, 150, 150),
                normalized_box=(50 / 512, 50 / 512, 150 / 512, 150 / 512),
                score=0.9,
            ),
            CandidateBox(
                region_id="R2",
                box_2d=(300, 300, 450, 450),
                normalized_box=(300 / 512, 300 / 512, 450 / 512, 450 / 512),
                score=0.8,
            ),
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].region_id, "R1")
        self.assertEqual(results[1].region_id, "R2")
        for result in results:
            self.assertEqual(result.mask.shape, (512, 512))
            self.assertEqual(result.mask.dtype, np.bool_)
            decoded = rle_decode(result.mask_rle)
            np.testing.assert_array_equal(decoded, result.mask)


@unittest.skipIf(
    _RUN_INTEGRATION,
    "Checkpoint verification test only meaningful when checkpoint is NOT verified.",
)
class TestMedSAMCheckpointNotVerified(unittest.TestCase):
    """Documents that the checkpoint is currently unavailable for integration."""

    def test_checkpoint_missing_or_unverified(self) -> None:
        """The checkpoint is incomplete/corrupt; real inference is blocked."""
        self.assertFalse(_checkpoint_verified())
        self.assertFalse(_RUN_INTEGRATION)


if __name__ == "__main__":
    unittest.main()
