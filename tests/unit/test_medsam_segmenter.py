"""Unit tests for FS-003 Stage 4: MedSAMRegionSegmenter (mocked).

These tests verify the MedSAMRegionSegmenter adapter behavior using mocks for
the heavy model/inference stack. No real checkpoint download or GPU is
required. Real MedSAM inference is covered by a separate integration smoke
test that only runs when the verified checkpoint and runtime dependencies
are available.

Traceability:
  - Feature Specification: FS-003 (MedSAM Segmentation Integration)
"""

import builtins
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from core.schemas import CandidateBox
from localization.rle import rle_decode
from localization.segmenter import (
    BaseRegionSegmenter,
    MedSAMRegionSegmenter,
    SegmenterConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 128, height: int = 96) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_box(
    region_id: str = "R1",
    ymin: int = 10,
    xmin: int = 20,
    ymax: int = 40,
    xmax: int = 60,
    score: float = 0.9,
) -> CandidateBox:
    return CandidateBox(
        region_id=region_id,
        box_2d=(ymin, xmin, ymax, xmax),
        normalized_box=(ymin / 96, xmin / 128, ymax / 96, xmax / 128),
        score=score,
        label="lesion",
    )


def _expected_checkpoint_bytes() -> bytes:
    """Return bytes of exactly the expected checkpoint size."""
    return b"\x00" * 375049145


class _SysModulesCleanupMixin:
    """Removes fake torch/segment_anything modules from sys.modules after each test.

    The _MedSAMHarness patches builtins.__import__ to inject MagicMock objects
    for torch and segment_anything. Python caches these in sys.modules, which
    would break later tests (including the integration test) that need the real
    torch module. This mixin cleans up after each test to guarantee order
    independence.
    """

    def tearDown(self) -> None:
        sys.modules.pop("torch", None)
        sys.modules.pop("segment_anything", None)
        super().tearDown()


class _MedSAMHarness:
    """Helper to construct MedSAMRegionSegmenter with mocked heavy deps."""

    def __init__(self) -> None:
        self._orig_import = builtins.__import__

    def make_mock_torch(self):
        mock_torch = mock.MagicMock()
        mock_torch.inference_mode.return_value.__enter__.return_value = None
        mock_torch.inference_mode.return_value.__exit__.return_value = None
        mock_torch.device = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        return mock_torch

    def make_mock_segment_anything(self, mock_torch):
        mock_sam = mock.MagicMock()
        mock_sam.sam_model_registry = {"vit_b": lambda checkpoint: mock.MagicMock()}
        mock_sam.SamPredictor = mock.MagicMock()
        return mock_sam

    def construct(
        self,
        device: str = "cpu",
        checkpoint_path: str | None = "/tmp/medsam_vit_b.pth",
        cuda_available: bool = False,
    ):
        """Construct MedSAMRegionSegmenter with all heavy deps mocked."""
        mock_torch = self.make_mock_torch()
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.device = lambda d: mock.MagicMock()

        fake_model = mock.MagicMock()
        fake_model.device = device

        mock_sam = self.make_mock_segment_anything(mock_torch)
        mock_sam.sam_model_registry = {"vit_b": lambda checkpoint: fake_model}
        fake_predictor = mock.MagicMock()
        mock_sam.SamPredictor.return_value = fake_predictor

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return mock_torch
            if name == "segment_anything":
                return mock_sam
            return self._orig_import(name, *args, **kwargs)

        cfg = SegmenterConfig(device=device, checkpoint_path=checkpoint_path)

        with mock.patch.object(MedSAMRegionSegmenter, "_validate_checkpoint"):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                seg = MedSAMRegionSegmenter(cfg)

        # Override internals with our mocks
        seg._torch = mock_torch
        seg._sam = fake_model
        seg._predictor = fake_predictor
        seg.device = mock_torch.device(device)

        return seg, mock_torch, fake_predictor


# ---------------------------------------------------------------------------
# A. Construction and configuration
# ---------------------------------------------------------------------------


class TestMedSAMConstruction(unittest.TestCase):
    """Tests for MedSAMRegionSegmenter construction and config validation."""

    def test_a01_is_subclass_of_base(self) -> None:
        self.assertTrue(issubclass(MedSAMRegionSegmenter, BaseRegionSegmenter))

    def test_a02_is_not_mock(self) -> None:
        self.assertFalse(MedSAMRegionSegmenter.IS_MOCK)

    def test_a03_requires_segmenter_config(self) -> None:
        with self.assertRaises(TypeError):
            MedSAMRegionSegmenter("not_a_config")  # type: ignore

    def test_a04_missing_checkpoint_raises(self) -> None:
        cfg = SegmenterConfig(device="cpu", checkpoint_path=None)
        with self.assertRaises(ValueError):
            MedSAMRegionSegmenter(cfg)

    def test_a05_nonexistent_checkpoint_raises(self) -> None:
        cfg = SegmenterConfig(
            device="cpu", checkpoint_path="/nonexistent/medsam_vit_b.pth"
        )
        with self.assertRaises(FileNotFoundError):
            MedSAMRegionSegmenter(cfg)

    def test_a06_wrong_size_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "medsam_vit_b.pth"
            path.write_bytes(b"\x00" * 100)  # wrong size
            cfg = SegmenterConfig(device="cpu", checkpoint_path=str(path))
            with self.assertRaises(ValueError):
                MedSAMRegionSegmenter(cfg)

    def test_a07_wrong_md5_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "medsam_vit_b.pth"
            # Write exactly expected size bytes but wrong content (wrong MD5)
            path.write_bytes(_expected_checkpoint_bytes())
            cfg = SegmenterConfig(device="cpu", checkpoint_path=str(path))
            with self.assertRaises(ValueError):
                MedSAMRegionSegmenter(cfg)


# ---------------------------------------------------------------------------
# B. Missing dependencies
# ---------------------------------------------------------------------------


class TestMedSAMMissingDependencies(unittest.TestCase):
    """Tests that missing torch/segment_anything produce informative ImportError."""

    def test_b01_import_error_when_torch_missing(self) -> None:
        cfg = SegmenterConfig(
            device="cpu", checkpoint_path="/tmp/medsam_vit_b.pth"
        )
        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return orig_import(name, *args, **kwargs)

        with mock.patch.object(MedSAMRegionSegmenter, "_validate_checkpoint"):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(ImportError) as ctx:
                    MedSAMRegionSegmenter(cfg)
            self.assertIn("torch", str(ctx.exception))

    def test_b02_import_error_when_segment_anything_missing(self) -> None:
        cfg = SegmenterConfig(
            device="cpu", checkpoint_path="/tmp/medsam_vit_b.pth"
        )
        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "segment_anything":
                raise ImportError("No module named 'segment_anything'")
            return orig_import(name, *args, **kwargs)

        with mock.patch.object(MedSAMRegionSegmenter, "_validate_checkpoint"):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(ImportError) as ctx:
                    MedSAMRegionSegmenter(cfg)
            self.assertIn("segment_anything", str(ctx.exception))

    def test_b03_segmenter_import_without_torch(self) -> None:
        """Importing localization.segmenter must not pull in torch."""
        # torch/segment_anything may already be in sys.modules from earlier
        # tests; pop them so we can verify segmenter itself does not import them.
        sys.modules.pop("torch", None)
        sys.modules.pop("segment_anything", None)
        # Re-import lightweight symbols (module already loaded, no heavy import)
        from localization.segmenter import (  # noqa: F401
            BaseRegionSegmenter,
            SegmenterConfig,
        )
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("segment_anything", sys.modules)


# ---------------------------------------------------------------------------
# C. CUDA-unavailable behavior
# ---------------------------------------------------------------------------


class TestMedSAMCudaBehavior(unittest.TestCase):
    """Tests explicit device handling when CUDA is requested but unavailable."""

    def test_c01_cuda_requested_unavailable_raises(self) -> None:
        cfg = SegmenterConfig(device="cuda", checkpoint_path="/tmp/medsam_vit_b.pth")

        harness = _MedSAMHarness()
        mock_torch = harness.make_mock_torch()
        mock_torch.cuda.is_available.return_value = False

        fake_model = mock.MagicMock()
        fake_model.device = "cuda"
        mock_sam = harness.make_mock_segment_anything(mock_torch)
        mock_sam.sam_model_registry = {"vit_b": lambda checkpoint: fake_model}
        fake_predictor = mock.MagicMock()
        mock_sam.SamPredictor.return_value = fake_predictor

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return mock_torch
            if name == "segment_anything":
                return mock_sam
            return harness._orig_import(name, *args, **kwargs)

        with mock.patch.object(MedSAMRegionSegmenter, "_validate_checkpoint"):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(RuntimeError) as ctx:
                    MedSAMRegionSegmenter(cfg)
            self.assertIn("CUDA is not available", str(ctx.exception))

    def test_c02_cpu_always_allowed(self) -> None:
        cfg = SegmenterConfig(device="cpu", checkpoint_path="/tmp/medsam_vit_b.pth")
        self.assertEqual(cfg.device, "cpu")


# ---------------------------------------------------------------------------
# D. Box coordinate conversion (project -> SAM)
# ---------------------------------------------------------------------------


class TestMedSAMBoxConversion(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests conversion from project (ymin,xmin,ymax,xmax) to SAM (xmin,ymin,xmax,ymax)."""

    def setUp(self) -> None:
        self.image = _make_image(width=128, height=96)
        self.boxes = [
            _make_box("R1", 10, 20, 40, 60),
            _make_box("R2", 30, 50, 70, 90),
        ]

    def test_d01_box_converted_to_sam_convention(self) -> None:
        """Verify (ymin,xmin,ymax,xmax) -> (xmin,ymin,xmax,ymax) conversion."""
        harness = _MedSAMHarness()
        seg, mock_torch, fake_predictor = harness.construct()

        fake_mask = np.zeros((96, 128), dtype=bool)
        fake_mask[10:40, 20:60] = True
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        seg.segment_boxes(self.image, self.boxes)

        call_args = fake_predictor.predict.call_args_list
        self.assertEqual(len(call_args), 2)
        # First box: (ymin=10, xmin=20, ymax=40, xmax=60) -> SAM (20, 10, 60, 40)
        np.testing.assert_array_equal(call_args[0].kwargs["box"], [20, 10, 60, 40])
        # Second box: (ymin=30, xmin=50, ymax=70, xmax=90) -> SAM (50, 30, 90, 70)
        np.testing.assert_array_equal(call_args[1].kwargs["box"], [50, 30, 90, 70])


# ---------------------------------------------------------------------------
# E. Mask dimensions and Boolean dtype
# ---------------------------------------------------------------------------


class TestMedSAMMaskProperties(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests that output masks have correct shape and dtype."""

    def test_e01_mask_shape_matches_image(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_mask = np.zeros((96, 128), dtype=bool)
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        results = seg.segment_boxes(_make_image(), [_make_box()])

        self.assertEqual(results[0].mask.shape, (96, 128))
        self.assertEqual(results[0].mask.dtype, np.bool_)

    def test_e02_non_square_image(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_mask = np.zeros((200, 50), dtype=bool)
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        tall = _make_image(width=50, height=200)
        results = seg.segment_boxes(tall, [_make_box()])

        self.assertEqual(results[0].mask.shape, (200, 50))
        self.assertEqual(results[0].mask.dtype, np.bool_)


# ---------------------------------------------------------------------------
# F. RLE encoding consistency
# ---------------------------------------------------------------------------


class TestMedSAMRLEConsistency(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests that mask_rle round-trips correctly via localization.rle."""

    def test_f01_rle_round_trip(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_mask = np.zeros((96, 128), dtype=bool)
        fake_mask[10:40, 20:60] = True
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        results = seg.segment_boxes(_make_image(), [_make_box()])

        decoded = rle_decode(results[0].mask_rle)
        np.testing.assert_array_equal(decoded, results[0].mask)

    def test_f02_rle_header_matches_image_dimensions(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_mask = np.zeros((96, 128), dtype=bool)
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        results = seg.segment_boxes(_make_image(), [_make_box()])

        header = results[0].mask_rle.split(":")[0]
        h_str, w_str = header.split(",")
        self.assertEqual(int(h_str), 96)
        self.assertEqual(int(w_str), 128)


# ---------------------------------------------------------------------------
# G. Multiple regions and input-order preservation
# ---------------------------------------------------------------------------


class TestMedSAMMultipleRegions(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests that multiple regions produce ordered results preserving region IDs."""

    def test_g01_order_preserved(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        call_order = []

        def fake_predict(*, box=None, multimask_output=False, return_logits=False):
            call_order.append(list(box))
            mask = np.zeros((96, 128), dtype=bool)
            return (mask[None, :, :], None, None)

        fake_predictor.predict.side_effect = fake_predict

        boxes = [
            _make_box("R7", 5, 5, 20, 20),
            _make_box("R1", 30, 30, 60, 90),
            _make_box("R3", 70, 10, 90, 50),
        ]
        results = seg.segment_boxes(_make_image(), boxes)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].region_id, "R7")
        self.assertEqual(results[1].region_id, "R1")
        self.assertEqual(results[2].region_id, "R3")
        self.assertEqual(call_order[0], [5, 5, 20, 20])
        np.testing.assert_array_equal(call_order[1], [30, 30, 90, 60])
        np.testing.assert_array_equal(call_order[2], [10, 70, 50, 90])


# ---------------------------------------------------------------------------
# H. Empty and boundary-touching boxes
# ---------------------------------------------------------------------------


class TestMedSAMEmptyBoundaryBoxes(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests empty input and boundary-touching boxes."""

    def test_h01_empty_boxes_returns_empty_list(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        results = seg.segment_boxes(_make_image(), [])

        self.assertEqual(results, [])
        fake_predictor.set_image.assert_not_called()

    def test_h02_boundary_touching_box(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_mask = np.zeros((96, 128), dtype=bool)
        fake_predictor.predict.return_value = (fake_mask[None, :, :], None, None)

        box = CandidateBox(
            region_id="R1",
            box_2d=(-30, 20, 0, 60),
            normalized_box=(0.0, 0.0, 0.0, 0.0),
            score=0.9,
        )
        results = seg.segment_boxes(_make_image(), [box])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].region_id, "R1")


# ---------------------------------------------------------------------------
# I. Inference errors and invalid inputs
# ---------------------------------------------------------------------------


class TestMedSAMErrors(_SysModulesCleanupMixin, unittest.TestCase):
    """Tests inference error handling and invalid inputs."""

    def test_i01_inference_failure_raises_runtime_error(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_predictor.predict.side_effect = RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError) as ctx:
            seg.segment_boxes(_make_image(), [_make_box()])
        self.assertIn("MedSAM inference failed", str(ctx.exception))

    def test_i02_wrong_mask_shape_raises(self) -> None:
        harness = _MedSAMHarness()
        seg, _, fake_predictor = harness.construct()

        fake_predictor.predict.return_value = (
            np.zeros((64, 64), dtype=bool)[None, :, :],
            None,
            None,
        )

        with self.assertRaises(RuntimeError) as ctx:
            seg.segment_boxes(_make_image(), [_make_box()])
        self.assertIn("expected", str(ctx.exception))

    def test_i03_non_pil_image_raises(self) -> None:
        harness = _MedSAMHarness()
        seg, _, _ = harness.construct()

        with self.assertRaises(TypeError):
            seg.segment_boxes("not_an_image", [])  # type: ignore

    def test_i04_non_list_boxes_raises(self) -> None:
        harness = _MedSAMHarness()
        seg, _, _ = harness.construct()

        with self.assertRaises(TypeError):
            seg.segment_boxes(_make_image(), (_make_box(),))  # type: ignore

    def test_i05_invalid_box_coordinates_raise(self) -> None:
        harness = _MedSAMHarness()
        seg, _, _ = harness.construct()

        bad_box = _make_box("R1", ymin=float("nan"))
        with self.assertRaises(ValueError):
            seg.segment_boxes(_make_image(), [bad_box])


# ---------------------------------------------------------------------------
# J. Package import — MedSAMRegionSegmenter lazy export
# ---------------------------------------------------------------------------


class TestMedSAMPackageExport(unittest.TestCase):
    """Tests that MedSAMRegionSegmenter is reachable via package lazy export."""

    def test_j01_exported_from_package(self) -> None:
        from localization import MedSAMRegionSegmenter as MRS

        self.assertIs(MRS, MedSAMRegionSegmenter)

    def test_j02_lightweight_import_no_torch(self) -> None:
        """Importing localization.rle or lightweight interfaces must not import torch."""
        # torch/segment_anything may already be in sys.modules from earlier tests
        sys.modules.pop("torch", None)
        sys.modules.pop("segment_anything", None)
        import localization.rle  # noqa: F401
        from localization.segmenter import (  # noqa: F401
            BaseRegionSegmenter,
            SegmenterConfig,
        )
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("segment_anything", sys.modules)


if __name__ == "__main__":
    unittest.main()
