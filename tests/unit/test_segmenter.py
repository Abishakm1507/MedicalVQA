"""Unit tests for FS-003 Stage 3: segmentation interface and MockRegionSegmenter.

Tests are grouped by concern:

A. SegmenterConfig — dataclass validation.
B. SegmentationResult — container validation.
C. BaseRegionSegmenter — abstract interface contract.
D. MockRegionSegmenter — interface behaviour.
E. MockRegionSegmenter — determinism and reproducibility.
F. MockRegionSegmenter — mask correctness (dtype, shape, pixel values).
G. MockRegionSegmenter — region_id preservation.
H. MockRegionSegmenter — RLE round-trip correctness.
I. MockRegionSegmenter — empty input.
J. MockRegionSegmenter — boundary / out-of-bounds boxes.
K. MockRegionSegmenter — multiple candidate boxes.
L. MockRegionSegmenter — invalid argument types.
M. Package import — no heavy ML dependencies required.

All test data is fully deterministic (no random state, no machine-specific paths,
no synthetic package injection, no mocked dependencies).

Tests in group M verify that importing ``localization.segmenter`` does not pull in
heavyweight ML dependencies. Because ``sys.modules`` is process-global and persists
across tests within a single pytest session, these checks are performed in a fresh
subprocess so they are independent of test execution order.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.schemas import CandidateBox
from localization.rle import rle_decode
from localization.segmenter import (
    BaseRegionSegmenter,
    MockRegionSegmenter,
    SegmentationResult,
    SegmenterConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width: int = 128, height: int = 96, mode: str = "RGB") -> Image.Image:
    """Return a solid-colour PIL Image for testing."""
    return Image.new(mode, (width, height), color=(128, 128, 128))


def _make_box(
    region_id: str = "R1",
    ymin: int = 10,
    xmin: int = 20,
    ymax: int = 40,
    xmax: int = 60,
    score: float = 0.9,
) -> CandidateBox:
    """Return a CandidateBox with simple absolute coordinates."""
    img_h, img_w = 96, 128
    return CandidateBox(
        region_id=region_id,
        box_2d=(ymin, xmin, ymax, xmax),
        normalized_box=(ymin / img_h, xmin / img_w, ymax / img_h, xmax / img_w),
        score=score,
        label="lesion",
    )


# ---------------------------------------------------------------------------
# A. SegmenterConfig
# ---------------------------------------------------------------------------

class TestSegmenterConfig(unittest.TestCase):

    def test_a01_defaults(self) -> None:
        """Default config constructs without error."""
        cfg = SegmenterConfig()
        self.assertEqual(cfg.device, "cpu")
        self.assertIsNone(cfg.checkpoint_path)
        self.assertEqual(cfg.image_size, 1024)

    def test_a02_valid_cpu_device(self) -> None:
        cfg = SegmenterConfig(device="cpu")
        self.assertEqual(cfg.device, "cpu")

    def test_a03_valid_cuda_device(self) -> None:
        cfg = SegmenterConfig(device="cuda")
        self.assertEqual(cfg.device, "cuda")

    def test_a04_valid_cuda_index_device(self) -> None:
        cfg = SegmenterConfig(device="cuda:0")
        self.assertEqual(cfg.device, "cuda:0")

    def test_a05_invalid_device_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(device="mps")

    def test_a06_zero_image_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(image_size=0)

    def test_a07_negative_image_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(image_size=-1)

    def test_a08_checkpoint_path_stored(self) -> None:
        cfg = SegmenterConfig(checkpoint_path="/path/to/medsam_vit_b.pth")
        self.assertEqual(cfg.checkpoint_path, "/path/to/medsam_vit_b.pth")

    def test_a09_cuda_malformed_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(device="cuda:abc")

    def test_a10_cuda_empty_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(device="cuda:")

    def test_a11_cuda_negative_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            SegmenterConfig(device="cuda:-1")


# ---------------------------------------------------------------------------
# B. SegmentationResult
# ---------------------------------------------------------------------------

class TestSegmentationResult(unittest.TestCase):

    def _valid_mask(self, h: int = 96, w: int = 128) -> np.ndarray:
        return np.zeros((h, w), dtype=bool)

    def test_b01_valid_construction(self) -> None:
        mask = self._valid_mask()
        rle = "96,128:12288"
        result = SegmentationResult(region_id="R1", mask=mask, mask_rle=rle)
        self.assertEqual(result.region_id, "R1")
        self.assertIs(result.mask, mask)
        self.assertEqual(result.mask_rle, rle)

    def test_b02_mask_must_be_ndarray(self) -> None:
        with self.assertRaises(TypeError):
            SegmentationResult(
                region_id="R1",
                mask=[[False, True], [True, False]],  # type: ignore
                mask_rle="2,2:1 1 1 1",
            )

    def test_b03_mask_must_be_2d(self) -> None:
        with self.assertRaises(ValueError):
            SegmentationResult(
                region_id="R1",
                mask=np.zeros((4,), dtype=bool),
                mask_rle="1,4:4",
            )

    def test_b04_mask_must_be_bool_dtype(self) -> None:
        with self.assertRaises(ValueError):
            SegmentationResult(
                region_id="R1",
                mask=np.zeros((4, 4), dtype=np.uint8),
                mask_rle="4,4:16",
            )


# ---------------------------------------------------------------------------
# C. BaseRegionSegmenter — abstract interface
# ---------------------------------------------------------------------------

class TestBaseRegionSegmenterAbstract(unittest.TestCase):

    def test_c01_cannot_instantiate_directly(self) -> None:
        """BaseRegionSegmenter is abstract and cannot be instantiated."""
        with self.assertRaises(TypeError):
            BaseRegionSegmenter()  # type: ignore

    def test_c02_concrete_subclass_must_implement_segment_boxes(self) -> None:
        """A subclass that does not implement segment_boxes raises TypeError."""
        class Incomplete(BaseRegionSegmenter):
            pass

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore

    def test_c03_minimal_concrete_subclass(self) -> None:
        """A minimal concrete subclass that implements segment_boxes is accepted."""
        class Minimal(BaseRegionSegmenter):
            def segment_boxes(self, image, boxes):
                return []

        seg = Minimal()
        result = seg.segment_boxes(_make_image(), [])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# D. MockRegionSegmenter — interface behaviour
# ---------------------------------------------------------------------------

class TestMockRegionSegmenterInterface(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image(width=128, height=96)

    def test_d01_is_subclass_of_base(self) -> None:
        self.assertIsInstance(self.seg, BaseRegionSegmenter)

    def test_d02_has_is_mock_flag(self) -> None:
        self.assertTrue(MockRegionSegmenter.IS_MOCK)

    def test_d03_non_pil_image_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(np.zeros((96, 128, 3), dtype=np.uint8), [])  # type: ignore

    def test_d04_non_list_boxes_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(self.image, (_make_box(),))  # type: ignore (tuple, not list)

    def test_d05_non_candidatebox_element_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(self.image, [{"region_id": "R1"}])  # type: ignore

    def test_d06_zero_dimension_image_raises_value_error(self) -> None:
        # PIL raises DecompressionBombError / ValueError for 0-size; test our guard
        # We bypass PIL constructor and patch size manually via subclass
        class TinyImage(Image.Image):
            mode = "RGB"
            @property
            def size(self):
                return (0, 100)

        with self.assertRaises(ValueError):
            self.seg.segment_boxes(TinyImage(), [_make_box()])

    def test_d07_grayscale_image_raises_value_error(self) -> None:
        gray_image = Image.new("L", (100, 100), color=128)
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(gray_image, [_make_box()])

    def test_d08_rgba_image_raises_value_error(self) -> None:
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(rgba_image, [_make_box()])


# ---------------------------------------------------------------------------
# E. MockRegionSegmenter — determinism
# ---------------------------------------------------------------------------

class TestMockSegmenterDeterminism(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image(width=200, height=150)
        self.boxes = [
            _make_box("R1", 10, 20, 50, 80),
            _make_box("R2", 60, 100, 120, 180),
        ]

    def test_e01_same_inputs_produce_identical_masks(self) -> None:
        results_a = self.seg.segment_boxes(self.image, self.boxes)
        results_b = self.seg.segment_boxes(self.image, self.boxes)
        self.assertEqual(len(results_a), len(results_b))
        for a, b in zip(results_a, results_b, strict=False):
            np.testing.assert_array_equal(a.mask, b.mask)
            self.assertEqual(a.mask_rle, b.mask_rle)

    def test_e02_same_inputs_different_segmenter_instances(self) -> None:
        seg2 = MockRegionSegmenter()
        results_a = self.seg.segment_boxes(self.image, self.boxes)
        results_b = seg2.segment_boxes(self.image, self.boxes)
        for a, b in zip(results_a, results_b, strict=False):
            np.testing.assert_array_equal(a.mask, b.mask)

    def test_e03_different_boxes_produce_different_masks(self) -> None:
        boxes_b = [_make_box("R1", 5, 5, 20, 20)]
        result_a = self.seg.segment_boxes(self.image, [_make_box("R1", 10, 10, 50, 50)])
        result_b = self.seg.segment_boxes(self.image, boxes_b)
        # Masks must differ because bounding boxes differ
        self.assertFalse(np.array_equal(result_a[0].mask, result_b[0].mask))


# ---------------------------------------------------------------------------
# F. MockRegionSegmenter — mask correctness
# ---------------------------------------------------------------------------

class TestMockSegmenterMaskCorrectness(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.w, self.h = 128, 96
        self.image = _make_image(width=self.w, height=self.h)

    def test_f01_mask_dtype_is_bool(self) -> None:
        box = _make_box("R1", 10, 20, 40, 60)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertEqual(results[0].mask.dtype, np.bool_)

    def test_f02_mask_shape_equals_image_dimensions(self) -> None:
        box = _make_box("R1", 10, 20, 40, 60)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertEqual(results[0].mask.shape, (self.h, self.w))

    def test_f03_pixels_inside_box_are_true(self) -> None:
        ymin, xmin, ymax, xmax = 10, 20, 40, 60
        box = _make_box("R1", ymin, xmin, ymax, xmax)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Every pixel strictly inside [ymin:ymax, xmin:xmax] is True
        self.assertTrue(np.all(mask[ymin:ymax, xmin:xmax]))

    def test_f04_pixels_outside_box_are_false(self) -> None:
        ymin, xmin, ymax, xmax = 10, 20, 40, 60
        box = _make_box("R1", ymin, xmin, ymax, xmax)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Pixel above the box
        self.assertFalse(mask[ymin - 1, xmin])
        # Pixel to the left of the box
        self.assertFalse(mask[ymin, xmin - 1])
        # Pixel below the box
        self.assertFalse(mask[ymax, xmin])
        # Pixel to the right of the box
        self.assertFalse(mask[ymin, xmax])

    def test_f05_foreground_pixel_count_matches_box_area(self) -> None:
        ymin, xmin, ymax, xmax = 10, 20, 40, 60
        expected_area = (ymax - ymin) * (xmax - xmin)
        box = _make_box("R1", ymin, xmin, ymax, xmax)
        results = self.seg.segment_boxes(self.image, [box])
        actual_count = int(np.sum(results[0].mask))
        self.assertEqual(actual_count, expected_area)

    def test_f06_all_black_mask_for_zero_area_box(self) -> None:
        """A box where ymin==ymax or xmin==xmax produces an all-False mask."""
        box = _make_box("R1", 20, 30, 20, 60)   # ymin == ymax → zero height
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_f07_non_square_image(self) -> None:
        """Test with a tall image (H > W)."""
        tall = _make_image(width=50, height=200)
        box = CandidateBox(
            region_id="R1",
            box_2d=(50, 10, 150, 40),
            normalized_box=(50 / 200, 10 / 50, 150 / 200, 40 / 50),
            score=0.8,
        )
        results = self.seg.segment_boxes(tall, [box])
        self.assertEqual(results[0].mask.shape, (200, 50))
        self.assertEqual(results[0].mask.dtype, np.bool_)
        self.assertTrue(np.all(results[0].mask[50:150, 10:40]))

    def test_f08_1x1_image(self) -> None:
        """Degenerate 1×1 image with a 1×1 box."""
        tiny = _make_image(width=1, height=1)
        box = CandidateBox(
            region_id="R1",
            box_2d=(0, 0, 1, 1),
            normalized_box=(0.0, 0.0, 1.0, 1.0),
            score=0.5,
        )
        results = self.seg.segment_boxes(tiny, [box])
        self.assertEqual(results[0].mask.shape, (1, 1))
        self.assertTrue(results[0].mask[0, 0])


# ---------------------------------------------------------------------------
# G. MockRegionSegmenter — region_id preservation
# ---------------------------------------------------------------------------

class TestMockSegmenterRegionIdPreservation(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image(width=128, height=96)

    def test_g01_single_region_id_preserved(self) -> None:
        box = _make_box("R3")
        results = self.seg.segment_boxes(self.image, [box])
        self.assertEqual(results[0].region_id, "R3")

    def test_g02_multiple_region_ids_preserved_in_order(self) -> None:
        boxes = [
            _make_box("R1", 5, 5, 20, 20),
            _make_box("R2", 30, 30, 60, 90),
            _make_box("R7", 70, 10, 90, 50),
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].region_id, "R1")
        self.assertEqual(results[1].region_id, "R2")
        self.assertEqual(results[2].region_id, "R7")

    def test_g03_result_count_equals_box_count(self) -> None:
        boxes = [_make_box(f"R{i}", i * 5, i * 5, i * 5 + 10, i * 5 + 10) for i in range(1, 6)]
        results = self.seg.segment_boxes(self.image, boxes)
        self.assertEqual(len(results), len(boxes))


# ---------------------------------------------------------------------------
# H. MockRegionSegmenter — RLE round-trip
# ---------------------------------------------------------------------------

class TestMockSegmenterRLERoundTrip(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image(width=128, height=96)

    def test_h01_rle_round_trip_single_box(self) -> None:
        box = _make_box("R1", 10, 20, 40, 60)
        results = self.seg.segment_boxes(self.image, [box])
        decoded = rle_decode(results[0].mask_rle)
        np.testing.assert_array_equal(decoded, results[0].mask)

    def test_h02_rle_header_encodes_image_dimensions(self) -> None:
        box = _make_box("R1", 10, 20, 40, 60)
        results = self.seg.segment_boxes(self.image, [box])
        header = results[0].mask_rle.split(":")[0]
        h_str, w_str = header.split(",")
        self.assertEqual(int(h_str), 96)   # image height
        self.assertEqual(int(w_str), 128)  # image width

    def test_h03_rle_round_trip_multiple_boxes(self) -> None:
        boxes = [
            _make_box("R1", 5, 5, 20, 30),
            _make_box("R2", 50, 70, 80, 110),
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        for result in results:
            decoded = rle_decode(result.mask_rle)
            np.testing.assert_array_equal(
                decoded, result.mask,
                err_msg=f"RLE round-trip failed for region {result.region_id}"
            )

    def test_h04_rle_for_empty_mask_is_valid(self) -> None:
        """Zero-area box produces an all-False mask and a valid RLE string."""
        box = _make_box("R1", 20, 30, 20, 60)   # zero-height box
        results = self.seg.segment_boxes(self.image, [box])
        decoded = rle_decode(results[0].mask_rle)
        np.testing.assert_array_equal(decoded, results[0].mask)
        self.assertFalse(np.any(decoded))


# ---------------------------------------------------------------------------
# I. MockRegionSegmenter — empty input
# ---------------------------------------------------------------------------

class TestMockSegmenterEmptyInput(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image(width=128, height=96)

    def test_i01_empty_boxes_returns_empty_list(self) -> None:
        results = self.seg.segment_boxes(self.image, [])
        self.assertEqual(results, [])

    def test_i02_empty_result_is_a_list(self) -> None:
        results = self.seg.segment_boxes(self.image, [])
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# J. MockRegionSegmenter — boundary and out-of-bounds boxes
# ---------------------------------------------------------------------------

class TestMockSegmenterBoundaryBoxes(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.w, self.h = 128, 96
        self.image = _make_image(width=self.w, height=self.h)

    def test_j01_box_exactly_at_image_boundary(self) -> None:
        """Box that fills the entire image exactly."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(0, 0, self.h, self.w),
            normalized_box=(0.0, 0.0, 1.0, 1.0),
            score=0.9,
        )
        results = self.seg.segment_boxes(self.image, [box])
        self.assertTrue(np.all(results[0].mask))  # every pixel True

    def test_j02_box_exceeds_image_boundary_clamped_safely(self) -> None:
        """Coordinates beyond image bounds are clamped, not raised as errors."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(-10, -20, self.h + 50, self.w + 100),
            normalized_box=(0.0, 0.0, 1.0, 1.0),
            score=0.9,
        )
        results = self.seg.segment_boxes(self.image, [box])
        self.assertEqual(results[0].mask.shape, (self.h, self.w))
        self.assertTrue(np.all(results[0].mask))  # all clamped to full image

    def test_j03_box_in_top_left_corner(self) -> None:
        box = CandidateBox(
            region_id="R1",
            box_2d=(0, 0, 10, 15),
            normalized_box=(0.0, 0.0, 10 / 96, 15 / 128),
            score=0.8,
        )
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        self.assertTrue(np.all(mask[0:10, 0:15]))
        self.assertFalse(mask[10, 0])
        self.assertFalse(mask[0, 15])

    def test_j04_box_in_bottom_right_corner(self) -> None:
        box = CandidateBox(
            region_id="R1",
            box_2d=(86, 118, 96, 128),
            normalized_box=(86 / 96, 118 / 128, 1.0, 1.0),
            score=0.8,
        )
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        self.assertTrue(np.all(mask[86:96, 118:128]))
        self.assertFalse(mask[85, 118])

    def test_j05_inverted_box_coordinates_produce_empty_mask(self) -> None:
        """When ymin > ymax or xmin > xmax after clamping, the mask is all-False."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(50, 60, 10, 20),   # ymin > ymax
            normalized_box=(0.5, 0.5, 0.1, 0.1),
            score=0.5,
        )
        results = self.seg.segment_boxes(self.image, [box])
        # After clamping ymin=50, ymax=10 → ymax(10) < ymin(50) → empty
        self.assertFalse(np.any(results[0].mask))

    def test_j06_single_row_box(self) -> None:
        """Box with height == 1 pixel."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(30, 20, 31, 80),
            normalized_box=(30 / 96, 20 / 128, 31 / 96, 80 / 128),
            score=0.7,
        )
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        self.assertTrue(np.all(mask[30:31, 20:80]))
        self.assertFalse(np.any(mask[29, :]))
        self.assertFalse(np.any(mask[31, :]))

    def test_j07_single_column_box(self) -> None:
        """Box with width == 1 pixel."""
        box = CandidateBox(
            region_id="R1",
            box_2d=(10, 50, 70, 51),
            normalized_box=(10 / 96, 50 / 128, 70 / 96, 51 / 128),
            score=0.7,
        )
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        self.assertTrue(np.all(mask[10:70, 50:51]))
        self.assertFalse(np.any(mask[:, 49]))
        self.assertFalse(np.any(mask[:, 51]))

    def test_j08_user_reported_bug_completely_below_image_produces_empty_mask(self) -> None:
        """User example: H=100, box_2d=(120, 10, 140, 30) completely below image -> empty mask."""
        img = _make_image(width=100, height=100)
        box = CandidateBox("R1", (120, 10, 140, 30), (1.2, 0.1, 1.4, 0.3), 0.9)
        results = self.seg.segment_boxes(img, [box])
        self.assertEqual(results[0].mask.shape, (100, 100))
        self.assertFalse(np.any(results[0].mask))
        self.assertEqual(results[0].mask_rle, "100,100:10000")

    def test_j09_completely_above_image_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (-50, 20, -10, 60), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j10_completely_left_of_image_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (20, -60, 50, -10), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j11_completely_right_of_image_produces_empty_mask(self) -> None:
        # width is 128
        box = CandidateBox("R1", (20, 150, 50, 190), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j12_touching_top_boundary_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (-30, 20, 0, 60), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j13_touching_bottom_boundary_produces_empty_mask(self) -> None:
        # height is 96
        box = CandidateBox("R1", (96, 20, 120, 60), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j14_touching_left_boundary_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (20, -40, 50, 0), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j15_touching_right_boundary_produces_empty_mask(self) -> None:
        # width is 128
        box = CandidateBox("R1", (20, 128, 50, 160), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j16_partially_outside_top_boundary(self) -> None:
        box = CandidateBox("R1", (-20, 10, 30, 50), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Intersection is [0:30, 10:50]
        self.assertTrue(np.all(mask[0:30, 10:50]))
        self.assertFalse(mask[30, 10])
        self.assertEqual(int(np.sum(mask)), 30 * 40)

    def test_j17_partially_outside_bottom_boundary(self) -> None:
        # height is 96
        box = CandidateBox("R1", (70, 10, 120, 50), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Intersection is [70:96, 10:50]
        self.assertTrue(np.all(mask[70:96, 10:50]))
        self.assertFalse(mask[69, 10])
        self.assertEqual(int(np.sum(mask)), 26 * 40)

    def test_j18_partially_outside_left_boundary(self) -> None:
        box = CandidateBox("R1", (10, -25, 40, 35), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Intersection is [10:40, 0:35]
        self.assertTrue(np.all(mask[10:40, 0:35]))
        self.assertFalse(mask[10, 35])
        self.assertEqual(int(np.sum(mask)), 30 * 35)

    def test_j19_partially_outside_right_boundary(self) -> None:
        # width is 128
        box = CandidateBox("R1", (10, 100, 40, 160), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        mask = results[0].mask
        # Intersection is [10:40, 100:128]
        self.assertTrue(np.all(mask[10:40, 100:128]))
        self.assertFalse(mask[10, 99])
        self.assertEqual(int(np.sum(mask)), 30 * 28)

    def test_j20_zero_width_box_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (10, 40, 30, 40), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))

    def test_j21_zero_height_box_produces_empty_mask(self) -> None:
        box = CandidateBox("R1", (25, 10, 25, 50), (0.0, 0.0, 0.0, 0.0), 0.9)
        results = self.seg.segment_boxes(self.image, [box])
        self.assertFalse(np.any(results[0].mask))


# ---------------------------------------------------------------------------
# K. MockRegionSegmenter — multiple candidate boxes
# ---------------------------------------------------------------------------

class TestMockSegmenterMultipleBoxes(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.w, self.h = 200, 150
        self.image = _make_image(width=self.w, height=self.h)

    def test_k01_two_non_overlapping_boxes(self) -> None:
        boxes = [
            CandidateBox("R1", (10, 10, 40, 60),
                         (10 / 150, 10 / 200, 40 / 150, 60 / 200), 0.9),
            CandidateBox("R2", (80, 120, 130, 180),
                         (80 / 150, 120 / 200, 130 / 150, 180 / 200), 0.8),
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        self.assertEqual(len(results), 2)

        m0, m1 = results[0].mask, results[1].mask

        # Both masks have the same shape
        self.assertEqual(m0.shape, (self.h, self.w))
        self.assertEqual(m1.shape, (self.h, self.w))

        # Pixels inside box 0 are True in m0 and False in m1
        self.assertTrue(np.all(m0[10:40, 10:60]))
        self.assertFalse(np.any(m1[10:40, 10:60]))

        # Pixels inside box 1 are True in m1 and False in m0
        self.assertTrue(np.all(m1[80:130, 120:180]))
        self.assertFalse(np.any(m0[80:130, 120:180]))

    def test_k02_seven_boxes_max_regions(self) -> None:
        """Up to 7 boxes (paper limit MAX_ALLOWED_REGIONS) all produce results."""
        boxes = [
            CandidateBox(f"R{i}", (i * 10, i * 10, i * 10 + 8, i * 10 + 8),
                         (0.0, 0.0, 0.1, 0.1), 0.5)
            for i in range(1, 8)
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        self.assertEqual(len(results), 7)
        for idx, result in enumerate(results):
            self.assertEqual(result.region_id, f"R{idx + 1}")

    def test_k03_masks_are_independent_objects(self) -> None:
        """Each result's mask is a distinct array (mutating one must not affect another)."""
        boxes = [
            _make_box("R1", 5, 5, 20, 30),
            _make_box("R2", 50, 50, 80, 100),
        ]
        results = self.seg.segment_boxes(self.image, boxes)
        # Mutate the first mask
        results[0].mask[0, 0] = True
        # Second mask must be unaffected at [0, 0]
        self.assertFalse(results[1].mask[0, 0])


# ---------------------------------------------------------------------------
# L. MockRegionSegmenter — invalid argument types
# ---------------------------------------------------------------------------

class TestMockSegmenterTypeErrors(unittest.TestCase):

    def setUp(self) -> None:
        self.seg = MockRegionSegmenter()
        self.image = _make_image()

    def test_l01_string_instead_of_image(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes("not_an_image", [_make_box()])  # type: ignore

    def test_l02_none_instead_of_image(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(None, [_make_box()])  # type: ignore

    def test_l03_tuple_instead_of_list(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(self.image, (_make_box(),))  # type: ignore

    def test_l04_dict_in_boxes_list(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(self.image, [{"region_id": "R1"}])  # type: ignore

    def test_l05_none_in_boxes_list(self) -> None:
        with self.assertRaises(TypeError):
            self.seg.segment_boxes(self.image, [None])  # type: ignore

    def test_l06_box_2d_not_4_tuple_raises(self) -> None:
        bad_box = CandidateBox("R1", (10, 20, 30), (0.1, 0.2, 0.3, 0.4), 0.8)  # type: ignore
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(self.image, [bad_box])

    def test_l07_box_2d_nan_coordinate_raises(self) -> None:
        bad_box = CandidateBox("R1", (float("nan"), 10, 30, 40), (0.1, 0.1, 0.3, 0.4), 0.8)  # type: ignore
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(self.image, [bad_box])

    def test_l08_box_2d_inf_coordinate_raises(self) -> None:
        bad_box = CandidateBox("R1", (10, float("inf"), 30, 40), (0.1, 0.1, 0.3, 0.4), 0.8)  # type: ignore
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(self.image, [bad_box])

    def test_l09_box_2d_string_coordinate_raises(self) -> None:
        bad_box = CandidateBox("R1", (10, "20", 30, 40), (0.1, 0.1, 0.3, 0.4), 0.8)  # type: ignore
        with self.assertRaises(ValueError):
            self.seg.segment_boxes(self.image, [bad_box])


# ---------------------------------------------------------------------------
# M. Package import — no heavy ML dependencies required
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _run_isolated_import_check(module_names: tuple[str, ...]) -> subprocess.CompletedProcess:
    """Spawn a fresh Python process to import localization.segmenter and check that
    none of *module_names* (e.g. 'torch', 'torchvision') appear in sys.modules.

    Running in a subprocess guarantees order independence: no prior test in the
    current pytest session can have polluted the new process's sys.modules.
    """
    code = (
        "import sys\n"
        "import localization.segmenter\n"
        f"missing = [m for m in {list(module_names)!r} if m in sys.modules]\n"
        "sys.exit(1 if missing else 0)\n"
    )
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = _PROJECT_ROOT
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_PROJECT_ROOT,
    )


class TestNoHeavyImportsRequired(unittest.TestCase):

    def test_m01_torch_not_imported_by_segmenter(self) -> None:
        """Importing localization.segmenter must not pull in torch."""
        result = _run_isolated_import_check(("torch",))
        self.assertEqual(
            result.returncode,
            0,
            msg="torch was imported by localization.segmenter: "
            + result.stderr
            + result.stdout,
        )

    def test_m02_torchvision_not_imported_by_segmenter(self) -> None:
        """Importing localization.segmenter must not pull in torchvision."""
        result = _run_isolated_import_check(("torchvision",))
        self.assertEqual(
            result.returncode,
            0,
            msg="torchvision was imported by localization.segmenter: "
            + result.stderr
            + result.stdout,
        )

    def test_m03_package_level_import_works(self) -> None:
        """localization.MockRegionSegmenter is reachable via package lazy export."""
        from localization import MockRegionSegmenter as MRS  # noqa: F401
        self.assertIs(MRS, MockRegionSegmenter)

    def test_m04_segmentation_result_importable_from_package(self) -> None:
        """localization.SegmentationResult is reachable via package lazy export."""
        from localization import SegmentationResult as SR  # noqa: F401
        self.assertIs(SR, SegmentationResult)


if __name__ == "__main__":
    unittest.main()
