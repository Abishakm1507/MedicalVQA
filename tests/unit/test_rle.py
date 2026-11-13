"""Unit tests for Run-Length Encoding (RLE) mask serialization.

Tests:
1. Empty mask (all zeros).
2. Full mask (all ones).
3. Single foreground pixel.
4. Single background pixel (1x1 False and 1x1 True).
5. Multiple disconnected foreground regions.
6. Non-square masks (tall and wide).
7. One-row and one-column masks (1xW, Hx1).
8. Round-trip encoding and decoding fidelity (identity).
9. Invalid input dimensions (1D, 3D, 0x0).
10. Invalid RLE syntax (missing colon, malformed header, non-integer tokens).
11. Invalid run counts (negative count, sum mismatch).
12. Decoded mask shape preservation.
13. Large mask efficiency check.
14. Type checking for non-ndarray and non-str inputs.
15. Non-boolean numpy arrays (e.g. uint8 binary 0/1 masks).
16. Non-binary values rejected (e.g. array with values 2, -1, 0.5).
17. Consecutive zero counts in RLE decoded string.
"""

import unittest

import numpy as np

from localization.rle import rle_decode, rle_encode


class TestRLESerialization(unittest.TestCase):
    """Test suite for binary mask Run-Length Encoding and Decoding."""

    def test_01_empty_mask(self) -> None:
        """1. Empty mask (all zeros / False)."""
        mask = np.zeros((10, 20), dtype=bool)
        encoded = rle_encode(mask)
        self.assertEqual(encoded, "10,20:200")
        decoded = rle_decode(encoded)
        self.assertEqual(decoded.shape, (10, 20))
        self.assertEqual(decoded.dtype, np.bool_)
        np.testing.assert_array_equal(decoded, mask)

    def test_02_full_mask(self) -> None:
        """2. Full mask (all ones / True)."""
        mask = np.ones((8, 12), dtype=bool)
        encoded = rle_encode(mask)
        # Leading 0 background, then 96 foreground
        self.assertEqual(encoded, "8,12:0 96")
        decoded = rle_decode(encoded)
        self.assertEqual(decoded.shape, (8, 12))
        np.testing.assert_array_equal(decoded, mask)

    def test_03_single_foreground_pixel(self) -> None:
        """3. Single foreground pixel in a larger canvas."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True  # Index in row-major: 2*5 + 3 = 13
        encoded = rle_encode(mask)
        # 13 background, 1 foreground, 11 background = 25 total
        self.assertEqual(encoded, "5,5:13 1 11")
        decoded = rle_decode(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_04_single_pixel_masks(self) -> None:
        """4. 1x1 single background pixel and 1x1 single foreground pixel."""
        # 1x1 background
        m_bg = np.array([[False]], dtype=bool)
        enc_bg = rle_encode(m_bg)
        self.assertEqual(enc_bg, "1,1:1")
        np.testing.assert_array_equal(rle_decode(enc_bg), m_bg)

        # 1x1 foreground
        m_fg = np.array([[True]], dtype=bool)
        enc_fg = rle_encode(m_fg)
        self.assertEqual(enc_fg, "1,1:0 1")
        np.testing.assert_array_equal(rle_decode(enc_fg), m_fg)

    def test_05_multiple_disconnected_regions(self) -> None:
        """5. Multiple disconnected foreground regions."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0:2] = True  # row 0: [True, True, False, False, False, False]
        mask[2, 4:] = True   # row 2: [False, False, False, False, True, True]
        mask[5, 1:3] = True  # row 5: [False, True, True, False, False, False]

        encoded = rle_encode(mask)
        decoded = rle_decode(encoded)
        self.assertEqual(decoded.shape, (6, 6))
        np.testing.assert_array_equal(decoded, mask)

    def test_06_non_square_masks(self) -> None:
        """6. Non-square masks (tall and wide)."""
        # Wide mask: 4 x 25
        wide = np.zeros((4, 25), dtype=bool)
        wide[1:3, 5:20] = True
        enc_wide = rle_encode(wide)
        self.assertTrue(enc_wide.startswith("4,25:"))
        np.testing.assert_array_equal(rle_decode(enc_wide), wide)

        # Tall mask: 30 x 7
        tall = np.zeros((30, 7), dtype=bool)
        tall[10:25, 2:5] = True
        enc_tall = rle_encode(tall)
        self.assertTrue(enc_tall.startswith("30,7:"))
        np.testing.assert_array_equal(rle_decode(enc_tall), tall)

    def test_07_one_row_and_one_column_masks(self) -> None:
        """7. 1-row (1 x W) and 1-column (H x 1) masks."""
        # 1 x 15 row
        row = np.array([[False, True, True, False, True, False, False, False, True, True, True, False, False, False, False]])
        enc_row = rle_encode(row)
        self.assertTrue(enc_row.startswith("1,15:"))
        np.testing.assert_array_equal(rle_decode(enc_row), row)

        # 10 x 1 column
        col = np.array([[False], [True], [True], [False], [False], [True], [False], [False], [False], [True]])
        enc_col = rle_encode(col)
        self.assertTrue(enc_col.startswith("10,1:"))
        np.testing.assert_array_equal(rle_decode(enc_col), col)

    def test_08_round_trip_random_patterns(self) -> None:
        """8. Round-trip encoding and decoding on various synthetic deterministic patterns."""
        rng = np.random.RandomState(42)
        shapes = [(16, 16), (23, 47), (100, 50), (37, 37)]
        for h, w in shapes:
            mask = rng.rand(h, w) > 0.65
            encoded = rle_encode(mask)
            decoded = rle_decode(encoded)
            self.assertEqual(decoded.shape, (h, w))
            self.assertEqual(decoded.dtype, np.bool_)
            np.testing.assert_array_equal(decoded, mask, err_msg=f"Failed round-trip for shape ({h}, {w})")

    def test_09_invalid_input_dimensions(self) -> None:
        """9. Invalid input dimensions (1D, 3D, zero dimensions)."""
        with self.assertRaises(ValueError):
            rle_encode(np.array([True, False, True]))  # 1D

        with self.assertRaises(ValueError):
            rle_encode(np.zeros((2, 2, 2), dtype=bool))  # 3D

        with self.assertRaises(ValueError):
            rle_encode(np.zeros((0, 10), dtype=bool))  # 0 height

        with self.assertRaises(ValueError):
            rle_encode(np.zeros((10, 0), dtype=bool))  # 0 width

    def test_10_invalid_rle_syntax(self) -> None:
        """10. Malformed RLE string syntax."""
        with self.assertRaises(ValueError):
            rle_decode("10,10")  # missing colon

        with self.assertRaises(ValueError):
            rle_decode("10:100")  # missing width

        with self.assertRaises(ValueError):
            rle_decode("ten,10:100")  # non-integer dimension

        with self.assertRaises(ValueError):
            rle_decode("10,10:")  # missing counts

        with self.assertRaises(ValueError):
            rle_decode("10,10:abc")  # non-integer counts

    def test_11_invalid_run_counts(self) -> None:
        """11. Invalid run counts (negative counts, sum mismatch)."""
        # Sum is 99, expected 100
        with self.assertRaises(ValueError):
            rle_decode("10,10:50 49")

        # Sum is 101, expected 100
        with self.assertRaises(ValueError):
            rle_decode("10,10:50 51")

        # Negative count
        with self.assertRaises(ValueError):
            rle_decode("10,10:50 -10 60")

        # Zero or negative dimensions in decode
        with self.assertRaises(ValueError):
            rle_decode("-5,10:50")

    def test_12_decoded_mask_shape_preservation(self) -> None:
        """12. Decoded mask preserves exact height and width (order='C')."""
        h, w = 17, 31
        mask = np.zeros((h, w), dtype=bool)
        mask[3, 5] = True
        mask[16, 30] = True
        encoded = rle_encode(mask)
        decoded = rle_decode(encoded)
        self.assertEqual(decoded.shape, (h, w))
        self.assertTrue(decoded[3, 5])
        self.assertTrue(decoded[16, 30])
        self.assertFalse(decoded[5, 3])

    def test_13_type_errors(self) -> None:
        """13. Type checking for non-ndarray and non-str inputs."""
        with self.assertRaises(TypeError):
            rle_encode([[True, False], [False, True]])  # type: ignore

        with self.assertRaises(TypeError):
            rle_decode(12345)  # type: ignore

    def test_14_large_mask_performance(self) -> None:
        """14. Verifies large radiograph mask (1024 x 1024) performance."""
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[300:500, 400:600] = True
        encoded = rle_encode(mask)
        decoded = rle_decode(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_15_non_boolean_input_array(self) -> None:
        """15. Binary integer mask (uint8 0 and 1) converts cleanly without error."""
        mask_uint8 = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        encoded = rle_encode(mask_uint8)
        self.assertEqual(encoded, "2,3:1 2 2 1")
        decoded = rle_decode(encoded)
        self.assertEqual(decoded.dtype, np.bool_)
        np.testing.assert_array_equal(decoded, mask_uint8.astype(bool))

    def test_16_non_binary_values_rejected(self) -> None:
        """16. Non-binary values (e.g. array containing 2, 3, or negative) must raise ValueError."""
        non_bin_arr = np.array([[0, 2], [1, 0]], dtype=np.int32)
        with self.assertRaises(ValueError):
            rle_encode(non_bin_arr)

        non_bin_float = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            rle_encode(non_bin_float)

    def test_17_consecutive_zero_counts_handling(self) -> None:
        """17. Decodes correctly even when consecutive zero counts appear in RLE string."""
        # e.g., '2,2:0 0 4' represents:
        # starts with background: run=0 -> flips to foreground: run=0 -> flips to background: run=4
        decoded = rle_decode("2,2:0 0 4")
        expected = np.zeros((2, 2), dtype=bool)
        np.testing.assert_array_equal(decoded, expected)


if __name__ == "__main__":
    unittest.main()
