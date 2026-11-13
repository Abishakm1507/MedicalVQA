"""Run-Length Encoding (RLE) serialization for binary masks.

Format:
    "{H},{W}:{counts}"

Where:
    - H: height of the 2D mask in pixels (positive integer)
    - W: width of the 2D mask in pixels (positive integer)
    - counts: whitespace-separated run lengths of alternating values.
              The traversal order is strictly ROW-MAJOR (C-contiguous).
              Per convention, counts always starts with the run-length of background (0 / False)
              pixels. If the mask begins with a foreground (1 / True) pixel, the first
              count is 0.

Traceability:
    - Feature Specification: FS-003 (Section 2, 3)
    - Schema Alignment: CandidateBox.mask_rle, GroundTruthRegion.mask_rle
"""

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Encodes a 2D boolean or binary mask into an RLE string.

    Traversal Order:
        Strictly row-major (C-contiguous, order='C'). Pixels are scanned row by row
        from left to right: (0, 0), (0, 1), ..., (0, W-1), (1, 0), ..., (H-1, W-1).

    Encoding Protocol:
        Runs alternate starting with background (0 / False). If the first pixel
        in row-major order is foreground (1 / True), a leading zero count '0'
        is prepended.

    Args:
        mask: 2D numpy array with boolean or binary (0/1) integer values.

    Returns:
        String in format '{H},{W}:{count_0} {count_1} ...'

    Raises:
        ValueError: If mask is not 2-dimensional, has zero/negative dimensions,
                    or contains non-binary values other than 0 and 1.
        TypeError: If mask is not a numpy ndarray.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected mask to be numpy.ndarray, got {type(mask).__name__}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-dimensional, got {mask.ndim} dimensions (shape={mask.shape})")

    height, width = mask.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got shape=({height}, {width})")

    # Flatten along row-major order without extra full-copy if already C-contiguous
    flat = mask.reshape(-1)

    # Validate that array elements are strictly binary (boolean or {0, 1})
    if flat.dtype == np.bool_:
        flat_bool = flat
    else:
        # Check for non-binary values (e.g. values other than 0 and 1)
        # Using fast numpy set-diff / min-max check
        if np.issubdtype(flat.dtype, np.number):
            unique_vals = np.unique(flat)
            if not np.all(np.isin(unique_vals, [0, 1])):
                raise ValueError(f"Mask contains non-binary values: {unique_vals}. Only binary (0/1/True/False) masks are accepted.")
            flat_bool = flat.astype(bool, copy=False)
        else:
            raise TypeError(f"Unsupported mask dtype: {flat.dtype}. Must be boolean or numeric binary (0/1).")

    total_pixels = flat_bool.size
    if total_pixels == 0:
        return f"{height},{width}:"

    # Find transition indices where values change
    diffs = flat_bool[1:] != flat_bool[:-1]
    change_indices = np.flatnonzero(diffs) + 1

    # Construct run lengths from split indices
    if change_indices.size == 0:
        # All pixels are identical
        if flat_bool[0]:
            # Entire mask is True: 0 background pixels, then total_pixels foreground
            return f"{height},{width}:0 {total_pixels}"
        else:
            # Entire mask is False: total_pixels background pixels
            return f"{height},{width}:{total_pixels}"

    # Build run lengths array
    run_starts = np.empty(change_indices.size + 1, dtype=np.int64)
    run_starts[0] = 0
    run_starts[1:] = change_indices

    run_lengths = np.empty(run_starts.size, dtype=np.int64)
    run_lengths[:-1] = np.diff(run_starts)
    run_lengths[-1] = total_pixels - run_starts[-1]

    counts: list[str] = []
    # If mask begins with foreground (True), prepend 0 background count
    if flat_bool[0]:
        counts.append("0")

    counts.extend(str(int(rl)) for rl in run_lengths)

    return f"{height},{width}:{' '.join(counts)}"


def rle_decode(rle: str) -> np.ndarray:
    """Decodes an RLE string back into a 2D boolean mask.

    Traversal Order:
        Strictly row-major (C-contiguous, order='C'). Reconstructed values fill
        (0, 0), (0, 1), ..., (0, W-1), (1, 0), ..., (H-1, W-1).

    Decoding Protocol:
        Interprets alternating counts starting from background (False).
        Verifies that the sum of run lengths equals exactly H * W.

    Args:
        rle: Encoded RLE string in format '{H},{W}:{counts}'.

    Returns:
        2D boolean numpy ndarray of shape (H, W).

    Raises:
        ValueError: If RLE format is malformed, dimensions are non-positive,
                    counts are negative, or total count does not equal H * W.
        TypeError: If rle is not a string.
    """
    if not isinstance(rle, str):
        raise TypeError(f"Expected rle to be str, got {type(rle).__name__}")

    if ":" not in rle:
        raise ValueError(f"Malformed RLE string: missing ':' separator in '{rle}'")

    header, counts_part = rle.split(":", 1)
    header_tokens = header.split(",")
    if len(header_tokens) != 2:
        raise ValueError(f"Malformed RLE header: expected 'H,W', got '{header}'")

    try:
        height = int(header_tokens[0].strip())
        width = int(header_tokens[1].strip())
    except ValueError as e:
        raise ValueError(f"Malformed RLE dimensions in header '{header}': must be integers") from e

    if height <= 0 or width <= 0:
        raise ValueError(f"RLE dimensions must be positive integers, got height={height}, width={width}")

    expected_total = height * width
    counts_tokens = counts_part.strip().split()

    if not counts_tokens:
        raise ValueError(f"Malformed RLE: missing count values for {height}x{width} mask")

    try:
        counts = [int(tok) for tok in counts_tokens]
    except ValueError as e:
        raise ValueError(f"Malformed run count in RLE string: '{counts_part}'") from e

    # Validate non-negative counts
    for c in counts:
        if c < 0:
            raise ValueError(f"Run count cannot be negative, got {c}")

    sum_counts = sum(counts)
    if sum_counts != expected_total:
        raise ValueError(
            f"RLE counts sum mismatch: expected {expected_total} (H={height} * W={width}), "
            f"but got sum of counts = {sum_counts}"
        )

    # Reconstruct flat boolean array
    flat = np.empty(expected_total, dtype=bool)
    current_val = False  # Always begins with background
    idx = 0

    for c in counts:
        if c > 0:
            flat[idx : idx + c] = current_val
            idx += c
        current_val = not current_val

    return flat.reshape((height, width), order="C")
