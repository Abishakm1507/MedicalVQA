"""Modality-aware image preprocessing and tensor preparation for region detection.

Traceability:
- Research Paper: Section III-B (Region Proposal)
- ADR 0006: Modality-Aware DICOM Preprocessing (CXR vs CT Decoupling)
"""

from typing import cast

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def prepare_image_for_detection(
    image: Image.Image,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Prepares a PIL Image for detection inference by converting to a normalized tensor.

    Args:
        image: Input PIL Image.
        device: Target torch device ('cpu' or 'cuda').

    Returns:
        Tensor of shape [3, H, W] in float32 normalized to [0.0, 1.0].

    Raises:
        ValueError: If image dimensions are non-positive.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: {width}x{height}. Dimensions must be positive."
        )

    # Ensure 3-channel RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    target_device = torch.device(device) if isinstance(device, str) else device
    tensor_img = TF.to_tensor(image).to(target_device)
    return cast(torch.Tensor, tensor_img)


def normalize_cxr_pixels(
    pixel_array: np.ndarray,
    photometric_interpretation: str = "MONOCHROME2",
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    """Preprocesses 2D Chest X-Ray (CXR) projection pixels into standard 8-bit representation.

    Note: Hounsfield Units do NOT apply to CXR projection radiographs.

    Args:
        pixel_array: Raw 2D numpy array of radiograph pixel values.
        photometric_interpretation: 'MONOCHROME1' (inverted) or 'MONOCHROME2' (standard).
        window_center: Optional VOI LUT window center (WC).
        window_width: Optional VOI LUT window width (WW).

    Returns:
        uint8 numpy array of shape [H, W] in range [0, 255].
    """
    arr = pixel_array.astype(np.float32)

    # Invert MONOCHROME1 so air is black and bone is bright (standard visual orientation)
    if photometric_interpretation.upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    if window_center is not None and window_width is not None and window_width > 0:
        # Standard DICOM linear VOI LUT windowing
        lower = window_center - 0.5 - (window_width - 1) / 2
        upper = window_center - 0.5 + (window_width - 1) / 2
        arr = np.clip(arr, lower, upper)
        norm_range = upper - lower
        if norm_range > 0:
            arr = ((arr - lower) / norm_range) * 255.0
        else:
            arr = np.zeros_like(arr)
    else:
        # Robust percentile scaling to protect against extreme dead-pixel outliers
        p_low, p_high = np.percentile(arr, (0.5, 99.5))
        if p_high > p_low:
            arr = np.clip(arr, p_low, p_high)
            arr = ((arr - p_low) / (p_high - p_low)) * 255.0
        else:
            min_v, max_v = np.min(arr), np.max(arr)
            if max_v > min_v:
                arr = ((arr - min_v) / (max_v - min_v)) * 255.0
            else:
                arr = np.zeros_like(arr)

    return cast(np.ndarray, np.round(arr).astype(np.uint8))


def normalize_ct_pixels(
    pixel_array: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    window_center: float = -600.0,
    window_width: float = 1500.0,
) -> np.ndarray:
    """Preprocesses volumetric Computed Tomography (CT) slices using calibrated Hounsfield Units.

    Strictly decoupled from CXR projection radiographs.

    Args:
        pixel_array: Raw CT slice pixel array.
        rescale_slope: DICOM RescaleSlope tag.
        rescale_intercept: DICOM RescaleIntercept tag.
        window_center: Target window level (default: -600 for lung window).
        window_width: Target window width (default: 1500 for lung window).

    Returns:
        uint8 numpy array of shape [H, W] in range [0, 255].
    """
    # 1. Transform raw detector counts to Hounsfield Units (HU)
    hu = pixel_array.astype(np.float32) * float(rescale_slope) + float(rescale_intercept)

    # 2. Apply clinical CT windowing
    lower = window_center - (window_width / 2.0)
    upper = window_center + (window_width / 2.0)
    windowed = np.clip(hu, lower, upper)

    norm_range = upper - lower
    if norm_range > 0:
        scaled = ((windowed - lower) / norm_range) * 255.0
    else:
        scaled = np.zeros_like(windowed)

    return cast(np.ndarray, np.round(scaled).astype(np.uint8))
