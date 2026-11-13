"""Base dataset loader and coordinate normalization utilities.

Provides common abstractions, coordinate normalization, bounding-box validation,
round-trip transformations, and patient-level split leakage verification.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from PIL import Image

from core.schemas import MedicalVQASample

logger = logging.getLogger(__name__)


def normalize_bounding_box(
    box: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
    input_format: Literal["xywh", "xyxy"] = "xywh",
    tolerance_pixels: float = 1.0,
    clip_out_of_bounds: bool = False,
) -> tuple[float, float, float, float] | None:
    """Normalizes bounding box coordinates into [ymin, xmin, ymax, xmax] in [0.0, 1.0].

    Strict validation: Does NOT silently shift or distort bounding boxes. Rejects degenerate
    or out-of-bounds annotations unless within small tolerance (e.g. 1 pixel rounding).

    Args:
        box: Coordinates as 4 floats.
        image_width: Width of image in pixels (must be > 0).
        image_height: Height of image in pixels (must be > 0).
        input_format: 'xywh' ([xmin, ymin, width, height]) or 'xyxy' ([xmin, ymin, xmax, ymax]).
        tolerance_pixels: Maximum allowable margin (in pixels) for coordinates slightly outside bounds.
        clip_out_of_bounds: If False, rejects out-of-bounds boxes. If True, clips with warning.

    Returns:
        Tuple of (ymin, xmin, ymax, xmax) normalized to [0.0, 1.0], or None if box is invalid.
    """
    if len(box) != 4 or image_width <= 0 or image_height <= 0:
        logger.warning(
            "Invalid normalization inputs: len(box)=%d, width=%d, height=%d",
            len(box),
            image_width,
            image_height,
        )
        return None

    c1, c2, c3, c4 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if input_format == "xywh":
        xmin, ymin = c1, c2
        xmax, ymax = c1 + c3, c2 + c4
    elif input_format == "xyxy":
        xmin, ymin = c1, c2
        xmax, ymax = c3, c4
    else:
        raise ValueError(f"Unsupported box format '{input_format}'. Use 'xywh' or 'xyxy'.")

    # Degenerate box validation: width and height must be strictly positive
    if xmax <= xmin or ymax <= ymin:
        logger.warning(
            "Rejected degenerate box: xmin=%.2f, ymin=%.2f, xmax=%.2f, ymax=%.2f",
            xmin,
            ymin,
            xmax,
            ymax,
        )
        return None

    # Check for out-of-bounds coordinates
    is_out_of_bounds = (
        xmin < -tolerance_pixels
        or ymin < -tolerance_pixels
        or xmax > (image_width + tolerance_pixels)
        or ymax > (image_height + tolerance_pixels)
    )

    if is_out_of_bounds:
        if not clip_out_of_bounds:
            logger.warning(
                "Rejected out-of-bounds box: [%.1f, %.1f, %.1f, %.1f] on image [%d x %d]",
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
            )
            return None
        logger.warning(
            "Clipping out-of-bounds box [%.1f, %.1f, %.1f, %.1f] to image bounds [%d x %d]",
            xmin,
            ymin,
            xmax,
            ymax,
            image_width,
            image_height,
        )

    # Clamping within bounds
    xmin_clamped = max(0.0, min(float(image_width), xmin))
    ymin_clamped = max(0.0, min(float(image_height), ymin))
    xmax_clamped = max(0.0, min(float(image_width), xmax))
    ymax_clamped = max(0.0, min(float(image_height), ymax))

    # Re-verify that clamping did not collapse the box
    if xmax_clamped <= xmin_clamped or ymax_clamped <= ymin_clamped:
        logger.warning("Box collapsed to zero area after boundary clamping.")
        return None

    # Compute normalized coordinates [ymin, xmin, ymax, xmax]
    ymin_norm = ymin_clamped / image_height
    xmin_norm = xmin_clamped / image_width
    ymax_norm = ymax_clamped / image_height
    xmax_norm = xmax_clamped / image_width

    return (ymin_norm, xmin_norm, ymax_norm, xmax_norm)


def denormalize_bounding_box(
    norm_box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    output_format: Literal["xywh", "xyxy"] = "xywh",
) -> tuple[float, float, float, float]:
    """Converts normalized [ymin, xmin, ymax, xmax] coordinates back to absolute pixel coordinates.

    Args:
        norm_box: Normalized coordinates in [0.0, 1.0].
        image_width: Target image width in pixels.
        image_height: Target image height in pixels.
        output_format: 'xywh' or 'xyxy'.

    Returns:
        Absolute pixel coordinates as 4 floats.
    """
    ymin_norm, xmin_norm, ymax_norm, xmax_norm = norm_box
    xmin = xmin_norm * image_width
    ymin = ymin_norm * image_height
    xmax = xmax_norm * image_width
    ymax = ymax_norm * image_height

    if output_format == "xywh":
        return (xmin, ymin, xmax - xmin, ymax - ymin)
    elif output_format == "xyxy":
        return (xmin, ymin, xmax, ymax)
    raise ValueError(f"Unsupported output_format '{output_format}'. Use 'xywh' or 'xyxy'.")


@dataclass
class PatientIsolationReport:
    """Detailed audit report for patient-level split isolation."""

    patient_ids_available: bool
    is_leakage_detected: bool
    leaked_patient_ids: set[str] = field(default_factory=set)
    total_patients_train: int = 0
    total_patients_test: int = 0
    total_patients_val: int = 0


def check_patient_leakage(
    train_samples: Iterable[MedicalVQASample],
    test_samples: Iterable[MedicalVQASample],
    val_samples: Iterable[MedicalVQASample] | None = None,
) -> PatientIsolationReport:
    """Verifies that patient identifiers are strictly isolated across splits.

    If samples do NOT have patient_id populated (patient_id is None), explicitly reports
    that patient identity is unavailable rather than assuming image IDs are patient IDs.
    """
    train_list = list(train_samples)
    test_list = list(test_samples)
    val_list = list(val_samples) if val_samples else []

    train_pids = {s.patient_id for s in train_list if s.patient_id is not None}
    test_pids = {s.patient_id for s in test_list if s.patient_id is not None}
    val_pids = {s.patient_id for s in val_list if s.patient_id is not None}

    # If no patient IDs exist in any split, patient identity is unavailable
    if not train_pids and not test_pids and not val_pids:
        logger.info(
            "Patient identifiers not provided in dataset; patient isolation cannot be verified."
        )
        return PatientIsolationReport(
            patient_ids_available=False,
            is_leakage_detected=False,
            leaked_patient_ids=set(),
            total_patients_train=0,
            total_patients_test=0,
            total_patients_val=0,
        )

    leaked = (train_pids & test_pids) | (train_pids & val_pids) | (val_pids & test_pids)
    return PatientIsolationReport(
        patient_ids_available=True,
        is_leakage_detected=len(leaked) > 0,
        leaked_patient_ids=leaked,
        total_patients_train=len(train_pids),
        total_patients_test=len(test_pids),
        total_patients_val=len(val_pids),
    )


class BaseDatasetLoader(ABC):
    """Abstract base class for Medical VQA dataset loaders."""

    def __init__(
        self,
        dataset_dir: str | Path,
        images_dir: str | Path | None = None,
        strict_image_exists: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = Path(images_dir) if images_dir else self.dataset_dir / "images"
        self.strict_image_exists = strict_image_exists

    @abstractmethod
    def stream_split(self, split: str) -> Generator[MedicalVQASample, None, None]:
        """Streams standardized MedicalVQASample objects for a given split."""
        pass

    def load_split(self, split: str) -> list[MedicalVQASample]:
        """Loads all samples for a given split into memory as a list."""
        return list(self.stream_split(split))

    def _get_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Retrieves image dimensions (width, height), falling back to (512, 512) if unavailable."""
        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    return cast(tuple[int, int], img.size)
            except Exception as e:
                logger.warning("Could not read image dimensions from %s: %s", image_path, e)
        elif self.strict_image_exists:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        return (512, 512)
