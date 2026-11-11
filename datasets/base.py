"""Base dataset loader and coordinate normalization utilities.

Provides common abstractions, coordinate normalization, bounding-box validation,
and patient-level split leakage verification.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from pathlib import Path

from PIL import Image

from core.schemas import MedicalVQASample

logger = logging.getLogger(__name__)


def normalize_bounding_box(
    box: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
    input_format: str = "xywh",
) -> tuple[float, float, float, float] | None:
    """Normalizes bounding box coordinates into [ymin, xmin, ymax, xmax] in [0.0, 1.0].

    Args:
        box: Coordinates as 4 floats.
        image_width: Width of image in pixels.
        image_height: Height of image in pixels.
        input_format: 'xywh' ([xmin, ymin, width, height]) or 'xyxy' ([xmin, ymin, xmax, ymax]).

    Returns:
        Tuple of (ymin, xmin, ymax, xmax) normalized to [0.0, 1.0], or None if box is invalid.
    """
    if len(box) != 4 or image_width <= 0 or image_height <= 0:
        return None

    c1, c2, c3, c4 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if input_format == "xywh":
        xmin, ymin = c1, c2
        xmax, ymax = c1 + c3, c2 + c4
    elif input_format == "xyxy":
        xmin, ymin, xmax, ymax = c1, c2, c3, c4
    else:
        raise ValueError(f"Unsupported box format '{input_format}'. Use 'xywh' or 'xyxy'.")

    # Clip to boundary with float epsilon tolerance
    ymin_norm = max(0.0, min(1.0, ymin / image_height))
    xmin_norm = max(0.0, min(1.0, xmin / image_width))
    ymax_norm = max(0.0, min(1.0, ymax / image_height))
    xmax_norm = max(0.0, min(1.0, xmax / image_width))

    # Reject degenerate (zero-area or inverted) bounding boxes
    if ymax_norm <= ymin_norm or xmax_norm <= xmin_norm:
        return None

    return (ymin_norm, xmin_norm, ymax_norm, xmax_norm)


def check_patient_leakage(
    train_samples: Iterable[MedicalVQASample],
    test_samples: Iterable[MedicalVQASample],
    val_samples: Iterable[MedicalVQASample] | None = None,
) -> tuple[bool, set[str]]:
    """Verifies that patient identifiers are strictly isolated across splits.

    Returns:
        (is_leakage_detected: bool, leaked_patient_ids: Set[str])
    """
    train_pids = {s.patient_id for s in train_samples if s.patient_id}
    test_pids = {s.patient_id for s in test_samples if s.patient_id}
    val_pids = {s.patient_id for s in val_samples if s.patient_id} if val_samples else set()

    leaked = (train_pids & test_pids) | (train_pids & val_pids) | (val_pids & test_pids)
    return (len(leaked) > 0, leaked)


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
                    return img.size  # (width, height)
            except Exception as e:
                logger.warning("Could not read image dimensions from %s: %s", image_path, e)
        elif self.strict_image_exists:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        return (512, 512)
