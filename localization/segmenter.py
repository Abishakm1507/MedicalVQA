"""Region segmentation interface and deterministic mock implementation.

This module defines:
  - SegmenterConfig       — configuration dataclass for segmenters.
  - SegmentationResult    — result container carrying region_id, mask, and mask_rle.
  - BaseRegionSegmenter   — abstract base class all segmenters must implement.
  - MockRegionSegmenter   — deterministic mock for unit testing and CI/CD.

Coordinate Convention
---------------------
All bounding-box coordinates follow the FS-002 / CandidateBox convention:
    box_2d = (ymin, xmin, ymax, xmax)  — absolute pixel integers, origin = top-left.

This convention is preserved verbatim throughout encoding, testing, and downstream
integration with FS-004 (visual overlay) and FS-007 (claim verification).

Separation of Concerns
----------------------
MockRegionSegmenter is intentionally isolated from MedSAM.
  - MockRegionSegmenter MUST NEVER be used as a silent fallback for a real model.
  - Callers must explicitly instantiate MockRegionSegmenter to obtain mock output.
  - MedSAMRegionSegmenter (Stage 4) will be a separate class in this module.

Traceability
------------
  - Feature Specification : FS-003 (MedSAM Segmentation Integration)
  - Architecture Decision  : ADR 0003 (Detector Selection and Licensing Review)
  - Research Paper         : Section III-B (Region Proposal)
  - Schemas                : core.schemas.CandidateBox, GroundTruthRegion.mask_rle
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.schemas import CandidateBox
from localization.rle import rle_encode

logger = logging.getLogger(__name__)

# Expected MedSAM checkpoint metadata
MEDSAM_CHECKPOINT_FILENAME = "medsam_vit_b.pth"
MEDSAM_CHECKPOINT_EXPECTED_SIZE = 375049145
MEDSAM_CHECKPOINT_EXPECTED_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"
MEDSAM_ZENODO_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmenterConfig:
    """Configuration for region segmenters.

    Attributes:
        device: Computation device string ('cpu', 'cuda', 'cuda:0', …).
            Only relevant for real model segmenters; ignored by the mock.
        checkpoint_path: Path to the MedSAM checkpoint file
            (medsam_vit_b.pth).  Must be provided for MedSAMRegionSegmenter.
            Unused by MockRegionSegmenter.
        image_size: Target image size fed into the SAM image encoder.
            Defaults to 1024 as used by the official MedSAM implementation.
    """

    device: str = "cpu"
    checkpoint_path: str | Path | None = None
    image_size: int = 1024

    def __post_init__(self) -> None:
        valid_cuda = (
            self.device == "cuda"
            or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        )
        if self.device != "cpu" and not valid_cuda:
            raise ValueError(
                f"Unsupported device '{self.device}'. Must be 'cpu', 'cuda', or 'cuda:N' where N is a non-negative integer."
            )
        if self.image_size <= 0:
            raise ValueError(
                f"image_size must be a positive integer, got {self.image_size}."
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Carries the segmentation output for a single candidate region.

    Attributes:
        region_id: Preserves the identifier from the originating CandidateBox
            (e.g. 'R1', 'R2').
        mask: 2-D boolean NumPy array of shape (H, W) where H and W are the
            height and width of the *source image* (not the bounding box).
            True pixels indicate the segmented foreground region.
        mask_rle: Run-Length Encoded representation of *mask*, serialized
            with ``localization.rle.rle_encode``.  Format: '{H},{W}:{counts}'.
    """

    region_id: str
    mask: np.ndarray          # shape (H, W), dtype bool
    mask_rle: str             # '{H},{W}:{counts}'

    def __post_init__(self) -> None:
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(
                f"mask must be a numpy.ndarray, got {type(self.mask).__name__}"
            )
        if self.mask.ndim != 2:
            raise ValueError(
                f"mask must be 2-dimensional, got shape {self.mask.shape}"
            )
        if self.mask.dtype != np.bool_:
            raise ValueError(
                f"mask dtype must be np.bool_, got {self.mask.dtype}"
            )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseRegionSegmenter(ABC):
    """Abstract base class for all region segmenters.

    Subclasses must implement ``segment_boxes``.

    Design constraints (enforced by all concrete implementations):
      1. ``segment_boxes`` MUST return one ``SegmentationResult`` per input
         ``CandidateBox``, in the same order.
      2. Each result's ``mask`` MUST have shape ``(image.height, image.width)``
         and dtype ``np.bool_``.
      3. ``region_id`` in each result MUST equal the corresponding
         ``CandidateBox.region_id``.
      4. An empty ``boxes`` list MUST return an empty list immediately.
      5. A missing or unloadable real-model checkpoint MUST raise
         ``FileNotFoundError`` — never silently fall back to mock output.
    """

    @abstractmethod
    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[CandidateBox],
    ) -> list[SegmentationResult]:
        """Generate binary segmentation masks for each candidate region.

        Args:
            image: RGB PIL Image representing the medical image.
            boxes: Candidate bounding boxes from the FS-002 detector.
                Each ``CandidateBox.box_2d`` is ``(ymin, xmin, ymax, xmax)``
                in absolute pixel coordinates.

        Returns:
            List of ``SegmentationResult`` objects, one per input box, in
            the same order.  Returns ``[]`` when ``boxes`` is empty.

        Raises:
            TypeError: If *image* is not a PIL Image or *boxes* is not a list.
            ValueError: If *image* has zero or negative dimensions.
        """


# ---------------------------------------------------------------------------
# Deterministic mock segmenter
# ---------------------------------------------------------------------------


class MockRegionSegmenter(BaseRegionSegmenter):
    """Deterministic mock segmenter for unit testing and CI/CD pipelines.

    CRITICAL NOTICE — RESEARCH INTEGRITY
    -------------------------------------
    This class exists solely for development, testing, and TDD pipelines.

    It MUST NEVER be:
      - used in production inference or clinical evaluation;
      - silently substituted for a real MedSAM prediction;
      - presented as actual MedSAM segmentation output;
      - used to generate or report calibration metrics.

    Callers must explicitly instantiate ``MockRegionSegmenter`` to obtain
    mock output.  ``MedSAMRegionSegmenter`` will NEVER fall back to this
    implementation when its checkpoint is unavailable — it will raise
    ``FileNotFoundError`` instead.

    Mask Generation Strategy
    ------------------------
    For each ``CandidateBox``, the mock fills *only* the pixels that lie
    strictly inside the bounding box with ``True``.  Coordinates are clamped
    to ``[0, image_width)`` and ``[0, image_height)`` so that boxes whose
    corners touch or exceed the image boundary are handled safely without
    raising exceptions.

    The mock is *deterministic*: the same ``(image_size, boxes)`` combination
    always produces the same masks.  It carries no internal random state.

    Parameters
    ----------
    None — the mock requires no configuration.
    """

    # Public marker so callers can distinguish mock output programmatically
    IS_MOCK: bool = True

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[CandidateBox],
    ) -> list[SegmentationResult]:
        """Generate deterministic rectangular masks for each candidate box.

        Each mask is a boolean array of shape (H, W) — the same as the source
        image — with ``True`` inside the clamped bounding box and ``False``
        everywhere else.

        Args:
            image: RGB PIL Image.  Mode is not enforced; only ``image.size``
                is used.
            boxes: List of ``CandidateBox`` instances.  Empty list returns [].

        Returns:
            List of ``SegmentationResult``, one per input box, in order.

        Raises:
            TypeError: If *image* is not a PIL Image or *boxes* is not a list.
            ValueError: If *image* has zero or negative dimensions.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"image must be a PIL.Image.Image, got {type(image).__name__}"
            )
        if image.mode != "RGB":
            raise ValueError(
                f"image must be in 'RGB' mode per interface contract, got mode '{image.mode}'"
            )
        if not isinstance(boxes, list):
            raise TypeError(
                f"boxes must be a list, got {type(boxes).__name__}"
            )

        img_w, img_h = image.size  # PIL: (width, height)
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"Image must have positive dimensions, got width={img_w}, height={img_h}"
            )

        if not boxes:
            return []

        results: list[SegmentationResult] = []

        for box in boxes:
            if not isinstance(box, CandidateBox):
                raise TypeError(
                    f"Each element of boxes must be a CandidateBox, "
                    f"got {type(box).__name__}"
                )

            if not (isinstance(box.box_2d, (tuple, list)) and len(box.box_2d) == 4):
                raise ValueError(
                    f"CandidateBox {box.region_id} must have box_2d as a 4-element tuple/list, "
                    f"got {box.box_2d}"
                )

            ymin_raw, xmin_raw, ymax_raw, xmax_raw = box.box_2d

            for c_name, c_val in (
                ("ymin", ymin_raw),
                ("xmin", xmin_raw),
                ("ymax", ymax_raw),
                ("xmax", xmax_raw),
            ):
                if not isinstance(c_val, (int, float, np.integer, np.floating)) or not np.isfinite(c_val):
                    raise ValueError(
                        f"CandidateBox {box.region_id} coordinate '{c_name}' must be finite numeric, "
                        f"got {c_val}"
                    )

            # Compute intersection between half-open candidate box [ymin, ymax) x [xmin, xmax)
            # and image canvas [0, img_h) x [0, img_w).
            ymin_clip = max(0, int(ymin_raw))
            ymax_clip = min(img_h, int(ymax_raw))
            xmin_clip = max(0, int(xmin_raw))
            xmax_clip = min(img_w, int(xmax_raw))

            # Build boolean mask of source image dimensions (img_h, img_w)
            mask = np.zeros((img_h, img_w), dtype=bool)

            # Intersection is non-empty iff clamped min < clamped max in both dimensions
            if ymin_clip < ymax_clip and xmin_clip < xmax_clip:
                mask[ymin_clip:ymax_clip, xmin_clip:xmax_clip] = True

            mask_rle = rle_encode(mask)

            results.append(
                SegmentationResult(
                    region_id=box.region_id,
                    mask=mask,
                    mask_rle=mask_rle,
                )
            )

        logger.debug(
            "MockRegionSegmenter produced %d mask(s) for image size %dx%d. "
            "These are MOCK outputs — not MedSAM predictions.",
            len(results),
            img_w,
            img_h,
        )
        return results


# ---------------------------------------------------------------------------
# MedSAM segmenter (Stage 4)
# ---------------------------------------------------------------------------


class MedSAMRegionSegmenter(BaseRegionSegmenter):
    """MedSAM ViT-B region segmenter using the official SamPredictor.

    Wraps the MedSAM ViT-B architecture (``sam_model_registry["vit_b"]``)
    and the official ``SamPredictor`` from the MedSAM-vendored
    ``segment_anything`` package.

    Coordinate Convention
    ---------------------
    Input boxes follow the project convention ``(ymin, xmin, ymax, xmax)``.
    They are converted to SAM's ``(xmin, ymin, xmax, ymax)`` pixel-coordinate
    convention before being passed to ``SamPredictor``.

    Design constraints:
      1. Requires an explicit, validated checkpoint path.
      2. Never silently falls back to MockRegionSegmenter.
      3. Processes regions sequentially (one predict call per box).
      4. Uses ``torch.inference_mode()`` for inference.
      5. Returns boolean masks with original image dimensions.
      6. RLE encodes masks with ``localization.rle.rle_encode``.
      7. Preserves region IDs and input ordering.
      8. Explicit device handling — raises if CUDA requested but unavailable.
    """

    IS_MOCK: bool = False

    def __init__(self, config: SegmenterConfig) -> None:
        """Initialize the MedSAM segmenter.

        Args:
            config: SegmenterConfig with device, checkpoint_path, and image_size.

        Raises:
            TypeError: If config is not a SegmenterConfig.
            ValueError: If checkpoint_path is None or empty.
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If checkpoint size or MD5 checksum does not match.
            RuntimeError: If checkpoint cannot be safely loaded (weights_only=True)
                or if CUDA is requested but unavailable.
            ImportError: If torch or segment_anything is not installed.
        """
        if not isinstance(config, SegmenterConfig):
            raise TypeError(
                f"config must be a SegmenterConfig, got {type(config).__name__}"
            )

        self.config = config

        if config.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for MedSAMRegionSegmenter. "
                "Provide a path to medsam_vit_b.pth via SegmenterConfig."
            )

        self.checkpoint_path = Path(config.checkpoint_path)

        # Validate checkpoint before loading the model
        self._validate_checkpoint()

        # Lazy imports — torch and segment_anything are heavyweight
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "MedSAMRegionSegmenter requires torch and segment_anything. "
                "Install the ml optional dependencies and the MedSAM-vendored "
                "segment_anything package. See pyproject.toml for details."
            ) from exc

        # Verify device availability explicitly — do not silently fall back
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{config.device}' but CUDA is not available on this system. "
                "Use device='cpu' for local execution or run on a GPU-enabled remote instance."
            )

        self.device = torch.device(config.device)
        self._torch = torch

        # Build MedSAM ViT-B model and load checkpoint
        self._sam = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
        self._sam.to(self.device)
        self._sam.eval()

        self._predictor = SamPredictor(self._sam)

        logger.info(
            "MedSAMRegionSegmenter initialized on device '%s' with checkpoint '%s'",
            self.device,
            self.checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Checkpoint validation
    # ------------------------------------------------------------------

    def _validate_checkpoint(self) -> None:
        """Validate checkpoint file existence, size, MD5, and safe loadability."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MedSAM checkpoint not found at: {self.checkpoint_path}"
            )

        actual_size = self.checkpoint_path.stat().st_size
        if actual_size != MEDSAM_CHECKPOINT_EXPECTED_SIZE:
            raise ValueError(
                f"MedSAM checkpoint size mismatch: expected "
                f"{MEDSAM_CHECKPOINT_EXPECTED_SIZE} bytes (~375.0 MB), "
                f"got {actual_size} bytes (~{actual_size / 1024 / 1024:.1f} MB). "
                "The download may be incomplete or corrupted."
            )

        actual_md5 = self._compute_md5()
        if actual_md5 != MEDSAM_CHECKPOINT_EXPECTED_MD5:
            raise ValueError(
                f"MedSAM checkpoint MD5 mismatch: expected "
                f"{MEDSAM_CHECKPOINT_EXPECTED_MD5}, got {actual_md5}. "
                "The checkpoint file may be corrupted."
            )

        # Validate that the checkpoint can be safely loaded with weights_only=True
        try:
            import torch
            with open(self.checkpoint_path, "rb") as f:
                torch.load(f, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"MedSAM checkpoint at {self.checkpoint_path} cannot be safely "
                f"loaded with weights_only=True: {exc}"
            ) from exc

    def _compute_md5(self) -> str:
        """Compute MD5 checksum of the checkpoint file."""
        import hashlib

        md5 = hashlib.md5()
        with open(self.checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    # ------------------------------------------------------------------
    # BaseRegionSegmenter interface
    # ------------------------------------------------------------------

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[CandidateBox],
    ) -> list[SegmentationResult]:
        """Generate binary segmentation masks for each candidate region.

        Args:
            image: RGB PIL Image representing the medical image.
            boxes: Candidate bounding boxes from the FS-002 detector.
                Each ``CandidateBox.box_2d`` is ``(ymin, xmin, ymax, xmax)``
                in absolute pixel coordinates.

        Returns:
            List of ``SegmentationResult`` objects, one per input box, in
            the same order.  Returns ``[]`` when ``boxes`` is empty.

        Raises:
            TypeError: If *image* is not a PIL Image or *boxes* is not a list.
            ValueError: If *image* has zero or negative dimensions, or if
                any box has invalid coordinates.
            RuntimeError: If inference fails for any region.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"image must be a PIL.Image.Image, got {type(image).__name__}"
            )
        if image.mode != "RGB":
            raise ValueError(
                f"image must be in 'RGB' mode per interface contract, got mode '{image.mode}'"
            )
        if not isinstance(boxes, list):
            raise TypeError(
                f"boxes must be a list, got {type(boxes).__name__}"
            )

        img_w, img_h = image.size  # PIL: (width, height)
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"Image must have positive dimensions, got width={img_w}, height={img_h}"
            )

        if not boxes:
            return []

        # Lazy imports
        import numpy as np

        image_np = np.array(image)  # HWC uint8 RGB

        results: list[SegmentationResult] = []

        with self._torch.inference_mode():
            # Set image once — computes the image embedding for all boxes
            self._predictor.set_image(image_np, image_format="RGB")

            for box in boxes:
                if not isinstance(box, CandidateBox):
                    raise TypeError(
                        f"Each element of boxes must be a CandidateBox, "
                        f"got {type(box).__name__}"
                    )

                if not (isinstance(box.box_2d, (tuple, list)) and len(box.box_2d) == 4):
                    raise ValueError(
                        f"CandidateBox {box.region_id} must have box_2d as a 4-element tuple/list, "
                        f"got {box.box_2d}"
                    )

                ymin_raw, xmin_raw, ymax_raw, xmax_raw = box.box_2d

                for c_name, c_val in (
                    ("ymin", ymin_raw),
                    ("xmin", xmin_raw),
                    ("ymax", ymax_raw),
                    ("xmax", xmax_raw),
                ):
                    if not isinstance(c_val, (int, float, np.integer, np.floating)) or not np.isfinite(c_val):
                        raise ValueError(
                            f"CandidateBox {box.region_id} coordinate '{c_name}' must be finite numeric, "
                            f"got {c_val}"
                        )

                # Convert project convention (ymin, xmin, ymax, xmax)
                # to SAM convention (xmin, ymin, xmax, ymax)
                sam_box = np.array(
                    [xmin_raw, ymin_raw, xmax_raw, ymax_raw], dtype=float
                )

                try:
                    masks, _, _ = self._predictor.predict(
                        box=sam_box,
                        multimask_output=False,
                        return_logits=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"MedSAM inference failed for region {box.region_id}: {exc}"
                    ) from exc

                # masks shape: (1, H, W) boolean — take first element
                mask = masks[0].astype(bool, copy=False)

                if mask.shape != (img_h, img_w):
                    raise RuntimeError(
                        f"MedSAM returned mask with shape {mask.shape}, expected ({img_h}, {img_w})"
                    )

                mask_rle = rle_encode(mask)

                results.append(
                    SegmentationResult(
                        region_id=box.region_id,
                        mask=mask,
                        mask_rle=mask_rle,
                    )
                )

        return results
