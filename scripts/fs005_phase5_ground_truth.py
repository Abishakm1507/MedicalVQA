"""Ground-truth region overlays for the FS-005 Phase 5 oracle pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def build_ground_truth_overlay(
    image: Image.Image, detection_path: str | Path
) -> tuple[Image.Image, list[str], list[dict[str, Any]]]:
    """Draw validated SLAKE XYWH annotations and return their R-number mapping."""
    path = Path(detection_path)
    try:
        detections = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground-truth annotations not found: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read ground-truth annotations at {path}: {exc}") from exc

    if not isinstance(detections, list):
        raise ValueError(f"Ground-truth annotations in {path} must be a JSON array")

    marked_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked_image)
    region_ids: list[str] = []
    regions: list[dict[str, Any]] = []

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict) or len(detection) != 1:
            raise ValueError(
                f"Annotation {index} in {path} must map exactly one region label to a box"
            )
        label, box = next(iter(detection.items()))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Annotation {index} in {path} has an invalid region label")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(
                f"Annotation {index} ({label!r}) in {path} must contain [x, y, width, height]"
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError(
                f"Annotation {index} ({label!r}) in {path} has non-numeric box coordinates"
            )

        x, y, width, height = (float(value) for value in box)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"Annotation {index} ({label!r}) in {path} has non-finite coordinates")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Annotation {index} ({label!r}) in {path} must have positive width and height"
            )

        x2 = x + width
        y2 = y + height
        if x < 0 or y < 0 or x2 > marked_image.width or y2 > marked_image.height:
            raise ValueError(
                f"Annotation {index} ({label!r}) box {(x, y, x2, y2)} exceeds "
                f"image bounds {(marked_image.width, marked_image.height)}"
            )

        region_id = f"R{index + 1}"
        draw.rectangle((x, y, x2, y2), outline=(255, 0, 0), width=2)
        text_bounds = draw.textbbox((0, 0), region_id)
        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]
        label_x = min(max(0, int(x)), max(0, marked_image.width - text_width - 4))
        label_y = int(y) - text_height - 4
        if label_y < 0:
            label_y = min(int(y) + 1, max(0, marked_image.height - text_height - 4))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 4, label_y + text_height + 4),
            fill=(255, 255, 0),
        )
        draw.text((label_x + 2, label_y + 2), region_id, fill=(0, 0, 0))

        region_ids.append(region_id)
        regions.append(
            {
                "region_id": region_id,
                "label": label,
                "box_xyxy": (x, y, x2, y2),
            }
        )

    return marked_image, region_ids, regions
