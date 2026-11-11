"""SLAKE dataset loader for MedicalVQA.

Reference: Section V-A, Table 1 of research paper.
Filters English questions, parses masks and bounding boxes, and assigns evaluation tracks.
"""

import json
from collections.abc import Generator
from pathlib import Path

from core.schemas import EvaluationTrack, GroundTruthRegion, MedicalVQASample
from datasets.base import BaseDatasetLoader, normalize_bounding_box


class SlakeDatasetLoader(BaseDatasetLoader):
    """Loader for the SLAKE medical visual question answering benchmark."""

    def __init__(
        self,
        dataset_dir: str | Path,
        images_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        language: str = "en",
        strict_image_exists: bool = False,
    ) -> None:
        super().__init__(dataset_dir, images_dir or Path(dataset_dir) / "imgs", strict_image_exists)
        self.masks_dir = Path(masks_dir) if masks_dir else self.dataset_dir / "masks"
        self.language = language.lower()

    def stream_split(self, split: str = "test") -> Generator[MedicalVQASample, None, None]:
        """Streams standardized samples for a SLAKE split.

        Args:
            split: 'train', 'validate', or 'test'.
        """
        split_file = self._resolve_split_file(split)
        if not split_file.exists():
            raise FileNotFoundError(f"SLAKE split file not found: {split_file}")

        with open(split_file, encoding="utf-8") as f:
            raw_entries = json.load(f)

        for entry in raw_entries:
            # 1. Filter by language (Paper specifies English questions)
            q_lang = str(entry.get("q_lang", "")).lower()
            if self.language and q_lang != self.language:
                continue

            # 2. Extract anatomical location & modality
            organ = str(entry.get("location", "chest")).lower()
            modality = str(entry.get("modality", "CXR")).upper()

            # 3. Resolve image file
            img_name = str(entry.get("img_name", ""))
            img_path = self._resolve_image_path(img_name)
            img_width, img_height = self._get_image_dimensions(img_path)

            # 4. Parse bounding box annotations (SLAKE format: [x, y, w, h])
            gt_regions = self._parse_regions(entry, img_width, img_height)

            # 5. Question type classification
            raw_q_type = str(entry.get("q_type", "open")).strip().lower()
            q_type = "closed" if raw_q_type in ("closed", "yes/no", "choice") else "open"

            # 6. Evaluation track compatibility
            compatible_tracks = [
                EvaluationTrack.TRACK_B_ORACLE_SLAKE,
                EvaluationTrack.TRACK_C_ANSWER_ONLY,
            ]
            if "chest" in organ or modality in ("CXR", "X-RAY", "CHEST X-RAY"):
                compatible_tracks.append(EvaluationTrack.TRACK_A_PREDICTED_CXR)

            # 7. Patient identifier (SLAKE does not expose explicit patient IDs, use img_id as patient proxy)
            patient_id = f"slake_case_{entry.get('img_id', 'unknown')}"
            sample_id = f"slake_{split}_{entry.get('id', entry.get('img_id', '0'))}"

            yield MedicalVQASample(
                sample_id=sample_id,
                dataset_name="slake",
                split=split,
                patient_id=patient_id,
                image_path=str(img_path),
                modality=modality,
                organ=organ,
                question=str(entry.get("question", "")).strip(),
                answer=str(entry.get("answer", "")).strip(),
                question_type=q_type,
                ground_truth_regions=gt_regions,
                has_ground_truth_regions=True,
                compatible_tracks=compatible_tracks,
                metadata={
                    "original_id": entry.get("id"),
                    "img_id": entry.get("img_id"),
                    "location": organ,
                    "base_type": entry.get("base_type", ""),
                    "triple": entry.get("triple", []),
                },
            )

    def _resolve_split_file(self, split: str) -> Path:
        """Finds the split file handling common naming variations."""
        candidates = [
            self.dataset_dir / f"{split}.json",
            self.dataset_dir / f"{split.capitalize()}.json",
            self.dataset_dir / f"slake_{split}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _resolve_image_path(self, img_name: str) -> Path:
        """Resolves the image file path handling potential subdirectory prefixes."""
        direct = self.images_dir / img_name
        if direct.exists():
            return direct
        basename_match = self.images_dir / Path(img_name).name
        if basename_match.exists():
            return basename_match
        return direct

    def _parse_regions(
        self, entry: dict, img_width: int, img_height: int
    ) -> list[GroundTruthRegion]:
        """Extracts and normalizes ground truth bounding boxes."""
        regions: list[GroundTruthRegion] = []
        raw_boxes = entry.get("boxes", [])
        raw_labels = entry.get("box_labels", [])

        for idx, box in enumerate(raw_boxes):
            label = raw_labels[idx] if idx < len(raw_labels) else "lesion"
            norm_box = normalize_bounding_box(box, img_width, img_height, input_format="xywh")
            if norm_box is not None:
                regions.append(GroundTruthRegion(label=str(label), box_2d=norm_box))
        return regions
