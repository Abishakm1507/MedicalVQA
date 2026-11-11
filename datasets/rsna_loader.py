"""RSNA Pneumonia Detection Challenge dataset loader for MedicalVQA.

Reference: Section III-B, Section V-A, Table 1 of research paper.
Parses chest radiographs with radiologist bounding boxes for lung opacity.
Accurately preserves official patientId as patient_id.
"""

import csv
import json
import logging
from collections.abc import Generator
from pathlib import Path

from core.schemas import EvaluationTrack, GroundTruthRegion, MedicalVQASample
from datasets.base import BaseDatasetLoader, normalize_bounding_box

logger = logging.getLogger(__name__)


class RSNADatasetLoader(BaseDatasetLoader):
    """Loader for RSNA Pneumonia Detection Challenge dataset."""

    def __init__(
        self,
        dataset_dir: str | Path,
        images_dir: str | Path | None = None,
        labels_file: str | Path | None = None,
        strict_image_exists: bool = False,
    ) -> None:
        super().__init__(
            dataset_dir,
            images_dir or Path(dataset_dir) / "stage_2_train_images",
            strict_image_exists,
        )
        self.labels_file = (
            Path(labels_file) if labels_file else self.dataset_dir / "stage_2_train_labels.csv"
        )

    def stream_split(self, split: str = "test") -> Generator[MedicalVQASample, None, None]:
        """Streams samples from RSNA dataset.

        Args:
            split: 'train', 'validate', or 'test'.
        """
        # If pre-partitioned split JSON exists, load it
        split_json = self.dataset_dir / f"rsna_{split}.json"
        if split_json.exists():
            with open(split_json, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                yield self._entry_to_sample(entry, split)
            return

        # Otherwise parse labels CSV
        if not self.labels_file.exists():
            raise FileNotFoundError(f"RSNA labels file not found: {self.labels_file}")

        patient_groups = self._parse_labels_csv()
        for patient_id, group in patient_groups.items():
            yield self._group_to_sample(patient_id, group, split)

    def _parse_labels_csv(self) -> dict[str, dict]:
        """Parses CSV and groups multiple bounding boxes by patientId."""
        patients: dict[str, dict] = {}
        with open(self.labels_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["patientId"]
                raw_target = row.get("Target")
                target = (
                    int(raw_target)
                    if raw_target is not None and str(raw_target).strip() != ""
                    else 0
                )

                if pid not in patients:
                    patients[pid] = {
                        "boxes": [],
                        "target": target,
                    }

                if target == 1 and row.get("x") and row.get("y"):
                    try:
                        x = float(row["x"])
                        y = float(row["y"])
                        w = float(row["width"])
                        h = float(row["height"])
                        patients[pid]["boxes"].append([x, y, w, h])
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Error parsing bounding box coordinates for patient %s: %s", pid, e
                        )

        return patients

    def _group_to_sample(self, patient_id: str, group: dict, split: str) -> MedicalVQASample:
        """Converts an RSNA patient record to a standardized MedicalVQASample."""
        img_path = self.images_dir / f"{patient_id}.png"
        if not img_path.exists():
            img_path = self.images_dir / f"{patient_id}.dcm"

        img_width, img_height = self._get_image_dimensions(img_path)
        gt_regions: list[GroundTruthRegion] = []

        for box in group["boxes"]:
            norm_box = normalize_bounding_box(
                box,
                img_width,
                img_height,
                input_format="xywh",
                clip_out_of_bounds=False,
            )
            if norm_box is not None:
                gt_regions.append(
                    GroundTruthRegion(
                        label="lung opacity",
                        box_2d=norm_box,
                    )
                )
            else:
                logger.warning("Rejected invalid box %s for RSNA patient %s", box, patient_id)

        has_opacity = len(gt_regions) > 0
        answer = "yes" if has_opacity else "no"

        # In RSNA, patientId is an official clinical patient identifier
        return MedicalVQASample(
            sample_id=f"rsna_{patient_id}",
            dataset_name="rsna",
            split=split,
            image_id=patient_id,
            patient_id=patient_id,  # Valid clinical patient identity
            image_path=str(img_path),
            modality="CXR",
            organ="chest",
            question="Is there lung opacity indicating pneumonia in this chest radiograph?",
            answer=answer,
            question_type="closed",
            ground_truth_regions=gt_regions,
            has_ground_truth_regions=True,
            compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
            metadata={
                "patient_id": patient_id,
                "target": group["target"],
                "box_count": len(gt_regions),
            },
        )

    def _entry_to_sample(self, entry: dict, split: str) -> MedicalVQASample:
        """Converts a pre-split JSON entry into a MedicalVQASample."""
        pid = entry.get("patient_id", "unknown")
        img_path = self.images_dir / f"{pid}.png"
        img_width, img_height = self._get_image_dimensions(img_path)

        gt_regions = []
        for box in entry.get("boxes", []):
            norm_box = normalize_bounding_box(
                box,
                img_width,
                img_height,
                input_format="xywh",
                clip_out_of_bounds=False,
            )
            if norm_box:
                gt_regions.append(GroundTruthRegion(label="lung opacity", box_2d=norm_box))

        return MedicalVQASample(
            sample_id=f"rsna_{split}_{pid}",
            dataset_name="rsna",
            split=split,
            image_id=pid,
            patient_id=pid,
            image_path=str(img_path),
            modality="CXR",
            organ="chest",
            question=entry.get(
                "question", "Is there lung opacity indicating pneumonia in this chest radiograph?"
            ),
            answer=entry.get("answer", "yes" if gt_regions else "no"),
            question_type="closed",
            ground_truth_regions=gt_regions,
            has_ground_truth_regions=True,
            compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
            metadata=entry.get("metadata", {}),
        )
