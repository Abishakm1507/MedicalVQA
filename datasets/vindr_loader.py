"""VinDr-CXR dataset loader for MedicalVQA.

Reference: Section III-B, Section V-A, Table 1 of research paper.
Parses multi-radiologist thoracic bounding boxes (xyxy format) from annotations CSV.
Respects PhysioNet credentialed access and open subsplit distribution terms.
"""

import csv
import json
from collections.abc import Generator
from pathlib import Path

from core.schemas import EvaluationTrack, GroundTruthRegion, MedicalVQASample
from datasets.base import BaseDatasetLoader, normalize_bounding_box

# Critical thoracic pathologies identified in paper and VinDr-CXR
VINDR_PATHOLOGY_CLASSES: set[str] = {
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
}


class VinDrCXRDatasetLoader(BaseDatasetLoader):
    """Loader for the VinDr-CXR thoracic finding detection dataset."""

    def __init__(
        self,
        dataset_dir: str | Path,
        images_dir: str | Path | None = None,
        annotations_file: str | Path | None = None,
        target_pathology: str | None = None,
        strict_image_exists: bool = False,
    ) -> None:
        super().__init__(
            dataset_dir, images_dir or Path(dataset_dir) / "images", strict_image_exists
        )
        self.annotations_file = Path(annotations_file) if annotations_file else None
        self.target_pathology = target_pathology.lower() if target_pathology else None

    def stream_split(self, split: str = "test") -> Generator[MedicalVQASample, None, None]:
        """Streams standardized samples from VinDr-CXR.

        Args:
            split: 'train' or 'test'.
        """
        # If pre-partitioned JSON exists, load it
        split_json = self.dataset_dir / f"vindr_{split}.json"
        if split_json.exists():
            with open(split_json, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                yield self._entry_to_sample(entry, split)
            return

        # Otherwise parse CSV
        ann_csv = self._resolve_annotations_csv(split)
        if not ann_csv.exists():
            raise FileNotFoundError(f"VinDr-CXR annotations file not found: {ann_csv}")

        image_groups = self._parse_annotations_csv(ann_csv)
        for image_id, group in image_groups.items():
            yield self._group_to_sample(image_id, group, split)

    def _resolve_annotations_csv(self, split: str) -> Path:
        """Finds annotations CSV for the requested split."""
        if self.annotations_file and self.annotations_file.exists():
            return self.annotations_file

        candidates = [
            self.dataset_dir / f"annotations_{split}.csv",
            self.dataset_dir / f"{split}_annotations.csv",
            self.dataset_dir / "annotations.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _parse_annotations_csv(self, csv_path: Path) -> dict[str, dict]:
        """Parses VinDr-CXR CSV grouping findings by image_id."""
        images: dict[str, dict] = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["image_id"]
                class_name = row.get("class_name", "No finding").strip()
                rad_id = row.get("rad_id", "R1")

                if img_id not in images:
                    images[img_id] = {"findings": [], "rad_ids": set()}

                images[img_id]["rad_ids"].add(rad_id)

                if class_name.lower() != "no finding" and row.get("x_min") and row.get("y_min"):
                    try:
                        xmin = float(row["x_min"])
                        ymin = float(row["y_min"])
                        xmax = float(row["x_max"])
                        ymax = float(row["y_max"])
                        images[img_id]["findings"].append(
                            {
                                "class": class_name,
                                "box_xyxy": [xmin, ymin, xmax, ymax],
                                "rad_id": rad_id,
                            }
                        )
                    except (ValueError, TypeError):
                        pass

        return images

    def _group_to_sample(self, image_id: str, group: dict, split: str) -> MedicalVQASample:
        """Converts an image record to a standardized MedicalVQASample."""
        img_path = self.images_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.dicom"

        img_width, img_height = self._get_image_dimensions(img_path)
        gt_regions: list[GroundTruthRegion] = []

        target_finding = self.target_pathology or "pleural effusion"
        matching_findings = []

        for finding in group["findings"]:
            c_name = finding["class"].lower()
            if self.target_pathology is None or self.target_pathology in c_name:
                matching_findings.append(finding)
                norm_box = normalize_bounding_box(
                    finding["box_xyxy"], img_width, img_height, input_format="xyxy"
                )
                if norm_box is not None:
                    gt_regions.append(GroundTruthRegion(label=finding["class"], box_2d=norm_box))

        has_target = len(matching_findings) > 0
        answer = "yes" if has_target else "no"
        question = f"Is there {target_finding} in this chest radiograph?"

        return MedicalVQASample(
            sample_id=f"vindr_{image_id}",
            dataset_name="vindr",
            split=split,
            patient_id=f"vindr_patient_{image_id[:12]}",  # image prefix proxy for patient
            image_path=str(img_path),
            modality="CXR",
            organ="chest",
            question=question,
            answer=answer,
            question_type="closed",
            ground_truth_regions=gt_regions,
            has_ground_truth_regions=True,
            compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
            metadata={
                "image_id": image_id,
                "rad_count": len(group["rad_ids"]),
                "total_findings": len(group["findings"]),
            },
        )

    def _entry_to_sample(self, entry: dict, split: str) -> MedicalVQASample:
        """Parses a pre-split JSON entry into a MedicalVQASample."""
        img_id = entry.get("image_id", "unknown")
        img_path = self.images_dir / f"{img_id}.png"
        img_width, img_height = self._get_image_dimensions(img_path)

        gt_regions = []
        for f in entry.get("findings", []):
            norm_box = normalize_bounding_box(
                f["box_xyxy"], img_width, img_height, input_format="xyxy"
            )
            if norm_box:
                gt_regions.append(
                    GroundTruthRegion(label=f.get("class", "lesion"), box_2d=norm_box)
                )

        return MedicalVQASample(
            sample_id=f"vindr_{split}_{img_id}",
            dataset_name="vindr",
            split=split,
            patient_id=entry.get("patient_id", f"vindr_{img_id[:12]}"),
            image_path=str(img_path),
            modality="CXR",
            organ="chest",
            question=entry.get("question", "Is there pleural effusion in this chest radiograph?"),
            answer=entry.get("answer", "yes" if gt_regions else "no"),
            question_type="closed",
            ground_truth_regions=gt_regions,
            has_ground_truth_regions=True,
            compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
            metadata=entry.get("metadata", {}),
        )
