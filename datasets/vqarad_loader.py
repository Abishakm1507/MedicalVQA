"""VQA-RAD dataset loader for MedicalVQA.

Reference: Section V-A, Table 1 of research paper.
Parses clinician-generated QA pairs for the ungrounded answer baseline (Track C).
Explicitly marks region annotations as unavailable (has_ground_truth_regions=False)
and patient_id=None as VQA-RAD does not publish clinical patient identifiers.
"""

import json
import logging
from collections.abc import Generator
from pathlib import Path

from core.schemas import EvaluationTrack, MedicalVQASample
from datasets.base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class VQARADDatasetLoader(BaseDatasetLoader):
    """Loader for the VQA-RAD radiology visual question answering benchmark."""

    def __init__(
        self,
        dataset_dir: str | Path,
        images_dir: str | Path | None = None,
        strict_image_exists: bool = False,
    ) -> None:
        super().__init__(
            dataset_dir, images_dir or Path(dataset_dir) / "images", strict_image_exists
        )

    def stream_split(self, split: str = "test") -> Generator[MedicalVQASample, None, None]:
        """Streams standardized samples from VQA-RAD.

        Args:
            split: 'train' or 'test'.
        """
        split_file = self._resolve_split_file(split)
        if not split_file.exists():
            raise FileNotFoundError(f"VQA-RAD split file not found: {split_file}")

        with open(split_file, encoding="utf-8") as f:
            raw_entries = json.load(f)

        if not isinstance(raw_entries, list):
            raise ValueError(
                f"Malformed VQA-RAD split file {split_file}: expected list, got {type(raw_entries)}."
            )

        for entry in raw_entries:
            if not isinstance(entry, dict):
                logger.warning("Skipping non-dictionary entry in %s: %s", split_file, entry)
                continue

            # 1. Resolve image file and image identifier
            img_name = str(entry.get("image_name", ""))
            img_id = Path(img_name).name
            img_path = self.images_dir / img_name
            if not img_path.exists():
                img_path = self.images_dir / img_id

            # 2. Extract organ and modality
            organ = str(entry.get("image_organ", "radiology")).lower()

            # 3. Question type classification
            raw_ans_type = str(entry.get("answer_type", "open")).strip().upper()
            q_type = "closed" if raw_ans_type == "CLOSED" else "open"

            qid = str(entry.get("qid", entry.get("id", "0")))
            sample_id = f"vqarad_{split}_{qid}"

            # 4. Patient identity: VQA-RAD does NOT publish clinical patient IDs.
            # Explicitly set patient_id to None; do NOT treat img_name as patient_id!
            patient_id = None

            # 5. VQA-RAD has no bounding boxes (Table 1: 'Questions and answers; no regions')
            # Assigns strictly to Track C (Answer-only evaluation)
            yield MedicalVQASample(
                sample_id=sample_id,
                dataset_name="vqarad",
                split=split,
                image_id=img_id,
                patient_id=patient_id,
                image_path=str(img_path),
                modality="RADIOLOGY",
                organ=organ,
                question=str(entry.get("question", "")).strip(),
                answer=str(entry.get("answer", "")).strip(),
                question_type=q_type,
                ground_truth_regions=[],
                has_ground_truth_regions=False,  # Explicitly flagged as unannotated for regions
                compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
                metadata={
                    "qid": entry.get("qid"),
                    "image_name": img_name,
                    "image_organ": entry.get("image_organ", ""),
                    "phrase_type": entry.get("phrase_type", ""),
                },
            )

    def _resolve_split_file(self, split: str) -> Path:
        """Finds the split file handling VQA-RAD naming conventions."""
        candidates = [
            self.dataset_dir / f"{split}.json",
            self.dataset_dir / f"VQA_RAD_{split.capitalize()}.json",
            self.dataset_dir / "VQA_RAD_Dataset.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]
