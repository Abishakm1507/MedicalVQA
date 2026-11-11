"""Validation CLI and smoke-testing tool for real dataset samples.

Usage:
    python -m datasets.validate_dataset --dataset slake --dir /path/to/slake --split test
    python -m datasets.validate_dataset --dataset vqarad --dir /path/to/vqarad --split test
    python -m datasets.validate_dataset --dataset rsna --dir /path/to/rsna --split train
    python -m datasets.validate_dataset --dataset vindr --dir /path/to/vindr --split test
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets.base import check_patient_leakage
from datasets.rsna_loader import RSNADatasetLoader
from datasets.slake_loader import SlakeDatasetLoader
from datasets.vindr_loader import VinDrCXRDatasetLoader
from datasets.vqarad_loader import VQARADDatasetLoader

DATASET_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "slake": {
        "required_files": ["train.json", "validate.json", "test.json"],
        "required_dirs": ["imgs/"],
        "optional_dirs": ["masks/"],
        "image_formats": [".jpg", ".png"],
        "annotation_format": "JSON array of objects with 'question', 'answer', 'q_lang', 'boxes', 'location', 'modality'",
        "patient_id_supported": False,
    },
    "vqarad": {
        "required_files": ["VQA_RAD_Dataset.json"],
        "required_dirs": ["images/"],
        "optional_dirs": [],
        "image_formats": [".jpg", ".png"],
        "annotation_format": "JSON array with 'qid', 'question', 'answer', 'answer_type', 'image_name', 'image_organ'",
        "patient_id_supported": False,
    },
    "rsna": {
        "required_files": ["stage_2_train_labels.csv"],
        "required_dirs": ["stage_2_train_images/"],
        "optional_dirs": [],
        "image_formats": [".dcm", ".png"],
        "annotation_format": "CSV with 'patientId', 'x', 'y', 'width', 'height', 'Target'",
        "patient_id_supported": True,
    },
    "vindr": {
        "required_files": ["annotations_train.csv"],
        "required_dirs": ["images/"],
        "optional_dirs": [],
        "image_formats": [".dicom", ".png"],
        "annotation_format": "CSV with 'image_id', 'rad_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max'",
        "patient_id_supported": False,
    },
}


def validate_dataset_sample(
    dataset_name: str,
    dataset_dir: str | Path,
    split: str = "test",
    max_samples: int = 50,
) -> dict[str, Any]:
    """Validates an official dataset directory against its expected schema and file structures."""
    p_dir = Path(dataset_dir)
    if not p_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {p_dir}")

    spec = DATASET_SPECIFICATIONS.get(dataset_name.lower())
    if not spec:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_SPECIFICATIONS.keys())}"
        )

    print("\n=======================================================")
    print(f" Validating Dataset: {dataset_name.upper()} ({p_dir})")
    print(f" Expected Patient ID Support: {spec['patient_id_supported']}")
    print("=======================================================")

    # 1. Directory Structure Audit
    required_files: list[str] = spec["required_files"]
    required_dirs: list[str] = spec["required_dirs"]
    missing_files = [f for f in required_files if not (p_dir / f).exists()]
    missing_dirs = [d for d in required_dirs if not (p_dir / d).exists()]

    if missing_files:
        print(f"  [!] Missing required files: {missing_files}")
    else:
        print("  [✓] All required split/label files present.")

    if missing_dirs:
        print(f"  [!] Missing required directories: {missing_dirs}")
    else:
        print("  [✓] All required directories present.")

    # 2. Loader Instantiation
    loader: Any
    if dataset_name == "slake":
        loader = SlakeDatasetLoader(dataset_dir=p_dir)
    elif dataset_name == "vqarad":
        loader = VQARADDatasetLoader(dataset_dir=p_dir)
    elif dataset_name == "rsna":
        loader = RSNADatasetLoader(dataset_dir=p_dir)
    elif dataset_name == "vindr":
        loader = VinDrCXRDatasetLoader(dataset_dir=p_dir)

    # 3. Stream Validation
    loaded_samples = []
    total_boxes = 0
    missing_image_count = 0
    degenerate_box_count = 0

    print(f"  Streaming first {max_samples} samples from split '{split}'...")
    try:
        for idx, sample in enumerate(loader.stream_split(split)):
            loaded_samples.append(sample)
            total_boxes += len(sample.ground_truth_regions)

            if not Path(sample.image_path).exists():
                missing_image_count += 1

            for region in sample.ground_truth_regions:
                ymin, xmin, ymax, xmax = region.box_2d
                if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
                    degenerate_box_count += 1

            if idx + 1 >= max_samples:
                break
    except Exception as e:
        print(f"  [ERROR] Streaming failed: {e}")
        return {"status": "FAILED", "error": str(e)}

    print(f"  [✓] Successfully ingested {len(loaded_samples)} samples.")
    print(f"  [i] Total ground-truth regions parsed: {total_boxes}")
    print(f"  [i] Missing image references: {missing_image_count}")
    print(f"  [i] Degenerate/invalid bounding boxes: {degenerate_box_count}")

    # 4. Patient Isolation Audit
    leakage_report = check_patient_leakage(loaded_samples, [])
    print(f"  [i] Patient identity verified available: {leakage_report.patient_ids_available}")

    summary = {
        "status": "PASSED"
        if not missing_files and not missing_dirs and degenerate_box_count == 0
        else "WARNING",
        "dataset": dataset_name,
        "samples_checked": len(loaded_samples),
        "total_regions": total_boxes,
        "missing_images": missing_image_count,
        "degenerate_boxes": degenerate_box_count,
        "patient_ids_available": leakage_report.patient_ids_available,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MedicalVQA dataset directory against specifications."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["slake", "vqarad", "rsna", "vindr"],
        help="Dataset name",
    )
    parser.add_argument("--dir", required=True, help="Path to dataset root folder")
    parser.add_argument("--split", default="test", help="Split name to inspect")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum samples to stream")

    args = parser.parse_args()
    res = validate_dataset_sample(args.dataset, args.dir, args.split, args.max_samples)
    print("\nValidation Summary:", res)
    if res["status"] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
