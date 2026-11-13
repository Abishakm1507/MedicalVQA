"""SLAKE sample inference pipeline for FS-004/FS-005 baseline.

Runs VLM inference on SLAKE test samples and saves predictions to JSONL.
Records failures, skipped samples, and inference errors.

Usage:
    python scripts/run_slake_inference.py [--model mock|qwen] [--limit N]
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from datasets.slake_loader import SlakeDatasetLoader
from vlm import MockVLMClient, QwenVLMClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "runs"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SLAKE_DIR = FIXTURES_DIR / "datasets" / "slake"
IMAGES_DIR = FIXTURES_DIR / "images"

logger = logging.getLogger("slake_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_inference(
    model: str = "mock",
    limit: int | None = None,
    output_jsonl: Path | None = None,
    output_csv: Path | None = None,
) -> dict:
    """Run VLM inference on SLAKE test samples.

    Returns a summary dict with counts and error details.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_jsonl is None:
        output_jsonl = EXPERIMENTS_DIR / f"slake_inference_{model}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.jsonl"
    if output_csv is None:
        output_csv = EXPERIMENTS_DIR / f"slake_inference_{model}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"

    # Load dataset
    loader = SlakeDatasetLoader(
        dataset_dir=SLAKE_DIR,
        images_dir=IMAGES_DIR,
        language="en",
    )
    samples = loader.load_split("test")
    logger.info("Loaded %d SLAKE test samples (English only)", len(samples))

    # Initialize VLM client
    if model == "mock":
        client = MockVLMClient(
            model_name="mock-vlm",
            device="cpu",
            default_answer="mock_prediction",
            findings=[],
        )
    elif model == "qwen":
        client = QwenVLMClient(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            quantization=None,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    logger.info("Using VLM client: %s (device=%s)", client.model_name, client.device)

    # Run inference
    total = len(samples) if limit is None else min(limit, len(samples))
    passed = 0
    failed = 0
    skipped = 0
    errors = []

    with open(output_jsonl, "w", encoding="utf-8") as jf, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as cf:
        csv_writer = csv.writer(cf)
        csv_writer.writerow(
            [
                "sample_id",
                "dataset_name",
                "image_id",
                "question",
                "ground_truth_answer",
                "predicted_answer",
                "model",
                "device",
                "inference_time_s",
                "status",
                "error",
            ]
        )

        for idx, sample in enumerate(samples):
            if limit is not None and idx >= limit:
                break

            sample_id = sample.sample_id
            image_path = Path(sample.image_path)

            # Check image exists
            if not image_path.exists():
                skipped += 1
                errors.append(
                    {
                        "sample_id": sample_id,
                        "error": f"Image not found: {image_path}",
                    }
                )
                logger.warning("Skipping %s: image not found at %s", sample_id, image_path)
                continue

            # Load image
            try:
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                failed += 1
                errors.append(
                    {
                        "sample_id": sample_id,
                        "error": f"Failed to load image: {exc}",
                    }
                )
                logger.error("Failed to load image for %s: %s", sample_id, exc)
                continue

            # Run inference
            try:
                t0 = time.perf_counter()
                structured = client.generate_structured(
                    marked_image=image,
                    question=sample.question,
                    region_ids=[r.label for r in sample.ground_truth_regions]
                    if sample.ground_truth_regions
                    else None,
                    temperature=0.0,
                    max_tokens=512,
                )
                t1 = time.perf_counter()
                inference_time = t1 - t0

                record = {
                    "sample_id": sample_id,
                    "dataset_name": sample.dataset_name,
                    "image_id": sample.image_id,
                    "image_path": sample.image_path,
                    "modality": sample.modality,
                    "organ": sample.organ,
                    "question": sample.question,
                    "ground_truth_answer": sample.answer,
                    "question_type": sample.question_type,
                    "predicted_answer": structured.answer,
                    "model": client.model_name,
                    "device": client.device,
                    "inference_time_s": round(inference_time, 4),
                    "status": "success",
                    "findings": [
                        {
                            "claim": f.claim,
                            "region_id": f.region_id,
                            "status": f.status,
                            "confidence": f.confidence,
                        }
                        for f in structured.findings
                    ],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }

                jf.write(json.dumps(record) + "\n")

                csv_writer.writerow(
                    [
                        sample_id,
                        sample.dataset_name,
                        sample.image_id,
                        sample.question,
                        sample.answer,
                        structured.answer,
                        client.model_name,
                        client.device,
                        round(inference_time, 4),
                        "success",
                        "",
                    ]
                )

                passed += 1
                logger.info(
                    "[%d/%d] %s: answer='%s' (gt='%s') in %.2fs",
                    idx + 1,
                    total,
                    sample_id,
                    structured.answer[:80],
                    sample.answer[:80],
                    inference_time,
                )

            except Exception as exc:
                failed += 1
                errors.append(
                    {
                        "sample_id": sample_id,
                        "error": str(exc),
                    }
                )
                logger.error("Inference failed for %s: %s", sample_id, exc)

                # Write error record
                error_record = {
                    "sample_id": sample_id,
                    "dataset_name": sample.dataset_name,
                    "image_id": sample.image_id,
                    "question": sample.question,
                    "ground_truth_answer": sample.answer,
                    "predicted_answer": "",
                    "model": client.model_name,
                    "device": client.device,
                    "inference_time_s": None,
                    "status": "error",
                    "error": str(exc),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
                jf.write(json.dumps(error_record) + "\n")

                csv_writer.writerow(
                    [
                        sample_id,
                        sample.dataset_name,
                        sample.image_id,
                        sample.question,
                        sample.answer,
                        "",
                        client.model_name,
                        client.device,
                        "",
                        "error",
                        str(exc),
                    ]
                )

    summary = {
        "model": client.model_name,
        "device": client.device,
        "total_samples": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "errors": errors,
    }

    print("\n" + "=" * 60)
    print("SLAKE INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Model:        {client.model_name}")
    print(f"Device:       {client.device}")
    print(f"Total:        {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"Skipped:      {skipped}")
    print(f"JSONL:        {output_jsonl}")
    print(f"CSV:          {output_csv}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAKE sample inference")
    parser.add_argument(
        "--model",
        choices=["mock", "qwen"],
        default="mock",
        help="VLM model to use (mock or qwen)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    args = parser.parse_args()

    summary = run_inference(
        model=args.model,
        limit=args.limit,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
    )

    sys.exit(0 if summary["failed"] == 0 else 1)
