"""FS-005 Phase 5 Colab T4 feasibility-pilot entry point.

Colocompatible execution script for the FS-005 Phase 5 feasibility pilot.
Upload this file to a Google Colab notebook (or run the cells directly) and
execute it cell by cell (cells are marked with ``# %%``).

DESIGN PRINCIPLES (read before running):
- Real model inference is GATED OFF by default (RUN_REAL_INFERENCE = False).
  The script runs a mock-mode dry run unless you explicitly flip the flag.
- No credentials are ever printed, echoed, or saved. HF_TOKEN is read only
  from Colab Secrets via os.environ.
- All persistent artifacts are written to Google Drive under
  MyDrive/MedicalVQA/outputs/fs005_phase5/ — never to ephemeral /content.
- Models are loaded SEQUENTIALLY and never resident in GPU VRAM at the same
  time (MedGemma-4B, then Qwen2.5-VL-3B, unloaded between).
- Conditions 2 and 3 reuse the SAME N=5 generation artifacts (only the
  abstention decision differs).
- Completed artifacts are skipped on resume; failed/corrupt artifacts are
  retried without overwriting valid results.

Traceability: FS-005 Phase 5 execution harness; ADR 0005 (N=5, confidence gate).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Pilot safety switches
# ---------------------------------------------------------------------------

# Flip to True ONLY when you are ready to run REAL MedGemma/Qwen inference on
# Colab T4. While False, the script executes a mock-mode dry run (no model
# weights are downloaded, no inference is performed).
RUN_REAL_INFERENCE = False

# Number of pilot samples (must be within 10-20).
PILOT_SIZE = 15
TRACK_B_PILOT_SIZE = 7

# ---------------------------------------------------------------------------
# 1. Colab environment detection
# ---------------------------------------------------------------------------


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Google Drive mount + persistent output path
# ---------------------------------------------------------------------------


def mount_drive_and_output() -> Path:
    """Mount Google Drive and return the persistent output directory.

    Returns the Path: /content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5
    """
    if not is_colab():
        # Local (non-Colab) fallback: use the repo-relative output dir.
        return Path("experiments/runs/fs005_phase5")

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive", force_remount=True)
    output_dir = Path("/content/drive/MyDrive/MedicalVQA/outputs/fs005_phase5")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generations").mkdir(parents=True, exist_ok=True)
    (output_dir / "conditions").mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration").mkdir(parents=True, exist_ok=True)
    print(f"Persistent output directory: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# 3. Hugging Face authentication via Colab Secrets (no printing)
# ---------------------------------------------------------------------------


def configure_hf_auth() -> None:
    """Authenticate to Hugging Face using the HF_TOKEN Colab Secret.

    Colab Secrets do NOT automatically populate ``os.environ``. They are read
    via ``google.colab.userdata.get("HF_TOKEN")``. We fall back to
    ``os.environ`` for non-Colab/local testing only. The token is never
    printed, echoed, persisted to disk, or committed.
    """
    token: str | None = None
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None

    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN secret not set. Add it in Colab Secrets (key: HF_TOKEN) "
            "and restart the runtime, then re-run this cell."
        )
    from huggingface_hub import login  # type: ignore

    login(token=token, add_to_git_credential=False)
    print("Hugging Face authentication configured (token not printed).")


# ---------------------------------------------------------------------------
# 4. Repository + dependency preparation
# ---------------------------------------------------------------------------


def prepare_repository() -> Path:
    """Ensure the MedicalVQA repository is importable.

    In Colab, clone the repo to /content/MedicalVQA if not already present.
    Returns the repo root Path.
    """
    repo_root = Path("/content/MedicalVQA") if is_colab() else Path.cwd()
    if is_colab() and not repo_root.exists():
        import subprocess

        repo_url = os.environ.get(
            "MEDICALVQA_REPO_URL",
            "https://github.com/Kilo-Org/MedicalVQA.git",
        )
        subprocess.run(
            ["git", "clone", repo_url, str(repo_root)], check=True
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print(f"Repository root: {repo_root}")
    return repo_root


def install_dependencies() -> None:
    """Install lightweight Colab-compatible dependencies.

    Heavy ML deps (torch, transformers, accelerate, bitsandbytes,
    safetensors, huggingface_hub) are pre-installed on Colab or installed on
    demand; this cell only ensures dev/lint tooling is present. Protect the
    active Colab PyTorch/CUDA runtime from dependency-manifest changes.
    """
    import subprocess

    requirements_path = Path(__file__).resolve().parents[1] / "requirements-dev.txt"
    protected_prefixes = ("torch", "triton", "nvidia-", "cuda-")
    for requirement in requirements_path.read_text(encoding="utf-8").splitlines():
        line = requirement.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise RuntimeError(
                f"Unreviewed requirements directive in {requirements_path}: {line}"
            )
        package_name = line
        for separator in "<=>!~[":
            package_name = package_name.split(separator, 1)[0]
        package_name = package_name.strip().lower().replace("_", "-")
        if any(
            package_name == prefix or package_name.startswith(prefix)
            for prefix in protected_prefixes
        ):
            raise RuntimeError(
                f"Refusing to install {package_name}: Colab's PyTorch/CUDA stack "
                "must remain untouched."
            )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(requirements_path),
        ],
        check=True,
    )
    print("Dependencies verified.")


# ---------------------------------------------------------------------------
# 5. Model clients (lazy, sequential)
# ---------------------------------------------------------------------------


def load_medgemma_client(device: str = "cuda"):
    """Load MedGemma-4B client (4-bit, GPU). Returns the client instance."""
    from vlm.medgemma_client import MedGemmaVLMClient  # type: ignore

    client = MedGemmaVLMClient(
        model_id="google/medgemma-4b-it",
        device=device,
        quantization="4bit",
    )
    # Trigger lazy load.
    client._load()  # noqa: SLF001
    print(f"MedGemma-4B loaded on {client.device}")
    return client


def unload_client(client) -> None:
    """Unload a model client to free GPU VRAM before loading the next model."""
    import torch  # type: ignore

    if client is None:
        return
    client._model = None  # noqa: SLF001
    client._processor = None  # noqa: SLF001
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded; GPU cache cleared.")


def load_qwen_client(device: str = "cuda"):
    """Load Qwen2.5-VL-3B client (4-bit, GPU). Returns the client instance."""
    from vlm.qwen_client import QwenVLMClient  # type: ignore

    client = QwenVLMClient(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        device=device,
        quantization="4bit",
        trust_remote_code=True,
    )
    client._load()  # noqa: SLF001
    print(f"Qwen2.5-VL-3B loaded on {client.device}")
    return client


# ---------------------------------------------------------------------------
# 6. Pilot cohort + image/sample preparation
# ---------------------------------------------------------------------------


def load_pilot_manifest(repo_root: Path) -> dict:
    """Load the validation-only pilot cohort manifest.

    The manifest is produced by scripts/fs005_phase5_pilot_cohort.py and must
    already exist (generated locally and copied to the repo, or regenerated
    here in dry-run mode).
    """
    manifest_path = repo_root / "experiments/runs/fs005_phase5/pilot_cohort/pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Pilot manifest not found. Run scripts/fs005_phase5_pilot_cohort.py "
            "first to generate experiments/runs/fs005_phase5/pilot_cohort/."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"Pilot cohort: {manifest['pilot_size']} samples, "
        f"{len(manifest['pilot_image_ids'])} distinct images"
    )
    return manifest


def build_input_samples(
    manifest: dict, repo_root: Path, track: str
) -> tuple[list[dict], dict[str, object]]:
    """Build GenerationRunner input samples + PIL images keyed by image_id.

    Images are read from data/SLAKE/imgs/xmlab{img_id}/source.jpg.
    Track B receives only samples with validated, visibly overlaid GT boxes.
    Track C receives the original image with no GT metadata or region IDs.
    """
    from PIL import Image  # type: ignore

    from scripts.fs005_phase5_ground_truth import build_ground_truth_overlay

    if track not in {"B", "C"}:
        raise ValueError(f"Unknown pilot track: {track!r}; expected 'B' or 'C'")

    samples: list[dict] = []
    images: dict[str, object] = {}

    slaake_dir = repo_root / "data" / "SLAKE"
    for entry in manifest["samples"]:
        qid = entry["qid"]
        iid = entry["img_id"]
        image_dir = slaake_dir / "imgs" / f"xmlab{iid}"
        img_path = image_dir / "source.jpg"
        with Image.open(img_path) as source:
            image = source.convert("RGB")

        region_ids: list[str] = []
        region_metadata: list[dict] = []
        input_image = image
        if track == "B":
            input_image, region_ids, region_metadata = build_ground_truth_overlay(
                image, image_dir / "detection.json"
            )
            if not region_ids:
                continue

        images[str(iid)] = input_image

        sample = {
            "question_id": qid,
            "image_id": iid,
            "question": entry["question"],
            "region_ids": region_ids,
        }
        if track == "B":
            sample["region_metadata"] = region_metadata
        samples.append(sample)

    if track == "B" and len(samples) != TRACK_B_PILOT_SIZE:
        raise RuntimeError(
            f"Track B requires exactly {TRACK_B_PILOT_SIZE} pilot samples with "
            f"usable ground-truth regions; found {len(samples)}."
        )
    if track == "C" and len(samples) != PILOT_SIZE:
        raise RuntimeError(
            f"Track C requires exactly {PILOT_SIZE} pilot samples; found {len(samples)}."
        )

    print(f"Prepared {len(samples)} input samples with {len(images)} images (track {track}).")
    return samples, images


# ---------------------------------------------------------------------------
# 7. GenerationRunner execution (Tracks B & C)
# ---------------------------------------------------------------------------


def run_tracks(
    repo_root: Path,
    output_dir: Path,
    samples: list[dict],
    images: dict[str, object],
    vlm_client,
    model_name: str,
    model_revision: str | None,
    track: str,
):
    """Execute one track with GenerationRunner.

    Persists generation artifacts and progress to the persistent output_dir.
    Conditions 2 and 3 reuse the same N=5 artifacts; resume skips completed.
    """
    from scripts.fs005_phase5_generation import GenerationRunner  # type: ignore

    runner = GenerationRunner(
        vlm_client=vlm_client,
        model_name=model_name,
        model_revision=model_revision,
        track=track,
        output_dir=output_dir,
        n_samples=5,
        temperature=0.7,
        random_seed=42,
    )
    summary = runner.run_batch(samples, images)
    print(json.dumps(summary, indent=2))
    return runner, summary


# ---------------------------------------------------------------------------
# 8. Main execution flow (mock by default)
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = prepare_repository()
    output_dir = mount_drive_and_output()

    manifest = load_pilot_manifest(repo_root)

    if not RUN_REAL_INFERENCE:
        print("\n=== MOCK MODE (dry run) — no real model inference ===")
        install_dependencies()
        from vlm.mock_client import MockVLMClient  # type: ignore

        client = MockVLMClient(model_name="mock-vlm")
        # Exercise both tracks with the mock client (cheap, no inference).
        for track in ("B", "C"):
            samples, images = build_input_samples(manifest, repo_root, track)
            run_tracks(
                repo_root=repo_root,
                output_dir=output_dir,
                samples=samples,
                images=images,
                vlm_client=client,
                model_name="mock-vlm",
                model_revision=None,
                track=track,
            )
        print(
            "\nMock dry run complete (both tracks exercised). Set "
            "RUN_REAL_INFERENCE=True for real inference."
        )
        return 0

    # ------------------------------------------------------------------
    # REAL MODEL EXECUTION (gated; present but not run in this task)
    # ------------------------------------------------------------------
    print("\n=== REAL MODEL EXECUTION (Colab T4) ===")
    configure_hf_auth()
    install_dependencies()

    # Pilot model/track matrix. Both models run both tracks; models are loaded
    # sequentially and never resident in GPU VRAM at the same time.
    matrix: list[tuple[str, str]] = [
        ("medgemma-4b", "B"),
        ("medgemma-4b", "C"),
        ("qwen2.5-vl-3b", "B"),
        ("qwen2.5-vl-3b", "C"),
    ]

    loaders = {
        "medgemma-4b": load_medgemma_client,
        "qwen2.5-vl-3b": load_qwen_client,
    }

    for model_name in ("medgemma-4b", "qwen2.5-vl-3b"):
        loader = loaders[model_name]
        client = loader(device="cuda")
        try:
            for mn, track in matrix:
                if mn != model_name:
                    continue
                samples, images = build_input_samples(manifest, repo_root, track)
                run_tracks(
                    repo_root=repo_root,
                    output_dir=output_dir,
                    samples=samples,
                    images=images,
                    vlm_client=client,
                    model_name=model_name,
                    model_revision=None,
                    track=track,
                )
        finally:
            unload_client(client)

    print("\nReal-model pilot execution complete. Artifacts persisted to Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
