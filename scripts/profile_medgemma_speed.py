"""Time MedGemma sampling on a real SLAKE image (GPU runtime only).

Usage (Colab, from the repo root):
    python scripts/profile_medgemma_speed.py [--n 5] [--max-tokens 256]

Compares 4-bit (bitsandbytes) against unquantized bf16 loading, because a T4
has no native bfloat16 and bitsandbytes 4-bit decoding can be very slow.
Reports seconds per sample_n call and the new-token count of every sample.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from scripts.fs005_phase5_full_run import IMG_DIR  # noqa: E402
from vlm.medgemma_client import MedGemmaVLMClient  # noqa: E402

QUESTIONS = [
    "What modality is used to take this image?",
    "Is the spleen present in this image?",
    "Which organ is shown in this image?",
]


def profile(quantization: str | None, n: int, max_tokens: int, image: Image.Image) -> None:
    import torch

    label = quantization or "bf16-unquantized"
    t0 = time.perf_counter()
    client = MedGemmaVLMClient(
        model_id="google/medgemma-4b-it", device="cuda", quantization=quantization
    )
    client._load()
    print(f"[{label}] load: {time.perf_counter() - t0:.1f}s")
    tok = client._processor.tokenizer
    for i, q in enumerate(QUESTIONS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = client.sample_n(image, q, n=n, temperature=0.7, max_tokens=max_tokens)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        lens = [len(tok(o, add_special_tokens=False)["input_ids"]) for o in out]
        print(
            f"[{label}] q{i}: {dt:.1f}s  new_tokens per sample={lens}  "
            f"-> {dt / max(max(lens), 1) * 1000:.0f} ms/step (longest sample)"
        )
    print(f"[{label}] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del client
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    img_path = sorted(IMG_DIR.glob("xmlab*/source.jpg"))[0]
    image = Image.open(img_path).convert("RGB")
    print("image:", img_path, image.size)
    profile("4bit", args.n, args.max_tokens, image)
    profile(None, args.n, args.max_tokens, image)


if __name__ == "__main__":
    main()
