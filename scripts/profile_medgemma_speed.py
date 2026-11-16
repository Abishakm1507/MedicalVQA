"""Time MedGemma sampling on a real SLAKE image (GPU runtime only).

Usage (Colab, from the repo root):
    python scripts/profile_medgemma_speed.py --configs fp16,fp16-4bit

Configs: bf16-4bit, bf16, fp16, fp16-4bit. A T4 has no native bfloat16, so
bf16 maths can be very slow there; fp16 is faster but Gemma 3 may overflow, so
the printed sample replies must be checked (not empty / valid JSON).
Reports prefill time (1 new token), full sample_n time and per-step time.
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
CONFIGS = {
    "bf16-4bit": ("4bit", None),
    "bf16": (None, None),
    "fp16": (None, "float16"),
    "fp16-4bit": ("4bit", "float16"),
}


def profile(name: str, n: int, max_tokens: int, image: Image.Image) -> None:
    import torch

    quantization, dtype = CONFIGS[name]
    t0 = time.perf_counter()
    client = MedGemmaVLMClient(
        model_id="google/medgemma-4b-it",
        device="cuda",
        quantization=quantization,
        dtype=dtype,
    )
    client._load()
    print(f"[{name}] load: {time.perf_counter() - t0:.1f}s")
    tok = client._processor.tokenizer

    torch.cuda.synchronize()
    t = time.perf_counter()
    client.sample_n(image, QUESTIONS[0], n=n, temperature=0.7, max_tokens=1)
    torch.cuda.synchronize()
    print(f"[{name}] prefill (1 new token, n={n}): {time.perf_counter() - t:.1f}s")

    for i, q in enumerate(QUESTIONS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = client.sample_n(image, q, n=n, temperature=0.7, max_tokens=max_tokens)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        lens = [len(tok(o, add_special_tokens=False)["input_ids"]) for o in out]
        print(
            f"[{name}] q{i}: {dt:.1f}s  new_tokens={lens}  "
            f"-> {dt / max(max(lens), 1) * 1000:.0f} ms/step"
        )
        if i == 0:
            print(f"[{name}] sample reply: {out[0]!r}")
    print(f"[{name}] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del client
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--configs", default="fp16,fp16-4bit")
    args = ap.parse_args()
    img_path = sorted(IMG_DIR.glob("xmlab*/source.jpg"))[0]
    image = Image.open(img_path).convert("RGB")
    print("image:", img_path, image.size)
    for name in args.configs.split(","):
        profile(name.strip(), args.n, args.max_tokens, image)


if __name__ == "__main__":
    main()
