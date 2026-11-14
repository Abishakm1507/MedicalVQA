"""Diagnostic script for MedGemma-4B empty generation responses.

Designed to run in Google Colab / Kaggle where the model weights are available.
Traces the complete pipeline for one genuine SLAKE sample:

  Image + question -> prompt -> processor -> model.generate() -> token IDs
  -> slicing -> decoding -> raw response

Prints processor input keys/shapes, input token count, model output type/shape,
total generated sequence length, number of new tokens, generated token IDs,
decoded output before parsing, and EOS/termination info.

Usage (Colab cell):
    python scripts/diagnose_medgemma_empty_response.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from PIL import Image

# Ensure the project root is on sys.path when run from Colab/Kaggle.
PROJECT_ROOT = os.environ.get("MEDICALVQA_ROOT", "/content/MedicalVQA")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vlm.medgemma_client import MedGemmaVLMClient  # noqa: E402

MODEL_ID = "google/medgemma-4b-it"
MANIFEST_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "runs", "slake_baseline_ollama", "selection_manifest.json"
)
SLAKE_IMAGES_DIR = "/content/slake/Slake1.0/imgs"  # Colab default; override via env


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA GPU required for this diagnostic.")

    # Load manifest and pick the first sample.
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    sample = manifest["samples"][0]
    qid = sample["qid"]
    img_name = sample["img_name"]
    question = sample["question"]
    gt = sample["answer"]
    print(f"\n=== Diagnosing sample qid={qid} ({img_name}) ===")
    print(f"Question: {question}")
    print(f"Ground truth: {gt}")

    image_path = os.path.join(SLAKE_IMAGES_DIR, img_name)
    if not os.path.exists(image_path):
        # Fallback: try nested layout.
        image_path = os.path.join("/content", "slake", img_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image.size}, mode={image.mode}")

    # ------------------------------------------------------------------
    # Load client (4-bit, bfloat16 - the fix)
    # ------------------------------------------------------------------
    client = MedGemmaVLMClient(model_id=MODEL_ID, device="cuda", quantization="4bit")
    t0 = time.time()
    client._load()
    t1 = time.time()
    print(f"\nModel loaded in {t1 - t0:.1f}s")

    # ------------------------------------------------------------------
    # Pipeline trace: build inputs
    # ------------------------------------------------------------------
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a medical visual question answering assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    # Single-call apply_chat_template (Gemma 3 recommended pattern).
    inputs = client._processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        images=[image],
    )

    print("\n--- Processor inputs ---")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]
    print(f"\nInput token count (input_length): {input_length}")
    print(f"Input IDs (first 30): {input_ids[0][:30].tolist()}")

    inputs_on_device = {k: v.to("cuda") for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Pipeline trace: model.generate()
    # ------------------------------------------------------------------
    print("\n--- model.generate() ---")
    t0 = time.time()
    with torch.inference_mode():
        output_ids = client._model.generate(
            **inputs_on_device,
            max_new_tokens=512,
            temperature=None,
            do_sample=False,
            pad_token_id=client._processor.tokenizer.pad_token_id,
            eos_token_id=client._processor.tokenizer.eos_token_id,
        )
    t1 = time.time()
    print(f"Generate elapsed: {t1 - t0:.2f}s")

    print(f"Output type: {type(output_ids).__name__}")
    if hasattr(output_ids, "shape"):
        print(f"Output shape: {tuple(output_ids.shape)}")
    total_seq_len = output_ids.shape[1]
    num_new = total_seq_len - input_length
    print(f"Total generated sequence length: {total_seq_len}")
    print(f"Number of new tokens: {num_new}")
    print(f"EOS token id: {client._processor.tokenizer.eos_token_id}")
    print(f"Output IDs (full): {output_ids[0].tolist()}")
    print(f"New token IDs: {output_ids[0][input_length:].tolist()}")

    # ------------------------------------------------------------------
    # Pipeline trace: decoding
    # ------------------------------------------------------------------
    new_ids = output_ids[0][input_length:]

    decoded_skip = client._processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    decoded_no_skip = client._processor.tokenizer.decode(new_ids, skip_special_tokens=False)

    print("\n--- Decoding ---")
    print(f"Decoded (skip_special_tokens=True):  '{decoded_skip}'")
    print(f"Decoded (skip_special_tokens=False): '{decoded_no_skip}'")

    # Full-sequence decode for reference.
    full_skip = client._processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Full sequence decoded (skip=True):   '{full_skip[:200]}'")

    # ------------------------------------------------------------------
    # Pipeline trace: client.generate() end-to-end
    # ------------------------------------------------------------------
    print("\n--- client.generate() end-to-end ---")
    t0 = time.time()
    response = client.generate(
        marked_image=image,
        question=question,
        temperature=0.0,
        max_tokens=512,
    )
    t1 = time.time()
    print(f"client.generate() elapsed: {t1 - t0:.2f}s")
    print(f"Raw response: '{response}'")
    print(f"Response length: {len(response)}")
    print(f"Response is empty: {response == ''}")

    # Termination info.
    eos_id = client._processor.tokenizer.eos_token_id
    new_ids_list = output_ids[0][input_length:].tolist()
    ended_with_eos = new_ids_list and new_ids_list[-1] == eos_id
    print(f"\nTermination: new tokens end with EOS={ended_with_eos}")
    print(f"New tokens: {new_ids_list}")

    print("\n=== Diagnostic complete ===")


if __name__ == "__main__":
    main()
