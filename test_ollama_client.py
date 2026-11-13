"""Test OllamaVLMClient with real SLAKE image."""
import sys
sys.path.insert(0, '.')

from PIL import Image
from vlm import OllamaVLMClient
from vlm.base import VLMClient

client = OllamaVLMClient()
print(f"Client: {client.model_name}, device: {client.device}")
print(f"Is VLMClient: {isinstance(client, VLMClient)}")

# Test with a real SLAKE image
image = Image.open("data/SLAKE/imgs/xmlab57/source.jpg").convert("RGB")
print(f"\nImage: {image.size}, mode: {image.mode}")

# Test generate
question = "What modality is used to take this image?"
print(f"\nQuestion: {question}")
response = client.generate(marked_image=image, question=question, temperature=0.0, max_tokens=512)
print(f"Response: {response}")
print(f"\nMetadata: {client.last_metadata}")

# Test generate_structured (inherited from VLMClient)
print(f"\n=== Testing generate_structured ===")
structured = client.generate_structured(
    marked_image=image,
    question=question,
    region_ids=["R1", "R2", "NONE"],
    temperature=0.0,
    max_tokens=512,
)
print(f"Answer: {structured.answer}")
print(f"Findings: {len(structured.findings)}")
for f in structured.findings:
    print(f"  - claim: {f.claim}, region_id: {f.region_id}, status: {f.status}, confidence: {f.confidence}")

# Test sample_n
print(f"\n=== Testing sample_n ===")
samples = client.sample_n(marked_image=image, question=question, n=2, temperature=0.7, max_tokens=256)
print(f"Got {len(samples)} samples")
for i, s in enumerate(samples):
    print(f"  Sample {i+1}: {s[:100]}")

print("\nOllamaVLMClient test PASSED")
