"""Test Ollama API with real SLAKE image."""
import sys
sys.path.insert(0, '.')

import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen2.5vl:3b"

slake_dir = Path("data/SLAKE")
img_path = slake_dir / "imgs" / "xmlab57" / "source.jpg"

# 1. Check Ollama server is reachable
try:
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    print(f"Ollama server reachable at {OLLAMA_BASE}")
    models = data.get("models", [])
    print(f"Installed models: {[m.get('name') for m in models]}")
    for m in models:
        if m.get("name") == MODEL:
            print(f"  Target model found: {m.get('name')}, size: {m.get('size')}, digest: {m.get('digest')[:20]}...")
except Exception as e:
    print(f"ERROR connecting to Ollama: {type(e).__name__}: {e}")
    sys.exit(1)

# 2. Load and encode a real SLAKE image
image = Image.open(img_path).convert("RGB")
print(f"\nImage: {img_path}, size: {image.size}, mode: {image.mode}")

# Resize if too large (Ollama may have limits)
max_dim = 1024
if max(image.size) > max_dim:
    image.thumbnail((max_dim, max_dim))
    print(f"Resized to: {image.size}")

import io
buf = io.BytesIO()
image.save(buf, format="JPEG", quality=85)
image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Image base64 length: {len(image_b64)}")

# 3. Test Ollama generate API with vision
question = "What modality is used to take this image?"

payload = {
    "model": MODEL,
    "prompt": question,
    "images": [image_b64],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 512,
    },
}

print(f"\nSending request to Ollama: {question}")
try:
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    
    print(f"\nResponse received:")
    print(f"  Response: {result.get('response', '')[:500]}")
    print(f"  Done: {result.get('done')}")
    print(f"  Total duration: {result.get('total_duration', 0) / 1e9:.2f}s")
    print(f"  Load duration: {result.get('load_duration', 0) / 1e9:.2f}s")
    print(f"  Prompt eval count: {result.get('prompt_eval_count')}")
    print(f"  Eval count: {result.get('eval_count')}")
    print(f"  Eval duration: {result.get('eval_duration', 0) / 1e9:.2f}s")
    
except Exception as e:
    print(f"ERROR calling Ollama API: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test with a second question
question2 = "Which part of the body does this image belong to?"
payload2 = {
    "model": MODEL,
    "prompt": question2,
    "images": [image_b64],
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 512},
}

print(f"\nSending second request: {question2}")
try:
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload2).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result2 = json.loads(resp.read())
    
    print(f"\nResponse received:")
    print(f"  Response: {result2.get('response', '')[:500]}")
    print(f"  Total duration: {result2.get('total_duration', 0) / 1e9:.2f}s")
    
except Exception as e:
    print(f"ERROR calling Ollama API: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nOllama API test completed")
