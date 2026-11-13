import sys
sys.path.insert(0, '.')
from localization.segmenter import MedSAMRegionSegmenter, SegmenterConfig
from pathlib import Path

ckpt = Path('data/models/medsam_vit_b.pth')
print(f'Checkpoint: {ckpt.exists()}, size: {ckpt.stat().st_size} bytes')

try:
    seg = MedSAMRegionSegmenter(SegmenterConfig(checkpoint_path=str(ckpt), device='cpu'))
    print('MedSAM loaded successfully on CPU')
    print(f'Device: {seg.device}')
    print(f'Has _sam: {hasattr(seg, "_sam")}')
    print(f'Has _predictor: {hasattr(seg, "_predictor")}')

    # Check state dict
    sd = seg._sam.state_dict()
    keys = list(sd.keys())
    print(f'State dict keys: {len(keys)}')
    vit_keys = [k for k in keys if "patch_embed" in k or "blocks." in k]
    print(f'ViT-B keys: {len(vit_keys)}')
    print(f'Sample keys: {keys[:3]} ... {keys[-3:]}')

except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
