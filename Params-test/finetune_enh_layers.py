"""Fine-tune an --enhance net on multi-scale layer data (Params-test/gen_enh_layers.py) with x64/GpuBuild/train_enh_fresh.py,
each photo's target read from its keep\\<name>\\gt.png (the shrunk layer) instead of the original photo. The trainer's own
ENH_INIT (start checkpoint) and ENH_LR (peak lr, cosine to 0) make it a fine-tune.

usage: python Params-test/finetune_enh_layers.py [steps]   env as train_enh_fresh.py"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, r"x64\GpuBuild"))
import train_enh_fresh as t

_paths = t.photo_paths


def photo_paths(split):
    return [(n, k, k + r"\gt.png" if os.path.exists(k + r"\gt.png") else f) for n, k, f in _paths(split)]


t.photo_paths = photo_paths
if __name__ == "__main__":
    t.STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 16000   # the trainer only reads argv when run directly
    print(f"fine-tune on layers from {t.INIT}, lr {t.LR}", flush=True)
    t.main()
