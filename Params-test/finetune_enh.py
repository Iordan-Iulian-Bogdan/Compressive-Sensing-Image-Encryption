"""Fine-tune an existing --enhance net with x64/GpuBuild/train_enh_fresh.py: same data, loss, EMA and val, but the net
starts from ENH_INIT and Adam starts at ENH_LR (cosine to 0 over the given steps) instead of 1e-3 from scratch.

usage: python Params-test/finetune_enh.py [steps]   env as train_enh_fresh.py, plus ENH_INIT (checkpoint, default the
       production net) and ENH_LR (default 2e-4)"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, r"x64\GpuBuild")
sys.path.insert(0, G)
import train_enh_fresh as t

INIT = os.environ.get("ENH_INIT", os.path.join(G, "enh_seg8a_x5_ema256_fresh_net_unet.pt"))
LR = float(os.environ.get("ENH_LR", 2e-4))

_make, _adam = t.make_net, torch.optim.Adam


def make_net(arch, extra):
    n = _make(arch, extra)
    n.load_state_dict(torch.load(INIT, map_location="cpu"))
    return n


t.make_net = make_net
t.torch.optim.Adam = lambda params, lr: _adam(params, lr=LR)
if __name__ == "__main__":
    t.STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 16000   # the trainer only reads argv when run directly
    print(f"fine-tune from {INIT}, lr {LR}", flush=True)
    t.main()
