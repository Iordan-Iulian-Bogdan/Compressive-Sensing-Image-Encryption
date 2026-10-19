"""LPIPS perceptual distance (deep-feature similarity, lower is better).

Usage: metrics_lpips.py <original.png> <decoded.png> [--net alex|vgg]
Needs torch + lpips (pip) and downloads the backbone weights once
(~100-500 MB under ~/.cache). AlexNet is the standard calibrated net;
VGG tracks texture/detail damage more closely. Runs on CPU by default
(hip device is flaky for this workload); images are compared at native
resolution in tiles only if memory demands it.

Typical natural-photo range: clean ~0.05-0.15, visible damage 0.2-0.4,
garbage > 0.5. Compare RELATIVELY between your own decodes.
"""
import sys
import numpy as np
import torch
from PIL import Image
import lpips


_ref_size = None


def load(path, is_ref=False):
    global _ref_size
    im = Image.open(path).convert("RGB")
    if is_ref:
        _ref_size = im.size
    elif _ref_size is not None and im.size != _ref_size:
        # half-res pipeline output: upscale to reference (mirrors compare)
        im = im.resize(_ref_size, Image.BILINEAR)
    a = np.asarray(im).astype(np.float32)
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    return t / 127.5 - 1.0


def main(orig_path, dec_path, net="alex"):
    dev = torch.device("cpu")
    model = lpips.LPIPS(net=net).to(dev)
    model.eval()
    with torch.no_grad():
        x = load(orig_path, is_ref=True).to(dev)
        y = load(dec_path).to(dev)
        assert x.shape == y.shape, (x.shape, y.shape)
        d = model(x, y)
    print("LPIPS-%s: %.4f  (lower is better)" % (net, float(d.view(-1)[0])))


if __name__ == "__main__":
    args = sys.argv[1:]
    net = "alex"
    if "--net" in args:
        i = args.index("--net")
        net = args[i + 1]
        del args[i:i + 2]
    main(args[0], args[1], net)
