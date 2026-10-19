"""Better-than-PSNR metrics: SSIM (structure) + CIEDE2000 (color).

Usage: metrics.py <original.png> <decoded.png>
PSNR averages squared error blindly: a global brightness shift and a
hue shift score the same if magnitudes match. SSIM reacts to structural
damage (blur, blocking, seams); CIEDE2000 reacts to color casts the way
human vision does (green/magenta shifts cost more than neutral ones).
"""
import sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000


def main(orig_path, dec_path):
    a = np.asarray(Image.open(orig_path).convert("RGB")).astype(np.float64)
    b = Image.open(dec_path).convert("RGB")
    if b.size != (a.shape[1], a.shape[0]):
        # half-res pipeline output: upscale to reference (mirrors compare)
        b = b.resize((a.shape[1], a.shape[0]), Image.BILINEAR)
    b = np.asarray(b).astype(np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)

    s = ssim(a / 255.0, b / 255.0, channel_axis=2, data_range=1.0)
    lab_a = rgb2lab(a / 255.0)
    lab_b = rgb2lab(b / 255.0)
    de = deltaE_ciede2000(lab_a, lab_b)
    print("SSIM:            %.4f" % s)
    print("CIEDE2000 mean:  %.3f  (just-noticeable ~1-2, visible >5)" % float(np.mean(de)))
    print("CIEDE2000 p95:   %.3f  (worst 5%% of pixels)" % float(np.quantile(de, 0.95)))
    print("CIEDE2000 p99:   %.3f  (worst 1%% of pixels)" % float(np.quantile(de, 0.99)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
