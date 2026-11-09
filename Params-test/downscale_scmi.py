"""SCMI30-IITRPR phone JPEGs shrunk by an integer factor -> PNGs: near-lossless phone-processed training targets
(ENH_SCMI in x64/GpuBuild/gen_enh_seg8tile_anchor.py). A 2x box average spans whole 2x2 luma blocks and exactly one
chroma sample of the 4:2:0 JPEG, so the blocking and the chroma subsampling mostly vanish while the phone's colour
rendering, sharpening and noise reduction stay. EXIF orientation applied first, sizes cropped to a multiple of the factor.
Same folder layout and stems as the source (so job_list's order and val split are unchanged). Resumable; NPROC processes.

usage: python Params-test/downscale_scmi.py [factor dst]   (default 2, F:\\scmi30_ds2)"""
import glob, os, sys, time
from concurrent.futures import ProcessPoolExecutor

SRC = r"G:\Download\archive\SCMI30-IITRPR"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DST = sys.argv[2] if len(sys.argv) > 2 else rf"F:\scmi30_ds{K}"


def shrink(f):
    import numpy as np
    from PIL import Image, ImageOps
    out = os.path.join(DST, os.path.splitext(os.path.relpath(f, SRC))[0] + ".png")
    if os.path.exists(out): return "kept"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    a = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"), np.float32)
    H, W = a.shape[0] // K * K, a.shape[1] // K * K
    a = a[:H, :W].reshape(H // K, K, W // K, K, 3).mean((1, 3))
    Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8)).save(out + ".part.png", compress_level=3)
    os.replace(out + ".part.png", out)
    return "made"


if __name__ == "__main__":
    files = sorted(f for f in glob.glob(SRC + r"\**\*", recursive=True) if f.lower().endswith((".jpg", ".jpeg")))
    t0, n = time.time(), {}
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 12))) as pool:
        for i, res in enumerate(pool.map(shrink, files, chunksize=8)):
            n[res] = n.get(res, 0) + 1
            if (i + 1) % 1000 == 0:
                print(f"[{i + 1}/{len(files)}] {n}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {n}, {(time.time() - t0) / 60:.1f} min", flush=True)
