"""Training pairs for an --enhance net on multi-scale (mode 20) layers, in the layout train_enh_fresh.py reads.

Per photo (gen_enh_prod.py's photos and train/val split): a scale s (2x 40%, 3x..6x 15% each) and a ratio (0.1 / 0.3 /
0.5) by crc32 of the name; the layer = the photo area-shrunk by s/2 (2x: the photo), even-sized, kept as keep\\<name>\\gt.png
(the net's target). It is encrypted as a one-layer `--multiscale` container, every 120 px tile at a random flat rate
(log-normal around 1.2 x ratio, 0.02..1, as the planner's tiles), the 2x layers with 450k closed-loop anchors, the others
with none (as cs_encrypt_multiscale); decoded without the net, with the 9-plane dumps (an empty anchor sidecar when there
are no anchors). Crops as gen_enh_prod.save_crops, against gt.png. Train with Params-test/finetune_enh_layers.py.

usage: python Params-test/gen_enh_layers.py   env OUT, ENH_EXE (pin one), ENH_DEC_PROX_OFF=1, ENH_DEC_ARGS, WORKER, NWORK, LIMIT"""
import os, sys, time, zlib
import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("ENH_CHROMA", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_enh_prod as g

Image.MAX_IMAGE_PIXELS = None
T = 120
SCALES = [2] * 8 + [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]   # 40% 2x, 15% each other
RATIOS = [0.1, 0.3, 0.5]


def one_photo(split, name, f, log):
    keep = g.OUT + rf"\keep\{name}"; os.makedirs(keep, exist_ok=True)
    h = zlib.crc32(name.encode())
    s, ratio = SCALES[h % len(SCALES)], RATIOS[(h // len(SCALES)) % len(RATIOS)]
    O = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    W, H = O.size
    if s == 2: L = O.crop((0, 0, W // 2 * 2, H // 2 * 2))
    else: L = O.resize((int(W / (s / 2)) // 2 * 2, int(H / (s / 2)) // 2 * 2), Image.BOX)
    L.save(keep + r"\gt.png", compress_level=1)
    lw, lh = L.size; th, tw = -(-lh // T), -(-lw // T)
    rng = np.random.default_rng(h)
    q = np.clip(np.exp(rng.normal(np.log(min(1.0, 1.2 * ratio)), 0.8, th * tw)), 0.02, 1.0)
    plan = np.zeros((th, tw, 3), np.uint8); plan[..., 0] = 2; plan[..., 1] = np.clip(np.rint(q * 255), 1, 254).reshape(th, tw)
    pp, enc = g.SCR + r"\plan.png", g.SCR + r"\container.png"
    Image.fromarray(plan[..., ::-1].copy(), "RGB").save(pp)
    anchors = ["--hr-detail", "450000", "--enhance", g.RANK_NET] if s == 2 else []
    t0 = time.time()
    rc = g.run(["encrypt", keep + r"\gt.png", enc, "--password", g.PW, "--no-preview", "--multiscale", pp, *anchors], g.ENV, g.SCR + r"\enc.log")
    if rc != 0 or not os.path.exists(enc):
        print("encode failed", name, rc, flush=True); return
    t1 = time.time()
    side = keep + r"\anchors_300k_sidecar.bin"
    if os.path.exists(side): os.remove(side)
    env = dict(g.DEC_ENV, CS_DUMP_YMASK=keep + r"\ymask.png", CS_DUMP_YSAMP=keep + r"\ysamp.png", CS_DUMP_HR=side,
               CS_DUMP_CHROMA=keep + "\\")
    rc = g.run(["decrypt", enc, keep + r"\dec_anchors.png", *g.DEC_FLAGS], env, g.SCR + r"\dec.log")
    if rc == 0 and not os.path.exists(side): open(side, "wb").write(b"\0\0\0\0")   # no anchors: K = 0
    size = os.path.getsize(enc); os.remove(enc)
    res = g.save_crops(split, name, keep + r"\gt.png") if rc == 0 else None
    if res is None:
        print("decode failed / shape mismatch", name, rc, flush=True); return
    p, W2, H2 = res
    log.write(f"{split},{name},{s},{ratio},{W2},{H2},{size},{p:.3f},{t1 - t0:.1f},{time.time() - t1:.1f}\n"); log.flush()
    return p, t1 - t0, time.time() - t1


def main():
    os.makedirs(g.SCR, exist_ok=True)
    jobs = g.gen_old.job_list()[g.WORKER::g.NWORK]
    print(f"worker {g.WORKER}/{g.NWORK}: {len(jobs)} photos -> {g.OUT}", flush=True)
    log = open(g.OUT + rf"\log_{g.WORKER}.csv", "a")
    t0, n_done = time.time(), 0
    for n, (split, p, f) in enumerate(jobs):
        name = f"{p}_{os.path.splitext(os.path.basename(f))[0]}"
        if os.path.exists(g.OUT + rf"\{split}\{name}.npz"): continue
        try:
            res = one_photo(split, name, f, log)
        except Exception as e:
            print("ERR", name, e, flush=True); continue
        if res:
            n_done += 1; el = time.time() - t0
            print(f"[{n + 1}/{len(jobs)}] {split} {name} {res[0]:.2f} dB enc {res[1]:.0f}s dec {res[2]:.0f}s  "
                  f"elapsed {el / 3600:.2f} h, ~{el / n_done * (len(jobs) - n - 1) / 3600:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
