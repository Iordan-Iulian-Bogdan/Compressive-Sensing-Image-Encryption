"""Training pairs for the --enhance net from the CURRENT production pipeline, in the layout train_enh_fresh.py reads.

Per photo (same photos and train/val split as x64/GpuBuild/gen_enh_seg8tile_anchor.py):
  encode: production format, --seg-auto 8 --ycc420 --seg-tile --ratio ENH_RATIO --hr-detail 300000, with the current net
          as --enhance for the closed-loop anchor ranking (ENH_RANK_NET, default the production net)
  decode: production decoder without the net (the net's input): --tv 0 cs-sr + ADMM, --hr-weight 1, CS_HR_NO_POSTHOC=1,
          the auto lambda law as built, and the dumps the trainer's planes need (CS_DUMP_YMASK / YSAMP / HR)
OUT\\keep\\<name>\\: dec_anchors.png, ymask.png, ysamp.png, anchors_300k_sidecar.bin (the container's own anchors);
OUT\\<split>\\<name>.npz: 32 seeded 128 px crops (dec, gt) at the positions train_enh_fresh.val_set rebuilds.
The container is not kept. Resumable; WORKER/NWORK processes share the list. One line per photo in OUT\\log_<worker>.csv.

usage: python Params-test/gen_enh_prod.py   env OUT, ENH_RATIO (0.9), ENH_RATIOS (e.g. "0.3 0.5 0.7 0.9": mixed-ratio set,
       overrides ENH_RATIO), ENH_RANK_NET, ENH_DEC_PROX (decode with this learned ADMM prox), ENH_DEC_PROX_OFF=1
       (decode with the identity prox: no NLM), ENH_DEC_ARGS (extra
       decode arguments, e.g. "--sr-pnp-iters 2"), ENH_ALL_PHONES=1 (every phone of the dataset, see job_list),
       ENH_SKIP_DATA ("a;b": skip photos those sets already hold), ENH_ANCHORS (--hr-detail count, 300000; the dump keeps
       the name anchors_300k_sidecar.bin, which the trainer reads), WORKER, NWORK, LIMIT"""
import importlib.util, os, subprocess, sys, time, zlib
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, r"x64\GpuBuild")
EXE = os.path.join(G, "ImgReconstruct_backend.exe")
OUT = os.environ.setdefault("OUT", r"C:\Users\iorda\AppData\Local\enh_data_prod_r09")   # outside OneDrive on purpose
RATIO = float(os.environ.get("ENH_RATIO", 0.9))
RATIOS = [float(r) for r in os.environ.get("ENH_RATIOS", "").split()]   # set: one of these per photo, by crc32 of its name
RANK_NET = os.environ.get("ENH_RANK_NET", os.path.join(ROOT, r"models\enh\enh_seg8a_x5_ema256_fresh_unet.onnx"))
PW = "enh-prod-train-1234"
CROP, N_CROPS = 128, 32
WORKER, NWORK = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORK", 1))
ENV = dict(os.environ, CS_GPU_TILES="1", CS_GPU_ADMM="1")
ENV.pop("CS_ADMM_PROX", None)                         # the encode never uses the learned prox
ENV.pop("CS_ADMM_PROX_OFF", None)                     # ... nor the identity prox
DEC_PROX = os.environ.get("ENH_DEC_PROX", "")         # decode only: learned ADMM prox weights (CS_ADMM_PROX)
DEC_PROX_OFF = os.environ.get("ENH_DEC_PROX_OFF", "0") == "1"   # decode only: identity prox, no NLM (CS_ADMM_PROX_OFF)
DEC_ARGS = os.environ.get("ENH_DEC_ARGS", "").split()   # decode only: extra arguments, e.g. "--sr-pnp-iters 2"
ANCHORS = os.environ.get("ENH_ANCHORS", "300000")    # encode: --hr-detail anchor count
SKIP = [d for d in os.environ.get("ENH_SKIP_DATA", "").split(";") if d]   # sets whose photos are not redone here
spec = importlib.util.spec_from_file_location("gen_old", os.path.join(G, "gen_enh_seg8tile_anchor.py"))
gen_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen_old)   # job_list only (creates OUT dirs)
SCR = OUT + rf"\_scratch{WORKER}"


def run(args, env, log):
    with open(log, "w") as f:
        return subprocess.run([EXE] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800).returncode


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))


def one_photo(split, name, f, log):
    keep = OUT + rf"\keep\{name}"; os.makedirs(keep, exist_ok=True)
    enc = SCR + r"\container.png"
    ratio = RATIOS[zlib.crc32(name.encode()) % len(RATIOS)] if RATIOS else RATIO
    t0 = time.time()
    rc = run(["encrypt", f, enc, "--password", PW, "--no-preview", "--ratio", str(ratio), "--seg-auto", "8", "--ycc420",
              "--seg-tile", "--hr-detail", ANCHORS, "--enhance", RANK_NET], ENV, SCR + r"\enc.log")
    if rc != 0 or not os.path.exists(enc):
        print("encode failed", name, rc, flush=True); return
    t1 = time.time()
    env = dict(ENV, CS_HR_NO_POSTHOC="1", CS_DUMP_YMASK=keep + r"\ymask.png", CS_DUMP_YSAMP=keep + r"\ysamp.png",
               CS_DUMP_HR=keep + r"\anchors_300k_sidecar.bin", **({"CS_ADMM_PROX": DEC_PROX} if DEC_PROX else {}),
               **({"CS_ADMM_PROX_OFF": "1"} if DEC_PROX_OFF else {}))
    dec_path = keep + r"\dec_anchors.png"
    rc = run(["decrypt", enc, dec_path, "--password", PW, "--no-preview", "--tv", "0", "--photo-upscaler", "cs-sr",
              "--sr-pnp", "admm", "--hr-weight", "1", "--device", "gpu", *DEC_ARGS], env, SCR + r"\dec.log")
    O = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
    D = np.asarray(Image.open(dec_path).convert("RGB")) if rc == 0 and os.path.exists(dec_path) else None
    if D is None or D.shape != O.shape or not all(os.path.exists(keep + "\\" + x) for x in
                                                  ("ymask.png", "ysamp.png", "anchors_300k_sidecar.bin")):
        print("decode failed / shape mismatch", name, rc, None if D is None else D.shape, O.shape, flush=True); return
    H, W, _ = O.shape
    r = np.random.default_rng(zlib.crc32(name.encode()))
    ys = r.integers(0, H - CROP, N_CROPS); xs = r.integers(0, W - CROP, N_CROPS)
    np.savez_compressed(OUT + rf"\{split}\{name}.npz", dec=np.stack([D[y:y + CROP, x:x + CROP] for y, x in zip(ys, xs)]),
                        gt=np.stack([O[y:y + CROP, x:x + CROP] for y, x in zip(ys, xs)]))
    p = psnr(O, D)
    log.write(f"{split},{name},{ratio},{W},{H},{os.path.getsize(enc)},{p:.3f},{t1 - t0:.1f},{time.time() - t1:.1f}\n"); log.flush()
    return p, t1 - t0, time.time() - t1


def main():
    os.makedirs(SCR, exist_ok=True)
    jobs = [j for j in gen_old.job_list() if not any(os.path.exists(d + rf"\{j[0]}\{j[1]}_{os.path.splitext(os.path.basename(j[2]))[0]}.npz")
                                                     for d in SKIP)][WORKER::NWORK]
    print(f"worker {WORKER}/{NWORK}: {len(jobs)} photos, {sum(j[0] == 'val' for j in jobs)} validation -> {OUT}", flush=True)
    log = open(OUT + rf"\log_{WORKER}.csv", "a")
    t0, n_done = time.time(), 0
    for n, (split, p, f) in enumerate(jobs):
        name = f"{p}_{os.path.splitext(os.path.basename(f))[0]}"
        if os.path.exists(OUT + rf"\{split}\{name}.npz"):
            continue
        try:
            res = one_photo(split, name, f, log)
        except Exception as e:
            print("ERR", name, e, flush=True); continue
        if res:
            n_done += 1
            el = time.time() - t0
            print(f"[{n + 1}/{len(jobs)}] {split} {name} {res[0]:.2f} dB enc {res[1]:.0f}s dec {res[2]:.0f}s  "
                  f"elapsed {el / 3600:.2f} h, ~{el / n_done * (len(jobs) - n - 1) / 3600:.1f} h left", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
