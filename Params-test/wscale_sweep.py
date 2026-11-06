"""wscale x coef sweep of the NLM-off decoder (identity prox, 2 ADMM rounds), scored after the enhance net.

The DCT threshold is coef * w(r,c) / 2 with w the DC-to-peak ramp 1..wscale; CS_COEF_MULT replaces the decoder's x0.2
NLM-off factor. Containers are the 8 ratio_sweep photos with 450k anchors (the mix450 training recipe), decoded with
the net trained on that recipe. PSNR, SSIM and CIEDE2000 p95/p99 go to wscale_sweep_results.csv in the repo root
(resumable); ratio_sweep.metrics deletes each decode once scored.

usage: python Params-test/wscale_sweep.py   env RATIOS ("0.3 0.9"), WSCALES ("1 1.5 2 3 4"), MULTS ("0.1 0.2 0.4"),
       NET (models/enh/enh_seg8a_x5_ema256_mix450c02_c32_unet.onnx)"""
import csv, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_sweep as rs

CSV = os.path.join(rs.ROOT, "wscale_sweep_results.csv")
OUTD = os.path.join(rs.WORK, "wsweep")
FIELDS = ["img", "ratio", "wscale", "mult", "psnr", "ssim", "de95", "de99"]
ANCHORS = 450000


def main():
    ratios = [float(r) for r in os.environ.get("RATIOS", "0.3 0.9").split()]
    wscales = os.environ.get("WSCALES", "1 1.5 2 3 4").split()
    mults = os.environ.get("MULTS", "0.1 0.2 0.4").split()
    net = os.environ.get("NET", os.path.join(rs.ROOT, r"models\enh\enh_seg8a_x5_ema256_mix450c02_c32_unet.onnx"))
    os.makedirs(OUTD, exist_ok=True)
    done = {}
    if os.path.exists(CSV):
        done = {tuple(r[k] for k in FIELDS[:4]): r for r in csv.DictReader(open(CSV))}
    else:
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    pool, futs = ProcessPoolExecutor(max_workers=6), []
    for ratio in ratios:
        for name in rs.PHOTOS:
            enc, _ = rs.encode(name, ratio, ANCHORS)
            if not enc:
                continue
            for ws in wscales:
                for mu in mults:
                    key = (name, f"{ratio:.2f}", ws, mu)
                    if key in done:
                        continue
                    out = os.path.join(OUTD, f"{name}_r{ratio:.2f}_w{ws}_m{mu}.dec.png")
                    env = dict(os.environ, CS_GPU_TILES="1", CS_GPU_ADMM="1", CS_ADMM_PROX_OFF="1", CS_COEF_MULT=mu)
                    cmd = [rs.EXE, "decrypt", enc, out, "--password", rs.PW, "--no-preview", *rs.DEC_BASE, "--tv", "0",
                           "--hr-weight", "1", "--sr-pnp-iters", "2", "--wscale", ws, "--enhance", net]
                    if subprocess.run(cmd, cwd=rs.ROOT, env=env, capture_output=True).returncode or not os.path.exists(out):
                        print("DECODE FAILED", key, flush=True); continue
                    futs.append((key, pool.submit(rs.metrics, rs.src_png(name), out)))
            print(f"{name} r{ratio}: queued", flush=True)
    for key, f in futs:
        m = f.result()
        done[key] = dict(zip(FIELDS, list(key) + [f"{x:.4f}" for x in m]))
        with open(CSV, "a", newline="") as fh:
            csv.writer(fh).writerow(list(key) + [f"{x:.4f}" for x in m])
    pool.shutdown()
    for ratio in ratios:
        r = f"{ratio:.2f}"
        ref = [done.get((i, r, "2", "0.2")) for i in rs.PHOTOS]
        print(f"\nr{r}: mean over photos, delta vs wscale 2 / x0.2 (PSNR {np.mean([float(x['psnr']) for x in ref if x]):.3f})")
        print("wscale  mult   dPSNR    dSSIM    dDE95   dDE99")
        for ws in wscales:
            for mu in mults:
                d = {k: [float(done[(i, r, ws, mu)][k]) - float(b[k]) for i, b in zip(rs.PHOTOS, ref)
                         if b and (i, r, ws, mu) in done] for k in ("psnr", "ssim", "de95", "de99")}
                if d["psnr"]:
                    print(f"{ws:>6} {mu:>5} {np.mean(d['psnr']):+8.3f} {np.mean(d['ssim']):+8.4f} "
                          f"{np.mean(d['de95']):+8.3f} {np.mean(d['de99']):+7.3f}")


if __name__ == "__main__":
    main()
