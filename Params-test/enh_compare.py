"""Compare --enhance nets on the production decode of the 8 ratio_sweep photos (300k-anchor containers).

Each net decodes every container at the given ratios with the production decoder; PSNR, SSIM and CIEDE2000 p95/p99
go to enh_compare_results.csv in the repo root (resumable), and the mean change of each net against the first one
is printed per ratio.

usage: python Params-test/enh_compare.py <net.onnx> [<net.onnx> ...]   env RATIOS (default "0.3 0.5 0.7 0.9")"""
import csv, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_sweep as rs

CSV = os.path.join(rs.ROOT, "enh_compare_results.csv")
OUTD = os.path.join(rs.WORK, "enhcmp")


def main():
    nets = sys.argv[1:]
    ratios = [float(r) for r in os.environ.get("RATIOS", "0.3 0.5 0.7 0.9").split()]
    os.makedirs(OUTD, exist_ok=True)
    done = {}
    if os.path.exists(CSV):
        done = {(r["img"], r["ratio"], r["net"]): r for r in csv.DictReader(open(CSV))}
    else:
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(["img", "ratio", "net", "psnr", "ssim", "de95", "de99"])
    env = dict(os.environ, CS_GPU_TILES="1", CS_GPU_ADMM="1")
    pool, futs = ProcessPoolExecutor(max_workers=8), []
    for ratio in ratios:
        for name in rs.PHOTOS:
            for net in nets:
                key = (name, f"{ratio:.2f}", os.path.basename(net))
                if key in done:
                    continue
                out = os.path.join(OUTD, f"{name}_r{ratio:.2f}_{os.path.basename(net)}.dec.png")
                cmd = [rs.EXE, "decrypt", rs.container(name, ratio, 300000), out, "--password", rs.PW, "--no-preview",
                       *rs.DEC_BASE, "--tv", "0", "--hr-weight", "1", "--enhance", net]
                if subprocess.run(cmd, cwd=rs.ROOT, env=env, capture_output=True).returncode or not os.path.exists(out):
                    print("DECODE FAILED", key, flush=True); continue
                futs.append((key, pool.submit(rs.metrics, rs.src_png(name), out)))
    for key, f in futs:
        m = f.result()
        done[key] = dict(zip(["img", "ratio", "net", "psnr", "ssim", "de95", "de99"], list(key) + [f"{x:.4f}" for x in m]))
        with open(CSV, "a", newline="") as fh:
            csv.writer(fh).writerow(list(key) + [f"{x:.4f}" for x in m])
    pool.shutdown()
    ref = os.path.basename(nets[0])
    for ratio in ratios:
        r = f"{ratio:.2f}"
        for net in map(os.path.basename, nets[1:]):
            d = {k: [float(done[(i, r, net)][k]) - float(done[(i, r, ref)][k]) for i in rs.PHOTOS
                     if (i, r, net) in done and (i, r, ref) in done] for k in ("psnr", "ssim", "de95", "de99")}
            print(f"r{r} {net} vs {ref}: dPSNR {np.mean(d['psnr']):+.3f} (min {min(d['psnr']):+.3f})  "
                  f"dSSIM {np.mean(d['ssim']):+.4f} (min {min(d['ssim']):+.4f})  dDE95 {np.mean(d['de95']):+.3f} "
                  f"(worst {max(d['de95']):+.3f})  dDE99 {np.mean(d['de99']):+.3f}  n={len(d['psnr'])}")


if __name__ == "__main__":
    main()
