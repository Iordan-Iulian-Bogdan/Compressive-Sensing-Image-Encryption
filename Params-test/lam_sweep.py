"""Global lambda multiplier sweep (CS_COEF_MULT) on the production decode.

tile_tune.py found x0.5 (its grid edge) best for the whole image at every ratio;
this extends the grid below it and scores PSNR, SSIM and CIEDE2000 p95/p99 so a
retune is promoted only if SSIM and dE-p95 agree. Appends to
lam_sweep_results.csv in the repo root (resumable).

Usage: python Params-test/lam_sweep.py [mult ...]   (default 0.2 0.3 0.4 0.5 1.0)
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_sweep as rs
import tile_tune as tt

CSV = os.path.join(rs.ROOT, "lam_sweep_results.csv")
RATIOS = [0.3, 0.5, 0.7, 0.9]


def main():
    mults = [float(a) for a in sys.argv[1:]] or [0.2, 0.3, 0.4, 0.5, 1.0]
    done = set()
    if os.path.exists(CSV):
        done = {(r["img"], r["ratio"], r["mult"]) for r in csv.DictReader(open(CSV))}
    else:
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(["img", "ratio", "mult", "psnr", "ssim", "de95", "de99"])
    os.makedirs(tt.TT, exist_ok=True)
    pool, futs = ProcessPoolExecutor(max_workers=8), []
    for ratio in RATIOS:
        for name in rs.PHOTOS:
            for m in mults:
                key = (name, f"{ratio:.2f}", f"{m:.2f}")
                if key in done:
                    continue
                out = os.path.join(tt.TT, f"lam_{name}_r{ratio:.2f}_m{m:.2f}.dec.png")
                if not tt.decode(rs.container(name, ratio, 300000), out, [], {"CS_COEF_MULT": str(m)}):
                    print(f"  DECODE FAILED {key}", flush=True)
                    continue
                os.remove(out + ".log")
                futs.append((key, pool.submit(rs.metrics, rs.src_png(name), out)))
    for key, f in futs:
        with open(CSV, "a", newline="") as fh:
            csv.writer(fh).writerow(list(key) + [f"{x:.4f}" for x in f.result()])
    pool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
