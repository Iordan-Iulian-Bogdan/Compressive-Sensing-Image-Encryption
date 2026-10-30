"""Per-tile tuning headroom of the decoder's FISTA / PnP-ADMM parameters.

For every (photo, ratio) it decodes the ratio_sweep production containers
(seg8-tile + 300k anchors) with the production decoder plus one-factor
variants: lambda x{0.5,0.7,1.4,2} (CS_COEF_MULT, stays in auto mode), wscale
{1,1.5,3}, PnP rho {0.05,0.2,0.4} and --per-tile-coef. Each output is scored
per 64x64 block (one ~32 px LR solve tile) against the original. The oracle
picks the best variant per block; the gap between that and the best single
variant for the whole image is what per-tile tuning could add over retuning
the global law. Upper bound: hard block seams, selection on the original.

Per-block SSEs and the base decode's block detail go to
%LOCALAPPDATA%\\ratio_sweep\\tt\\<photo>_r<ratio>.npz (resumable); the summary
appends to tile_tune_results.csv in the repo root.

Usage: python Params-test/tile_tune.py [ratio ...]   (default 0.3 0.5 0.7 0.9)
"""
import csv, os, subprocess, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_sweep as rs

B = 64
TT = os.path.join(rs.WORK, "tt")
CSV = os.path.join(rs.ROOT, "tile_tune_results.csv")
BASE_ARGS = rs.DEC_BASE + ["--tv", "0", "--hr-weight", "1", "--enhance", rs.NET]
VARIANTS = {"base": ([], {})}
for m in (0.5, 0.7, 1.4, 2.0):
    VARIANTS[f"lam{m}"] = ([], {"CS_COEF_MULT": str(m)})
for w in (1.0, 1.5, 3.0):
    VARIANTS[f"ws{w}"] = (["--wscale", str(w)], {})
for r in (0.05, 0.2, 0.4):
    VARIANTS[f"rho{r}"] = (["--sr-pnp-rho", str(r)], {})
VARIANTS["ptc"] = (["--per-tile-coef"], {})
FAMILIES = {"lam": "lam", "ws": "ws", "rho": "rho", "ptc": "ptc", "all": ""}


def decode(enc, out, args, env):
    e = dict(os.environ, CS_GPU_TILES="1", CS_GPU_ADMM="1", **env)
    cmd = [rs.EXE, "decrypt", enc, out, "--password", rs.PW, "--no-preview"] + BASE_ARGS + args
    with open(out + ".log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=rs.ROOT).returncode
    return rc == 0 and os.path.exists(out)


def blocks_sse(a, b):
    h, w = a.shape[0] // B * B, a.shape[1] // B * B
    d = (a[:h, :w].astype(np.float32) - b[:h, :w].astype(np.float32)) ** 2
    return d.reshape(h // B, B, w // B, B, 3).sum(axis=(1, 3, 4))


def block_detail(img):
    y = img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:])
    lap = np.pad(lap, 1)
    h, w = y.shape[0] // B * B, y.shape[1] // B * B
    return lap[:h, :w].reshape(h // B, B, w // B, B).mean(axis=(1, 3))


def composite(imgs, pick):
    out = imgs[0].copy()
    for k, im in enumerate(imgs):
        m = np.kron(pick == k, np.ones((B, B), bool))
        out[:m.shape[0], :m.shape[1]][m] = im[:m.shape[0], :m.shape[1]][m]
    return out


def ssim_de(a, b):
    from skimage.metrics import structural_similarity as ssim
    from skimage.color import rgb2lab, deltaE_ciede2000
    a, b = a / 255.0, b / 255.0
    s = ssim(a, b, channel_axis=2, data_range=1.0)
    de = deltaE_ciede2000(rgb2lab(a), rgb2lab(b))
    return s, float(np.quantile(de, 0.95))


def psnr(sse, npx):
    return 10 * np.log10(255.0 ** 2 * npx * 3 / sse)


def run_group(name, ratio):
    npz = os.path.join(TT, f"{name}_r{ratio:.2f}.npz")
    if os.path.exists(npz):
        return None
    enc = rs.container(name, ratio, 300000)
    if not os.path.exists(enc):
        print(f"  missing container {enc}", flush=True)
        return None
    orig = np.asarray(Image.open(rs.src_png(name)).convert("RGB"))
    names, imgs, sse, secs = [], [], [], []
    for v, (args, env) in VARIANTS.items():
        out = os.path.join(TT, f"{name}_r{ratio:.2f}_{v}.dec.png")
        t0 = time.time()
        if not decode(enc, out, args, env):
            print(f"  DECODE FAILED {name} r{ratio} {v}", flush=True)
            continue
        secs.append(time.time() - t0)
        im = np.asarray(Image.open(out).convert("RGB"))
        os.remove(out)
        os.remove(out + ".log")
        names.append(v); imgs.append(im); sse.append(blocks_sse(orig, im))
    sse = np.stack(sse)
    det = block_detail(imgs[0])
    np.savez(npz, names=np.array(names), sse=sse, detail=det)
    npx = sse.shape[1] * sse.shape[2] * B * B
    tot = sse.reshape(len(names), -1).sum(axis=1)
    row = {"img": name, "ratio": f"{ratio:.2f}", "base_psnr": f"{psnr(tot[0], npx):.3f}",
           "dec_s": f"{np.mean(secs):.2f}"}
    for fam, pre in FAMILIES.items():
        idx = [0] + [k for k, v in enumerate(names) if k and v.startswith(pre)]
        g = int(idx[np.argmin(tot[idx])])
        pick = np.array(idx)[np.argmin(sse[idx], axis=0)]
        orc = np.take_along_axis(sse, pick[None], axis=0).sum()
        row[f"{fam}_global"] = names[g]
        row[f"{fam}_global_gain"] = f"{psnr(tot[g], npx) - psnr(tot[0], npx):.3f}"
        row[f"{fam}_tile_gain"] = f"{psnr(orc, npx) - psnr(tot[0], npx):.3f}"
        if fam == "all":
            s0, d0 = ssim_de(orig.astype(np.float64), imgs[0].astype(np.float64))
            s1, d1 = ssim_de(orig.astype(np.float64), composite(imgs, pick).astype(np.float64))
            row.update(base_ssim=f"{s0:.4f}", orc_ssim=f"{s1:.4f}", base_de95=f"{d0:.3f}", orc_de95=f"{d1:.3f}")
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def main():
    ratios = [float(a) for a in sys.argv[1:]] or [0.3, 0.5, 0.7, 0.9]
    os.makedirs(TT, exist_ok=True)
    t0 = time.time()
    for ratio in ratios:
        for name in rs.PHOTOS:
            row = run_group(name, ratio)
            if row:
                print(f"{name:12s} r{ratio:.2f} base {row['base_psnr']} dB | global/tile gain: "
                      + " ".join(f"{f} {row[f + '_global_gain']}/{row[f + '_tile_gain']}" for f in FAMILIES)
                      + f" | SSIM {row['base_ssim']}->{row['orc_ssim']} dE95 {row['base_de95']}->{row['orc_de95']}"
                      + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
