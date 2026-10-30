"""Upper bound on what reference-tile ("side information") CS could save.

For every 64x64 luma tile at the pipeline's half resolution, find the best
already-decoded reference block (raster-causal) and compare the DCT sparsity of
the tile x with that of the difference d = x - ref. CS needs roughly
m ~ k*(1+ln(n/k)) samples (= n at k = n), with k the number of DCT coefficients that
reach the target error. A dependent tile costs min(m_x, m_d + side info), the
l1-l1 view: a bad reference never costs more than ignoring it.

Optimistic on purpose: the reference is the ORIGINAL block, not its noisy
reconstruction, and the side info is a flat 2 samples per dependent tile.

Usage: python Params-test/sideinfo_bound.py [psnr_db ...]   (default 38 33)
"""
import os, sys
import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = r"G:\Download\archive\SCMI30-IITRPR"
PHOTOS = {
    "IMG_3690": os.path.join(ROOT, r"ImgReconstruct_backend\IMG_3690.png"),
    "IMG_0007": os.path.join(ROOT, r"ImgReconstruct_backend\IMG_0007.png"),
    "IMG_1297": os.path.join(ROOT, r"ImgReconstruct_backend\IMG_1297.png"),
    "IMG_9321": os.path.join(ROOT, r"ImgReconstruct_backend\IMG_9321.png"),
    "D28_rnd_123": DS + r"\Random\D28_Apple_Iphone_13\D28_rnd_123.jpg",
    "D06_rnd_39": DS + r"\Random\D06_Apple_Iphone_12Mini\D06_rnd_39.jpg",
    "D17_rnd_71": DS + r"\Random\D17_Apple_Iphone_12\D17_rnd_71.jpg",
    "D17_nat_18": DS + r"\Similar\D17_Apple_Iphone_12\natural\D17_nat_18.jpg",
}
T, R, N = 64, 64, 64 * 64  # tile, shift search radius, samples per tile
SIDE = 2.0                   # side-info cost in samples (ref index + offset)


def luma_half(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((im.width // 2, im.height // 2), Image.BOX)
    a = np.asarray(im, dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def samples(block, eps):
    """CS samples for `block` at RMS error eps: k*(1+ln(N/k)), monotone, = N at k = N."""
    e = np.sort(dctn(block, norm="ortho").ravel() ** 2)  # ascending
    tail = np.cumsum(e)                                  # energy of the smallest j coefs
    dropped = np.searchsorted(tail, eps * eps * N, side="right")
    k = max(N - dropped, 1)
    return min(float(N), k * (1 + np.log(N / k)))


def best_shifted(Y, ty, tx):
    """Best causal block (any pixel offset within +-R) by SSD, or None."""
    y0, x0 = max(ty - R, 0), max(tx - R, 0)
    y1, x1 = min(ty + T, Y.shape[0]), min(tx + T + R, Y.shape[1])
    W, tile = Y[y0:y1, x0:x1], Y[ty:ty + T, tx:tx + T]
    if W.shape[0] < T or W.shape[1] < T:
        return None
    ones = np.ones((T, T))
    w2 = fftconvolve(W * W, ones, mode="valid")
    wt = fftconvolve(W, tile[::-1, ::-1], mode="valid")
    ssd = w2 - 2 * wt + np.sum(tile * tile)
    ys = np.arange(ssd.shape[0])[:, None] + y0
    xs = np.arange(ssd.shape[1])[None, :] + x0
    # every pixel of the candidate must already be decoded (raster order)
    ok = (ys <= ty - T) | ((ys <= ty) & (xs <= tx - T))
    if not ok.any():
        return None
    ssd = np.where(ok, ssd, np.inf)
    i, j = np.unravel_index(np.argmin(ssd), ssd.shape)
    return Y[y0 + i:y0 + i + T, x0 + j:x0 + j + T]


def lax_candidates(Y, ty, tx, K=8):
    """Top-K causal blocks (+-R) by zero-mean normalized correlation, i.e.
    similar up to brightness and contrast."""
    y0, x0 = max(ty - R, 0), max(tx - R, 0)
    y1, x1 = min(ty + T, Y.shape[0]), min(tx + T + R, Y.shape[1])
    W, tile = Y[y0:y1, x0:x1], Y[ty:ty + T, tx:tx + T]
    if W.shape[0] < T or W.shape[1] < T:
        return []
    t0 = tile - tile.mean()
    ones = np.ones((T, T))
    s1 = fftconvolve(W, ones, mode="valid")
    s2 = fftconvolve(W * W, ones, mode="valid")
    num = fftconvolve(W, t0[::-1, ::-1], mode="valid")
    ncc = num / np.sqrt(np.maximum(s2 - s1 * s1 / N, 1e-9) * max(np.sum(t0 * t0), 1e-9))
    ys = np.arange(ncc.shape[0])[:, None] + y0
    xs = np.arange(ncc.shape[1])[None, :] + x0
    ok = (ys <= ty - T) | ((ys <= ty) & (xs <= tx - T))
    ncc = np.where(ok, ncc, -np.inf).ravel()
    top = np.argsort(ncc)[::-1][:K]
    top = top[np.isfinite(ncc[top])]
    w = x1 - x0 - T + 1
    return [Y[y0 + i // w:y0 + i // w + T, x0 + i % w:x0 + i % w + T] for i in top]


LOW = np.add.outer(np.arange(T), np.arange(T)) < 16  # low-frequency DCT band


def lax_cost(tile, cands, eps):
    """Cheapest of: a*ref+b, and the low band of a*ref+b only (+3 side info)."""
    best = np.inf
    for ref in cands:
        A = np.stack([ref.ravel(), np.ones(N)], axis=1)
        a, b = np.linalg.lstsq(A, tile.ravel(), rcond=None)[0]
        pred = a * ref + b
        best = min(best, samples(tile - pred, eps))
        c = dctn(tile, norm="ortho") - np.where(LOW, dctn(pred, norm="ortho"), 0.0)
        best = min(best, samples_coefs(c, eps))
    return best + SIDE + 1


def samples_coefs(c, eps):
    e = np.sort(c.ravel() ** 2)
    dropped = np.searchsorted(np.cumsum(e), eps * eps * N, side="right")
    k = max(N - dropped, 1)
    return min(float(N), k * (1 + np.log(N / k)))


def analyse(Y, eps_list):
    th, tw = Y.shape[0] // T, Y.shape[1] // T
    tiles = [Y[i * T:(i + 1) * T, j * T:(j + 1) * T] for i in range(th) for j in range(tw)]
    flat = np.stack([t.ravel() for t in tiles])
    # detail = mean |Laplacian| (same signal the encoder's rate law uses)
    det = np.array([np.mean(np.abs(4 * t[1:-1, 1:-1] - t[:-2, 1:-1] - t[2:, 1:-1]
                                   - t[1:-1, :-2] - t[1:-1, 2:])) for t in tiles])
    # aligned: best earlier grid tile, any distance (Gram-matrix SSD)
    sq = np.sum(flat * flat, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * flat @ flat.T
    shifted = [best_shifted(Y, (n // tw) * T, (n % tw) * T) for n in range(len(tiles))]
    lax = [lax_candidates(Y, (n // tw) * T, (n % tw) * T) for n in range(len(tiles))]
    out = {}
    for eps in eps_list:
        mx = np.array([samples(t, eps) for t in tiles])
        res = {"lax": np.minimum(mx, [lax_cost(t, c, eps) for t, c in zip(tiles, lax)])}
        for mode in ("aligned", "shifted"):
            md = np.full(len(tiles), np.inf)
            for n, t in enumerate(tiles):
                if mode == "aligned":
                    if n == 0:
                        continue
                    ref = tiles[int(np.argmin(d2[n, :n]))]
                else:
                    ref = shifted[n]
                    if ref is None:
                        continue
                md[n] = samples(t - ref, eps) + SIDE
            res[mode] = np.minimum(mx, md)
        out[eps] = (mx, res, det)
    return out


def main():
    dbs = [float(a) for a in sys.argv[1:]] or [38.0, 33.0]
    eps_list = [255.0 / 10 ** (db / 20) for db in dbs]
    rows = []
    for name, path in PHOTOS.items():
        if not os.path.exists(path):
            print(f"skip {name}: missing {path}")
            continue
        Y = luma_half(path)
        for db, (eps, (mx, res, det)) in zip(dbs, analyse(Y, eps_list).items()):
            terc = np.quantile(det, [1 / 3, 2 / 3])
            band = np.digitize(det, terc)  # 0 flat, 1 mid, 2 textured
            line = {"photo": name, "psnr": db, "tiles": len(mx),
                    "base_rate": mx.sum() / (N * len(mx))}
            for mode, mc in res.items():
                line[f"{mode}_save"] = 1 - mc.sum() / mx.sum()
                line[f"{mode}_dep"] = np.mean(mc < mx)  # tiles that use a reference
                for b, bn in enumerate(("flat", "mid", "tex")):
                    m = band == b
                    line[f"{mode}_{bn}"] = (mx[m].sum() - mc[m].sum()) / mx.sum()
            rows.append(line)
            print(f"{name:12s} {db:4.0f}dB rate={line['base_rate']:.3f} "
                  f"aligned save={line['aligned_save']:6.1%} (dep {line['aligned_dep']:5.1%}) "
                  f"shifted save={line['shifted_save']:6.1%} (dep {line['shifted_dep']:5.1%}) "
                  f"[shifted share flat/mid/tex: {line['shifted_flat']:5.1%} "
                  f"{line['shifted_mid']:5.1%} {line['shifted_tex']:5.1%}] "
                  f"lax save={line['lax_save']:6.1%} (dep {line['lax_dep']:5.1%})", flush=True)
    for db in dbs:
        r = [x for x in rows if x["psnr"] == db]
        if r:
            print(f"MEAN {db:.0f}dB: base rate {np.mean([x['base_rate'] for x in r]):.3f}, "
                  f"aligned save {np.mean([x['aligned_save'] for x in r]):.1%}, "
                  f"shifted save {np.mean([x['shifted_save'] for x in r]):.1%} "
                  f"(flat {np.mean([x['shifted_flat'] for x in r]):.1%} / "
                  f"mid {np.mean([x['shifted_mid'] for x in r]):.1%} / "
                  f"tex {np.mean([x['shifted_tex'] for x in r]):.1%}), "
                  f"lax save {np.mean([x['lax_save'] for x in r]):.1%} "
                  f"(flat {np.mean([x['lax_flat'] for x in r]):.1%} / "
                  f"mid {np.mean([x['lax_mid'] for x in r]):.1%} / "
                  f"tex {np.mean([x['lax_tex'] for x in r]):.1%})")


if __name__ == "__main__":
    main()
