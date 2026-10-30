"""Can a decoder-side rule capture the per-tile oracle of tile_tune.py?

Rule: bin each 64x64 block by the detail (mean |Laplacian|) of the base decode,
which the decoder has, and use per (ratio, bin) the variant with the least
summed SSE on the other photos. Scored leave-one-photo-out against
(a) the base and (b) a global retune: the single variant best on the other
photos at that ratio. Gains are mean dB over the held-out photos.

Usage: python Params-test/tile_tune_predict.py [bins]   (default 4)
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_tune as tt

FAMS = {"lam": "lam", "ws": "ws", "rho": "rho", "all": ""}


def load():
    data = {}
    for p in glob.glob(os.path.join(tt.TT, "*.npz")):
        name, r = os.path.basename(p)[:-4].rsplit("_r", 1)
        z = np.load(p)
        data.setdefault(float(r), {})[name] = (list(z["names"]), z["sse"], z["detail"])
    return data


def psnr_gain(sse_sel, sse_base):
    return 10 * np.log10(sse_base / sse_sel)


def main():
    nbins = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    data = load()
    for r in sorted(data):
        photos = data[r]
        print(f"ratio {r:.2f} ({len(photos)} photos)")
        for fam, pre in FAMS.items():
            g_or, g_glob, g_rule = [], [], []
            for held in photos:
                names, sse, det = photos[held]
                idx = [0] + [k for k, v in enumerate(names) if k and v.startswith(pre)]
                train = [p for p in photos if p != held and photos[p][0] == names]
                if not train:
                    continue
                base = sse[0].sum()
                g_or.append(psnr_gain(np.min(sse[idx], axis=0).sum(), base))
                # global retune
                tot = sum(photos[p][1][idx].reshape(len(idx), -1).sum(axis=1) / photos[p][1][0].sum()
                          for p in train)
                g_glob.append(psnr_gain(sse[idx[int(np.argmin(tot))]].sum(), base))
                # detail-binned rule, thresholds from the training blocks
                tdet = np.concatenate([photos[p][2].ravel() for p in train])
                edges = np.quantile(tdet, np.linspace(0, 1, nbins + 1)[1:-1])
                choice = []
                for b in range(nbins):
                    acc = np.zeros(len(idx))
                    for p in train:
                        _, s, d = photos[p]
                        m = np.digitize(d, edges) == b
                        acc += s[idx][:, m].sum(axis=1) / s[0].sum()
                    choice.append(idx[int(np.argmin(acc))])
                hb = np.digitize(det, edges)
                sel = sum(sse[choice[b]][hb == b].sum() for b in range(nbins))
                g_rule.append(psnr_gain(sel, base))
            if g_or:
                print(f"  {fam:4s} oracle {np.mean(g_or):+.3f}  global-retune {np.mean(g_glob):+.3f}  "
                      f"detail-rule {np.mean(g_rule):+.3f}  (rule worst photo {np.min(g_rule):+.3f})")


if __name__ == "__main__":
    main()
