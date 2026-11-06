"""Per-ratio settings sweep for the production seg8-tile + anchors pipeline.

For every (photo, ratio) it encodes with a few anchor budgets and decodes with
one-factor-at-a-time variations of the production decoder, scoring PSNR, SSIM
and CIEDE2000 p95/p99 (same code as metrics.py). Metrics run in a process pool
while the GPU decodes. Results append to ratio_sweep_results.csv in the repo
root (resumable: done rows are skipped). Containers and decodes go to
%LOCALAPPDATA%\\ratio_sweep (not OneDrive); only the 300k containers are kept.

Usage: python Params-test/ratio_sweep.py [stage]   (stage: 1 = OFAT grid, 2 = combos from stage2.json)
"""
import csv, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, r"x64\GpuBuild\ImgReconstruct_backend.exe")
NET = os.path.join(ROOT, r"models\enh\enh_seg8a_x5_ema256_fresh_unet.onnx")
WORK = os.path.join(os.environ["LOCALAPPDATA"], "ratio_sweep")
CSV = os.path.join(ROOT, "ratio_sweep_results.csv")
PW = "ratio-sweep-1234"
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
RATIOS = [round(0.2 + 0.05 * i, 2) for i in range(16)]  # 0.20 .. 0.95

ENC_BASE = ["--seg-auto", "8", "--ycc420", "--seg-tile"]
DEC_BASE = ["--photo-upscaler", "cs-sr", "--sr-pnp", "admm", "--device", "gpu"]

# stage 1: encoder anchor budgets x decoder variants (one factor at a time)
DEC_VARIANTS = {
    "base":     ["--tv", "0", "--hr-weight", "1", "--enhance", NET],
    "nonet":    ["--tv", "0", "--hr-weight", "1"],
    "tv005":    ["--tv", "0.005", "--hr-weight", "1", "--enhance", NET],
    "tv01":     ["--tv", "0.01", "--hr-weight", "1", "--enhance", NET],
    "tv02":     ["--tv", "0.02", "--hr-weight", "1", "--enhance", NET],
    "hrw05":    ["--tv", "0", "--hr-weight", "0.5", "--enhance", NET],
    "hrw2":     ["--tv", "0", "--hr-weight", "2", "--enhance", NET],
    "fast":     ["--tv", "0", "--hr-weight", "1", "--enhance", NET, "--fast"],
}
STAGE1 = {  # anchors -> decoder variants
    300000: list(DEC_VARIANTS),
    0: ["base", "nonet"],
    150000: ["base", "nonet"],
    600000: ["base", "nonet"],
}


def metrics(orig_path, dec_path):
    from skimage.metrics import structural_similarity as ssim
    from skimage.color import rgb2lab, deltaE_ciede2000
    a = np.asarray(Image.open(orig_path).convert("RGB")).astype(np.float64)
    b = np.asarray(Image.open(dec_path).convert("RGB")).astype(np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    mse = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    s = ssim(a / 255.0, b / 255.0, channel_axis=2, data_range=1.0)
    de = deltaE_ciede2000(rgb2lab(a / 255.0), rgb2lab(b / 255.0))
    return psnr, s, float(np.quantile(de, 0.95)), float(np.quantile(de, 0.99))


def run(args, log):
    env = dict(os.environ, CS_GPU_TILES="1", CS_GPU_ADMM="1")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([EXE] + args, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
    return rc, time.time() - t0


def src_png(name):
    p = PHOTOS[name]
    if p.endswith(".png"):
        return p
    out = os.path.join(WORK, "src", name + ".png")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.open(p).convert("RGB").save(out)
    return out


def container(name, ratio, anchors):
    return os.path.join(WORK, "enc", f"{name}_r{ratio:.2f}_a{anchors // 1000}k.enc.png")


def encode(name, ratio, anchors):
    enc = container(name, ratio, anchors)
    if os.path.exists(enc):
        return enc, 0.0
    os.makedirs(os.path.dirname(enc), exist_ok=True)
    args = ["encrypt", src_png(name), enc, "--password", PW, "--ratio", str(ratio), "--no-preview"] + ENC_BASE
    if anchors:
        args += ["--hr-detail", str(anchors), "--enhance", NET]
    rc, dt = run(args, enc + ".log")
    if rc != 0 or not os.path.exists(enc):
        print(f"  ENCODE FAILED {name} r{ratio} a{anchors}: rc={rc}", flush=True)
        return None, dt
    return enc, dt


def main(stage):
    os.makedirs(os.path.join(WORK, "dec"), exist_ok=True)
    fields = ["stage", "img", "ratio", "anchors", "dec", "mb", "enc_s", "dec_s", "psnr", "ssim", "de95", "de99"]
    done = set()
    if os.path.exists(CSV):
        with open(CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["stage"], r["img"], r["ratio"], r["anchors"], r["dec"]))
    else:
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    if stage == "1":
        jobs = [(n, r, a, v) for r in RATIOS for n in PHOTOS for a, vs in STAGE1.items() for v in vs]
        variants = {v: DEC_VARIANTS[v] for v in DEC_VARIANTS}
    else:  # stage 2: {"ratio": {"name": [anchors, [decoder args]]}}
        spec = json.load(open(os.path.join(ROOT, "Params-test", "ratio_sweep_stage2.json")))
        jobs, variants = [], {}
        for r, combos in spec.items():
            for v, (a, dargs) in combos.items():
                variants[v] = [x.replace("NET", NET) for x in dargs]
                jobs += [(n, float(r), a, v) for n in PHOTOS]
    jobs = [j for j in jobs if (stage, j[0], f"{j[1]:.2f}", str(j[2]), j[3]) not in done]
    print(f"stage {stage}: {len(jobs)} decodes to run", flush=True)

    pool = ProcessPoolExecutor(max_workers=10)
    pending = []

    def flush(block):
        keep = []
        for fut, row in pending:
            if block or fut.done():
                try:
                    row += [f"{x:.4f}" for x in fut.result()]
                except Exception as e:
                    print(f"  METRICS FAILED {row[:5]}: {e}", flush=True)
                    row += ["", "", "", ""]
                for _ in range(60):  # OneDrive sync can lock the file briefly
                    try:
                        with open(CSV, "a", newline="") as f:
                            csv.writer(f).writerow(row)
                        break
                    except PermissionError:
                        time.sleep(1)
            else:
                keep.append((fut, row))
        pending[:] = keep

    t_start = time.time()
    for i, (name, ratio, anchors, v) in enumerate(jobs):
        enc, enc_s = encode(name, ratio, anchors)
        if enc is None:
            continue
        dec = os.path.join(WORK, "dec", f"{name}_r{ratio:.2f}_a{anchors // 1000}k_{v}.dec.png")
        args = ["decrypt", enc, dec, "--password", PW, "--no-preview"] + DEC_BASE + variants[v]
        rc, dec_s = run(args, dec + ".log")
        if rc != 0 or not os.path.exists(dec):
            print(f"  DECODE FAILED {name} r{ratio} a{anchors} {v}: rc={rc}", flush=True)
            continue
        mb = os.path.getsize(enc) / 1e6
        row = [stage, name, f"{ratio:.2f}", str(anchors), v, f"{mb:.3f}", f"{enc_s:.2f}", f"{dec_s:.2f}"]
        pending.append((pool.submit(metrics, src_png(name), dec), row))
        while len(pending) > 20:  # keep disk and RAM bounded
            flush(False)
            time.sleep(0.2)
        flush(False)
        if i % 20 == 0:
            el = time.time() - t_start
            print(f"[{i + 1}/{len(jobs)}] {name} r{ratio:.2f} a{anchors} {v}  {el / 60:.1f} min, "
                  f"eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min", flush=True)
        # drop non-production containers once their last decode is queued
        if anchors != 300000 and stage == "1" and v == STAGE1[anchors][-1]:
            os.remove(enc)
    flush(True)
    pool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
