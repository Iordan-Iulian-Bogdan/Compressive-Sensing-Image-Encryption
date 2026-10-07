# Decrypt benchmark — IMG_3690 photo (2026-10-07)

21-config speed/quality sweep, `x64\Cpu` build, ratio 0.5, auto params,
24 threads. Times are wall decrypt (encrypt is ~0.3 s for all). The main
binary has no quality module, so PSNR/MAE only, no SSIM. Raw per-config
outputs are not committed (reproduce: `bench_sweep.ps1` pattern —
encrypt each sampling variant, decrypt, `compare`).

## Sampling modes (default decrypt)

| Container | PSNR | MAE | Decrypt | Container |
|---|---|---|---|---|
| random (baseline) | 26.80 | 6.11 | 6.1 s | 4.58 MB |
| adaptive | 27.19 | 6.09 | 6.0 s | 4.56 MB |
| two-pass | 26.59 | 6.13 | 6.2 s | 4.55 MB |
| spectral | 27.15 | 6.08 | 6.1 s | 4.56 MB |
| periodic | 14.10 (BROKEN, see below) | 33.0 | 6.2 s | 4.53 MB |
| ycc420 | 26.65 | 6.44 | 4.2 s | 2.17 MB |
| ycc422 | 26.69 | 6.36 | 4.9 s | 2.82 MB |
| hf-focus | 26.71 | 6.18 | 6.3 s | 4.58 MB |
| ycc420+hf | 26.56 | 6.50 | 4.3 s | 2.19 MB |
| ycc422+hf | 26.60 | 6.43 | 5.0 s | 2.84 MB |

## Solvers / upscalers (same baseline container)

| Decrypt | PSNR (Δ) | MAE | Time (×) |
|---|---|---|---|
| avir (baseline) | 26.80 | 6.11 | 6.2 s (1×) |
| joint | 27.06 (+0.26) | 5.85 | 8.4 s (1.4×) |
| wavelet | 26.50 (−0.30) | 5.48 | 36 s (5.9×) |
| joint+wavelet | 26.74 (−0.06) | 5.25 | 75 s (12×) |
| tv 0.01 | 26.70 (−0.10) | 5.77 | 41 s (6.6×) |
| fsrcnn | 26.80 (+0.00, no-op, see below) | 6.11 | 6.2 s (1×) |
| cs-sr | 27.17 (+0.37) | 5.69 | 19 s (3.1×) |
| cs-sr + red 0.5 | 27.29 (+0.49) | 5.45 | 20 s (3.3×) |
| cs-sr + cascade | 27.00 (+0.20) | 5.67 | 18 s (3.0×) |
| sr-direct | 27.20 (+0.40) | 6.09 | 18 s (2.9×) |
| sr-direct + cascade | 27.00 (+0.20) | 5.67 | 18 s (3.0×) |

## Takeaways

- Best quality: `cs-sr + red 0.5` (27.29 dB). `sr-direct` (cold zeros)
  is right behind (27.20); both cascade variants converge to 27.00 dB.
- Best speed/size: `ycc420` — 0.15 dB under baseline at 0.68× time and
  0.47× container. `ycc422` costs +0.65 MB for +0.04 dB. Adaptive/spectral
  LOD are free (+0.35–0.39 dB, same cost); two-pass loses here (−0.21).
- Not worth it here: wavelet (6×, −0.3 dB), TV backtracking (7×, −0.1 dB
  at 0.01), joint+wavelet (12×, −0.06). Plain `joint` is the only cheap
  solver win (+0.26 dB, 1.4×).

## Open issues (not fixed, recorded 2026-10-07)

1. **Periodic container collapse**: `--periodic` at ratio 0.5 decrypts to
   14.10 dB, reproduced bit-identically on re-run — deterministic
   decrypt-side mishandling of the periodic container (or a genuinely bad
   default `--tile-size` interaction at this ratio), not flakiness.
2. **`fsrcnn` silent no-op in the main binary**: output bit-matches AVIR
   (time included) with zero warnings. Root cause: the 4.90 build lacks
   `opencv2/dnn_superres.hpp`, so `cs_fsrcnn_upscale_tile_2x` takes the
   `#if !__has_include` early-false branch. Only `cs_tests` (4.13+contrib)
   can run it. Fix options: gate the CLI flag per-build, or emit the
   warn-once on that path too.

## Caveats

Single photo, single ratio, auto params. Not benchmarked: waifu2x/ncnn/
realcugan subprocess backends, `--sr-dict` (no trained `.csd2` in repo),
dncnn-RED (no model vendored), GPU/HIP path.
