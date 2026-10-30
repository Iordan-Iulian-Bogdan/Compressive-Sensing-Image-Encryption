# Ratio sweep: best settings per ratio, 0.20-0.95 (2026-10-30)

Which encoder and decoder settings are best at each `--ratio` from 0.20 to 0.95 in steps of 0.05, on the production
pipeline (mode 19: `--seg-auto 8 --ycc420 --seg-tile --hr-detail N --enhance <fresh net>`; decoder
`--tv 0 --photo-upscaler cs-sr --sr-pnp admm --hr-weight 1 --device gpu --enhance <fresh net>`,
`CS_GPU_TILES=1 CS_GPU_ADMM=1`; `build_gpu.bat main` binary at e3875df).

## Method

- 8 photos (IMG_3690, IMG_0007, IMG_1297, IMG_9321, D28_rnd_123, D06_rnd_39, D17_rnd_71, D17_nat_18), 16 ratios.
- Stage 1, one factor at a time against production: decoder variants on the 300k container (no net, tv 0.005 / 0.01 /
  0.02, hr-weight 0.5 / 2, `--fast`), and anchor budgets 0 / 150k / 600k decoded with and without the net. 1792 decodes.
- Stage 2: 600k and 1M anchors with hr-weight 0.25 / 0.5 / 1. 768 decodes.
- Metrics: PSNR, SSIM, CIEDE2000 p95/p99 (same code as `Params-test/metrics.py`), means over the 8 photos. "Wins" counts
  photos better than the reference. Run-to-run noise between encodes ~0.01 dB.
- Script `Params-test/ratio_sweep.py` (resumable), raw rows in `ratio_sweep_results.csv`. Containers in
  `%LOCALAPPDATA%\ratio_sweep`.

## Recommended settings per ratio

The same settings are best at every ratio. Nothing in the sweep has a crossover anywhere from 0.20 to 0.95:

- Encoder: production flags, with `--hr-detail 600000` if the ~0.3 MB extra is acceptable (see below).
- Decoder: production (`--tv 0 --hr-weight 1` + net). Optional `--hr-weight 0.5` at ratio <= 0.35 (+0.05 dB).
  `--fast` is fine from 0.5 up (<= 0.07 dB).

| ratio | MB (300k) | PSNR | SSIM | ΔE p95 | ΔE p99 | MB (600k) | PSNR 600k | SSIM 600k | ΔE p95 600k | no net | `--fast` | tv 0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 1.25 | 30.33 | 0.8003 | 8.54 | 13.66 | 1.55 | 30.59 | 0.8032 | 8.42 | -0.78 | -0.21 | -0.31 |
| 0.25 | 1.45 | 31.46 | 0.8261 | 7.71 | 12.44 | 1.75 | 31.72 | 0.8287 | 7.60 | -0.91 | -0.17 | -0.41 |
| 0.30 | 1.64 | 32.44 | 0.8467 | 7.04 | 11.44 | 1.94 | 32.71 | 0.8492 | 6.94 | -1.02 | -0.14 | -0.47 |
| 0.35 | 1.84 | 33.31 | 0.8639 | 6.48 | 10.57 | 2.14 | 33.60 | 0.8662 | 6.38 | -1.12 | -0.11 | -0.50 |
| 0.40 | 2.04 | 34.10 | 0.8785 | 6.00 | 9.80 | 2.34 | 34.38 | 0.8806 | 5.91 | -1.23 | -0.10 | -0.50 |
| 0.45 | 2.23 | 34.81 | 0.8909 | 5.58 | 9.11 | 2.53 | 35.07 | 0.8927 | 5.50 | -1.34 | -0.09 | -0.48 |
| 0.50 | 2.43 | 35.45 | 0.9013 | 5.22 | 8.51 | 2.73 | 35.70 | 0.9031 | 5.14 | -1.44 | -0.07 | -0.44 |
| 0.55 | 2.62 | 36.03 | 0.9105 | 4.90 | 7.96 | 2.92 | 36.27 | 0.9121 | 4.83 | -1.55 | -0.05 | -0.40 |
| 0.60 | 2.82 | 36.56 | 0.9185 | 4.62 | 7.47 | 3.12 | 36.80 | 0.9202 | 4.55 | -1.66 | -0.05 | -0.36 |
| 0.65 | 3.02 | 37.03 | 0.9256 | 4.36 | 7.02 | 3.32 | 37.27 | 0.9272 | 4.30 | -1.75 | -0.04 | -0.32 |
| 0.70 | 3.21 | 37.48 | 0.9319 | 4.14 | 6.59 | 3.51 | 37.71 | 0.9334 | 4.08 | -1.84 | -0.04 | -0.31 |
| 0.75 | 3.40 | 37.87 | 0.9375 | 3.94 | 6.21 | 3.70 | 38.10 | 0.9388 | 3.89 | -1.93 | -0.04 | -0.30 |
| 0.80 | 3.60 | 38.22 | 0.9425 | 3.78 | 5.89 | 3.90 | 38.45 | 0.9437 | 3.72 | -2.00 | -0.03 | -0.30 |
| 0.85 | 3.79 | 38.52 | 0.9469 | 3.64 | 5.64 | 4.09 | 38.76 | 0.9481 | 3.58 | -2.06 | -0.03 | -0.30 |
| 0.90 | 3.98 | 38.79 | 0.9510 | 3.53 | 5.44 | 4.28 | 39.03 | 0.9521 | 3.47 | -2.13 | -0.03 | -0.30 |
| 0.95 | 4.17 | 39.01 | 0.9545 | 3.44 | 5.31 | 4.47 | 39.27 | 0.9556 | 3.38 | -2.18 | -0.02 | -0.30 |

The last three columns are PSNR deltas against production (300k). Decode time is 1.9-2.1 s for every row; `--fast`
is ~1.1 s; tv > 0 is ~12 s.

## Findings

- **Enhance net**: helps at every ratio although it was trained at 0.9 only, and the gain grows with ratio (+0.78 dB
  at 0.20, +2.18 at 0.95). SSIM and ΔE p95 are better on all 8 photos at every ratio.
- **TV**: never helps. tv 0.005 / 0.01 / 0.02 lose 0.1-0.2 / 0.3-0.5 / 0.7-1.05 dB with worse SSIM and ΔE, and cost 6x
  the decode time. The auto rule's tv 0.01 below 0.5 applies to BGR seg only, not to these containers.
- **hr-weight**: 0.5 gains +0.06 dB at 0.20-0.30 (7-8 of 8 photos), +0.02-0.05 up to 0.65 and nothing above 0.70.
  0.25 is the same as 0.5; 2 is slightly worse. Too small to change the default.
- **`--fast`**: costs 0.21 dB at 0.20, falling to 0.02 at 0.95.
- **Anchor budget at a fixed ratio**: more is always better. 0 → 150k → 300k → 600k gives about -1.1 to -1.5 / -0.3 /
  0 / +0.25 dB, and 600k beats 300k on all 8 photos at every ratio (SSIM +0.001-0.003, ΔE p95 -0.06 to -0.12) for
  +0.30 MB. 1M stores the same as ~600k: the closed-loop pass flags 5% of the 16x16 blocks (2381 on IMG_3690 =
  609,536 pixels), so ~610k is the effective maximum.
- **Anchor budget at a fixed container size**: here the anchors lose. Spending the same bytes on a higher ratio with
  fewer anchors scores better. The table below compares each anchor setting interpolated to the size of the 300k
  container at that ratio. 0k (a higher ratio with no anchors) has the best SSIM and ΔE up to ~0.75. 150k has the best
  PSNR everywhere and ties ΔE from 0.70 up. 600k is worst on every metric. So if the target is a file size rather
  than a ratio, raise `--ratio` and use 0-150k anchors.

| size of 300k at ratio | MB | 0k | 150k | 300k | 600k |
|---|---|---|---|---|---|
| 0.30 | 1.64 | 32.54 / 0.8637 / 6.60 | 32.76 / 0.8575 / 6.72 | 32.44 / 0.8467 / 7.04 | 31.13 / 0.8154 / 8.03 |
| 0.50 | 2.43 | 35.31 / 0.9098 / 4.99 | 35.63 / 0.9068 / 5.05 | 35.45 / 0.9013 / 5.22 | 34.71 / 0.8863 / 5.72 |
| 0.70 | 3.21 | 37.01 / 0.9362 / 4.06 | 37.54 / 0.9350 / 4.05 | 37.48 / 0.9319 / 4.14 | 37.02 / 0.9233 / 4.43 |
| 0.85 | 3.79 | 37.80 / 0.9493 / 3.70 | 38.47 / 0.9490 / 3.61 | 38.52 / 0.9469 / 3.64 | 38.25 / 0.9409 / 3.82 |

(PSNR / SSIM / ΔE p95, linear interpolation between the measured ratios of each anchor setting.)

## Caveats

- The 600k / 1M stage-2 containers were re-encoded (new salt), so stage-2 comparisons to stage 1 carry the ~0.01 dB
  encode noise.
- Per-photo spread is wide (at 0.50: IMG_1297 21.2 dB, D28_rnd_123 44.3 dB); the means hide that, but the win counts
  show the rankings hold photo by photo.
- The net was trained on ratio-0.9, 300k-anchor containers only. A net trained on mixed ratios or 600k anchors may
  shift the anchor/ratio trade-off.
