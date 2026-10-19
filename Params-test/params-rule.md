# Automatic coef/tv selection (Params-test)

Sweep: bird photo (IMG_3690, 4032x3024), seg-auto-8 vs uniform, ratios
0.15/0.3/0.5 (half-res) + 0.2/0.4 (full-res), tiles 64/96/128 (+1 for the
auto path), coef 0.01-0.3, tv 0/0.01/0.03, iters 8, wscale 2. 532 cells in
`results.csv` (`pertile` = 1 for the `--per-tile-coef` arm). Reproduce with
`.\Params-test\sweep.ps1` (resumable; containers `enc-*.enc.png` reused
across cells).

## Headline findings (half-res, tiles=64)

| ratio | seg best | uniform best |
|---|---|---|
| 0.15 | 0.2 + tv.01 → 24.3 | 0.12, tv0 → 26.1 |
| 0.3 | 0.12, tv0 → 27.1 | 0.05, tv0 → 28.4 |
| 0.5 | 0.08, tv0 → 29.0 | 0.035, tv0 → 30.2 |

Full-res agrees (seg 0.2 → 0.2+; 0.4 → 0.08; uniform 0.2 → 0.08; 0.4 → 0.05).

- Optimum coef RISES as ratios fall and is HIGHER than the old auto rule
  everywhere below 0.5 (old flat 0.045 was 2-4x too weak; seg@0.15 lost 8 dB).
- Tiles barely matter for quality (64→128: +0.2-0.5 dB at 2.5x time);
  64 is the sweet spot (6 s vs 16 s per cell half-res).
- tv=0.01 beats tv=0 only left of the coef peak (under-regularized) and
  for seg below 0.5; tv=0.03 never wins. tv costs 3-10x time (ROF).
- `--per-tile-coef` on seg: +1-2 dB at 0.15-0.3, +0.2-0.3 at 0.5;
  on uniform: bit-identical (uniform counts → scale 1).

## Shipped rule (`cs_auto_reg_coefficient` / `cs_auto_reg_tv`)

```
r = clamp(ratio, 0.05, 1.0)
seg:    coef = 0.045 * r^-0.75  in [0.02, 0.3],  x tilefactor
uniform: coef = 0.01875 / r      in [0.01, 0.3]
tilefactor = ((ntiles/64)^0.15 in [0.5, 1.2]) for seg always,
             for uniform only when r < 0.5  (dense uniform tiles
             are self-similar; single tiles need ~half)
tv = (seg && r < 0.5) ? 0.01 : 0.0
```

Covered modes: BGR random + segmask. YCC/adaptive/periodic/HF/T-tetromino
keep legacy behavior. Applies when `--coef`/`--tv` are absent (sentinel
-1.0), in auto AND manual mode; explicit values always win. Unit-pinned in
`test_auto_reg_params`.

## Validation (same-container law checks, tiles=1)

| cell | shipped value | measured | sweep optimum |
|---|---|---|---|
| seg 0.15 | 0.100 | 23.55 | 23.95 |
| seg 0.3 | 0.0595 | 25.40 | 25.39 |
| seg 0.5 | 0.0406 | 26.79 | 26.95 |
| uni 0.15 | 0.067 | 25.07 | 25.25 |
| uni 0.3 | 0.034 | 26.55 | 26.52 |
| uni 0.5 | 0.0375 | 27.63 | 27.67 |

(All within 0.4 dB of the per-cell optima. Fresh-container runs vary ±1 dB
by salt draw; these checks reuse sweep containers for exactness.)

## Cross-metric validation (SSIM + CIEDE2000 + LPIPS)

The suite was re-run with `-Metrics`, adding SSIM (structure), CIEDE2000
mean/p95 (perceptual color), and LPIPS-alex (deep-feature) to each cell.
Columns `ssim,de_mean,de_p95,lpips` appended to `results.csv`; helpers
`metrics.py` (SSIM/ΔE) and `metrics_lpips.py` (LPIPS, ~5-7 s/cell).

Per-cell argmax under each metric, over the 36 suite cells (2 geom x 3
ratios x 2 samplers x 3 tile grids):

- **PSNR, SSIM and ΔE-p95 pick the SAME (coef,tv) in 33/36 cells.** The
  three disagreements are off-by-one coef steps or tv=0 vs 0.01 ties on
  uniform 0.5/0.15 (e.g. full 0.5 uni t64: PSNR/ΔE 0.035/0 vs SSIM
  0.02/0.01, 0.2 dB and 0.003 SSIM apart).
- **No metric prefers a different sampler or tile count.** The metric
  choice does not overturn any structural conclusion in this file.
- `seg` optima sit at the top of the default coef grid (0.08) everywhere.
  Where the extended range was swept (half seg t64) the peak moved to
  coef 0.12-0.2 — PSNR/ΔE at 0.2, SSIM at 0.12 — i.e. the metrics agree
  within one step and all say the default grid under-covers `seg`.
- `seg` prefers tv=0.01 at every ratio; `uniform` prefers tv=0 at low
  ratio and is tv-insensitive at 0.5.

The earlier PSNR-only green-cast artefact (full-res YCC seg-solve vs BGR:
PSNR/SSIM/LPIPS favored YCC, ΔE-p95 favored BGR) is the one case where a
single metric lies; see the metric suite only after `-Metrics` runs.

## High-ratio extension (0.6 - 0.9)

The suite was extended to ratios 0.6, 0.7, 0.8, 0.9 (same grid, `-Metrics`).
Over the extended 84-cell grid (2 geom x 7 ratios x 2 samplers x 3 tiles)
PSNR / SSIM / ΔE-p95 pick the identical (coef,tv) in 56/84 cells, and every
disagreement is negligible: worst case ΔSSIM = 0.006 and worst ΔΔE-p95 =
0.52 (usually <=0.25, often 0.00). The PSNR optimum coincides with the
perceptual optimum; no metric flips the sampler, tile count or tv choice.

Best coef falls monotonically with ratio, matching the 1/r law:

| ratio | full uni best | full seg best | half uni best | half seg best |
|---|---|---|---|---|
| 0.15 | 0.08 | 0.08 | 0.08 | 0.08 |
| 0.3  | 0.05 | 0.08 | 0.05 | 0.08 |
| 0.5  | 0.035 | 0.08 | 0.035 | 0.08 |
| 0.6  | 0.02 | 0.05 | 0.02 | 0.08 |
| 0.7  | 0.02 | 0.05 | 0.02 | 0.05 |
| 0.8  | 0.01 | 0.035 | 0.01 | 0.05 |
| 0.9  | 0.01 | 0.035 | 0.01 | 0.035 |

- At ratio >=0.8 the **uniform** optimum sits on the 0.01 grid floor, i.e.
  the true peak is at or below 0.01 while the shipped uniform law
  `0.01875/r` predicts ~0.021-0.023 there. The high-ratio end of the
  uniform coef law is a touch too strong; consider a lower floor
  (re-fit pending).
- `seg` holds tv=0.01 across the whole high-ratio range; `uniform` holds
  tv=0 at every ratio >=0.3.
- **Tile sweet spot shifts upward with ratio.** Winner of {64,96,128} at
  the totimum, and t128-t64 gain at the best coef:

  | sampler | r0.15 | r0.3 | r0.5 | r0.7 | r0.9 |
  |---|---|---|---|---|---|
  | half seg | 64 (-0.5) | 128 (+0.2) | 128 (+0.3) | 128 (+0.2) | 128 (+0.0) |
  | full seg | 64 (-0.5) | 64 (-0.8) | 128 (+0.4) | 128 (+0.4) | 128 (+0.3) |
  | half uniform | 128 (+0.1) | 128 (+0.3) | 128 (+0.3) | 128 (+0.2) | 128 (+0.1) |
  | full uniform | 64 (-0.2) | 128 (+0.4) | 128 (+0.4) | 128 (+0.3) | 128 (+0.3) |

  At low ratio 64 tiles wins (fewer boundary seams); from ratio ~0.5 up,
  128 tiles wins by +0.1 to +0.4 dB. The earlier flat "64 is the sweet
  spot" claim came from the low-ratio sweep only.

Structural reading of the added ratios (t64, best coef, PSNR):

| sampler | 0.15 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| half seg | 20.5 | 26.9 | 29.0 | 29.5 | 30.4 | 30.8 | 31.3 |
| half uni | 26.0 | 28.4 | 30.2 | 30.9 | 31.5 | 32.1 | 32.6 |
| full seg | 16.7 | 26.4 | 31.4 | 32.7 | 33.6 | 34.4 | 35.3 |
| full uni | 28.8 | 32.7 | 35.9 | 37.7 | 39.3 | 41.7 | 44.4 |

## Caveats

- Fit on ONE photo (bird): re-run `sweep.ps1` on new content types before
  trusting blindly; the functional forms (1/r-like + tile factor) should
  transfer, the constants may not.
- tv=0.01 costs 3-10x decrypt time; the rule uses it only where measured
  better (seg below 0.5).
- Sparse YCC needs `--iterations 12+` regardless of sampler (uniform YCC
  0.25 fails at defaults too, 10.78 dB) — separate axis, unchanged.
