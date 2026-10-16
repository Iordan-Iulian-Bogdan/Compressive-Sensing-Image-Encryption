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

## Caveats

- Fit on ONE photo (bird): re-run `sweep.ps1` on new content types before
  trusting blindly; the functional forms (1/r-like + tile factor) should
  transfer, the constants may not.
- tv=0.01 costs 3-10x decrypt time; the rule uses it only where measured
  better (seg below 0.5).
- Sparse YCC needs `--iterations 12+` regardless of sampler (uniform YCC
  0.25 fails at defaults too, 10.78 dB) — separate axis, unchanged.
