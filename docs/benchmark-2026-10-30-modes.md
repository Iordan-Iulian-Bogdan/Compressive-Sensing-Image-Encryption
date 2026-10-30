# Mode sweep: encoder and decoder modes (2026-10-30)

Which container modes and decoder options are still worth having, measured on the current pipeline (branch
`local-sync/clang-gpu` after ddf01c0, `build_gpu.bat main` binary). Supersedes the mode tables of
`benchmark-2026-10-07-decrypt.md`, which predate seg-tile containers, in-container anchors and the enhance net, and
compared containers of very different sizes.

## Method

- Photo: `ImgReconstruct_backend/IMG_3690.png` (4032x3024). One photo only: confirm on the 8-photo set before removing code.
- Encoder sweep: every mode at `--ratio 0.9`, decoded with the production decoder without the net
  (`--tv 0 --photo-upscaler cs-sr --sr-pnp admm --hr-weight 1 --device gpu`, `CS_GPU_TILES=1 CS_GPU_ADMM=1 CS_HR_NO_POSTHOC=1`).
  Sizes differ per mode, so a mode is called dominated when another mode is at least as good on every metric with no
  larger container.
- Decoder sweep: every option on the production container (mode 19, 4.06 MB), `CS_GPU_TILES=1 CS_GPU_ADMM=1`.
- Metrics: PSNR, SSIM and CIEDE2000 p95/p99 (`Params-test/metrics.py`). Times are wall clock while an enhance-net
  training run shared the GPU (a few percent high).
- Run-to-run noise between two encodes of the same photo: about 0.01 dB (separate test, same day).
- Outputs in the repo root: `modes_enc_*`, `modes_dec_*`, scores in `modes_results.csv` and `modes_results_rest.csv`.

## Encoder modes (ratio 0.9, production decoder, no net)

| mode | flags | MB | PSNR | SSIM | ΔE p95 | ΔE p99 | verdict |
|---|---|---|---|---|---|---|---|
| 19 seg8-tile + anchors | `--ycc420 --seg-auto 8 --seg-tile --hr-detail 300000` | 4.06 | **36.11** | **0.9285** | **3.44** | **5.85** | production, best |
| 11 seg8-tile | `--ycc420 --seg-auto 8 --seg-tile` | 3.77 | 33.40 | 0.9169 | 4.30 | 8.82 | good for its size |
| 3 + adaptive | `--ycc420 --adaptive` | 3.71 | 33.33 | 0.9144 | 4.33 | 8.84 | good for its size |
| 3 ycc420 | `--ycc420` | 3.73 | 33.11 | 0.9122 | 4.45 | 9.09 | dominated by ycc420 + adaptive |
| 6 ycc420 + hf | `--ycc420 --hf-focus` | 3.73 | 33.10 | 0.9122 | 4.45 | 9.11 | dominated; identical to plain ycc420 |
| 15 ycc420 + tcluster | `--ycc420 --sampling tcluster` | 3.60 | 32.97 | 0.9108 | 4.51 | 9.25 | dominated |
| 16 ycc420 + tmeans | `--ycc420 --sampling tmeans` | — | — | — | — | — | **decode fails** with cs-sr |
| 9 ycc420 + anchors | `--ycc420 --hr-detail 300000` | 4.27 | 33.35 | 0.9141 | 4.36 | 8.80 | dominated by mode 19 |
| 11 seg8 (no tile) | `--ycc420 --seg-auto 8` | 2.26 | 32.46 | 0.8976 | 4.93 | 9.58 | weakest; 0.9 dB below seg-tile |
| 12 ycc422 seg8 | `--ycc422 --seg-auto 8` | 2.92 | 32.52 | 0.8983 | 4.77 | 9.53 | poor |
| 7 ycc422 | `--ycc422` | 4.85 | 33.16 | 0.9130 | 4.33 | 9.07 | dominated |
| 8 ycc422 + hf | `--ycc422 --hf-focus` | 4.85 | 33.15 | 0.9129 | 4.34 | 9.08 | dominated; identical to plain ycc422 |
| 10 seg8 BGR | `--seg-auto 8` | 4.67 | 32.64 | 0.8967 | 4.39 | 9.29 | dominated |
| 13 tcluster BGR | `--sampling tcluster` | 6.00 | 33.15 | 0.9145 | 4.14 | 9.09 | dominated |
| 14 tmeans BGR | `--sampling tmeans` | — | — | — | — | — | **decode fails** with cs-sr |
| 0 random | (none) | 7.86 | 33.30 | 0.9159 | 4.06 | 8.91 | dominated |
| 1 periodic | `--periodic` | 7.78 | 33.29 | 0.9159 | 4.07 | 8.93 | dominated; identical to random |
| 5 hf-focus | `--hf-focus` | 7.85 | 33.30 | 0.9160 | 4.06 | 8.91 | dominated; identical to random |
| 2 adaptive | `--adaptive` | 7.83 | 33.55 | 0.9183 | 3.90 | 8.63 | dominated |
| 4 BGR + anchors | `--hr-detail 300000` | 8.41 | 33.53 | 0.9178 | 3.96 | 8.63 | dominated |

Findings:
- Mode 19 beats everything. Every BGR mode is about twice its size and 2.6-2.8 dB worse with higher ΔE.
- The anchors are only worth their bytes in mode 19: in mode 9 (plain ycc420) and mode 4 (BGR) they add 0.2-0.5 MB
  for 0.0-0.25 dB.
- `--hf-focus` (modes 5, 6, 8) and `--periodic` (mode 1) change nothing measurable against their plain counterparts.
- `--seg-tile` is what makes seg8 work: without it seg8 is the weakest mode.
- `--sampling tmeans` (modes 14, 16) cannot be decoded by the production decoder: "--sampling tmeans needs the
  standard/AVIR path (no sr-direct/cs-sr/dip operator yet)".

## Decoder options (production container, mode 19)

| option | time | PSNR | SSIM | ΔE p95 | ΔE p99 | verdict |
|---|---|---|---|---|---|---|
| production + `--enhance` (fresh net) | 2.2 s | **38.32** | **0.9579** | **3.02** | **4.75** | best |
| production (cs-sr + ADMM) | 2.2 s | 36.13 | 0.9282 | 3.50 | 5.86 | baseline |
| `--fast` | 1.4 s | 36.07 | 0.9284 | 3.48 | 5.92 | fine: −0.06 dB for 35% less time |
| `--final-polish waifu2x-ncnn` | 9.4 s | — | 0.9259 | 3.61 | — | dominated: slower and slightly worse |
| `--final-polish realcugan` | 16.7 s | 30.00 | 0.9029 | 7.90 | 16.32 | harmful: −6 dB, ΔE doubles |
| `--sr-pnp off` | 4.0 s | 34.46 | 0.9340 | 3.98 | 7.88 | dominated |
| `--photo-upscaler sr-direct` | 3.7 s | 34.20 | 0.9395 | 3.96 | 8.44 | dominated |
| `--basis wavelet` | 3.6 s | 32.94 | 0.9038 | 4.81 | 10.32 | dominated |
| `--photo-upscaler avir` | 1.5 s | 33.25 | 0.9090 | 4.42 | 9.03 | dominated |
| `--sr-pnp prox` | 8.8 s | 32.49 | 0.8916 | 4.84 | 9.85 | dominated: 4x slower, −3.6 dB |
| `--photo-upscaler fsrcnn` | 1.4 s | 33.25 | 0.9090 | 4.42 | 9.03 | no-op: output identical to avir |
| `--solver joint` | 2.0 s | 36.13 | 0.9282 | 3.50 | 5.86 | no-op on ycc (warns, falls back to per-channel FISTA) |
| `--device cpu` | 2.1 s | 36.13 | 0.9282 | 3.50 | 5.86 | no-op while `CS_GPU_TILES=1` is set |
| `--photo-upscaler waifu2x` | ~16 h | — | — | — | — | unusable: stopped at ~383 of 3969 tiles after 1 h 42 min |
| `--photo-upscaler waifu2x-ncnn` / `realcugan` | hours | — | — | — | — | unusable by design; not run |
| `--photo-upscaler dip` | > 5 min | — | — | — | — | unusable: stopped at the 5 min cap |
| old AGENTS.md default (`--manual --tiles 64 --overlap 24 --full-res --iterations 10 --per-tile-coef --y-confidence 0.3`) | 3.3 s | — | 0.8920 | 6.51 | 12.77 | **wrong geometry**: half-size output, 0/300000 anchors applied |

Findings:
- The net is the largest single gain (+2.2 dB, SSIM +0.03, ΔE p95 −0.48) at no measurable time cost.
- `--sr-pnp off` and `sr-direct` have a higher SSIM than production without the net (0.934 / 0.940 vs 0.928) but lose
  1.7-1.9 dB and are clearly worse on ΔE; both lose to production + net on every metric.
- `--full-res` disables the 2x cs-sr path on ycc containers (`is_sr && !full_res` in `image_decryption.cpp`), so the
  decode stays on the LR luma grid and the anchors are skipped as out of range.
- `waifu2x`, `waifu2x-ncnn` and `realcugan` as `--photo-upscaler` run one external process per tile (`std::system` in
  `photo_upscaler.cpp`), 3969 times per 12 MP decode. They are only practical as `--final-polish` (one call on the
  whole image), where they do not help.

## Anchor ranking after the net (encrypt `--enhance`, now the default)

The closed-loop anchor pass decodes the anchor-free container and flags the worst 5% of 16x16 blocks. With
`--enhance <onnx>` on encrypt (seg + ycc420 + hr-detail only; otherwise ignored with a warning) the trial decode runs
that net first, so the anchors go where the error is left after it. Same container size, same 300k anchors, about
70 ms more per encode. Decoded with the fresh net, no post-hoc anchor write-back, 8 photos:

| photo | PSNR normal → after-net | SSIM | ΔE p95 |
|---|---|---|---|
| IMG_3690 | 38.428 → 38.634 | 0.9592 → 0.9604 | 2.843 → 2.816 |
| IMG_0007 | 39.810 → 39.808 | 0.9621 → 0.9620 | 2.204 → 2.203 |
| IMG_1297 | 25.659 → 25.676 | 0.9088 → 0.9090 | 9.653 → 9.641 |
| IMG_9321 | 43.999 → 44.187 | 0.9949 = | 2.152 → 2.138 |
| D28_rnd_123 | 48.444 → 48.453 | 0.9912 = | 1.634 → 1.633 |
| D06_rnd_39 | 29.616 → 29.632 | 0.9010 → 0.9013 | 5.829 → 5.823 |
| D17_rnd_71 | 38.692 → 38.698 | 0.8963 → 0.8964 | 2.012 → 2.011 |
| D17_nat_18 | 45.181 → 45.241 | 0.9925 = | 1.942 → 1.938 |

Mean +0.06 dB; ΔE p95 and p99 better on all 8; run-to-run noise between two normal encodes ≤ 0.009 dB.

Retraining the net on containers made this way did not help: on the same after-net containers it scored −0.08 dB and
ΔE p95 +0.06 against the fresh net (worse on all 8), although its own validation score was higher (36.804 vs 36.738 dB,
measured on different decodes). Keep the fresh net.

Post-hoc anchor replacement after the net cost quality: decoding the same after-net IMG_3690 container with the exact
anchor luma written back over the net's output gave 38.508 dB / SSIM 0.9591 / ΔE p95 2.991, against 38.629 / 0.9605 /
2.816 without. The ycc decoder now skips that write-back whenever `--enhance` ran (`image_decryption.cpp`); decodes
without the net still apply it. With the change, the default decode (no `CS_HR_NO_POSTHOC`) reproduces the table
above exactly on all 8 photos. The "production + `--enhance`" row of the decoder-options table was measured before
this change, with the write-back, and so is about 0.1 dB low.

## Closed-loop sample allocation (tested, not adopted)

Question: would spending the fixed luma budget where it improves the post-net output most beat the seg-tile
heuristic (per 64x64 LR tile, one of 8 levels from the mean |Laplacian|; per class, level weights spanning ~1.65x; a
global scale for the budget)? If so, a net predicting the gain per sample would be worth training. The encoder can
measure that gain directly, so this was tested first without a net:

- Probe: encode at the heuristic rates and at 1.5x them (same key, so the probe's samples contain the base ones),
  decode both with the enhance net, and score each tile (or label) by its error drop per extra sample. Two extra trial
  decodes, encode 1.6 s -> 3.4 s. Production pipeline otherwise (`--enhance` on encrypt and decrypt), container size
  unchanged (±0.2%).
- The switches below were a temporary experiment in `image_encryption.cpp` and have been removed again.
- Mode 1 (`CS_TILE_LOOP=1`): keep each level's tile count, hand the levels out in measured-gain order.
- Mode 2 (`CS_TILE_LOOP=2`): per-label rates moved toward equal marginal gain (r *= sqrt(g / mean g)), scale re-solved.
- Level spread (`CS_TILE_SPREAD=R`): geometric level weights with top/bottom ratio R instead of ~1.65x.

PSNR vs the heuristic (same run, noise ~0.01 dB):

| photo | heuristic | mode 1 | mode 2 | per-label gain spread (CV) |
|---|---|---|---|---|
| IMG_3690 | 38.633 | +0.005 | −0.050 | 0.13 |
| IMG_0007 | 39.810 | −0.019 | +0.036 | 1.88 |
| IMG_1297 | 25.681 | −0.115 | +0.023 (SSIM −0.0017) | 0.51 |
| IMG_9321 | 44.184 | −0.040 | +0.035 | 2.13 |
| D28_rnd_123 | 48.452 | −0.007 | +0.042 | 0.46 |
| D06_rnd_39 | 29.633 | −0.158 | +0.062 (SSIM −0.0004) | 0.50 |
| D17_rnd_71 | 38.696 | −0.041 | +0.011 | 0.34 |
| D17_nat_18 | 45.236 | +0.011 | +0.005 | 0.09 |
| mean | | **−0.045** | **+0.020** | |

Level spread on IMG_3690: 4x 38.628 (= heuristic), 16x 38.398 (−0.23); with mode 1 on top, 38.636 and 38.406.

Findings:
- Re-ranking tiles inside a class by measured gain is worse than the Laplacian ranking (6 of 8 photos, up to −0.16 dB):
  one probe's per-tile gain is noisier than the Laplacian.
- Moving budget between labels toward equal marginal gain gains +0.02 dB on average with mixed SSIM, for twice the
  encode time. Even where the measured gains are far from equal (CV ~2 on IMG_0007 and IMG_9321) the rebalance buys
  only ~0.035 dB.
- A wider level spread does not help (4x) or hurts (16x).
- Conclusion: the seg-tile heuristic allocation is already close to the best this budget allows with the current
  decoder and net. A learned allocator (predicting what this closed loop measures) has under ~0.05 dB to win; not worth
  building. The anchors placed after the net (+0.06 dB) and the net itself (+2.2 dB) were the levers.

## Code-level findings

- All 20 `CS_MODE_*` constants in `crypto_utils.hpp` are live: each is written by some encode-flag combination and
  accepted by the header parser in `crypto_utils.cpp`. Modes 4/9 never appear by name in `image_decryption.cpp` (the
  parser maps them to `hr_detail`), mode 6 is the default argument of `encrypt_ycc420_hf`, and mode 1 is written as the
  literal `1` (`image_encryption.cpp:802`).
- Test gaps: periodic (mode 1) has no test; `CS_MODE_YCC422_SEG` (12) and `CS_MODE_YCC420_SEG_HRDETAIL` (19, the
  production format) are not referenced by name in `tests/` (the seg encoder is exercised by `test_main.cpp` and
  `test_modes.cpp`; which modes those produce was not checked).
- Every decode prints `[gpu] CPU-only build: no HIP device, staying on CPU`, from the CPU stub of the old `--device gpu`
  API (`cs_gpu_cpu.cpp`). It is misleading: the GPU tile path and the net run on the GPU.
- `--final-polish waifu2x-ncnn` falls back to AVIR with only a warning when `waifu2x-ncnn-vulkan` is not on PATH. The
  binary lives in `waifu2x-ncnn-vulkan-20250915-windows\waifu2x-ncnn-vulkan-20250915-windows\` in the repo root.

## Caveats

- One photo; the 8-photo set (IMG_3690, IMG_0007, IMG_1297, IMG_9321, D28_rnd_123, D06_rnd_39, D17_rnd_71,
  D17_nat_18) should confirm any removal.
- The encoder sweep scores the decode without the net; the net was trained only on mode 19 containers.
- Old modes may still be needed to decrypt existing containers even if they are no longer worth encoding.
