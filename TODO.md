# TODO

## GPU

- [ ] **Port the WMMA kernels to rocWMMA.** They call `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
  directly (helpers in `ImgReconstruct_backend/gpu_wmma.hpp`), which ties them to RDNA3 (`gfx1100`): the
  fragment layout they rely on (lanes 16-31 repeat lanes 0-15, outputs at rows `2i + lane/16`) changed on
  RDNA4 (gfx12), and CDNA uses MFMA. rocWMMA would make them portable.
  - Kernels: `gpu_swin.hip` (`k_stem`, `k_layer`, `k_conv`), `k_fista_w` in `gpu_gemm.hpp`, and the
    `gpu_enh.hip` kernel (also on `gpu_wmma.hpp`).
  - Done when: same quality and speed on gfx1100 (`docs/gpu-optimizations.md` sections 9-10), and it builds
    and passes on one more architecture.
  - Tried (section 11): rocWMMA 2.2 ships with ROCm 7.2 and works on gfx1100; it reads the packed weight
    tiles as `col_major`, ld 16, and agrees with the builtin to float rounding. `k_stem` on rocWMMA
    (accumulators staged in LDS, LayerNorm from LDS) was 5-10% slower (15.4-16.7 -> 17.2-17.5 ms per
    3969 tiles), reverted. `k_conv`'s A rows (16 pixels over two halo rows) have no constant stride, so
    `load_matrix_sync` needs each tap's rows copied first; `k_layer` / `k_fista_w` would stage every
    epilogue through LDS. Expect a slowdown on gfx1100, and only gfx1100 is here to test on.
  - Options when it is picked up: a full rocWMMA rewrite (measure the gfx1100 slowdown), or the builtin
    kernels for gfx11 with a fallback elsewhere (fp32 FISTA loop, ONNX Runtime prior); best done with the
    target GPU at hand to test on.

- [x] **Profile the LR FISTA stage (`gpu_tiles.hip`) before optimizing it.** Done (2026-11-04,
  `docs/gpu-optimizations.md` section 12): build + upload 30-70 ms per batch, the luma loop 108-148 ms, no
  backtracking tiles in production; the loop was bound by LDS bank conflicts / occupancy, not launches. `[gpu-tiles-ycc] gpu` is
  225-335 ms per 12 MP decode, the largest GPU stage, while its floors are ~10-30 ms of VRAM traffic and a few
  ms of fp32 math (luma batch: 9 arrays of ~3.05M floats, ~110 MB, ~17 array passes per FISTA step, 16-40
  steps x 1-5 reweights). So it is not bandwidth-bound; find what it is.
  - Measure: per-kernel times (Radeon GPU Profiler, or hipEvents around `synth` / `adj` / `k_step`), the
    `Batch::build` + upload share, and how many tiles take `solve_backtrack` (tv > 0 from the per-class seg
    params?): its 25-trial line search does four blocking `hipMemcpy` round trips per trial.
  - Done when: the 225-335 ms is attributed (host prep / transfers / kernels / syncs) in
    `docs/gpu-optimizations.md`.

- [x] **Fuse the LR FISTA loop per tile, as `k_fista_w` does for cs-sr.** Done for the fixed-step branch
  (`k_lr_fista`, bit-identical, luma loop 87 -> 30-40 ms; section 12). The backtracking branch still runs
  the launch-per-step loop with host round trips (no production tile takes it). LR tiles are small (luma ~32x24,
  chroma ~16x12: ~3 KB per fp32 array), so a tile's whole FISTA loop (both DCTs, scatter residual, soft
  threshold, momentum) fits in one block's LDS and registers. That removes the ~17 whole-image array passes
  and ~8 launches per step, and keeps the working set out of the Infinity Cache question entirely (today the
  luma batch's ~110 MB + samples exceeds the 80 MB cache; the chroma batch, ~55 MB, fits). The backtracking
  branch would move its line search on-device (per-tile accept, no host round trips).
  - Depends on the profiling item: only worth it if kernels or syncs, not host prep, dominate.
  - Done when: bit-comparable decodes (or PSNR within 0.01 dB on the 8 photos) and the stage time measured.

- [x] **Band-fuse the enhance U-Net's full-res layers (`gpu_enh.hip`).** Done (`CS_ENH_BAND`, default 128
  rows, bit-identical; section 12). The gain was the allocation (93 -> 17-20 ms with the aliased quarter-res
  buffers); the network time did not measurably change on the shared GPU. Retime it on an idle GPU. Every conv streams a whole-image
  NHWC fp16 tensor through VRAM (12 MP: ~385 MB per tensor at c16, ~770 MB at c32), far beyond the 80 MB
  Infinity Cache. Run head / e0 / tail (and fuse0) in horizontal bands sized so a ResBlock's in / mid / out
  stay cache-resident (c32: 64-row band ~16 MB per tensor), recomputing the 3x3 halo rows at band edges.
  - Measured today (`runs/20261101-235435_enh_hip_time`): network 36 ms (c16) / 99 ms (c32); c32 is 2.75x
    c16 for 4x the MACs, so the narrow layers are memory- or launch-bound. Realistic upside ~20-40 ms of c32.
  - Done when: identical output to the unfused path and the network time measured on c16 and c32.

- [x] **Find why `[enhance] upload + planes` scales with net width.** It was device allocation of the
  activations (74-76 of 82 ms at c32); the banded layers fixed it (above). 14 ms (c16) vs 57 ms (c32) on
  D01_nat_32, 26 vs 118 ms on IMG_3690, though the planes do not depend on the width: probably the weight
  load / fragment packing is counted there. Cache the loaded net across decodes, or move it out of the timed
  phase if it is already cached.

- [x] **Replace AMD-specific intrinsics with standard forms where it costs nothing.** Checked per swap
  against the device assembly, the decode timings and bit-identical decode PNGs (section 11).
  - Done, identical machine code: `__float2int_rn(x)` -> `(int)rintf(x)` (`gpu_admm.hip`, `gpu_sr.hip`,
    `gpu_composite.hip`, `gpu_swin.hip`), `__fmaf_rn` -> `fmaf` (AVIR dither in `gpu_sr.hip`), `wave_sync()` ->
    `__syncwarp()` (`gpu_swin.hip`; HIP's `__syncwarp` is the same three calls).
  - Kept, measured slower: `rsqrtf` (adds a denormal guard; network +1-2%), `__fdividef(1, x)` (compiles to
    the IEEE division sequence, not `v_rcp`), `__funnelshift_r` / a 64-bit shift for
    `__builtin_amdgcn_alignbyte` (`k_nlm` +2% / +8%), `memcpy` for `ld16` (changes the WMMA loops: network
    +20%, `k_fista_w` +14%).
  - Kept, measured or required before: DPP reductions `dpp_xor` (vs `__shfl_xor`: layer 185 -> 156 ms per
    512 tiles; rocPRIM `warp_reduce` untried), `__builtin_amdgcn_udot4` and `__builtin_amdgcn_sched_barrier`
    in NLM (2x slower without), `pin()` (compiler moving WMMA into a divergent branch), the WMMA builtin.

## Evaluation

- [ ] **Compare the format against conventional codecs fairly.** PSNR / SSIM barely penalise JPEG's 8x8
  blocking, and no full-reference metric separates recovered detail from detail the enhance net invents. The
  last comparison (JPEG ~1.3-1.5 MB and AVIF ~0.7 MB matching CS r0.9 at 4.4 MB) used single points and
  those metrics.
  - Metrics built for codec comparison: SSIMULACRA2, Butteraugli (max / p-norm for local failures), DISTS next
    to LPIPS (DISTS tolerates different-but-plausible texture; disagreement hints at replaced texture), ΔE
    p95/p99 for colour.
  - Invented detail: per wavelet / frequency band and local window, gain (how much of the original's detail
    survives, < 1 = blur) and spurious energy (decoded detail uncorrelated with the original). JPEG blocking
    shows as spurious energy at the 8 px frequency and its harmonics, net texture in the high bands; output
    spurious-energy maps. Task checks where invention matters: OCR of text crops, face-embedding similarity
    of face crops (original vs decode).
  - RD curves and BD-rate per metric over the 8 photos, not single points. Baselines: mozjpeg, JPEG + an ML
    restoration net (e.g. FBCNN: the fair "conventional codec + ML" opponent), AVIF, JPEG XL, and our own
    decode without the enhance net (the no-invention curve). Encryption gives no size credit: the
    alternative is codec + AES-GCM, ~28 bytes over the codec.
  - Final verdict: a small side-by-side preference test on matched-bpp crops.
  - Expect the metrics to narrow a ~3x gap, not close it; the spurious-energy measure may count against the
    net.
  - Done when: one script sweeps all codecs on the 8 photos and writes the RD curves, BD-rate table and
    spurious-energy maps, with the results summarised in `docs/`.

## Enhance net

- [ ] **Try NAFNet-style blocks in the same U-Net.** Replace the ReLU ResBlocks of c16d2 with NAFNet blocks
  (LayerNorm, 1x1 -> depthwise 3x3 -> SimpleGate -> simplified channel attention -> 1x1, learned residual
  scale). NAFNet gets more dB per MAC on denoise / deblur in the literature, but here depthwise convs are
  memory-bound and skip WMMA, and LayerNorm / channel attention add reductions (channel attention's global
  pool breaks single-pass banded execution; a per-band or per-tile pool would avoid it).
  - Compare at equal HIP network ms, not equal MACs: first a PyTorch val-dB check at a few widths, then a HIP
    port only for a variant that wins there.
  - Done when: an 8-photo 6-bit HIP decode comparison against the production net (PSNR, SSIM, ΔE95/99, ms),
    promoted only if SSIM and ΔE95 agree.
- [ ] **Check whether the density plane (plane 0) is still needed.** It is the luma sample mask blurred
  (Gaussian σ 4 on the LR grid, `enh_inputs.BLUR_LR`), from the 3-plane version; since the LR sample mask
  (plane 3) became an input, the net could derive the local rate itself. Retrain the production recipe
  without plane 0 (same steps, data, seed) and compare val dB; dropping it also saves the blur in the decoder.
  - Done when: val dB plus an 8-photo HIP decode comparison show whether it can go (drop it only if PSNR,
    SSIM and ΔE95 do not get worse).

## Codec

- [ ] **Per-region downsampling scale instead of a fixed 2x.** Today the luma is solved on a 2x grid
  everywhere, with 1x only in the anchor blocks (450k pixels, ~4% of a 12 MP photo). Let each 64x64 tile pick
  a scale of 1, 2 or 4: 4x for flat areas (sky, blurred background), 1x for detail, 2x for the rest.
  - Why: at high ratios the 2x grid is the ceiling (the sample ratio caps at 1, so r1.0 gives the same
    container as r0.9; above about r0.83 extra bytes have nowhere to go), and at low ratios CS on the 2x grid
    is weak (bird r0.05: 27.3 dB before the net), where a 4x grid sampled at ~25% costs the same bytes. Likely
    to pay off more on 50 MP+ photos (phone 50 MP sensors resolve well below their pixel count, and 4x
    regions also cut decode time); there are no such test photos yet, all 8 are 12 MP.
  - Design: scales 1/2/4 only, because they nest with the 2x2 luma / 4x4 chroma blocks that `cs_enh_project`
    and the net's planes use (3x breaks that alignment). Scale map at 2 bits per tile (<1 KB at 12 MP). The
    encoder picks scales by error vs bytes per tile, like the closed-loop anchor pass. A prototype could run
    the whole pipeline on separate sub-images, but that adds seams and per-container overhead.
  - Cost: CS solve per scale grid; a 4x upscaler (cs-sr twice or a new one), none for 1x (the `--full-res`
    path); seam handling where scales meet; mixed tile sizes in the GPU batch; a scale plane for the enhance
    net plus a new training set and retrain (~1 day of data, ~3 h of training). Roughly one to two weeks to
    the first fair comparison.
  - First, a cheap CPU test on the 8 test photos: per 64x64 tile, the error of downsample-by-s plus a good
    upsampler (s = 1, 2, 4) against the production decode's error on that tile. It gives the share of each
    photo that could go to 4x at no loss (a lower bound, since the decoder beats a plain upsampler), the
    bytes that frees, and how much of the remaining error is the 2x limit that only 1x fixes. Build the
    full version only if both are substantial.
  - Done 2026-11-06 (scratchpad `scale/oracle.py`, 23688 tiles, luma, box-down + bicubic-up vs the production decode,
    LR sample counts from the ymask dumps): tiles where 4x is already as good as the decode are 81 / 49 / 25 / 12% at
    r0.1 / 0.3 / 0.5 / 0.9, freeing 0 / 7.8 / 7.6 / 9.1% of the luma samples (12.9 / 16.1 / 17.4% at r0.3 / 0.5 / 0.9
    if 4x may cost up to 1 dB in those tiles). At r0.1 nothing is freed because those tiles hold fewer samples than
    a full 4x grid (256 per tile). The r0.9 decode (the 2x limit, with net and anchors) is 13 / 28 / 51% of the
    error at r0.1 / 0.3 / 0.5, and at r0.9 the top 10% of tiles hold 63% of it (top 25%: 90%); the anchors already
    target those blocks. Production r0.9 beats the 2x bicubic floor on every photo (by 0.0 to 7.1 dB).
    Verdict: a modest 4x gain (~8% of luma samples, a lower bound) and a 1x side that overlaps the anchors. Next:
    a tighter 4x test through the real pipeline (encode the 2x-downsampled photo, upscale its decode).
  - Done 2026-11-06, the tighter test (scratchpad `scale/oracle2.py`): the 2x-downsampled photos encoded at
    r0.1/0.3/0.5/0.9 and decoded with the production decode, bicubic back to full size, as per-tile 4x options next
    to the production 2x decodes at r0.05/0.1/0.3/0.5/0.9; per photo a Lagrangian picks one option per 64x64 tile
    at production's luma-sample budget. Mean luma dB over the 8 photos at r0.1 / 0.3 / 0.5 / 0.9: production 31.61 /
    35.52 / 37.54 / 39.28; 2x options only +0.98 / +0.91 / +0.55 / +0.08; adding 4x only +0.13 / +0.05 / +0.04 /
    +0.02 more (4x picked for 23 / 13 / 8 / 4% of tiles). **Parked:** on 12 MP photos 4x adds little over a better
    per-tile rate on the 2x grid; revisit for 50 MP+ photos. Optimistic in both columns: hindsight choice per tile,
    tiles treated as independent, luma only, anchor bytes not counted (the 4x options had no anchors).
  - 48 MP, 2026-11-06 (`ImgReconstruct_backend\IMG_0611.HEIC`, iPhone 8064x6048, converted to `IMG_0611.png`;
    scratchpad `scale/big_run.py`, `big_oracle.py`): the same test with 2x / 3x / 4x / 5x / 6x (pre-shrink 1 / 1.5 /
    2 / 2.5 / 3 before the pipeline's 2x) at r0.05 to r0.9, 11844 tiles. Luma dB vs production at r0.1 / 0.3 / 0.5:
    production 29.70 / 33.40 / 36.19; 2x rates only +0.63 / +1.55 / +1.33; +3x +1.16 / +2.07 / +1.45; +3x,4x
    +1.72 / +2.33 / +1.51; all scales +1.98 / +2.40 / +1.53 (r0.1: 6x on 39% of tiles, 5x 22%, 4x 25%, 2x 10%). So at
    48 MP the larger scales add +1.35 / +0.85 / +0.20 dB over the best 2x rates (12 MP: +0.13 / +0.05 / +0.04),
    and the per-region scale is worth it at low ratios. One photo; 3x and 5x went through a non-integer pre-shrink.
    r0.9 gains nothing here: its budget already samples every 2x tile fully, and there is no 1x option.
  - Prototype, 2026-11-06 (scratchpad `scale/layers.py`, no codec changes): per 120x120 tile the (scale, ratio) the
    allocation above picks; one container per scale layer (the pre-shrunk photo, an external `--seg-mask` label
    map with that layer's tiles at a flat rate and the others at 0, 6-bit luma, no anchors); each layer through the
    production decode, scaled up, blended per tile (8 px feather). Real bytes, RGB metrics, vs production
    interpolated to the same bytes (log-bytes, between r0.05/0.1/0.3/0.5):

    | budget (bytes) | 2x only, per-tile rates | all scales 2x-6x |
    |---|---|---|
    | r0.1 (1.68 MB) | +0.50 dB, SSIM +0.042, ΔE95 −0.79 | +1.26 dB, SSIM +0.074, ΔE95 −1.02 |
    | r0.3 (4.96 MB) | +0.87 dB, SSIM +0.022, ΔE95 −0.74 | +1.03 dB, SSIM +0.029, ΔE95 −0.51 |
    | r0.5 (8.28 MB) | +0.80 dB, SSIM +0.010, ΔE95 −0.41 | +0.36 dB, SSIM +0.009, ΔE95 ±0 |

    Multi-scale wins at low ratios; at r0.5 it loses colour to 2x only, because each layer draws its chroma
    uniformly over the whole layer (the unused tiles' share is wasted) and the larger scales have coarse chroma
    (6x: 12x12 blocks). The tile choice came from full encodes and decodes at every option (hindsight), which a
    real encoder would have to estimate.
  - Built, 2026-11-06: mode 20 (`encrypt --multiscale <plan.png>`, plan = per-120 px tile scale 2..6 and rate):
    one seg-mask layer container per scale inside one sealed outer container; seg prefix version 2 draws chroma
    only inside the label support. The decoder decodes each layer, upsamples bicubic, blends with a 17x17 feather.
    The GPU solve no longer drops a layer to the CPU path for its zero-sample tiles (4x layer 5.9 -> 0.62 s).
    48 MP IMG_0611 r0.3: 34.38 dB, SSIM 0.896, ΔE95 3.24 at 4.98 MB vs production 33.10 / 0.879 / 4.03 at 5.64 MB.
  - 11 photos, 24.5 MP (IMG_0615..0625), hindsight plans, each container matched to production's bytes within
    1% (scratchpad `scale/eval_ms.py`, `ms_eval.csv`); mean delta vs production at the same bytes:

    | ratio | PSNR | SSIM | ΔE95 | ΔE99 | wins PSNR / SSIM / ΔE95 | decode |
    |---|---|---|---|---|---|---|
    | r0.1 (~1.4 MB) | +1.29 dB | +0.030 | −0.95 | −1.52 | 10 / 11 / 11 | 3.9 s vs 1.2 s |
    | r0.3 (~3.0 MB) | +0.39 dB | +0.010 | −0.34 | −0.55 | 7 / 11 / 11 | 4.2 s vs 1.3 s |
    | r0.5 (~4.6 MB) | +0.03 dB | +0.005 | −0.05 | ±0 | 7 / 11 / 9 | 4.3 s vs 1.3 s |

    SSIM better on all 33. Outliers: IMG_0623 r0.3 / r0.5 −2.37 / −1.94 dB, IMG_0620 r0.5 −0.62 dB.
    IMG_0623's loss was the 2x layer having no anchors: the plan took its 2x tile errors from production encodes
    (450k anchors), so it gave thin dark lines on light ground 4-5% rates that only hold with anchors.
  - 2x layer anchored, 2026-11-06 (`encrypt --multiscale <plan> --hr-detail 450000 --enhance <rank net>`: the
    production closed loop on the 2x layer, flags limited to its tiles, 450k x its tile share), same 11 photos at
    production's bytes:

    | ratio | PSNR | SSIM | ΔE95 | ΔE99 | wins PSNR / SSIM / ΔE95 |
    |---|---|---|---|---|---|
    | r0.1 | +2.01 dB | +0.028 | −0.90 | −1.90 | 11 / 11 / 11 |
    | r0.3 | +1.11 dB | +0.009 | −0.33 | −0.76 | 10 / 11 / 10 |
    | r0.5 | +0.61 dB | +0.004 | −0.09 | −0.17 | 10 / 11 / 10 |

    IMG_0623 now +1.57 / +1.01 / +0.61 dB. The one loser is IMG_0620 (smooth, production MSE ~2-3) at r0.3 / r0.5,
    −0.84 / −0.40 dB, ΔE95 +0.01 / +0.28: 70% of its excess error lies within 10 px of a scale boundary (the
    17 px feather), which the per-tile plan does not price, the rest in low-rate 4x-6x tiles. Decode ~4.2 s vs
    1.3 s.
  - Done 2026-11-06:
    - Decode 4.3 -> 2.8 s (24.5 MP). The composite runs tile by tile, so a tile with one scale in its 3x3
      neighbourhood is that layer resampled; mixed tiles get separable box weights over the tile grid. It took
      1.14 s and now takes 0.10 s, with the same output to 1 level. One PBKDF2 per container: the layers share the
      outer salt and key (each header keeps its own IV), and `cs_derive_key` caches the last derivation. The
      remaining time is the five layer decodes (2x 1.25 s, 3x-6x 0.14-0.46 s) plus the PNG write. A GPU
      composite would save at most 0.1 s, so it was not built.
    - Seams: each layer also samples 8 px past its own tiles (`kMsMargin`), at the rate of the nearest tile of
      its scale. Tested at r0.3 on hindsight plans, against production at equal bytes:

      | photo | no margin | 8 px margin | 16 px margin |
      |---|---|---|---|
      | IMG_0620 | −0.86 dB | −0.20 dB | 0.00 dB |
      | IMG_0625 | +2.32 dB | +2.29 dB | +1.94 dB |
      | IMG_0615 | +1.07 dB | +1.05 dB | +0.90 dB |

      All three use the 17 px feather. Feathers of 0, 4 and 16 px did no better.
    - 2x layer anchors are ranked with `--enhance`. Before this fix the flag was dropped on multi-scale
      encrypts.
    - In-encoder planner: `encrypt --multiscale auto --ratio r`, about 1 s of the ~2.8 s encode.
      - Per 120 px tile and scale it computes three features (`ms_tile_features`):
        - D: the scale's floor, the RGB MSE of the layer upscaled back;
        - G: luma detail between the layer and its LR;
        - V: the LR's Laplacian energy.
      - Error model: MSE = D + exp(a·f(lD, lG, lV) + lq·min(b·[1, lD, lG, lV], −0.05)), so the error falls as
        the rate rises. Features are clamped to the training range.
      - Fit: 16 dataset photos, none of them evaluation photos (scratchpad `plan/collect.py`, `fit2.py`). Each
        scale is encoded as a whole-photo layer, with a random rate held over each 6x6-tile block.
      - Two failures came first. With per-tile random rates, a starved tile borrows from its neighbours, so the
        error at low rates was under-predicted. A free quadratic model then predicted error below the scale floor
        (one 5x tile at a 2% rate in feather texture decoded to a flat patch). Both gave −3..−4.5 dB at r0.5 on
        IMG_0620/0621/0622.
      - Allocation: a Lagrangian over 5 scales x 20 rates per tile, at production's budget for --ratio. The budget
        is 1.2 x ratio x LR pixels of luma plus 450k anchors, with one anchor counted as 0.8 sample. The plan is
        redone with the margin samples counted, and containers land at 0.92-0.97x production's bytes.
      - 11 photos (24.5 MP), against production at equal bytes:

        | ratio | PSNR | SSIM | ΔE95 | ΔE99 | wins PSNR / SSIM / ΔE95 | hindsight plans |
        |---|---|---|---|---|---|---|
        | r0.1 | +2.12 dB | +0.029 | −1.05 | −2.17 | 11 / 11 / 11 | +2.01 dB |
        | r0.3 | +1.32 dB | +0.008 | −0.38 | −0.91 | 11 / 11 / 11 | +1.11 dB |
        | r0.5 | +0.77 dB | +0.003 | −0.17 | −0.39 | 11 / 11 / 11 | +0.61 dB |

      - Decode 2.6-2.8 s vs 1.2-1.3 s. IMG_0620 is now +1.91 / +0.47 / +0.09 dB.
    - Decode speed, 2026-11-06 (IMG_0621 r0.3, idle GPU, wall incl. process start): 1.73-1.77 s vs production
      1.37-1.41 s. Done:
      - one key derivation per container;
      - layer tiles without samples skip cs-sr (bilinear) and the LR solve (flat fill);
      - one thread per layer, with the GPU stages serialized by a lock;
      - parallel PNG writer.

      Per stage, summed over the 5 layers vs production:

      | stage | 5 layers | production |
      |---|---|---|
      | LR solve | 411 ms (the first layer's 195 ms includes the GPU start-up wait) | 273 ms |
      | cs-sr (5679 sampled tiles of 15742) | 222 ms | 379 ms |
      | net network | 337 ms (2x layer alone 172) | 176 ms |
      | composite | 64 ms | — |

      - The 2x layer's frame is full size, so its nets cost as much as production's although it holds 39% of
        the tiles.
      - Tried and reverted, 2026-11-06: nets cropped to each layer's sample bounding box + 48 px (one run per
        layer). The layer's tiles are scattered, so the box is 38-90% of the frame (only 1-3 layers per photo
        under 90%). Network −21 ms per photo on 11 photos, inside the wall-time noise; output max 1 level off.
      - Done, 2026-11-06: masked conv kernels (`gpu_enh::Inputs::active`).
        - The decoder passes each layer's 8x8 blocks that the composite reads: tiles of its scale + the 8 px
          feather + the cubic taps (`cs_ms_needed_blocks`).
        - `gpu_enh` dilates them by the net's receptive field (69 px for c16d2) and k_conv skips the other output
          blocks; pixels outside keep the input image.
        - The first net's blocks are dilated again by the second net's field, since its input there must be exact
          too. Without that, 0.002% of the pixels were 1 level off.
        - 20-73% of the blocks are active.
        - On 11 photos at r0.3 (ABBA): network 340 -> 221 ms, wall 1887 -> 1793 ms per photo; output identical on
          all 11, production decode identical. `CS_MS_ENH_ALL=1` turns it off.
      - Done, 2026-11-06: the tile merge reads the cs-sr output that is still on the device, plus shared flat tiles.
        Merge 151-165 -> 41 ms over 5 layers; 11 photos 1786 -> 1620 ms wall; output identical. IMG_0621 1.40 s,
        about production's time.
      - Full write-up: `docs/gpu-optimizations.md` section 18.
    - `--ms-coarse <a>` (0..1, built 2026-11-06, exe `ms26`), planner lever.
      - It moves the plan's coarseness (area mean of log(s/2)/log 3) the fraction a of the way from the plain
        plan's value to all-6x. A second multiplier is bisected at the same budget.
      - a = 0 gives the same plan as before. The target is hit to 0.001 at every a.
      - IMG_0621 r0.3, 2x / 6x tiles: a 0.2: 447 / 855; 0.5: 331 / 1223; 0.8: 113 / 1511.
      - From a ≈ 0.3 the budget can no longer be spent: the fine tiles are already at full rate. The container
        shrinks: 2.81 MB at a = 0, 2.20 at 0.5, 1.32 at 0.8.
      - Most of the shift is 2x -> 6x; 3x-5x thin out, because the error model rates 6x as cheaper than 4x for the
        flat tiles.
      - Quality not measured yet. (An earlier (2/s)^a error weight jumped straight to all-6x by a = 2; replaced.)
- [x] **Anchor count scales with the photo area** (2026-11-06, exe `ms27`). Opt-in since exe `ms30`: `--hr-area`; the
  default is the fixed count again, because the scaling lost at equal bytes (below).
  - `--hr-detail K` is now the count for a 4032x3024 (12.2 MP) photo; other sizes get K x area / 12.2 MP
    (`cs_hr_detail_scaled`). `CS_HRDETAIL_MAX_SAMPLES` is raised from 2^20 to 2^24.
  - Production r0.3, 450k, single runs (fixed -> scaled anchors):

    | photo | bytes | PSNR | SSIM | ΔE95 |
    |---|---|---|---|---|
    | IMG_3690 (12.2 MP, 450k either way) | 1.83 MB = | 35.87 = | 0.921 = | 3.34 = |
    | IMG_0621 (24.5 MP, 903k) | 2.93 -> 3.39 MB | 43.13 -> 43.53 | 0.9733 -> 0.9741 | 1.72 -> 1.69 |
    | IMG_0611 (48.8 MP, 1.8M) | 5.64 -> 6.99 MB | 33.11 -> 33.83 | 0.879 -> 0.886 | 4.03 -> 3.79 |

  - Not measured at equal bytes yet.
  - The multi-scale-vs-production numbers above used fixed 450k on 24.5 MP photos (both sides).
  - `Params-test/gen_enh_prod.py` (300k) now scales the same way for photos that are not 12.2 MP.
  - `--hr-scale m` (0..16, default 1, exe `ms29`) multiplies the anchor count; the container grows with it.
    - The closed-loop flagging used to take a fixed 5% of the 16x16 blocks, so anchors saturated at 74%..100% fill
      (609,536 on IMG_3690, at m ≈ 1.35).
    - `cs_error_anchor_flags(..., count)` now flags more blocks once `count` would fill 5% past 74%, production's
      fill. Default encodes still flag exactly 5%.
    - IMG_3690 r0.3, single runs:

      | m | anchors | blocks | bytes | PSNR | SSIM | ΔE95 |
      |---|---|---|---|---|---|---|
      | 0 | 0 | — | 1.38 MB | 32.67 | 0.903 | 4.48 |
      | 0.5 | 225k | 2381 | 1.60 MB | 35.43 | 0.920 | 3.51 |
      | 1 | 450k | 2381 | 1.83 MB | 35.85 | 0.921 | 3.35 |
      | 2 | 900k | 4751 | 2.28 MB | 37.04 | 0.927 | 2.98 |
      | 4 | 1.8M | 9502 | 3.18 MB | 38.72 | 0.937 | 2.60 |

    - m = 0 runs the anchor-trained nets without anchors.
  - Equal bytes, 2026-11-06 (scratchpad `an/hr_eq.py`, `hr_cmp.py`; 5 photos: IMG_3690 12.2 MP + IMG_0615 / 0618 / 0621 /
    0624 at 24.5 MP).
    - Method: each m point against that photo's m = 1 curve over `--ratio` 0.2..1.0, log-bytes interpolation.
    - Fewer anchors win, more lose:

      | point | PSNR | SSIM | ΔE95 | wins PSNR / SSIM / ΔE95 |
      |---|---|---|---|---|
      | r0.3 m0.5 | +0.66 dB | +0.0096 | −0.25 | 5 / 5 / 5 |
      | r0.3 m2 | −0.56 | −0.0078 | +0.17 | 0 / 0 / 0 |
      | r0.3 m4 | −1.09 | −0.0133 | +0.27 | 1 / 0 / 1 |
      | r0.5 m0.5 | +0.30 | +0.0035 | −0.08 | 3 / 5 / 5 |
      | r0.5 m2 | −0.13 | −0.0034 | +0.02 | 2 / 0 / 2 |

    - On the 24.5 MP photos m0.5 is the old fixed 450k. So the area scaling loses there at equal bytes: +0.34..0.95 dB
      for the old count at r0.3.
    - IMG_3690 also prefers 225k to 450k at r0.3 (+0.33 dB, SSIM +0.008).
    - The exception is IMG_3690 at r0.5 m2 / m4: +0.58 / +1.16 dB PSNR, but SSIM −0.004 / −0.007.
    - Open: a lower default count (m 0.25..0.75 not tried) or reverting the scaling. The nets were trained with 450k
      at ~12 MP.
  - 48.8 MP `iPhone14Pro_ProRAW48MP.png`, 2026-11-06 (scratchpad `an/hr48*.py`; decodes kept in `prores48_anchors\`).
    - Method: equal bytes against the 450k curve (r0.2..1.0; r0.2 = 4.51 MB, 32.55 dB, SSIM 0.930).

      | point | 113k (m0.25) | 225k (m0.5) | 900k (m2) | 1.8M (m4 = --hr-area) |
      |---|---|---|---|---|
      | r0.3 | −0.17 dB, SSIM +0.003, ΔE95 −0.11 | −0.07, +0.002, −0.07 | −0.09, −0.002 | −0.66, −0.007 |
      | r0.5 | −0.37 dB, SSIM +0.001 | −0.22, +0.001 | +0.29, −0.001 | +0.41, −0.002 |
      | r0.7 | −0.66, −0.000 | −0.40, −0.000 | +0.61, +0.000 | +1.19, +0.000 |
      | r0.9 (above the curve's top, 15.23 MB) | −0.78 | −0.49 | +0.69, +0.001 | +1.40, +0.001 |

    - The ratio curve flattens above r0.7 (r0.9 = r1.0: 1.2 x 0.9 caps the luma rate at 1). Up there anchors are
      the only way to spend more bytes, and they pay: the 1.8M point is +1.40 dB, ΔE95 −0.13 beyond what any
      ratio reaches.
    - At r0.3 more anchors lose, as on the 24.5 MP photos. Fewer anchors trade a little PSNR for SSIM and ΔE.
      - Tried and reverted: bands of 512 / 1024 rows (2.08 / 2.00 s vs 1.93), ~5-8 ms fixed cost per extra net
        run. That cost is not allocation: the net's buffers come from the device arena (0 ms on 9 of 10 calls).
        The exception is the first full-frame call, where S0 (372 MB) misses the arena (32 ms, production too).
    - Layer-tuned net (`6bclay_c16d2`, 16k-step fine-tune of `6bc450_c16d2` on layer data; run
      `x64/GpuBuild/runs/20261106-134902_enh_layers`; validation 26.64 -> 26.80 dB) is worse on the 11 photos
      than the production nets.
      - Alone, vs the production pair: −0.16 / −0.12 / −0.17 dB, ΔE95 +0.08..0.10.
      - Followed by u2dc: −0.04 / −0.05 / −0.10 dB, ΔE95 −0.01 / +0.01 / +0.02 (r0.1 / 0.3 / 0.5).
      - Not promoted.
  - Settings oracle, 2026-11-06: should a net choose the encode settings per photo? (scratchpad `oracle/oracle.py`,
    `oracle_metrics.py`, `analyze.py`; decodes kept in `F:\oracle\<photo>\`.)
    - Photos: 88 in all, 72 at ≤13 MP and 16 at >20 MP. They are the 77 validation photos of the enhance-net set plus
      IMG_0615..0625.
    - Configs: 8 in all, each at `--ratio` 0.2 / 0.3 / 0.5 / 0.7 / 0.9, with exe `ms30` and the production decode.
      - `P_m*`: production at `--hr-scale` 0.5 / 1 / 2 / 4.
      - `M_c0_m*`: `--multiscale auto` at `--hr-scale` 0.5 / 1 / 2.
      - `M_c0.3_m1`: `--multiscale auto --ms-coarse 0.3`.
    - Method: equal bytes. Every config is read at the production default's (`P_m1`) bytes by log-bytes interpolation.
      - rule = the one config with the best mean at that ratio.
      - oracle = the best config per photo.
    - Gains over production (`P_m1`). The rule shown is the SSIM-best config, except at r0.7 / 0.9 (picked by ΔE95):

      | ratio | rule | rule PSNR / SSIM / ΔE95 | oracle − rule (SSIM pick / ΔE95 pick / PSNR pick) |
      |---|---|---|---|
      | 0.2 | M_c0_m0.5 | +1.81 dB / +0.0285 / −1.01 | +0.0013 / −0.05 / +0.21 dB |
      | 0.3 | M_c0_m0.5 | +1.31 / +0.0157 / −0.59 | +0.0006 / −0.06 / +0.13 |
      | 0.5 | M_c0_m0.5 | +0.54 / +0.0062 / −0.22 | +0.0005 / −0.04 / +0.18 |
      | 0.7 | M_c0_m2 (ΔE95) or M_c0_m0.5 (SSIM) | +0.27 / −0.0005 / −0.07 or −0.04 / +0.0023 / −0.02 | +0.0009 / −0.03 / +0.18 |
      | 0.9 | M_c0_m2 (ΔE95); P_m1 (SSIM) | +0.35 / −0.0001 / −0.08 | +0.0012 / −0.02 / +0.18 |

    - On >20 MP the headroom is smaller still: oracle − rule is ≤ 0.0003 SSIM, ≤ 0.025 ΔE95 and ≤ 0.10 dB.
      - The ΔE-best rule there uses more anchors one step earlier: m1 at r0.3 / 0.5, m2 from r0.7.
    - Verdict: a per-photo settings net is not worth building. Picking by ratio captures nearly all the gain. The
      remaining headroom (≤ 0.0015 SSIM, ≤ 0.06 ΔE95) is at the noise level of what we promote.
    - The gain is in the settings themselves: multi-scale with anchors rising with ratio.
      - Proposed rule: m0.5 up to r0.5, m2 from r0.7.
      - At r0.2, `--ms-coarse 0.3` is the ΔE-best (−1.08 vs −1.01) but SSIM-worse (+0.024 vs +0.029).
    - Cost, mean over all runs:
      - encode 4.6 s multi-scale vs 3.4 s production;
      - decode 2.6 s vs 2.2 s (two GPU workers running at once, so absolute times are contended).
    - Adopted 2026-11-06 (exe `ms31`, `ImgReconstruct_backend.cpp` main):
      - The production flags (`--seg-auto --ycc420 --seg-tile --hr-detail`) now default to `--multiscale auto`.
      - Without an explicit `--hr-scale`, the multiplier is m = 0.5 · 4^clamp((r − 0.5) / 0.2, 0, 1): 0.5 up to
        r0.5, 1 at r0.6, 2 from r0.7.
      - `--multiscale off` restores the single-scale container. `Params-test/gen_enh_prod.py` passes it, so the
        training data stays in the nets' format.
      - `--multiscale auto` now also works in batch mode (`encrypt @list`), one plan per photo.
      - Checked on two oracle photos at r0.3 / 0.9, against the oracle runs:
        - anchors are identical (145,240 / 900,000), and bytes are within 0.02%;
        - PSNR is within +0.06 dB. Encodes are not byte-reproducible (a fresh salt each time), so decodes are
          compared by metrics, not pixels.
        - Batch decodes match their single runs.
      - Not done: the size-dependent variant (>20 MP prefers m1 at r0.3 / 0.5 by ΔE95). `--show-mask` has no mask
        view for mode 20 containers.
    - Sweep speed: the first version computed metrics (`ratio_sweep.metrics`, ~19 s per 12 MP decode) inside the
      encode/decode worker, which left the GPU idle. Moving them to a separate pool of 8–12 processes cut it from
      ~11 s per run per worker to ~2 s.
    - Layer-trained net, 2026-11-07 (not promoted): `6bclaylo_c16d2`, c16d2 9-plane from scratch, 128k steps, on
      `F:\Training Data\enh_data_6bc_laylo` (whole-photo layers, 2x 40% at 225k anchors, 3x–6x 15% each, r0.1–0.3).
      On the default multi-scale containers (32 photos) it is behind `6bclo5` at every ratio, alone and with u2dc
      (details and numbers: `docs/enhance-nets.md`, run `20261106-233103_enh_laylo`). Training on layers at
      these scales gives the net nothing the mixed low-ratio data does not.
      - Found on the way: `6bclo5` + u2dc beats production's two stages at r0.1 / 0.2 (+0.26 / +0.06 dB, ΔE95
        −0.24 / −0.07, SSIM +0.0045 / +0.0008) and loses at r0.3 (−0.06 dB). Adopted 2026-11-07 as
        `--enhance-low` (picked below 0.43 luma sample bits per pixel; `docs/enhance-nets.md`, "Net choice by ratio").
      - Not tried: training on layers cut from real planner containers (partial layers, layer borders, real
        per-tile rates); this result makes it a low priority.
  - Open: the 48 MP photo with the planner; the 12 MP test photos.
- [ ] **Per-tile sampling rate chosen closed-loop.** The scale test above found that choosing each 64x64 tile's
  sampling rate from the decodes at r0.05 / 0.1 / 0.3 / 0.5 / 0.9 (same total luma samples) beats the seg-auto
  per-class rates by +0.98 / +0.91 / +0.55 / +0.08 dB luma at r0.1 / 0.3 / 0.5 / 0.9 (8 photos, oracle upper bound:
  hindsight, independent tiles, luma only). Most of the gain is moving samples from tiles the net already
  reconstructs well to tiles it does not.
  - First: check how the per-tile rates reach the decoder (the seg-tile labels and per-class rates in the
    container) and whether a per-tile rate fits the format; then an encoder pass like the anchor ranking (decode,
    measure the error after the net per tile, move samples), checked on the 8 photos.
  - Done when: an 8-photo HIP decode comparison against production at r0.1 to r0.9 (PSNR, SSIM, ΔE95/99, encode
    ms), promoted only if SSIM and ΔE95 agree.
  - Done when: an 8-photo HIP decode RD comparison against the production decode (PSNR, SSIM, ΔE95/99, ms)
    at r0.05 to r0.9, plus at least one 50 MP+ photo.
