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
