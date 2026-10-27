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
