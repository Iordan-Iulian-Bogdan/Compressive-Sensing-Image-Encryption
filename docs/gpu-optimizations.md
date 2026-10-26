# GPU decrypt path: design, optimizations and measurements

This documents the experimental GPU build (`build_gpu.bat main`, `-DCS_GPU_TILES -DCS_ONNX_DML`):
what runs where, every optimization that was made (with the measurement that justified it), the
experiments that did **not** pay off, how the profiling was done, and what the remaining bottlenecks are.
All GPU stages are opt-in through environment variables; the CPU build and the default behaviour are
unchanged (the CPU path was verified bit-identical to the pre-GPU output).

Hardware / software used for every number below: Ryzen 9 7900 (24 threads), Radeon RX 7900 XT
(gfx1100, 20 GB), Windows 11, ROCm/HIP 7.2, ONNX Runtime 1.22.1 DirectML.
Workload: `IMG_3690` (seg-auto 8 container, ratio 0.9) decrypted with
`--manual --tiles 64 --overlap 24 --iterations 10 --per-tile-coef --y-confidence 0.3 --photo-upscaler cs-sr
--sr-prior-model models\swin2sr\onnx\model.onnx --sr-pnp admm`: a 64x64 grid = 4096 tiles, LR tiles 48x56,
HR (2x) tiles 96x112. Wall times are for the whole process, run unprofiled, median of 2-3 runs.

## 1. Result in one table

| Step | Wall time | What changed |
| --- | --- | --- |
| CPU only | 168-170 s | baseline (ADMM stage 617 thread-seconds: NLM prox 58 %, cs-sr FISTA core 40 %) |
| GPU LR solve + GPU ADMM, CPU ONNX prior | 82 s | `gpu_tiles` + `gpu_admm` + broker; 64 worker threads |
| + DirectML prior, batched | 54 s | prior on the GPU through a batching front end |
| + flat tile loop | 31 s | removed the wavefront barrier so prior and ADMM overlap |
| + LDS-tiled GEMM DCT with fused epilogues | 28 s | ADMM GPU time 20 s -> 9 s |
| + shape-folded Swin2SR graph | 19.3 s | 19,113 -> 2,018 ONNX nodes; prior GPU time 22 s -> 9 s |
| + startup: no dynamic session, prefetch, disk cache | **13.8 s** | 6.2 s of serial startup removed |

The GPU output differs from the CPU output by 56.3 dB PSNR (max 11 of 255 levels, from the LR solve
that skips neighbour warm starts, see 3.1); the CPU path itself is unchanged.

## 2. Architecture

```
decrypt_tiles()                                   image_decryption.cpp
 |- GPU LR pre-pass  (CS_GPU_TILES)               gpu_decrypt.cpp -> gpu_tiles.hip
 |     all 4096 tiles x 3 channels in one batch, reweighted-L1 FISTA, no neighbour warm start
 |- prefetch of the folded DirectML session       onnx_denoise.cpp (background thread)
 `- flat parallel loop over all tiles (64 workers)
       per tile: cs_sr_upscale_tile_2x()          superres.cpp
         |- Swin2SR prior  ----------------------> batcher (onnx_denoise.cpp) -> DirectML
         `- PnP-ADMM loop  ----------------------> broker (gpu_admm_broker.cpp) -> gpu_admm.hip
```

| File | Role |
| --- | --- |
| `gpu_common.hpp` | tile layout record, shared dense-DCT kernels, `DevBuf`, DCT table builder |
| `gpu_gemm.hpp` | LDS-tiled batched GEMM with fused epilogues (the DCT engine of `gpu_admm`) |
| `gpu_tiles.hip/.h`, `gpu_decrypt.cpp/.hpp` | batched LR solve (reweighted-L1 FISTA, DCT basis) |
| `gpu_admm.hip/.h` | PnP-ADMM stage: cs-sr FISTA core, OpenCV-exact NLM prox, dual update |
| `gpu_admm_broker.cpp/.hpp` | turns per-tile blocking calls from 64 worker threads into GPU batches |
| `onnx_denoise.cpp` | DirectML execution provider, shape folding, batching of prior runs, session cache |
| `scripts/make_swin2sr_fp16.py` | builds the optional fp16 prior model |
| `tests/gpu_tiles_test.cpp`, `tests/gpu_admm_test.cpp` | accuracy checks against CPU references + batch-scaling sweep |

Design rule that everything else follows from: the CPU code processes one tile per thread with blocking
calls, a GPU is only efficient on large batches. Every GPU stage therefore keeps all state resident on
the device for its whole loop, uses `blockIdx.y` as the tile index (tiles of different size share a launch,
each with its own DCT tables), and is fed by a batching front end.

## 3. Optimizations

### 3.1 Batched LR solve without neighbour warm starts (`gpu_tiles`)

- The CPU solves tiles in an anti-diagonal wavefront because each tile warm-starts from its already solved
  west/north neighbours. That serialises the work into ~127 waves.
- Dropping the warm start (each tile starts from a smooth fill of its own samples, transformed to DCT
  coefficients) makes all tiles independent. It was measured to cost no reconstruction quality.
- All 4096 tiles x 3 channels go to the GPU in 2 batches (grouped by iteration count / reweights / wscale):
  1.27 s total versus 29 thread-seconds of CPU solve. Accuracy check (`gpu_tiles_test`): worst relative
  coefficient error 1.4e-7, min PSNR vs the CPU reference 138 dB.
- DCTs are dense matrix products (`T = X Cc^T`, then `Cr T`), which is cheaper than an FFT at ~100 px and
  needs no library.
- This is also what makes 3.3 possible: with every tile pre-solved, tiles no longer depend on neighbours.

### 3.2 PnP-ADMM stage on the GPU (`gpu_admm`)

The CPU loop (per tile, 8 iterations): anchor/DCT, a cs-sr FISTA core (40 inner steps), inverse DCT,
NLM prox on 8-bit luma, dual update. Profiling the CPU version showed NLM 359 s and the core 247 s of
617 thread-seconds, DCTs ~1 %, so both had to move.

- The whole loop runs on the device with no host round trips; only the initial HR plane goes in and the
  final plane comes out.
- **NLM matches OpenCV**, not merely approximates it: integer 7x7 SSD, the `>> 6` distance quantization,
  fixed-point weights (`INT_MAX/(15*15*255)`, weights below 0.001 dropped), reflect-101 border, rounded
  integer division. The padded tile is staged in LDS. Test vs `cv::fastNlMeansDenoising` on 25 planes:
  2 of 139,920 pixels differ, by 1. NLM is only 4-8 % of the GPU time.
- Whole-loop check vs a CPU reference built from `cv::dct` + OpenCV NLM: min PSNR 53 dB (rounding flips in
  the 8-bit NLM input), identical mean PSNR vs ground truth (28.50 dB both).
- The CPU fallback is kept: tiles with TV, HR anchors, a block DCT, or an ONNX denoiser prox run the old
  loop, and any GPU failure returns false so the caller falls back per tile.

### 3.3 Flat tile loop instead of wavefront (`image_decryption.cpp`)

Measured before the change: the DirectML prior (~34 s) and the ADMM stage (~16 s) added up to the wall
time, because within a wave every tile does the prior and then every tile does ADMM, with a barrier between
waves (127 of them): the two GPU stages never overlapped.

When the GPU pre-pass solved every non-empty tile, `decrypt_tiles` now runs one flat `omp parallel for`
over all tiles (`flat_tiles`). Tiles at different stages overlap, batches get bigger.
**54 s -> 31 s.** `CS_GPU_WAVES=1` restores the wavefront for A/B comparison.

### 3.4 Batching broker (`gpu_admm_broker.cpp`)

- Workers call `cs_gpu_admm_run()` and block; the first waiting thread becomes the leader, takes the
  queue, runs one batched solve, wakes everyone. A batch closes at `target` waiting jobs or after
  `CS_GPU_ADMM_WAIT_MS` (default 30 ms) so tail batches do not stall.
- Worker count: batches are only as large as the number of threads that can be waiting at once, and the
  default 8 threads gave batches of ~2 tiles (1,973 batches, GPU latency-bound). With the GPU enabled the
  loop uses one worker per tile of a wave, up to 64 (`cs_gpu_admm_threads`; blocked workers cost no CPU).
  Sweep (full decrypt): 16 workers 31.5 s, 32 -> 22.8 s, 48 -> 19.9 s, 64 / 96 / 128 -> 19.5 s (plateau).
- The batch wait time (5-30 ms) and the prior batch target (4-16) were measured to make no difference
  (19.2-19.7 s); 60 ms costs ~1 s.
- GPU launch floor: a solve costs ~13 ms regardless of size (about 1,300 launches at ~10 us), so single-tile
  batches are ~15x less efficient than 48-tile batches (13.2 ms vs 0.89 ms per tile).

### 3.5 LDS-tiled GEMM with fused epilogues (`gpu_gemm.hpp`)

Profile of the first version (per-kernel `hipEvent` timing, see 5): the four dense-DCT kernels were ~77 % of
GPU time at ~1.5 % of peak FLOPs; each thread did ~100 multiply-adds with two global loads each (no reuse),
plus a memset (9 %), a sparse scatter (6 %) and the soft-threshold step (10 %) as separate kernels.

- One templated kernel `k_gemm<A, B, EPILOGUE>` computes `OUT = A * B` for every tile of a batch. Operands
  are tile data or the tile's DCT table (normal or transposed), which covers all four transforms
  (inverse passes A/B, forward passes A/B) without ever materialising a transpose.
- 32x32 output block per 64-thread workgroup, 4x4 register block per thread, 16-wide K chunks staged in LDS.
  54 VGPRs, no scratch spills, LDS 4.6 KB per block.
- **Fused epilogues.** Because a thread owns a 4x4 register block, a 2x2 neighbourhood never straddles two
  threads, so:
  - `EPI_RESID` (inverse pass B): computes the 2x2 average of the reconstructed HR image, subtracts the
    sample on the LR grid and writes the residual replicated to the four children, or zero where there is no
    sample. This replaces the memset, the scatter kernel and the downsample.
  - `EPI_STEP` (forward pass B): applies the gradient, soft threshold and FISTA momentum in registers.
  Result: **4 kernels per FISTA step instead of 7.** Samples live on a per-tile LR grid (sentinel for
  "no sample"), so the scatter and its index arrays are gone.
- **Load/compute overlap.** The first tiled version was *slower* at small batches (81 ms floor vs 17 ms)
  because every thread issued its 16 global loads one after another (~8 us of latency per K chunk). Loads
  for a chunk are now issued back to back into registers, and the next chunk is prefetched into registers
  while the current one is multiplied.
- `gpu_admm_test` batch scaling (48 tiles, ADMM loop): naive 138 ms (2.87 ms/tile) -> tiled 42.8 ms
  (0.89 ms/tile), floor 17 ms -> 13 ms; the accuracy check is unchanged (53.1 dB min vs the reference).
  In the full decrypt the ADMM GPU time dropped from ~20 s to ~9 s and later to ~6.4 s once overlapped.
- Why not rocBLAS/hipBLASLt: the matrices are tiny (~100 x 100) and need fused epilogues; the batched
  call overhead and the extra elementwise kernels would give back what the library saves.

### 3.6 DirectML prior: batching, shape folding (`onnx_denoise.cpp`)

- **Serial runs are the worst case.** One shared DirectML session cannot run concurrently, so the prior
  runs sat behind a mutex at ~56 ms each, while the CPU ONNX prior (270 ms per run but 24 in parallel)
  costs ~11 ms of wall time per tile. Moving only the prior to the GPU made the full decrypt *slower*
  (110 s with GPU ADMM, vs 82 s with the CPU prior). Tile workers now hand their input to a batcher:
  same-size tiles are stacked into one `N x 3 x h x w` run (ADMM on the CPU: 131 s serial -> 74 s
  batched; all-GPU: 110 s -> 54.7 s).
- **The Swin2SR graph is mostly bookkeeping.** The HuggingFace export has 19,113 nodes; ~6,500 are
  `Constant`, the rest are `Shape`/`Gather`/`Range`/`Expand`/`Where`/`ScatterND` that compute window and
  mask indices from the input shape on every run. DirectML does not constant-fold them (folding only
  touches CPU-assigned nodes), so each run dispatched ~5,100 tiny DML ops (profile: 183 ms of a 214 ms run,
  real MatMul/Conv ~17 ms).
- Fix: build a CPU session with the free dimensions pinned (`batch_size`, `num_channels`, `height`,
  `width` via `AddFreeDimensionOverrideByName`) at `ORT_ENABLE_BASIC`, which writes a constant-folded
  model: **19,113 -> 2,018 nodes**, outputs identical to 2e-6. DirectML then fuses it into a single
  `DmlFusedNode`. Python lab (16 tiles, 48x56): dynamic 195 ms -> folded 74 ms (12.2 -> 4.6 ms/tile).
  All four dimensions must be pinned: overriding only height/width gave no gain.
- One folded session per tile size (here one size: 48x56 for every prior run), fixed batch **2**; a
  short chunk is padded by repeating its last sample. Per-tile GPU cost is ~4.5 ms for any batch >= 4 and
  6.2 ms for batch 1, so the batch size is a latency/padding trade-off: end-to-end sweep (s): N=1 20.0,
  **N=2 19.3**, N=3 19.8, N=4 20.5, N=8 21.2, N=16 22.5. Prior GPU time 22 s -> 9 s.
- Falls back to the dynamic DirectML session (lazily built) if folding fails or a size cannot be folded.

### 3.7 Startup (6.2 s of serial stall removed)

Measured with `CS_GPU_PROF=1`: before the first tile could use the prior, the process built the dynamic
DirectML session (3.8 s, never used once folding worked) and then the folded one (2.4 s), all workers
waiting.

- No session at all is created for the SR model entry when folding is on; only the file is checked
  (fail-fast contract kept, a bare CPU session on this graph still costs 4.7 s). The folded session
  provides the input/output names.
- The folded session for each tile size is built on a background thread (`cs_onnx_sr_prefetch`) as soon
  as the GPU LR pre-pass sees the size, overlapping the rest of the setup and the LR solve.
- The folded graph is cached in `%TEMP%\cs_fold_cache` (key: model path, size, timestamp, ORT version,
  batch, tile size, precision; atomic publish via `.part` + rename). Warm build 2.4 s -> 0.65 s.
- 19.5 s -> 13.8 s.

## 4. Experiments that did not pay off

| Experiment | Result | Why |
| --- | --- | --- |
| fp16 Swin2SR (naive conversion) | 19.4 dB vs fp32 | final LayerNorm squares token deviations up to ~500 (2.45e5 > 65,504) and overflows |
| fp16 with 1/16 pre-scale of that LayerNorm | 59-65 dB vs fp32, 1.5x faster in isolation | **slower end-to-end** (+0.6-1 s): batch 2 gives no per-tile gain and the prior is no longer alone on the critical path |
| Larger prior batches (8, 16) | 21-22.5 s vs 19.3 s | more padding, bigger latency, same per-tile cost |
| Broker wait 5-60 ms, prior batch target 4-16 | no effect | pipeline is saturated, not latency-limited |
| First tiled GEMM without load prefetch | 81 ms floor vs 17 ms | serialized global loads |
| Per-op fp32 block lists for fp16 | model fails to load | the converter turns the `Pow` exponent constant into fp16 |

The fp16 model is still built by `scripts/make_swin2sr_fp16.py` (pre-scaled LayerNorm; fp32 result
unchanged at 130 dB) and selectable with `CS_GPU_ONNX_PRECISION=fp16`; fp32 stays the default, both files
are kept.

## 5. How it was profiled (and the caveats)

- ROCm's profiler is not part of the Windows install (no `rocprofv3`, `roctracer`, `rocm-smi` in
  `ROCm\7.2\bin`), so there are no hardware counters; claims about being "load-bound" are inferences from
  timings, not measurements.
- `CS_GPU_PROF=1`: `hipEvent` pairs around every launch of `gpu_admm`, aggregated per kernel and printed at
  exit, plus batch-size / tile-size histograms and run time of the prior, and session creation times.
  **The events themselves cost time** (ADMM GPU time reads 14.9 s profiled vs 6.4 s unprofiled), so use the
  profile for proportions and the unprofiled `[gpu-admm]` / wall numbers for absolutes.
- Thread-time accounting (temporary timers in `cs_sr_upscale_tile_2x`, reverted): of ~960 thread-seconds,
  only ~16 s were CPU compute; ~98 % of worker time is spent blocked on the GPU, i.e. the CPU is not a
  bottleneck.
- Windows GPU engine counters (`typeperf`) were unreliable for HIP/DirectML work and were not used.
- An ONNX Runtime Python lab (`onnxruntime-directml`) was used for node placement, op profiling, batch
  sweeps, folding and fp16 experiments before touching the C++.
- Pitfalls worth knowing: piping the decrypt through `Select-Object -First 1` kills it on the first match
  (bogus 8-11 s timings once); ROCm 7.2 clang needs the MSVC 14.29 toolset for the `.hip` objects; the
  `.hip` objects must be built before `vcvars64` sets the newer include paths (see `build_gpu.bat`).

## 6. Where the time goes now (~13.8 s)

| Part | Time | Notes |
| --- | --- | --- |
| DirectML prior | ~9 s of GPU time | 1,749 tiles x ~5 ms; one fused DML node, real GPU execution time. Main limit |
| ADMM stage | ~6.4 s of GPU time | mostly overlapped with the prior; 4 GEMM kernels = ~88 % of it, NLM 8 %, ~55-60 us per launch at batches of ~26 tiles |
| LR solve | ~1.3 s | one-off, before the loop |
| Fixed CPU work | ~1.5 s | blending 0.9 s, PNG write 0.24 s, image read, key derivation |
| CPU compute in workers | negligible | threads are ~98 % blocked on the GPU |

Ideas not done: look inside the DirectML graph (needs a GPU profiler, e.g. Radeon GPU Profiler, a download
that has not been made); fuse the two GEMM passes of a transform through LDS to halve the launches; capture
the ADMM inner loop in a HIP graph to cut the 13 ms launch floor; use a larger register block / WMMA for the
GEMM; skip prior tiles that the segment gate already sends to the CPU cs-sr path earlier; cache or persist
the folded graph next to the model instead of in `%TEMP%`.

## 7. Switches

| Variable | Default | Effect |
| --- | --- | --- |
| `CS_GPU_TILES=1` | off | GPU LR pre-pass; with it, the flat tile loop (`CS_GPU_WAVES=1` keeps the wavefront) |
| `CS_GPU_ADMM=1` | off | GPU ADMM stage |
| `CS_GPU_ONNX=1` | off | `--sr-prior-model` on DirectML |
| `CS_GPU_ONNX_FOLD=0` | on | disable shape folding (dynamic-shape session) |
| `CS_GPU_ONNX_FOLD_N` | 2 | batch of the folded session |
| `CS_GPU_ONNX_PRECISION=fp16` | fp32 | use `<model>_fp16.onnx` |
| `CS_GPU_ONNX_BATCH` | 16 | prior batcher target (only used by the dynamic path) |
| `CS_GPU_ADMM_WAIT_MS` | 30 | broker batch wait |
| `CS_GPU_WORKERS` | 64 | cap of GPU worker threads |
| `CS_GPU_PROF=1` | off | per-kernel / batch / session profiling at exit |

Reproduce the table in section 1: build with `build_gpu.bat main`, set the three switches
(`CS_GPU_TILES`, `CS_GPU_ADMM`, `CS_GPU_ONNX` = 1), run the decrypt command above; the first run builds and
caches the folded graph, so time the second.
Unit checks: `tests\build_gpu_tiles_test.bat`, `tests\build_gpu_admm_test.bat run`.

## 8. Second round: the ycc + anchors decode (6.1 s -> 2.1 s, bit-identical)

Workload: `IMG_3690` seg8-tile r0.9 ycc420 container + 300k HR anchors (sidecar, `CS_HR_NO_POSTHOC=1`),
`decrypt --no-preview --tv 0 --photo-upscaler cs-sr --sr-pnp admm --hr-weight 1` (auto grid: 63 x 63 tiles,
overlap 24, 8 ADMM rounds x 24 FISTA steps on 42.7 Mpx of overlapped HR tiles). Unprofiled wall time
including process start, median of 7. Every step was checked byte for byte against the previous output.

| Step | Wall time | What changed |
| --- | --- | --- |
| start | 6.1 s | |
| anchors sorted once, per-tile binary search; parallel SR prep | 4.55 s | each of 3969 tiles scanned all 300k anchors (1.6 s) |
| one ADMM batch for anchored + anchor-free tiles | | per-tile step (`Epi::hrt`) instead of two serial batches |
| NLM: sliding 7-row window of row sums, weight table | 3.8 s | exact integer SSD; NLM 1.13 -> 0.48 s |
| LR solve DCTs on the tiled GEMM | 3.6 s | `gpu_tiles` still used the naive one-thread-per-output kernels |
| seg calm: one-pass low-density test | 3.25 s | 38 classes x 4 full-frame passes for nothing (no class was low) |
| parallel tile prep, HR tiles taken without copy | | |
| HIP runtime warm-up thread at decode start | 3.13 s | overlaps imread / KDF / unpack |
| fixed-step ADMM without dX/dZC/dWS (weights computed in place) | 2.99 s | -0.5 GB device memory, less epilogue traffic |
| fused GEMM pairs (`k_gemm2`, LDS stripe) | | no gain (FMA-bound), kept: half the launches |
| parallel PNG writer (`png_fast.cpp`, AOCL zlib) | 2.74 s | 272 -> 24 ms, same size, same pixels |
| seg luma draw: class bands/pools in parallel, chroma draws concurrent | | 127 -> 64 ms |
| NLM: dot4 cross term, sums of squares in registers | 2.45 s | NLM 0.48 -> 0.27 s |
| device arena, pooled HR tiles, deduplicated blend weights, parallel host packing | 2.26 s | allocations and repacking |
| AVIR dither as an anti-diagonal wavefront | | 35 ms -> a few ms |
| marker event around each ADMM launch | 2.13 s | ADMM span 454 -> 419 ms (fast settings), measured, not explained |

Faster settings (not bit-identical; same build, all decode-side flags, quality vs the original image):

| Flags | Wall | PSNR | SSIM | dE2000 p95 |
| --- | --- | --- | --- | --- |
| (auto) | 2.13 s | 36.120 | 0.9285 | 3.444 |
| `--fista-iters 12` | | 36.118 | 0.9284 | 3.445 |
| `--fast` (= overlap 8, `--fista-iters 12`; explicit flags win) | 0.95 s | 35.996 | 0.9283 | 3.486 |
| `--sr-pnp-iters 4` (fewer ADMM rounds) | | 35.794 | 0.9254 | 3.552 |

Notes:
- Fewer ADMM rounds cost real quality; fewer inner FISTA steps and a smaller overlap barely do.
- The GEMMs are bound by the FMA loop (~9 TFLOP/s): fusion, fewer launches and less memory traffic did not move
  them. Halving their work needs the even/odd DCT factorisation, which changes rounding.
- `build_gpu.bat` now needs AOCL-Compression (`C:\Program Files\AMD\AOCL-Windows\amd-compression`) for the PNG
  writer; only the main exe defines `CS_FAST_PNG`.
- Where the 0.96 s goes: process start + PNG read + KDF ~80 ms, unpack + tile prep ~125 ms, LR stage ~80 ms,
  SR setup ~70 ms, ADMM ~420 ms (GEMMs ~240, NLM ~150), composite + write ~50 ms, frees + exit ~60 ms.
