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
| `CS_GPU_SWIN=0` | on | keep `--sr-prior-model` on ONNX Runtime instead of the HIP network (section 9) |
| `CS_GPU_ADMM_WMMA=0` | on | fp32 `k_gemm2` FISTA loop instead of `k_fista_w` (section 10) |

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

### 8.1 Second-round baseline profile (6.12 s)

Stage marks (temporary timestamps, each after `hipDeviceSynchronize()`), one profiled run:

| Stage | Time | Runs on |
| --- | --- | --- |
| SR GPU pass (two serial ADMM batches 2.27 s: NLM 1,128 ms, step GEMM 460, other GEMMs 184-222 each) | 2,540 ms | GPU |
| SR host prep (per-tile scan of all 300k anchors) | 1,620 ms | CPU |
| LR solve (incl. ~130 ms HIP runtime start) | 707 ms | GPU |
| Seg boundary calm (38 classes x 4 full-frame passes; no class was low, nothing changed) | 351 ms | CPU |
| PNG write (OpenCV, one thread) | 268 ms | CPU |
| Composite (incl. repacking and 128 MB upload) | 131 ms | GPU |
| Tile prep (serial) | 130 ms | CPU |
| Unpack: sample-position draws | 124 ms | CPU |
| Process start, PNG read, exit | 123 ms | CPU |
| Post-SR tile copies | 91 ms | CPU |
| Key derivation (PBKDF2) | 32 ms | CPU |

### 8.2 What did not work (second round)

| Attempt | Measured | Lesson |
| --- | --- | --- |
| NLM sliding row sums, fully unrolled | 2x slower (1.13 -> 2.33 s) | 256 VGPRs, 284 B/lane spills: the compiler hoisted all 154 LDS loads. A shifting 7-entry ring + `__builtin_amdgcn_sched_barrier` per row fixed it (105 VGPRs) |
| NLM with an LDS table of window sums of squares | no gain | LDS bandwidth/occupancy bound, not load-count bound; sums of squares in registers (dot4 of bytes with themselves) gave 0.27 s |
| Sparser scheduling barriers in NLM | no gain | same limit |
| Fused GEMM pairs (`k_gemm2`) | within noise (943 vs 965 ms) | GEMMs are bound by the FMA loop (~9 TFLOP/s), 178-185 VGPRs; kept for half the launches |
| One flush event after each ADMM launch | no gain | only a marker before AND after each launch reproduces the profiled speed |
| 4 GB device arena | first GPU stage waited ~30 ms | reservation must finish within the CPU-only window; 2 GB does |
| First wavefront dither | 0.002% of values off by up to 3 | the serial kernel's `L[j+1] += noise*c` is contracted to one FMA; carrying the noise and `__fmaf_rn` at the use restored bit-exactness |

Use `-Rpass-analysis=kernel-resource-usage` (compiled with the 14.29 MSVC env, as in `build_gpu.bat :hipobj`) before reasoning about instruction counts: two of these failures were register pressure and occupancy.

### 8.3 Final-build timings per setting (median of 3, wall incl. process start)

| Flags | Wall | PSNR | SSIM | dE2000 p95 |
| --- | --- | --- | --- | --- |
| (auto) | 2.135 s | 36.120 | 0.9285 | 3.444 |
| `--fista-iters 12` | 1.622 s | 36.118 | 0.9284 | 3.445 |
| `--fista-iters 8` | 1.446 s | 36.047 | 0.9282 | 3.461 |
| `--manual --tiles 63 --iterations 6 --overlap 12` | 1.603 s | 35.987 | 0.9274 | 3.501 |
| `--manual --tiles 63 --iterations 6 --overlap 8` | 1.234 s | 35.999 | 0.9283 | 3.486 |
| `--fast` | 0.953 s | 35.996 | 0.9283 | 3.486 |
| overlap 8 + `--fista-iters 8` | 0.887 s | 35.912 | 0.9281 | 3.503 |
| overlap 4 + `--fista-iters 12` | 0.882 s | 35.930 | 0.9279 | 3.504 |
| `--tiles 48`, overlap 8, `--fista-iters 12` | 0.894 s | 35.940 | 0.9267 | 3.534 |
| `--sr-pnp-iters 4` | 1.503 s | 35.794 | 0.9254 | 3.552 |
| `--sr-pnp-iters 2` | 1.192 s | 34.378 | 0.9123 | 4.045 |

### 8.4 The neural prior is not in these timings

The workload above runs no network: `--sr-pnp admm` without `--sr-prior-model` uses the NLM prox and the AVIR
warm start. With the Swin2SR prior (`--sr-prior-model models\swin2sr\onnx\model.onnx`, DirectML, folded fp32
graph) the batched SR pass declined every tile and the per-tile broker path ran instead:

| Decode | Wall | PSNR | SSIM | dE2000 p95 |
| --- | --- | --- | --- | --- |
| default, Swin2SR prior | 18.9 s | 35.833 | 0.9284 | 3.439 |
| `--fast`, Swin2SR prior | 10.8 s | 35.614 | 0.9281 | 3.476 |

Section 9 moves the network into the batched pass (3.1 s / 1.42 s).

### 8.5 How the second round was profiled

- Stage marks: ~30 temporary `steady_clock` timestamps (`CS_PROFILE`-gated), each preceded by `hipDeviceSynchronize()`
  in GPU code; removed before the final measurements (the syncs themselves cost ~10 ms).
- Kernel split: `CS_GPU_PROF=1`. Note that it changes timing (it now matches the marker events, see the table above).
- Exactness: every step compared byte for byte against the previous decode PNG (`max |diff| = 0`).
- Timing: process wall time from PowerShell, median of 5-7 runs; drift between sessions was up to ~40 ms, so close
  comparisons were run back to back.

## 9. Third round: the Swin2SR prior on HIP (18.9 s -> 3.1 s)

Same workload as section 8 plus `--sr-prior-model models\swin2sr\onnx\model.onnx` (Swin2SR-lightweight x2).
Every LR tile (48 x 56, 3969 of them) goes through the network once to give the warm start of its ADMM solve.

| Decode | Before (DirectML, per-tile path) | After (HIP, batched pass) | PSNR | SSIM | dE2000 p95 / p99 |
| --- | --- | --- | --- | --- | --- |
| default | 18.9 s | 3.1 s | 35.833 (same) | 0.9284 (same) | 3.439 / 6.016 (was 6.015) |
| `--fast` | 10.8 s | 1.42 s | 35.613 (was 35.614) | 0.9281 (same) | 3.477 (was 3.476) |

The final image differs from the DirectML decode by 0.08 levels on average (0.4% of values by more than one level,
max 7, where a rounding edge in the warm start sends ADMM elsewhere).

### 9.1 Why DirectML could not be tuned

DirectML ran the folded graph (2018 nodes, 2.1 MFLOP per LR pixel) at ~1.5-2.1 us per LR pixel whatever the batch,
patch size or precision: ~1.4 TFLOP/s, about 3% of the GPU. 10.7 M LR pixels of tiles cost 17.3 s of the 18.9 s.
The graph is many small ops on 60 channels; each round-trips memory. The fix was a hand-written network.

### 9.2 Design

- `scripts/export_swin2sr_weights.py` reads the ONNX graph, recovers the anonymous MatMul weights by their bias
  consumers, precomputes the relative-position bias (`16*sigmoid(cpb MLP)`, `[6][64][64]`) and the clamped logit
  scales, and writes `<model>.swin2sr.bin` (5.9 MB). `--check` runs its PyTorch forward of the same math against
  ONNX Runtime (max diff 1.6e-6).
- `gpu_swin.hip` packs the weights once into fp16 WMMA B fragments and runs the network on device-resident tiles:
  `gpu_sr::run` hands it the LR buffer it already uploaded and gets the 8-bit HR warm start back in the buffer the
  AVIR kernels would fill, so the rest of the batched pass (YCrCb, ADMM, merge) is unchanged.
- Four kernels per tile batch (chunks of 2 M LR pixels): `k_stem` (first conv + patch embed + LN), `k_layer` x 24
  (one 8 x 8 window per workgroup: q/k/v, cosine attention, projection, MLP, both post-norms), `k_conv<4,1>` x 4
  (stage conv3x3 + conv1x1 + residual, implicit GEMM over a 10 x 10 fp16 halo) and `k_conv<4,0>` / `k_conv<1,2>`
  (conv after body, then the 12-channel conv with pixel shuffle, mean and the 8-bit store).
- The matrix products use `v_wmma_f32_16x16x16_f16` (fp16 in, fp32 accumulate); the residual stream, LayerNorm,
  softmax and GELU stay fp32. Channels are padded 60 -> 64, head dims 10 -> 16, the MLP 120 -> 128.
- Accuracy against ONNX Runtime CPU on real tiles: 0.41% of 8-bit outputs off by one, none by more. Emulating the
  fp16 rounding in PyTorch first (`HMODE`, section 9.4) predicted this before any kernel was written.

### 9.3 Steps (512 tiles, harness timing per network pass)

| Step | Time | Notes |
| --- | --- | --- |
| fp32 VALU port (one thread per pixel / window) | 1,560 ms | 2 TFLOP/s; ~12.7 s for the image |
| WMMA layer and convs (fp16 inputs) | 228 ms | 6.8x |
| one head group instead of two (more LDS, fewer barriers) | 221 ms | grouping to raise occupancy was slower |
| DPP `row_xmask` reductions instead of `__shfl_xor` (`ds_bpermute`) | 192 ms | ~900 reductions per layer |
| hardware `rcp` / `rsq`, tanh-form GELU | 148 ms | IEEE division and `erff` dominated the VALU work |
| fused stem (first conv + embedding, coalesced) | 133 ms | 16 -> 1.9 ms |

In the decode the network then takes ~1.0 s of the 2.48 s batched SR pass (ADMM 1.24 s). Removing the
DirectML session check at startup (`cs_sr_gpu_prior_ready`) saved another ~70 ms.

### 9.4 What went wrong on the way

| Symptom | Cause | Lesson |
| --- | --- | --- |
| warm start 96% off ORT, up to 105 levels | the loader read 24 layers then 4 stage convs; the file interleaves them per stage | a NumPy forward that reads the `.bin` with the loader's own offsets found it in one step |
| WMMA output rows 12-15 wrong, only in the up conv | the compiler sank the last WMMA into the `if (r16 < 12)` store branch; WMMA with lanes 12-15 off drops their A rows | `pin()` (empty `asm volatile` on the accumulator) after every product loop |
| VGPRs 227 (layer), 256 + spills (stage conv) | hoisted fragments | did not matter here: weights read from LDS instead of global memory gained only 6% |

Ruled out by measurement: global weight loads (6%), the softmax VALU (14 ms of 113), smaller head groups for
occupancy (slower), `#pragma unroll 1` on the conv taps (no change).

### 9.5 Switches and files

- `CS_GPU_SWIN=0` keeps the prior on ONNX Runtime; without the `.swin2sr.bin` the decode falls back to it too.
- The HIP path covers `--sr-pnp admm` on the batched pass without a segment gate (`seg_lr`, the BGR pipeline) and
  tiles up to 256 px a side (one ONNX patch), sides multiples of 8.
- `CS_GPU_SR_VERIFY=<n>` compares every n-th tile's warm start with `cs_onnx_sr_upscale_bgr` and the final tile
  with the per-tile CPU path: 0.42% of warm-start values off by one (max 1), final tiles 0.071 levels mean.
- Export once per model: `py -3.12 scripts\export_swin2sr_weights.py [model.onnx] [--check]`.

## 10. Fourth round: the ADMM FISTA loop on WMMA (prior decode 3.1 s -> 2.34 s)

### 10.1 Where the 3.19 s went (temporary stage marks, device synced at each GPU mark)

| Stage | Time |
| --- | --- |
| imread, key derivation + header, unpack, tile prep (CPU) | 211 ms |
| LR solve (GPU) | 304 ms |
| SR host prep, upload | 47 ms |
| Swin2SR warm start | 1,120 ms |
| ADMM (FISTA GEMMs ~860 ms, NLM ~260 ms) | 1,319 ms |
| download, composite, PNG write | 67 ms |

Keeping the tiles on the device between the LR solve, the SR pass and the composite would save only the
uploads and downloads (~50 ms), so it was not done. The two GPU stages are 2.44 s of 3.19 s.

### 10.2 The FISTA loop

The fixed-step loop ran 2 launches per step (`k_gemm2`, four fp32 DCT products with fused epilogues) over
192 steps. What was tried:

| Attempt | ADMM GEMM time | Notes |
| --- | --- | --- |
| fp32 `k_gemm2` (before) | ~860 ms | |
| WMMA form of `k_gemm2`, operands split into hi + lo fp16 (3 products) | 1,250 ms | slower |
| same, plain fp16 operands | 748 ms | quality unchanged: 35.832 dB vs 35.833 |
| run all steps on chunks of 128-512 tiles (cache-sized state) | slower at every size | not memory bound; smaller grids underfill the GPU |
| one block per tile for the whole round (`k_fista_w`) | 879 ms | 256 VGPRs, 848 B/lane of spills |
| same, 2 column tiles per wave, 768-thread bound, no runtime-selected register arrays | **184 ms** | 226-240 VGPRs, no spills |

`k_fista_w` (gpu_gemm.hpp) keeps y, x_prev, the ADMM anchor DCT, the HR anchors and the frequency weights in
registers in WMMA output layout; each step is four fp16 products (DCT tables pre-packed as fragments) through
46 KB of LDS, with 4 barriers. The residual's 2x2 average is a DPP lane ^ 1 plus a lane ^ 16 exchange. It writes
the round's pixel image directly, replacing the per-round inverse DCT too. ADMM 1,319 -> 557 ms; NLM (256 ms)
is now its largest part. `CS_GPU_ADMM_WMMA=0` restores the fp32 loop (bit-identical to earlier builds).

The same register-pressure lesson as section 9.4: a lambda called with `last ? xp : y` (an array chosen at run
time) forced the state arrays to scratch; copying z into y before the last pass removed most of the spills.

### 10.3 The network

q | k | v now come out of one 12-tile product (64 columns each) instead of 18 per-head padded tiles, with the
q / k norms computed once from LDS and applied to the scores; accuracy unchanged (0.41% of outputs off by one
level vs ONNX Runtime), time within noise (~1.06-1.12 s in the decode). The layer kernel holds 2 workgroups
per WGP (56 KB LDS); with WMMA at ~25% of peak, it is latency bound.

### 10.4 Results (median of 3, wall incl. process start)

| Decode | Before | After | PSNR | SSIM | dE2000 p95 |
| --- | --- | --- | --- | --- | --- |
| Swin2SR prior | 3.10 s | 2.34 s | 35.834 (was 35.833) | 0.9284 | 3.439 |
| Swin2SR prior, `--fast` | 1.42 s | 1.27 s | 35.612 (was 35.613) | 0.9280 (was 0.9281) | 3.476 (was 3.477) |
| no prior | 2.13 s | 1.35 s | 36.121 (was 36.120) | 0.9285 | 3.445 (was 3.444) |
| no prior, `--fast` | 0.95 s | 0.76 s | 35.994 (was 35.996) | 0.9283 | 3.486 |

The decode is no longer bit-identical to the fp32 build (fp16 products in FISTA); the difference is ~0.1
level on average.

## 11. Intrinsics vs standard forms, and rocWMMA

Each candidate swap was compiled to device assembly (`--cuda-device-only -S`) and compared per kernel with
the baseline, then timed (median of 5 decodes per setting) with the decode PNGs compared byte for byte.

| Intrinsic | Standard form | Result |
| --- | --- | --- |
| `__float2int_rn(x)` | `(int)rintf(x)` | identical machine code: swapped |
| `__fmaf_rn(a, b, c)` | `fmaf(a, b, c)` | identical: swapped (still a guaranteed fused FMA, which the bit-exact dither needs) |
| `wave_sync()` (wavefront fences + `__builtin_amdgcn_wave_barrier`) | `__syncwarp()` | identical (HIP defines it as those calls): swapped |
| `__builtin_amdgcn_rsqf(x)` | `rsqrtf(x)` | adds a denormal guard (compare, select, scale): network +1-2%, kept |
| `__builtin_amdgcn_rcpf(x)` | `__fdividef(1.0f, x)` | compiles to the IEEE division sequence: kept |
| `__builtin_amdgcn_alignbyte(hi, lo, sh)` | 64-bit shift / `__funnelshift_r(lo, hi, 8 * sh)` | extra shifts in the NLM inner loop: `k_nlm` +8% / +2%, kept |
| `ld16` (two 8-wide loads + shuffle) | `memcpy` into the 16-wide vector | changes how the WMMA loops unroll: network warm start 1,070 -> 1,280 ms, `k_fista_w` 185 -> 211 ms, kept |

Decodes after the kept swaps: 2,325 / 1,266 / 1,385 / 803 ms (prior, prior `--fast`, no prior, no prior
`--fast`) against 2,302 / 1,241 / 1,364 / 801 ms before, run-to-run noise on identical machine code; all
four PNGs byte-identical.

rocWMMA 2.2 (ROCm 7.2) builds and runs on gfx1100 with this toolchain. A 16 x 16 x 16 test agrees with the
builtin to float rounding, and `load_matrix_sync(..., col_major, 16)` reads the packed weight tiles as they
are. Porting `k_stem` (accumulators stored to LDS, bias + LayerNorm per pixel from LDS) gave the same output
(max diff 3e-6) but 5-10% more time (15.4-16.7 -> 17.2-17.5 ms per 3969 tiles), so it was reverted. The
other kernels fit rocWMMA less well: `k_conv`'s 16-pixel A tiles span two halo rows (no single leading
dimension), and `k_layer` / `k_fista_w` read accumulators by lane in every epilogue (LayerNorm, softmax,
the 2x2 exchange, register-resident FISTA state), which rocWMMA keeps opaque. The port stays open
(`TODO.md`).
