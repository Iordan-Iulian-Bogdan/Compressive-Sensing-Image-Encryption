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
| `CS_GPU_LR_FUSED=0` | on | launch-per-step LR FISTA loop instead of `k_lr_fista` (section 12) |
| `CS_ENH_BAND` | 128 | row band of the enhance net's full-res layers; 0 = whole image at once (section 12) |

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

## 12. Fifth round: host prep, the LR FISTA loop, banded enhance net (bit-identical)

Production decode (prox-off, 450k anchors, c32 net), IMG_3690. Measured with the enhance-net training
sharing the GPU, so absolute GPU times are high; every step was checked byte for byte against the
production exe (8 photos x r0.3/0.5/0.7/0.9: 32/32 identical) and `cs_tests` / `gpu_tiles_test` pass.

Where the time was (`CS_PROFILE=1 CS_GPU_PROF=1`, which now also prints the LR build / loop / download
split, the ADMM host phases, the enhance allocation and the warm-up thread):

| Stage | Before | Cause |
| --- | --- | --- |
| LR FISTA, luma batch loop | 108-148 ms | not launch-bound (fusing alone gave ~10%): LDS bank conflicts and one block per CU |
| LR build (layout + upload) | 30-70 ms per batch | per-sample tile ids, W / WS arrays, separate row / column indices |
| SR host layout | 44 ms | AVIR 2x tables rebuilt serially per decode (R + C resizes per tile size) |
| ADMM host layout + upload | 45 + 21 ms | dense 43 MB anchor map built on the CPU, copied again, uploaded |
| Enhance "upload + planes" | 82 ms | 74-76 ms of it allocation: ~4.9 GB of activations at c32, beyond the 2 GB arena |

What changed:

| Change | Files | Effect |
| --- | --- | --- |
| `k_lr_fista`: one block per tile runs every step and reweight; Y / SC column-major and TMP in LDS, the DCT tables (plus transposed copies) through the cache, every operand one 16-byte load | `gpu_tiles.hip` | luma loop 87 -> 30-40 ms (fused, tables in LDS: 87 ms; unfused: 97-148 ms) |
| lean batch for the fused path: one packed sample index, no tile ids, no W / WS | `gpu_tiles.hip` | upload 11-28 -> 8-17 ms per batch |
| AVIR table probes in parallel (OpenMP) | `gpu_avir_tables.cpp` | SR host layout 44 -> 12 ms |
| sparse HR anchors (index, value) scattered on the device over a sentinel fill | `superres.cpp`, `gpu_sr.*`, `gpu_admm.*` | ADMM host layout 45 -> 15 ms, upload 21 -> 10 ms |
| enhance full-res layers in 128-row bands (X, head, e0 into a full S0; up1, fuse0, tail per band; each band recomputes its halo rows) | `gpu_enh.hip` | allocation 93 -> 29 ms |
| quarter-res buffers carved out of idle half-res ones | `gpu_enh.hip` | allocation 29 -> 17-20 ms |

`k_conv` takes per-buffer row bases (`ConvP::ia / ib / ob / rb`) and an output row window (`y0`, `ny`); all
zero and `ny = OH` is the whole-image conv as before. Band heights of 36, 64, 128 and the whole image give
the same bytes; the network time stayed within the run-to-run noise of the shared GPU (130-160 ms).

Result: median wall time of the 32 decodes (incl. process start) 1.38 -> 1.12 s (-19%), total 44.5 -> 36.2 s.

Still open:
- The warm-up thread needs ~258 ms (HIP runtime) + ~65 ms (2 GB arena). At low ratios the CPU reaches the
  first LR batch sooner and waits ~90 ms for it (r0.3; none at r0.9).
- LR layout (10-30 ms per batch) and the CPU warm start of the LR planes (`prep`, ~40 ms) are host work
  that could move to the device.
- The enhance network itself (~130-145 ms at c32) and `k_fista_w` (~60 ms) are unchanged.

## 13. Sixth round: arena blocks, overlapped host work, cs-sr -> composite on the device (bit-identical)

Same production decode as section 12. The GPU kernels were busy for only ~0.35 s of the ~1.0 s decode; the rest
was serial host work, transfers, allocations and start-up. Checked byte for byte against the section 12 exe
(16729dd): 8 photos x r0.3/0.5/0.7/0.9, 32/32 identical; a CPU-path decode (`--device cpu`, GPU stages off) is
identical too; `cs_tests` passes and `build_cpu.bat` builds.

What decided the changes was a timeline of one decode: ms since process creation at every stage edge (a temporary
instrumentation, not kept in the code). IMG_3690 r0.9, before, with the training sharing the GPU:

| ms | Event | Note |
| --- | --- | --- |
| 0-58 | DLL loading | before `main` |
| 59 | warm-up thread starts | 0.7 ms after `main`: it cannot start earlier |
| 70 / 307 | HIP runtime / device context ready | `hipGetDeviceCount` 11 ms, `hipMemGetInfo` ~235 ms |
| 357 -> 378 | first LR batch waits | for the 2 GB arena reserve, which held the arena lock for its whole `hipMalloc` (72 ms); ~115 ms at r0.3 |
| 533-584 | LR tail + tile merge (CPU) | inverse DCT, chroma resize, YCrCb -> BGR per tile |
| 585-615 | cs-sr host prep (CPU) | samples, anchor lists, ADMM groups: independent of the LR result |
| inside cs-sr | ADMM layout 21 ms, alloc 20 ms | four 163 MB and two 6 MB buffers overflow the arena (peak ~2.7 GB) |
| 884-909 | composite | re-uploads the 128 MB of HR tiles the cs-sr stage had just downloaded |
| 910-935 | `gpu_enh::load` | read, pack and upload the 7 MB weight file |
| inside enhance | alloc 20 ms | two 372 MB buffers overflow the arena (peak ~2.45 GB) |
| 1171 -> ~1219 | process exit | after `main` returns |

What changed:

| Change | Files | Effect |
| --- | --- | --- |
| The arena is a list of blocks (pieces go to the first block with room; each block keeps the stack discipline); `arena_reserve` adds a block and runs `hipMalloc` outside the lock | `gpu_tiles.hip`, `gpu_common.hpp` | stages keep allocating while a block is being reserved |
| Warm-up reserves a 1 GiB block first (enough for the LR solve), then the rest; total min(free / 2, 3 GiB), was 2 GiB | `gpu_tiles.hip` | the first LR batch no longer waits for the reserve; ADMM and enhance allocation 20 -> 0 ms each |
| `cs_enhance_prefetch`: the decrypt entry loads the HIP weights on a side thread (next to the warm-up); `gpu_enh::load` takes them from plain `hipMalloc`, not the arena (a long-lived piece would strand the space below it) | `enh_net.*`, `gpu_enh.*`, `image_decryption.cpp` | 25 ms before the enhance stage -> 0 |
| `cs_sr_gpu_batch_prepare`: the host part of the cs-sr batch that does not read the LR pixels; `decrypt_tiles_ycc420` builds the tile list before the LR solve and runs it on 8 OpenMP threads from `cs_gpu_solve_tiles_ycc`'s new `on_gpu` hook (when the LR batches reach the GPU); `cs_sr_gpu_batch` only takes the solved tiles (a tile whose size changed is dropped) | `superres.cpp`, `helper_functions.hpp`, `gpu_decrypt.*`, `image_decryption.cpp` | `sr enter` -> `sr gpu enter` ~28 -> 1 ms. With all 24 threads it slowed the second LR batch (97 -> 118 ms); 8 leave it at ~91 ms |
| `resident_put` / `resident_take`: `gpu_sr::run` keeps its packed output tiles on the device in a one-slot handoff keyed on the host range it downloaded them to; `gpu_composite::blend_tiles` takes that buffer instead of uploading the same bytes, and empties the slot on every return | `gpu_tiles.hip`, `gpu_common.hpp`, `gpu_sr.hip`, `gpu_composite.hip` | composite 21-28 -> 13-15 ms; the host tiles are still downloaded, so every fallback sees the same bytes |

Result on an idle GPU: median wall time of the 32 decodes (incl. process start) 1.03 -> 0.98 s, total
32.9 -> 31.3 s (-50 ms per decode, -4.9%); the network time is unchanged (144 / 142 ms median). With the enhance
training sharing the GPU the same A/B gave 1.18 -> 1.14 s (-28 ms per decode). The single-decode timelines
suggested ~100 ms: part of what left the critical path (the cs-sr prep, the second arena block) still competes
for CPU or device time with what remains on it.

Tried: starting the warm-up earlier (it already starts 0.7 ms after `main`; the 58-88 ms before `main` is DLL
loading).

Still open:
- LR -> cs-sr on the device (~60 ms): the CPU inverse DCT and tile merge (~45 ms) and cs-sr's tile copy and
  upload (~16 ms). The inverse DCT is `cv::dct` today; a GPU one rounds some pixels differently, so the decode
  would no longer be bit-identical.
- Composite -> enhance on the device (~4 ms of upload): the seg boundary calm edits the image on the host in
  between at low ratios.
- The cs-sr prep state (`CsSrBatchPrep`) is freed when `decrypt_tiles_ycc420` returns (~12 ms of frees; before,
  inside `cs_sr_gpu_batch`, so not a new cost).
- Process exit takes ~48 ms after `main` returns.

## 14. Seventh round: LR -> cs-sr on the device (no longer bit-identical)

Section 13's still-open list, measured on an idle GPU (IMG_3690 r0.9 with `CS_GPU_PROF=1`, then the 32-decode
A/B against 4730b84; production decode path, c32 net).

| Change | Where | Effect |
| --- | --- | --- |
| `gpu_tiles::solve_ycc_bgr`: the LR planes' inverse DCT (the batch's dense GEMM, `Batch::synth`) and the tile merge (`k_ycc_merge`: chroma bilinear as `cv::resize` INTER_LINEAR, convertTo 8 bit, YCrCb -> BGR in OpenCV's fixed point) run on the device. The BGR tiles land back to back in one host buffer, which the `mats_out` tiles now view, and the device copy goes into the resident slot. `gpu_sr::run` takes it when its LR tiles are exactly that buffer, instead of gathering and uploading them. The output buffer is allocated before the solves, so the arena reclaims their scratch above it. `CS_GPU_LR_HOST_TAIL=1` keeps the host tail. | `gpu_tiles.*`, `gpu_decrypt.*`, `gpu_sr.hip`, `gpu_common.hpp` | LR stage 253 -> 225 ms (host tail 11 ms and merge ~34 ms gone; the 64 MB coefficient download becomes a 32 MB BGR one). cs-sr host copies 11.5 -> 4.8 ms, upload 2.6 -> 0.3 ms. Solve phase 450 -> 413 ms |
| The cs-sr prep state (`CsSrBatchPrep`) is handed out of `decrypt_tiles_ycc420` and freed with the tile buffers in `tile_frees` | `image_decryption.cpp` | solve -15 ms in the phase profile, ~-6 ms end to end |

Result: median wall time of the 32 decodes 0.96 -> 0.93 s, total 31.0 -> 29.9 s (-34 ms per decode, two runs).
The decode is no longer bit-identical to the host tail. A GPU-summed inverse DCT puts a few LR pixels on the other
side of a rounding step, and ADMM plus the net carry that into 0.5-8% of the output pixels, at most 3-11 levels.
On the 8 photos x 4 ratios the quality stays put:

| ratio | dPSNR | dSSIM | dDE95 | dDE99 | max abs diff | pixels that differ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | -0.0000 | -0.00000 | +0.0000 | +0.0001 | 8 | 3.00% |
| 0.5 | -0.0000 | +0.00000 | +0.0000 | +0.0000 | 9 | 2.47% |
| 0.7 | -0.0000 | +0.00000 | -0.0000 | +0.0000 | 9 | 2.45% |
| 0.9 | -0.0000 | -0.00000 | +0.0000 | +0.0000 | 11 | 2.20% |

(Per photo, |dPSNR| is at most 0.0002 dB with both signs; IMG_1297 and D06_rnd_39 have the most changed pixels.)

Looked at and left:
- Process exit: the 48 ms was mostly the Python harness. `main`'s own teardown is ~7-12 ms; ending the
  process early is not worth it.
- Composite -> enhance on the device (~3-4 ms of upload): at low ratios the seg boundary calm edits the host
  image in between, so a device copy can be stale. Not done.

The start-up wait (timestamps on one clock, IMG_3690 r0.9, ms after the warm-up thread starts): `hipMemGetInfo`
returns at +319 (the device start itself, ~310 ms), the 1 GiB block at +352, and the LR stage's first upload is
issued at +364 but returns only at +424, right after the 2 GiB `hipMalloc` of the rest (+419): a `hipMalloc` holds
the runtime's lock. The warm-up now reserves the rest in 256 MiB blocks with a 0.5 ms pause after each, so the
waiting upload gets in after one block (`gpu_tiles.hip`). Without the pause the warm-up thread takes the lock
straight back (256 MiB blocks alone: 61 -> 32 ms):

| second reserve | first-upload wait | solve | profile total |
| --- | --- | --- | --- |
| one 2 GiB block | 56-61 ms | 406-417 | 686-696 |
| 256 MiB blocks, 0.5 ms pause | 2-3 ms | 368-370 | 645-652 |
| 384 MiB blocks, 0.5 ms pause | 3.5 ms | 375 | 650 |
| 128 MiB blocks, 0.5 ms pause | 1.3 ms | 403 | 685 (the reserve ends too late for cs-sr) |

(4 photos x 2 at r0.9, c16 net.) The reserve now ends ~115 ms after the LR stage starts, before cs-sr needs it;
the arena misses are unchanged (one 162 MB plane on some photos, with or without the blocks). Against f29297a, 32
decodes on the c16 net: 32/32 byte-identical. Per phase at r0.3 + r0.9: solve -17.5 ms, total -28 ms. Wall time:
-12 ms per decode (median 0.81 s both). r0.3 gains less: its LR stage starts earlier, when the device start
itself (~310 ms) is what it waits for.

Still open:
- Device start-up (~310 ms in `hipMemGetInfo`) now bounds the LR stage's start when the host stages before it
  are short (r0.3).

## 15. Unpack and OpenMP idle waits (bit-identical)

c16 net from here on (`enh_seg8a_x5_ema256_mix450c02_c16_unet.onnx`), production decode path otherwise.

Unpack (`get_sampled_ycc420`, IMG_3690 r0.9, timestamps): label PNG 5 ms, per-class pixel pools 7.6, keyed luma
draws 41.7, keyed shuffle 7.2, HR-anchor payload read 25.8, zeroed planes and scatter 12.3 (105 ms; the chroma
draws already ran alongside).

| Change | Where | Effect |
| --- | --- | --- |
| The seg-luma modes read the HR-anchor payload on a side thread during the luma draw (its offset is the prefix length from the header pad plus m and the chroma count) | `image_decryption.cpp` | 25.8 ms off the critical path |
| Partial Fisher-Yates draws: each batch of 256 picks is drawn first (same generator calls, same order), then the swaps run with the pool entries prefetched 16 ahead; the output lists are reserved | `CS_encryption.cpp` (shared with the encoder) | draws 41.7 -> 21.7 ms |
| Seg-luma and chroma sample scatter in parallel (their positions are distinct) | `image_decryption.cpp` | 12.3 -> 8.5 ms |

Unpack 105 -> 54 ms at r0.9, 65 -> 36 ms at r0.3. Alone it barely moved the wall time: the LR stage then started
earlier and waited on device start-up. Splitting the wall time (QPC on both sides) showed what did:

| Same exe | start-up | main | teardown | wall |
| --- | --- | --- | --- | --- |
| default | 36.9 | 743.7 | 41.4 | 832.5 |
| `OMP_WAIT_POLICY=passive` | 34.0 | 687.9 | 36.2 | 757.2 |
| `KMP_BLOCKTIME=0` | 38.1 | 687.0 | 36.1 | 768.6 |

libomp's workers spin for 200 ms after every parallel region and take the cores the decode's other threads (GPU
driving, the side preps, PNG write) need. `main` now sets `OMP_WAIT_POLICY=PASSIVE` unless one is set. It has to
go into the OS environment block (`SetEnvironmentVariableA`): libomp links its own CRT and reads that block when
it initializes, so `_putenv_s` is not seen. It also has to happen before that initialization:
`gpu_admm_broker.cpp`'s `g_target` was a static initializer calling `omp_get_max_threads()`, which started libomp
before `main`. It now reads it at first use (`ImgReconstruct_backend.cpp`, `gpu_admm_broker.cpp`).

Result against f29297a (section 14's start-up wait fix plus this section): 32 decodes 32/32 byte-identical, median 0.82 ->
0.76 s, total 26.3 -> 24.4 s (-59 ms per decode, -7%). CPU decode (IMG_3690 r0.9) 9.8 -> 9.3 s, identical output.
Encrypt (production format) 1.58 -> 1.51 s. A container from the new encoder decodes byte-identically on the
f29297a exe. `cs_tests` pass; `build_cpu.bat` builds.

Still open:
- Device start-up (~310 ms in `hipMemGetInfo`) bounds when the LR stage can start once the host stages before it
  are short.

## 16. Host layouts in GPU time, submitting queued launches (bit-identical)

Most host work before the LR stage is hidden behind device start-up (~310 ms) now, so moving it to the GPU would
gain nothing; what is left after start-up is host work the GPU waits for.

| Change | Where | Effect |
| --- | --- | --- |
| `gpu_admm::prepare_host_layout`: the ADMM batch's host layout (tile layout, LR sample grid, anchor lists; no z, no NLM strengths) is built separately; `gpu_sr::run` builds it while the AVIR warm start runs and passes it to `solve_pnp_admm_device` | `gpu_admm.*`, `gpu_sr.hip` | ADMM host layout 14 -> 1.4 ms on the critical path |
| `hipStreamQuery(nullptr)` after queuing work the host then does something else next to: the warm start in `gpu_sr::run`, each group in `solve_ycc_bgr`. On Windows the launches otherwise sit unsubmitted until the next synchronizing call | `gpu_sr.hip`, `gpu_tiles.hip` | without it the warm start grew 54 -> 65 ms with the layout moved into it; with it, 50-52 ms |

Per stage (16 decodes, no `CS_GPU_PROF`): LR 219.5 -> 206.4 ms, cs-sr 190.3 -> 180.4, solve 414.6 -> 393.0. Same exe
with and without the two submissions: -34 ms wall. Against d1811cb, 64 paired decodes (ABBA order, 0.5 s between
runs, both exes under fresh file names): wall median 738.7 -> 711.9 ms (-29 ms), in `main` -27 ms; 32/32
byte-identical; `cs_tests` pass.

Measuring note: the exe named `ImgReconstruct_backend.exe` starts ~12 ms slower than a byte-identical copy under
another name (start-up to `main` 32.7 vs 20.6 ms; likely Windows' per-exe launch history). A/Bs that run the new
build under that name and the baseline as a renamed copy understate the gain by ~12 ms; back-to-back GPU
processes also disturb each other.

## 17. Batch encode / decode in one process (2026-11-05, bit-identical)

`encrypt @<pairs.tsv> [options]` and `decrypt @<pairs.tsv> [options]` run a list of jobs (one `input<TAB>output` per
line, `\r\n` accepted) in one process with one set of options (`ImgReconstruct_backend.cpp`). Per-process costs are
paid once: process start (~50 ms), HIP start-up and kernel loads, the `--enhance` weights. `decrypt` also reads the
next container on a side thread while the current one decodes. A failed job is reported (`Error: job failed: <input>`)
and the batch goes on; the exit code is 1 if any failed. Each job prints `[batch] i/n <output> <ms> ms`. Preview
windows are off in a batch (`--show-mask` is ignored).

The decoder's training dumps take `*` for a per-job path: `CS_DUMP_YMASK=*`, `CS_DUMP_YSAMP=*`, `CS_DUMP_HR=*` and
`CS_DUMP_CHROMA=*` write `<output>.ymask.png`, `.ysamp.png`, `.anchors.bin` and `.crmask.png` / `.crsamp.png` /
`.cbmask.png` / `.cbsamp.png` next to each output (an explicit path still works for single decodes).
`Params-test/gen_enh_prod.py` uses this with `ENH_BATCH=N`: per chunk of N photos one batch encode per ratio and one
batch decode, crops cut in a process pool while the next chunk runs.

Measured on the 8 test photos (12 MP), production flags (encode r0.5 with `--hr-detail 450000` and the ranking net;
prox-off decode with c16d2), RX 7900 XT, totals divided by 8:

| Setup | Decode per photo | Encode per photo |
| --- | --- | --- |
| one process per photo | 687 ms | 1352 ms |
| two processes at a time | 523 ms | 877 ms |
| one batch process | 574 ms (515 ms after the first job) | 1146 ms |
| two batch processes (list split in half) | **439 ms** | **732 ms** |

Two batch processes are fastest: one decode's CPU stages (container read, unpack 73 ms, prep 60 ms, PNG write 26 ms)
run while the other's GPU stages do. Decode output is byte-identical to single decodes (8/8). Batch-encoded
containers decode to the same quality as single encodes within the salt's run-to-run noise (IMG_3690, D28_rnd_123,
IMG_9321: within 0.015 dB, SSIM and ΔE-p95 equal). Training-data generation (2 workers, `ENH_BATCH=32`): 68 s per
32-photo chunk per worker, ~2.0 h for the 6,847-photo set against 2.9 h with one process per encode and decode.

## 18. Multi-scale (mode 20) layered decode (2026-11-06)

### 18.1 What the decode does

A mode-20 container (`encrypt --multiscale auto --ratio r`) holds a descriptor (photo size, tile side T = 120, per-tile
scale map 2x..6x) and one seg-mask layer container per scale used. Each layer is the photo shrunk by its scale s. It is
sampled only on the tiles of that scale plus an 8 px margin (`kMsMargin`; seg prefix version 2 also draws chroma only
there). The 2x layer also carries the closed-loop anchors.

The decoder (`decrypt_image_tiled`, the `mode == 20` branch) works in four steps:
1. It decodes every layer with the production ycc path: LR solve, cs-sr, tile merge, calm, enhance nets, data
   consistency.
2. It then composites tile by tile. A tile whose 3x3 neighbourhood holds one scale is that layer resampled (cubic).
   The other tiles blend their neighbours' layers with the weights of a 17x17 box over the scale map, done
   separably on the tile grid.
3. Each layer's output is width / s x 2 (the 2x layer is full size).
4. It writes the PNG.

Workload for every number below:
- the auto-planned r0.3 containers of the 11 dataset photos `IMG_0615`..`IMG_0625` (24.5 MP in total), plus
  production containers of the same photos;
- the production decode command (`--photo-upscaler cs-sr --sr-pnp admm --sr-pnp-iters 2 --tv 0 --hr-weight 1`,
  `--enhance 6bc450_c16d2 --enhance2 6bc450_u2dc`, `CS_ADMM_PROX_OFF=1 CS_GPU_TILES=1 CS_GPU_ADMM=1`);
- an idle GPU (no training running).

Wall times include process start. Each exe was pinned under its own name (`x64/GpuBuild/ImgReconstruct_backend.msNN.exe`)
and timed alternately (ABBA), 1 s apart. On IMG_0621 the production decode takes 1.37-1.41 s.

### 18.2 Result in one table

Per-step ladder, re-timed on 2026-11-06 from the pinned exes, mean of 4 runs. Output is compared with the step
before.

| Step (exe) | IMG_0621 | IMG_0615 | Output vs the step before |
| --- | --- | --- | --- |
| one key, planner v2 (ms14) | 2864 ms | 3054 ms | — |
| + cs-sr skipped on tiles without luma samples (ms15) | 2623 ms (−241) | 2901 ms (−153) | identical |
| + LR solve skipped on tiles without any sample (ms16) | 2274 ms (−349) | 2606 ms (−295) | max 2 levels on 0.09 / 0.22% of pixels |
| + cs-sr host prep skip + parallel PNG writer (ms17) | 1900 ms (−374) | 2118 ms (−488) | identical |
| + one thread per layer, GPU stages under a lock (ms18) | 1797 ms (−103) | 1910 ms (−208) | identical |
| + enhance nets only on the blocks the composite reads (ms23) | 1628 ms (−169) | 1831 ms (−79) | identical |
| + tile merge reads the resident cs-sr output, flat tiles shared (ms24) | 1406 ms (−222) | 1650 ms (−181) | identical |

On all 11 photos (ABBA, 4 runs each), with output identical on 11 / 11 and the production decode identical too:
- ms20 (= ms18) against ms23: wall 1887 -> 1793 ms per photo;
- ms23 against ms24: wall 1786 -> 1620 ms per photo.

IMG_0621 now decodes in 1.40-1.41 s, about production's time (1.37-1.49 s).

### 18.3 The optimizations

**One key derivation per container** (`crypto_utils.cpp`: `cs_derive_key` over `cs_derive_key_pbkdf2`;
`image_encryption.cpp`: `use_key`)
- What changed:
  - Before, every layer had its own salt, so a 5-layer decode ran PBKDF2-HMAC-SHA256 (200,000 iterations) five
    times.
  - Now the encoder derives the outer container's key once and hands it to every layer (`CsEncryptor::use_key`).
    The headers still get their own random IVs.
  - `cs_derive_key` keeps the last (password, salt) -> key under a mutex, so the four layers after the first get it
    from the cache.
- Measured: one derivation takes ~50 ms (hashlib, same parameters), so 5 -> 1 removes ~200 ms. This is not
  re-measured in the decoder: the ladder starts at ms14, which already has it.

**cs-sr skipped on tiles without luma samples** (`decrypt_tiles_ycc420`, `ms_layer` flag)
- What changed:
  - Each layer decodes its whole frame, but only its own tiles and their margin have samples.
  - The other tiles get zero weight in the composite. They are left out of the batched cs-sr tile list
    (`sr_batch`: `if (ms_layer && y_idx[i][j].ri_x_g.empty()) continue;`).
  - The tile loop gives them a bilinear 2x (`cv::resize(..., INTER_LINEAR)`) instead of the solve.
- Gating: `ms_layer` is set from seg prefix version 2 (`get_sampled_ycc420`: `ms_layer = chroma_masked`), so
  production containers never take this path.
- Measured on IMG_0621: cs-sr ran on 5679 of the 15742 tiles of the five layer grids; production runs on all 7921
  of its tiles.

  | Layer | Grid tiles | Tiles with samples |
  | --- | --- | --- |
  | 2x | 7921 | 3116 |
  | 3x | 3600 | 1012 |
  | 4x | 2025 | 849 |
  | 5x | 1296 | 232 |
  | 6x | 900 | 477 |

  - cs-sr summed over the layers went from ~800 to ~300 ms (production: 380 ms).
  - On an idle GPU that is 40 µs per tile here against 49 µs in production. The saving is the tile count, not
    faster tiles.

**LR solve skipped on tiles without any sample** (`decrypt_tiles_ycc420`, before the GPU job list)
- What changed:
  - A layer tile with no luma, Cr or Cb sample is filled with flat gray (128) and marked `gpu_done`.
  - The GPU job loop skips it.
- Output: the fill feeds the tile merge's feather at tile edges, so 0.09-0.22% of the composited pixels move by at
  most 2 levels. They are next to zero-weight tiles.
- Measured: the LR solve summed over the layers went from ~750 to ~310 ms.

**cs-sr host prep skip and parallel PNG writer** (`decrypt_tiles_ycc420`; mode-20 branch: `cs_write_image`)
- What changed:
  - The host prep of the cs-sr batch (`cs_sr_gpu_batch_prepare`: samples, anchor lists, ADMM groups) is built only
    for the batch's tiles, so the empty tiles cost nothing on the CPU either.
  - The composite was written with `cv::imwrite`; it now goes through `cs_write_image` (`png_fast.cpp`, the
    strip-parallel deflate the production path already used).
- Measured: −374 / −488 ms for the two changes together (they were built together). The PNG writer alone measured
  −280 ms when it went in.

**One thread per layer, GPU stages under a lock** (mode-20 branch; `t_ms_gpu_lock`)
- What changed:
  - The layers decode on one thread each.
  - A thread-local `std::mutex*` is set to a shared lock, held from the grid solve (after `prof.mark("prep")`)
    through `prof.mark("enhance")`.
  - The container read, unpack and prep of one layer run while another layer holds the GPU.
  - The GPU stages stay one layer at a time: they share the arena, the resident handoff slot and the HIP stream, and
    the per-stage code is not written for concurrent use.
- Measured: −103 / −208 ms; output identical.

**Enhance nets only on the blocks the composite reads** (`gpu_enh.hip`: `Inputs::active`, `rf_of`, the `k_conv`
early exit; `image_decryption.cpp`: `cs_ms_needed_blocks`, `t_ms_need`; `enh_net.*`: `CsEnhInputs::active`,
`cs_enhance_receptive_field`)
- The problem: each layer ran both nets over its whole frame, so the 2x layer's nets cost as much as production's
  (full 12 MP) although that layer holds 39% of the tiles.
- What the decoder passes, per layer:
  - the 8x8 output blocks of the layer that the composite reads: blocks within 8 px (the feather box) plus the cubic
    taps (2 layer px) of a tile of the layer's scale;
  - for the first net, the same blocks dilated by the second net's receptive field, because the second net's input
    there must be exact too.
- What `gpu_enh::run` does with them:
  - It dilates them by the net's receptive field: 69 px for c16d2. This is an upper bound summed over the convs:
    full-res x1, half x2, quarter x4, plus 8 px of slack.
  - It ORs them into half- and quarter-res block flags.
  - Each `k_conv` workgroup exits at once when its 8x8 output block is not flagged.
- Why the output is exact on the needed blocks:
  - Every conv runs on at least the dilated region.
  - A conv's output is exact wherever its 3x3 inputs are. The stale values outside the region (arena memory, never
    written) can reach inward by at most one conv radius per conv, so by at most the receptive field.
  - `dout` starts as a copy of the input image, so unflagged pixels hold the un-enhanced decode, not stale bytes.
- Active blocks on IMG_0621:

  | Layer | First net | Second net |
  | --- | --- | --- |
  | 5x | 28% | 20% |
  | 6x | 72% | 58% |
  | 4x | 73% | 52% |
  | 3x | 46% | 33% |
  | 2x | 45% | 41% |

- Measured:
  - Network time summed over the layers fell from 337 to 172 ms on IMG_0621, and from 340 to 221 ms per photo on
    the 11 photos.
  - Output identical on 11 / 11. Without the second-net dilation, 0.002% of the pixels were 1 level off.
- Switch: `CS_MS_ENH_ALL=1` turns it off. Production decodes never pass `active`.

**Tile merge on the resident cs-sr output, flat tiles shared** (`gpu_composite.hip`: `blend_tiles`; `gpu_tiles.hip` /
`gpu_common.hpp`: `resident_range`; `decrypt_tiles_ycc420`: `ms_flat_hr`)
- The problem:
  - The tile merge (`cs_composite_tiles` -> `gpu_composite::blend_tiles`, the `profile[ycc] composite` stage) took
    150-160 ms over the five layers, 74-87 ms of it in the 2x layer, against 15 ms in production.
  - `blend_tiles` could use the device copy of the cs-sr output (`resident_take`) only when every tile sat back to
    back in exactly that buffer.
  - In a layer, the tiles cs-sr skipped (bilinear) are separate `cv::Mat`s between the pooled ones. So it repacked
    every tile on the host (~255 MB for the 2x layer) and uploaded the lot, while the cs-sr output on the device was
    freed unused.
- What changed:
  - `resident_range` reports the host range the handoff slot holds.
  - `blend_tiles` reads every tile inside that range from the slot's device buffer at its offset. It uploads only the
    other tiles, once per distinct host buffer, after the resident part: one device-side copy into a combined buffer,
    then one upload.
  - With every tile resident (production), it reads the slot's buffer itself as before. With no slot and every tile
    packed, it uploads that buffer without repacking, as before.
  - A slot that holds none of the tiles is ignored (freed by the guard as before).
  - In a layer, the flat-filled tiles (no luma, Cr or Cb sample; the LR fill is gray 128) take a shared 2x tile per
    size (`ms_flat_hr`, gray 128, read-only), which is exactly the bilinear 2x of the fill. So the composite uploads
    one copy instead of one per tile. Tiles with chroma samples but no luma still get their own bilinear 2x.
- Output: `CS_GPU_COMPOSITE_VERIFY=1` reports 0 values different from `blendTilesWithImage` on all five layers. The
  decode is identical on 11 / 11 photos and in production.
- Measured:
  - Tile merge over the five layers fell from 151-165 to 40-42 ms on IMG_0621 (2x layer: 74-87 -> 19 ms).
  - Decode (internal) fell from 1454-1478 to 1272 ms. That is more than the merge stage alone; probably the 255 MB
    host repack also slowed the other layer threads' CPU work, but that was not measured.
  - Wall time on the 11 photos fell from 1786 to 1620 ms per photo.

### 18.4 Tried and not kept

**Banded enhance per layer** (bands of 512 / 1024 rows, each cropped to the columns of its samples, one net call per
band)
- Output: max 1 level.
- Time: 2.08 / 2.00 s against 1.93 s whole-frame. It made 28-48 net calls instead of 10, and each call has a fixed
  cost of ~5-8 ms (uploads, plane kernels, three device syncs, download, data consistency).

**Reusing the net's device buffers across calls** (proposed as the fix for the banded version)
- Measured, not built: allocation is not the per-call cost.
- With `CS_GPU_PROF=1`, `gpu_enh: device allocation` is 0.0 ms on 9 of the 10 calls of a layered decode; the
  buffers come from the device arena.
- The exception is the first full-frame call, where S0 (372 MB, more than the 256 MiB arena blocks) falls back to
  `hipMalloc`: 32 ms. Production pays this too.
- A second call of the same size takes 0 ms, because the driver keeps the freed allocation and hands it back.

**Nets cropped to each layer's bounding box** (+48 px of context, one call per layer)
- Exe `ms21`.
- A layer's tiles are spread over the frame, so the box covers 38-90% of it, and only 1-3 layers per photo came in
  under 90%.
- Network −21 ms per photo on 11 photos, inside the wall-time noise; output max 1 level off. Superseded by the
  block mask.

**Layer-tuned enhance net** (`6bclay_c16d2`, a quality change, not speed)
- A 16k-step fine-tune on layer decodes. Validation rose from 26.64 to 26.80 dB, but on the 11 photos it lost to
  the production pair at equal bytes.
  - Alone: −0.16 / −0.12 / −0.17 dB at r0.1 / 0.3 / 0.5.
  - Followed by u2dc: −0.04 / −0.05 / −0.10 dB.
- Not promoted (`TODO.md`).

### 18.5 Where the remaining time goes

IMG_0621 r0.3, ms23, `CS_GPU_PROF=1 CS_PROFILE=1`. The left column sums each stage over the five layers.

| Stage | 5 layers | Production | Difference | On the critical path |
| --- | --- | --- | --- | --- |
| LR solve (`tiles on the GPU in`) | 444 ms | 261 ms | +183 | yes (GPU lock) |
| cs-sr | 227 ms | 388 ms | −161 | yes |
| tile merge (`profile[ycc] composite`) | 160 ms | 15 ms | +146 | yes |
| calm | 69 ms | 39 ms | +30 | yes |
| net uploads + planes / downloads | 89 / 34 ms | 51 / 18 ms | +54 | yes |
| net | 170 ms | 175 ms | −5 | yes |
| data consistency | 22 ms | 21 ms | +1 | yes |
| multi-scale composite | 63-78 ms | — | +70 | yes (after the layers) |
| PNG write | ~70 ms | ~40 ms | +30 | yes |
| unpack + prep (CPU) | 215 + 172 ms | 112 + 92 ms | | mostly hidden behind other layers' GPU time |

Layer finish times: 5x 346 ms, 6x 429, 4x 578, 3x 779, 2x 1364 ms; composited at 1497 ms. The four small layers hold
the GPU for ~430 ms before the 2x layer can start. The 2x layer then holds it for 585 ms.

What is still slower than production, largest first:
1. **LR solve, +183 ms.**
   - The first layer to reach the GPU (5x, 227 tiles) spends 195 ms waiting for the HIP runtime and the arena reserve.
     Production hides that wait behind its 112 ms unpack and 92 ms prep. A layer thread reaches the GPU after ~30 ms.
   - The other four layers cost 22-38 ms each, even the 480-tile 6x layer: a fixed cost per solve call (layout,
     uploads, two batches, device tail).
2. **Tile merge, +146 ms (fixed in ms24: now 41 ms over the five layers, +26 against production).**
   - In a layer, the skipped tiles' bilinear outputs were separate `cv::Mat`s, so the tiles were no longer one
     packed buffer.
   - `gpu_composite::blend_tiles` then repacked every tile on the host and uploaded it, instead of taking the cs-sr
     output still on the device. See the ms24 entry in 18.3.
3. **Multi-scale composite, ~70 ms.** A CPU pass after all layers, tile-parallel: a cubic remap per used layer per
   tile, then the box-weight blend.
4. **Net transfers, +54 ms.** Per call: an upload of the full frame, planes built on the GPU, a download of the full
   frame. That is ten calls instead of two, and the masked nets still move whole frames.
5. **Calm, +30 ms.** `cs_seg_calm_boundaries` runs on every layer's full frame.
6. **PNG write, +30 ms.** The composite is written after the GPU stages end, so nothing overlaps it. Production is
   in the same position, but its write measured 40 ms against ~70 here.

Together: about +480 ms of extra stage time against −161 ms of cs-sr. With the overlaps that leaves the
measured gap of ~0.3-0.4 s (1.63-1.83 s against 1.37-1.41 s).
