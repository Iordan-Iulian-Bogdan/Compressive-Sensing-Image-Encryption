# AGENTS.md

## Project
C++ image encryption via compressive sensing with FISTA reconstruction (per-channel + SOMP-structured joint; HIP GPU offload). MSVC / Visual Studio solution `ImgReconstruct_backend.sln`, x64. Dependencies resolved via vcpkg manifest (`vcpkg.json`). Uses Windows CNG crypto (`crypto_utils`).

## Branch `production-only`
This branch carries only the production formats and decode. Encode: the multi-scale container (mode 20, default) and the single-scale seg8-tile container (mode 19, `--multiscale off`); mode 11 stays inside them as the multi-scale layer format. Decode: those two containers on GPU and CPU, batch mode, `--enhance` / `--enhance2` / `--enhance-low`; any other container fails with "not a production format". Everything else was removed (uniform / periodic / adaptive / HF / T-tetromino / 4:2:2 / BGR / region / seg-blend sampling, the other tile solvers (`--solver` joint, GAP-TV, ADMM-L1TV, PnP-NLM), the wavelet / dictionary bases, dual-TV / grain-TV / y-confidence, `--full-res`, `--manual` parameters, every other `--photo-upscaler` (avir, fsrcnn, waifu2x, realcugan, dip, sr-direct, ONNX priors, Swin2SR) and `--final-polish`, the rate loop, `--fast`).
- CLI flags: `--password --no-preview --device --ratio --seg-auto --ycc420 --seg-tile --hr-detail --multiscale --hr-scale --sample-bits --show-mask --enhance` (encrypt) and `--tv --photo-upscaler cs-sr --sr-pnp admm --sr-pnp-iters --hr-weight --enhance --enhance2 --enhance-low` (decrypt); the decode values that were flags are fixed at their production values (auto tile grid / coef / overlap, FISTA, DCT, wscale 2, 2 reweights, 4:2:0). The production command lines below run unchanged.
- Env switches removed with their features: `CS_ADMM_ITERS CS_ADMM_L1 CS_ADMM_TV CS_DIP_DEBUG CS_DUMP_LRTILES CS_DUMP_LR_EVERY CS_DUMP_SEG CS_ENC_PROD_DECODE CS_GAPTV_* CS_GPU_ONNX_BATCH CS_GPU_ONNX_FOLD* CS_GPU_ONNX_PRECISION CS_GPU_SWIN CS_PNP_* CS_RATE_LOOP* CS_SEG_CHROMA_MASKED CS_SEG_EDGE_NOSPILL CS_WAIFU2X_*`. The production, dump and GPU switches stay.
- Verified 2026-11-09: with all randomness fixed, 43 outputs (10 encodes incl. mask views and batch, 10 GPU decodes, 2 CPU decodes, batch decode, training-data dumps, the NLM-prox anchor decode) are byte-identical to master at 086b225, and the CPU build (`build_cpu.bat`) decodes byte-identically to master's CPU build. `cs_tests.exe` keeps the crypto, tiling, sample-bits, segmentation, anchor and enhance-plane tests plus production roundtrips (155 checks).
- The training scripts the pipeline calls (`train_enh_fresh.py`, `gen_enh_seg8tile_anchor.py`, the ONNX export) live untracked in the main checkout's `x64/GpuBuild/`, as on master.

## Working principles
1. **Think Before Coding** — Don't assume. Don't hide confusion. Surface tradeoffs.
   - State assumptions explicitly — if uncertain, ask rather than guess.
   - Present multiple interpretations — don't pick silently when ambiguity exists.
   - Push back when warranted — if a simpler approach exists, say so.
   - Stop when confused — name what's unclear and ask for clarification.
2. **Simplicity First** — Minimum code that solves the problem. Nothing speculative.
   - No features beyond what was asked.
   - No abstractions for single-use code.
   - No "flexibility" or "configurability" that wasn't requested.
   - No error handling for impossible scenarios.
   - If 200 lines could be 50, rewrite it.
   - Test: would a senior engineer say this is overcomplicated? If yes, simplify.
3. **Surgical Changes** — Touch only what you must. Clean up only your own mess.
   - Don't "improve" adjacent code, comments, or formatting.
   - Don't refactor things that aren't broken.
   - Match existing style, even if you'd do it differently.
   - If you notice unrelated dead code, mention it — don't delete it.
   - When your changes create orphans: remove imports/variables/functions that YOUR changes made unused; don't remove pre-existing dead code unless asked.
   - Test: every changed line should trace directly to the user's request.
4. **Goal-Driven Execution** — Define success criteria. Loop until verified.
   - "Add validation" → "Write tests for invalid inputs, then make them pass."
   - "Fix the bug" → "Write a test that reproduces it, then make it pass."
   - "Refactor X" → "Ensure tests pass before and after."

## Layout
- `ImgReconstruct_backend/` - main sources (`ImgReconstruct_backend.vcxproj`): `CS_encryption`, `image_encryption`, `image_decryption`, `helper_functions`, `crypto_utils`, `cs_gpu` (HIP)
- `tests/` - `cs_tests` sources (`test_main.cpp`, `test_*.cpp`) and the standalone HIP kernel tests (`gpu_tiles_test.cpp`, `gpu_admm_test.cpp`)
- `x64/`, `.vs/`, `.obj`/`.exe` files in root - build output (never read/edit)

## Build
- Default build: `.\build_gpu.bat main` (LLVM 23 `clang-cl` + hipcc, Release x64, outputs `x64\GpuBuild\ImgReconstruct_backend.exe`; drop `main` to also build `cs_tests.exe`). It compiles the `*.hip` files under the VS 18 Insiders 14.29 toolset (ROCm 7.2 clang rejects the newer MSVC STL) and the rest under VS 18 BuildTools. Needs AOCL-Compression for the PNG writer. GPU paths are opt-in at run time: `--device gpu` plus `CS_GPU_TILES=1`, `CS_GPU_ADMM=1` (see `docs/gpu-optimizations.md`).
- `.\build_cpu.bat` builds without ROCm (CPU stub for the `cs_gpu` API).
- Stale, do not use: `build_clang.bat` (hardcodes VS 2022, which is not installed, and misses `enh_net.cpp`, `gpu_*.cpp`, `png_fast.cpp`) and the MSVC solution `ImgReconstruct_backend.sln` (its vcxproj lacks the same sources).

## rtk (command-output condenser)
- The shell environment pipes big-output commands (`rg`, `grep`, `git log/diff/status`, test runners, etc.) through `rtk`, which condenses output: every signal kept, noise dropped.
- Treat condensed output as the COMPLETE result; run commands normally and batch related commands into one call.
- Truncated results state their recovery path in their own output.
- Re-run as `rtk proxy <cmd>` only when the result is unusable: empty when output was clearly expected, contradicting its exit code, or garbled.
- Direct helpers: `rtk rg`/`rtk grep` (search), `rtk git log/diff/status`, `rtk test`, `rtk smart`/`rtk read` (file summaries).

## Conventions
- For large logs or file analysis, use sandboxed processing (ctx tools) and only return derived summaries.
- Every optimization that works (speed or quality, kept in the code) gets a detailed write-up in the same task, in the doc for its area (`docs/gpu-optimizations.md` for decode / encode speed, `docs/enhance-nets.md` for nets, otherwise the closest doc): what changed and why it is correct, the files / functions / env switches, the workload and method (pinned exe names, ABBA, idle GPU), the measured before / after, and whether the output is identical (else by how much). Tried-and-reverted attempts get their numbers and the reason too. A one-line TODO.md note is not enough.
- Test commands for the user: write ALL output files (decrypts, comparisons, error maps) to the repo root `C:\Users\iorda\OneDrive\Documents\Compressive-Sensing-Image-Encryption-CPU_Limited_memory_BFGS` using relative paths (e.g. `wavelet.dec.png`, never `$env:TEMP` or absolute temp paths).
- Production path (since 2026-11-03, CPU and GPU alike): the prox-off decoder with the net trained on it. Use it unless the user says otherwise.
- Encrypt commands: production format is `--seg-auto 8 --ycc420 --seg-tile --ratio 0.9 --hr-detail 450000 --enhance models\enh\enh_seg8a_x5_ema256_fresh_unet.onnx` (since 2026-11-06 these flags write a multi-scale container by default: mode 20, `--multiscale auto`, with the anchor count x0.5 up to r0.5 and x2 from r0.7, log-linear between; +1.81 / +1.31 / +0.54 dB, SSIM +0.029 / +0.016 / +0.006, ΔE95 −1.01 / −0.59 / −0.22 at r0.2 / 0.3 / 0.5 at equal bytes over 88 photos, settings oracle in `TODO.md`; explicit `--hr-scale` overrides the multiplier. `--multiscale off` keeps the single-scale seg8-tile container, mode 19; `Params-test/gen_enh_prod.py` uses it for training data. `--hr-detail` is a fixed count at any photo size, the cap is 2^24. Opt-in since 2026-11-06: `--hr-area` makes it the count per 4032x3024 / 12.2 MP, scaled by the photo's area (lost at equal bytes on 24.5 MP photos, `TODO.md`). `--hr-scale m` multiplies the count, and the flagged blocks grow past 5% to keep production's 74% fill) with `CS_GPU_TILES=1 CS_GPU_ADMM=1` set. Since 2026-11-05 luma is 6-bit by default with these flags (sample ratio 1.2x `--ratio`, capped at 1, for the same bytes; `--sample-bits 8` keeps 8-bit). On encrypt, `--enhance` makes the closed-loop pass rank the anchor blocks by the error left after that net (+0.06 dB mean, ΔE better on all 8 test photos, ~70 ms); the `fresh` net here is the ranking net the prox-off net's training containers were encoded with (`Params-test/gen_enh_prod.py`).
- Decrypt commands: default to `--tv 0 --photo-upscaler cs-sr --sr-pnp admm --sr-pnp-iters 2 --hr-weight 1 --device gpu --enhance models\enh\enh_seg8a_x5_ema256_6bc450_c16d2_unet.onnx --enhance2 models\enh\enh_seg8a_x5_ema256_6bc450_u2dc_unet.onnx --enhance-low models\enh\enh_seg8a_x5_ema256_6bclo5_c16d2_unet.onnx` with `CS_ADMM_PROX_OFF=1 CS_GPU_TILES=1 CS_GPU_ADMM=1` set (identity ADMM prox, no NLM; the decoder then scales the auto coef x0.2 itself). For a CPU decode use `--device cpu` and `CS_GPU_TILES=0 CS_GPU_ADMM=0 CS_GPU_ONNX=0 CS_GPU_ENH=0` (the GPU build enables those stages by default); its output is within max 6 levels of the GPU's. The net (c16d2 with 9 extra planes: the 5 below plus the Cr / Cb sample masks and residuals; run `x64/GpuBuild/runs/20261105-080017_enh_6bitc`) beat the 5-plane `6b450_c16d2` on all 32 HIP decodes (2026-11-05, both followed by `cs_enh_project`: +0.17/+0.18/+0.20/+0.22 dB, SSIM 8/8 better at every ratio, ΔE-p95 −0.10/−0.10/−0.11/−0.12 at r0.3/0.5/0.7/0.9), network 45 ms; IMG_3690 r0.3 35.70 dB, SSIM 0.920, ΔE-p95 3.46; r0.9 38.77 dB, SSIM 0.956, ΔE-p95 2.58. The c32 width of it (`6bc450_c32d2`) is +0.09..0.11 dB but 144 ms. Every net in `models/enh` (recipes, results, status): `docs/enhance-nets.md`. `--enhance-low` (since 2026-11-07) swaps the first net for the low-ratio `6bclo5` on containers under 0.43 luma sample bits per pixel (about r0.25 and below; the decode prints its choice): +0.26 / +0.06 dB, ΔE-p95 −0.24 / −0.07 at r0.1 / 0.2 over 32 photos, r0.3 and up unchanged, no time cost. Second unrolled stage (default since 2026-11-05; drop `--enhance2` for the single net): `--enhance2` runs another c16d2 on the projected output, with its planes rebuilt against that output, then projects again (trained on the frozen 9-plane net; run `x64/GpuBuild/runs/20261105-103025_enh_unrolled`): all 32 HIP decodes better, +0.17/+0.12/+0.09/+0.07 dB, ΔE-p95 −0.13/−0.10/−0.08/−0.06, network 96 ms instead of 48 (2026-11-05); IMG_3690 r0.9 38.88 dB, ΔE-p95 2.54. The 5-plane net before it (c16, two extra full-res residual blocks, trained on 6-bit 450k mixed-ratio containers; run `x64/GpuBuild/runs/20261104-205447_enh_6bit`) beat `mix450c02_c16` on the 8-photo HIP decode of 6-bit containers at every ratio (2026-11-05: +0.17/+0.38/+0.55/+0.65 dB, ΔE-p95 0/−0.01/−0.03/−0.03 at r0.3/0.5/0.7/0.9), network 47 ms. IMG_3690 6-bit, 2026-11-05: r0.3 35.43 dB, SSIM 0.918, ΔE-p95 3.61; r0.9 38.38 dB, SSIM 0.954, ΔE-p95 2.79; GPU decode ~0.9 s. After the net the decoder projects its output onto the container's measurements (`cs_enh_project`: 4x4 Cr/Cb means and 2x2 luma means into their samples' quantization bins, anchors exact; 12 ms CPU; `CS_ENH_DC=0` skips it): all 32 test decodes better, +0.04/+0.06/+0.09/+0.10 dB, ΔE-p95 −0.02/−0.03/−0.04/−0.05 at r0.3/0.5/0.7/0.9 (2026-11-05, on top of the numbers above), almost all of it from chroma. The c32 net (`mix450c02_c32`) on the 8-bit container was 39.05 dB, SSIM 0.962, ΔE-p95 2.57 on IMG_3690 r0.9.
  - Previous path (NLM prox, 300k anchors): decrypt `--tv 0 --photo-upscaler cs-sr --sr-pnp admm --hr-weight 1 --device gpu --enhance models\enh\enh_seg8a_x5_ema256_fresh_unet.onnx` on `--hr-detail 300000` containers (IMG_3690: 38.71 dB, SSIM 0.961, ΔE-p95 2.71; GPU 2.3 s, CPU 57 s). After the net the decoder keeps its output as is (no whole-image anchor write-back; that measured −0.12 dB because its 8-bit YCrCb round trip moves every pixel, not because of the anchors); `CS_HR_NO_POSTHOC=1` is only needed for decodes without `--enhance`. `--fast` gave ~35% less time at −0.06 dB there.
- Many images (sweeps, evaluations, training data): use batch mode, `encrypt|decrypt @<pairs.tsv> [options]` (one `input<TAB>output` per line, one process, same options for all), split into two batch processes run at once for the most throughput (decode 439 vs 687 ms, encode 732 vs 1352 ms per 12 MP photo; `docs/gpu-optimizations.md` section 17). Decoder dumps take `*` for per-image paths (`CS_DUMP_YMASK=*` etc. write `<output>.ymask.png`). `Params-test/gen_enh_prod.py` batches with `ENH_BATCH=32`.
- Encrypt commands written for the user to run: pass `--show-mask` and display the resulting `*.mask.png` view. Encrypts run through an agent's shell tool: use `--no-preview` instead (the mask view's keypress window blocks the call).
- NEVER delete `third_party/` (untracked local deps: onnxruntime, nunif — builds fail without it, and it is invisible to git). If a build fails on missing `onnxruntime_cxx_api.h`, restore from the `BFG2S` clone on the Desktop, never re-download.
- Quality rule of thumb: judge **sharpness by SSIM**, **color by CIEDE2000 p95/p99**, treat **PSNR as a tiebreaker** (it is the solver's optimization target, not a perceptual verdict). LPIPS (AlexNet, `Params-test/metrics_lpips.py`) confirms structure/texture. Optimize PSNR, but promote a decode only if SSIM and ΔE-p95 agree. Helpers: `Params-test/metrics.py <orig> <dec>` (SSIM + ΔE), `Params-test/metrics_lpips.py` (LPIPS).

# context-mode - MANDATORY routing rules

context-mode MCP tools available. Rules protect context window from flooding. One unrouted command dumps 56 KB into context.

## Think in Code - MANDATORY

Analyze/count/filter/compare/search/parse/transform data: **write code** via `context-mode_ctx_execute(language, code)`, `console.log()` only the answer. Do NOT read raw data into context. PROGRAM the analysis, not COMPUTE it. Pure JavaScript - Node.js built-ins only (`fs`, `path`, `child_process`). `try/catch`, handle `null`/`undefined`. One script replaces ten tool calls.

## BLOCKED - do NOT attempt

### curl / wget - BLOCKED
Shell `curl`/`wget` intercepted and blocked. Do NOT retry.
Use: `context-mode_ctx_fetch_and_index(url, source)` or `context-mode_ctx_execute(language: "javascript", code: "const r = await fetch(...)")`

### Inline HTTP - BLOCKED
`fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, `http.request(` - intercepted. Do NOT retry.
Use: `context-mode_ctx_execute(language, code)` - only stdout enters context

### Direct web fetching - BLOCKED
Use: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)`

## REDIRECTED - use sandbox

### Shell (>20 lines output)
Shell ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`.
Otherwise: `context-mode_ctx_batch_execute(commands, queries)` or `context-mode_ctx_execute(language: "javascript", code: "...")`. Use `language: "shell"` only when code matches the host shell.

### File reading (for analysis)
Reading to **edit** - reading correct. Reading to **analyze/explore/summarize** - `context-mode_ctx_execute_file(path, language, code)`.

### grep / search (large results)
Use `context-mode_ctx_execute(language: "javascript", code: "...")` in sandbox for portable filtering/counting.

## Tool selection

0. **MEMORY**: `context-mode_ctx_search(sort: "timeline")` - after resume, check prior context before asking user.
1. **GATHER**: `context-mode_ctx_batch_execute(commands, queries)` - runs all commands, auto-indexes, returns search. ONE call replaces 30+. Each command: `{label: "header", command: "..."}`.
2. **FOLLOW-UP**: `context-mode_ctx_search(queries: ["q1", "q2", ...])` - all questions as array, ONE call (default relevance mode).
3. **PROCESSING**: `context-mode_ctx_execute(language, code)` | `context-mode_ctx_execute_file(path, language, code)` - sandbox, only stdout enters context.
4. **WEB**: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)` - raw HTML never enters context.
5. **INDEX**: `context-mode_ctx_index(content, source)` - store in FTS5 for later search.

## Parallel I/O batches

For multi-URL fetches or multi-API calls, **always** include `concurrency: N` (1-8):

- `context-mode_ctx_batch_execute(commands: [3+ network commands], concurrency: 5)` - gh, curl, dig, docker inspect, multi-region cloud queries
- `context-mode_ctx_fetch_and_index(requests: [{url, source}, ...], concurrency: 5)` - multi-URL batch fetch

**Use concurrency 4-8** for I/O-bound work (network calls, API queries). **Keep concurrency 1** for CPU-bound (npm test, build, lint) or commands sharing state (ports, lock files, same-repo writes).

## Output

Write artifacts to FILES - never inline. Return: file path + 1-line description.
Descriptive source labels for `search(source: "label")`.

## Memory

Session history is persistent and searchable. On resume, search BEFORE asking the user:

| Need | Command |
|------|---------|
| What did we decide? | `context-mode_ctx_search(queries: ["decision"], source: "decision", sort: "timeline")` |
| What constraints exist? | `context-mode_ctx_search(queries: ["constraint"], source: "constraint")` |

DO NOT ask "what were we working on?" - SEARCH FIRST.
If search returns 0 results, proceed as a fresh session.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call `stats` MCP tool, display full output verbatim |
| `ctx doctor` | Call `doctor` MCP tool, run returned shell command, display as checklist |
| `ctx upgrade` | Call `upgrade` MCP tool, run returned shell command, display as checklist |
| `ctx purge` | Call `purge` MCP tool with confirm: true. Warns before wiping knowledge base. |

After /clear or /compact: knowledge base and session stats preserved. Use `ctx purge` to start fresh.
