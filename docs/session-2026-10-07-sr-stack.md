# Session notes — super-resolution stack (2026-10-07)

> Reconstructed session summary, not a verbatim transcript.
> Code changes from this session are in commit `d8e991f` on branch `local-sync/sr-stack`.

## Goal
Push the accumulated super-resolution / sampling work to GitHub on a new branch.

## What was built (in commit d8e991f, +2938/−148, 22 files)
- **CS-SR refinement backend** (`--photo-upscaler cs-sr`): per-tile HR-DCT FISTA
  refinement of luma against the original LR samples (box-downsample forward
  model, AVIR anchor, TV). Y-only solve.
- **Multiscale cascade warm start** (`cs_sr_cascade_init`): solve at 1.5x from a
  bilinear init, then warm-start the 2x solve. Measured on photos: 36.76 dB
  (cascade) vs 36.71 dB (direct start), ~2.4x refinement cost. Kept.
- **RED-lite priors** (`--red`, `--red-denoiser nlmeans|dncnn`): per-pass proximal
  blend toward fastNlMeans (luma) or the vendored color DnCNN via in-process
  ONNX Runtime (residual convention verified by test). `--red 0` is bit-identical.
- **Coupled dictionaries** (new `ImgReconstruct_backend/cs_dict.{cpp,hpp}`):
  Batch-OMP/Cholesky, K-SVD, CSD2 format, joint [Xh;Yl] training,
  `train-dict` subcommand, `--sr-dict` fused SR path for every container mode.
- **Sampling upgrades**: spectral-DCT LOD scorer (`--spectral`), YCC 4:2:2
  (mode 7), YCC422+HF-focus (mode 8).
- **FSRCNN backend** (`--photo-upscaler fsrcnn`, OpenCV `dnn_superres`,
  thread-local sessions). Weights committed as `FSRCNN_x2.pb` (39 KB) so the
  default `--fsrcnn-model FSRCNN_x2.pb` resolves on fresh clones.
- **sr-direct experiment** (`cs_sr_direct_luma` harness): single-stage solve
  measured *worse* than two-stage on photos — no production flag ships.
- Tests (`tests/test_main.cpp`, +593 lines) and README updates included.

## Key decisions
- `FSRCNN_x2.pb` committed deliberately (tiny, required by the default path;
  same precedent as tracked `cs_dict.dict`). Revert that file if models should
  stay download-only.
- `models/` stays gitignored; `x64/` build outputs are ignored and were not pushed.
- Trained `sr_coupled_256.csd2` was left in temp (0.5 MB experiment artifact that
  underperformed: 28.5 dB HR vs 29.4 dB plain AVIR). Retrain via `train-dict`.
- Branch `local-sync/sr-stack` was cut from `cpu-only-no-rocm` HEAD (`b8a71fd`),
  which is not an ancestor of `origin/master` — no rebase attempted.

## Test / experiment results worth remembering
- `cs_tests`: ALL TESTS PASSED (MSVC build, `x64/Cpu`).
- ycc422 photo: chroma PSNR 45–47 dB at 0.24–0.25 ratios.
- Mode 8 HF-focus photo: 30.55 dB vs 30.64 dB mode 7 baseline (worse on Y, keep off).
- sr-direct photo: HR 31.00 dB vs AVIR 30.58 dB — but 1.1 dB *below* the
  decrypt path with `cs_sr_direct_luma`; LOD masks off where gains appear.
- DnCNN path: residual convention confirmed (`clip(y - f(y))` denoises,
  `clip(f(y))` destroys); color residual beats per-channel gray residual.

## Branch / remote state
- Branch: `local-sync/sr-stack`, tracking `origin/local-sync/sr-stack`.
- Commit: `d8e991f` "Super-resolution stack: CS-SR refinement, RED priors,
  coupled dictionaries, sampling upgrades".
- This notes file was committed on top as a follow-up.

## Open follow-ups
- Decide PR base (`master` vs stacked on the `cpu-only-no-rocm` line).
- Optional: DnCNN FP16/int8 + CUDA provider probes; spectral scorer per-image
  tuning; HF-focus default ratios per mode.
