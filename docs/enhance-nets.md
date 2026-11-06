# Enhance nets (`--enhance`, `--enhance2`)

Every restoration net in `models/enh/`. What each one is, how it was trained, how it scored, and whether it is in use.
Each net is two files: `<name>.onnx` (ONNX Runtime path) and `<name>.enh.bin` (HIP path). Both are written by
`x64/GpuBuild/export_enh_onnx.py <checkpoint.pt> <planes> <out.onnx>`. The PyTorch checkpoints stay in `x64/GpuBuild/`
as `enh_<tag>_net_unet.pt`.

## Production (since 2026-11-05)

Decode with two stages and a projection after each:

```
--enhance  models\enh\enh_seg8a_x5_ema256_6bc450_c16d2_unet.onnx
--enhance2 models\enh\enh_seg8a_x5_ema256_6bc450_u2dc_unet.onnx
```

1. The 9-plane net (`6bc450_c16d2`) runs on the pre-net decode.
2. `cs_enh_project` moves its output back onto the container's measurements.
3. The second-stage net (`6bc450_u2dc`) runs on that output. Its planes are rebuilt against the stage-1 output, so they
   show what stage 1 still got wrong.
4. `cs_enh_project` runs again.

Encrypt keeps `enh_seg8a_x5_ema256_fresh_unet.onnx` as the anchor-ranking net (`--enhance` on encrypt). The training
containers for every net from `mix450` onward were encoded with that ranking net.

IMG_3690, 6-bit 450k-anchor containers, HIP decode:

| Ratio | PSNR | SSIM | ΔE95 | ΔE99 |
|---|---|---|---|---|
| r0.3 | 35.85 dB | 0.921 | 3.35 | 5.94 |
| r0.5 | 37.84 dB | 0.941 | 2.77 | 4.59 |
| r0.7 | 38.64 dB | 0.952 | 2.58 | 4.28 |
| r0.9 | 38.88 dB | 0.956 | 2.54 | 4.21 |

r1.0 gives the same container as r0.9 and the same decode (6-bit luma samples at 1.2 × the ratio, capped at 1).

## Architecture

All nets are `UNetX` from `x64/GpuBuild/train_enh_fresh.py` (`ENH_ARCH=unet`):

- **Shape:** 3 levels.
- **Width:** `ENH_WIDTH` channels at full resolution (c16 or c32).
- **Depth:** `ENH_D0` extra residual blocks at full resolution. The `d2` nets have 2.
- **Output:** the enhanced RGB image.

**Inputs:** RGB in [0, 1] plus extra planes, built by `enh_inputs.py` in training and by `cs_enh_planes` in the decoder.

| # | Plane | Nets |
|---|---|---|
| 0 | luma sampling density: the LR sample mask blurred (Gaussian σ 4 on the LR grid), minus 0.5 | 3, 5, 9 |
| 1 | anchor indicator | 3, 5, 9 |
| 2 | anchor residual: (exact anchor luma − decoded luma) / 64 | 3, 5, 9 |
| 3 | LR luma sample mask, on its 2×2 block | 5, 9 |
| 4 | LR luma sample residual: (sample − 2×2 mean of decoded luma) / 8 | 5, 9 |
| 5, 6 | Cr sample mask and residual: (sample − 4×4 mean of decoded Cr) / 4 | 9 |
| 7, 8 | Cb sample mask and residual, the same way | 9 |

All residuals are clipped to [−1, 1]. The decoder takes the plane count from the model's input channels: 6, 8 or 12.

**HIP path** (`gpu_enh.hip`): it keeps activations in fp16. A net is only safe there when its peak activation is well
below 65504; check it with `act_check.py`. `mix128k` is the example of what happens otherwise (see the experiments below).

## Data consistency (`cs_enh_project`, `enh_net.cpp`)

The projection runs after every net. It moves the output onto what the container measured:

- **Chroma:** each sampled 4×4 block's mean Cr / Cb goes into its sample's quantization bin ± 0.75, with luma kept.
- **Luma:** each sampled 2×2 block's mean luma goes into its 6- or 8-bit bin ± 0.75.
- **Anchors:** each anchor pixel's luma becomes exact.

It takes about 12 ms on the CPU, and `CS_ENH_DC=0` turns it off.

| Net it follows | Gain on the 32 HIP decodes (r0.3 / 0.5 / 0.7 / 0.9) |
|---|---|
| 5-plane `6b450_c16d2` | +0.04 / +0.06 / +0.09 / +0.10 dB, ΔE95 −0.02 to −0.05, almost all from chroma |
| 9-plane net | about +0.005 dB, because that net already sees the chroma residuals |

The PyTorch version used in training is `x64/GpuBuild/enh_dc.py`. It has `dc` and `replanes`, which rebuilds the
residual planes against a stage's output.

## Training recipe (all nets from `mix450` on)

`x64/GpuBuild/train_enh_fresh.py <steps>` with:

- `ENH_DATA=<set>`, `ENH_ALL_PHONES=1`, `ENH_ARCH=unet`, `ENH_EXTRA=1`
- `ENH_PLANES=5|9`, `ENH_WIDTH=16|32`, `ENH_D0=0|2`
- `ENH_LR=5e-4`, `ENH_CLIP=0.05`, `ENH_EMA=0.999`, `ENH_CROP=256`, loss L1
- 128k steps from scratch, batch 8 of 256 px crops. The best EMA checkpoint by validation PSNR is kept.

**Training sets** are made by `Params-test/gen_enh_prod.py`. Each photo is encoded with the production format at a
ratio picked from {0.3, 0.5, 0.7, 0.9} by crc32 of its name, then decoded on the production prox-off path before the
net (`--sr-pnp-iters 2`).

| Set | Photos | Content |
|---|---|---|
| `F:\Training Data\enh_data_6b_mix450` | 6847 | 6-bit luma, 450k anchors |
| `F:\Training Data\enh_data_6bc_mix450` | 6847 | the same, plus Cr / Cb mask and sample dumps (`ENH_CHROMA=1`) |
| `F:\Training Data\enh_data_6bc_lo5_450` | 6847 | as 6bc_mix450, at r0.05 / 0.1 / 0.2 / 0.3 / 0.4 (129 GB; replaced the deleted mix sets) |
| `G:\Training Data\enh_data_6bc_hi5_450` | 6847 | as 6bc_mix450, at r0.5 / 0.6 / 0.7 / 0.8 / 0.9 (~146 GB) |

**Trainer switches** for the unrolled experiments:

| Switch | Effect |
|---|---|
| `ENH_DC=1` | projection inside the training graph |
| `ENH_STAGES=2` | two stages with `replanes` between them |
| `ENH_INIT=<.pt>` | start from a checkpoint (stage 1 when STAGES=2) |
| `ENH_FREEZE1=1` | train stage 2 only |

**Scoring.** A net is judged on the 8-photo × 4-ratio HIP decode of production containers: PSNR, SSIM, ΔE95 and
ΔE99, with `cmp6.py`-style scripts. It is promoted only if SSIM and ΔE95 agree with PSNR. Validation PSNR alone has
misled before: for `mix` and the c32 control below, it did not predict the decode verdict.

## The current lineage

| Net | Date | Planes / width / D0 | Data | Val PSNR | Result | Status |
|---|---|---|---|---|---|---|
| `6bc450_u2dc` | 11-05 | 9 / c16 / 2, **stage 2** | 6bc_mix450 | 34.834 (2 stages) | vs the 9-plane net, all 32 HIP decodes better: +0.17/+0.12/+0.09/+0.07 dB, SSIM 8/8 at every ratio, ΔE95 −0.13/−0.10/−0.08/−0.06; network 96 ms vs 48 | **production, `--enhance2`** |
| `6bc450_c16d2` | 11-05 | 9 / c16 / 2 | 6bc_mix450 | 34.751 | vs `6b450_c16d2`, all 32 better: +0.17/+0.18/+0.20/+0.22 dB, SSIM 8/8, ΔE95 −0.10/−0.10/−0.11/−0.12; 45–50 ms | **production, `--enhance`** |
| `6bclo5_c16d2` | 11-06 | 9 / c16 / 2 | 6bc_lo5_450 (r0.05/0.1/0.2/0.3/0.4) | 29.477 (low-ratio val) | vs the 9-plane net at r0.05/0.1/0.2/0.3/0.4: +0.84/+0.56/+0.22/+0.08/−0.01 dB, SSIM 8/8/8/4/2 of 8, ΔE95 −0.66/−0.38/−0.17/−0.08/−0.02; 46 ms. Production + u2dc there: +0.12/+0.16/+0.18/+0.17/+0.14 | not used yet: wins at r ≤ 0.2 (beats production's two stages there), but the decoder cannot pick a net by ratio |
| `6bchi5_c16d2` | 11-06 | 9 / c16 / 2 | 6bc_hi5_450 (r0.5/0.6/0.7/0.8/0.9) | 36.573 (high-ratio val) | vs the 9-plane net at r0.5–0.9: −0.04/−0.02/−0.00/+0.01/+0.01 dB, SSIM 0–5 of 8, ΔE95 within ±0.03; 50 ms | not used: no gain, the mixed 0.3–0.9 net already covers these ratios |
| `6bc450_c32d2` | 11-05 | 9 / c32 / 2 | 6bc_mix450 | 34.892 | vs the 9-plane c16: +0.11/+0.09/+0.09/+0.09 dB, ΔE95 −0.07/−0.05/−0.04/−0.04; 144 ms | not used: u2dc is better on colour and at low ratios, at two thirds of the cost |
| `6b450_c16d2` | 11-05 | 5 / c16 / 2 | 6b_mix450 | 34.659 | vs `mix450c02_c16` on 6-bit containers: +0.17/+0.38/+0.55/+0.65 dB, ΔE95 0/−0.01/−0.03/−0.03; 47 ms | production 11-05 morning, replaced the same day |
| `6b450_c16` | 11-05 | 5 / c16 / 0 | 6b_mix450 | 34.620 | ~0.03 dB and ~0.01 ΔE95 behind d2; 40 ms | not used |
| `mix450c02_c16` | 11-04 | 5 / c16 / 0 | `G:\enh_data_prod_mix450` (8-bit, 450k) | 33.83 | vs the c32: −0.20/−0.23/−0.27/−0.31 dB; 42 ms vs 142 | production 11-04, chosen for speed |
| `mix450c02_c32` | 11-03 | 5 / c32 / 0 | prod_mix450 | 34.03 | vs `fresh` on the old path: r0.3 +1.08 dB, r0.9 −0.33; with the prox-off path, IMG_3690 r0.9 39.05 dB (8-bit container) | production 11-03 |

Training runs:

| Net | Run (`x64/GpuBuild/runs/`) |
|---|---|
| `6bclo5_c16d2` | `20261105-223802_enh_lowratio`, recipe `lowratio5_chain.sh` |
| `6bchi5_c16d2` | `20261106-035111_enh_highratio`, recipe `highratio5_chain.sh` |
| 9-plane nets | `20261105-080017_enh_6bitc` |
| unrolled (`u2dc`, `ftdc`, `ftctl`) | `20261105-103025_enh_unrolled` |
| c32 9-plane | `20261105-171207_enh_c32_9p` |
| 6-bit 5-plane | `20261104-205447_enh_6bit` |
| `mix450c02_c16` | `20261104-122350_enh_c16`, recipe `c16_chain.sh` |
| `mix450c02_c32` | recipe `mix450_chain.sh` |

### The unrolled experiments (11-05; no files exported except u2dc)

The run was `20261105-103025_enh_unrolled`, scored in PyTorch on the 8 photos against the 9-plane net:

| Variant | What | Val PSNR | 8-photo result |
|---|---|---|---|
| `u2dc` | stage 1 = the 9-plane net, frozen; stage 2 = a fresh c16d2; projection after both; 64k steps, lr 5e-4 | 34.834 | +0.18/+0.14/+0.11/+0.10 dB (HIP: +0.17/+0.12/+0.09/+0.07) |
| `ftdc` | the 9-plane net fine-tuned with the projection inside the loss, 32k steps, lr 1e-4 | fell from 34.757 to 34.54 | +0.01, the same as the control; the projection cuts the gradient at active bounds |
| `ftctl` | the same fine-tune without the projection (control) | 34.760 | +0.01 |

HIP comes in 0.01–0.02 dB below PyTorch. The decoder rounds the stage-1 output to 8 bits before stage 2, while
training passed it on as float.

## Earlier nets (5 planes, the old NLM-prox decode path)

| Net | Date | What is recorded | Status |
|---|---|---|---|
| `fresh` | 10-30 | c16 (from its 1.77 MB weights; c32 files are 7 MB), 5 planes, 128k steps; EMA and the rest of the recipe not recorded; trained on r0.9 seg8-tile decodes (that dataset is gone) | production until 11-03; **still the encoder's anchor-ranking net** |
| `prod128k` | 10-31 | 128k retrain of `fresh` on `enh_data_prod_r09` (1438 photos, r0.9 only), EMA 0.999 | lost to `fresh` (−0.11..−0.22 dB) |
| `ft16k` | 10-31 | 16k fine-tune of `fresh` on prod_r09 (lr 2e-4 cosine) | tied at r0.9, lost at r0.3–0.7 |
| `mixft16k` | 10-31 | 16k fine-tune of `fresh` on mixed ratios `enh_data_prod_mix` | +0.74 dB r0.3, −0.06 r0.9; showed `fresh` was an r0.9 specialist |
| `mix128k` | 10-31 | 128k from scratch on prod_mix, lr 1e-3 | **fp16 overflow** (activations ~2.7e6, 0/255 contour artifacts on HIP); the reason for the activation check |
| `low234` c16 / c32 | 11-03 | c32 (and c16) 128k, 5 planes, run `20261103-160926_enh_full_eval_low234`; the data behind the tag is not recorded | 77-photo eval: c32 +0.06 dB over `mix450c02_c32` on average; not used |
| `mix450c02_c32_ssim` / `_lab` / `_mix` | 11-04 | `mix450c02_c32` recipe with `ENH_LOSS=ssim`, `lab` or `mix`; run `20261103-210601_enh_loss` | ssim worse on everything; lab ΔE95 −0.04..−0.06 but SSIM worse; mix worse on everything; not used |
| `h04`, `postnet` | 10-30 | 5-plane HIP nets compared in `h04cmp_*.log` / `pncmp_*.log` (repo root); recipe not recorded | not used |
| `enh_seg8a_x5_ema256`, `enh_seg8a05_x5_ema256` | 10-27 / 10-30 | the first seg8a nets, before `fresh`; recipe not recorded | superseded |

## Open items (TODO.md, "Enhance net")

- NAFNet-style blocks in the same U-Net, compared at equal HIP ms.
- Whether density (plane 0) is still needed now that plane 3 carries the raw mask.
- Not yet done: training stage 2 on the 8-bit-rounded stage-1 output, to close the 0.01–0.02 dB gap to PyTorch.
- Not yet done: the fp16 headroom check (`act_check.py`) for the 9-plane, u2dc and c32 nets. It hung in a GPU driver
  call on 11-05. All HIP test decodes of these nets were clean.
