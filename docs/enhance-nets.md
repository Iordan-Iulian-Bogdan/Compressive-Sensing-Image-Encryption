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
--enhance-low models\enh\enh_seg8a_x5_ema256_6bclo5_c16d2_unet.onnx
```

`--enhance-low` (since 2026-11-07) swaps the first-stage net for the low-ratio net on low-ratio containers; see
"Net choice by ratio" below.

1. The 9-plane net (`6bc450_c16d2`) runs on the pre-net decode.
2. `cs_enh_project` moves its output back onto the container's measurements.
3. The second-stage net (`6bc450_u2dc`) runs on that output. Its planes are rebuilt against the stage-1 output, so they
   show what stage 1 still got wrong.
4. `cs_enh_project` runs again.

Encrypt keeps `enh_seg8a_x5_ema256_fresh_unet.onnx` as the anchor-ranking net (`--enhance` on encrypt). The training
containers for every net from `mix450` onward were encoded with that ranking net.

#### Anchor ranking with the production decode (11-08; opt-in, not yet the default)

**What.** The encoder's closed-loop pass (`anchor_decode` in `ImgReconstruct_backend.cpp`) decodes the anchor-free
container in memory and anchors the worst 16x16 blocks after the `--enhance` net. That decode was not production's: it
used the `--fast` settings (overlap 8, 12 FISTA steps), the NLM prox with the default 8 ADMM rounds, and the `fresh` net
alone. `CS_ENC_PROD_DECODE=1` (new) makes the pass use the decode's own overlap and FISTA steps; with
`CS_ADMM_PROX_OFF=1`, `--sr-pnp-iters 2` and the decode nets on the encrypt command line it is the production decode,
projection included:

`CS_ENC_PROD_DECODE=1 CS_ADMM_PROX_OFF=1 CS_GPU_TILES=1 CS_GPU_ADMM=1 encrypt <in> <out> --seg-auto 8 --ycc420 --seg-tile
--ratio r --hr-detail 450000 --sr-pnp-iters 2 --enhance models\enh\enh_seg8a_x5_ema256_6bc450_c16d2_unet.onnx --enhance2
models\enh\enh_seg8a_x5_ema256_6bc450_u2dc_unet.onnx --enhance-low models\enh\enh_seg8a_x5_ema256_6bclo5_c16d2_unet.onnx`

Without the switch the encoder is unchanged. Encodes are not byte-repeatable (the random salt moves the key-drawn anchor
positions): three plain encodes of IMG_3690 r0.3 decoded to 36.264-36.280 dB, so one photo's difference is noise below
~0.015 dB.

**Evaluation** (`eval_encdec.py` in run `20261108-003201_enh_ds2`, rows `encprod` in its `fk_eval.csv`, ms34 encode,
production two-stage decode, against the `prod` rows of the same photos from plain ms33 encodes). 84 photos: the 18
`ImgReconstruct_backend` photos, 32 FiveK test, 34 edltauSCI iPhone HEICs. Mean change, photos better:

| Set | r0.3 | r0.5 | r0.7 | r0.9 |
|---|---|---|---|---|
| all 84 | +0.024 dB, 60/84; ΔE99 −0.011 | +0.008, 59/84 | +0.008, 70/84 | +0.008, 70/84 |
| 18 phone photos | +0.035, 17/18 | +0.010, 14/18 | +0.015, 18/18 | +0.014, 18/18 |
| iPhone 13 Pro Max (17) | +0.024, 11/17 | +0.006, 13/17 | +0.005, 14/17 | +0.005, 17/17 |
| iPhone 17 Pro (17) | +0.041, 15/17; ΔE99 −0.037 | +0.020, 15/17 | +0.013, 17/17 | +0.014, 17/17 |
| 32 FiveK | +0.009, 17/32 | +0.002, 17/32 | +0.004, 21/32 | +0.004, 18/32 |

SSIM moves by at most +0.0003 and ΔE95 by −0.007..+0.002 (level). The decode nets were trained on containers ranked by
`fresh`; a regenerated training set ranked this way might add to it.

**Encode time** (ms33 plain vs ms34 with the switch, ABBA x2, medians): IMG_3690 12 MP r0.3 1188 -> 1272 ms, r0.9 1501 ->
1494; IP17_IMG_8595 24.5 MP r0.3 2664 -> 2831, r0.9 3042 -> 3122 (+0..6%: two nets instead of one, the decode's overlap).

IMG_3690, 6-bit 450k-anchor containers, HIP decode:

| Ratio | PSNR | SSIM | ΔE95 | ΔE99 |
|---|---|---|---|---|
| r0.3 | 35.85 dB | 0.921 | 3.35 | 5.94 |
| r0.5 | 37.84 dB | 0.941 | 2.77 | 4.59 |
| r0.7 | 38.64 dB | 0.952 | 2.58 | 4.28 |
| r0.9 | 38.88 dB | 0.956 | 2.54 | 4.21 |

r1.0 gives the same container as r0.9 and the same decode (6-bit luma samples at 1.2 × the ratio, capped at 1).

### Net choice by ratio (`--enhance-low`, 2026-11-07)

**What it does.** With `--enhance-low <model>`, the decoder counts the container's luma sample bits per output pixel:
the luma measurement count `m` times the sample bits, over the decoded width × height. For a multi-scale container
(mode 20) the sum runs over all its layers. Their headers are parsed once before the layer threads start, and the one
choice applies to every layer. Below 0.43 bits per pixel, the first stage runs the `--enhance-low` net instead of
`--enhance`. `--enhance2` and both projections are unchanged. The decode prints the value and the choice
(`enhance: 0.386 luma sample bits per pixel, low-ratio net`).

**Why bits per pixel.** The multi-scale descriptor does not store the encode ratio. Bits per pixel are in every
container, and they separate the ratios cleanly:

| Container | r0.1 | r0.2 | r0.3 |
|---|---|---|---|
| default multi-scale (32 photos) | 0.19–0.29 | 0.37–0.46 | 0.55–0.64 |
| single-scale (`--multiscale off`, 6-bit) | 0.18 | 0.36 | 0.54 |

Single-scale containers give 1.8 × r (6-bit samples at 1.2 × r on the half-size grid). Multi-scale ones give more
for the same bytes, more so on smaller photos, because they carry half the anchors at these ratios.

**The cutoff.** It was chosen from the eval decodes of `20261106-233103_enh_laylo` (`6bclo5` + u2dc minus production's
two stages, by container bits per pixel):

| Bits per pixel | n | PSNR | SSIM | ΔE95 | PSNR wins |
|---|---|---|---|---|---|
| < 0.30 | 32 | +0.257 | +0.0045 | −0.240 | 31 |
| 0.30–0.40 | 16 | +0.096 | +0.0018 | −0.095 | 12 |
| 0.40–0.43 | 7 | +0.070 | +0.0006 | −0.089 | 4 |
| 0.43–0.47 | 9 | −0.014 | −0.0009 | −0.006 | 4 |
| 0.50–0.58 | 17 | −0.045 | −0.0001 | −0.008 | 4 |
| 0.58–0.70 | 15 | −0.080 | −0.0014 | +0.019 | 1 |

0.43 is the crossover. It was chosen on the same 32 photos it is scored on, so the r0.2 gain below is slightly
optimistic.

**Result** (ms33 decodes, default multi-scale containers, 32 photos, vs production's two stages):

| Ratio | Low net on | PSNR | SSIM | ΔE95 | ΔE99 | Better (PSNR / SSIM / ΔE95) |
|---|---|---|---|---|---|---|
| r0.1 | 32 / 32 | +0.257 | +0.0045 | −0.240 | −0.335 | 31 / 26 / 31 |
| r0.2 | 23 / 32 | +0.063 | +0.0010 | −0.067 | −0.112 | 16 / 15 / 18 |
| r0.3 | 0 / 32 | 0 | 0 | 0 | 0 | identical |

**Checks.**
- All 96 decodes are bit-identical (max difference 0) to the earlier decodes with the same fixed nets (`lo5u` or
  `prod` in `laylo_eval.csv`).
- No time cost: the third net is prefetched with the other two. ABBA on IMG_0615, 4 runs each: r0.1 1,528 vs
  1,532 ms, r0.3 1,470 vs 1,496 ms (with vs without the flag).
- Code: `cs_enhance_pick` and `CS_ENHANCE_LOW_BITS` in `image_decryption.cpp`, the mode-20 block in
  `decrypt_image_tiled`, `enhance_low_model` in `photo_upscaler.hpp`, the flag in `ImgReconstruct_backend.cpp`.

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
| `F:\Training Data\enh_data_6bc_laylo` | 6847 | multi-scale layers (`gen_enh_layers.py`): one whole-photo layer per photo, 2x for 40% (225k anchors) and 3x / 4x / 5x / 6x for 15% each, at r0.1 / 0.2 / 0.3, log-normal per-tile rates around 1.2r; target = the photo shrunk by s/2 (`INTER_AREA`) |

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
| `6bclo5_c16d2` | 11-06 | 9 / c16 / 2 | 6bc_lo5_450 (r0.05/0.1/0.2/0.3/0.4) | 29.477 (low-ratio val) | vs the 9-plane net at r0.05/0.1/0.2/0.3/0.4: +0.84/+0.56/+0.22/+0.08/−0.01 dB, SSIM 8/8/8/4/2 of 8, ΔE95 −0.66/−0.38/−0.17/−0.08/−0.02; 46 ms. Production + u2dc there: +0.12/+0.16/+0.18/+0.17/+0.14 | **production, `--enhance-low`** (11-07): first stage below 0.43 luma bits per pixel, with u2dc after it |
| `6bclaylo_c16d2` | 11-07 | 9 / c16 / 2 | 6bc_laylo (layers, r0.1/0.2/0.3) | 26.247 (layer val; 24.46 before the net) | on real multi-scale containers (ms31 default encode, 32 photos), vs production two stages at r0.1/0.2/0.3: alone +0.16/−0.04/−0.17 dB, ΔE95 −0.06/+0.06/+0.11; + u2dc +0.25/+0.04/−0.09 dB, ΔE95 −0.20/−0.05/+0.02. `6bclo5` on the same containers: alone +0.20/−0.00/−0.13, ΔE95 −0.14/+0.01/+0.07; + u2dc +0.26/+0.06/−0.06, ΔE95 −0.24/−0.07/+0.01. Peak activation 24 (fp16 safe) | not used: behind `6bclo5` at every ratio, alone and with u2dc |
| `fk450_c16d2` | 11-07 | 9 / c16 / 2 | `F:\Training Data\enh_data_fk_mix450` (FiveK raws, 4920 photos) | 35.998 (FiveK val, 34.71 before the net; not comparable to the JPEG val) | vs `6bc450_c16d2` (both single stage): FiveK test +0.53/+0.53/+0.55/+0.55 dB, ΔE95 −0.49/−0.46/−0.41/−0.41, 32/32 photos; the 18 phone photos −0.17/−0.28/−0.39/−0.42 dB, SSIM −0.003/−0.002/−0.002/−0.002, ΔE95 +0.09/+0.10/+0.09/+0.08, 1–2 of 18 better. Peak activation 11.9 | not used: worse on the phone photos (see "FiveK training data") |
| `ds2_c16d2` | 11-08 | 9 / c16 / 2 | `F:\Training Data\enh_data_ds2_mix450` (SCMI30 shrunk 2x, 6847 photos, 112.5k anchors) | 30.461 (shrunk val, 28.61 before the net; not comparable) | vs `6bc450_c16d2` (both single stage) at r0.3/0.5/0.7/0.9: 18 phone photos −0.08/−0.10/−0.11/−0.13 dB, ΔE95 +0.01; 34 edltauSCI iPhone HEICs −0.01..−0.14 dB, 0–5 of 17 better per phone; FiveK test +0.06..+0.01 dB. Peak activation 43.4 | not used: slightly worse on every phone set (see "Downscaled phone targets") |
| `6bchi5_c16d2` | 11-06 | 9 / c16 / 2 | 6bc_hi5_450 (r0.5/0.6/0.7/0.8/0.9) | 36.573 (high-ratio val) | vs the 9-plane net at r0.5–0.9: −0.04/−0.02/−0.00/+0.01/+0.01 dB, SSIM 0–5 of 8, ΔE95 within ±0.03; 50 ms | not used: no gain, the mixed 0.3–0.9 net already covers these ratios |
| `6bc450_c32d2` | 11-05 | 9 / c32 / 2 | 6bc_mix450 | 34.892 | vs the 9-plane c16: +0.11/+0.09/+0.09/+0.09 dB, ΔE95 −0.07/−0.05/−0.04/−0.04; 144 ms | not used: u2dc is better on colour and at low ratios, at two thirds of the cost |
| `6b450_c16d2` | 11-05 | 5 / c16 / 2 | 6b_mix450 | 34.659 | vs `mix450c02_c16` on 6-bit containers: +0.17/+0.38/+0.55/+0.65 dB, ΔE95 0/−0.01/−0.03/−0.03; 47 ms | production 11-05 morning, replaced the same day |
| `6b450_c16` | 11-05 | 5 / c16 / 0 | 6b_mix450 | 34.620 | ~0.03 dB and ~0.01 ΔE95 behind d2; 40 ms | not used |
| `mix450c02_c16` | 11-04 | 5 / c16 / 0 | `G:\enh_data_prod_mix450` (8-bit, 450k) | 33.83 | vs the c32: −0.20/−0.23/−0.27/−0.31 dB; 42 ms vs 142 | production 11-04, chosen for speed |
| `mix450c02_c32` | 11-03 | 5 / c32 / 0 | prod_mix450 | 34.03 | vs `fresh` on the old path: r0.3 +1.08 dB, r0.9 −0.33; with the prox-off path, IMG_3690 r0.9 39.05 dB (8-bit container) | production 11-03 |

Training runs:

| Net | Run (`x64/GpuBuild/runs/`) |
|---|---|
| `fk450_c16d2` | `20261107-173052_enh_fivek`, recipe `fivek_chain.sh` (`eval_fk.py`, `fk_eval.csv`, `crops_fk.py`) |
| `ds2_c16d2` | `20261108-003201_enh_ds2`, recipe `ds2_chain.sh` (`eval_fk.py`, `fk_eval.csv` with all four nets on all three sets, `crops_fk.py`, `ip_eval_chain.sh`) |
| `6bclo5_c16d2` | `20261105-223802_enh_lowratio`, recipe `lowratio5_chain.sh` |
| `6bclaylo_c16d2` | `20261106-233103_enh_laylo`, recipe `laylo_chain.sh` (`summary.txt`, `laylo_eval.csv`) |
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

### FiveK training data (11-07; not promoted)

**Dropped 11-08: FiveK is no longer used, and its data, nets and scripts are deleted.** The FiveK raws and develops, the
`fk450` net (`.pt`, `.onnx`, `.enh.bin`), the FiveK decodes and crops, `develop_fivek*.py`, the chain scripts and the
`ENH_FIVEK` path in `gen_enh_seg8tile_anchor.py` are gone. The results below are kept as the record. A last attempt
rendered the raws with a phone-like look (local tone mapping, detail boost, light noise reduction and sharpening). Its
settings were fitted by coordinate search so that the median tone, local contrast, edge overshoot, noise and top
spectrum band of 24 renders matched 100 SCMI30 iPhone photos. A first fit was rejected: fitting chroma cut saturation to
0.55, and the sharpening came out too strong. The fit matched the median noise (0.0040 against 0.0042), but the tail did
not. 16% of the renders were noisier than the iPhones' 90th percentile (0.0102), up to 0.16 on pushed night shots,
because one fixed noise reduction cannot do what a phone's ISO-adaptive, multi-frame one does. Its c16d2 net
(`fkp450_c16d2`) was stopped at step 11k of 128k (val 32.14 dB against 31.31 dB before the net; not comparable with the
other sets). The dataset was judged unsuitable however it was processed.

**Why.** Every net so far trains on SCMI30-IITRPR phone photos (`G:\Download\archive\SCMI30-IITRPR`), which are all JPEG
4:2:0 at luma quantization ~5–7. The targets are therefore already smoothed, and their chroma is 4:2:0 like our
`--ycc420` container. The question was whether uncompressed targets train a better net.

**Data.** MIT-Adobe FiveK: 5000 DNGs (49 GB, `F:\fivek\dng`, fetched from data.csail.mit.edu by a resumable HTTP Range
reader over the tar, `fivek_get.py` in the run folder). `Params-test/develop_fivek.py` develops the 4952 plain 2x2
Bayer raws with LibRaw (rawpy 0.28): camera white balance, AHD, sRGB 8-bit, `auto_bright_thr=0.001`, full resolution
(`F:\fivek\png`, 72 GB). Skipped: 42 Fujifilm SuperCCD (4x4, interpolated) and 6 already-demosaiced files (Canon 40D
sRAW 4:2:2, Nikon E990). `ENH_FIVEK=F:\fivek\png` makes `gen_enh_seg8tile_anchor.job_list()` (and so
`gen_enh_prod.py` and the trainer) use them: seeded shuffle, 32 test photos held out of every job, 77 val, 4843 train,
names `FK_<file>`. The set was made with the production single-scale recipe of `6bc450_c16d2` (6-bit, 450k anchors,
r0.3/0.5/0.7/0.9, prox-off with 2 ADMM rounds, ENH_CHROMA=1; 4920 photos, ~1.5 h) and trained with the same recipe
(c16d2, 9 planes, 128k steps, ~4 h).

**Evaluation** (`eval_fk.py`, 2026-11-08). Production default encode (multi-scale, ms33), decoded on the production path
with each net as a single stage, plus production's two stages for reference; r0.3/0.5/0.7/0.9. Means
(PSNR / SSIM / ΔE95):

| Photos | Net | r0.3 | r0.5 | r0.7 | r0.9 |
|---|---|---|---|---|---|
| 18 phone photos (`ImgReconstruct_backend`) | old `6bc450_c16d2` | 37.23 / 0.9313 / 3.70 | 39.51 / 0.9553 / 2.80 | 41.35 / 0.9695 / 2.35 | 41.72 / 0.9733 / 2.30 |
| | FiveK `fk450_c16d2` | 37.05 / 0.9285 / 3.79 | 39.23 / 0.9534 / 2.89 | 40.96 / 0.9680 / 2.44 | 41.29 / 0.9717 / 2.38 |
| | production (old + u2dc) | 37.31 / 0.9319 / 3.60 | 39.58 / 0.9557 / 2.72 | 41.43 / 0.9698 / 2.29 | 41.80 / 0.9736 / 2.23 |
| 32 FiveK test | old | 35.34 / 0.8477 / 6.98 | 36.23 / 0.8674 / 6.47 | 37.13 / 0.8874 / 6.02 | 37.36 / 0.8936 / 5.88 |
| | FiveK | 35.87 / 0.8535 / 6.49 | 36.76 / 0.8739 / 6.02 | 37.68 / 0.8945 / 5.60 | 37.91 / 0.9012 / 5.47 |
| | production | 35.51 / 0.8489 / 6.90 | 36.36 / 0.8685 / 6.42 | 37.22 / 0.8885 / 5.98 | 37.44 / 0.8949 / 5.85 |

On FiveK the FiveK net wins on all 32 photos at every ratio; on the phone photos it loses on 16–17 of 18 (wins: `2.png`
+0.96 dB, `IMG_9321` +0.25 at r0.3; worst: `IMG_3690` −0.74, `IMG_1297` −0.44 dB / SSIM −0.028 / ΔE95 +1.17). The
crops (`fk_vs_old_crops\`, r0.3, original | old | FiveK at each photo's best and worst 384 px window) show why: on fine
periodic texture (IMG_3690 feathers, IMG_1297 grass) the FiveK net adds sharp high-frequency structure that does not
match the phone original, which was itself smoothed by the phone's processing and compression. The net learns the
statistics of its targets, so it is a domain match, not a better net: keep the JPEG/phone-trained nets for phone
photos. A mixed set (phones + FiveK) is the untried next step if raw-camera photos matter.

### Downscaled phone targets (11-08; not promoted)

**Why.** FiveK showed the net learns its targets' look, so the targets should be phone-processed; the SCMI30 JPEGs are,
but they carry JPEG blocking and 4:2:0 chroma. Shrinking them 2x by a 2x2 box average spans whole luma blocks and
exactly one chroma sample, so those artefacts mostly vanish while the phone's colour, sharpening and noise reduction stay
(a standard way to make clean targets). The cost: detail sits at half the scale of a full-size photo.

**Data.** `Params-test/downscale_scmi.py` (EXIF transpose, crop to even size, 2x2 mean, PNG; same folder layout and
stems, 12 processes) wrote all 6847 SCMI30 JPEGs to `F:\scmi30_ds2`. `ENH_SCMI` (new in
`x64/GpuBuild/gen_enh_seg8tile_anchor.py`) points `job_list()` there, so the photos, phones and val split are the
`6bc_mix450` ones. Recipe as `6bc450_c16d2` except 112,500 anchors (450k / 4, the same anchors per pixel as the
full-size set; kept by the user's choice). Generation 0.7 h (two workers), training 128k steps ~3 h.

**Evaluation** (`eval_fk.py` with a `ds2` variant, 2026-11-08; same method as FiveK above). New third set: the 34
edltauSCI iPhone HEICs (Zenodo 10.5281/zenodo.17640511, CC-BY 4.0, `F:\edltauSCI`; 17 iPhone 13 Pro Max 12 MP iOS 18.6,
17 iPhone 17 Pro 24.5 MP iOS 26.0, converted to PNG in `F:\edltauSCI\png` as `IP13_*` / `IP17_*`). Means
(PSNR / SSIM / ΔE95):

| Photos | Net | r0.3 | r0.5 | r0.7 | r0.9 |
|---|---|---|---|---|---|
| 18 phone photos | old `6bc450_c16d2` | 37.23 / 0.9313 / 3.70 | 39.51 / 0.9553 / 2.80 | 41.35 / 0.9695 / 2.35 | 41.72 / 0.9733 / 2.30 |
| | ds2 | 37.15 / 0.9305 / 3.72 | 39.41 / 0.9547 / 2.81 | 41.24 / 0.9690 / 2.37 | 41.58 / 0.9727 / 2.31 |
| 32 FiveK test | old | 35.34 / 0.8477 / 6.98 | 36.23 / 0.8674 / 6.47 | 37.13 / 0.8874 / 6.02 | 37.36 / 0.8936 / 5.88 |
| | ds2 | 35.39 / 0.8478 / 6.95 | 36.28 / 0.8674 / 6.45 | 37.15 / 0.8872 / 6.00 | 37.37 / 0.8934 / 5.87 |
| 17 iPhone 13 Pro Max | old | 37.06 / 0.8941 / 2.73 | 38.31 / 0.9137 / 2.36 | 40.00 / 0.9353 / 2.06 | 40.45 / 0.9422 / 2.00 |
| | FiveK | 36.81 / 0.8919 / 2.79 | 38.01 / 0.9113 / 2.44 | 39.69 / 0.9329 / 2.14 | 40.10 / 0.9396 / 2.08 |
| | ds2 | 37.05 / 0.8935 / 2.73 | 38.29 / 0.9131 / 2.37 | 39.96 / 0.9346 / 2.07 | 40.40 / 0.9414 / 2.01 |
| | production | 37.11 / 0.8944 / 2.66 | 38.34 / 0.9139 / 2.32 | 40.05 / 0.9355 / 2.02 | 40.50 / 0.9424 / 1.96 |
| 17 iPhone 17 Pro | old | 38.89 / 0.9358 / 2.42 | 41.13 / 0.9571 / 1.91 | 43.03 / 0.9710 / 1.67 | 43.48 / 0.9746 / 1.64 |
| | FiveK | 38.51 / 0.9329 / 2.61 | 40.75 / 0.9552 / 2.06 | 42.57 / 0.9693 / 1.80 | 42.96 / 0.9727 / 1.77 |
| | ds2 | 38.83 / 0.9353 / 2.44 | 41.06 / 0.9566 / 1.93 | 42.91 / 0.9704 / 1.68 | 43.34 / 0.9739 / 1.65 |
| | production | 38.92 / 0.9360 / 2.35 | 41.16 / 0.9572 / 1.86 | 43.06 / 0.9711 / 1.62 | 43.52 / 0.9747 / 1.58 |

ds2 is behind the old net on every phone set at every ratio, by little (−0.01..−0.14 dB, SSIM −0.0005..−0.0009, ΔE95
+0.01..+0.02; it wins on 0–5 of 17 iPhone photos and 1–2 of 18 phone photos), and level with it on FiveK (+0.01..+0.06
dB). On the iPhone HEICs the FiveK net loses on all 34 photos (−0.25..−0.52 dB, ΔE95 +0.07..+0.19), and production's
two stages win on all 34 (+0.03..+0.05 dB, ΔE95 −0.04..−0.07), as on the other phone photos. The crops
(`fk_vs_old_crops\iphone\` for FiveK, `\iphone\ds2\` and `\ds2\` for ds2, r0.3) show ds2 a little softer on fine
foliage than the old net; FiveK adds structure the phone original does not have. Reading: the full-size JPEG targets
already match what a phone photo (HEIC included) looks like at full size; removing their compression by shrinking moves
the detail scale away from the test photos and costs more than the cleaner targets gain. The current phone-trained nets
stay; the ranking of the nets on newer iPhones (13 Pro Max, 17 Pro, iOS 18/26) is the same as on the other phone photos.

### SCMI30 + Google Pixel HDR+ photos (11-08; not promoted)

**Data.** Google's HDR+ burst dataset (`gs://hdrplusdata/20171106/results_20171023/<burst>/`, CC-BY-SA; 3,640 bursts,
the `final.jpg` of every burst 15.3 GB, `merged.dng` 73.6 GB and not pixel-aligned with the final, so not used). The
finals are 4:2:0 JPEGs with a luma quantization table mean of 5.8, like SCMI30's. Reading the 64 KB EXIF header of every
final (70 s, range requests) found 821 shot on a Pixel or Pixel XL (makers' names, `sailfish` / `marlin`;
4048×3036, 3036×4048 or 3264×2448, no EXIF rotation); only those finals were downloaded, to `F:\hdrplus\pixel`
(lists and EXIF in `F:\hdrplus\meta`). `ENH_ADD="PX=F:\hdrplus\pixel"` (new in `gen_enh_seg8tile_anchor.py`,
`add_split()`) seeds a shuffle of that folder, holds out the first 32 as a test set (`F:\hdrplus\meta\pixel_test.txt`)
and appends the other 789 as `PX_*` training jobs after the SCMI30 ones; the SCMI30 photos, phones and val split are the
`6bc_mix450` ones. Recipe as `6bc450_c16d2` (450k anchors, r0.3/0.5/0.7/0.9, prox-off decode, 9 planes, c16d2, L1, lr
5e-4, clip 0.05, EMA 0.999, crop 256, 128k steps). Chain `runs/px_chain.sh`, run `20261108-192137_enh_pixel`:
generation 2.5 h (two workers, 7636 photos, no failures), training 3.1 h. Val 34.862 dB (32.43 before the net) against
production's 34.751 (32.41): +0.11 dB on the same 77 SCMI30 photos, +0.15 dB at step 21k. Max |activation| 26, fp16-safe.
Exported `models/enh/enh_seg8a_x5_ema256_px450_c16d2_unet.onnx` (DirectML 0.21 s per 12 MP; ONNX = torch to 0 levels).

**Evaluation** (`eval_px.py` in the run folder, 2026-11-09; HIP decodes of fresh production containers in
`F:\pxeval`, exe `ms33`, metrics in `px_eval.csv`). 18 backend phone photos, the 34 edltauSCI iPhones and the 32
held-out Pixel photos, r0.3/0.5/0.7/0.9; each container decoded with the old first-stage net, `px`, production
(old + `u2dc`) and `pxprod` (`px` + the same `u2dc`). `px` minus old, single stage (PSNR dB / SSIM / ΔE95, photos
better on SSIM / ΔE95):

| Photos | r0.3 | r0.5 | r0.7 | r0.9 |
|---|---|---|---|---|
| 18 phone photos | −0.003 / −0.0001 / −0.004 (3, 14) | −0.025 / 0 / 0 (4, 11) | −0.024 / 0 / +0.002 (3, 3) | −0.023 / 0 / +0.001 (3, 7) |
| 34 iPhones | +0.003 / +0.0001 / −0.008 (15, 31) | −0.008 / 0 / −0.001 (17, 17) | −0.004 / +0.0001 / 0 (20, 14) | −0.004 / 0 / −0.001 (17, 15) |
| 32 Pixel test | +0.026 / +0.0004 / −0.028 (24, 32) | +0.015 / +0.0003 / −0.017 (23, 31) | +0.024 / +0.0002 / −0.009 (27, 30) | +0.025 / +0.0003 / −0.008 (28, 28) |

With the second stage (`pxprod` minus production) the differences are about the same or smaller: Pixel +0.013..+0.021 dB,
ΔE95 −0.005..−0.021 (27–32 of 32 better); phone photos −0.008..−0.027 dB; iPhones −0.008..0 dB. Means over all ratios,
production / `pxprod`: phone photos 40.03 / 40.01 dB, SSIM 0.9578 / 0.9577, ΔE95 2.71 / 2.71; iPhones 40.33 / 40.33,
0.9407 / 0.9407, 2.04 / 2.04; Pixel 35.02 / 35.03, 0.9156 / 0.9158, 4.28 / 4.27.

**Reading.** The Pixel photos help only on Pixel photos, and by little (+0.02 dB, ΔE95 −0.01..−0.03, consistent across
photos). On the other phones it is level (iPhones) or slightly behind (phone photos, 15–16 of 18 worse on SSIM by
≤0.0001). The +0.11 dB on SCMI30 val did not carry over to any test set. 789 photos are 10% of the set; the net already
generalises across phones, so more photos of the same kind (phone JPEGs) add nothing measurable. Production stays.

**Matched second stage `px450_u2dc` (11-09; not promoted).** `pxprod` reuses production's `u2dc`, which was trained on the
old net's output, so a second stage was trained on `px`'s: the `u2dc` recipe (`runs/unrolled_chain.sh`: `ENH_STAGES=2`,
`px450_c16d2` frozen as stage 1, fresh c16d2 stage 2, `ENH_DC=1`, lr 5e-4, 64k steps) on the SCMI30 + Pixel set; chain
`runs/pxu2_chain.sh` (started after a 2 h wait at the user's request), 1.9 h. Val 34.917 dB, +0.055 over `px` alone
(production's `u2dc` added +0.083 over its first stage). Max |activation| 13.9. Exported
`models/enh/enh_seg8a_x5_ema256_px450_u2dc_unet.onnx`; decoded as variant `pxu2` (px + px u2dc) by `eval_px.py` on the
same containers. Against production (PSNR dB / SSIM / ΔE95, ranges over r0.3..0.9): phone photos −0.04..−0.07 /
−0.0002..−0.0004 (0 of 18 better on SSIM) / +0.01..+0.06; iPhones −0.01..−0.03 / −0.0001 / +0.02..+0.04 (1–3 of 34
better on ΔE95); Pixel −0.005..+0.001 / −0.0000..−0.0006 / −0.009..+0.041. It is also behind `pxprod` (the mismatched
`u2dc`) on every set and ratio, mostly in ΔE95 (+0.01..+0.05) and on SSIM for nearly every photo. Means over all ratios,
production / `pxu2`: phone photos 40.03 / 39.97 dB, SSIM 0.9578 / 0.9575, ΔE95 2.71 / 2.75; iPhones 40.33 / 40.31,
0.9407 / 0.9406, 2.05 / 2.08; Pixel 35.02 / 35.02, 0.9156 / 0.9154, 4.28 / 4.30. So the matched stage loses its
first stage's small Pixel gain and worsens colour; one retrain of a 64k-step second stage evidently varies by more than
the mismatch costs. Production's pair stays.
`F:\Training Data\enh_data_6bc_px_mix450` (the generated set) and `F:\pxeval` are kept.

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
