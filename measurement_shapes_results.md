# Measurement-shape experiment

## Implementation and reproduction

- Benchmark: `tests/bench_measurement_shapes.cpp`
- Build/run: `scripts/bench_measurement_shapes.bat`
- Source: `ImgReconstruct_backend/IMG_3690.png`, 4032 × 3024.
- Reference idea: Katz et al., *Rethinking Compressive Sensing: A Compression Framework for Video Super-Resolution*, CVPRW 2025, Sections 3.2–3.3. This experiment tests single-image four-group averaging, not their temporally changing, two-measurement VSR system.

From the repository root:

```powershell
.\scripts\bench_measurement_shapes.bat --self-test
.\x64\Cpu\bench_measurement_shapes.exe ImgReconstruct_backend\IMG_3690.png shape.full.uniform 100 0.005 17
.\x64\Cpu\bench_measurement_shapes.exe ImgReconstruct_backend\IMG_3690.png shape.full.segment 100 0.005 17 scene.colors.png 1
```

Arguments: input, output prefix, steps per reweight, lambda, seed, optional scene palette mask, optional adaptive rate scale (0 < scale <= 1).

Each run writes a CSV, three reconstructions, and three raw measurement payloads in the repository root. Payloads are experimental data, not authenticated encrypted containers. The seed, geometry, and optional segmentation schedule must be supplied to reproduce the operator; the binary files alone are not standalone image files.

## Matched design

Three operators per nonoverlapping 4 × 4 acquisition cell:

1. Four unique random individual pixel measurements.
2. Four nonoverlapping 2 × 2 square means.
3. Four nonoverlapping T means, covering all 16 pixels:

```text
A A A C
B A C C
B B D C
B D D D
```

All operators use one unsigned 8-bit value per measurement per BGR channel. Means are rounded to nearest integer, with half ties rounded up. The solver uses these quantized values, not unquantized source means. No YCC subsampling, TV, neural super-resolution, class-specific reconstruction, or boundary postprocessing is applied.

Common reconstruction: serial CPU, 64 × 64 DCT-FISTA patches, stride 48, 100 steps per reweight, two reweights, lambda 0.005, zero initialization, unpenalized DC, identical overlap feathering. Averages are scaled by sqrt(group size) internally so every disjoint measurement row has unit norm; all operators therefore have the same valid FISTA step. Lambda is matched in this normalized objective. This normalization does not make quantization-error variances identical: 8-bit mean quantization and 8-bit point quantization have different effective noise after row scaling.

Overlapping reconstruction patches reuse acquisition measurements. A global-cell keyed PRNG gives the same groups/point locations wherever a cell appears in overlapping patches. Replicated edge padding is included in the reported sample count and payload size. PSNR/MAE use the original unpadded image.

Self-tests cover exact T geometry, disjoint/complete averaging partitions, measurement count, forward/adjoint dot-product identity, unit row norms, and constant-image reconstruction.

## Full-resolution uniform budget

Exactly **3,076,480 measurements/channel**, **9,229,440 payload bytes/operator**, including padding. Nominal unpadded measurement ratio: 4/16 = 25%.

| Operator | PSNR dB | MAE | Seconds |
|---|---:|---:|---:|
| Random points, seed 17 | 31.4040 | 3.47838 | 140.609 |
| Square means | **34.1689** | **2.47087** | 166.521 |
| T means | 33.7393 | 2.71001 | 167.328 |

Outputs: `shape.full.uniform.{random_points,square_means,tetromino_means}.png` and `.measurements.bin`; CSV: `shape.full.uniform.csv`.

Square means lead T means by 0.43 dB here. Changing the seed does not change the full-budget square/T measurement sets: all four groups are acquired; only their order changes. Thumbnail seeds 17/29/43 also favored square means (27.98 vs 27.68 dB); random points were about 23.5 dB.

## Segmentation-guided budget

This experimental extension supports 0–4 measurements/cell. Relative scene priorities are `[.2, .825, .2, .2, .5, .8, .5, .5]`, normalized by .825 and scaled to a maximum of four measurements. A cell's target averages its 16 pixel priorities. Carry-based rounding produces an integer count; a seeded group order selects that many distinct square/T groups, or that many distinct random points. The schedule is **identical across shapes**. Group selection may cross segment contours. This initial adaptive experiment does not apply the production `--seg-blend` density smoothing.

Consequently the highest-priority bird interior gets 25% measurements, not 82.5%: this benchmark is capped at four measurements/cell to preserve the initial experiment's operator family. It does not reproduce the production `--seg-rates` absolute sampling rates.

Exactly **1,856,671 measurements/channel**, **5,570,013 payload bytes/operator**, including padding, for both tested seeds.

| Operator | Seed | PSNR dB | MAE | Seconds |
|---|---:|---:|---:|---:|
| Random points | 17 | 27.7457 | 5.99524 | 140.230 |
| Square means | 17 | 33.0566 | 3.34612 | 150.100 |
| T means | 17 | **33.1384** | **3.26129** | 149.657 |
| Random points | 29 | 27.7061 | 6.01874 | 137.034 |
| Square means | 29 | 33.0542 | 3.34718 | 151.175 |
| T means | 29 | **33.1386** | **3.26182** | 150.517 |

Outputs: `shape.full.segment.*` and `shape.full.segment.s29.*`.

## Interpretation and next integration step

- Square averaging wins the uniform four-measurement experiment on this photograph.
- T averaging narrowly wins the tested segmentation-guided budget, by about 0.08 dB on this photograph for two seeds. This is a small, image-specific advantage; adaptive thumbnail tests at 300 steps did not consistently favor T over square.
- Neither result establishes a universal winner. Test more images and matched convergence/lambda sweeps before choosing a default. These PSNR values are not directly comparable to the production segment solver's earlier results: acquisition budgets, color representation, regularization, and iteration conventions differ.
- The requested adaptive combination is implemented in this standalone benchmark for all three operators, with one shared reconstruction pass. Production encrypted-container support is not implemented. That would require shipping the operator/version and budget schedule, matching forward/adjoint support in production FISTA, and rules for measurements intersecting reconstruction-tile boundaries. It should preserve existing point-sample containers rather than reinterpret their values as averages.
- Timings include acquisition/payload export and reconstruction in this deliberately serial reference implementation. They are not performance forecasts for a parallel production implementation.
