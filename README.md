This is an implementation of an encryption algorithm for images using compressive sensing, this is meant to be more of a proof of concept where we use compressive sensing in a novel way to encrypt images. CS is typically used to sample signals in an already compressed format but a by-product of this process is that the signal also becomes uniquely encoded upon measurement, this can be used for encryption .

The overall algorthm for this is as follows : 

```
Inputs : 𝜓 measurement matrix, 𝑥 vectorized image to be encrypted, 𝑁 signal length of 𝑥, 𝑘number of iterations for ADM, 𝜏, 𝛽 parameters for ADM, 𝑒𝑘, θ auxiliary vectors

// encryption step
𝑦 ← 𝜓𝑥 // extracting the measurements, this effectively encrypts the signal 𝑥 into y

//decryption step
For 𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // construct dictionary 𝐴
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝐴 : , 𝑖 = 𝜓 ∗ θ
End For

𝑠 = 𝐴𝐷𝑀(𝐴, 𝑦, 𝜏,𝛽, 𝑘) // solves the 𝐴 * 𝑥 = 𝑦 equation, where 𝑥 is the unknown encrypted signal using LMBFGS

For  𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // reconstruct signal 𝑥 which now represents the decrypted signal 𝑥′
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝑥′ = 𝑥′ + θ ∗ 𝑠(𝑖)
Sf - End

Outputs : 𝑥′ vectorized decrypted image
```

This works only if the  measurement matrix ```𝜓``` is identical upon encryption and decryption. ```𝜓``` is meant to be a random matrix but by using a deterministic number generator which is seeded using a passphrase we can encrypt and decrypt an arbitrary signal. Keep in mind that this method is not lossless, the reconstructed signal will not be 100% identical, this is why you'd only want to use something like this for things like images. ```𝑦``` represents the encrypted image, since it's obtained by multiplying the original image with a random matrix it will contain a bunch of seemeingly random numbers.

Because ```𝑥``` is a vectorized image which means it can have millions of elements the dictionary ```𝐴``` is going to be a matrix with potentially billions of elements (so dozens of GB in size). The challenge in doing something like this comes from the fact that the matrices involved occupy so much memory that it's impossible to solve this problem on a regular computer as is, however, we can divide the original image in smaller chunks that can fit in the memory of a typical computer.

The tile solves run reweighted-L1 FISTA (per-channel, or SOMP-structured joint over R/G/B), with an optional HIP GPU path (see `--device` below).

This method processes the image in tiles, it should be noted that this is technically not equivalent to solving this problem for one single large image, however for something like images it works quite well and can even improve quality in some ways (lower noise) when the compression ratio is higher.

## Security design (v2 container format)

The passphrase is the single root secret. It is stretched with **PBKDF2-HMAC-SHA256** (Windows CNG, 200,000 iterations, 16-byte random per-image salt) into a 256-bit key. The old `std::hash`-based `generate_seeds` chain (hash-then-pick-then-reseed funnel) was removed; the derived key is the only seeding input, so there is no accidental entropy bottleneck and offline passphrase guessing is deliberately expensive.

The 256-bit key is used for three things:

- **Header encryption** — the metadata header (`m|rows|cols|height|width`) is AES-256-CTR encrypted, so image dimensions are no longer recoverable from the ciphertext without the passphrase.
- **Authentication** — an HMAC-SHA256 tag (truncated to 128 bits), computed encrypt-then-MAC over the encrypted header *and* the entire measurement body, is stored in the header. Decryption verifies it first: a wrong password or any tampering with the container is rejected before the expensive tile solves start.
- **Index regeneration** — MT19937 is seeded via `std::seed_seq` directly from the key bytes; the pixel shuffle seed is the first 4 key bytes. Both sides derive identical sampling indices from passphrase + salt alone.

Header layout (v2, 32 pixels = 96 bytes at the start of the container):

```
[0]      version byte (2)
[1..16]  PBKDF2 salt (random, per image)
[17..32] AES-CTR IV (random, per image)
[33..64] metadata (32 bytes) XOR AES-256-CTR keystream
[65..80] HMAC-SHA256 tag, truncated to 16 bytes, over [0..64] + measurement body
[81..95] sampling mode + geometry (authenticated):
          [81]     mode: 0 = random, 1 = periodic, 2 = adaptive,
                   3 = YCC420, 5 = HF-focus, 6 = YCC420 + HF-focus,
                   7 = YCC422, 8 = YCC422 + HF-focus
         [82..83] tile_size (LE, modes 1 and 2)
         [84..85] mode 1: samples_per_tile; mode 2: lod byte count (sanity)
         [86..87] mode 2: weight_base (LE uint16; strength → base mapping below)
```

### Adaptive sampling (`--adaptive`)

Mode 2 splits the measurement budget across a tile grid by an 8-bit
**level-of-detail** score per tile (mean |Laplacian| by default, min-max
normalized, floored with `--adaptive-floor`, default 32). Each tile's share is weighted
by `tile_pixels * (weight_base + lod)` — capacity keeps total allocation
proportional to tile size, while `weight_base` controls how hard detail tiles
are pushed over flat ones. `--adaptive-strength <s>` (default 0.5) sets that
base: `s=0` → 65535 (essentially uniform, like random), `s=0.5` → 256
(tuned default, density ratio near 1.8× so flat tiles retain enough coverage
for the wavefront warm-start while detail tiles still get a clear bonus),
`s=1` → 1 (maximum LOD bias, density ratio near 7.8×). The base is stored in
the authenticated pad so decryption regenerates the exact counts. The lod
byte array is stored immediately after the
header (one byte per tile, row-major, zero-padded to a whole container
pixel); measurements follow. Decryption regenerates identical indices from
`(key, lod, m, geometry, weight_base)` via `cs_compute_tile_sample_counts`
(deterministic largest-remainder split with capacity clamp/redistribute) plus
a per-tile Fisher–Yates draw — no decrypt-side flag needed (mode travels in
the header). `--show-mask` (encrypt/roundtrip) writes `<output>.mask.png`
(source | binary mask | cyan overlay) and opens the same view in a window
after sampling. **Resolution:** by default encrypt downsamples the input 2×
and decrypt 2×-upscales the solve (half-res pipeline). Pass `--full-res` on
**both** encrypt and decrypt/roundtrip to keep native geometry end-to-end.

**LOD scorers:** `--two-pass` replaces the Laplacian with a pilot uniform
sample + cheap per-tile recon (residual scoring); `--spectral` replaces it
with per-tile DCT band energy whitened by the image's global radial spectral
decay (tiles pay for high-frequency energy the natural-image prior does not
predict, so smooth gradients score low even when bright). `--two-pass` and
`--spectral` are mutually exclusive; both imply the adaptive container and
need no decrypt flags. `--regions` blends on top of any base scorer, and
`--lod-smooth` blurs the shipped bytes. Measured on IMG_3690 at ratio 0.5:
spectral ties Laplacian within ±0.1 dB at default and maximum strength
(27.2/27.6 dB) — rank orders correlate on natural content; spectral is the
cheaper deterministic alternative to the pilot solve, complementary where
edge magnitude misleads (smooth high-contrast gradients).

### YCC 4:2:0 split sampling (`--ycc420`)

Mode 3 applies the JPEG insight directly: encrypt converts to YCrCb,
downsamples chroma 2×, samples luma at the full `--ratio` budget and each
chroma plane on its own coarse grid (domain-separated key streams), and packs
the body as raw bytes `[Y][Cr][Cb]`. The container holds ~1.5 bytes/px instead
of 3 at the same ratio, and chroma counts re-derive from the luma budget so no
extra header fields were needed (mode byte in the authenticated pad).
Decrypt auto-selects a dedicated tile pipeline: luma solves full-res
(OWL-QN/FISTA/ADMM per `--solver`), chroma solves natively coarse with
reweighted-L1 FISTA (the near-flat chroma warm start stalls OWL-QN's orthant
projection; proximal updates converge directly), then merges YCrCb→BGR per
tile. Measured: 21.9 dB / 0.51 SSIM synthetic at ratio 0.5 in ~half the
container bytes (BGR modes sit ~16–17 dB there); 29.2 dB at ratio 1.0.
The adaptive LOD flags (`--adaptive`, `--two-pass`, `--regions`,
`--lod-smooth`, `--adaptive-strength/floor`) are absorbed by `--ycc420`:
they drive the luma budget (LOD bytes ship in the container exactly like
mode 2) while chroma stays uniform; `--periodic` is ignored with a warning.
Chroma planes solve natively on their coarse grids. Joint solvers degrade to per-channel FISTA here.

### YCC 4:2:2 split sampling (`--ycc422`)

Mode 7 is the same luma/chroma-split construction with chroma downsampled
horizontally only (full height, half width). The container holds ~2 bytes/px
instead of 3, and chroma edges (text, colored boundaries) reconstruct
sharper than 4:2:0 at a larger container. Same body layout (`[Y][Cr][Cb]`,
counts re-derived from the luma budget), same absorbed adaptive flags, same
decrypt pipeline with full-height chroma tiles; `--periodic` is ignored with
a warning, `--ycc420` is mutually exclusive (error), and `--hf-focus`
selects the mode-8 HF-weighted variant (thumbnail + edge-weighted luma,
uniform 4:2:2 chroma). `--sample-bits`
`L[,C]` applies to luma/chroma exactly like 4:2:0. Measured on the 480×360
synthetic suite at ratio 0.5: 23.37 dB / 0.75 SSIM vs 22.68 dB for 4:2:0 at a
1.33× container.

### HF-focus sampling (`--hf-focus`)

Modes 5, 6 and 8 store an authenticated, lossless PNG thumbnail (maximum dimension
128px) in the container. Per-tile sample counts remain nearly uniform, but
sample positions are drawn with probability weighted by the thumbnail's
Laplacian magnitude. This concentrates measurements on edges while preserving
coverage in smooth tiles. The thumbnail is encrypted with AES-CTR using a
domain-separated counter and is also used as a first-wave solver warm start.
Use `--tile-size 32` for a fine HF grid; decrypt detects the mode from the
container.

`--hf-focus` is an encrypt/roundtrip option and cannot be combined with
`--periodic`, `--adaptive`, `--two-pass`, `--regions`, `--lod-smooth`,
`--lod-full`, or a non-default adaptive strength. Combine it with `--ycc420`
for mode 6 or `--ycc422` for mode 8: HF-weighted luma sampling plus uniform
coarse-grid chroma (4:2:0 halves both chroma axes, 4:2:2 halves the width
only, so mode 8 keeps full-height chroma detail).

`--lod-full <n>` (encrypt/roundtrip only, default 0 = off) is the detail
guarantee on top of adaptive sampling: tiles scoring LOD >= n (1–255) are
sampled at 100% regardless of `--ratio`, before the remaining budget splits
over the rest. Qualifying tiles are taken in LOD-descending order while they
fit the budget (top-detail wins; a warning names the shortfall); the
threshold travels in the authenticated header pad and decrypt recomputes the
identical counts, so no decrypt-side flags are needed. Implies adaptive
scoring (absorbed into the LOD-luma variant under `--ycc420`); rejected with
`--hf-focus`, which has its own budgeting. Alone it selects the adaptive
container. Measured on 256×256, ratio 0.75: two full tiles hit the AVIR
ceiling (25.2/22.5 dB per quadrant) while the pool shares the rest, matching
proportional-split total PSNR (22.21 dB) with the detail tiles protected.
Over-subscribed guarantees (qualifiers exceeding the budget) leave pool
tiles starved — the warning tells you when that happens; lower the threshold
or raise the ratio.

### Measurement bit-depth (`--sample-bits`, encrypt/roundtrip only)

Stored samples are plaintext in secret order, so fewer bits per value is
the structural size lever (besides ratio and YCC-vs-BGR). Use
`--sample-bits L[,C]`, with each depth in [1, 8]. One value applies to
both planes; with two values, luma uses L and chroma uses C. Depths and an
extension marker are written in the header pad (also bringing the pad under
the header MAC). A zero chroma byte means "same as luma," so older headers
still decode as 8/8. LOD/thumbnail prefixes remain raw. Decrypt
auto-detects both values; works in every sampling mode (0/1/2/3/5/6).

In BGR sampling modes there are no explicit Y/Cr/Cb planes: G uses the
luma depth while B/R use the chroma depth in the interleaved bitstream.
This retains the legacy stream when both depths are equal.

Measured on a 256x256 photo, ratio 0.5, full-res (ADMM 8 iters):

| mode | depths | container | PSNR |
|---|---|---|---|
| standard | 8 | 76 KB | 28.96 dB |
| standard | 4 | 43 KB (−44%) | 27.36 dB (−1.6 dB) |
| ycc420 | 6,4 | 33 KB | 26.76 dB |
| hf-focus | 5 | 62 KB | 28.57 dB |
| adaptive | 4 | 43 KB | 27.36 dB |

PNG byte sizes vary slightly with the random salt and resulting payload bytes.

Security caveats (this is still a proof of concept):

- **MT19937 is not cryptographically secure.** With 624 consecutive known outputs an attacker can recover the generator state. The PBKDF2 KDF keeps guessing the passphrase expensive, but the sampling stream itself must not be trusted against an adversary who can infer measurement values. A cryptographically secure stream (ChaCha20 / AES-CTR) driving the index selection would close this.
- Key material (`cs_key`) is wiped with a volatile write loop after sealing (encrypt) and in the `CSencryption` destructor (decrypt), but passphrase copies held in `std::string` by callers are outside this module's control.
- **Format change**: images encrypted with the previous format (11-pixel plaintext header, no version byte, `std::hash` seeding) can no longer be decrypted and are rejected with a legacy-format error — re-encrypt them.

Performance note : the PBKDF2 stretch adds ~0.2–0.4s per encrypt/decrypt operation (once per image, not per tile), which is negligible next to the solve time.

## Building

Requirements: C++17 compiler, OpenCV (developed against 4.13.0; any 4.x with core, imgproc, imgcodecs, highgui, photo), OpenMP, and a crypto backend (Windows CNG is built in; Linux/macOS use OpenSSL). x86-64 with AVX2+FMA is assumed for the solver kernels.

### Clang (Windows, default)

`build_clang.bat` is the default build: LLVM `clang-cl` 23 (via `winget install LLVM.LLVM`) with VS 2022 headers/link, ROCm HIP for `cs_gpu.hip`, OpenCV at `C:\opencv` (main binary, 4.90 world) plus the static 4.13 set with contrib under `Documents\opencv\Release` (test binary, mirrors the vcxproj split):

```bat
.\build_clang.bat
```

Outputs land in `x64\Clang\` (`ImgReconstruct_backend.exe`, `cs_tests.exe`), leaving the MSVC outputs untouched.

### Visual Studio (Windows, legacy)

1. Open `ImgReconstruct_backend.sln` (MSVC v143 / VS 2022 or newer).
2. The project expects OpenCV headers/libs at `C:\opencv\include` and `C:\opencv\lib` (see the `IncludePath`/`LibraryPath` entries in `ImgReconstruct_backend.vcxproj`); adjust if yours differs.
3. Build `Release | x64`. `bcrypt.lib` (CNG) is linked automatically. For `--device gpu`, the AMD HIP SDK (7.2 tested, `hipcc` + `amdhip64.lib`) must be installed; `hip_build.targets` (imported by both vcxprojs) compiles `cs_gpu.hip` for `HipArch` (default `gfx1100`) and links/copies the HIP runtime.

### CMake + vcpkg (Windows/Linux)

A vcpkg manifest is included (`vcpkg.json`: `opencv4`, `openssl`).

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build            # runs the cs_tests suite
```

Alternatively point CMake at any OpenCV with CMake config files via `-DOpenCV_DIR=<path>`; on Windows the CNG backend means OpenSSL is not required.

### Test suite

`cs_tests` covers: PBKDF2/seal/verify unit tests, wrong-password and tamper rejection, seed determinism, shuffle determinism, tile-grid helpers, synthetic encrypt→decrypt, adaptive sample-count/roundtrip (including --lod-full counts identity/full/fit-budget/tiny-budget plus roundtrip header/tamper/range/hf-reject checks), YCC420 roundtrip, HF-focus mode 5 and mode 6 roundtrips, sample-bits quantize/pack/unpack unit checks plus 8-bit/4-bit/split-depth YCC packed roundtrips with header/tamper checks, AVIR upscaler geometry and waifu2x fallback/argument composition, and an opt-in photo roundtrip. Run with `ctest` or directly (`tests\\x64\\Release\\cs_tests.exe`).

### CLI usage

```
ImgReconstruct_backend encrypt  <input.png> <output.png> [--password <pw>] [--ratio R] [--periodic] [--adaptive] [--hf-focus] [--ycc420] [--tile-size N] [--adaptive-floor N] [--adaptive-strength S] [--lod-full N] [--sample-bits L[,C]] [--show-mask] [--full-res]
ImgReconstruct_backend decrypt  <input.png> <output.png> [--password <pw>] [--tiles N] [--overlap N] [--iterations N] [--threads N] [--coef F] [--per-tile-coef] [--per-tile-tv] [--manual] [--no-preview] [--photo-upscaler avir|waifu2x] [--device cpu|gpu]
ImgReconstruct_backend roundtrip <input.png> <output.png> [options]
ImgReconstruct_backend compare   <original.png> <decrypted.png> <out_prefix> [--amp F]
```

`compare` scores a decryption against the original (PSNR, SSIM, MAE, max abs error) and writes `<out_prefix>_sidebyside.png` (original | decrypted | JET error map) plus `<out_prefix>_error.png` (amplified absolute-error map) for inspecting tile seams, blur, and solver noise. `--amp` sets a fixed error-map gain (default: auto-scale to peak).

Without `--password` the passphrase is read from `CS_ENCRYPTION_PASSWORD` or prompted. Parameter overrides (`--tiles` etc.) switch the decrypt to manual mode; `--tiles` accepts any count >= 1 (auto mode derives 24); otherwise parameters are derived automatically from the container.

The standard decrypt output is 2× the solve geometry, which restores the
original input dimensions after the default 2× encrypt downscale.
`--full-res` keeps native solve geometry.

`--per-tile-coef` (decrypt only, default off) scales the L1 coefficient per
tile from its sample count (`coef·sqrt(mean/count)`, clamped to
[0.25×, 4×]; tv stays global unless --per-tile-tv is given (per-tile TV can draw the tile grid).
Starved tiles regularize more, rich tiles relax. Measured ~zero effect on a
4032×3024 photo (identical PSNR, identical boundary/interior error split):
overlap sharing homogenizes per-tile counts and reweighted FISTA absorbs the
residual λ differences. Harmless, occasionally useful on strongly
non-uniform sampling; watch zoomed tile boundaries when trying it.
`--per-tile-tv` applies the same law to the TV weight (luma only; base tv 0
stays 0, independent flag so seam effects stay attributable). It measures
bit-identical on tested content: unlike the discontinuous L1 threshold,
which flips marginal coefficients into visibly different supports, the
smooth TV perturbation lands below 8U output precision. Same verdict.
`--photo-upscaler avir|cs-sr|sr-direct|fsrcnn|waifu2x` selects the per-tile 2x upscaler backend (default
`avir`). Waifu2x batches tiles through the optional nunif Python package and
falls back to AVIR if the command or any tile fails. Set `CS_WAIFU2X_CMD` if
Python/nunif needs a custom launcher; `--waifu2x-cmd` and `--waifu2x-args`
override it. To denoise while upscaling, use
`--waifu2x-method noise_scale --waifu2x-noise 2` (noise range 0–3).

`--photo-upscaler fsrcnn` runs FSRCNN 2x in-process via OpenCV dnn_superres
(CPU, one session per worker thread, ~10–20 ms/tile). Needs the weights file
(`--fsrcnn-model`, default `FSRCNN_x2.pb` resolved against the working
directory); a missing model or a build without `opencv2/dnn_superres.hpp`
falls back to AVIR per tile with one warning. Measured at ratio 0.5:
IMG_3690 photo 26.79 dB / 0.772 SSIM vs. AVIR's 26.81 dB / 0.768 in ~26 s vs.
~16 s — i.e. a tie, not a win: FSRCNN is trained on clean bicubic-downsampled
images while decrypt feeds it FISTA-reconstructed tiles with solver noise
(degradation mismatch, the BSRGAN-literature effect), so the learned prior
cannot deploy. The CS-SR path, which refits the actual measurements
(27.31 dB / 0.813 with `--red 0.75`), stays ahead here; fsrcnn remains the
fast deterministic neural option (and the harness fits ESPCN/ECBSR/SPAN
weights with the same 2x contract).

`--photo-upscaler cs-sr` enables CPU 2× compressed-sensing super-resolution
for BGR containers in the default half-resolution pipeline. Each reconstructed
LR tile is first upscaled with AVIR; a reweighted-L1 FISTA solve then refines a
sparse HR DCT correction of the **luma channel only** against the **original
sampled LR luma** (BGR samples combined with the YCrCb weights) using a 2×2
box-downsample forward model, a quadratic anchor to the AVIR image, and the
optional `--tv` smoothness penalty. The anchor preserves HR detail that LR
measurements alone cannot determine; chroma comes straight from the AVIR
upscale, so one HR solve runs instead of three. It uses the existing tile
overlap and feather blending; small/unsampled tiles and wavelet-basis solves
use AVIR. `--ycc420`/`--ycc422` also use AVIR because split luma/chroma
sampling needs a different SR operator; `--full-res` skips upscaling entirely.
Example:
`ImgReconstruct_backend roundtrip input.png output.png --ratio 0.5 --photo-upscaler cs-sr --no-preview`.
Measured at ratio 0.5: IMG_3690 photo 27.17 dB / 0.784 SSIM vs. AVIR's
26.81 dB / 0.768, taking 46 s vs. 16 s; 256px synthetic 19.36 dB / 0.624 vs.
19.22 dB / 0.624 in 0.4 s (CPU; image dependent). An explicit iterative
back-projection closure was trialed and removed — pulling toward the noisy LR
solve hurt SSIM with no PSNR gain.

`--red <f>` in [0, 1] (default 0 = off) adds a RED-lite natural-image prior
to the cs-sr solve: after each outer FISTA pass the HR luma is blended
toward its fastNlMeans-denoised self (`x ← x − red·(x − D(x))`, Romano et
al.). Unlike DCT sparsity, the denoiser knows what natural texture looks
like, so structure (SSIM) improves even where point-wise error (PSNR)
barely moves. Same IMG_3690 setup: `--red 0.5` → 27.30 dB / 0.810,
`--red 0.75` → 27.31 dB / 0.813 (measured best), `--red 1.0` → 27.27 dB /
0.808 (over-smoothing starts); cost rises only ~4 s (NLM runs once per
outer pass, not per iteration — a per-gradient-step denoiser is infeasible
on CPU at these tile sizes, which is also why BM3D is left out: minutes per
call). Synthetic 256px: SSIM 0.624 → 0.674 at `--red 0.5` with PSNR flat,
the classic denoiser-prior signature.

`--photo-upscaler sr-direct` selects the single-stage variant: the HR luma
is solved directly from the LR samples (`cs_sr_direct_luma`, no LR-solve
warm start inside the coefficients) and merged with the AVIR chroma.
Measured on the fixture it trails the two-stage path on real photos
(warm-start economics dominate: integrated direct 26.79 dB / 0.707 in
128 s vs 27.31 dB / 0.813 in 50 s for two-stage+RED), but it is available
for BGR containers when the LR stage is best skipped; ycc containers fall
back to AVIR and `--red` is ignored there (single-stage solve has no RED
passes). `--sr-dict` takes precedence over both CS backends.

`--sr-init avir|cascade` (default `avir`, decrypt/roundtrip only) selects
the warm start for `cs-sr` and `sr-direct`: `cascade` runs the multiscale
coarse-to-fine binned-FISTA chain (`cs_sr_cascade_init`) instead of the
AVIR upscale to seed the HR solve. On the fixture, cascade + standard
budget reaches 36.76 dB in 0.016 s, beating zeros + 4× budget (36.05 dB
in 0.055 s) — the blocker was init quality, not iteration count
(HF-focus thumbnails already ship this idea at image scale).

`--device gpu` (decrypt, `--solver fista|joint`) offloads the batched tile solves to an AMD GPU over HIP: tile/channel solves queue up and flush as batched kernel groups (proximal-gradient + momentum over FFT-DCT transforms with sampled gradients; joint: row-coupled group threshold over stacked planes), while CDF97 and any device failure silently fall back to the CPU path (output differs from CPU only by float rounding). Set `CS_GPU_DEBUG=1` for per-batch timing lines. Building requires an AMD HIP SDK (7.2 tested) with the target GPU in `HipArch` (`hip_build.targets` compiles `cs_gpu.hip` through `hipcc` at link time).

Quality notes: tiles are composited with a **cosine-feathered** weight ramp (width = tile overlap) instead of a fixed alpha blend — overlap zones sum to a smooth transition and the composite is order-independent. Tiles are solved in **wavefront (anti-diagonal) order**: every tile is warm-started from the already-solved west/north neighbors' overlap strips, which turned out to be by far the largest quality lever — roundtrip baselines went from ~19.2/16.7 dB (independent solves) to **~26.8/30.4 dB** (synthetic/photo) with no wall-time penalty, since the better warm-starts converge faster than the wave barriers cost. A YCrCb-domain solve was re-tested properly on top of the wavefront (neighbor chroma strips, softened chroma l1) after the first attempt turned out to have a channel-ordering bug — the corrected result is a tie with BGR (26.7/30.3 vs 26.8/30.4 dB) at higher cost, so BGR remains the default; the path is kept behind a flag for future tuning experiments (see `decrypt_image::decrypt`).

Performance : the decrypt solver saturates by ~5 L-BFGS iterations across compression ratios 0.25–1.0 (`tests/bench_iterations.cpp`: PSNR plateau at 5 steps everywhere, time linear beyond it), so auto mode allocates `ceil(5/ratio)` iterations capped at 8 — previously `10/ratio` capped at 30, which overshot 2–6× with identical output quality.

Performance : 

~1.0 seconds to decompress and decrypt a 4032 X 3024 image on a Ryzen 7900 after adding upscaling using [this library](https://github.com/avaneev/avir).

TODO List : 

- [ ] Improve quality of decrypted image, right now there are fair amount of artifacts upon closer inspection.

Example :

Original image :

![IMG_36902](https://github.com/user-attachments/assets/52e80e7a-c58d-4466-9c72-11882fc827a3)

Encrypted image : 

![encrypted_img_g2](https://github.com/user-attachments/assets/639e7ad4-c9aa-4c1d-ab6e-74e2bdd701a4)

Decrypting the image :

![unknown_2025 04 04-22 37_1-ezgif com-cut](https://github.com/user-attachments/assets/37cdb048-2e94-4be1-9b65-555813523fde)
