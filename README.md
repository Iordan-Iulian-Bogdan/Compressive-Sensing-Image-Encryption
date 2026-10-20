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
                   7 = YCC422, 8 = YCC422 + HF-focus,
                   13 = T-cluster points, 14 = T-means groups
                   (4 = BGR HR-detail, 9 = YCC420 HR-detail,
                   10/11/12 = seg-mask family)
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
`s=1` → 1 (maximum LOD bias, density ratio near 7.8×). Past that,
`--adaptive-gamma <g>` in [0.1, 4] (default 1 = off) reshapes the shipped
detail bytes as `floor + ((v-floor)/(255-floor))^g`: `g > 1` pushes
mid-detail down for stronger edge focus (verified visually: `g=3` collapses
sampling onto the subject's edges), `g < 1` flattens toward uniform. It
applies encrypt-side to the final shipped bytes (after scoring, region
blend, and smoothing), so decrypt recomputes identical counts with no format
change; combine with `--adaptive-floor 0` for the maximum contrast. Past
even that, `--adaptive-cutoff <n>` in [0, 255] (default 0 = off) hard-zeros
tiles scoring below `n`: proportional allocation can never fully starve the
background (saturated edge tiles cap at 100% and the rest spills), but a
cutoff genuinely blacklists it — survivors absorb the budget up to capacity
and any unplaceable remainder is dropped (smaller container as a side
effect). The cutoff travels in the authenticated pad (adaptive mode 2 and
LOD-luma YCC) so decrypt agrees with no extra flags; cutoff wins over
`--lod-full` on overlap. Verified on IMG_3690 (`--ycc420 --ratio 0.25
--adaptive --tile-size 16 --strength 1 --floor 0 --gamma 3 --cutoff 100
--full-res`): background essentially black, samples pinned to the subject's
edges, container 1.35 MB. Start around 40–60 for a wider net; 100+ keeps
only the hottest edges and starves the decrypt warm-start in flat areas. The base is stored in
the authenticated pad so decryption regenerates the exact counts. The lod
byte array is stored immediately after the
header (one byte per tile, row-major, zero-padded to a whole container
pixel); measurements follow. Decryption regenerates identical indices from
`(key, lod, m, geometry, weight_base)` via `cs_compute_tile_sample_counts`
(deterministic largest-remainder split with capacity clamp/redistribute) plus
a per-tile Fisher–Yates draw — no decrypt-side flag needed (mode travels in
the header). Tile grids are capped at 65535 tiles (16-bit lod-count header
field): finer grids are refused at encrypt with the minimum viable
`--tile-size`, and count-mismatched containers are rejected at decrypt
instead of decoding to silent garbage. `--show-mask` (encrypt/roundtrip) writes `<output>.mask.png`
(source | binary mask | cyan overlay) and opens the same view in a window
after sampling. `--sample-split <prefix>` (encrypt/roundtrip) writes
`<prefix>_under.png` / `<prefix>_over.png`: source tiles from
under/over-sampled regions (split by mean tile count) at original positions,
rest black — no window, headless-safe. `--grain-tv <f>` (decrypt/roundtrip)
smooths only grainy tiles: post-solve Laplacian noise estimate per tile in
gray levels; tiles above `<f>` get a luma ROF smooth proportional to the
excess, clean tiles pass through untouched (any container, works with
`--tv 0`). **Resolution:** by default encrypt downsamples the input 2×
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

**Per-segment sampling:** `--seg-mask <f> --seg-rates <l>` (mode 10) samples
each segmentation class at its own rate instead of scoring tiles. `<f>` is an
8-bit label map (pixel value = class id, resized to the pipeline geometry
with nearest-neighbor) or a BGR color mask matched to the scene palette by
nearest color; `<l>` is a comma list of per-class rates in [0,1] in id order
(ids past the list score 0), e.g. `0.02,0.6,0.02,0.02,0.05,0.08,0.05,0.05`
(digits and commas only, so it survives PowerShell native-arg passing).
`--ratio` scales every rate globally. Rates are quantized to u8 and shipped
with a PNG of the label map in the body prefix, so decrypt regenerates the
identical per-class draws with no flags. `--seg-auto <k>` (2..255) replaces
the mask file with built-in k-means scene segmentation in Lab space (480px
working copy, classes sorted by size desc, speckle components reassigned to
the neighborhood mode, deterministic fixed seed) plus detail-driven rates
(mean Sobel energy per class: flat classes ~0.08, top detail 1.0, flat images
uniform 1.0); `--seg-rates` overrides the auto rates. The auto label map
ships identically, so decrypt needs no new flags. Exclusive with all other
sampling flags except `--ycc420`/`--ycc422`, which absorb it like the adaptive LOD
path: the rates drive the luma budget (modes 11/12) while chroma stays
uniform. `--show-mask`/`--sample-split` work unchanged.
`--seg-blend <w>` (0..128 pixels, default 0 = off) smoothly filters the
per-pixel target rates with a compact separable triangular kernel, radius
`w` in each axis at encryption geometry. Unlike a flat boosted band, the
density ramps across the frontier and returns continuously to the original
rates. A straight 20%/80% boundary meets near 50%; the high side gives up
some samples as the low side gains them. Constant interiors outside the
kernel support are unchanged (at corners support is a square, not a radial
distance cutoff). Nearby classes contribute locally, including at junctions.
The resulting 16-bit probabilities drive a deterministic keyed draw without
replacement of pixel positions; totals fluctuate slightly around the target.
With YCC this changes luma density; chroma remains uniformly sampled with
its count derived from the realized luma count.

New blended containers use a versioned prefix `[0,1,width,nclasses,...]`;
decrypt selects the stored algorithm automatically. Existing unblended and
flat-band containers retain their original sampling algorithms. Re-encrypt
to obtain the smooth ramp; decrypting an old file cannot change its samples.
Width 0 retains the original class-budget sampler. Tests:
`scripts\test_seg_blend.bat` (after `build_cpu.bat main`).

**Per-segment solve:** `--seg-solve <f>` (decrypt/roundtrip, BGR or YCC) runs
a single solver stage over all tiles: each tile is solved exactly once with
the parameters of its majority class from the label/color mask `<f>`
(majority vote over the tile rect; ties prefer the higher sampled class, or
the lower id without rate information). `--seg-params` overrides apply per
tile through the same vote. Neighboring tiles solved with different settings
meet through the standard cosine-feathered tile blend — there are no
per-segment solver passes and no cross-canvas seam reconciliation. The run
logs per-class tile counts. After the single solve, low-density segment
boundaries receive an edge-aware bilateral calm (13-pixel window, color
sigma 18, spatial sigma 4, fading out one overlap width inside the class;
restricted to the boundary bands' bounding region, ~7 s at full-res).
YCC chroma follows the tile's class parameters (half the coefficient and
roughly half the iterations, as in ordinary YCC reconstruction). The live
preview is the standard tile-progress window. Measured on the full-res
scene (96-grid, blend-6 container): single stage base-8/bird-12 + calm,
55.8 s, 35.40 dB; single stage global-12 + calm, 59.0 s, 35.83 dB —
against ~74 s / 36.01 dB for the former multi-pass solver. The remaining
gap is concentrated at segment boundaries; global iterations recover the
rest everywhere else.
`--seg-params "cls:knob=value,...;..."` overrides solver
settings per class (needs `--seg-solve`), e.g.
`1:iterations=8,coef=0.03,tv=0.05;5:iterations=6` — knobs are
`iterations|coef|tv|solver|fista-iters|reweights|basis|wscale` (`solver`
`fista|joint`, `basis` `dct|wavelet`); unlisted classes inherit the global
decrypt settings, and malformed entries fail before any solve runs.
`scripts/sweep_segsolve.ps1` grid-searches these knobs across `--ratio`
values (seg-solve cells plus an optional joint baseline per ratio),
recording PSNR/edge-PSNR/timings to CSV with a best-per-ratio summary —
run it on the thumbnail for speed, then validate the winner at full size.

### T-tetromino sampling (`--sampling tcluster|tmeans`)

`--sampling` selects the sampling atom (default `points` = uniform random;
exclusive with other sampling flags except `--ycc420`/`--ycc422`; decrypt
auto-detects):
- `tcluster` (mode 13): random subset of (4x4 cell, T-group) pairs from the
fixed bench partition; all 4 pixels of each selected pair stored as
individual point values (gaps between clusters). Same container/solver
machinery as points. Measured full-res crop: 22.07 dB at 0.5 (vs 20.63
points), 13.72 at 0.25 (vs 15.60) — helps at high ratios, hurts in gaps.
- `tmeans` (mode 14): one rounded mean per selected pair (max useful ratio
0.25 = full pair coverage; higher requests clamp with a note). Needs the
weight-aware FISTA scatter (true-average rows, Lipschitz-rescaled step);
standard/AVIR/joint paths supported, sr-direct/cs-sr/dip/GPU rejected
with a clear error. Measured full-res crop: 32.56 dB at 0.25 (vs 15.60
points), 30.06 at 0.2, 16.13 at 0.1; auto 0.2 reaches 23.92 dB (vs 23.58
points at 0.5 — wins with 2.5x fewer measurements).
- With `--ycc420`/`--ycc422` the T draw applies to luma (modes 15-18,
chroma stays uniform): tcluster 17.18/16.79 dB at 0.5 (vs 17.42 points),
tmeans 31.74/31.78 dB at 0.25 with `--iterations 40` (vs 28.50 points at
equal bytes). Sparse YCC needs iterations to converge — same physics as
uniform YCC at matched ratios (standard 0.25 also needs it40); defaults
undercook both equally.

### YCC 4:2:0 split sampling (`--ycc420`)

**Measurement-shape experiment:** See [measurement_shapes_results.md](measurement_shapes_results.md)
for the matched random-point / square-average / T-tetromino-average benchmark,
raw 8-bit payloads, full-resolution results, and experimental segmentation-guided
budgets. Build and run with `scripts\bench_measurement_shapes.bat`. This is a
standalone operator experiment, not a new encrypted-container mode.

On decrypt, `--y-confidence 0.3` enables an optional second CPU FISTA pass
for luma in YCC420/422 containers. It blurs the binary sample mask (5-pixel
sigma), compares local coverage with the tile's mean, and uses a soft
quadratic anchor only in locally under-sampled **unmeasured** pixels. The
anchor is a lightly smoothed first-pass Y reconstruction (1.2-pixel sigma);
the original measurements retain their data term. Strength is in [0,1];
0 (default) does not change the current decode. DCT basis only; chroma uses
the existing reconstruction. This is a decrypt option: existing containers
need no re-encryption. Try 0.2–0.5 and compare against 0 on the same image;
it adds one Y solve and can soften real texture where samples are scarce.

For tiles that contain both dense and sparse sampling, `--y-dual-tv 0.02`
detects mixed-density reconstruction tiles and performs **two independent Y
solves** from the same warm start: high-TV first as the base, then low-TV
(`--tv`, default 0). The two tile grids are composited separately, including
overlap feathering and AVIR upscaling. **After all tiles are solved**, Sobel
edges of the stable high-TV composite define where to copy low-TV detail;
the blurred sample mask gates edges with inadequate coverage. The mask has
soft boundaries and spans reconstruction-tile seams. Uniform-density tiles
use the normal single solve. Dual-TV uses AVIR for both candidates (other
photo upscalers are ignored when dual-TV is active).
Try `--y-dual-tv 0.01`, `0.02`, `0.05` against the same container; the
high-TV value must exceed `--tv`. This adds another full Y solve and may
smooth actual detail in sparse areas. It can also be combined with
`--y-confidence`, which refines the low-TV candidate before blending. On
IMG_3690's existing YCC container: baseline 27.63 dB (~9 s), confidence
0.3 alone 28.46 dB (~15 s), post-composite filled-segment mask + confidence
0.3 + dual-TV 0.02 reached 28.73 dB (~68 s) with binary color-segment selection.
High-TV backtracking makes dual solves costly. `--y-dual-tv-mask out.png`
writes the exact binary selection mask alongside the decode: black pixels
come from the high-TV composite, white pixels from the low-TV composite. It
copies whole BGR pixels (the two candidates use the same chroma solve), so
the mask exactly describes which candidate supplies every output pixel. The
edge/coverage decision can be
inspected directly. `--dual-tv-split <prefix>` (needs `--y-dual-tv`, YCC
only) writes `<prefix>_lowtv.png` / `<prefix>_hightv.png`: solved tiles taken
mostly from the low/high-TV solve (split by mean tile mask weight) at
original positions, rest black. `--dual-tv-cutouts <prefix>` writes
`<prefix>_low.png` / `<prefix>_high.png`: pixel-exact cutouts (low-candidate
pixels where the mask is white, blended pixels where black, else black). The map holds solid detail segments (strong high-TV edges,
closed morphologically and flood-filled from a padded outside border, gated
by sample coverage and a maximum component size). Hue-connected saturated
regions supported by those segments can expand into smooth parts of the same
object (e.g. a bird's crown), without selecting similarly smooth blue sky;
this is a color heuristic, not semantic object segmentation. This is a hard
cut at segment borders (only the earlier reconstruction-tile overlaps are
feathered). These
decrypt-side options require a YCC container; BGR modes ignore them.

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
128px) in the container. Per-tile sample counts remain nearly uniform, and
within each tile the budget splits 50/50: half drawn uniformly (so no region
can starve), half drawn with probability weighted by the log-compressed
thumbnail Laplacian (`log1p`, ~3x edge-vs-flat boost instead of the raw
100:1 spikes that used to pile whole tiles onto single features). This
concentrates measurements on edges while preserving coverage in smooth tiles.
The thumbnail is encrypted with AES-CTR using a
domain-separated counter and is also used as a first-wave solver warm start.
Use `--tile-size 32` for a fine HF grid; decrypt detects the mode from the
container. Note: the within-tile draw changed under the same mode numbers,
so mode 5/6/8 containers written by older binaries must be re-encrypted;
decrypting them with a new binary misplaces samples.

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
ImgReconstruct_backend encrypt  <input.png> <output.png> [--password <pw>] [--ratio R] [--periodic] [--adaptive] [--hf-focus] [--ycc420] [--tile-size N] [--adaptive-floor N] [--adaptive-strength S] [--lod-full N] [--seg-mask F --seg-rates L] [--sample-bits L[,C]] [--show-mask] [--full-res]
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
`--final-polish waifu2x|waifu2x-ncnn|realcugan` (unset by default) runs one
whole-image pass after everything else: the finished composite is upscaled
2x with the neural backend (configured by the `--waifu2x-*`/`--realcugan-*`
options above) and downsampled back with `INTER_AREA`, which smooths solver
and tiling artifacts that survive the per-tile stages. Applies to BGR and
ycc decrypts alike; a backend failure skips the pass with a warning, leaving
the image untouched.

`--basis dct|wavelet` selects the sparsifying basis (default `dct`,
whole-tile). `--dct-block N` (0 = whole-tile default, else N x N,
decrypt/roundtrip only, CPU only) switches DCT to independent per-block
transforms. Measured on a 1000x744 photo crop, full-res, ratio 0.5: whole-tile
20.7 dB, B=64 16.3 dB, B=8 14.0 dB — smaller blocks lose monotonically because
pixel-domain random sampling is maximally incoherent with the global DCT while
per-block coefficient groups starve (each block's coefficients see only local
samples). The wavelet basis (+12 dB over global DCT here) wins through
multiscale structure, not locality. `--wscale F` (default 2) is the threshold
ramp: coarsest x1, finest xF for wavelet, and a DC(x1)-to-peak(xF) frequency
ramp for DCT (1 = legacy uniform). The DCT ramp is the high-leverage prior:
ratio 0.5 full-res goes 20.7 dB (w=1) to 27.4 dB (w=2) to 34.3 dB (w=4),
past wavelet's 33.0 dB; w=8 saturates (34.5 dB). Auto (half-res + upscale)
pipelines are upscale-bound and gain ~0.1 dB.

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
in the default half-resolution pipeline. Each reconstructed
LR tile is first upscaled with AVIR; a reweighted-L1 FISTA solve then refines a
sparse HR DCT correction of the **luma channel only** against the **original
sampled LR luma** (BGR samples combined with the YCrCb weights) using a 2×2
box-downsample forward model, a quadratic anchor to the AVIR image, and the
optional `--tv` smoothness penalty. The anchor preserves HR detail that LR
measurements alone cannot determine; chroma comes straight from the AVIR
upscale, so one HR solve runs instead of three. It uses the existing tile
overlap and feather blending; small/unsampled tiles and wavelet-basis solves
use AVIR. `--ycc420`/`--ycc422` refine luma-only against the tile's LR luma
samples the same way (the YCrCb weights collapse to 1.0 on the luma
container); `--full-res` skips upscaling entirely.
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

`--red-denoiser scunet` swaps NLM for the blind SCUNet-real color model
(same RED plumbing, 256px runner tiles; weights gitignored at
`models/scunet/scunet_color_real_psnr.onnx` — fetch `scunet_color_real_psnr.onnx`
plus its `.onnx.data` from `Heliosoph/scunet-onnx`). `--red-denoiser restormer`
is the same interface awaiting weights at `models/restormer/restormer_color.onnx`;
`--red-model <f>` overrides either default. `--red-decay <f>` in [0,1]
schedules the blend per outer pass HQS-style (1.0 = legacy identical passes,
<1 decays geometrically: strong early, gentle late).

`--onsager` adds the LDAMP-style Onsager correction to the per-channel FISTA
tile solver: residuals are debiased by the previous residual scaled by the
soft-threshold support fraction (CPU only; the joint solver and the HIP kernel
are unchanged). Off by default; output is bit-identical with it off.

`--photo-upscaler sr-direct` selects the single-stage variant: the HR luma
is solved directly from the LR samples (`cs_sr_direct_luma`, no LR-solve
warm start inside the coefficients) and merged with the AVIR chroma.
Measured on the fixture it trails the two-stage path on real photos
(warm-start economics dominate: integrated direct 26.79 dB / 0.707 in
128 s vs 27.31 dB / 0.813 in 50 s for two-stage+RED), but it is available
for BGR containers when the LR stage is best skipped; ycc containers refine
luma-only the same way and `--red` is ignored there (single-stage solve has no RED
passes). `--sr-dict` takes precedence over both CS backends.

`--sr-init avir|cascade` (default `avir`, decrypt/roundtrip only) selects
the warm start for `cs-sr` and `sr-direct`: `cascade` runs the multiscale
coarse-to-fine binned-FISTA chain (`cs_sr_cascade_init`) instead of the
AVIR upscale to seed the HR solve. On the fixture, cascade + standard
budget reaches 36.76 dB in 0.016 s, beating zeros + 4× budget (36.05 dB
in 0.055 s) — the blocker was init quality, not iteration count
(HF-focus thumbnails already ship this idea at image scale).

`--sr-prior-model <f>` (decrypt/roundtrip, cs-sr backend only) folds a strong
learned SR prior into the CS-SR solve: a 2x ONNX photo-SR net (Real-ESRGAN /
SwinIR contract — RGB planar [0,1] NCHW in, 2x clean output) replaces the AVIR
anchor, so the net proposes the HR image and the LR measurements enforce it.
`--sr-pnp off|prox|admm` (default `off`) selects the denoiser step: `prox` is a
true proximal operator (NLM sigma-matched), `admm` is an outer PnP-ADMM pass
(`--sr-pnp-rho` in (0,8], default 0.1; `--sr-pnp-iters` in [1,50], default 8).
Absent a model the path is inactive and legacy CS-SR is bit-identical. A
missing/unreadable model fails fast (exit -3) rather than silently dropping the
prior. Recommended prior: `Xenova/swin2SR-lightweight-x2-64` (Apache-2.0, 2x,
~8 MB) — fetch with
`hf download Xenova/swin2SR-lightweight-x2-64 onnx/model.onnx --local-dir models\swin2sr`.

Reference commands (bird photo IMG_3690, seg-auto 8 + YCC 4:2:0, ratio 0.9;
`<pw>` is the passphrase; outputs land in the working directory):

```
ImgReconstruct_backend.exe encrypt ImgReconstruct_backend\IMG_3690.png IMG_3690segycc420r90.enc.png --password <pw> --ratio 0.9 --seg-auto 8 --ycc420 --no-preview
ImgReconstruct_backend.exe decrypt IMG_3690segycc420r90.enc.png IMG_3690segycc420r90.legacy.dec.png --password <pw> --photo-upscaler cs-sr --manual --tiles 64 --overlap 24 --iterations 10 --per-tile-coef --y-confidence 0.3 --no-preview
ImgReconstruct_backend.exe decrypt IMG_3690segycc420r90.enc.png IMG_3690segycc420r90.srprior.dec.png --password <pw> --photo-upscaler cs-sr --sr-prior-model models\swin2sr\onnx\model.onnx --manual --tiles 64 --overlap 24 --iterations 10 --per-tile-coef --y-confidence 0.3 --no-preview
ImgReconstruct_backend.exe decrypt IMG_3690segycc420r90.enc.png IMG_3690segycc420r90.srprior_admm.dec.png --password <pw> --photo-upscaler cs-sr --sr-prior-model models\swin2sr\onnx\model.onnx --sr-pnp admm --manual --tiles 64 --overlap 24 --iterations 10 --per-tile-coef --y-confidence 0.3 --no-preview
python Params-test\metrics.py ImgReconstruct_backend\IMG_3690.png IMG_3690segycc420r90.srprior.dec.png
python Params-test\metrics_lpips.py ImgReconstruct_backend\IMG_3690.png IMG_3690segycc420r90.srprior.dec.png
```

Measured on that container (SSIM / ΔE-p95 / ΔE-p99 / LPIPS / time): legacy
0.9077 / 5.144 / 10.038 / 0.1959 / 14 s; + SR prior 0.9113 / 5.162 / 10.733 /
0.1574 / 165 s; + SR prior + admm 0.8947 / 4.808 / 9.291 / 0.2817 / 230 s.
`--sr-prior-model` alone is the promoted config (best SSIM and LPIPS); PnP-ADMM
improves ΔE-p95/p99 but regresses SSIM/LPIPS — leave `--sr-pnp off`.

`--photo-upscaler dip` selects the untrained-prior variant: per tile, a
tiny Deep Decoder generator (nearest-upsample + channel mixes + ReLU,
final sigmoid; underparameterized vs the HR pixels) is fit by Adam from a
fixed seed against the tile's own LR luma samples (2x2-box forward model)
plus the usual AVIR anchor — no training data, no weights, no basis, CPU
only, luma-only with AVIR chroma (ycc containers refine luma-only the same way).
The generator is a residual head over the AVIR luma (zero-initialised output
weights, so it starts exactly at AVIR). Every fifth measured sample is held
out; Adam fits the rest, and the weights from the best held-out step are
kept (early stop after 100 steps without improvement). On a 1000x750 crop of
IMG_3690 at ratio 0.5, one tile: AVIR 23.21 dB, DIP 23.15 dB at about 140 s
(about 280x slower). The previous non-residual version scored 22.92 dB on the
same crop. The earlier 19.59 dB synthetic result did not transfer to photos
and should be disregarded. DIP is kept as an experiment, not a quality path.

`--hr-detail <n>` (encrypt/roundtrip only, prototype) stores `n` native-luma
edge anchors as a trailing AES-CTR payload (BGR mode 4, `--ycc420` mode 9;
uniform sampling + 8-bit samples only, decrypt auto-detects from the mode
byte). Anchors are top Sobel-magnitude luma pixels with per-cell top-2
bucketing for spatial spread (6 bytes each: u32 index + u8 value). Decrypt
applies them as exact post-composite luma replacements, reverting anchors
whose exact luma plus solver chroma falls outside BGR gamut instead of
injecting clip artifacts. With `--photo-upscaler cs-sr` the anchors also
enter the HR coefficient solve as exact data terms beside the LR samples
(point-sampling is orthonormal, so each anchor pulls its own pixel at full
weight with a `2*hr_w` Lipschitz fold), improving neighborhoods too, not
just anchor pixels. `--hr-weight <f>` in [0, 16] (default 1) scales that
term; 0 keeps post-hoc replacement only. Measured on IMG_3690, ycc420 ratio
0.5, 16384 anchors (+96 KB, +4.4%): converged `--tiles 12 --iterations 16`
gives 29.56 dB / edge 21.81 dB post-hoc-only vs 29.57 dB / edge 21.82 dB
in-solve, edge-MAE 15.397 → 15.386, admitted anchors 16354 → 16377 —
small but consistent across every metric (synthetic rig: anchor pixels
−33%, neighborhoods −2%). Starved settings (few giant tiles, few
iterations) saturate global metrics and show almost nothing; use `compare`'s
Edge-PSNR/Edge-MAE lines to see edge movement.

`--hr-red-radius <r>` (default 0 = uniform RED) scales the RED-lite blend per
pixel by `1-exp(-d²/2r²)` from the nearest in-tile anchor, so exact anchors
take (almost) no denoising pull. Measured negative: same setup plus
`--red 0.5` gives 29.63 dB / edge 21.85 dB uniform vs 29.61 dB / 21.84 dB at
`r=8` (edge-MAE 15.323 → 15.351), while `r=0.5` (anchor pixels only) is
bit-for-bit equivalent to uniform. Mechanism: anchor pixels are post-hoc
pinned exactly anyway, so their in-solve RED pull was harmless — suppressing
RED around them only deprives edge neighborhoods of denoising exactly where
the solve is noisiest. Kept as an opt-in experiment; uniform RED stands.

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
