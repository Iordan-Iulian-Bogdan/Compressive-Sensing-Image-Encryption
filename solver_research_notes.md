# Alternative Solvers for CS Reconstruction — Research Notes

Context: current solvers are OWL-QN (L-BFGS on L1), FISTA (reweighted L1),
FISTA-Joint (SOMP-structured) over DCT / CDF 9/7 bases, ~5-8 iterations per
tile, CPU-only. Baselines: 26.8 dB synthetic / 30.3 dB photo @ ratio 1.0,
~17 dB @ ratio 0.5. ONNX Runtime already vendored (clip_scorer).

## 1. Classical optimization solvers (drop-in, no training)

### ADMM (Alternating Direction Method of Multipliers)
- Splits L1/TV problems into subproblems with **closed-form solutions**
  (element-wise soft-thresholding + a linear solve) — no line search, no
  Lipschitz tuning.
- Reaches quality in fewer iterations than alternatives: one study measured
  24 dB at 8 ADMM iterations vs 20 for a competing method; hyperspectral
  study found ADMM the *most parameter-robust* of all tested solvers
  (PSNR varies only 0.84 dB across lambda sweep vs 3.31 for FISTA).
- Proven local linear convergence for isotropic TV compressed sensing
  (arXiv:2505.01240); equivalent to G-prox PDHG which scales to very large
  images.
- Fits this codebase: the existing `--tv` fusion path is already a splitting
  problem; ADMM formalizes it. Per-iteration cost is one DCT/wavelet
  synthesis + gradient + soft threshold.

### GAP (Generalized Alternating Projection) + TV
- From the GAP-TV paper (arXiv:1511.03890), at 50% CS ratio: **30.93 dB**
  (GAP-TV) / 32.01 dB (Acc-GAP-TV) vs 12.12 dB for TwIST+TV — a massive
  margin at ratios matching this project's tests.
- Cheapest per-iteration option: projection onto measurement constraint +
  shrinkage in transform domain. GAP-TV == ADMM-TV numerically in that
  paper but simpler to code.

### Greedy: CoSaMP / gOMP / Subspace Pursuit / BIHT
- Can beat FISTA on peak PSNR (CoSaMP up to 70.98 dB on hyperspectral) but
  **require the sparsity level kappa as input** and are fragile to it
  (PSNR swings 10-14 dB across kappa vs 0.84 for ADMM).
- BIHT / SAMP (sparsity-adaptive variants) are the robust picks.
- Project already has SOMP-structured joint solver — CoSaMP is the same
  family with backtracking (prunes wrong support), a natural extension.

### Primal-dual (PDHG / Chambolle-Pock)
- Same family as ADMM; strong for TV terms; scales to large images.
- Worth mentioning only if ADMM's linear solve becomes a bottleneck.

## 2. Deep unrolling (learned solvers, fixed iteration count)

- **LISTA** (Learned ISTA): unrolls ISTA into a feed-forward net; achieves
  *linear* convergence vs sublinear ISTA/FISTA (NeurIPS 2018).
- **ISTA-Net / ISTA-Net+** (CVPR 2018): nonlinear learned transforms +
  thresholds; beats optimization-based methods "by large margins".
- **ISTA-Net++** (TIP 2021): multi-ratio flexible; +0.95..1.74 dB over
  ISTA-Net+ on BSD68/Set11/Urban100/DIV2K. Current SOTA-ish baseline.
- **SALSA-Net, ADMM-CSNet, AMP-Net, CSformer**: same idea from ADMM/AMP
  origins.
- Deployment cost: a fixed stack of conv/matmul layers — fast on CPU.
- Key advantage for this project: iteration count and step sizes are
  *learned* per-ratio, removing the hand-tuned (coef, iters) grid search
  the repo already does with `--tune`.

## 3. Plug-and-Play priors (PnP / RED) — best effort/quality tradeoff

- Replace the regularizer's proximal step with an arbitrary **denoiser**:
  PnP-ADMM / PnP-GAP. Venkatakrishnan et al. 2013 introduced it; now standard.
- Quantitative: PnP(denoiser) matches/beats ISTA-Net+ on CS-MRI
  (34.56 vs 34.65 dB @ 10%, 44.24 vs 44.12 @ 50%) **without training a
  reconstruction network** (arXiv:2106.03668).
- PnP with an artifact-removal prior significantly beats generative-model CS
  methods in the same paper.
- Classic CPU version: **PnP-BM3D** — deblurring example 31.29 dB vs TV
  29.97 dB (+1.3 dB) with zero training (Purdue, Wang & Chan). BM3D/OpenCV
  `fastNlMeansDenoising` already available — the project even has a denoise
  post-pass flag.
- Suggested integration: keep the existing measurement-projection step,
  replace/augment the TV term with a denoiser call inside the same loop.

## 4. Recommendation ranking for this repo

| Priority | Solver | Why | Effort |
|---|---|---|---|
| 1 | **PnP-ADMM/GAP with BM3D or OpenCV NL-means** | +1-3 dB typical, no training, reuses existing loop + `fastNlMeansDenoising` already linked | Low |
| 2 | **GAP-TV** (replaces current solve loop) | 30.9 dB @ 50% in the paper vs ~17 dB here; simplest algorithm of all | Low |
| 3 | **ADMM-L1/TV** formalizing `--tv` path | Fewest iterations, most parameter-robust solver measured in comparative studies | Low-Med |
| 4 | **CoSaMP/SP** next to existing SOMP joint solver | Same family, backtracking fixes wrong support picks | Med |
| 5 | **ISTA-Net++ via ONNX** | +1-1.7 dB over prior SOTA, fixed CPU cost; ONNX Runtime already vendored | Med-High (training) |
| 6 | PnP with learned denoiser via ONNX | Matches deep CS nets inside the model-based loop | High |

## Key sources
- GAP-TV: arXiv:1511.03890 (PSNR tables at multiple CS ratios)
- CS solver comparative study: arXiv:2401.14762 (FISTA/ADMM/gOMP/BIHT/CoSaMP)
- ADMM linear convergence for TV-CS: arXiv:2505.01240
- ISTA-Net: arXiv:1706.07929 (CVPR 2018); ISTA-Net++: arXiv:2103.11554
- LISTA linear convergence: NeurIPS 2018 (cf8c9be2...)
- PnP/RED recovery analysis: arXiv:2106.03668 (PnP vs ISTA-Net+ tables)
- PnP-BM3D: Wang & Chan, "Parameter-free Plug-and-Play ADMM"
- CoSaMP: Needell & Tropp 2009
