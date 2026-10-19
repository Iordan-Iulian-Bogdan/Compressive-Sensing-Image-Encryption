# Improving Reconstruction & Super-Resolution Quality — Literature Survey & Recommendations

Scope: alternative algorithms and priors for the CS reconstruction solver and the SR/upscaling path
of `ImgReconstruct_backend`, with the constraints of the project (x64 **CPU**, limited memory via
per-tile blocking, optional **HIP** GPU, **ONNX Runtime** already vendored, `nunif`/waifu2x present).

Companion to `AGENTS.md` quality rule: **SSIM = sharpness, CIEDE2000 p95/p99 = color, PSNR = tiebreaker,
LPIPS confirms structure.**

---

## 0. TL;DR — where the gains actually are

Ranked by **expected quality gain per unit of effort** for *this* codebase:

1. **Fix the TV smoothing.** The solver currently uses a *smoothed* TV (curvature ≈ `8·tv`) rather than
   the exact TV prox/dual. This over-smooths edges — the single cheapest fidelity win. Use an exact
   TV prox (Chambolle) or a primal–dual solver (Chambolle–Pock) instead → sharper edges at equal cost.
2. **Better optimization, no training.** Add **adaptive restart** / **monotone FISTA** / **OGM** to the
   existing FISTA + IRLS. Purely algorithmic, CPU-cheap, often 1.5–3× faster convergence or higher
   quality at a fixed iteration budget.
3. **Turn the existing denoisers into *in-loop* priors.** The code already ships DNCNN / SCUNet /
   Restormer behind `--red-denoiser`, but only as a post/RED-lite step. Wire them as the **proximal
   operator inside the solver (PnP-ISTA / PnP-ADMM)** — this is where modern quality comes from.
4. **Change the prior class.** Move from `ℓ1(+TV)` toward **TGV** (kills TV staircase) and, most
   importantly, **denoiser-based priors** (PnP/RED) and **learned priors** (deep unfolding).
5. **Fix the SR path.** `AVIR`/`FSRCNN` are dated; move to a **Real-ESRGAN / SwinIR / HAT** model via
   ONNX (already available), or fold SR into the CS inverse problem (you already have `sr-direct`).
6. **Optimize the sampling mask** (blue-noise / learned). Because the encoder is a *subsampled orthonormal
   transform*, the mask — not the solver — bounds how much information is even recoverable.

**Critical framing (see §1):** this is **not** classical dense-random-matrix CS. It is *coefficient-domain
recovery* (inpainting of missing DCT coefficients). For this model, **prior + mask dominate**; marginal
solver tweaks (more iterations) yield little. Invest in the prior and the mask.

---

## 1. What the code does today (grounding)

Verified against source (`file:line`):

- **Encoder.** Each tile solves `min_x ½‖P·Φx − b‖² + λ‖x‖₁ (+ tv·TV(x))`, where measurements are
  **row-selected samples of an orthonormal transform** `A = P·IDCT` — i.e. a *mask/partial-observation*
  model, not a dense random Gaussian matrix (`helper_functions.cpp`; mode enum `crypto_utils.hpp:94-112`).
  Modes: RANDOM, PERIODIC, ADAPTIVE, YCC420/422, HR-DETAIL, HF-FOCUS, SEGMASK, TCLUSTER, TMEANS.
- **Solver.** Only solver is **FISTA** (Nesterov accelerated proximal gradient) with **reweighted L1
  (IRLS)** and an optional **group-L2,1 "joint"** variant (`cs_fista_core_single :528`, `..._joint :630`).
  Step size is exact for the orthonormal DCT basis (`L=2`), plus a **smoothed-TV curvature `≈8·tv`**;
  the CDF97 wavelet path uses backtracking. **No ADMM / no L-BFGS remain** (removed; the `--iterations`
  unit is a legacy L-BFGS scale, `helper_functions.hpp:157`).
- **"SOMP" is a convex surrogate.** Greedy SOMP is infeasible at ~170k DCT atoms/tile, so the joint
  solve is a **group-`ℓ2,1` FISTA over stacked [R|G|B] with one shared DCT support** — a convex relaxation
  of joint support selection (`:250-267`).
- **GPU.** HIP offloads *batched reweighted-L1 FISTA*, DCT only, with a CPU fallback (`cs_gpu.hip`).
- **SR / upscaling** (`--photo-upscaler`): `avir` (default), `fsrcnn` (OpenCV `dnn_superres`), `cs-sr`
  (own FISTA CS refinement), `sr-direct` (LR→HR FISTA), `dip`, + subprocess `waifu2x[-ncnn]`,
  `realcugan`; plus a **coupled dictionary** (`--sr-dict`, Yang TIP-2010) and **RED-lite**
  (`--red`, `--red-denoiser nlmeans|dncnn|scunet|restormer`). ONNX Runtime is already loaded on `_WIN32`.
- **Memory model.** "Limited memory" = **per-tile blocking** (peak working set ≈ one tile). No explicit
  `--memory` budget exists.

---

## 2. Diagnosis

1. **Prior is weak/over-smoothed.** `ℓ1` in DCT + *smoothed* TV → softened edges, staircase, loss of
   fine texture. This matches the observed failure mode (SSIM/sharpness lag).
2. **Information ceiling from the mask.** With `A = P·IDCT` the problem is *recovering unobserved
   transform coefficients*. Neither more iterations nor a fancier solver creates missing information;
   the **mask design** and the **image prior** set the ceiling.
3. **Per-tile independence** creates seam/blocking artifacts (mitigated by overlap blending but not
   removed) — a *global* consistency term helps.
4. **SR is decoupled and dated.** AVIR/FSRCNN are weak; `cs-sr`/`sr-direct` already gesture at the right
   idea (SR as part of the inverse problem) but without a strong learned prior.
5. **Learning is barely used.** The project has ONNX Runtime and strong denoisers available but does not
   yet use a *trained* prior in the reconstruction solve.

---

## 3. Reconstruction — alternative algorithms & priors (tiered)

### 3.1 Optimization upgrades (no training; cheapest, lowest risk)
- **Adaptive restart FISTA** [O'Donoghue & Candès 2015] and **gradient restart** [Fercoq & Qu 2019]:
  reset momentum when the objective/gradient turns non-monotone → avoids FISTA's oscillations, gives
  near-linear local convergence.
- **Parameter-free FISTA (Free-FISTA)** [Saban & Pesquet 2023/2024, SIAM J. Opt.]: adaptive restart +
  non-monotone backtracking estimates both `L` and `µ`; **also removes the need for the hard-coded step
  size**.
- **Monotone FISTA (MFISTA)** [Beck & Teboulle 2009] and **OGM** [Kim & Fessler 2016]: monotone variant
  for stability; OGM gives up to a **√2 fewer iterations** at equal cost. Good for your `--iterations`
  budget.
- **Over-relaxation / tuned momentum** in the existing loop: trivial to A/B against the current baseline.
- **Exact TV proximal** (Chambolle's dual projection) or a **primal–dual (Chambolle–Pock) / Douglas–Rachford
  solver**: removes the `8·tv` smoothing bias; markedly sharper edges. Highest-value cheap change.
- **IRLS tightening**: current `w = eps/(|x|+eps)`, 2 passes. More passes / better `eps` schedule
  (or `ℓp` with `p→1`) improves sparse recovery at modest cost.

### 3.2 Better priors **without training** (denoiser-based)
- **Plug-and-Play (PnP)** [Venkatakrishnan 2013]: replace the prox with an off-the-shelf denoiser in
  ISTA/ADMM → PnP-ISTA / PnP-ADMM. State-of-the-art with BM3D / NLM and with CNN denoisers.
- **RED — Regularization by Denoising** [Romano, Elad, Milanfar 2017; clarifications Reehorst & Schniter
  2018]: explicit regularizer `R(x) = ½ xᵀ(x − D(x))` usable in gradient descent. **You already have a
  RED-lite path** — upgrading it to proper PnP-ADMM is the win.
- **Denoising-based AMP (D-AMP)** [Metzler, Maleki, Baraniuk 2016] and **D-VAMP** [Schniter, Rangan,
  Fletcher 2016]: AMP with an **Onsager correction** turns the iterate perturbation into white Gaussian
  noise so any denoiser (BM3D) applies. **Important:** AMP diverges under structured/mildly ill-conditioned
  matrices — and your `A` is *structured* → use **VAMP** (robust to right-rotationally-invariant matrices),
  not plain AMP. You already expose `--onsager`; extending it toward **D-VAMP/LDAMP** is well-founded.
- **TGV** (Total Generalized Variation) [Bredies et al. 2010] and **NLTV / directional TV**: second-order
  generalization that removes the TV staircase while keeping edges.
- **Averaged / nonexpansive deep denoisers** [Nair & Chaudhury 2022] give PnP/RED **convergence guarantees**.

### 3.3 Learned reconstruction (moderate effort; ONNX-friendly)
- **Deep unfolding** (interpretable, usually lightweight, ONNX-exportable):
  - **ISTA-Net / ISTA-Net⁺** [Zhang & Ghanem, CVPR 2018]: learn the sparsifying transform + shrinkage;
    `ISTA-Net⁺` works in the **residual domain** (more compressible). Direct upgrade of your FISTA.
  - **FISTA-Net** and **ISTA-Net⁺⁺** (flexible, multi-ratio) [ICME 2021].
  - **AMP-Net / OAMP-Net** [Zhang & Kim; He et al. 2020]: unfolded AMP/OAMP with a denoiser module.
  - **CSNet / CSCA / CSFormer** (conv + transformer hybrids: `CSNet` ICIP 2020, `CSformer` ICLR 2023).
  - **TransCS / FusionOpt-Net** [2024: Transformer + FISTA momentum] — strong PSNR/SSIM and noise-robust.
  - **UFC-Net** (CVPR 2024): unrolled fixed-point continuous network — SOTA deep CS.
  - **LDAMP** [Metzler, Mousavi, Baraniuk, NeurIPS 2017]: learned D-AMP; magnitude faster than BM3D-AMP.
  - Practical: pick **ISTA-Net⁺** or **FISTA-Net** first — smallest, matches your exact `min ½‖Ax−b‖² + λ‖·‖₁`
    form, exportable to ONNX, CPU-viable at patch level; heavy ones (UFC-Net/CSformer) if GPU budget allows.
- **PnP/RED with CNN denoisers** (DnCNN, **DRUNet**, SCUNet, Restormer) *inside* the loop — you already
  have these denoisers available via `--red-denoiser`.
- **Trained per-mask**: if you also **learn the sampling mask** end-to-end (see §4), unfolded nets trained
  for *your* mask beat generic reconstruction (LDAMP shows mask-transfer heuristic).

### 3.4 Generative / diffusion priors (heaviest; use with care)
- **DDNM** (range-nullspace decomposition), **DPS**, **DiffPIR**, **PnP-diffusion** [Wang et al. 2025,
  arXiv:2509.09365], **IDM (Invertible Diffusion Models)** [TPAMI 2025] — very high quality but
  **hallucinate**. For an *encryption-validation* pipeline that must be faithful, treat these as
  perceptual refiners only, and score with LPIPS **plus** a fidelity guard (SSIM/ΔE must not regress).

---

## 4. Sampling / encoding — the biggest lever for this model

Because `A = P·IDCT` is *partial observation of an orthonormal basis*, recoverability is governed by the
mask and by the image's sparsity/structure:

- **Mask design.** Replace heuristic RANDOM/PERIODIC masks with **blue-noise / low-coherence masks** and
  masks optimized for the reconstruction basis. Incoherent bases (e.g. **noiselets** vs Haar wavelets,
  [arXiv:1606.04535]) improve conditioning without storing a dense matrix.
- **Learned sampling.** Jointly optimize mask+reconstructor end-to-end [differentiable cascade, e.g.
  arXiv:2203.14945]; **learned sampling beats random/engineered masks at high compression**.
- **Structured/orthogonal patterns.** Hadamard/DCT with randomized sign flips, differential/complementary
  measurement (SNR-robust) — mature in single-pixel imaging [Duarte et al. 2008].
- **Adaptive / LOD / HF-focus masks** (already present) are a good direction — refine via *measured*
  per-tile error rather than hand-tuned rules.
- **Onsager-aware normalization:** "pseudo-inverse gets *worse* with more samples" under noise; a
  **zero-mean inverse measurement** formulation fixes it [Noise Suppression in Compressive SPI, 2020].

---

## 5. Super-resolution

### 5.1 Fidelity-oriented (recommended for this pipeline)
- **Classic CNN**: EDSR, **RCAN** (channel attention), **HAN** — high PSNR/SSIM, deterministic, no hallucination.
- **Transformer**: **SwinIR** (window attention) → **HAT / DAT / DRCT / HAAT** [arXiv:2309.05239, 2411.18003].
  HAT-M currently tops classic SR benchmarks (e.g. ×2: HAT 38.63 vs SwinIR 38.42 PSNR on DIV2K).
- **Efficient / on-device**: the **AIM 2025 Efficient Perceptual SR study** [arXiv:2510.12765] shows
  Real-ESRGAN-beating quality at mobile compute.

### 5.2 Perceptual / generative
- **ESRGAN / Real-ESRGAN / BSRGAN**; diffusion **SR3 / StableSR / ResShift / LDM**-based.
  Survey: *Diffusion Models, Image Super-Resolution And Everything* [arXiv:2401.00736].
- Caveat: GAN/diffusion SR **invents texture** → visually pleasing but can *falsely* raise LPIPS while
  damaging fidelity in a validation context. Prefer deterministic SR unless the goal is aesthetics.

### 5.3 Joint CS + SR (most promising for this codebase)
You already have `--sr-direct` and `cs-sr`: SR **inside the inverse problem**. Extend it to a **learned
prior** (unfolded net or PnP-denoiser) so the same solve simultaneously enforces measurement consistency
and SR. This avoids the two-stage SR artifacts and is more memory-efficient than a separate SR model.
Reference-based SR (HR guide) is another option when a guide image exists.

### 5.4 Deep dive — folding SR into the inverse problem (grounded in `sr-direct` / `cs-sr`)

**What the code already does.** `cs_sr_upscale_tile_2x` (`helper_functions.cpp:2154`) and the single-stage
`cs_sr_direct_upscale_tile_2x` → `cs_sr_direct_luma` (`:2363`, `:2818`) solve for **HR luma coefficients `x`**
under the fused operator

```
y = P · D · Ψ · x + n        (Ψ = inverse DCT, D = 2x2-box average `cs_sr_downsample:1493`,
                              P = row-selection by the sample mask (rix,riy), b = LR luma samples)
```

with objective `min_x ½‖P D Ψ x − b‖² + λ‖x‖₁ + tv·TV(x) + w‖x − x_avır‖²`, solved by reweighted FISTA
(`cs_sr_core_single:2079`; gradient `cs_sr_fista_grad:1507`). So SR is already an **SR/deconvolution inverse
problem** whose prior is `ℓ1+TV` and whose warm start/anchor is the AVIR net (`:2207`, `:2385`). `sr-direct`
is the single-stage variant (no LR-solve warm start); `cs_sr_cascade_init:2930` is a LapSRN-style pyramid
warm start.

**Why folding beats two-stage (`LR-solve → quantize → HR-refine`).** It keeps float precision and
**re-enforces the measurements** (the solve is anchored to `b`), so the output cannot drift from the observed
samples. This matters most here because `P` is *your own irregular mask*, which no off-the-shelf SR net can
invert.

**Blueprint — USRNet** (*Deep Unfolding Network for Image Super-Resolution*, CVPR 2020, arXiv:2003.10422):
solves `min_x ½‖y − D x‖² + λ R(x)` with a **learned denoiser prior** and a **pluggable degradation `D`**
(blur + downsampler + scale). Your solver is a hand-designed USRNet with a weak prior; the upgrade path is
to keep the skeleton and strengthen the **prior** and the **degradation model**.

**Concrete upgrades (ranked):**

1. **Strengthen the anchor, keep the loop.** Replace the AVIR warm start/anchor (`cs_upscale_2x_avir` at
   `:2207`/`:2385`) with a strong SR net (**SwinIR / HAT / Real-ESRGAN** via the ONNX path already linked).
   The existing FISTA stage then becomes a **data-consistency refinement of a strong prior** (PnP/diffusion
   pattern: prior proposes, forward model enforces). One-location change, largest expected payoff.
2. **PnP prox instead of RED gradient.** The ONNX denoisers are wired for `cs-sr`
   (`cs_onnx_denoise_bgr`, `red_kind`, `:2298`) but applied as `x −= red·(x − D(x))`
   (`cs_sr_red_proximal:1655`) — a RED *gradient* step, not a true proximal operator. Move to **PnP-ADMM**
   (real prox per outer pass) — the USRNet-consistent formulation; texture/SSIM gains live here.
3. **Match or learn `D`.** `D` is a hard-coded 2x2 box; if the true formation differs, consistency is
   enforced on the wrong model. Match `D` to the real LR formation, or go **blind SR**:
   IKC (arXiv:1904.07396), KernelGAN (arXiv:1909.06581), DASR/BSRGAN.
4. **Guarantee data consistency at the end.** **Range–nullspace projection** (DDNM, ICLR 2023,
   arXiv:2212.00490): satisfy `P D Ψ x = b` exactly on the observed subspace; prior fills the nullspace.
5. **Use every observation as a data term.** HR-detail anchors are already an extra data term
   (`hr_idx/hr_val`, `:1546`) and an HF thumbnail is shipped — that is *multi-frame/burst SR* in disguise.
   Add LR tile + HR anchors + thumbnail as simultaneous data terms (classical robust MFSR [Farsiu 2004] /
   Deep Burst SR, arXiv:2101.10997); cheap — more terms in the existing gradient.
6. **Optional: unroll** into a small Learned-ISTA/LDAMP net trained end-to-end for *your* `P D Ψ`. Largest
   PSNR/SSIM jump, ONNX-friendly. Diffusion refiners are possible but **hallucinate** — guard fidelity.

**Caveats.** (a) Information ceiling `m < n_hr` — beyond captured information SR is hallucination; gate with
SSIM/ΔE-p95. (b) Kernel mismatch silently caps quality — verify `D` first. (c) Prefer deterministic SR /
contractive PnP denoisers over GAN/diffusion for validation. (d) Per-tile FISTA + one ONNX SR call per tile
fits the blocking memory model. Validate by A/B on `IMG_3690` with `Params-test/metrics.py` +
`metrics_lpips.py`, swept via `sweep.ps1`.

---

## 6. Concrete roadmap (ordered, with rationale)

| # | Change | Type | Where | Effort | Expected gain |
|---|--------|------|-------|--------|---------------|
| 1 | Exact TV prox / primal–dual (drop smoothed-TV bias) | solver | `helper_functions.cpp` FISTA core | Low | **Edges/SSIM** ↑, low risk |
| 2 | Adaptive restart + monotone + OGM variants | solver | `cs_fista_core_*` | Low | Faster conv. / quality at fixed iters |
| 3 | PnP: use DNCNN/SCUNet/Restormer as in-loop prox (PnP-ISTA / PnP-ADMM) | prior | RED-lite → solver loop | Med | **Large** (texture/SSIM) |
| 4 | D-VAMP with BM3D (extend `--onsager`; VAMP for structured A) | prior | new solver path | Med | Large at low ratios |
| 5 | TGV / NLTV as alternative regularizers | prior | FISTA penalty | Low–Med | Removes staircase |
| 6 | Optimize sampling mask (blue-noise → learned) | encoder | `CS_encryption.cpp` mask modes | Med | Raises information ceiling |
| 7 | Unfolded net (**ISTA-Net⁺ / FISTA-Net**) via ONNX, trained per mask | learned | new + ONNX | Med–High | **Largest** PSNR/SSIM |
| 8 | Replace AVIR/FSRCNN SR with **Real-ESRGAN or SwinIR/HAT** (ONNX) / AIM-2025 efficient | SR | `photo_upscaler.*` | Med | Sharper SR |
| 9 | Joint CS+SR with learned prior (extend `sr-direct`) | SR | `cs_sr_*` | High | Best SR consistency |
| 10 | Diffusion prior (DDNM/DPS/PnP-diffusion) — perceptual only; guard fidelity | prior/SR | new (GPU) | High | Aesthetics only |

Memory/CPU fit: #1–#6 are CPU-native and respect per-tile blocking. #7–#8 run patch- or tile-wise (ONNX,
already vendored). #10 needs GPU and hallucinates — exclude for encryption validation.

---

## 7. References (selected)

**Deep unfolding / learned CS**
- Zhang & Ghanem, *ISTA-Net*, CVPR 2018 — arXiv:1706.07929. ISTA-Net⁺ residual domain.
- ISTA-Net⁺⁺, ICME 2021; FISTA-Net; AMP-Net (Zhang & Kim); OAMP-Net (He et al., 2020).
- Metzler, Mousavi, Baraniuk, *Learned D-AMP (LDAMP)*, NeurIPS 2017.
- UFC-Net, CVPR 2024; CSformer, ICLR 2023; FusionOpt-Net (Transformer+FISTA), 2024 (PMC11435510).

**Denoiser-based CS**
- Metzler, Maleki, Baraniuk, *From Denoising to Compressed Sensing (D-AMP)*, IEEE IT 2016 — arXiv:1406.4175.
- Schniter, Rangan, Fletcher, *Denoising-based VAMP*, 2016 — arXiv:1611.01376 (VAMP robust to structured A).

**PnP / RED**
- Venkatakrishnan et al., *Plug-and-Play Priors*, 2013.
- Romano, Elad, Milanfar, *RED*, SIAM J. Imaging Sci. 2017; Reehorst & Schniter, *RED: Clarifications*, 2018.
- Kamilov et al., *PnP denoisers: regularization & optimization perspectives*, 2025 — arXiv:2509.03475.
- Nair & Chaudhury, *Averaged Deep Denoisers*, 2022 — arXiv:2207.07321 (convergence).
- Terris et al., *Equivariant PnP*, CVPR 2024 (stability/quality).

**Diffusion for inverse problems**
- Wang et al., *PnP Diffusion for CS with Data-Consistency Projection*, 2025 — arXiv:2509.09365.
- *Invertible Diffusion Models for CS (IDM)*, TPAMI 2025 — DOI 10.1109/TPAMI.2025.3538896 (~2.6 dB PSNR over SOTA CS nets).
- DDNM, DPS, DiffPIR (cited therein).

**Optimization**
- O'Donoghue & Candès, *Adaptive restart for accelerated gradient schemes*, FoCM 2015.
- Fercoq & Qu, *Adaptive restart under local quadratic growth*, IMA J. Numer. Anal. 2019 — arXiv:1709.02300.
- Saban & Pesquet, *Parameter-Free FISTA by Adaptive Restart and Backtracking*, SIAM J. Opt. 2024 — arXiv:2307.14323.
- Beck & Teboulle, *Monotone FISTA (MFISTA)*, 2009; Kim & Fessler, *OGM*, 2016 (√2 speedup).
- Giselsson & Boyd, *Monotonicity and Restart in Fast Gradient Methods*.

**Regularizers**
- Bredies, Kunisch, Pock, *Total Generalized Variation (TGV)*, SIAM J. Imaging Sci. 2010.
- Chambolle, *exact TV prox*; Chambolle & Pock, *primal–dual*, 2011.

**Sampling / single-pixel**
- Duarte et al., *Single-Pixel Imaging via Compressive Sampling*, IEEE SPM 2008.
- *Noiselets for compressed single-pixel imaging* — arXiv:1606.04535.
- *Learned compressive sampling* (differentiable cascade) — arXiv:2203.14945.
- *Noise Suppression in Compressive Single-Pixel Imaging*, 2020 (zero-mean inverse measurement).

**Super-resolution**
- Chen et al., *HAT: Hybrid Attention Transformer*, TPAMI — arXiv:2309.05239 (SOTA classic SR).
- HAAT — arXiv:2411.18003; SwinIR (ICCVW 2021); DAT (CVPR 2023); DRCT.
- *AIM 2025 Efficient Perceptual SR: Study & Benchmark* — arXiv:2510.12765.
- *Diffusion Models, Image Super-Resolution And Everything: A Survey* — arXiv:2401.00736.
- Yang et al., *Coupled dictionary training for SR*, IEEE TIP 2010 (already in code as `--sr-dict`).
- **USS/Inverse-problem SR:** Zhang et al., *USRNet — Deep Unfolding Network for SR*, CVPR 2020 — arXiv:2003.10422.
- Wang et al., *DDNM — Denoising Diffusion Null-space Model*, ICLR 2023 — arXiv:2212.00490 (range–nullspace data consistency).
- **Blind SR / kernels:** Gu et al., *IKC — Iterative Kernel Correction*, CVPR 2019 — arXiv:1904.07396;
  Bell-Kligler et al., *KernelGAN (Blind SR kernel estimation)*, ICCV 2019 — arXiv:1909.06581; DASR (CVPR 2021); BSRGAN.
- **Multi-frame / burst SR:** Farsiu et al., *Fast and Robust Multi-Frame SR*, IEEE TIP 2004;
  Bhat et al., *Deep Burst Super-Resolution*, 2021 — arXiv:2101.10997.
- Lai et al., *LapSRN (Laplacian pyramid SR)*, CVPR 2017 — arXiv:1704.03915 (basis for the `cs_sr_cascade_init` pyramid).
