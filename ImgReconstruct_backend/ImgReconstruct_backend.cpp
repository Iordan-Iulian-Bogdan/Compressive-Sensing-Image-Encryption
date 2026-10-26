#include "image_decryption.hpp"
#include "cs_wavelet.hpp"
#include "image_encryption.hpp"
#include "photo_upscaler.hpp"
#include "cs_dict.hpp"
#include "quality_utils.hpp"
#include "cs_gpu.h"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <iostream>

namespace {

void print_usage(const char* exe) {
    std::cout <<
        "Compressive-sensing image encryption\n"
        "\n"
        "Usage:\n"
        "  " << exe << " encrypt <input.png> <output.png> [options]\n"
        "  " << exe << " decrypt <input.png> <output.png> [options]\n"
        "  " << exe << " roundtrip <input.png> <output.png> [options]\n"
        "  " << exe << " compare <original.png> <decrypted.png> <out_prefix> [options]\n"
  "  " << exe << " train-dict <out.csd2> <img1.png> [img2.png ...] [atoms=N] [iters=N] [pairs=N]\n"
        "\n"
"Options:\n"
        "  --password <pw>       passphrase (else $CS_PASSWORD, else interactive prompt)\n"
        "  --ratio <f>           encryption sampling ratio in (0.001, 1.0]; default 1.0\n"
        "  --tiles <n>           decrypt tile count >= 1 (manual mode only,\n"
        "                        default 1: whole image is one tile)\n"
        "  --overlap <n>         decrypt tile overlap in [0, 96] (manual mode only;\n"
  "                        0 disables overlap: tiles solve independently and\n"
  "                        composite without feathering)\n"
        "  --fast                decrypt preset: tile overlap 8 and 12 FISTA steps per\n"
        "                        pass (explicit --overlap/--fista-iters win); about 2x\n"
        "                        faster for a small quality cost\n"
        "  --iterations <n>      decrypt solver iterations (manual mode only)\n"
        "  --threads <n>         decrypt worker threads (manual mode only)\n"
        "  --coef <f>            decrypt solver coefficient (>0; higher = sparser,\n"
        "                        no upper bound). Default: auto-derived from the\n"
        "                        container ratio, sampling mode and tile grid\n"
        "                        (Params-test fitted laws); explicit values win\n"
        "                        (and switch to manual mode).\n"
        "  --per-tile-coef       scale the L1 coefficient per tile from its\n"
        "                        sample count (starved tiles regularized more,\n"
        "                        rich tiles relaxed; tv stays global unless\n"
        "                        --per-tile-tv is also given)\n"
  "  --per-tile-tv         scale the TV weight per tile the same way\n"
  "                        (smooth/starved tiles smoothed more; watch\n"
  "                        for seams on flat gradients)\n"
  "  --grain-tv <f>        smooth only grainy tiles: post-solve Laplacian\n"
  "                        noise estimate per tile (gray levels); tiles above\n"
  "                        <f> get a luma ROF smooth proportional to the\n"
  "                        excess, clean tiles pass through (any container)\n"
  "  --y-confidence <f>   YCC luma confidence prior [0,1] on decrypt:\n"
  "                        smooth locally sparse unsampled areas (default 0;\n"
  "                        adds a CPU FISTA refinement pass)\n"
  "  --y-dual-tv <f>      YCC luma: solve again with this high TV weight,\n"
  "                        blend into sparse regions via a soft sample mask\n"
  "                        (0 = off; uses --tv for the low-TV solve)\n"
  "  --y-dual-tv-mask <p>  write the binary dual-TV selection mask to PNG\n"
  "                        (black = high-TV, white = low-TV composite;\n"
  "                        needs --y-dual-tv)\n"
  "  --dual-tv-split <p>  write <p>_lowtv.png / <p>_hightv.png: solved\n"
  "                        tiles taken mostly from the low/high-TV solve\n"
  "                        (split by mean tile mask weight) at original\n"
  "                        positions, rest black; needs --y-dual-tv\n"
  "                        (YCC containers only)\n"
  "  --dual-tv-cutouts <p> write <p>_low.png / <p>_high.png: pixel-exact\n"
  "                        cutouts (low-candidate pixels where the mask is\n"
  "                        white, blended pixels where black, else black);\n"
  "                        needs --y-dual-tv (YCC containers only)\n"
        "  --manual              force manual parameter mode (auto is the default and\n"
        "                        derives tiles/overlap/iterations/coef from the image)\n"
        "  --no-preview          disable the live decryption preview window\n"
        "  --photo-upscaler <b>  per-tile 2x upscaler backend: avir (default,\n"
        "                        vendored resampler), fsrcnn (neural 2x, needs weights),\n"
        "                        cs-sr (CS refinement: the\n"
        "                        HR tile is re-solved against its LR samples),\n"
        "                        sr-direct (single-stage HR solve directly\n"
        "                        from the LR samples, no LR-solve warm start;\n"
        "                        luma path in ycc containers),\n"
        "                        dip (untrained Deep Decoder prior fit to the\n"
        "                        tile's own samples, no weights; luma path\n"
        "                        in ycc containers),\n"
        "                        waifu2x (one batched\n"
        "                        nunif subprocess), waifu2x-ncnn (native\n"
        "                        waifu2x-ncnn-vulkan binary) or realcugan\n"
        "                        (native realcugan-ncnn-vulkan binary); the\n"
        "                        ncnn binaries run on Vulkan (AMD included)\n"
        "                        with no Python, and fall back to AVIR when\n"
        "                        unavailable)\n"
        "  --fsrcnn-model <f>    FSRCNN weights for --photo-upscaler fsrcnn\n"
        "                        (default FSRCNN_x2.pb, resolved against the\n"
        "                        working directory)\n"
        "  --waifu2x-cmd <s>     waifu2x command prefix (default\n"
        "                        \"python -m waifu2x.cli\" or $CS_WAIFU2X_CMD;\n"
        "                        e.g. \"py -m waifu2x.cli\" on Windows)\n"
        "  --waifu2x-args <s>    full-arg override appended after -i/-o\n"
        "                        (default composes --style photo --method\n"
        "                        <method> -n <noise> -g -1)\n"
        "  --waifu2x-method <m>  waifu2x denoising: scale (pure 2x upscale,\n"
        "                        default) or noise_scale (denoise + 2x)\n"
  "  --waifu2x-noise <n>   waifu2x denoise strength in [0, 3]\n"
  "                        (default 0; used with noise_scale)\n"
  "  --final-polish <b>    after everything else, upscale the final image\n"
  "                        2x with a neural backend and downsample back\n"
  "                        (artifact smoother): waifu2x, waifu2x-ncnn or\n"
  "                        realcugan (configured by the --waifu2x-*/--realcugan-*\n"
  "                        options above); unset by default; a backend\n"
  "                        failure skips the pass with a warning\n"
  "  --tv <f>              total-variation fusion weight for the solver\n"
  "                        (>0 enables TV, smoother edges, helps at high\n"
  "                        compression; 0 = off). Default: auto (0.01 for\n"
  "                        segment sampling below ratio 0.5, else 0).\n"
  "  --red <f>             RED-lite NLM strength for --photo-upscaler cs-sr\n"
  "                        in [0, 1] (0 = off): blends each outer FISTA pass\n"
  "                        toward its fastNlMeans-denoised self; stronger\n"
  "                        natural-image prior than TV, slower\n"
  "  --red-denoiser <d>    RED denoiser for cs-sr: nlmeans (default, luma\n"
  "                        fastNlMeans), dncnn (repo-vendored color DnCNN),\n"
  "                        scunet (blind real-camera color model) or\n"
  "                        restormer (blind color transformer); ONNX models\n"
  "                        run in-process via ONNX Runtime, BGR tile/pass\n"
  "  --dncnn-model <f>     DnCNN ONNX path (default models/dncnn/\n"
  "                        dncnn_color.onnx)\n"
  "  --red-model <f>       model file for the scunet/restormer RED denoiser\n"
  "                        (default models/scunet/scunet_color_real_psnr.onnx\n"
  "                        resp. models/restormer/restormer_color.onnx)\n"
        "  --red-decay <f>       HQS schedule on the RED blend per outer pass in\n"
        "                        [0, 1] (default 1 = every pass identical; <1\n"
        "                        decays geometrically: strong early, gentle late)\n"
        "  --hr-weight <f>       HR-anchor data-term weight for cs-sr in\n"
        "                        [0, 16] (default 1: each anchor pulls its HR\n"
        "                        pixel at full weight beside the LR samples;\n"
        "                        0 = post-hoc replacement only)\n"
        "  --hr-red-radius <f>   anchor-proximity RED radius in HR pixels in\n"
        "                        [0, 64] (default 0 = legacy uniform RED): the\n"
        "                        RED blend scales per pixel from 0 at anchor\n"
        "                        pixels to full beyond a few radii; needs\n"
        "                        --hr-detail anchors and --red > 0\n"
  "  --onsager             LDAMP-style Onsager correction in the per-channel\n"
  "                        FISTA tile solver (debiases residuals by the shrink\n"
  "                        support fraction; CPU only, joint solver unchanged)\n"
  "  --sr-dict <f>         coupled-dictionary SR model (CSD2) for decrypt/\n"
  "                        roundtrip: patch-OMP synthesis replaces the FISTA\n"
  "                        HR solve in the fused 2x path (any container mode);\n"
  "                        --red is ignored there; see train-dict\n"
  "  --sr-init <m>         CS SR warm start: avir (default, AVIR upscale of\n"
  "                        the solved tile) or cascade (multiscale binned-\n"
  "                        FISTA chain landing HR pixels); applies to cs-sr\n"
  "                        and sr-direct, decrypt/roundtrip only\n"
  "  --sr-prior-model <f>  ONNX 2x photo-SR net (Real-ESRGAN/SwinIR: RGB\n"
  "                        planar [0,1] NCHW, 2x clean output) whose upscale\n"
  "                        replaces the AVIR anchor in the cs-sr fused solve,\n"
  "                        so a strong prior proposes the HR image and the LR\n"
  "                        measurements enforce it (decrypt/roundtrip only)\n"
  "  --sr-pnp <m>          PnP step for the cs-sr denoiser: off (default,\n"
  "                        legacy RED blend), prox (denoiser as a true\n"
  "                        proximal operator, NLM sigma-matched) or admm\n"
  "                        (outer PnP-ADMM; denoiser = prox, measurement\n"
  "                        model enforces) (decrypt/roundtrip only)\n"
  "  --sr-pnp-rho <f>      ADMM penalty for --sr-pnp admm in (0, 8]\n"
  "                        (default 0.1)\n"
  "  --sr-pnp-iters <n>    ADMM outer passes for --sr-pnp admm in [1, 50]\n"
  "                        (default 8)\n"
  "  --solver <name>      tile solver: fista (default, reweighted-L1\n"
  "                        FISTA per channel), joint (SOMP-structured\n"
  "                        group-sparsity FISTA over R/G/B together +\n"
  "                        reweighting)\n"
  "  --fista-iters <n>    steps per reweight pass (0 = auto-map from\n"
  "                        --iterations as N*4 clamped to [16,40]; max 500)\n"
  "  --reweights <n>      outer reweight passes (1-5, default 2;\n"
  "                        1 = plain unweighted solve)\n"
  "  --basis <name>       sparsifying basis: dct (default, whole-tile;\n"
  "                        add --dct-block for block-DCT) or wavelet\n"
  "                        (multilevel CDF 9/7, sparser on natural images)\n"
  "  --dct-block <n>      DCT block size: 0 = whole-tile (default),\n"
  "                        else independent DCT per n x n block (try 8,\n"
  "                        16, 32; decrypt/roundtrip only, CPU only)\n"
  "  --pad-radix          decrypt/roundtrip only, CPU only: solve each DCT\n"
  "                        tile on a power-of-two canvas (pad -> transform\n"
  "                        -> crop) so cv::dct hits its fast radix path;\n"
  "                        ~1.2-1.5x faster, very slightly softer\n"
  "                        (container unchanged)\n"
  "  --wscale <f>         per-scale threshold ramp (in (0,16], default\n"
  "                        2: coarsest band x1, finest band x wscale, LL\n"
  "                        protected; 1 = uniform; DCT maps it to a\n"
  "                        DC-to-peak frequency ramp)\n"
        "  --device <name>      solve device: cpu (default) or gpu\n"
        "                        (HIP offload; dct basis, any tv; other\n"
        "                        combinations fall back to cpu per call)\n"
        "  --periodic            encrypt with periodic tile sampling: one random\n"
        "                        per-tile pattern repeated across the image\n"
        "                        (encrypt/roundtrip only; decrypt auto-detects\n"
        "                        the mode from the container)\n"
        "  --tile-size <n>       periodic tile size (default 64)\n"
        "  --sampling <name>     point atom: points (default, uniform random),\n"
        "                        tcluster (random T-tetromino 4-pixel clusters),\n"
        "                        tmeans (one mean per random T-tetromino; max\n"
        "                        useful ratio 0.25). Works with --ycc420/--ycc422\n"
        "                        (T luma + uniform chroma, modes 15-18);\n"
        "                        exclusive with other sampling flags; decrypt\n"
        "                        auto-detects.\n"
        "  --ycc420              encrypt with luma/chroma-split 4:2:0 sampling: Y\n"
  "                        at full resolution, each chroma plane sampled on\n"
  "                        its half-resolution grid (~1.5 bytes/px stored\n"
  "                        instead of 3; decrypt auto-detects from the\n"
  "                        container). Adaptive flags (--adaptive, --two-pass,\n"
  "                        --regions, --lod-smooth, --adaptive-strength/floor,\n"
  "                        --lod-full) are absorbed: they drive the luma budget\n"
  "                        while chroma stays uniform. --periodic is ignored\n"
  "                        under --ycc420.\n"
  "  --ycc422              encrypt with luma/chroma-split 4:2:2 sampling: Y\n"
  "                        at full resolution, each chroma plane sampled on\n"
  "                        a half-width, full-height grid (~2 bytes/px;\n"
  "                        sharper chroma edges than 4:2:0 at a larger\n"
  "                        container; decrypt auto-detects mode 7).\n"
  "                        Same absorbed flags as --ycc420; mutually\n"
  "                        exclusive with --ycc420; --hf-focus selects the\n"
  "                        mode-8 HF-weighted variant.\n"
        "  --hf-focus            encrypt a stored LF thumbnail plus HF-weighted\n"
        "                        samples (BGR mode 5; with --ycc420, mode 6;\n"
        "                        with --ycc422, mode 8)\n"
        "  --hr-detail <n>       encrypt K native-luma edge anchors as a\n"
        "                        trailing payload (BGR mode 4, ycc420 mode 9;\n"
        "                        uniform sampling + 8-bit samples only; decrypt\n"
        "                        applies them as exact luma replacements)\n"
  "  --sample-bits <L[,C]> bits for luma and optional chroma, each [1, 8]\n"
  "                        (one value applies to both; default 8; lower values shrink\n"
  "                        the container near-linearly at the cost of\n"
  "                        quantization noise; encrypt/roundtrip only;\n"
  "                        decrypt auto-detects from the header)\n"
        "  --adaptive            encrypt with LOD-based adaptive sampling: each\n"
        "                        tile gets an 8-bit detail score shipped in the\n"
        "                        container; sample budget splits by score so\n"
        "                        detailed tiles are over-sampled and flat tiles\n"
        "                        under-sampled (encrypt/roundtrip only; decrypt\n"
        "                        auto-detects from the container)\n"
  "  --adaptive-floor <n>  minimum LOD byte in adaptive mode (default 32;\n"
  "                        keeps a floor share for flat tiles)\n"
  "  --lod-full <n>       tiles scoring >= n detail are fully sampled\n"
  "                        regardless of --ratio (0 = off, default; 1-255\n"
  "                        LOD byte; implies adaptive scoring, top-detail\n"
  "                        tiles win when the budget cannot fit them all;\n"
  "                        encrypt/roundtrip only, decrypt auto-detects)\n"
  "  --adaptive-strength <s>  how aggressively detail tiles are over-sampled\n"
        "                        and flat tiles under-sampled, s in [0, 1]\n"
        "                        (default 0.5; 0 = uniform like random sampling,\n"
        "                        1 = maximum LOD bias)\n"
        "  --adaptive-gamma <g>  LOD contrast exponent in [0.1, 4] (default\n"
        "                        1 = off): reshapes shipped detail scores as\n"
        "                        floor + ((v-floor)/(255-floor))^g; >1 pushes\n"
        "                        mid-detail down for stronger edge focus\n"
        "                        past strength 1.0, <1 flattens toward uniform\n"
        "                        (adaptive paths only, decrypt auto-detects)\n"
        "  --adaptive-cutoff <n> hard LOD floor in [0, 255] (default 0 =\n"
        "                        off): tiles scoring below n get zero samples\n"
        "                        (background blackout); survivors absorb the\n"
        "                        budget up to capacity, unplaceable remainder\n"
        "                        is dropped (smaller container). Needs an\n"
        "                        adaptive path; decrypt auto-detects\n"
  "  --two-pass            adaptive scoring via a pilot uniform sample +\n"
  "                        cheap per-tile recon (residual scoring) instead\n"
  "                        of the Laplacian; implies adaptive, decrypt\n"
  "                        needs no new flags (encrypt/roundtrip only)\n"
  "  --pilot-ratio <f>     pilot fraction of each tile's pixels for\n"
  "                        --two-pass scoring in [0.01, 0.25]\n"
  "                        (default 0.05; min 16 samples/tile)\n"
  "  --spectral            adaptive scoring via per-tile DCT energy whitened\n"
  "                        by the global spectral decay (band-energy\n"
  "                        scoring) instead of the Laplacian; implies adaptive,\n"
  "                        decrypt needs no new flags (encrypt/roundtrip only)\n"
  "  --regions <json>      inline LLM region spec, e.g.\n"
  "                        '{\"regions\": [{\"box\": [325,165,605,445],\n"
  "                        \"detail\": 1.0}]}' (boxes in 0-1000, origin\n"
  "                        top-left; detail in [0,1]). Single quotes work\n"
  "                        too (handy in PowerShell, which mangles embedded\n"
  "                        double quotes). Tiles overlapped by boxes gain\n"
  "                        sample share; implies adaptive, decrypt needs\n"
  "                        no new flags\n"
  "  --region-blend <f>    LLM-vs-signal mix for --regions in [0, 1]\n"
  "                        (default 0.5; 1 = LLM only, 0 = signal only)\n"
  "  --lod-smooth <f>      Gaussian blur (in tiles) of the LOD density\n"
  "                        field before budget allocation in [0, 8]\n"
  "                        (default 0 = off; 1-2 removes the abrupt\n"
  "                        per-tile density rectangles)\n"
  "  --seg-mask <f>        encrypt with per-segment sampling: <f> is an 8-bit\n"
  "                        label map (pixel value = class id) or a BGR color\n"
  "                        mask matched to the scene palette; --seg-rates\n"
  "                        gives the per-class sampling rate in [0,1]\n"
  "                        (class order = id order). Each segment is sampled\n"
  "                        at its own rate; decrypt auto-detects (mode 10).\n"
  "                        Exclusive with all other sampling flags except\n"
  "                        --ycc420/--ycc422 (seg-luma split, modes 11/12).\n"
  "  --seg-rates <l>       comma list, e.g. 0.02,0.6,0.02,0.02,0.05,0.08,0.05,0.05\n"
  "                        (digits and commas only, PowerShell-safe).\n"
  "                        --ratio scales every rate globally. With --ycc420\n"
  "                        or --ycc422 the rates drive the luma budget while\n"
  "                        chroma stays uniform (modes 11/12).\n"
  "  --seg-auto <k>        encrypt/roundtrip only: built-in k-means scene\n"
  "                        segmentation into <k> classes (2..255) in Lab space\n"
  "                        with detail-driven per-class rates (flat classes get\n"
  "                        ~0.08, top detail 1.0; --seg-rates overrides). No\n"
  "                        mask file needed; the label map ships in the\n"
  "                        container, so decrypt needs no new flags. Decrypt\n"
  "                        auto-scales each class's solver coef by its own\n"
  "                        rate (per-segment parameters). Works with\n"
  "                        --ycc420/--ycc422 and --seg-blend.\n"
  "  --seg-edge            encrypt/roundtrip only, needs --seg-auto <= 127:\n"
  "                        within-class edge refinement: the sharpest half of\n"
  "                        each class's 16x16 blocks is sampled at 1.5x the\n"
  "                        class rate, the rest lowered (same class budget).\n"
  "  --seg-blend <w>       encrypt/roundtrip only, needs --seg-mask or --seg-auto: sample\n"
  "                        a <w>-pixel band (0..128, 0 = off) on each\n"
  "                        side of every segment frontier with a smooth local\n"
  "                        density ramp (compact triangular filter). Keeps\n"
  "                        distant interiors at their own rate; width ships so\n"
  "                        decrypt regenerates the identical draw.\n"
  "  --seg-solve <f>       decrypt/roundtrip only: single solver stage where\n"
  "                        each tile uses the parameters of its majority\n"
  "                        segment from mask <f> (BGR/YCC). No per-segment\n"
  "                        passes; tiles meet via the standard tile blend.\n"
  "                        See --seg-params for the per-class settings.\n"
  "  --seg-params <s>      decrypt/roundtrip only, needs --seg-solve: per-class\n"
  "                        solver overrides as cls:knob=value,...;... e.g.\n"
  "                        1:iterations=8,coef=0.03,tv=0.05;5:iterations=6\n"
  "                        knobs: iterations|coef|tv|solver|fista-iters|\n"
  "                        reweights|basis|wscale (solver fista|joint,\n"
  "                        basis dct|wavelet). Unlisted classes inherit the\n"
  "                        global decrypt settings.\n"
        "  --show-mask           after encrypt/roundtrip sampling, write\n"
        "                        <output>.mask.png and open a preview window\n"
        "                        (source | binary mask | cyan overlay)\n"
        "  --sample-split <p>   after encrypt/roundtrip sampling, write\n"
        "                        <p>_under.png and <p>_over.png: source tiles\n"
        "                        from under/over-sampled regions (split by mean\n"
        "                        tile count) at original positions, rest black\n"
        "  --full-res            keep native resolution end-to-end (skip the\n"
        "                        default encrypt 2x downscale and decrypt\n"
        "                        2x upscale); use on encrypt AND decrypt/roundtrip\n"
        "\n"
        "  roundtrip encrypts the input and then decrypts the in-memory result\n"
        "  compare scores decryption quality (PSNR/SSIM/MAE/max) and writes\n"
        "    <out_prefix>_sidebyside.png  original | decrypted | JET error map\n"
        "    <out_prefix>_error.png       amplified absolute-error map\n"
        "\n"
        "Examples:\n"
        "  " << exe << " encrypt photo.png photo.enc.png\n"
        "  " << exe << " decrypt photo.enc.png photo.dec.png --password \"my secret\"\n"
        "  " << exe << " roundtrip photo.png photo.dec.png --password \"my secret\"\n"
        "  " << exe << " encrypt photo.png photo.enc.png --ratio 0.5 --periodic --tile-size 64\n"
        "  " << exe << " roundtrip photo.png photo.dec.png --password \"my secret\" --ratio 0.5 --adaptive\n"
        "  " << exe << " encrypt photo.png photo.enc.png --ratio 0.5 --adaptive --show-mask\n"
        "  " << exe << " roundtrip photo.png photo.dec.png --password \"my secret\" --ratio 0.5 --adaptive --full-res\n"
        "  " << exe << " compare photo.png photo.dec.png report --amp 4\n";
}

int run_compare(const std::string& original_path, const std::string& decoded_path,
                const std::string& out_prefix, double amplification)
{
    // full-size photos can exhaust GPU buffers in OpenCL UMat paths
    cv::ocl::setUseOpenCL(false);
    cv::Mat original = cv::imread(original_path, cv::IMREAD_COLOR);
    if (original.empty()) {
        std::cerr << "Error: cannot load original '" << original_path << "'" << std::endl;
        return 1;
    }
    cv::Mat decoded = cv::imread(decoded_path, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        std::cerr << "Error: cannot load decrypted '" << decoded_path << "'" << std::endl;
        return 1;
    }
    if (decoded.size() != original.size()) {
        cv::resize(decoded, decoded, original.size());
    }

    const double p = cs_quality::psnr(original, decoded);
    const double s = cs_quality::ssim(original, decoded);
    const double mae = cs_quality::mean_abs_error(original, decoded);
    const double mx = cs_quality::max_abs_error(original, decoded);
    double edge_p = -1.0, edge_mae = -1.0, edge_share = 0.0;
    cs_quality::edge_scores(original, decoded, edge_p, edge_mae, edge_share);

    std::printf("PSNR: %.2f dB\n", p);
    if (s >= 0.0) std::printf("SSIM: %.4f\n", s);
    else          std::printf("SSIM: n/a (OpenCV quality module not available)\n");
    std::printf("MAE:  %.3f\n", mae);
    std::printf("MAX:  %.1f\n", mx);
    if (edge_p >= 0.0)
        std::printf("Edge-PSNR (top-10%% gradient, share %.3f): %.2f dB\n", edge_share, edge_p);
    if (edge_mae >= 0.0)
        std::printf("Edge-MAE: %.3f\n", edge_mae);

    const cv::Mat err = cs_quality::error_map(original, decoded, amplification);
    const cv::Mat strip = cs_quality::side_by_side(original, decoded, amplification);
    if (err.empty() || strip.empty()) {
        std::cerr << "Error: failed to build error visualization" << std::endl;
        return 1;
    }
    const std::string err_path = out_prefix + "_error.png";
    const std::string strip_path = out_prefix + "_sidebyside.png";
    if (!cv::imwrite(err_path, err)) {
        std::cerr << "Error: cannot write '" << err_path << "'" << std::endl;
        return 1;
    }
    if (!cv::imwrite(strip_path, strip)) {
        std::cerr << "Error: cannot write '" << strip_path << "'" << std::endl;
        return 1;
    }
    std::printf("Wrote %s\n", err_path.c_str());
    std::printf("Wrote %s\n", strip_path.c_str());
    return 0;
}

std::string read_password() {
    std::string env;
    const char* env_pw = std::getenv("CS_ENCRYPTION_PASSWORD");
    if (env_pw && env_pw[0]) {
        std::cout << "Using passphrase from CS_ENCRYPTION_PASSWORD" << std::endl;
        return env_pw;
    }
    std::cout << "Enter passphrase: " << std::flush;
    std::string pw;
    std::getline(std::cin, pw);
    return pw;
}

bool parse_int(const char* s, int& out) {
    try { size_t pos; long v = std::stol(s, &pos); if (pos != std::strlen(s)) return false; out = (int)v; return true; }
    catch (...) { return false; }
}

bool parse_float(const char* s, float& out) {
    try { size_t pos; float v = std::stof(s, &pos); if (pos != std::strlen(s)) return false; out = v; return true; }
    catch (...) { return false; }
}

// Value helpers for the option table in main(). They return false when the
// value does not parse or fails `bad`, printing `msg` only when one is given:
// flags without their own message fail silently, as they always have.
bool set_int(const char* v, int& out, bool (*bad)(int), const char* msg) {
    if (!parse_int(v, out)) return false;
    if (bad && bad(out)) { std::cerr << msg << std::endl; return false; }
    return true;
}

bool set_float(const char* v, float& out, bool (*bad)(float), const char* msg) {
    if (!parse_float(v, out)) return false;
    if (bad && bad(out)) { std::cerr << msg << std::endl; return false; }
    return true;
}

// Whole-token integer via std::stoi; any failure prints `msg`.
bool set_int_strict(const char* v, int& out, const char* msg) {
    try {
        size_t pos = 0;
        out = std::stoi(v, &pos);
        if (pos != std::string(v).size()) throw std::runtime_error("trailing characters");
    }
    catch (...) {
        std::cerr << msg << std::endl;
        return false;
    }
    return true;
}

// Decimal in [0, 1] for the luma-confidence flags. /fp:fast may fold
// isfinite(NaN), so nan/inf spellings are rejected before conversion.
bool set_unit_decimal(const char* v, float& out, const char* msg) {
    bool decimal = *v != '\0';
    bool digit = false;
    for (const char* p = v; *p; ++p) {
        if (*p >= '0' && *p <= '9') digit = true;
        else if (*p != '.' && *p != '+' && *p != '-' && *p != 'e' && *p != 'E') decimal = false;
    }
    if (!decimal || !digit || !parse_float(v, out) || out < 0.0f || out > 1.0f) {
        std::cerr << msg << std::endl;
        return false;
    }
    return true;
}

// Exact-match choice. fold_case lowercases the value first and stores the
// folded spelling (used by --sr-init and --sr-pnp).
bool set_choice(const char* v, std::string& out, std::initializer_list<const char*> choices,
                const char* msg, bool fold_case = false) {
    std::string s = v;
    if (fold_case) for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    for (const char* choice : choices) {
        if (s == choice) { out = s; return true; }
    }
    std::cerr << msg << std::endl;
    return false;
}

} // namespace

#ifdef CS_GPU_TILES
// GPU build: the GPU stages are the default. CS_GPU_TILES (LR tile solve, batched cs-sr tile
// stage, compositing), CS_GPU_ADMM (PnP-ADMM) and CS_GPU_ONNX (DirectML prior/denoiser) default
// to on; set one to 0 to run that stage on the CPU. Any other explicit value is kept as given.
static void cs_gpu_env_defaults()
{
    for (const char* name : { "CS_GPU_TILES", "CS_GPU_ADMM", "CS_GPU_ONNX" }) {
        const char* v = std::getenv(name);
        if (!v || !*v) _putenv_s(name, "1");
        else if (v[0] == '0' && v[1] == '\0' && std::strcmp(name, "CS_GPU_TILES") == 0)
            _putenv_s(name, "");   // CS_GPU_TILES is tested for presence, so "0" must unset it
    }
}
#endif

int main(int argc, char* argv[])
{
#ifdef CS_GPU_TILES
    cs_gpu_env_defaults();
#endif
    if (argc < 2) {
        print_usage(argv[0]);
        return 64; // usage error
    }

    const std::string mode = argv[1];
    if (mode != "encrypt" && mode != "decrypt" && mode != "roundtrip" && mode != "compare" && mode != "train-dict") {
        std::cerr << "Error: unknown mode '" << mode << "'" << std::endl;
        print_usage(argv[0]);
        return 64;
    }

    if (mode == "train-dict") {
        // train-dict <out.csd2> <img1.png> [img2.png ...] [atoms=N] [iters=N] [pairs=N] [plr=8]
        if (argc < 4) {
            std::cerr << "Error: train-dict needs <out.csd2> <img1.png> [img2.png ...]" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
        const std::string out_path = argv[2];
        int atoms = 256, iters = 10, max_pairs = 30000, plr = 8;
        std::vector<cv::Mat> images;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("atoms=", 0) == 0) { atoms = std::atoi(a.c_str() + 6); continue; }
            if (a.rfind("iters=", 0) == 0) { iters = std::atoi(a.c_str() + 6); continue; }
            if (a.rfind("pairs=", 0) == 0) { max_pairs = std::atoi(a.c_str() + 6); continue; }
            if (a.rfind("plr=", 0) == 0) { plr = std::atoi(a.c_str() + 4); continue; }
            cv::Mat img = cv::imread(a, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Error: cannot load training image '" << a << "'" << std::endl;
                return 1;
            }
            std::cout << "using " << a << " (" << img.cols << "x" << img.rows << ")" << std::endl;
            images.push_back(img);
        }
        if (images.empty()) {
            std::cerr << "Error: train-dict needs at least one input image" << std::endl;
            return 64;
        }
        if (atoms <= 0 || atoms > 4096 || iters <= 0 || max_pairs <= 0 || plr != 8) {
            std::cerr << "Error: bad train-dict params (atoms 1..4096, iters >= 1, pairs >= 1, plr 8)" << std::endl;
            return 64;
        }
        std::cout << "training coupled dictionary: atoms=" << atoms << " iters=" << iters
            << " pairs=" << max_pairs << " plr=" << plr << std::endl;
        cs_coupled_dict dict;
        if (!cs_train_coupled_dictionary(images, atoms, iters, max_pairs, plr, dict)) {
            std::cerr << "Error: coupled training failed (not enough patch data?)" << std::endl;
            return 1;
        }
        if (!cs_save_coupled_dictionary(out_path, dict)) {
            std::cerr << "Error: cannot write '" << out_path << "'" << std::endl;
            return 1;
        }
        std::cout << "saved " << dict.atoms << " atoms (lr " << dict.plr << "x" << dict.plr
            << " -> hr " << dict.phr << "x" << dict.phr << ") to " << out_path << std::endl;
        return 0;
    }

    if (mode == "compare") {
        if (argc < 5) {
            std::cerr << "Error: compare needs <original.png> <decrypted.png> <out_prefix>" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
        double amp = 0.0;
        for (int i = 5; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--amp" && i + 1 < argc) {
                amp = std::atof(argv[++i]);
            }
            else {
                std::cerr << "Error: unknown compare option '" << a << "'" << std::endl;
                print_usage(argv[0]);
                return 64;
            }
        }
        return run_compare(argv[2], argv[3], argv[4], amp);
    }

    if (argc < 4) {
        std::cerr << "Error: missing input/output paths" << std::endl;
        print_usage(argv[0]);
        return 64;
    }

    const std::string input = argv[2];
    const std::string output = argv[3];

    std::string password;
    bool have_password = false;
    float ratio = 1.0f;
    int tiles = 1, overlap = 24, iterations = 5, threads = 8;
    float coef = -1.0f;
    bool per_tile_coef = false;
    bool per_tile_tv = false;
    float grain_tv = 0.0f;
    bool use_onsager = false;
    float tv_lambda = -1.0f;
    bool manual = false;
    bool show_preview = true;
    CsPhotoUpscalerOptions photo_up;
    bool periodic = false;
    int tile_size = 64;
    bool adaptive = false;
    bool ycc420 = false;
    bool ycc422 = false;
    bool hf_focus = false;
    int hr_detail = 0;
    int adaptive_floor = 32;    float adaptive_strength = 0.5f;
    float adaptive_gamma = 1.0f;
    int adaptive_cutoff = 0;
    int lod_full = 0;
    bool two_pass = false;
    bool spectral = false;
    float pilot_ratio = 0.05f;
    std::string regions_json;
    float region_blend = 0.5f;
    float lod_smooth = 0.0f;
    std::string seg_mask;
    std::string seg_rates;
    int seg_blend = 0;
    std::string sampling = "points";
    int seg_auto = 0;
    bool seg_edge = false;
    std::string seg_solve;
    std::string seg_params;
    bool show_mask = false;
    std::string sample_split;
    bool full_res = false;
    int sample_bits = 8;
    int chroma_sample_bits = 8;
    std::string solver_name = "fista";
    int solver = CS_SOLVER_FISTA;
    int fista_iters = 0;
    bool fast = false, overlap_set = false, fista_iters_set = false;
    int reweights = 2;
    std::string basis_name = "dct";
    int basis = CS_BASIS_DCT;
    float wscale = 2.0f;
    int dct_block = 0;

    // One row per flag. `hint` names the value in the missing-value error
    // (null for switches); `manual` switches decrypt to manual parameters
    // when the flag parses; `apply` returns false on a bad value (exit 64).
    struct Opt {
        const char* name;
        const char* hint;
        bool manual;
        std::function<bool(const char*)> apply;
    };
    const std::vector<Opt> opts = {
        {"--password", "passphrase text", false, [&](const char* v) { password = v; have_password = true; return true; }},
        {"--ratio", "floating point", false, [&](const char* v) { return set_float(v, ratio, nullptr, nullptr); }},
        {"--tiles", "integer", true, [&](const char* v) { return set_int(v, tiles, nullptr, nullptr); }},
        {"--overlap", "integer", true, [&](const char* v) { overlap_set = true; return set_int(v, overlap, nullptr, nullptr); }},
        {"--iterations", "integer", true, [&](const char* v) { return set_int(v, iterations, nullptr, nullptr); }},
        {"--threads", "integer", true, [&](const char* v) { return set_int(v, threads, nullptr, nullptr); }},
        {"--coef", "floating point", true, [&](const char* v) { return set_float(v, coef, nullptr, nullptr); }},
        // decrypt-side switches: per-tile counts travel in the container, so
        // none of these may flip decrypt into manual parameter mode
        {"--per-tile-coef", nullptr, false, [&](const char*) { per_tile_coef = true; return true; }},
        {"--per-tile-tv", nullptr, false, [&](const char*) { per_tile_tv = true; return true; }},
        {"--grain-tv", "floating point in (0, 64]", false, [&](const char* v) {
            return set_float(v, grain_tv, [](float x) { return x <= 0.0f || x > 64.0f; },
                             "Error: --grain-tv must be in (0, 64]");
        }},
        {"--y-confidence", "strength in [0,1]", false, [&](const char* v) {
            return set_unit_decimal(v, photo_up.y_confidence, "Error: --y-confidence must be in [0,1]");
        }},
        {"--y-dual-tv", "TV weight in (0,1]", false, [&](const char* v) {
            return set_unit_decimal(v, photo_up.y_dual_tv, "Error: --y-dual-tv must be in [0,1]");
        }},
        {"--y-dual-tv-mask", "output PNG path", false, [&](const char* v) { photo_up.y_dual_tv_mask = v; return true; }},
        {"--dual-tv-split", "output prefix", false, [&](const char* v) { photo_up.dual_tv_split = v; return true; }},
        {"--dual-tv-cutouts", "output prefix", false, [&](const char* v) { photo_up.dual_tv_cutouts = v; return true; }},
        {"--manual", nullptr, true, [&](const char*) { return true; }},
        {"--fast", nullptr, false, [&](const char*) { fast = true; return true; }},
        {"--no-preview", nullptr, false, [&](const char*) { show_preview = false; return true; }},
        {"--photo-upscaler", "avir, cs-sr, sr-direct, dip, fsrcnn, waifu2x, waifu2x-ncnn or realcugan", false, [&](const char* v) {
            return set_choice(v, photo_up.backend,
                {"avir", "cs-sr", "cs", "sr-direct", "dip", "fsrcnn", "waifu2x", "waifu2x-ncnn", "realcugan"},
                "Error: --photo-upscaler must be avir, cs-sr, sr-direct, dip, fsrcnn, waifu2x, waifu2x-ncnn or realcugan");
        }},
        {"--fsrcnn-model", "FSRCNN .pb path", false, [&](const char* v) { photo_up.fsrcnn_model = v; return true; }},
        {"--waifu2x-cmd", "command", false, [&](const char* v) { photo_up.waifu2x_cmd = v; return true; }},
        {"--waifu2x-args", "arguments", false, [&](const char* v) { photo_up.waifu2x_args = v; return true; }},
        {"--waifu2x-method", "scale or noise_scale", false, [&](const char* v) {
            return set_choice(v, photo_up.waifu2x_method, {"scale", "noise_scale"},
                              "Error: --waifu2x-method must be scale or noise_scale");
        }},
        {"--waifu2x-noise", "integer in [0,3]", false, [&](const char* v) {
            return set_int(v, photo_up.waifu2x_noise, [](int x) { return x < 0 || x > 3; },
                           "Error: --waifu2x-noise must be in [0, 3]");
        }},
        {"--waifu2x-ncnn-cmd", "command", false, [&](const char* v) { photo_up.waifu2x_ncnn_cmd = v; return true; }},
        {"--waifu2x-ncnn-args", "arguments", false, [&](const char* v) { photo_up.waifu2x_ncnn_args = v; return true; }},
        {"--waifu2x-ncnn-model", "model-dir name or explicit path", false, [&](const char* v) { photo_up.waifu2x_ncnn_model = v; return true; }},
        {"--realcugan-cmd", "command", false, [&](const char* v) { photo_up.realcugan_cmd = v; return true; }},
        {"--realcugan-args", "arguments", false, [&](const char* v) { photo_up.realcugan_args = v; return true; }},
        {"--realcugan-model", "se, pro, nose or an explicit model-dir path", false, [&](const char* v) { photo_up.realcugan_model = v; return true; }},
        // post-pipeline whole-image up-then-down pass (decrypt-side only)
        {"--final-polish", "waifu2x, waifu2x-ncnn or realcugan", false, [&](const char* v) {
            return set_choice(v, photo_up.final_polish, {"waifu2x", "waifu2x-ncnn", "realcugan"},
                              "Error: --final-polish must be waifu2x, waifu2x-ncnn or realcugan");
        }},
        // weights below do NOT force manual mode: AUTO still derives
        // coef/iterations from the container ratio; only the weight changes
        {"--tv", "floating point", false, [&](const char* v) { return set_float(v, tv_lambda, nullptr, nullptr); }},
        {"--red", "floating point in [0, 1]", false, [&](const char* v) {
            float red = 0.0f;
            if (!set_float(v, red, [](float x) { return x < 0.0f || x > 1.0f; }, "Error: --red must be in [0, 1]")) return false;
            photo_up.sr_red = red;
            return true;
        }},
        {"--red-denoiser", "nlmeans, dncnn, scunet or restormer", false, [&](const char* v) {
            return set_choice(v, photo_up.sr_red_denoiser, {"nlmeans", "dncnn", "scunet", "restormer"},
                              "Error: --red-denoiser must be nlmeans, dncnn, scunet or restormer");
        }},
        {"--dncnn-model", "DnCNN ONNX model path", false, [&](const char* v) { photo_up.sr_dncnn_model = v; return true; }},
        {"--red-model", "ONNX model file", false, [&](const char* v) { photo_up.sr_red_model = v; return true; }},
        {"--red-decay", "floating point in [0, 1]", false, [&](const char* v) {
            float decay = 1.0f;
            if (!set_float(v, decay, [](float x) { return x < 0.0f || x > 1.0f; }, "Error: --red-decay must be in [0, 1]")) return false;
            photo_up.sr_red_decay = decay;
            return true;
        }},
        {"--hr-weight", "floating point in [0, 16]", false, [&](const char* v) {
            float hrw = 1.0f;
            if (!set_float(v, hrw, [](float x) { return x < 0.0f || x > 16.0f; }, "Error: --hr-weight must be in [0, 16]")) return false;
            photo_up.sr_hr_weight = hrw;
            return true;
        }},
        {"--hr-red-radius", "floating point in [0, 64]", false, [&](const char* v) {
            float hrr = 0.0f;
            if (!set_float(v, hrr, [](float x) { return x < 0.0f || x > 64.0f; }, "Error: --hr-red-radius must be in [0, 64]")) return false;
            photo_up.sr_hr_red_radius = hrr;
            return true;
        }},
        {"--onsager", nullptr, false, [&](const char*) { use_onsager = true; return true; }},
        {"--sr-dict", "CSD2 dictionary file", false, [&](const char* v) { photo_up.sr_dict = v; return true; }},
        {"--sr-init", "avir or cascade", false, [&](const char* v) {
            return set_choice(v, photo_up.sr_init, {"avir", "cascade"},
                              "Error: --sr-init must be avir or cascade", true);
        }},
        {"--sr-prior-model", "ONNX SR model file", false, [&](const char* v) { photo_up.sr_prior_model = v; return true; }},
        {"--sr-pnp", "off, prox or admm", false, [&](const char* v) {
            return set_choice(v, photo_up.sr_pnp, {"off", "prox", "admm"},
                              "Error: --sr-pnp must be off, prox or admm", true);
        }},
        {"--sr-pnp-rho", "floating point in (0, 8]", false, [&](const char* v) {
            float rho = 0.1f;
            if (!set_float(v, rho, [](float x) { return !(x > 0.0f) || x > 8.0f; }, "Error: --sr-pnp-rho must be in (0, 8]")) return false;
            photo_up.sr_pnp_rho = rho;
            return true;
        }},
        {"--sr-pnp-iters", "integer in [1, 50]", false, [&](const char* v) {
            int it = 8;
            if (!set_int(v, it, [](int x) { return x < 1 || x > 50; }, "Error: --sr-pnp-iters must be in [1, 50]")) return false;
            photo_up.sr_pnp_iters = it;
            return true;
        }},
        // encrypt-side switches: decrypt reads the mode from the container header
        {"--periodic", nullptr, false, [&](const char*) { periodic = true; return true; }},
        {"--tile-size", "integer", false, [&](const char* v) { return set_int(v, tile_size, nullptr, nullptr); }},
        {"--ycc420", nullptr, false, [&](const char*) { ycc420 = true; return true; }},
        {"--ycc422", nullptr, false, [&](const char*) { ycc422 = true; return true; }},
        {"--hf-focus", nullptr, false, [&](const char*) { hf_focus = true; return true; }},
        {"--hr-detail", "integer", false, [&](const char* v) { return set_int(v, hr_detail, nullptr, nullptr); }},
        {"--adaptive", nullptr, false, [&](const char*) { adaptive = true; return true; }},
        {"--adaptive-floor", "integer", false, [&](const char* v) {
            return set_int(v, adaptive_floor, [](int x) { return x < 0 || x > 255; }, "Error: --adaptive-floor must be in [0, 255]");
        }},
        {"--lod-full", "integer in [0, 255]", false, [&](const char* v) {
            return set_int(v, lod_full, [](int x) { return x < 0 || x > 255; }, "Error: --lod-full must be in [0, 255] (0 = off)");
        }},
        {"--adaptive-strength", "floating point in [0,1]", false, [&](const char* v) {
            return set_float(v, adaptive_strength, [](float x) { return x < 0.0f || x > 1.0f; }, "Error: --adaptive-strength must be in [0, 1]");
        }},
        {"--adaptive-gamma", "floating point in [0.1, 4]", false, [&](const char* v) {
            return set_float(v, adaptive_gamma, [](float x) { return x < 0.1f || x > 4.0f; }, "Error: --adaptive-gamma must be in [0.1, 4]");
        }},
        {"--adaptive-cutoff", "integer in [0, 255]", false, [&](const char* v) {
            return set_int(v, adaptive_cutoff, [](int x) { return x < 0 || x > 255; }, "Error: --adaptive-cutoff must be in [0, 255]");
        }},
        {"--two-pass", nullptr, false, [&](const char*) { two_pass = true; return true; }},
        {"--spectral", nullptr, false, [&](const char*) { spectral = true; return true; }},
        {"--pilot-ratio", "floating point in [0.01,0.25]", false, [&](const char* v) {
            return set_float(v, pilot_ratio, [](float x) { return x < 0.01f || x > 0.25f; }, "Error: --pilot-ratio must be in [0.01, 0.25]");
        }},
        {"--regions", "JSON text", false, [&](const char* v) { regions_json = v; return true; }},
        {"--region-blend", "floating point in [0,1]", false, [&](const char* v) {
            return set_float(v, region_blend, [](float x) { return x < 0.0f || x > 1.0f; }, "Error: --region-blend must be in [0, 1]");
        }},
        {"--lod-smooth", "floating point in [0,8]", false, [&](const char* v) {
            return set_float(v, lod_smooth, [](float x) { return x < 0.0f || x > 8.0f; }, "Error: --lod-smooth must be in [0, 8]");
        }},
        {"--seg-mask", "label/color mask PNG path", false, [&](const char* v) { seg_mask = v; return true; }},
        {"--seg-rates", "comma-separated per-class rates", false, [&](const char* v) { seg_rates = v; return true; }},
        {"--seg-blend", "boundary blend width in pixels", false, [&](const char* v) {
            return set_int_strict(v, seg_blend, "Error: --seg-blend needs an integer in [0, 128]");
        }},
        {"--seg-solve", "label/color mask PNG path", false, [&](const char* v) { seg_solve = v; return true; }},
        {"--seg-auto", "class count 2..255", false, [&](const char* v) {
            return set_int_strict(v, seg_auto, "Error: --seg-auto needs an integer in [2, 255]");
        }},
        {"--seg-edge", nullptr, false, [&](const char*) { seg_edge = true; return true; }},
        {"--sampling", "points|tcluster|tmeans", false, [&](const char* v) {
            return set_choice(v, sampling, {"points", "tcluster", "tmeans"},
                              "Error: --sampling must be points|tcluster|tmeans");
        }},
        {"--seg-params", "per-class solver overrides", false, [&](const char* v) { seg_params = v; return true; }},
        {"--show-mask", nullptr, false, [&](const char*) { show_mask = true; return true; }},
        {"--sample-split", "output prefix", false, [&](const char* v) { sample_split = v; return true; }},
        {"--sample-bits", "L or L,C, each integer in [1, 8]", false, [&](const char* v) {
            const std::string arg(v);
            const size_t comma = arg.find(',');
            const std::string luma = arg.substr(0, comma);
            if (!parse_int(luma.c_str(), sample_bits)) return false;
            chroma_sample_bits = sample_bits;
            if (comma != std::string::npos) {
                if (arg.find(',', comma + 1) != std::string::npos) {
                    std::cerr << "Error: --sample-bits takes L or L,C" << std::endl;
                    return false;
                }
                const std::string chroma = arg.substr(comma + 1);
                if (!parse_int(chroma.c_str(), chroma_sample_bits)) return false;
            }
            if (sample_bits < 1 || sample_bits > 8 || chroma_sample_bits < 1 || chroma_sample_bits > 8) {
                std::cerr << "Error: --sample-bits luma and chroma values must each be in [1, 8]" << std::endl;
                return false;
            }
            return true;
        }},
        {"--full-res", nullptr, false, [&](const char*) { full_res = true; return true; }},
        {"--solver", "fista|joint", false, [&](const char* v) {
            solver_name = v;
            if (cs_solver_from_name(solver_name, solver) != 0) {
                std::cerr << "Error: unknown --solver '" << solver_name << "' (fista|joint)" << std::endl;
                return false;
            }
            return true;
        }},
        {"--fista-iters", "integer", false, [&](const char* v) {
            fista_iters_set = true;
            return set_int(v, fista_iters, [](int x) { return x < 0 || x > 500; }, "Error: --fista-iters must be in [0, 500]");
        }},
        {"--reweights", "integer", false, [&](const char* v) {
            return set_int(v, reweights, [](int x) { return x < 1 || x > 5; }, "Error: --reweights must be in [1, 5]");
        }},
        {"--basis", "dct|wavelet", false, [&](const char* v) {
            basis_name = v;
            if (cs_basis_from_name(basis_name, basis) != 0) {
                std::cerr << "Error: unknown --basis '" << basis_name << "' (dct|wavelet)" << std::endl;
                return false;
            }
            return true;
        }},
        {"--wscale", "float", false, [&](const char* v) {
            return set_float(v, wscale, [](float x) { return x <= 0.0f || x > 16.0f; }, "Error: --wscale must be in (0, 16]");
        }},
        {"--dct-block", "integer", false, [&](const char* v) {
            if (!set_int(v, dct_block, [](int x) { return x != 0 && (x < 2 || x > 1024); },
                         "Error: --dct-block must be 0 (whole-plane) or in [2, 1024]")) return false;
            cs_set_default_dct_block(dct_block);
            return true;
        }},
        {"--pad-radix", nullptr, false, [&](const char*) { cs_set_default_pad_radix(1); return true; }},
        {"--device", "cpu|gpu", false, [&](const char* v) {
            if (std::strcmp(v, "gpu") == 0) {
                cs_gpu::set_enabled(true);
            } else if (std::strcmp(v, "cpu") == 0) {
                cs_gpu::set_enabled(false);
            } else {
                std::cerr << "Error: unknown --device '" << v << "' (cpu|gpu)" << std::endl;
                return false;
            }
            return true;
        }},
    };

    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        const Opt* opt = nullptr;
        for (const Opt& candidate : opts) {
            if (a == candidate.name) { opt = &candidate; break; }
        }
        if (!opt) {
            std::cerr << "Error: unknown option '" << a << "'" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
        const char* v = nullptr;
        if (opt->hint) {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires a value (" << opt->hint << ")" << std::endl;
                return 64;
            }
            v = argv[++i];
        }
        if (!opt->apply(v)) return 64;
        if (opt->manual) manual = true;
    }

    // --fast decode preset: tile overlap 8 and 12 FISTA steps per pass, unless given explicitly
    // (IMG_3690 ycc420 + anchors: 2.1 s -> 0.96 s, -0.12 dB PSNR, SSIM 0.9285 -> 0.9283).
    if (fast) {
        if (!overlap_set) { overlap = 8; photo_up.auto_overlap = 8; }
        if (!fista_iters_set) fista_iters = 12;
    }

    if (!have_password) {
        password = read_password();
    }

    if (password.size() < 10) {
        std::cerr << "Error: password should be at least 10 characters" << std::endl;
        return 64;
    }

    if (two_pass && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --two-pass is encrypt-side only; ignoring for decrypt" << std::endl;
        two_pass = false;
    }
    if (lod_full > 0 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --lod-full is encrypt-side only; ignoring for decrypt" << std::endl;
        lod_full = 0;
    }
    if (adaptive_gamma != 1.0f && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --adaptive-gamma is encrypt-side only; ignoring for decrypt" << std::endl;
        adaptive_gamma = 1.0f;
    }
    if (adaptive_cutoff != 0 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --adaptive-cutoff is encrypt-side only; ignoring for decrypt" << std::endl;
        adaptive_cutoff = 0;
    }
    if (!regions_json.empty() && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --regions is encrypt-side only; ignoring for decrypt" << std::endl;
        regions_json.clear();
    }
    if ((!seg_mask.empty() || !seg_rates.empty()) && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-mask/--seg-rates is encrypt-side only; ignoring for decrypt" << std::endl;
        seg_mask.clear();
        seg_rates.clear();
    }
    if (seg_auto != 0 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-auto is encrypt-side only; ignoring for decrypt" << std::endl;
        seg_auto = 0;
    }
    if (seg_edge && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-edge is encrypt-side only; ignoring for decrypt" << std::endl;
        seg_edge = false;
    }
    if (seg_blend != 0 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-blend is encrypt-side only; ignoring for decrypt" << std::endl;
        seg_blend = 0;
    }
    if (sampling != "points" && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --sampling is encrypt-side only (decrypt auto-detects); ignoring for decrypt" << std::endl;
        sampling = "points";
    }
    if (seg_blend != 0 && (seg_mask.empty() || seg_rates.empty()) && seg_auto == 0) {
        std::cerr << "Error: --seg-blend needs --seg-mask/--seg-rates or --seg-auto" << std::endl;
        return 64;
    }
    if (seg_blend < 0 || seg_blend > 128) {
        std::cerr << "Error: --seg-blend must be in [0, 128] pixels" << std::endl;
        return 64;
    }
    if (!seg_solve.empty() && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-solve is decrypt-side only; ignoring for encrypt" << std::endl;
        seg_solve.clear();
    }
    if (!seg_params.empty() && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --seg-params is decrypt-side only; ignoring for encrypt" << std::endl;
        seg_params.clear();
    }
    if (!seg_params.empty() && seg_solve.empty() && (mode == "decrypt" || mode == "roundtrip")) {
        std::cerr << "Error: --seg-params needs --seg-solve" << std::endl;
        return 64;
    }
    if (ycc420 && ycc422) {
        std::cerr << "Error: --ycc420 and --ycc422 are mutually exclusive" << std::endl;
        return 64;
    }
    if ((ycc420 || ycc422) && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --ycc420/--ycc422 is encrypt-side only; ignoring for decrypt" << std::endl;
        ycc420 = false;
        ycc422 = false;
    }
    if ((ycc420 || ycc422) && periodic) {
        std::cerr << "Warning: --periodic is meaningless with --ycc420/--ycc422; ignoring" << std::endl;
        periodic = false;
    }
    if (hf_focus && periodic) {
        std::cerr << "Error: --hf-focus cannot be combined with --periodic" << std::endl;
        return 64;
    }
    if (hf_focus && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --hf-focus is encrypt-side only; ignoring for decrypt" << std::endl;
        hf_focus = false;
    }
    if (hr_detail != 0 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --hr-detail is encrypt-side only; ignoring for decrypt" << std::endl;
        hr_detail = 0;
    }
    if (!photo_up.sr_dict.empty() && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --sr-dict is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.sr_dict.clear();
    }
    if (!photo_up.final_polish.empty() && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --final-polish is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.final_polish.clear();
    }
    if (photo_up.sr_init != "avir" && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --sr-init is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.sr_init = "avir";
    }
    if ((!photo_up.sr_prior_model.empty() || photo_up.sr_pnp != "off") &&
        mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --sr-prior-model/--sr-pnp are decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.sr_prior_model.clear();
        photo_up.sr_pnp = "off";
    }
    if (photo_up.y_confidence > 0.0f && mode == "encrypt") {
        std::cerr << "Warning: --y-confidence is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.y_confidence = 0.0f;
    }
    if (photo_up.y_dual_tv > 0.0f && mode == "encrypt") {
        std::cerr << "Warning: --y-dual-tv is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.y_dual_tv = 0.0f;
    }
    if (photo_up.y_dual_tv > 0.0f && photo_up.y_dual_tv <= tv_lambda && mode != "encrypt") {
        std::cerr << "Error: --y-dual-tv must exceed --tv (the low-TV solve)" << std::endl;
        return 64;
    }
    if (!photo_up.y_dual_tv_mask.empty() && mode == "encrypt") {
        std::cerr << "Warning: --y-dual-tv-mask is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.y_dual_tv_mask.clear();
    }
    if (!photo_up.y_dual_tv_mask.empty() && photo_up.y_dual_tv <= 0.0f && mode != "encrypt") {
        std::cerr << "Warning: --y-dual-tv-mask needs --y-dual-tv; no mask will be written" << std::endl;
    }
    if (!photo_up.dual_tv_split.empty() && mode == "encrypt") {
        std::cerr << "Warning: --dual-tv-split is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.dual_tv_split.clear();
    }
    if (!photo_up.dual_tv_split.empty() && photo_up.y_dual_tv <= 0.0f && mode != "encrypt") {
        std::cerr << "Warning: --dual-tv-split needs --y-dual-tv (YCC containers only); nothing will be written" << std::endl;
    }
    if (!photo_up.dual_tv_cutouts.empty() && mode == "encrypt") {
        std::cerr << "Warning: --dual-tv-cutouts is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.dual_tv_cutouts.clear();
    }
    if (!photo_up.dual_tv_cutouts.empty() && photo_up.y_dual_tv <= 0.0f && mode != "encrypt") {
        std::cerr << "Warning: --dual-tv-cutouts needs --y-dual-tv (YCC containers only); nothing will be written" << std::endl;
    }
    if (!sample_split.empty() && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --sample-split is encrypt-side only; ignoring for decrypt" << std::endl;
        sample_split.clear();
    }
    if (grain_tv > 0.0f && mode == "encrypt") {
        std::cerr << "Warning: --grain-tv is decrypt-side only; ignoring for encrypt" << std::endl;
        grain_tv = 0.0f;
    }
    if (photo_up.sr_hr_weight != 1.0f && mode == "encrypt") {
        std::cerr << "Warning: --hr-weight is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.sr_hr_weight = 1.0f;
    }
    if (photo_up.sr_hr_red_radius != 0.0f && mode == "encrypt") {
        std::cerr << "Warning: --hr-red-radius is decrypt-side only; ignoring for encrypt" << std::endl;
        photo_up.sr_hr_red_radius = 0.0f;
    }
    if (!photo_up.sr_red_model.empty() && cs_onnx_kind_for(photo_up.sr_red_denoiser) < 0 && mode != "encrypt") {
        std::cerr << "Warning: --red-model needs an ONNX --red-denoiser (dncnn/scunet/restormer); ignoring" << std::endl;
        photo_up.sr_red_model.clear();
    }
    if ((photo_up.sr_red_decay < 1.0f || !photo_up.sr_red_model.empty()) && mode == "encrypt") {
        if (photo_up.sr_red_decay < 1.0f) {
            std::cerr << "Warning: --red-decay is decrypt-side only; ignoring for encrypt" << std::endl;
            photo_up.sr_red_decay = 1.0f;
        }
        if (!photo_up.sr_red_model.empty()) {
            std::cerr << "Warning: --red-model is decrypt-side only; ignoring for encrypt" << std::endl;
            photo_up.sr_red_model.clear();
        }
    }

    CSencryption::params = manual ? MANUAL_PARAM : AUTO_PARAM;

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

    try {
        // tile_size > 0 selects periodic sampling on the encrypt side; the
        // mode travels inside the authenticated container, so decrypt needs
        // no sampling flags. --adaptive/--two-pass win over --periodic when
        // combined (--two-pass implies the adaptive container), unless
        // --ycc420/--ycc422 is set: then the adaptive-family flags are absorbed into
        // the LOD-luma variant of the split container instead.
        const bool use_adaptive = adaptive || two_pass || spectral || !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0;
        const int periodic_tile_arg = (use_adaptive || periodic) ? tile_size : 0;
        if (mode == "encrypt") {
            rc = encrypt_image::encrypt_image_tiled(input,
output, password, ratio, periodic_tile_arg, nullptr, use_adaptive, adaptive_floor, adaptive_strength, show_mask,
full_res, two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, ycc420, hf_focus, sample_bits,
chroma_sample_bits, lod_full, ycc422, spectral, sample_split, hr_detail, adaptive_gamma, adaptive_cutoff, seg_mask,
seg_rates, seg_blend, sampling, seg_auto, seg_edge);
        }
        else if (mode == "decrypt") {
            rc = decrypt_image::decrypt_image_tiled(input, output, password,
                tiles, overlap, iterations, threads, coef, show_preview, tv_lambda, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv, grain_tv, use_onsager, seg_solve, seg_params);
        }
        else { // roundtrip: decrypt the in-memory encrypted image, no disk roundtrip
            cv::Mat encrypted;
            rc = encrypt_image::encrypt_image_tiled(input,
output, password, ratio, periodic_tile_arg, &encrypted, use_adaptive, adaptive_floor, adaptive_strength, show_mask,
full_res, two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, ycc420, hf_focus, sample_bits,
chroma_sample_bits, lod_full, ycc422, spectral, sample_split, hr_detail, adaptive_gamma, adaptive_cutoff, seg_mask,
seg_rates, seg_blend, sampling, seg_auto, seg_edge);
            if (rc == 0) {
                    rc = decrypt_image::decrypt_image_tiled(encrypted, output, password,
                        tiles, overlap, iterations, threads, coef, show_preview, tv_lambda, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv, grain_tv, use_onsager, nullptr, seg_solve, seg_params);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 3;
    }

    if (rc != 0) {
        std::cerr << (mode == "decrypt" ? "Decryption" : "Encryption") << " failed (code " << rc << ")" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    return 0;
}



