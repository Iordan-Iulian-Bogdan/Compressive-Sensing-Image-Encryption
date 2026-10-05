#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include "photo_upscaler.hpp"
#include "quality_utils.hpp"
#include "cs_gpu.h"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
        "\n"
"Options:\n"
        "  --password <pw>       passphrase (else $CS_PASSWORD, else interactive prompt)\n"
        "  --ratio <f>           encryption sampling ratio in (0.001, 1.0]; default 1.0\n"
        "  --tiles <n>           decrypt tile count >= 1 (manual mode only,\n"
        "                        default 1: whole image is one tile)\n"
        "  --overlap <n>         decrypt tile overlap in [0, 96] (manual mode only;\n"
  "                        0 disables overlap: tiles solve independently and\n"
  "                        composite without feathering)\n"
        "  --iterations <n>      decrypt solver iterations (manual mode only)\n"
        "  --threads <n>         decrypt worker threads (manual mode only)\n"
        "  --coef <f>            decrypt solver coefficient (>0; higher = sparser,\n"
        "                        no upper bound; manual mode only)\n"
        "  --per-tile-coef       scale the L1 coefficient per tile from its\n"
        "                        sample count (starved tiles regularized more,\n"
        "                        rich tiles relaxed; tv stays global unless\n"
        "                        --per-tile-tv is also given)\n"
        "  --per-tile-tv         scale the TV weight per tile the same way\n"
        "                        (smooth/starved tiles smoothed more; watch\n"
        "                        for seams on flat gradients)\n"
        "  --manual              force manual parameter mode (auto is the default and\n"
        "                        derives tiles/overlap/iterations/coef from the image)\n"
        "  --no-preview          disable the live decryption preview window\n"
        "  --photo-upscaler <b>  per-tile 2x upscaler backend: avir (default,\n"
        "                        vendored resampler), waifu2x (one batched\n"
        "                        nunif subprocess), waifu2x-ncnn (native\n"
        "                        waifu2x-ncnn-vulkan binary) or realcugan\n"
        "                        (native realcugan-ncnn-vulkan binary); the\n"
        "                        ncnn binaries run on Vulkan (AMD included)\n"
        "                        with no Python, and fall back to AVIR when\n"
        "                        unavailable)\n"
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
  "  --tv <f>              total-variation fusion weight for the solver\n"
  "                        (>0 enables TV, smoother edges, helps at high\n"
  "                        compression; 0 = off)\n"
  "  --solver <name>      tile solver: fista (default, reweighted-L1\n"
  "                        FISTA per channel), joint (SOMP-structured\n"
  "                        group-sparsity FISTA over R/G/B together +\n"
  "                        reweighting)\n"
  "  --fista-iters <n>    steps per reweight pass (0 = auto-map from\n"
  "                        --iterations as N*4 clamped to [16,40]; max 500)\n"
  "  --reweights <n>      outer reweight passes (1-5, default 2;\n"
  "                        1 = plain unweighted solve)\n"
  "  --basis <name>       sparsifying basis: dct (default, block-DCT)\n"
  "                        or wavelet (multilevel CDF 9/7, sparser on\n"
  "                        natural images)\n"
  "  --wscale <f>         wavelet per-scale threshold ramp\n"
  "                        (in (0,16], default 2: coarsest band x1, finest\n"
  "                        band x wscale, LL protected; 1 = uniform;\n"
  "                        DCT ignores it)\n"
        "  --device <name>      solve device: cpu (default) or gpu\n"
        "                        (HIP offload; dct basis, any tv; other\n"
        "                        combinations fall back to cpu per call)\n"
        "  --periodic            encrypt with periodic tile sampling: one random\n"
        "                        per-tile pattern repeated across the image\n"
        "                        (encrypt/roundtrip only; decrypt auto-detects\n"
        "                        the mode from the container)\n"
        "  --tile-size <n>       periodic tile size (default 64)\n"
        "  --ycc420              encrypt with luma/chroma-split 4:2:0 sampling: Y\n"
  "                        at full resolution, each chroma plane sampled on\n"
  "                        its half-resolution grid (~1.5 bytes/px stored\n"
  "                        instead of 3; decrypt auto-detects from the\n"
  "                        container). Adaptive flags (--adaptive, --two-pass,\n"
  "                        --regions, --lod-smooth, --adaptive-strength/floor,\n"
  "                        --lod-full) are absorbed: they drive the luma budget\n"
  "                        while chroma stays uniform. --periodic is ignored\n"
  "                        under --ycc420.\n"
  "  --hf-focus            encrypt a stored LF thumbnail plus HF-weighted\n"
  "                        samples (BGR mode 5; with --ycc420, mode 6)\n"
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
  "  --two-pass            adaptive scoring via a pilot uniform sample +\n"
  "                        cheap per-tile recon (residual scoring) instead\n"
  "                        of the Laplacian; implies adaptive, decrypt\n"
  "                        needs no new flags (encrypt/roundtrip only)\n"
  "  --pilot-ratio <f>     pilot fraction of each tile's pixels for\n"
  "                        --two-pass scoring in [0.01, 0.25]\n"
  "                        (default 0.05; min 16 samples/tile)\n"
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
        "  --show-mask           after encrypt/roundtrip sampling, write\n"
        "                        <output>.mask.png and open a preview window\n"
        "                        (source | binary mask | cyan overlay)\n"
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

    std::printf("PSNR: %.2f dB\n", p);
    if (s >= 0.0) std::printf("SSIM: %.4f\n", s);
    else          std::printf("SSIM: n/a (OpenCV quality module not available)\n");
    std::printf("MAE:  %.3f\n", mae);
    std::printf("MAX:  %.1f\n", mx);

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

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 64; // usage error
    }

    const std::string mode = argv[1];
    if (mode != "encrypt" && mode != "decrypt" && mode != "roundtrip" && mode != "compare") {
        std::cerr << "Error: unknown mode '" << mode << "'" << std::endl;
        print_usage(argv[0]);
        return 64;
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
    float coef = 0.01f;
    bool per_tile_coef = false;
    bool per_tile_tv = false;
    float tv_lambda = 0.0f;
    bool manual = false;
    bool show_preview = true;
    CsPhotoUpscalerOptions photo_up;
    bool periodic = false;
    int tile_size = 64;
    bool adaptive = false;
    bool ycc420 = false;
    bool hf_focus = false;
    int adaptive_floor = 32;    float adaptive_strength = 0.5f;
    int lod_full = 0;
    bool two_pass = false;
    float pilot_ratio = 0.05f;
    std::string regions_json;
    float region_blend = 0.5f;
    float lod_smooth = 0.0f;
    bool show_mask = false;
    bool full_res = false;
    int sample_bits = 8;
    int chroma_sample_bits = 8;
    std::string solver_name = "fista";
    int solver = CS_SOLVER_FISTA;
    int fista_iters = 0;
    int reweights = 2;
    std::string basis_name = "dct";
    int basis = CS_BASIS_DCT;
    float wscale = 2.0f;

    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires a value (" << what << ")" << std::endl;
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--password") {
            const char* v = next("passphrase text");
            if (!v) return 64;
            password = v; have_password = true;
        }
        else if (a == "--ratio") {
            const char* v = next("floating point");
            if (!v || !parse_float(v, ratio)) return 64;
        }
        else if (a == "--tiles") {
            const char* v = next("integer");
            if (!v || !parse_int(v, tiles)) return 64;
            manual = true;
        }
        else if (a == "--overlap") {
            const char* v = next("integer");
            if (!v || !parse_int(v, overlap)) return 64;
            manual = true;
        }
        else if (a == "--iterations") {
            const char* v = next("integer");
            if (!v || !parse_int(v, iterations)) return 64;
            manual = true;
        }
        else if (a == "--threads") {
            const char* v = next("integer");
            if (!v || !parse_int(v, threads)) return 64;
            manual = true;
        }
        else if (a == "--coef") {
            const char* v = next("floating point");
            if (!v || !parse_float(v, coef)) return 64;
            manual = true;
        }
        else if (a == "--per-tile-coef") {
            // decrypt-side only; per-tile counts travel in the container,
            // so this must not flip decrypt into manual parameter mode
            per_tile_coef = true;
        }
        else if (a == "--per-tile-tv") {
            // same, for the TV weight (independent flag: tv seams are the
            // failure mode to watch, keep it separable from coef)
            per_tile_tv = true;
        }
        else if (a == "--manual") {
            manual = true;
        }
        else if (a == "--no-preview") {
            show_preview = false;
        }
        else if (a == "--photo-upscaler") {
            const char* v = next("avir, waifu2x, waifu2x-ncnn or realcugan");
            if (!v) return 64;
            if (v != std::string("avir") && v != std::string("waifu2x") &&
                v != std::string("waifu2x-ncnn") && v != std::string("realcugan")) {
                std::cerr << "Error: --photo-upscaler must be avir, waifu2x, waifu2x-ncnn or realcugan" << std::endl;
                return 64;
            }
            photo_up.backend = v;
        }
        else if (a == "--waifu2x-cmd") {
            const char* v = next("command");
            if (!v) return 64;
            photo_up.waifu2x_cmd = v;
        }
        else if (a == "--waifu2x-args") {
            const char* v = next("arguments");
            if (!v) return 64;
            photo_up.waifu2x_args = v;
        }
        else if (a == "--waifu2x-method") {
            const char* v = next("scale or noise_scale");
            if (!v) return 64;
            if (v != std::string("scale") && v != std::string("noise_scale")) {
                std::cerr << "Error: --waifu2x-method must be scale or noise_scale" << std::endl;
                return 64;
            }
            photo_up.waifu2x_method = v;
        }
        else if (a == "--waifu2x-noise") {
            const char* v = next("integer in [0,3]");
            if (!v || !parse_int(v, photo_up.waifu2x_noise)) return 64;
            if (photo_up.waifu2x_noise < 0 || photo_up.waifu2x_noise > 3) {
                std::cerr << "Error: --waifu2x-noise must be in [0, 3]" << std::endl;
                return 64;
            }
        }
        else if (a == "--waifu2x-ncnn-cmd") {
            const char* v = next("command");
            if (!v) return 64;
            photo_up.waifu2x_ncnn_cmd = v;
        }
        else if (a == "--waifu2x-ncnn-args") {
            const char* v = next("arguments");
            if (!v) return 64;
            photo_up.waifu2x_ncnn_args = v;
        }
        else if (a == "--realcugan-cmd") {
            const char* v = next("command");
            if (!v) return 64;
            photo_up.realcugan_cmd = v;
        }
        else if (a == "--realcugan-args") {
            const char* v = next("arguments");
            if (!v) return 64;
            photo_up.realcugan_args = v;
        }
        else if (a == "--realcugan-model") {
            const char* v = next("se, pro, nose or an explicit model-dir path");
            if (!v) return 64;
            photo_up.realcugan_model = v;
        }
        else if (a == "--tv") {
            // does NOT force manual mode: AUTO still derives coef/iterations
            // from the container ratio; only the TV weight is overridden
            const char* v = next("floating point");
            if (!v || !parse_float(v, tv_lambda)) return 64;
        }
        else if (a == "--periodic") {
            // encrypt-side only: decrypt auto-detects the mode from the
            // container header, so this must not flip decrypt into manual
            // parameter mode (CLI default coef/iterations are wrong for
            // ratios other than 1.0 and collapse quality to ~11 dB)
            periodic = true;
        }
        else if (a == "--tile-size") {
            const char* v = next("integer");
            if (!v || !parse_int(v, tile_size)) return 64;
        }
        else if (a == "--ycc420") {
            // encrypt-side only (same as --periodic): luma/chroma-split 4:2:0
            // sampling; decrypt auto-detects mode 3 from the container header
            ycc420 = true;
        }
        else if (a == "--hf-focus") {
            hf_focus = true;
        }
        else if (a == "--adaptive") {
            // encrypt-side only (same as --periodic): never forces manual
            // decrypt parameters — mode travels in the authenticated header
            adaptive = true;
        }
        else if (a == "--adaptive-floor") {
            const char* v = next("integer");
            if (!v || !parse_int(v, adaptive_floor)) return 64;
            if (adaptive_floor < 0 || adaptive_floor > 255) {
                std::cerr << "Error: --adaptive-floor must be in [0, 255]" << std::endl;
                return 64;
            }
        }
        else if (a == "--lod-full") {
            // encrypt-side detail guarantee; decrypt restores the threshold
            // from the header, so this must not flip decrypt into manual
            const char* v = next("integer in [0, 255]");
            if (!v || !parse_int(v, lod_full)) return 64;
            if (lod_full < 0 || lod_full > 255) {
                std::cerr << "Error: --lod-full must be in [0, 255] (0 = off)" << std::endl;
                return 64;
            }
        }
        else if (a == "--adaptive-strength") {
            const char* v = next("floating point in [0,1]");
            if (!v || !parse_float(v, adaptive_strength)) return 64;
            if (adaptive_strength < 0.0f || adaptive_strength > 1.0f) {
                std::cerr << "Error: --adaptive-strength must be in [0, 1]" << std::endl;
                return 64;
            }
        }
        else if (a == "--two-pass") {
            // encrypt-side scoring variant: implies the adaptive container
            // (mode travels in the header, decrypt auto-detects)
            two_pass = true;
        }
        else if (a == "--pilot-ratio") {
            const char* v = next("floating point in [0.01,0.25]");
            if (!v || !parse_float(v, pilot_ratio)) return 64;
            if (pilot_ratio < 0.01f || pilot_ratio > 0.25f) {
                std::cerr << "Error: --pilot-ratio must be in [0.01, 0.25]" << std::endl;
                return 64;
            }
        }
        else if (a == "--regions") {
            const char* v = next("JSON text");
            if (!v) return 64;
            regions_json = v;
        }
        else if (a == "--region-blend") {
            const char* v = next("floating point in [0,1]");
            if (!v || !parse_float(v, region_blend)) return 64;
            if (region_blend < 0.0f || region_blend > 1.0f) {
                std::cerr << "Error: --region-blend must be in [0, 1]" << std::endl;
                return 64;
            }
        }
        else if (a == "--lod-smooth") {
            const char* v = next("floating point in [0,8]");
            if (!v || !parse_float(v, lod_smooth)) return 64;
            if (lod_smooth < 0.0f || lod_smooth > 8.0f) {
                std::cerr << "Error: --lod-smooth must be in [0, 8]" << std::endl;
                return 64;
            }
        }
        else if (a == "--show-mask") {
            // encrypt/roundtrip only: after sampling, save + open the mask
            show_mask = true;
        }
        else if (a == "--sample-bits") {
            // encrypt-side only (container size lever); decrypt auto-detects
            // from the header, so this must not flip decrypt into manual
            const char* v = next("L or L,C, each integer in [1, 8]");
            if (!v) return 64;
            const std::string arg(v);
            const size_t comma = arg.find(',');
            const std::string luma = arg.substr(0, comma);
            if (!parse_int(luma.c_str(), sample_bits)) return 64;
            chroma_sample_bits = sample_bits;
            if (comma != std::string::npos) {
                if (arg.find(',', comma + 1) != std::string::npos) {
                    std::cerr << "Error: --sample-bits takes L or L,C" << std::endl;
                    return 64;
                }
                const std::string chroma = arg.substr(comma + 1);
                if (!parse_int(chroma.c_str(), chroma_sample_bits)) return 64;
            }
            if (sample_bits < 1 || sample_bits > 8 || chroma_sample_bits < 1 || chroma_sample_bits > 8) {
                std::cerr << "Error: --sample-bits luma and chroma values must each be in [1, 8]" << std::endl;
                return 64;
            }
        }
        else if (a == "--full-res") {
            // both sides: skip encrypt 2x downscale AND decrypt 2x upscale
            full_res = true;
        }
        else if (a == "--solver") {
            const char* v = next("fista|joint");
            if (!v) return 64;
            solver_name = v;
            if (cs_solver_from_name(solver_name, solver) != 0) {
                std::cerr << "Error: unknown --solver '" << solver_name << "' (fista|joint)" << std::endl;
                return 64;
            }
        }
        else if (a == "--fista-iters") {
            const char* v = next("integer");
            if (!v || !parse_int(v, fista_iters)) return 64;
            if (fista_iters < 0 || fista_iters > 500) {
                std::cerr << "Error: --fista-iters must be in [0, 500]" << std::endl;
                return 64;
            }
        }
        else if (a == "--reweights") {
            const char* v = next("integer");
            if (!v || !parse_int(v, reweights)) return 64;
            if (reweights < 1 || reweights > 5) {
                std::cerr << "Error: --reweights must be in [1, 5]" << std::endl;
                return 64;
            }
        }
        else if (a == "--basis") {
            const char* v = next("dct|wavelet");
            if (!v) return 64;
            basis_name = v;
            if (cs_basis_from_name(basis_name, basis) != 0) {
                std::cerr << "Error: unknown --basis '" << basis_name << "' (dct|wavelet)" << std::endl;
                return 64;
            }
        }
        else if (a == "--wscale") {
            const char* v = next("float");
            if (!v || !parse_float(v, wscale)) return 64;
            if (wscale <= 0.0f || wscale > 16.0f) {
                std::cerr << "Error: --wscale must be in (0, 16]" << std::endl;
                return 64;
            }
        }
        else if (a == "--device") {
            const char* v = next("cpu|gpu");
            if (!v) return 64;
            if (std::strcmp(v, "gpu") == 0) {
                cs_gpu::set_enabled(true);
            } else if (std::strcmp(v, "cpu") == 0) {
                cs_gpu::set_enabled(false);
            } else {
                std::cerr << "Error: unknown --device '" << v << "' (cpu|gpu)" << std::endl;
                return 64;
            }
        }
        else {
            std::cerr << "Error: unknown option '" << a << "'" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
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
    if (!regions_json.empty() && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --regions is encrypt-side only; ignoring for decrypt" << std::endl;
        regions_json.clear();
    }
    if (ycc420 && mode != "encrypt" && mode != "roundtrip") {
        std::cerr << "Warning: --ycc420 is encrypt-side only; ignoring for decrypt" << std::endl;
        ycc420 = false;
    }
    if (ycc420 && periodic) {
        std::cerr << "Warning: --periodic is meaningless with --ycc420; ignoring" << std::endl;
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

    CSencryption::params = manual ? MANUAL_PARAM : AUTO_PARAM;

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

    try {
        // tile_size > 0 selects periodic sampling on the encrypt side; the
        // mode travels inside the authenticated container, so decrypt needs
        // no sampling flags. --adaptive/--two-pass win over --periodic when
        // combined (--two-pass implies the adaptive container), unless
        // --ycc420 is set: then the adaptive-family flags are absorbed into
        // the LOD-luma variant of the split container instead.
        const bool use_adaptive = adaptive || two_pass || !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0;
        const int periodic_tile_arg = (use_adaptive || periodic) ? tile_size : 0;
        if (mode == "encrypt") {
            rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, periodic_tile_arg, nullptr, use_adaptive, adaptive_floor, adaptive_strength, show_mask, full_res, two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, ycc420, hf_focus, sample_bits, chroma_sample_bits, lod_full);
        }
        else if (mode == "decrypt") {
            rc = decrypt_image::decrypt_image_tiled(input, output, password,
                tiles, overlap, iterations, threads, coef, show_preview, tv_lambda, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv);
        }
        else { // roundtrip: decrypt the in-memory encrypted image, no disk roundtrip
            cv::Mat encrypted;
            rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, periodic_tile_arg, &encrypted, use_adaptive, adaptive_floor, adaptive_strength, show_mask, full_res, two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, ycc420, hf_focus, sample_bits, chroma_sample_bits, lod_full);
            if (rc == 0) {
                    rc = decrypt_image::decrypt_image_tiled(encrypted, output, password,
                        tiles, overlap, iterations, threads, coef, show_preview, tv_lambda, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv);
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



