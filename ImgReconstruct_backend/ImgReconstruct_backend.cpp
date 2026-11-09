#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include "photo_upscaler.hpp"
#include "quality_utils.hpp"
#include "enh_net.hpp"
#include "cs_gpu.h"
#ifdef CS_GPU_TILES
#include "gpu_tiles.h"
#endif
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

void print_usage(const char* exe) {
    std::cout <<
        "Compressive-sensing image encryption (production format only)\n"
        "\n"
        "Usage:\n"
        "  " << exe << " encrypt <input.png> <output.png> [options]\n"
        "  " << exe << " decrypt <input.png> <output.png> [options]\n"
        "  " << exe << " encrypt|decrypt @<pairs.tsv> [options]   (batch: one 'input<TAB>output' per line, one process)\n"
        "  " << exe << " compare <original.png> <decrypted.png> <out_prefix> [--amp <f>]\n"
        "\n"
        "Options for both:\n"
        "  --password <pw>       passphrase, at least 10 characters (else\n"
        "                        $CS_ENCRYPTION_PASSWORD, else interactive prompt)\n"
        "  --no-preview          disable the live decryption preview window\n"
        "  --device <name>       cpu (default) or gpu (HIP offload)\n"
        "  --enhance <model>     restoration CNN (ONNX). Decrypt: run on the finished\n"
        "                        decode with the decoder's sample density, anchor and\n"
        "                        LR-sample maps as extra inputs, then project onto the\n"
        "                        container's samples. Encrypt: the net the anchor pass\n"
        "                        ranks the blocks after (the error left after it)\n"
        "\n"
        "Encrypt (the production container; the first five are required):\n"
        "  --seg-auto <k>        k-means scene classes (2..32; production 8)\n"
        "  --ycc420              luma at full resolution, chroma on the 4:2:0 grid\n"
        "  --seg-tile            each class split into 8 detail levels of 64x64 tiles\n"
        "  --ratio <f>           byte budget as a sampling ratio in (0.001, 1.0]\n"
        "  --hr-detail <n>       exact native-luma anchors, placed on the worst\n"
        "                        16x16 blocks of an in-memory decode (cap 2^24)\n"
        "  --multiscale auto|off auto (default): multi-scale container (mode 20), each\n"
        "                        120 px tile at its own scale 2x..6x, planned for the\n"
        "                        budget of --ratio; off: single-scale (mode 19)\n"
        "  --hr-scale <m>        multiply the --hr-detail count by m (0..16). Default\n"
        "                        1 single-scale; multi-scale 0.5 up to --ratio 0.5,\n"
        "                        2 from 0.7, log-linear between\n"
        "  --sample-bits <L[,C]> luma / chroma sample bits, each in [1, 8]. Default:\n"
        "                        6-bit luma / 8-bit chroma at the byte budget of 8-bit\n"
        "                        --ratio (~1.2x the samples, capped at full sampling)\n"
        "  --show-mask           write <output>.mask.png (source | binary mask | cyan\n"
        "                        overlay) and show it in a window\n"
        "\n"
        "Decrypt (the container's format travels in its header):\n"
        "  --tv <f>              TV weight of the tile solve (production 0; default:\n"
        "                        auto, 0.01 below ratio 0.5)\n"
        "  --photo-upscaler cs-sr  the 2x stage (the only one; accepted for old commands)\n"
        "  --sr-pnp admm         its PnP-ADMM loop (the only one; accepted for old commands)\n"
        "  --sr-pnp-iters <n>    ADMM outer passes in [1, 50] (production 2; default 8)\n"
        "  --hr-weight <f>       anchor data-term weight in [0, 16] (default 1)\n"
        "  --enhance2 <model>    second unrolled stage on the projected output of the\n"
        "                        first, its planes rebuilt against it, then projected\n"
        "  --enhance-low <model> use this net instead of --enhance on containers with\n"
        "                        under 0.43 luma sample bits per output pixel (about\n"
        "                        r0.25 and below); the decode prints its choice\n"
        "\n"
        "  Production decode: --tv 0 --photo-upscaler cs-sr --sr-pnp admm --sr-pnp-iters 2\n"
        "  --hr-weight 1 --device gpu with CS_ADMM_PROX_OFF=1 (identity ADMM prox; the\n"
        "  decoder then scales the auto coef x0.2) and CS_GPU_TILES=1 CS_GPU_ADMM=1.\n"
        "  compare scores decryption quality (PSNR/SSIM/MAE/max) and writes\n"
        "    <out_prefix>_sidebyside.png  original | decrypted | JET error map\n"
        "    <out_prefix>_error.png       amplified absolute-error map\n"
        "\n"
        "Examples:\n"
        "  " << exe << " encrypt photo.png photo.enc.png --password \"my secret\" --seg-auto 8 --ycc420 --seg-tile\n"
        "      --ratio 0.3 --hr-detail 450000 --enhance models\\enh\\enh_seg8a_x5_ema256_fresh_unet.onnx\n"
        "  " << exe << " decrypt photo.enc.png photo.dec.png --password \"my secret\" --tv 0 --photo-upscaler cs-sr\n"
        "      --sr-pnp admm --sr-pnp-iters 2 --hr-weight 1 --device gpu --enhance <net> --enhance2 <net> --enhance-low <net>\n"
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
// value does not parse or fails `bad`, printing `msg`.
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

} // namespace

#ifdef CS_GPU_TILES
// GPU build: the GPU stages are the default. CS_GPU_TILES (LR tile solve, batched cs-sr tile
// stage, compositing), CS_GPU_ADMM (PnP-ADMM) and CS_GPU_ONNX (DirectML enhance net) default
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
    // OpenMP workers sleep between parallel regions instead of spinning: spinning workers took the cores the decode's
    // other threads (GPU driving, side preps, PNG write) need. -64 ms per GPU decode, -0.4 s on a CPU decode. An
    // OMP_WAIT_POLICY already set is kept. libomp has its own CRT and reads the OS environment block on first use, so
    // the variable goes there (_putenv_s would only change this CRT's copy).
    if (GetEnvironmentVariableA("OMP_WAIT_POLICY", nullptr, 0) == 0) SetEnvironmentVariableA("OMP_WAIT_POLICY", "PASSIVE");
#ifdef CS_GPU_TILES
    cs_gpu_env_defaults();
#endif
    if (argc < 2) {
        print_usage(argv[0]);
        return 64; // usage error
    }

    const std::string mode = argv[1];
    if (mode != "encrypt" && mode != "decrypt" && mode != "compare") {
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

    // encrypt / decrypt @<pairs.tsv>: a batch of jobs in this process (no output path argument)
    const bool batch = argc >= 3 && argv[2][0] == '@';
    if (argc < (batch ? 3 : 4)) {
        std::cerr << "Error: missing input/output paths" << std::endl;
        print_usage(argv[0]);
        return 64;
    }
    const bool enc = mode == "encrypt";

    const std::string input = argv[2];
    const std::string output = batch ? std::string() : argv[3];

    std::string password;
    bool have_password = false;
    bool show_preview = true;
    CsPhotoUpscalerOptions photo_up;
    // encrypt
    float ratio = 1.0f;
    int seg_auto = 0, hr_detail = 0;
    bool ycc420 = false, seg_tile = false, multiscale = true, show_mask = false;
    float hr_scale = 1.0f;
    bool hr_scale_set = false;
    int sample_bits = 8, chroma_sample_bits = 8;
    bool sample_bits_set = false;
    // decrypt
    float tv_lambda = -1.0f;

    // One row per flag. `hint` names the value in the missing-value error
    // (null for switches); `modes` is where it applies ('e' encrypt, 'd' decrypt);
    // `apply` returns false on a bad value (exit 64).
    struct Opt {
        const char* name;
        const char* hint;
        const char* modes;
        std::function<bool(const char*)> apply;
    };
    const std::vector<Opt> opts = {
        {"--password", "passphrase text", "ed", [&](const char* v) { password = v; have_password = true; return true; }},
        {"--no-preview", nullptr, "ed", [&](const char*) { show_preview = false; return true; }},
        {"--device", "cpu|gpu", "ed", [&](const char* v) {
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
        {"--enhance", "ONNX model path", "ed", [&](const char* v) { photo_up.enhance_model = v; return true; }},
        {"--ratio", "floating point", "e", [&](const char* v) { return set_float(v, ratio, nullptr, nullptr); }},
        {"--seg-auto", "class count 2..32", "e", [&](const char* v) {
            return set_int(v, seg_auto, [](int x) { return x < 2 || x > 32; }, "Error: --seg-auto needs an integer in [2, 32]");
        }},
        {"--ycc420", nullptr, "e", [&](const char*) { ycc420 = true; return true; }},
        {"--seg-tile", nullptr, "e", [&](const char*) { seg_tile = true; return true; }},
        {"--hr-detail", "integer", "e", [&](const char* v) { return set_int(v, hr_detail, nullptr, nullptr); }},
        {"--multiscale", "auto or off", "e", [&](const char* v) {
            if (std::strcmp(v, "auto") != 0 && std::strcmp(v, "off") != 0) {
                std::cerr << "Error: --multiscale must be auto or off" << std::endl;
                return false;
            }
            multiscale = std::strcmp(v, "auto") == 0;
            return true;
        }},
        {"--hr-scale", "floating point in [0,16]", "e", [&](const char* v) {
            hr_scale_set = true;
            return set_float(v, hr_scale, [](float x) { return x < 0.0f || x > 16.0f; }, "Error: --hr-scale must be in [0, 16]");
        }},
        {"--sample-bits", "L or L,C, each integer in [1, 8]", "e", [&](const char* v) {
            const std::string arg(v);
            sample_bits_set = true;
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
        {"--show-mask", nullptr, "e", [&](const char*) { show_mask = true; return true; }},
        {"--tv", "floating point", "d", [&](const char* v) { return set_float(v, tv_lambda, nullptr, nullptr); }},
        {"--photo-upscaler", "cs-sr", "d", [&](const char* v) {
            if (std::strcmp(v, "cs-sr") == 0) return true;
            std::cerr << "Error: --photo-upscaler: only cs-sr is in this build" << std::endl;
            return false;
        }},
        {"--sr-pnp", "admm", "d", [&](const char* v) {
            if (std::strcmp(v, "admm") == 0) return true;
            std::cerr << "Error: --sr-pnp: only admm is in this build" << std::endl;
            return false;
        }},
        {"--sr-pnp-iters", "integer in [1, 50]", "d", [&](const char* v) {
            return set_int(v, photo_up.sr_pnp_iters, [](int x) { return x < 1 || x > 50; }, "Error: --sr-pnp-iters must be in [1, 50]");
        }},
        {"--hr-weight", "floating point in [0, 16]", "d", [&](const char* v) {
            return set_float(v, photo_up.sr_hr_weight, [](float x) { return x < 0.0f || x > 16.0f; }, "Error: --hr-weight must be in [0, 16]");
        }},
        {"--enhance2", "ONNX model path", "d", [&](const char* v) { photo_up.enhance2_model = v; return true; }},
        {"--enhance-low", "ONNX model path", "d", [&](const char* v) { photo_up.enhance_low_model = v; return true; }},
    };

    for (int i = batch ? 3 : 4; i < argc; i++) {
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
        if (!std::strchr(opt->modes, enc ? 'e' : 'd')) {
            std::cerr << "Error: " << a << " is " << (enc ? "a decrypt" : "an encrypt") << " option" << std::endl;
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
    }

    if (!have_password) {
        password = read_password();
    }

    if (password.size() < 10) {
        std::cerr << "Error: password should be at least 10 characters" << std::endl;
        return 64;
    }

    if (enc) {
        if (seg_auto == 0 || !ycc420 || !seg_tile || hr_detail <= 0) {
            std::cerr << "Error: encrypt writes the production container only: pass --seg-auto <k> --ycc420 --seg-tile "
                         "--hr-detail <n> (production: --seg-auto 8 --ycc420 --seg-tile --hr-detail 450000)" << std::endl;
            return 64;
        }
        // The multi-scale container (the default) scales the anchor count by --ratio: x0.5 up to r0.5, x2 from r0.7,
        // log-linear between (settings oracle, 88 photos, 2026-11-06: TODO.md). --hr-scale overrides m.
        if (multiscale && !hr_scale_set) hr_scale = 0.5f * std::pow(4.0f, std::clamp((ratio - 0.5f) / 0.2f, 0.0f, 1.0f));
        if (hr_scale != 1.0f)
            hr_detail = (int)std::lround((double)hr_detail * hr_scale);
    }

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

    try {
        // Batch (encrypt / decrypt @<pairs.tsv>): one process for the whole list, so HIP start-up, kernel loads and
        // the --enhance weights happen once. A failed job is reported and the batch goes on; rc = 1 if any failed.
        std::vector<std::pair<std::string, std::string>> jobs;
        if (batch) {
            std::ifstream lf(input.substr(1));
            if (!lf) { std::cerr << "Error: cannot read batch list '" << input.substr(1) << "'" << std::endl; return 64; }
            for (std::string line; std::getline(lf, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                const size_t tab = line.find('\t');
                if (tab == std::string::npos) { std::cerr << "Error: batch line without a tab: " << line << std::endl; return 64; }
                jobs.emplace_back(line.substr(0, tab), line.substr(tab + 1));
            }
        }
        const auto batch_line = [&](size_t j, int r, std::chrono::high_resolution_clock::time_point tj) {
            if (r != 0) std::cerr << "Error: job failed: " << jobs[j].first << std::endl;
            std::cout << "[batch] " << (j + 1) << "/" << jobs.size() << " " << jobs[j].second << " " <<
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tj).count() << " ms" << std::endl;
        };
        if (enc) {
            // The encoder decodes its own anchor-free container (cs-sr + PnP-ADMM, tv 0, tile overlap 8 and 12 FISTA
            // steps per pass: it only ranks 16x16 blocks by error, after the --enhance net when given) in memory, then
            // anchors the worst blocks. Flag map and anchor luma ride in the container.
            const std::function<cv::Mat(const cv::Mat&)> anchor_decode = [&](const cv::Mat& container) {
                CsPhotoUpscalerOptions pass_up = photo_up;
                pass_up.auto_overlap = 8;
                cv::Mat dec;
                if (decrypt_image::decrypt_image_tiled(container, "", password, false, 0.0f, 12, pass_up, &dec) != 0)
                    dec.release();
                return dec;
            };
#ifdef CS_GPU_TILES
            // HIP start-up overlaps the image read and segmentation (the in-memory decode has no warm-up of its own)
            struct JoinedThread { std::thread t; ~JoinedThread() { if (t.joinable()) t.join(); } } gpu_warm;
            if (std::getenv("CS_GPU_TILES")) gpu_warm.t = std::thread(gpu_tiles::warmup);
#endif
            if (!batch) jobs.emplace_back(input, output);
            int failed = 0;
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto tj = std::chrono::high_resolution_clock::now();
                const int r = multiscale
                    ? cs_encrypt_multiscale(jobs[j].first, jobs[j].second, password, hr_detail, anchor_decode, ratio,
                          show_mask && !batch)
                    : encrypt_image::encrypt_image_tiled(jobs[j].first, jobs[j].second, password, ratio, seg_auto,
                          hr_detail, sample_bits_set ? sample_bits : 0,   // 0 = the default 6-bit luma
                          chroma_sample_bits, show_mask && !batch, anchor_decode);
                if (!batch) { rc = r; break; }
                failed += r != 0;
                batch_line(j, r, tj);
            }
            if (batch) rc = failed ? 1 : 0;
        }
        else if (batch) {
            // A side thread reads the next container while the current one decodes.
            struct JoinedThread { std::thread t; ~JoinedThread() { if (t.joinable()) t.join(); } } gpu_warm, enh_pre;
#ifdef CS_GPU_TILES
            if (std::getenv("CS_GPU_TILES")) gpu_warm.t = std::thread(gpu_tiles::warmup);
            if (!photo_up.enhance_model.empty())
                enh_pre.t = std::thread([&photo_up] {
                    cs_enhance_prefetch(photo_up.enhance_model);
                    if (!photo_up.enhance2_model.empty()) cs_enhance_prefetch(photo_up.enhance2_model);
                    if (!photo_up.enhance_low_model.empty()) cs_enhance_prefetch(photo_up.enhance_low_model);
                });
#endif
            cv::Mat next;
            if (!jobs.empty()) next = cv::imread(jobs[0].first, cv::IMREAD_COLOR);
            int failed = 0;
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto tj = std::chrono::high_resolution_clock::now();
                cv::Mat cur = std::move(next);
                std::thread reader;
                if (j + 1 < jobs.size()) reader = std::thread([&next, &jobs, j] { next = cv::imread(jobs[j + 1].first, cv::IMREAD_COLOR); });
                if (gpu_warm.t.joinable()) gpu_warm.t.join();   // the decode's own launches wait for the runtime anyway
                const int r = cur.empty() ? -1 : decrypt_image::decrypt_image_tiled(cur, jobs[j].second, password,
                    false, tv_lambda, 0, photo_up);
                if (reader.joinable()) reader.join();
                failed += r != 0;
                batch_line(j, r, tj);
            }
            rc = failed ? 1 : 0;
        }
        else {
            rc = decrypt_image::decrypt_image_tiled(input, output, password, show_preview, tv_lambda, 0, photo_up);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 3;
    }

    if (rc != 0) {
        std::cerr << (enc ? "Encryption" : "Decryption") << " failed (code " << rc << ")" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    return 0;
}
