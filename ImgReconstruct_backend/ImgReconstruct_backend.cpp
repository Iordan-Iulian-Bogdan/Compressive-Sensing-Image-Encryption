#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include "quality_utils.hpp"
#include "clip_scorer.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
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
        "  --overlap <n>         decrypt tile overlap in (24, 96) (manual mode only)\n"
        "  --iterations <n>      decrypt solver iterations (manual mode only)\n"
        "  --threads <n>         decrypt worker threads (manual mode only)\n"
        "  --coef <f>            decrypt solver coefficient in (0.01, 0.05) (manual mode only)\n"
        "  --manual              force manual parameter mode (auto is the default and\n"
        "                        derives tiles/overlap/iterations/coef from the image)\n"
        "  --no-preview          disable the live decryption preview window\n"
        "  --denoise             apply a final non-local-means denoise pass\n"
        "                        (smooths solver noise, blurs fine detail)\n"
        "  --tv <f>              total-variation fusion weight for the solver\n"
        "                        (>0 enables TV, smoother edges, helps at high\n"
        "                        compression; 0 = off)\n"
        "  --dict <file>         solve with a learned K-SVD patch dictionary\n"
        "                        instead of the DCT basis (see the trainer tool)\n"
        "  --periodic            encrypt with periodic tile sampling: one random\n"
        "                        per-tile pattern repeated across the image\n"
        "                        (encrypt/roundtrip only; decrypt auto-detects\n"
        "                        the mode from the container)\n"
        "  --tile-size <n>       periodic tile size (default 64)\n"
        "  --adaptive            encrypt with LOD-based adaptive sampling: each\n"
        "                        tile gets an 8-bit detail score shipped in the\n"
        "                        container; sample budget splits by score so\n"
        "                        detailed tiles are over-sampled and flat tiles\n"
        "                        under-sampled (encrypt/roundtrip only; decrypt\n"
        "                        auto-detects from the container)\n"
        "  --adaptive-floor <n>  minimum LOD byte in adaptive mode (default 32;\n"
        "                        keeps a floor share for flat tiles)\n"
        "  --adaptive-strength <s>  how aggressively detail tiles are over-sampled\n"
        "                        and flat tiles under-sampled, s in [0, 1]\n"
        "                        (default 0.5; 0 = uniform like random sampling,\n"
        "                        1 = maximum LOD bias)\n"
        "  --show-mask           after encrypt/roundtrip sampling, write\n"
        "                        <output>.mask.png and open a preview window\n"
        "                        (source | binary mask | cyan overlay)\n"
        "  --full-res            keep native resolution end-to-end (skip the\n"
        "                        default encrypt 2x downscale and decrypt\n"
        "                        2x upscale); use on encrypt AND decrypt/roundtrip\n"
        "  --desc <text>         CLIP restart selection: decrypt N candidates\n"
        "                        with perturbed warm-starts and keep the one\n"
        "                        scoring highest against this description\n"
        "                        (decrypt/roundtrip only; needs models/clip/)\n"
        "  --restarts <n>        candidate count for --desc (default 4, >= 1;\n"
        "                        candidate 0 is the unperturbed decrypt)\n"
        "  --clip-dir <dir>      CLIP model directory (default models/clip\n"
        "                        or $CS_CLIP_DIR)\n"
        "  --tune                CLIP-guided coordinate descent: search\n"
        "                        (coef, tv, iterations) to maximize\n"
        "                        alpha*CLIP - beta*measurement-residual,\n"
        "                        starting from the AUTO decrypt (requires\n"
        "                        --desc; cannot combine with --restarts)\n"
        "  --tune-rounds <n>     coordinate-descent rounds (default 1)\n"
        "  --tune-alpha <f>      CLIP weight (default 1.0)\n"
        "  --tune-beta <f>       residual weight (default 0.005)\n"
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

// CLIP restart selection: decrypt N candidates (candidate 0 unperturbed,
// rest with seeded warm-start noise) and keep the highest-scoring one.
// input_path is used for decrypt mode; encrypted (non-empty) for roundtrip.
int run_clip_selection(const std::string& input_path, const cv::Mat& encrypted,
    const std::string& output, const std::string& password,
    int tiles, int overlap, int iterations, int threads, float coef,
    bool show_preview, bool denoise, float tv_lambda, const std::string& dict_path,
    bool full_res, const std::string& desc, int restarts, const std::string& clip_dir)
{
    ClipScorer clip;
    if (!clip.load(clip_dir)) {
        std::cerr << "Error: CLIP load failed: " << clip.error() << std::endl;
        std::cerr << "Run scripts\\download_clip.ps1 first (models/clip/)" << std::endl;
        return 2;
    }
    std::string best_path;
    float best_score = -2.0f;
    for (int k = 0; k < restarts; ++k) {
        const int seed = (k == 0) ? -1 : (k - 1);
        const std::string cand = output + ".cand" + std::to_string(k) + ".png";
        int rc;
        if (!input_path.empty())
            rc = decrypt_image::decrypt_image_tiled(input_path, cand, password,
                tiles, overlap, iterations, threads, coef, show_preview,
                denoise, tv_lambda, dict_path, full_res, seed);
        else
            rc = decrypt_image::decrypt_image_tiled(encrypted, cand, password,
                tiles, overlap, iterations, threads, coef, show_preview,
                denoise, tv_lambda, dict_path, full_res, seed);
        if (rc != 0) {
            std::cerr << "candidate " << k << " decrypt failed (code " << rc << ")" << std::endl;
            std::remove(cand.c_str());
            return 1;
        }
        cv::Mat img = cv::imread(cand, cv::IMREAD_COLOR);
        const float s = clip.score(img, desc);
        std::printf("candidate %d (seed %d): CLIP %.4f\n", k, seed, s);
        if (s > best_score) {
            if (!best_path.empty()) std::remove(best_path.c_str());
            best_score = s;
            best_path = cand;
        } else {
            std::remove(cand.c_str());
        }
    }
    std::printf("selected %s (CLIP %.4f)\n", best_path.c_str(), best_score);
    std::remove(output.c_str());
    if (std::rename(best_path.c_str(), output.c_str()) != 0) {
        std::cerr << "Error: cannot rename best candidate to output" << std::endl;
        return 1;
    }
    return 0;
}

// Measurement consistency: mean |candidate - measurement| over the sampled
// pixels (in [0,1] units). Guards CLIP against rewarding confabulation that
// contradicts the container. Candidate is the final full-res output; it is
// downscaled to the solve geometry before comparison (unless --full-res).
double measurement_mae(const cv::Mat& encrypted, const std::string& password,
                       const cv::Mat& candidate) {
    decrypt_image dimgs(encrypted, password);
    cv::Mat sampled, masked;
    dimgs.get_sampled_mat(sampled, masked);
    if (sampled.empty() || candidate.empty()) return 1e9;
    cv::Mat cand;
    if (candidate.size() != sampled.size())
        cv::resize(candidate, cand, sampled.size(), 0, 0, cv::INTER_AREA);
    else
        cand = candidate;
    cv::Mat a, b;
    cand.convertTo(a, CV_32FC3, 1.0 / 255.0);
    sampled.convertTo(b, CV_32FC3, 1.0 / 255.0);
    double sum = 0;
    long n = 0;
    for (int y = 0; y < a.rows; ++y) {
        const cv::Vec3b* m = masked.ptr<cv::Vec3b>(y);
        const cv::Vec3f* pa = a.ptr<cv::Vec3f>(y);
        const cv::Vec3f* pb = b.ptr<cv::Vec3f>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (m[x] == cv::Vec3b(1, 1, 1)) {
                sum += std::fabs(pa[x][0] - pb[x][0])
                     + std::fabs(pa[x][1] - pb[x][1])
                     + std::fabs(pa[x][2] - pb[x][2]);
                ++n;
            }
        }
    }
    if (n == 0) return 1e9;
    return sum / (3.0 * n);
}

// CLIP-guided coordinate descent over (coef, tv, iterations): the gradient-
// free stand-in for a clip_lambda term in evaluate(). Trial 0 is the AUTO
// decrypt (incumbent, never-worse guarantee); phases then sweep one axis at
// a time in MANUAL mode (tiles/overlap mirror AUTO: 24/24). Objective:
// alpha*CLIP - beta*residual. Evaluated points are cached across rounds.
int run_tune(const std::string& input_path, const cv::Mat& encrypted,
    const std::string& output, const std::string& password,
    int tiles, int overlap, int iterations, int threads, float coef,
    bool show_preview, bool denoise, float tv_lambda, const std::string& dict_path,
    bool full_res, const std::string& desc, const std::string& clip_dir,
    int rounds, double alpha, double beta)
{
    ClipScorer clip;
    if (!clip.load(clip_dir)) {
        std::cerr << "Error: CLIP load failed: " << clip.error() << std::endl;
        std::cerr << "Run scripts\\download_clip.ps1 first (models/clip/)" << std::endl;
        return 2;
    }
    cv::Mat enc = input_path.empty() ? encrypted
                                     : cv::imread(input_path, cv::IMREAD_COLOR);
    if (enc.empty()) {
        std::cerr << "Error: cannot load encrypted input" << std::endl;
        return 1;
    }
    // text embedding is description-only: compute once, reuse for all trials
    const std::vector<int> text_ids = clip.tokenize_public(desc);

    const int saved_params = CSencryption::params;
    std::map<std::string, double> cache;
    std::string best_path;
    double best_score = -1e100;
    float best_coef = coef;
    float best_tv = tv_lambda;
    int best_iters = iterations;
    int trial_no = 0;

    const char* best_label = "auto";
    auto eval_point = [&](float c, float tv, int it, const char* label) -> int {
        // key includes the label: the AUTO trial's derived params are
        // unknown to the CLI, so its score must not collide with a manual
        // trial that happens to share the CLI-default numbers
        char key[80];
        std::snprintf(key, sizeof(key), "%s|%.4f|%.4f|%d", label, c, tv, it);
        double s;
        auto hit = cache.find(key);
        if (hit != cache.end()) {
            s = hit->second;
            if (std::strcmp(label, "auto") == 0)
                std::printf("trial %d [auto]: cached score %.4f\n", trial_no, s);
            else
                std::printf("trial %d [%s] coef=%.3f tv=%.3f iters=%d: cached score %.4f%s\n",
                    trial_no, label, c, tv, it, s, s > best_score ? " *" : "");
        } else {
            const std::string cand = output + ".tune" + std::to_string(trial_no) + ".png";
            int rc;
            if (std::strcmp(label, "auto") == 0) {
                CSencryption::params = AUTO_PARAM;
                rc = decrypt_image::decrypt_image_tiled(enc, cand, password,
                    tiles, overlap, iterations, threads, coef, show_preview,
                    denoise, tv_lambda, dict_path, full_res);
            } else {
                CSencryption::params = MANUAL_PARAM;
                rc = decrypt_image::decrypt_image_tiled(enc, cand, password,
                    tiles, overlap, it, threads, c, show_preview,
                    denoise, tv, dict_path, full_res);
            }
            if (rc != 0) {
                std::cerr << "trial " << trial_no << " decrypt failed (code " << rc << ")" << std::endl;
                std::remove(cand.c_str());
                CSencryption::params = saved_params;
                return 1;
            }
            cv::Mat img = cv::imread(cand, cv::IMREAD_COLOR);
            const float cs = clip.score_embed(text_ids, img);
            const double rs = measurement_mae(enc, password, img);
            s = alpha * cs - beta * rs;
            cache[key] = s;
            const bool is_auto = std::strcmp(label, "auto") == 0;
            if (is_auto)
                std::printf("trial %d [auto, derived params]: CLIP %.4f resid %.4f score %.4f%s\n",
                    trial_no, cs, rs, s, s > best_score ? " *" : "");
            else
                std::printf("trial %d [%s] coef=%.3f tv=%.3f iters=%d: CLIP %.4f resid %.4f score %.4f%s\n",
                    trial_no, label, c, tv, it, cs, rs, s, s > best_score ? " *" : "");
            if (s > best_score) {
                if (!best_path.empty()) std::remove(best_path.c_str());
                best_score = s;
                best_path = cand;
                best_coef = c; best_tv = tv; best_iters = it;
                best_label = label;
            } else {
                std::remove(cand.c_str());
            }
        }
        // note: a cache hit can never beat best_score (same value was
        // already considered), so best_path always names a live file
        ++trial_no;
        return 0;
    };

    int rc = 0;
    // coef grid spans the MANUAL legal range (0.01, 0.05); note AUTO derives
    // its own coef inside decrypt (e.g. 0.075 at ratio 0.25), which may lie
    // outside this range — the manual trials explore the legal neighborhood
    const float coef_grid[] = { 0.01f, 0.02f, 0.03f, 0.045f };
    const float tv_grid[] = { 0.0f, 0.05f, 0.2f };
    const int iter_grid[] = { 5, 8, 12, 16 };
    for (int r = 0; r < rounds && rc == 0; ++r) {
        std::printf("--- tune round %d ---\n", r + 1);
        rc = eval_point(coef, tv_lambda, iterations, r == 0 ? "auto" : "auto");
        for (float c : coef_grid) {
            if (rc != 0) break;
            rc = eval_point(c, best_tv, best_iters, "coef");
        }
        for (float tv : tv_grid) {
            if (rc != 0) break;
            rc = eval_point(best_coef, tv, best_iters, "tv");
        }
        for (int it : iter_grid) {
            if (rc != 0) break;
            rc = eval_point(best_coef, best_tv, it, "iters");
        }
    }
    CSencryption::params = saved_params;
    if (rc != 0) {
        if (!best_path.empty()) std::remove(best_path.c_str());
        return rc;
    }
    if (std::strcmp(best_label, "auto") == 0)
        std::printf("tuned: AUTO params kept (score %.4f)\n", best_score);
    else
        std::printf("tuned coef=%.3f tv=%.3f iters=%d (score %.4f)\n",
            best_coef, best_tv, best_iters, best_score);
    std::remove(output.c_str());
    if (std::rename(best_path.c_str(), output.c_str()) != 0) {
        std::cerr << "Error: cannot rename best trial to output" << std::endl;
        return 1;
    }
    return 0;
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
    float tv_lambda = 0.0f;
    std::string dict_path;
    bool manual = false;
    bool show_preview = true;
    bool denoise = false;
    bool periodic = false;
    int tile_size = 64;
    bool adaptive = false;
    int adaptive_floor = 32;
    float adaptive_strength = 0.5f;
    bool show_mask = false;
    bool full_res = false;
    std::string desc;
    int restarts = 4;
    std::string clip_dir;
    bool tune = false;
    int tune_rounds = 1;
    double tune_alpha = 1.0;
    double tune_beta = 0.005;

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
        else if (a == "--manual") {
            manual = true;
        }
        else if (a == "--no-preview") {
            show_preview = false;
        }
        else if (a == "--denoise") {
            denoise = true;
        }
        else if (a == "--tv") {
            // does NOT force manual mode: AUTO still derives coef/iterations
            // from the container ratio; only the TV weight is overridden
            const char* v = next("floating point");
            if (!v || !parse_float(v, tv_lambda)) return 64;
        }
        else if (a == "--dict") {
            const char* v = next("dictionary file");
            if (!v) return 64;
            dict_path = v;
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
        else if (a == "--adaptive-strength") {
            const char* v = next("floating point in [0,1]");
            if (!v || !parse_float(v, adaptive_strength)) return 64;
            if (adaptive_strength < 0.0f || adaptive_strength > 1.0f) {
                std::cerr << "Error: --adaptive-strength must be in [0, 1]" << std::endl;
                return 64;
            }
        }
        else if (a == "--show-mask") {
            // encrypt/roundtrip only: after sampling, save + open the mask
            show_mask = true;
        }
        else if (a == "--full-res") {
            // both sides: skip encrypt 2x downscale AND decrypt 2x upscale
            full_res = true;
        }
        else if (a == "--desc") {
            const char* v = next("description text");
            if (!v) return 64;
            desc = v;
        }
        else if (a == "--restarts") {
            const char* v = next("integer");
            if (!v || !parse_int(v, restarts)) return 64;
            if (restarts < 1) {
                std::cerr << "Error: --restarts must be >= 1" << std::endl;
                return 64;
            }
        }
        else if (a == "--clip-dir") {
            const char* v = next("directory");
            if (!v) return 64;
            clip_dir = v;
        }
        else if (a == "--tune") {
            tune = true;
        }
        else if (a == "--tune-rounds") {
            const char* v = next("integer");
            if (!v || !parse_int(v, tune_rounds)) return 64;
            if (tune_rounds < 1) {
                std::cerr << "Error: --tune-rounds must be >= 1" << std::endl;
                return 64;
            }
        }
        else if (a == "--tune-alpha") {
            const char* v = next("floating point");
            if (!v) return 64;
            tune_alpha = std::atof(v);
        }
        else if (a == "--tune-beta") {
            const char* v = next("floating point");
            if (!v) return 64;
            tune_beta = std::atof(v);
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

    if (!desc.empty() && mode == "encrypt") {
        std::cerr << "Warning: --desc is decrypt-side only; ignoring for encrypt" << std::endl;
        desc.clear();
    }
    if (tune && mode == "encrypt") {
        std::cerr << "Warning: --tune is decrypt-side only; ignoring for encrypt" << std::endl;
        tune = false;
    }
    if (tune && desc.empty()) {
        std::cerr << "Error: --tune requires --desc" << std::endl;
        return 64;
    }
    if (tune && restarts != 4) {
        std::cerr << "Error: --tune cannot be combined with --restarts" << std::endl;
        return 64;
    }

    CSencryption::params = manual ? MANUAL_PARAM : AUTO_PARAM;

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

    try {
        // tile_size > 0 selects periodic sampling on the encrypt side; the
        // mode travels inside the authenticated container, so decrypt needs
        // no sampling flags. --adaptive wins over --periodic when both given.
        const int periodic_tile_arg = (adaptive || periodic) ? tile_size : 0;
        if (mode == "encrypt") {
            rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, periodic_tile_arg, nullptr, adaptive, adaptive_floor, adaptive_strength, show_mask, full_res);
        }
        else if (mode == "decrypt") {
            if (tune) {
                rc = run_tune(input, cv::Mat(), output, password,
                    tiles, overlap, iterations, threads, coef, show_preview, denoise,
                    tv_lambda, dict_path, full_res, desc, clip_dir,
                    tune_rounds, tune_alpha, tune_beta);
            } else if (desc.empty()) {
                rc = decrypt_image::decrypt_image_tiled(input, output, password,
                    tiles, overlap, iterations, threads, coef, show_preview, denoise, tv_lambda, dict_path, full_res);
            } else {
                rc = run_clip_selection(input, cv::Mat(), output, password,
                    tiles, overlap, iterations, threads, coef, show_preview, denoise,
                    tv_lambda, dict_path, full_res, desc, restarts, clip_dir);
            }
        }
        else { // roundtrip: decrypt the in-memory encrypted image, no disk roundtrip
            cv::Mat encrypted;
            rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, periodic_tile_arg, &encrypted, adaptive, adaptive_floor, adaptive_strength, show_mask, full_res);
            if (rc == 0) {
                if (tune) {
                    rc = run_tune(std::string(), encrypted, output, password,
                        tiles, overlap, iterations, threads, coef, show_preview, denoise,
                        tv_lambda, dict_path, full_res, desc, clip_dir,
                        tune_rounds, tune_alpha, tune_beta);
                } else if (desc.empty()) {
                    rc = decrypt_image::decrypt_image_tiled(encrypted, output, password,
                        tiles, overlap, iterations, threads, coef, show_preview, denoise, tv_lambda, dict_path, full_res);
                } else {
                    rc = run_clip_selection(std::string(), encrypted, output, password,
                        tiles, overlap, iterations, threads, coef, show_preview, denoise,
                        tv_lambda, dict_path, full_res, desc, restarts, clip_dir);
                }
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



