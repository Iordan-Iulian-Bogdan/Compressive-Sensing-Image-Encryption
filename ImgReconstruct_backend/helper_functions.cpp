#include "helper_functions.hpp"
#include "cs_gpu.h"
#include "crypto_utils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

cs_solve_profile g_solveprof;

namespace {
// file-local square sizing (same convention as the encrypt paths; their
// next_perfect_square lives in another TU)
int cs_square_total(size_t bytes) {
    const size_t pixels = (bytes + 2) / 3;
    int side = (int)std::ceil(std::sqrt((double)pixels + 8.0));
    if (side < 1) side = 1;
    while ((long long)side * side < (long long)pixels + 8) ++side;
    return side * side;
}
} // namespace

cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<cs_body_section>& sections) {
    const size_t raw_bytes = raw.total() * raw.elemSize();
    if (sections.empty()) {
        return raw.clone();
    }
    const size_t first = sections[0].offset;
    if (first > raw_bytes) {
        throw std::runtime_error("packed container: section starts past end of body");
    }
    bool any_packed = false;
    size_t packed_len = first;
    for (const auto& s : sections) {
        if (!cs_sample_bits_valid(s.bits) ||
            (s.interleaved_bgr && !cs_sample_bits_valid(s.chroma_bits))) {
            throw std::runtime_error("sample-bits out of range [1, 8]");
        }
        if (s.offset < first || s.offset > raw_bytes || s.count > raw_bytes - s.offset) {
            throw std::runtime_error("packed container: sections out of order");
        }
        if (s.interleaved_bgr && s.count % 3 != 0) {
            throw std::runtime_error("packed BGR section count is not divisible by 3");
        }
        any_packed = any_packed || s.bits < 8 ||
            (s.interleaved_bgr && s.chroma_bits < 8);
        packed_len += s.interleaved_bgr
            ? cs_packed_bgr_bytes(s.count / 3, s.bits, s.chroma_bits)
            : cs_packed_bytes(s.count, s.bits);
    }
    if (!any_packed) return raw.clone();
    const int total_p = cs_square_total(packed_len);
    cv::Mat out(1, total_p, CV_8UC3, cv::Scalar(0, 0, 0));
    std::memcpy(out.data, raw.data, first);
    size_t woff = first;
    for (const auto& s : sections) {
        if (s.interleaved_bgr) {
            if (s.count % 3 != 0) throw std::runtime_error("packed BGR section count is not divisible by 3");
            cs_pack_samples_bgr(raw.data + s.offset, s.count / 3,
                s.bits, s.chroma_bits, out.data + woff);
            woff += cs_packed_bgr_bytes(s.count / 3, s.bits, s.chroma_bits);
        } else {
            cs_pack_samples(raw.data + s.offset, s.count, s.bits, out.data + woff);
            woff += cs_packed_bytes(s.count, s.bits);
        }
    }
    return out;
}

cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<std::pair<size_t, size_t>>& sections, int bits) {
    std::vector<cs_body_section> converted;
    converted.reserve(sections.size());
    for (const auto& s : sections) converted.push_back({ s.first, s.second, bits, 0, false });
    return cs_pack_container_body(raw, converted);
}

size_t cs_unpacked_body_bytes(const std::vector<cs_body_section>& sections) {
    if (sections.empty()) return 0;
    size_t len = sections[0].offset;
    for (const auto& s : sections) len += s.count;
    return len;
}

size_t cs_unpacked_body_bytes(const std::vector<std::pair<size_t, size_t>>& sections) {
    if (sections.empty()) return 0;
    size_t len = sections[0].first;
    for (const auto& s : sections) len += s.second;
    return len;
}

void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<cs_body_section>& sections, uint8_t* raw) {
    if (sections.empty()) return;
    const size_t first = sections[0].offset;
    size_t needed = first;
    for (const auto& s : sections) {
        if (!cs_sample_bits_valid(s.bits) ||
            (s.interleaved_bgr && !cs_sample_bits_valid(s.chroma_bits))) {
            throw std::runtime_error("sample-bits out of range [1, 8]");
        }
        if (s.offset < first || (s.interleaved_bgr && s.count % 3 != 0)) {
            throw std::runtime_error("invalid packed container section");
        }
        needed += s.interleaved_bgr
            ? cs_packed_bgr_bytes(s.count / 3, s.bits, s.chroma_bits)
            : cs_packed_bytes(s.count, s.bits);
    }
    if (needed > packed_bytes) {
        throw std::runtime_error("packed container: body shorter than section table");
    }
    std::memcpy(raw, packed, first);
    size_t roff = first;
    for (const auto& s : sections) {
        if (s.interleaved_bgr) {
            if (s.count % 3 != 0) throw std::runtime_error("packed BGR section count is not divisible by 3");
            cs_unpack_samples_bgr(packed + roff, s.count / 3,
                s.bits, s.chroma_bits, raw + s.offset);
            roff += cs_packed_bgr_bytes(s.count / 3, s.bits, s.chroma_bits);
        } else {
            cs_unpack_samples(packed + roff, s.count, s.bits, raw + s.offset);
            roff += cs_packed_bytes(s.count, s.bits);
        }
    }
}

void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<std::pair<size_t, size_t>>& sections, int bits, uint8_t* raw) {
    std::vector<cs_body_section> converted;
    converted.reserve(sections.size());
    for (const auto& s : sections) converted.push_back({ s.first, s.second, bits, 0, false });
    cs_unpack_container_body(packed, packed_bytes, converted, raw);
}

namespace {
bool sp_env() {
    const char* v = std::getenv("CS_PROFILE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
inline long long sp_ns_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
} // namespace

void cs_solveprof_reset() {
    g_solveprof.on = sp_env();
    if (!g_solveprof.on) return;
    g_solveprof.warm_ns = 0; g_solveprof.ctor_ns = 0; g_solveprof.call_ns = 0;
    g_solveprof.wave_wall_ns = 0; g_solveprof.wave_cap_ns = 0;
    g_solveprof.tiles = 0; g_solveprof.samples = 0;
    g_solveprof.wrap_setup_ns = 0; g_solveprof.wrap_core_ns = 0;
    g_solveprof.wrap_rw_ns = 0; g_solveprof.wrap_tail_ns = 0;
    g_solveprof.wrap_calls = 0; g_solveprof.rw_passes = 0;
    g_solveprof.admm_setup_ns = 0; g_solveprof.admm_rhs_ns = 0;
    g_solveprof.admm_x_ns = 0; g_solveprof.admm_cg_ns = 0;
    g_solveprof.admm_z_ns = 0; g_solveprof.admm_dual_ns = 0;
    g_solveprof.admm_iters = 0;
    g_solveprof.fs_grad_ns = 0; g_solveprof.fs_shrink_ns = 0;
    g_solveprof.fs_mom_ns = 0; g_solveprof.fs_iters = 0;
    g_solveprof.owl_setup_ns = 0; g_solveprof.owl_lbfgs_ns = 0; g_solveprof.owl_tail_ns = 0;
    g_solveprof.owl_eval_idct_ns = 0; g_solveprof.owl_eval_data_ns = 0;
    g_solveprof.owl_eval_dct_ns = 0; g_solveprof.owl_evals = 0;
}

// one "profile[solve ...]" line; all times in milliseconds
void cs_solveprof_dump(int num_tiles, int num_threads, long long wall_ns) {
    if (!g_solveprof.on) return;
    const double ms = 1e-6;
    const long long work = g_solveprof.warm_ns + g_solveprof.ctor_ns + g_solveprof.call_ns;
    const double eff = g_solveprof.wave_cap_ns > 0
        ? 100.0 * (double)work / (double)g_solveprof.wave_cap_ns : 0.0;
    std::fprintf(stderr,
        "profile[solve tiles=%d threads=%d]: wall=%.1f warm=%.1f ctor=%.1f call=%.1f"
        " | wave wall=%.1f cap=%.1f eff=%.1f%% tiles=%lld m_sum=%lld"
        " | wrap setup=%.1f core=%.1f rw=%.1f tail=%.1f calls=%lld rwp=%lld"
        " | admm setup=%.1f rhs=%.1f x=%.1f cg=%.1f z=%.1f dual=%.1f iters=%lld"
        " | fsta grad=%.1f shrink=%.1f mom=%.1f iters=%lld"
        " | owl setup=%.1f lbfgs=%.1f tail=%.1f eval idct=%.1f data=%.1f dct=%.1f evals=%lld\n",
        num_tiles, num_threads, wall_ns * ms,
        g_solveprof.warm_ns.load() * ms, g_solveprof.ctor_ns.load() * ms, g_solveprof.call_ns.load() * ms,
        g_solveprof.wave_wall_ns.load() * ms, g_solveprof.wave_cap_ns.load() * ms, eff,
        g_solveprof.tiles.load(), g_solveprof.samples.load(),
        g_solveprof.wrap_setup_ns.load() * ms, g_solveprof.wrap_core_ns.load() * ms,
        g_solveprof.wrap_rw_ns.load() * ms, g_solveprof.wrap_tail_ns.load() * ms,
        g_solveprof.wrap_calls.load(), g_solveprof.rw_passes.load(),
        g_solveprof.admm_setup_ns.load() * ms, g_solveprof.admm_rhs_ns.load() * ms,
        g_solveprof.admm_x_ns.load() * ms, g_solveprof.admm_cg_ns.load() * ms,
        g_solveprof.admm_z_ns.load() * ms, g_solveprof.admm_dual_ns.load() * ms,
        g_solveprof.admm_iters.load(),
        g_solveprof.fs_grad_ns.load() * ms, g_solveprof.fs_shrink_ns.load() * ms,
        g_solveprof.fs_mom_ns.load() * ms, g_solveprof.fs_iters.load(),
        g_solveprof.owl_setup_ns.load() * ms, g_solveprof.owl_lbfgs_ns.load() * ms,
        g_solveprof.owl_tail_ns.load() * ms, g_solveprof.owl_eval_idct_ns.load() * ms,
        g_solveprof.owl_eval_data_ns.load() * ms, g_solveprof.owl_eval_dct_ns.load() * ms,
        g_solveprof.owl_evals.load());
    cs_solveprof_reset();
}

int nextClosestDivisible(int x, int y) {
    // Ensure y is not zero to avoid division by zero error
    if (y == 0) {
        throw std::invalid_argument("y must not be zero");
    }

    // Find the next multiple of y greater than x
    int nextMultiple = ((x + y - 1) / y) * y;

    return nextMultiple;
}

inline void updateAxb2AndComputeFx(float* x_copy, const int* ri_x, const int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n) {
    __m256 fx_vec = _mm256_setzero_ps();  // Accumulator for fx

    int i = 0;
    for (; i <= n - 8; i += 8) {
        // Gather indices (aka coordinates of sampled pixels)
        int idx[8];
        for (int k = 0; k < 8; k++) {
            idx[k] = ri_x[i + k] * cols + ri_y[i + k];
        }

        // Load x_copy values using gather
        __m256 x_val = _mm256_i32gather_ps(x_copy, _mm256_load_si256((__m256i*) & idx[0]), 4);

        // Load b values (measurment aka sampled tile)
        __m256 b_val = _mm256_load_ps(&b[i]);

        // Compute differences
        __m256 diff = _mm256_sub_ps(x_val, b_val);

        // Accumulate fx (diff * diff)
        fx_vec = _mm256_fmadd_ps(diff, diff, fx_vec);

        // Store differences to Axb2_vec
        alignas(32) float temp[8];
        _mm256_store_ps(temp, diff);
        for (int k = 0; k < 8; k++) {
            Axb2_vec[idx[k]] = temp[k];
        }
    }

    // Handle remaining elements
    float fx_temp = 0.0f;
    for (; i < n; ++i) {
        int idx = ri_x[i] * cols + ri_y[i];
        float diff = x_copy[idx] - b[i];
        fx_temp += diff * diff;
        Axb2_vec[idx] = diff;
    }

    // Reduce fx_vec to scalar
    __m128 hi = _mm256_extractf128_ps(fx_vec, 1);
    __m128 lo = _mm256_castps256_ps128(fx_vec);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    fx = _mm_cvtss_f32(sum) + fx_temp;
}

inline void eval_g(float* Axb2, float* g, int n) {
    __m256 scalar = _mm256_set1_ps(2.0f); // Set scalar to 2.0f
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&Axb2[i]); 
        _mm256_store_ps(&g[i], _mm256_mul_ps(vecData, scalar));  // Multiply and store
    }

    // Process remaining elements
    for (; i < n; ++i) {
        g[i] = Axb2[i] * 2.0f;
    }
}

inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n) {
    __m256 factor = _mm256_set1_ps(0.0f);
    int i = 0;
    // Process multiples of 8
    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&x_copy[i], vecData);  // Copy to x_copy
        _mm256_store_ps(&Axb2_vec[i], factor); // Set Axb2_vec to 0
    }

    // Process remaining elements
    for (; i < n; ++i) {
        x_copy[i] = x[i];
        Axb2_vec[i] = 0.0f;
    }
}


// here we are basically evaluating the objective function
// as well as evaluating the error
// looks very unreadable because I tried to optimize it as much as possible
// DCTs are the limiting performance factor
float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
)
{
    float fx = 0;
    const bool sp = g_solveprof.on;
    auto sp_t = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    copy_x(data.x_copy, (float*)x, data.Axb2, n);
    cv::Mat Ax(data.rows, data.cols, CV_32F, data.x_copy);
    dct(Ax, Ax, cv::DCT_INVERSE);
    if (sp) { g_solveprof.owl_eval_idct_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
    updateAxb2AndComputeFx(data.x_copy, data.ri_x, data.ri_y, data.Axb2, data.b, data.cols, fx, data.m);
    if (sp) { g_solveprof.owl_eval_data_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
    cv::Mat Axb2(data.rows, data.cols, CV_32F, data.Axb2);
    dct(Axb2, Axb2);
    eval_g(data.Axb2, g, n);
    if (sp) { g_solveprof.owl_eval_dct_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }

    // Optional total-variation fusion (smoothed isotropic TV on the
    // pixel-domain plane, which data.x_copy holds at this point):
    //   fx += tv_lambda * sum(sqrt(dx^2 + dy^2 + eps^2) - eps)
    //   gradient back-projected through the DCT and added to g.
    // The gradient is accumulated into the Axb2 buffer, which is free after
    // eval_g consumed its DCT of the data residual.
    if (data.tv_lambda > 0.0f) {
        const int rows = data.rows, cols = data.cols;
        const float eps = 1e-3f; // normalized-domain smoothing (x in [0,1])
        float* xpix = data.x_copy;
        float* tvgrad = data.Axb2;
        float phi = 0.0f;
        std::memset(tvgrad, 0, sizeof(float) * n);
        for (int i = 0; i < rows; ++i) {
            const bool has_down = i + 1 < rows;
            for (int j = 0; j < cols; ++j) {
                const int idx = i * cols + j;
                const float a = has_down ? xpix[idx + cols] - xpix[idx] : 0.0f;
                const float b = (j + 1 < cols) ? xpix[idx + 1] - xpix[idx] : 0.0f;
                const float d = std::sqrt(a * a + b * b + eps * eps);
                phi += d - eps;
                const float ua = a / d;
                const float vb = b / d;
                // phi = sum(sqrt(a^2+b^2)); dphi/dx[p,q] = -div(u, v)
                tvgrad[idx] -= ua + vb;
                if (has_down) {
                    tvgrad[idx + cols] += ua;
                }
                if (j + 1 < cols) {
                    tvgrad[idx + 1] += vb;
                }
            }
        }
        cv::Mat tvgrad_m(data.rows, data.cols, CV_32F, tvgrad);
        dct(tvgrad_m, tvgrad_m);
        for (int i = 0; i < n; ++i) {
            g[i] += data.tv_lambda * tvgrad[i];
        }
        fx += data.tv_lambda * phi;
    }
    // optional TV tail folded into the dct bucket (tv=0 in profile runs)
    if (sp) { g_solveprof.owl_eval_dct_ns += sp_ns_since(sp_t); g_solveprof.owl_evals += 1; }

    return fx;
}

// prints out convergence metrics with every iterations
// this is more for debugging purposes, it's not necesarry to be called
int progress(
    void* instance,
    const float* x,
    const float* g,
    const float fx,
    const float xnorm,
    const float gnorm,
    const float step,
    int n,
    int k,
    int ls
)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");

    return 0;
}

// this function creates initial solutions for each color channel
// we use a generic reference image to create them
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols) {
    cv::Mat ref = cv::imread("ref.png", cv::IMREAD_COLOR);

    // the warm-start asset is optional; fall back to neutral gray if missing
    // instead of crashing inside cv::resize on an empty Mat
    if (ref.empty()) {
        ref = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128));
    }
    else {
        // resizing to accomodate the size of the tiles
        cv::resize(ref, ref, cv::Size(rows, cols));
    }

    std::vector<cv::Mat> c;
    cv::split(ref, c);

    for (int i = 0; i < 3; i++) {
        c[i].convertTo(c[i], CV_32F);
        c[i] = c[i] / 255.0f;
        cv::dct(c[i], c[i], 0);
        c[i] = c[i] / 10.0f;
    }

    return c;
}

// reconstructs a color channel using LBFGS
void reconstruct_color_channel(const cv::Mat& pixel_measurements, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref, float tv) {

    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    int n = rows * cols; // size of solution (size of vectorized image)
    float fx;
    /* Initialize the parameters for the optimization. */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // this tells lbfgs to do OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;
    int lbfgs_ret;
    std::vector<float> b;

    // reserving space to avoid realocations
    b.reserve(ri_x.size());

    //auto update_progress = progress;
    lbfgs_progress_t update_progress = NULL;
    eval_data data;
    std::vector<float> Axb2(n);
    std::vector<float> x_copy(n);

    // extracting pixel measurements from encrypted image
    for (int i = CS_HEADER_PIXELS; i < ri_x.size() + CS_HEADER_PIXELS && i < pixel_measurements.total(); i++) {
        b.push_back(pixel_measurements.at<cv::Vec3b>(i)[k] / 255.0f);
    }

    // sometimes the number of sampled pixels in a tile wont be exactly ri_x.size()
    // so we just make the rest of them 0
    for (int i = b.size(); i < ri_x.size(); i++) {
        b.push_back(0.0f);
    }

    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.m = ri_x.size();
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = rows;
    data.cols = cols;
    data.tv_lambda = tv;

    // LBFGS optimization
    if (sp) {
        g_solveprof.owl_setup_ns += sp_ns_since(sp_t0);
        g_solveprof.wrap_calls += 1;
        g_solveprof.samples += (long long)ri_x.size();
    }
    auto sp_l = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    lbfgs_ret = lbfgs(n, (float*)ref.data, data, &fx, evaluate, update_progress, NULL, &param);
    if (sp) g_solveprof.owl_lbfgs_ns += sp_ns_since(sp_l);

    // we are copying the current solution to the next solution for faster convergence.
    // NOTE: this used to stride i in bytes against total() (an element count),
    // copying only the first quarter of the float plane; a full memcpy carries
    // the whole solved plane (including correlated high frequencies) forward.
    auto sp_t1 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (copy_next_ref && !next_ref.empty()) {
        if (!next_ref.isContinuous()) next_ref = next_ref.clone();
        std::memcpy(next_ref.data, ref.data, sizeof(float) * (size_t)n);
    }

    cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref.data);
    dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
    AtAxb2 = AtAxb2 * 255.0f;
    if (sp) g_solveprof.owl_tail_ns += sp_ns_since(sp_t1);
}

// ---------------------------------------------------------------------------
// FISTA + reweighted L1 + SOMP-structured joint sparsity
// ---------------------------------------------------------------------------
// Alternative to the OWL-QN (liblbfgs) path in reconstruct_color_channel.
// Under DCT the forward operator A = P * IDCT is a row-selected orthonormal
// transform, so ||A|| = 1 and the data term f(x) = ||Ax - b||^2 has
// Lipschitz constant L = 2 exactly (plus ~8*tv when the smoothed-TV fusion
// is enabled). That makes FISTA's proximal step exact and cheap: one IDCT +
// one DCT per iteration, no line search, versus several evaluate() calls per
// OWL-QN step. Under CDF97 (biorthogonal, multilevel) the step is chosen by
// backtracking instead. Reweighting (Candes et al.) runs 2 outer passes with normalized
// weights w = eps/(|x|+eps) so large coefficients are protected while small
// ones are pushed harder toward zero. The joint mode replaces the
// per-channel independent solves with one group-L2,1 FISTA over the stacked
// [R|G|B] planes: a single DCT support shared across channels, which is the
// convex (and tractable) form of SOMP-style simultaneous greedy selection.
// A literal greedy SOMP over ~170k DCT atoms per tile would need a full DCT
// per candidate atom and is infeasible on CPU; group thresholding converges
// to the same joint-support structure at the cost of one extra DCT per
// channel per iteration.

int cs_solver_from_name(const std::string& name, int& out) {
    std::string s;
    s.reserve(name.size());
    for (char c : name) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        s.push_back(c);
    }
    if (s == "owlqn" || s == "lbfgs" || s == "owl-qn" || s == "0") { out = CS_SOLVER_OWLQN; return 0; }
    if (s == "fista" || s == "1") { out = CS_SOLVER_FISTA; return 0; }
    if (s == "joint" || s == "fista-joint" || s == "fistajoint" || s == "somp" || s == "2") { out = CS_SOLVER_FISTA_JOINT; return 0; }
    if (s == "admm" || s == "3") { out = CS_SOLVER_ADMM; return 0; }
    return -1;
}

// Smoothed isotropic TV energy + gradient in the pixel domain (same
// convention as evaluate()). tvgrad must hold n floats; returns phi.
static float cs_tv_grad_phi(const float* pix, float* tvgrad, int rows, int cols) {
    const int n = rows * cols;
    const float eps = 1e-3f;
    std::memset(tvgrad, 0, sizeof(float) * (size_t)n);
    float phi = 0.0f;
    for (int i = 0; i < rows; ++i) {
        const bool has_down = i + 1 < rows;
        for (int j = 0; j < cols; ++j) {
            const int idx = i * cols + j;
            const float a = has_down ? pix[idx + cols] - pix[idx] : 0.0f;
            const float bb = (j + 1 < cols) ? pix[idx + 1] - pix[idx] : 0.0f;
            const float d = std::sqrt(a * a + bb * bb + eps * eps);
            phi += d - eps;
            const float ua = a / d;
            const float vb = bb / d;
            tvgrad[idx] -= ua + vb;
            if (has_down) tvgrad[idx + cols] += ua;
            if (j + 1 < cols) tvgrad[idx + 1] += vb;
        }
    }
    return phi;
}

// Basis-aware synthesis: coefficient domain -> pixel plane (in place).
inline void cs_synth(float* p, int rows, int cols, const cs_fista_basis& bx) {
    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_inverse(p, rows, cols, bx.levels);
    } else {
        cv::Mat P(rows, cols, CV_32F, p);
        cv::dct(P, P, cv::DCT_INVERSE);
    }
}

// Adjoint synthesis: pixel/residual plane -> coefficient-domain gradient
// (in place). DCT is orthogonal (adjoint = forward DCT); CDF97 uses the true
// adjoint of the lifting synthesis — the forward analysis DWT would be the
// WRONG operator here (biorthogonality) and freezes the solve.
inline void cs_synth_adj(float* p, int rows, int cols, const cs_fista_basis& bx) {
    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_synth_adjoint(p, rows, cols, bx.levels);
    } else {
        cv::Mat P(rows, cols, CV_32F, p);
        cv::dct(P, P, 0);
    }
}

// Gradient of the smooth part in the coefficient domain:
//   g = 2*S^T(scatter(S(y) - b)) [+ tv*S^T(tvgrad(S(y)))] with S =
//   synthesis (IDCT/IDWT). pix/sc are scratch (size n). Returns the
//   smooth-objective value f(y) = ||Ax-b||^2 (+ tv*TV) for backtracking.
static float cs_fista_grad(const float* y, float* pix, float* sc, float* g,
    const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, float tv_lambda, const cs_fista_basis& bx)
{
    const int n = rows * cols;
    std::memcpy(pix, y, sizeof(float) * (size_t)n);
    cs_synth(pix, rows, cols, bx);
    std::memset(sc, 0, sizeof(float) * (size_t)n);
    float fx = 0.0f;
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * cols + riy[k];
        const float diff = pix[idx] - b[k];
        sc[idx] = diff;
        fx += diff * diff;
    }
    cs_synth_adj(sc, rows, cols, bx);
    for (int i = 0; i < n; ++i) g[i] = 2.0f * sc[i];

    if (tv_lambda > 0.0f) {
        float* tvgrad = sc; // scatter buffer is free now (g holds the data grad)
        const float phi = cs_tv_grad_phi(pix, tvgrad, rows, cols);
        cs_synth_adj(tvgrad, rows, cols, bx);
        for (int i = 0; i < n; ++i) g[i] += tv_lambda * tvgrad[i];
        fx += tv_lambda * phi;
    }
    return fx;
}

// Smooth-objective value f(xc) for a candidate (backtracking trials).
// pix/sc are scratch (size n).
static float cs_smooth_fx(const float* xc, float* pix, float* sc,
    const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, float tv_lambda, const cs_fista_basis& bx)
{
    const int n = rows * cols;
    std::memcpy(pix, xc, sizeof(float) * (size_t)n);
    cs_synth(pix, rows, cols, bx);
    float fx = 0.0f;
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * cols + riy[k];
        const float diff = pix[idx] - b[k];
        fx += diff * diff;
    }
    if (tv_lambda > 0.0f) {
        fx += tv_lambda * cs_tv_grad_phi(pix, sc, rows, cols);
    }
    return fx;
}

// One weighted-L1 FISTA run: min ||Ax-b||^2 (+ tv*TV) + lambda*sum(w|x|).
// x is the warm start in, solution out. y/x_prev/grad/pix/sc are scratch.
// DCT uses the exact fixed step (previous behavior, bit-identical); CDF97
// (biorthogonal, no exact Lipschitz) runs FISTA-with-backtracking: L grows
// until the quadratic upper bound holds, which guarantees convergence for
// any starting L.
static void cs_fista_core_single(float* x, const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, const float* w, float lambda, float tv_lambda, int iters,
    const cs_fista_basis& bx,
    std::vector<float>& y, std::vector<float>& x_prev,
    std::vector<float>& grad, std::vector<float>& pix, std::vector<float>& sc)
{
    const int n = rows * cols;
    const bool bt = (bx.basis == CS_BASIS_CDF97);
    const bool sp = g_solveprof.on;
    float L = bx.lip + 8.0f * tv_lambda;
    std::memcpy(y.data(), x, sizeof(float) * (size_t)n);
    std::memcpy(x_prev.data(), x, sizeof(float) * (size_t)n);
    std::vector<float> z((size_t)n);
    float* yy = y.data();
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        auto sp_t = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const float fy = cs_fista_grad(yy, pix.data(), sc.data(), grad.data(), b, rix, riy, m, rows, cols, tv_lambda, bx);
        if (sp) { g_solveprof.fs_grad_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
        float step = 1.0f / L;
        if (bt) {
            // backtracking: shrink step until f(z) <= Q_L(z, y)
            for (int trial = 0; ; ++trial) {
                step = 1.0f / L;
                const float base = lambda * step;
                for (int i = 0; i < n; ++i) {
                    const float v = yy[i] - step * grad[i];
                    const float thr = base * w[i] * bx.wscale[i];
                    const float av = std::fabs(v);
                    z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
                }
                const float fz = cs_smooth_fx(z.data(), pix.data(), sc.data(), b, rix, riy, m, rows, cols, tv_lambda, bx);
                double dot = 0.0, dz2 = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double dz = (double)z[(size_t)i] - yy[i];
                    dot += (double)grad[i] * dz;
                    dz2 += dz * dz;
                }
                const double Q = (double)fy + dot + 0.5 * (double)L * dz2;
                if ((double)fz <= Q + 1e-7 * (1.0 + std::fabs((double)fy)) || trial >= 24) break;
                L *= 2.0f;
            }
        } else {
            const float base = lambda * step;
            for (int i = 0; i < n; ++i) {
                const float v = yy[i] - step * grad[i];
                const float thr = base * w[i] * bx.wscale[i];
                const float av = std::fabs(v);
                z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
            }
        }
        if (sp) { g_solveprof.fs_shrink_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
        const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float mom = (t - 1.0f) / t_new;
        for (int i = 0; i < n; ++i) yy[i] = z[(size_t)i] + mom * (z[(size_t)i] - x_prev[(size_t)i]);
        std::memcpy(x_prev.data(), z.data(), sizeof(float) * (size_t)n);
        t = t_new;
        if (sp) { g_solveprof.fs_mom_ns += sp_ns_since(sp_t); g_solveprof.fs_iters += 1; }
    }
    std::memcpy(x, z.data(), sizeof(float) * (size_t)n);
}

// One weighted group-L2,1 FISTA run over stacked [x0|x1|x2]:
//   min sum_c ||A xc - bc||^2 (+ tv*TV each) + lambda*sum_i w_i*||row_i||_2.
// Row norms couple the channels, so surviving atoms are shared (joint
// support, SOMP-structured). Buffers y/xp/grad/z hold 3n floats. DCT uses
// the exact fixed step; CDF97 backtracks (see cs_fista_core_single).
static void cs_fista_core_joint(float* x0, float* x1, float* x2,
    const float* b0, const float* b1, const float* b2,
    const int* rix, const int* riy, int m, int rows, int cols,
    const float* w, float lambda, float tv_lambda, int iters,
    const cs_fista_basis& bx,
    std::vector<float>& y, std::vector<float>& xp,
    std::vector<float>& grad, std::vector<float>& z,
    std::vector<float>& pix, std::vector<float>& sc)
{
    const int n = rows * cols;
    const bool bt = (bx.basis == CS_BASIS_CDF97);
    float L = bx.lip + 8.0f * tv_lambda;
    float* xx[3] = { x0, x1, x2 };
    const float* bb[3] = { b0, b1, b2 };
    float* yc[3] = { y.data(), y.data() + n, y.data() + 2 * n };
    float* pc[3] = { xp.data(), xp.data() + n, xp.data() + 2 * n };
    float* gc[3] = { grad.data(), grad.data() + n, grad.data() + 2 * n };
    float* zc[3] = { z.data(), z.data() + n, z.data() + 2 * n };
    for (int c = 0; c < 3; ++c) {
        std::memcpy(yc[c], xx[c], sizeof(float) * (size_t)n);
        std::memcpy(pc[c], xx[c], sizeof(float) * (size_t)n);
    }
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        float fy = 0.0f;
        for (int c = 0; c < 3; ++c) {
            fy += cs_fista_grad(yc[c], pix.data(), sc.data(), gc[c], bb[c], rix, riy, m, rows, cols, tv_lambda, bx);
        }
        if (bt) {
            for (int trial = 0; ; ++trial) {
                const float step = 1.0f / L;
                const float base = lambda * step;
                for (int i = 0; i < n; ++i) {
                    const float v0 = yc[0][i] - step * gc[0][i];
                    const float v1 = yc[1][i] - step * gc[1][i];
                    const float v2 = yc[2][i] - step * gc[2][i];
                    const float rn = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
                    const float thr = base * w[i] * bx.wscale[i];
                    const float s = (rn > thr && rn > 0.0f) ? (1.0f - thr / rn) : 0.0f;
                    zc[0][i] = v0 * s; zc[1][i] = v1 * s; zc[2][i] = v2 * s;
                }
                float fz = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    fz += cs_smooth_fx(zc[c], pix.data(), sc.data(), bb[c], rix, riy, m, rows, cols, tv_lambda, bx);
                }
                double dot = 0.0, dz2 = 0.0;
                for (int c = 0; c < 3; ++c) {
                    for (int i = 0; i < n; ++i) {
                        const double dz = (double)zc[c][i] - yc[c][i];
                        dot += (double)gc[c][i] * dz;
                        dz2 += dz * dz;
                    }
                }
                const double Q = (double)fy + dot + 0.5 * (double)L * dz2;
                if ((double)fz <= Q + 1e-7 * (1.0 + std::fabs((double)fy)) || trial >= 24) break;
                L *= 2.0f;
            }
        } else {
            const float step = 1.0f / L;
            const float base = lambda * step;
            for (int i = 0; i < n; ++i) {
                const float v0 = yc[0][i] - step * gc[0][i];
                const float v1 = yc[1][i] - step * gc[1][i];
                const float v2 = yc[2][i] - step * gc[2][i];
                const float rn = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
                const float thr = base * w[i] * bx.wscale[i];
                const float s = (rn > thr && rn > 0.0f) ? (1.0f - thr / rn) : 0.0f;
                zc[0][i] = v0 * s; zc[1][i] = v1 * s; zc[2][i] = v2 * s;
            }
        }
        const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float mom = (t - 1.0f) / t_new;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < n; ++i) {
                yc[c][i] = zc[c][i] + mom * (zc[c][i] - pc[c][i]);
                pc[c][i] = zc[c][i];
            }
        }
        t = t_new;
    }
    for (int c = 0; c < 3; ++c) {
        std::memcpy(xx[c], zc[c], sizeof(float) * (size_t)n);
    }
}

// FISTA needs more steps than OWL-QN (no Hessian); map the L-BFGS-scale
// iteration budget to a per-reweight FISTA count. Total DCT pairs stay
// within ~3-5x of an OWL-QN solve at the same setting. A positive
// fista_iters override (CLI --fista-iters) bypasses the map and sets the
// per-pass step count directly.
static int cs_fista_inner_iters(int iterations, int fista_iters) {
    if (fista_iters > 0) {
        if (fista_iters > 500) fista_iters = 500;
        return fista_iters;
    }
    int inner = iterations * 4;
    if (inner < 16) inner = 16;
    if (inner > 40) inner = 40;
    return inner;
}

static void cs_extract_channel_measurements(const cv::Mat& pixel_measurements, int channel,
    int m, std::vector<float>& b)
{
    b.assign((size_t)m, 0.0f);
    const int avail = (int)pixel_measurements.total() - CS_HEADER_PIXELS;
    const int cnt = avail < m ? (avail < 0 ? 0 : avail) : m;
    for (int i = 0; i < cnt; ++i) {
        b[(size_t)i] = pixel_measurements.at<cv::Vec3b>(i + CS_HEADER_PIXELS)[channel] / 255.0f;
    }
}

void reconstruct_color_channel_fista(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref,
    float tv, int reweights, int fista_iters, int basis, float wscale)
{
    if (!ref.isContinuous()) ref = ref.clone();
    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    const cs_fista_basis bx = cs_make_basis(rows, cols, basis, wscale);

    std::vector<float> b;
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);

    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> y((size_t)n), x_prev((size_t)n), grad((size_t)n), pix((size_t)n), sc((size_t)n);
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    float* x = (float*)ref.data;
    if (sp) g_solveprof.wrap_setup_ns += sp_ns_since(sp_t0);
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97) {
        // GPU path: batched with concurrent callers inside cs_gpu; the tail
        // (ref chain copy + IDCT + 255) stays on the CPU below, unchanged.
        // tv rides per-problem: 0 takes the sampled fast path, >0 the TV
        // branch (same cs_fista_grad math as the CPU core).
        cs_gpu::FistaProblem pb;
        pb.tv = tv;
        pb.b = b.data();
        pb.rix = ri_x.data();
        pb.riy = ri_y.data();
        pb.m = m;
        pb.rows = rows;
        pb.cols = cols;
        pb.x = x;
        pb.lambda = param_c;
        pb.inner = inner;
        pb.reweights = reweights;
        auto sp_g = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        cs_gpu::fista_solve_blocking(pb);
        if (sp) {
            g_solveprof.wrap_core_ns += sp_ns_since(sp_g);
            g_solveprof.rw_passes += (reweights > 1) ? (reweights - 1) : 0;
        }
    } else for (int r = 0; r < reweights; ++r) {
        auto sp_c = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        cs_fista_core_single(x, b.data(), ri_x.data(), ri_y.data(), m, rows, cols,
            w.data(), param_c, tv, inner, bx, y, x_prev, grad, pix, sc);
        if (sp) g_solveprof.wrap_core_ns += sp_ns_since(sp_c);
        if (r + 1 < reweights) {
            auto sp_w = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float av = std::fabs(x[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i) w[(size_t)i] = eps / (std::fabs(x[i]) + eps);
            if (sp) { g_solveprof.wrap_rw_ns += sp_ns_since(sp_w); g_solveprof.rw_passes += 1; }
        }
    }

    auto sp_t1 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (copy_next_ref && !next_ref.empty()) {
        if (!next_ref.isContinuous()) next_ref = next_ref.clone();
        std::memcpy(next_ref.data, ref.data, sizeof(float) * (size_t)n);
    }

    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_inverse((float*)ref.data, rows, cols, bx.levels);
        cv::Mat plane(rows, cols, CV_32F, ref.data);
        plane = plane * 255.0f;
    } else {
        cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref.data);
        dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
        AtAxb2 = AtAxb2 * 255.0f;
    }
    if (sp) {
        g_solveprof.wrap_tail_ns += sp_ns_since(sp_t1);
        g_solveprof.wrap_calls += 1;
        g_solveprof.samples += m;
    }
}

void reconstruct_image_fista_joint(const cv::Mat& pixel_measurements, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat refs[3], float tv, int reweights, int fista_iters, int basis, float wscale)
{
    for (int ch = 0; ch < 3; ++ch) {
        if (!refs[ch].isContinuous()) refs[ch] = refs[ch].clone();
    }
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    const cs_fista_basis bx = cs_make_basis(rows, cols, basis, wscale);

    std::vector<float> b0, b1, b2;
    cs_extract_channel_measurements(pixel_measurements, 0, m, b0);
    cs_extract_channel_measurements(pixel_measurements, 1, m, b1);
    cs_extract_channel_measurements(pixel_measurements, 2, m, b2);

    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> y((size_t)3 * n), xp((size_t)3 * n), grad((size_t)3 * n), z((size_t)3 * n);
    std::vector<float> pix((size_t)n), sc((size_t)n);
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    float* x0 = (float*)refs[0].data;
    float* x1 = (float*)refs[1].data;
    float* x2 = (float*)refs[2].data;
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97) {
        // GPU path: stacked group solve inside cs_gpu (same batching as the
        // single-channel path); tails below stay on the CPU, unchanged.
        cs_gpu::FistaProblem pb;
        pb.tv = tv;
        pb.b = b0.data();
        pb.rix = ri_x.data();
        pb.riy = ri_y.data();
        pb.m = m;
        pb.rows = rows;
        pb.cols = cols;
        pb.x = x0;
        pb.lambda = param_c;
        pb.inner = inner;
        pb.reweights = reweights;
        pb.b1 = b1.data();
        pb.b2 = b2.data();
        pb.x1 = x1;
        pb.x2 = x2;
        pb.joint = 1;
        cs_gpu::fista_solve_blocking(pb);
    } else for (int r = 0; r < reweights; ++r) {
        cs_fista_core_joint(x0, x1, x2, b0.data(), b1.data(), b2.data(),
            ri_x.data(), ri_y.data(), m, rows, cols, w.data(), param_c, tv, inner, bx,
            y, xp, grad, z, pix, sc);
        if (r + 1 < reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float rn = std::sqrt(x0[i] * x0[i] + x1[i] * x1[i] + x2[i] * x2[i]);
                if (rn > mx) mx = rn;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i) {
                const float rn = std::sqrt(x0[i] * x0[i] + x1[i] * x1[i] + x2[i] * x2[i]);
                w[(size_t)i] = eps / (rn + eps);
            }
        }
    }

    for (int ch = 0; ch < 3; ++ch) {
        if (bx.basis == CS_BASIS_CDF97) {
            cs_dwt_inverse((float*)refs[ch].data, rows, cols, bx.levels);
            cv::Mat plane(rows, cols, CV_32F, refs[ch].data);
            plane = plane * 255.0f;
        } else {
            cv::Mat plane(rows, cols, CV_32F, refs[ch].data);
            dct(plane, plane, cv::DCT_INVERSE);
            plane = plane * 255.0f;
        }
    }
}

// One consensus-ADMM run: min f(x) + g(z) s.t. x = z with
//   f(x) = ||P(S x) - b||^2 (+ tv*TV(S x)),  g(z) = lambda*sum(w*wscale*|z|).
// The x-update keeps the data term exact (quadratic) and linearizes TV at
// x^k with the (tau/2)||x - x^k||^2 stabilizer (tau = 8*tv, the same TV
// gradient bound FISTA uses), which gives the linear system
//   (2 S^T M S + sigma I) x = 2 S^T P^T b - tv*S^T gradTV(S x^k)
//                             + tau*x^k + rho*(z - u),   sigma = rho + tau.
// Orthonormal synthesis (DCT): S^T(2M + sigma*I)S equals that matrix, so the
// solve is closed form, x = S^T D^{-1} S rhs with D = 2M + sigma*I diagonal
// in the pixel domain (two transforms + one division).
// Biorthogonal CDF97: S^{-1} != S^T, so the identity fails; solve the same
// SPD system with CG instead (sigma > 0 bounds the condition number and the
// previous x warm-starts it; the relative-residual exit keeps the cost at a
// few matvecs once ADMM settles).
// z is the exact soft-threshold, u the scaled dual, and rho follows Boyd's
// residual balancing (check every step, mu = 10, tau = 2) so a fixed
// 16-40 step budget is not wasted on a badly scaled penalty. x is warm
// start in, solution out; z/u are reset from x at entry (each reweight pass
// starts a fresh consensus on the new weights).
static void cs_admm_core(float* x, const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, const float* w, float lambda, float tv_lambda, int iters,
    const cs_fista_basis& bx,
    std::vector<float>& z, std::vector<float>& u,
    std::vector<float>& rhs, std::vector<float>& pix,
    std::vector<float>& sc, std::vector<float>& cgp)
{
    const int n = rows * cols;
    const bool closed = (bx.basis == CS_BASIS_DCT);
    const bool sp = g_solveprof.on;
    const auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    // constant pieces of the x-update: Atb = 2*S^T P^T b and the diagonal
    // 2M of the data-term Hessian (2 per measured pixel, 0 elsewhere)
    std::vector<float> atb((size_t)n, 0.0f), mask2((size_t)n, 0.0f);
    std::memset(pix.data(), 0, sizeof(float) * (size_t)n);
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * cols + riy[k];
        pix[(size_t)idx] += b[k];
        mask2[(size_t)idx] += 2.0f;
    }
    cs_synth_adj(pix.data(), rows, cols, bx);
    for (int i = 0; i < n; ++i) atb[(size_t)i] = 2.0f * pix[(size_t)i];

    // initial penalty at the data-Hessian mean eigenvalue
    // (trace(2 S^T M S) = 2m over n dims) so the first x-update balances
    // measurement fit against consensus; residual balancing adjusts from there
    std::memcpy(z.data(), x, sizeof(float) * (size_t)n);
    std::memset(u.data(), 0, sizeof(float) * (size_t)n);
    float rho = 2.0f * (float)m / (float)n;
    if (rho < 0.05f) rho = 0.05f;

    const float tau = 8.0f * tv_lambda;
    std::vector<float> zprev((size_t)n);
    if (sp) g_solveprof.admm_setup_ns += sp_ns_since(sp_t0);
    for (int k = 0; k < iters; ++k) {
        auto sp_t = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const float sigma = rho + tau;

        // RHS = 2*Atb - tv*S^T gradTV(S x^k) + tau*x^k + rho*(z - u)
        for (int i = 0; i < n; ++i)
            rhs[(size_t)i] = atb[(size_t)i] + tau * x[i] + rho * (z[(size_t)i] - u[(size_t)i]);
        if (tv_lambda > 0.0f) {
            std::memcpy(pix.data(), x, sizeof(float) * (size_t)n);
            cs_synth(pix.data(), rows, cols, bx);
            cs_tv_grad_phi(pix.data(), sc.data(), rows, cols); // grad TV in pixels
            cs_synth_adj(sc.data(), rows, cols, bx);           // S^T gradTV
            for (int i = 0; i < n; ++i) rhs[(size_t)i] -= tv_lambda * sc[(size_t)i];
        }
        if (sp) { g_solveprof.admm_rhs_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }

        if (closed) {
            // x = S^T (2M + sigma*I)^{-1} S rhs  (S orthonormal)
            std::memcpy(pix.data(), rhs.data(), sizeof(float) * (size_t)n);
            cs_synth(pix.data(), rows, cols, bx);
            for (int i = 0; i < n; ++i) pix[(size_t)i] /= (mask2[(size_t)i] + sigma);
            cs_synth_adj(pix.data(), rows, cols, bx);
            std::memcpy(x, pix.data(), sizeof(float) * (size_t)n);
        } else {
            // CG on (2 S^T M S + sigma I) x = rhs, warm-started at x^k.
            // r lives in rhs afterwards (the RHS is not needed again), the
            // direction in cgp, Ap in sc, and pix is synthesis scratch.
            std::memcpy(pix.data(), x, sizeof(float) * (size_t)n);
            cs_synth(pix.data(), rows, cols, bx);
            for (int i = 0; i < n; ++i) pix[(size_t)i] *= mask2[(size_t)i];
            cs_synth_adj(pix.data(), rows, cols, bx);
            for (int i = 0; i < n; ++i) rhs[(size_t)i] -= pix[(size_t)i] + sigma * x[i];
            double rr = 0.0;
            for (int i = 0; i < n; ++i) {
                cgp[(size_t)i] = rhs[(size_t)i];
                rr += (double)rhs[(size_t)i] * rhs[(size_t)i];
            }
            const double rr0 = rr;
            for (int cg = 0; cg < 20 && rr > 1e-10 * (rr0 + 1e-30); ++cg) {
                std::memcpy(pix.data(), cgp.data(), sizeof(float) * (size_t)n);
                cs_synth(pix.data(), rows, cols, bx);
                for (int i = 0; i < n; ++i) pix[(size_t)i] *= mask2[(size_t)i];
                cs_synth_adj(pix.data(), rows, cols, bx);
                for (int i = 0; i < n; ++i) sc[(size_t)i] = pix[(size_t)i] + sigma * cgp[(size_t)i];
                double pAp = 0.0;
                for (int i = 0; i < n; ++i) pAp += (double)cgp[(size_t)i] * sc[(size_t)i];
                if (!(pAp > 0.0)) break;
                const double alpha = rr / pAp;
                for (int i = 0; i < n; ++i) {
                    x[i] += (float)alpha * cgp[(size_t)i];
                    rhs[(size_t)i] -= (float)alpha * sc[(size_t)i];
                }
                double rr_new = 0.0;
                for (int i = 0; i < n; ++i) rr_new += (double)rhs[(size_t)i] * rhs[(size_t)i];
                const float beta = (float)(rr_new / rr);
                for (int i = 0; i < n; ++i) cgp[(size_t)i] = rhs[(size_t)i] + beta * cgp[(size_t)i];
                rr = rr_new;
            }
        }
        if (sp) {
            if (closed) g_solveprof.admm_x_ns += sp_ns_since(sp_t);
            else g_solveprof.admm_cg_ns += sp_ns_since(sp_t);
            sp_t = std::chrono::steady_clock::now();
        }

        // z-update: exact soft-threshold of (x + u) at lambda*w*wscale/rho
        std::memcpy(zprev.data(), z.data(), sizeof(float) * (size_t)n);
        for (int i = 0; i < n; ++i) {
            const float v = x[i] + u[i];
            const float thr = lambda * w[i] * bx.wscale[(size_t)i] / rho;
            const float av = std::fabs(v);
            z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
        }
        if (sp) { g_solveprof.admm_z_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }

        // scaled-dual update + residual norms for balancing
        double rnorm2 = 0.0, snorm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            u[(size_t)i] += x[i] - z[(size_t)i];
            const double r = (double)x[i] - z[(size_t)i];
            const double s = (double)rho * (z[(size_t)i] - zprev[(size_t)i]);
            rnorm2 += r * r;
            snorm2 += s * s;
        }

        // Boyd residual balancing: grow/shrink rho by 2x and rescale the
        // scaled dual to keep y = rho*u fixed
        const double rnorm = std::sqrt(rnorm2), snorm = std::sqrt(snorm2);
        if (rnorm > 10.0 * snorm && rho < 100.0f) {
            const float old = rho;
            rho *= 2.0f;
            const float s = old / rho;
            for (int i = 0; i < n; ++i) u[(size_t)i] *= s;
        }
        else if (snorm > 10.0 * rnorm && rho > 1e-3f) {
            const float old = rho;
            rho /= 2.0f;
            const float s = old / rho;
            for (int i = 0; i < n; ++i) u[(size_t)i] *= s;
        }
        if (sp) { g_solveprof.admm_dual_ns += sp_ns_since(sp_t); g_solveprof.admm_iters += 1; }
    }
}

// CPU re-execution of the DCT/tv=0 ADMM wrapper contract for the GPU worker:
// runs when the device fails mid-batch. The problem arrives with inner and
// reweights already clamped, so this mirrors the wrapper loop exactly.
static void cs_gpu_cpu_fallback(const cs_gpu::AdmmProblem& p)
{
    const int n = p.rows * p.cols;
    const cs_fista_basis bx = cs_make_basis(p.rows, p.cols, CS_BASIS_DCT, 1.0f);
    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> z((size_t)n), u((size_t)n), rhs((size_t)n),
        pix((size_t)n), sc((size_t)n), cgp((size_t)n);
    for (int r = 0; r < p.reweights; ++r) {
        cs_admm_core(p.x, p.b, p.rix, p.riy, p.m, p.rows, p.cols, w.data(),
            p.lambda, 0.0f, p.inner, bx, z, u, rhs, pix, sc, cgp);
        if (r + 1 < p.reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float av = std::fabs(p.x[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i)
                w[(size_t)i] = eps / (std::fabs(p.x[i]) + eps);
        }
    }
}

static const struct CsGpuFallbackReg {
    CsGpuFallbackReg() { cs_gpu::register_cpu_fallback(&cs_gpu_cpu_fallback); }
} cs_gpu_fallback_reg;

// CPU re-execution of the DCT FISTA wrapper contract for the GPU
// worker: runs when the device fails mid-batch. The problem arrives with
// inner and reweights already mapped, so this mirrors the wrapper loop
// exactly (same scratch shapes, same reweight schedule).
static void cs_gpu_cpu_fallback_fista(const cs_gpu::FistaProblem& p)
{
    const int n = p.rows * p.cols;
    const cs_fista_basis bx = cs_make_basis(p.rows, p.cols, CS_BASIS_DCT, 1.0f);
    if (p.joint == 1 && p.b1 && p.b2 && p.x1 && p.x2) {
        // stacked group solve, same contract as reconstruct_image_fista_joint
        std::vector<float> w((size_t)n, 1.0f);
        std::vector<float> y((size_t)3 * n), xp((size_t)3 * n),
            grad((size_t)3 * n), z((size_t)3 * n);
        std::vector<float> pix((size_t)n), sc((size_t)n);
        for (int r = 0; r < p.reweights; ++r) {
            cs_fista_core_joint(p.x, p.x1, p.x2, p.b, p.b1, p.b2,
                p.rix, p.riy, p.m, p.rows, p.cols, w.data(), p.lambda, p.tv,
                p.inner, bx, y, xp, grad, z, pix, sc);
            if (r + 1 < p.reweights) {
                float mx = 0.0f;
                for (int i = 0; i < n; ++i) {
                    const float rn = std::sqrt(p.x[i] * p.x[i] +
                        p.x1[i] * p.x1[i] + p.x2[i] * p.x2[i]);
                    if (rn > mx) mx = rn;
                }
                float eps = 0.02f * mx;
                if (eps < 1e-3f) eps = 1e-3f;
                for (int i = 0; i < n; ++i) {
                    const float rn = std::sqrt(p.x[i] * p.x[i] +
                        p.x1[i] * p.x1[i] + p.x2[i] * p.x2[i]);
                    w[(size_t)i] = eps / (rn + eps);
                }
            }
        }
        return;
    }
    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> y((size_t)n), x_prev((size_t)n), grad((size_t)n),
        pix((size_t)n), sc((size_t)n);
    for (int r = 0; r < p.reweights; ++r) {
        cs_fista_core_single(p.x, p.b, p.rix, p.riy, p.m, p.rows, p.cols,
            w.data(), p.lambda, p.tv, p.inner, bx, y, x_prev, grad, pix, sc);
        if (r + 1 < p.reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float av = std::fabs(p.x[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i)
                w[(size_t)i] = eps / (std::fabs(p.x[i]) + eps);
        }
    }
}

static const struct CsGpuFallbackRegFista {
    CsGpuFallbackRegFista() { cs_gpu::register_cpu_fallback_fista(&cs_gpu_cpu_fallback_fista); }
} cs_gpu_fallback_reg_fista;

// Reweighted-L1 consensus ADMM for one channel. Same driver structure as
// reconstruct_color_channel_fista (outer IRLS reweights, same ref in/out
// convention), only the core optimizer differs.
void reconstruct_color_channel_admm(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref,
    float tv, int reweights, int admm_iters, int basis, float wscale)
{
    if (!ref.isContinuous()) ref = ref.clone();
    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    const cs_fista_basis bx = cs_make_basis(rows, cols, basis, wscale);

    std::vector<float> b;
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);

    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> z((size_t)n), u((size_t)n), rhs((size_t)n), pix((size_t)n), sc((size_t)n), cgp((size_t)n);
    const int inner = cs_fista_inner_iters(iterations, admm_iters);
    float* x = (float*)ref.data;
    if (sp) g_solveprof.wrap_setup_ns += sp_ns_since(sp_t0);
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97 && tv == 0.0f) {
        // GPU path: batched with concurrent callers inside cs_gpu; the tail
        // (ref chain copy + IDCT + 255) stays on the CPU below, unchanged.
        cs_gpu::AdmmProblem pb;
        pb.b = b.data();
        pb.rix = ri_x.data();
        pb.riy = ri_y.data();
        pb.m = m;
        pb.rows = rows;
        pb.cols = cols;
        pb.x = x;
        pb.lambda = param_c;
        pb.inner = inner;
        pb.reweights = reweights;
        auto sp_g = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        cs_gpu::admm_solve_blocking(pb);
        if (sp) {
            g_solveprof.wrap_core_ns += sp_ns_since(sp_g);
            g_solveprof.rw_passes += (reweights > 1) ? (reweights - 1) : 0;
        }
    } else for (int r = 0; r < reweights; ++r) {
        auto sp_c = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        cs_admm_core(x, b.data(), ri_x.data(), ri_y.data(), m, rows, cols,
            w.data(), param_c, tv, inner, bx, z, u, rhs, pix, sc, cgp);
        if (sp) g_solveprof.wrap_core_ns += sp_ns_since(sp_c);
        if (r + 1 < reweights) {
            auto sp_w = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float av = std::fabs(x[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i) w[(size_t)i] = eps / (std::fabs(x[i]) + eps);
            if (sp) { g_solveprof.wrap_rw_ns += sp_ns_since(sp_w); g_solveprof.rw_passes += 1; }
        }
    }

    auto sp_t1 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (copy_next_ref && !next_ref.empty()) {
        if (!next_ref.isContinuous()) next_ref = next_ref.clone();
        std::memcpy(next_ref.data, ref.data, sizeof(float) * (size_t)n);
    }

    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_inverse((float*)ref.data, rows, cols, bx.levels);
        cv::Mat plane(rows, cols, CV_32F, ref.data);
        plane = plane * 255.0f;
    } else {
        cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref.data);
        dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
        AtAxb2 = AtAxb2 * 255.0f;
    }
    if (sp) {
        g_solveprof.wrap_tail_ns += sp_ns_since(sp_t1);
        g_solveprof.wrap_calls += 1;
        g_solveprof.samples += m;
    }
}

float evaluate_coarse(void* instance, const float* x, eval_data data, float* g, const int n, const float step)
{
    float fx = 0;

    // x: coarse-plane DCT coefficients -> pixel plane (coarse)
    copy_x(data.x_copy, (float*)x, data.Axb2, n);
    cv::Mat C(data.rows, data.cols, CV_32F, data.x_copy);
    cv::dct(C, C, cv::DCT_INVERSE);

    // forward operator: upsample the coarse chroma plane to the full tile
    // grid and gather the scattered full-res measurements there
    cv::Mat full;
    cv::resize(C, full, cv::Size(data.full_cols, data.full_rows), 0, 0, cv::INTER_LINEAR);

    const int mm = data.m;
    cv::Mat R(data.full_rows, data.full_cols, CV_32F, cv::Scalar(0));
    float* rp = (float*)R.data;
    for (int k = 0; k < mm; ++k) {
        const int r = data.ri_x[k], c = data.ri_y[k];
        const float diff = full.at<float>(r, c) - data.b[k];
        R.at<float>(r, c) = diff;
        fx += diff * diff;
    }

    // gradient: 2x2 area-average the full-res residual back to the coarse
    // grid, then DCT it and scale
    cv::resize(R, C, cv::Size(data.cols, data.rows), 0, 0, cv::INTER_AREA);
    cv::dct(C, C, 0);
    eval_g((float*)C.data, g, n);

    return fx;
}

void reconstruct_color_channel_subchroma(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref)
{
    const int crows = (rows + 1) / 2, ccols = (cols + 1) / 2;
    const int nC = crows * ccols;
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;

    // measurements for this channel at the scattered full-res positions
    std::vector<float> b;
    b.reserve(ri_x.size());
    for (int i = CS_HEADER_PIXELS; i < (int)ri_x.size() + CS_HEADER_PIXELS && i < pixel_measurements.total(); i++) {
        b.push_back(pixel_measurements.at<cv::Vec3b>(i)[channel] / 255.0f);
    }
    for (int i = (int)b.size(); i < (int)ri_x.size(); i++) {
        b.push_back(0.0f);
    }

    eval_data data;
    std::vector<float> Axb2(nC), x_copy(nC);
    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.m = (int)ri_x.size();
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = crows;      // coarse unknown dims
    data.cols = ccols;
    data.full_rows = rows;  // full-res measurement grid
    data.full_cols = cols;

    float fx = 0.0f;
    lbfgs(nC, (float*)ref.data, data, &fx, evaluate_coarse, NULL, NULL, &param);

    // solved coarse DCT plane -> pixel plane -> upsample to the full tile size
    cv::Mat C(crows, ccols, CV_32F, ref.data);
    cv::dct(C, C, cv::DCT_INVERSE);
    C *= 255.0f;
    cv::resize(C, ref, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
}


