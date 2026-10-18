#include "helper_functions.hpp"
#include "cs_gpu.h"
#include "crypto_utils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#ifdef _WIN32
#include <onnxruntime_cxx_api.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

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
    g_solveprof.fs_grad_ns = 0; g_solveprof.fs_shrink_ns = 0;
    g_solveprof.fs_mom_ns = 0; g_solveprof.fs_iters = 0;
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
        " | fsta grad=%.1f shrink=%.1f mom=%.1f iters=%lld\n",
        num_tiles, num_threads, wall_ns * ms,
        g_solveprof.warm_ns.load() * ms, g_solveprof.ctor_ns.load() * ms, g_solveprof.call_ns.load() * ms,
        g_solveprof.wave_wall_ns.load() * ms, g_solveprof.wave_cap_ns.load() * ms, eff,
        g_solveprof.tiles.load(), g_solveprof.samples.load(),
        g_solveprof.wrap_setup_ns.load() * ms, g_solveprof.wrap_core_ns.load() * ms,
        g_solveprof.wrap_rw_ns.load() * ms, g_solveprof.wrap_tail_ns.load() * ms,
        g_solveprof.wrap_calls.load(), g_solveprof.rw_passes.load(),
        g_solveprof.fs_grad_ns.load() * ms, g_solveprof.fs_shrink_ns.load() * ms,
        g_solveprof.fs_mom_ns.load() * ms, g_solveprof.fs_iters.load());
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

// (OWL-QN helpers updateAxb2AndComputeFx/eval_g/copy_x removed with the
// solver; FISTA cores below are self-contained.)


// (OWL-QN objective evaluate() and its progress printer removed with the
// solver; FISTA cores below are self-contained.)

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
        cs_dct_forward(c[i], cs_default_dct_block());
        c[i] = c[i] / 10.0f;
    }

    return c;
}

// (OWL-QN channel solve reconstruct_color_channel removed; FISTA below is
// the only solver.)

// ---------------------------------------------------------------------------
// FISTA + reweighted L1 + SOMP-structured joint sparsity
// ---------------------------------------------------------------------------
// Under DCT the forward operator A = P * IDCT is a row-selected orthonormal
// transform, so ||A|| = 1 and the data term f(x) = ||Ax - b||^2 has
// Lipschitz constant L = 2 exactly (plus ~8*tv when the smoothed-TV fusion
// is enabled). That makes FISTA's proximal step exact and cheap: one IDCT +
// one DCT per iteration with no line search. Under CDF97 (biorthogonal, multilevel) the step is chosen by
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
    if (s == "fista" || s == "1") { out = CS_SOLVER_FISTA; return 0; }
    if (s == "joint" || s == "fista-joint" || s == "fistajoint" || s == "somp" || s == "2") { out = CS_SOLVER_FISTA_JOINT; return 0; }
    return -1;
}

// Smoothed isotropic TV energy + gradient in the pixel domain.
// Gather formulation: each output pixel is computed from the
// forward-difference cells that touch it (own + above + left), so every
// iteration writes only its own pixel -- no scatter-carried dependence,
// no memset, and ~40% fewer instructions than the scatter form
// (microbenched 0.100 vs 0.166 ms/call at 259x343; max diff 7e-7).
// tvgrad must hold n floats; returns phi.
static float cs_tv_grad_phi(const float* pix, float* tvgrad, int rows, int cols) {
    const int n = rows * cols;
    // Smoothing corner: the smoothed-TV gradient's local curvature scales as
    // tv/eps, but the DCT path steps with the fixed 1/(lip + 8*tv) assuming
    // curvature ~8*tv. With eps much smaller than that bound the step is
    // locally far too big in flat zones, and FISTA momentum sustains the
    // resulting Nyquist flip-flop as a checkerboard dot pattern (visible
    // only with tv > 0, only in smooth areas). eps = 1e-2 keeps the corner
    // at 1% intensity (real edges have far larger gradients) while cutting
    // the stiffness 10x vs 1e-3. Keep in sync with tvgrad_k in cs_gpu.hip.
    const float eps = 1e-2f;
    float phi = 0.0f;
    for (int i = 0; i < rows; ++i) {
        const bool has_down = i + 1 < rows;
        for (int j = 0; j < cols; ++j) {
            const int t = i * cols + j;
            const float c = pix[t];
            const float a0 = has_down ? pix[t + cols] - c : 0.0f;
            const float b0 = (j + 1 < cols) ? pix[t + 1] - c : 0.0f;
            const float d0 = std::sqrt(a0 * a0 + b0 * b0 + eps * eps);
            phi += d0 - eps;
            float acc = -(a0 / d0 + b0 / d0);
            if (i > 0) {
                const float up = pix[t - cols];
                const float aT = c - up;
                const float bT = (j + 1 < cols) ? pix[t - cols + 1] - up : 0.0f;
                const float dT = std::sqrt(aT * aT + bT * bT + eps * eps);
                acc += aT / dT;
            }
            if (j > 0) {
                const float lf = pix[t - 1];
                const float aL = has_down ? pix[t - 1 + cols] - lf : 0.0f;
                const float bL = c - lf;
                const float dL = std::sqrt(aL * aL + bL * bL + eps * eps);
                acc += bL / dL;
            }
            tvgrad[t] = acc;
        }
    }
    return phi;
}

float cs_auto_reg_coefficient(int sampling_mode, float ratio, int num_tiles) {
    if (cs_is_ycc(sampling_mode)) return -1.0f;
    float r = ratio < 0.05f ? 0.05f : (ratio > 1.0f ? 1.0f : ratio);
    const int nt = num_tiles > 0 ? num_tiles : 1;
    float base;
    // Tile-grid factor: small tiles hold fewer pixels per solve and need a
    // stronger prior. Seg needs it at every ratio (single global coef must
    // compromise across density variation: 0.46-0.6x at tiles=1); uniform
    // only when starved (ratio < 0.5), where it recovers ~1 dB at tiles=1.
    float tf = 1.0f;
    const bool segm = (sampling_mode == CS_MODE_SEGMASK);
    if (segm || r < 0.5f) {
        tf = std::pow((float)nt / 64.0f, 0.15f);
        if (tf < 0.5f) tf = 0.5f;
        if (tf > 1.2f) tf = 1.2f;
    }
    if (segm) {
        base = 0.045f * std::pow(r, -0.75f);
    } else if (sampling_mode == CS_MODE_RANDOM) {
        base = 0.01875f / r;
    } else {
        return -1.0f;
    }
    float coef = base * tf;
    if (coef < 0.01f) coef = 0.01f;
    if (coef > 0.3f) coef = 0.3f;
    return coef;
}

float cs_auto_reg_tv(int sampling_mode, float ratio) {
    if (cs_is_ycc(sampling_mode)) return -1.0f;
    if (sampling_mode == CS_MODE_SEGMASK) {
        return ratio < 0.5f ? 0.01f : 0.0f;
    }
    if (sampling_mode == CS_MODE_RANDOM) {
        return 0.0f;
    }
    return -1.0f;
}

// Block-aware 2D DCT (see helper_functions.hpp): whole-plane legacy at
// block <= 0, per-BxB orthonormal DCTs otherwise. Edge blocks clipped to
// the plane stay orthogonal (cv::dct handles arbitrary sizes), so forward
// remains the exact adjoint of inverse block-wise.
void cs_dct_forward(cv::Mat& plane, int dct_block) {
    if (dct_block <= 0) {
        cv::dct(plane, plane, 0);
        return;
    }
    for (int y0 = 0; y0 < plane.rows; y0 += dct_block) {
        const int bh = (std::min)(dct_block, plane.rows - y0);
        for (int x0 = 0; x0 < plane.cols; x0 += dct_block) {
            const int bw = (std::min)(dct_block, plane.cols - x0);
            cv::Mat roi = plane(cv::Rect(x0, y0, bw, bh));
            cv::dct(roi, roi, 0);
        }
    }
}

void cs_dct_inverse(cv::Mat& plane, int dct_block) {
    if (dct_block <= 0) {
        cv::dct(plane, plane, cv::DCT_INVERSE);
        return;
    }
    for (int y0 = 0; y0 < plane.rows; y0 += dct_block) {
        const int bh = (std::min)(dct_block, plane.rows - y0);
        for (int x0 = 0; x0 < plane.cols; x0 += dct_block) {
            const int bw = (std::min)(dct_block, plane.cols - x0);
            cv::Mat roi = plane(cv::Rect(x0, y0, bw, bh));
            cv::dct(roi, roi, cv::DCT_INVERSE);
        }
    }
}

// Basis-aware synthesis: coefficient domain -> pixel plane (in place).
inline void cs_synth(float* p, int rows, int cols, const cs_fista_basis& bx) {
    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_inverse(p, rows, cols, bx.levels);
    } else {
        cv::Mat P(rows, cols, CV_32F, p);
        cs_dct_inverse(P, bx.dct_block);
    }
}

// Adjoint synthesis: pixel/residual plane -> coefficient-domain gradient
// (in place). DCT is orthogonal (adjoint = forward DCT, block-wise too);
// CDF97 uses the true adjoint of the lifting synthesis — the forward
// analysis DWT would be the WRONG operator here (biorthogonality) and
// freezes the solve.
inline void cs_synth_adj(float* p, int rows, int cols, const cs_fista_basis& bx) {
    if (bx.basis == CS_BASIS_CDF97) {
        cs_dwt_synth_adjoint(p, rows, cols, bx.levels);
    } else {
        cv::Mat P(rows, cols, CV_32F, p);
        cs_dct_forward(P, bx.dct_block);
    }
}

// Gradient of the smooth part in the coefficient domain:
//   g = 2*S^T(scatter(S(y) - b)) [+ tv*S^T(tvgrad(S(y)))] with S =
//   synthesis (IDCT/IDWT). pix/sc are scratch (size n). Returns the
//   smooth-objective value f(y) = ||Ax-b||^2 (+ tv*TV) for backtracking.
static float cs_fista_grad(const float* y, float* pix, float* sc, float* g,
    const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, float tv_lambda, const cs_fista_basis& bx,
    const float* zcorr = nullptr, float oc = 0.0f, float* zsave = nullptr,
    const cs_meas_groups* groups = nullptr)
{
    const int n = rows * cols;
    std::memcpy(pix, y, sizeof(float) * (size_t)n);
    cs_synth(pix, rows, cols, bx);
    std::memset(sc, 0, sizeof(float) * (size_t)n);
    float fx = 0.0f;
    if (groups && groups->ng > 0) {
        // Averaged measurements: b[k] is the k-th group mean; predict it
        // from its weighted pixels, scatter w*residual back (adjoint of
        // the normalized averaging rows). Groups are disjoint by
        // construction; += keeps it correct regardless.
        for (int k = 0; k < groups->ng; ++k) {
            float pred = 0.0f;
            for (int e = groups->off[k]; e < groups->off[k + 1]; ++e)
                pred += groups->w[e] * pix[groups->px[e]];
            float diff = pred - b[k];
            if (zcorr) diff -= oc * zcorr[k];
            for (int e = groups->off[k]; e < groups->off[k + 1]; ++e)
                sc[groups->px[e]] += groups->w[e] * diff;
            if (zsave) zsave[k] = diff;
            fx += diff * diff;
        }
    }
    else for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * cols + riy[k];
        // LDAMP-style Onsager: debias the residual with the previous
        // residual scaled by the denoiser divergence (zcorr/oc supplied by
        // the core; null keeps the legacy plain residual, bit-identical).
        float diff = pix[idx] - b[k];
        if (zcorr) diff -= oc * zcorr[k];
        sc[idx] = diff;
        if (zsave) zsave[k] = diff;
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
    int rows, int cols, float tv_lambda, const cs_fista_basis& bx,
    const cs_meas_groups* groups = nullptr)
{
    const int n = rows * cols;
    std::memcpy(pix, xc, sizeof(float) * (size_t)n);
    cs_synth(pix, rows, cols, bx);
    float fx = 0.0f;
    if (groups && groups->ng > 0) {
        for (int k = 0; k < groups->ng; ++k) {
            float pred = 0.0f;
            for (int e = groups->off[k]; e < groups->off[k + 1]; ++e)
                pred += groups->w[e] * pix[groups->px[e]];
            const float diff = pred - b[k];
            fx += diff * diff;
        }
    }
    else for (int k = 0; k < m; ++k) {
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
    std::vector<float>& grad, std::vector<float>& pix, std::vector<float>& sc,
    bool use_onsager = false, const cs_meas_groups* groups = nullptr)
{
    const int n = rows * cols;
    // Grouped measurements (T-means) count groups, not pixels, for the
    // Onsager scaling/state below.
    const int meff = (groups && groups->ng > 0) ? groups->ng : m;
    // LDAMP-style Onsager state: previous corrected residual (size m) and
    // the soft-threshold divergence (support fraction) that scales it.
    // Per-call state (resets each reweight pass, whose weights/thresholds
    // define a fresh problem); disabled keeps legacy behavior exactly.
    const bool oc_on = use_onsager && meff > 0;
    std::vector<float> zprev(oc_on ? (size_t)meff : 0, 0.0f);
    float dof_prev = 0.0f;
    // Backtrack whenever the fixed step is not known-safe: CDF97
    // (biorthogonal, no exact Lipschitz) or any tv > 0. The smoothed-TV
    // gradient's local curvature scales as tv/eps, far above the 8*tv
    // folded into L in flat zones; stepping 1/L blindly there sustains a
    // Nyquist flip-flop (checkerboard dots, only with tv > 0). Growing L
    // until the quadratic upper bound holds guarantees a stable step.
    // tv = 0 keeps the exact fixed step (previous behavior, bit-identical).
    const bool bt = (bx.basis == CS_BASIS_CDF97) || (tv_lambda > 0.0f);
    const bool sp = g_solveprof.on;
    float L = bx.lip + 8.0f * tv_lambda;
    std::memcpy(y.data(), x, sizeof(float) * (size_t)n);
    std::memcpy(x_prev.data(), x, sizeof(float) * (size_t)n);
    std::vector<float> z((size_t)n);
    float* yy = y.data();
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        auto sp_t = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const float oc = oc_on ? dof_prev / (float)meff : 0.0f;
        const float fy = cs_fista_grad(yy, pix.data(), sc.data(), grad.data(), b, rix, riy, m, rows, cols, tv_lambda, bx,
            oc_on ? zprev.data() : nullptr, oc, oc_on ? zprev.data() : nullptr, groups);
        if (sp) { g_solveprof.fs_grad_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
        float step = 1.0f / L;
        if (bt) {
            // backtracking: shrink step until f(z) <= Q_L(z, y)
            for (int trial = 0; ; ++trial) {
                step = 1.0f / L;
                const float base = lambda * step;
                // simd-safe: z is local; yy/grad/w/wscale are distinct inputs.
                #pragma omp simd
                for (int i = 0; i < n; ++i) {
                    const float v = yy[i] - step * grad[i];
                    const float thr = base * w[i] * bx.wscale[i];
                    const float av = std::fabs(v);
                    z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
                }
                const float fz = cs_smooth_fx(z.data(), pix.data(), sc.data(), b, rix, riy, m, rows, cols, tv_lambda, bx, groups);
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
            // simd-safe: z is local; yy/grad/w/wscale are distinct inputs.
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                const float v = yy[i] - step * grad[i];
                const float thr = base * w[i] * bx.wscale[i];
                const float av = std::fabs(v);
                z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
            }
        }
        if (sp) { g_solveprof.fs_shrink_ns += sp_ns_since(sp_t); sp_t = std::chrono::steady_clock::now(); }
        if (oc_on) {
            // Divergence of soft-thresholding = support fraction: counts
            // the coefficients the shrink kept (z != 0 iff |v| > thr).
            // (Backtracking trials reuse the final z; the Q-model stays
            // approximate there, same as any inexact-gradient FISTA.)
            long long nnz = 0;
            for (int i = 0; i < n; ++i) if (z[(size_t)i] != 0.0f) ++nnz;
            dof_prev = (float)nnz / (float)n;
        }
        const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float mom = (t - 1.0f) / t_new;
        // simd-safe: yy/x_prev are distinct caller vectors, z is local.
        #pragma omp simd
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
    std::vector<float>& pix, std::vector<float>& sc,
    const cs_meas_groups* groups = nullptr)
{
    const int n = rows * cols;
    // Backtrack under tv > 0 for the same reason as cs_fista_core_single:
    // the fixed 1/(lip + 8*tv) step is unsafe against the smoothed-TV
    // local curvature in flat zones. tv = 0 keeps the exact fixed step.
    const bool bt = (bx.basis == CS_BASIS_CDF97) || (tv_lambda > 0.0f);
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
            fy += cs_fista_grad(yc[c], pix.data(), sc.data(), gc[c], bb[c], rix, riy, m, rows, cols, tv_lambda, bx,
                nullptr, 0.0f, nullptr, groups);
        }
        if (bt) {
            for (int trial = 0; ; ++trial) {
                const float step = 1.0f / L;
                const float base = lambda * step;
                // simd-safe: zc is local; yc/gc/w/wscale are distinct inputs.
                #pragma omp simd
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
                    fz += cs_smooth_fx(zc[c], pix.data(), sc.data(), bb[c], rix, riy, m, rows, cols, tv_lambda, bx, groups);
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
            // simd-safe: zc is local; yc/gc/w/wscale are distinct inputs.
            #pragma omp simd
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
        // simd-safe: yc/xp are distinct caller vectors, zc is local.
        for (int c = 0; c < 3; ++c) {
            #pragma omp simd
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
    float tv, int reweights, int fista_iters, int basis, float wscale, bool use_onsager,
    const cs_meas_groups* groups)
{
    if (!ref.isContinuous()) ref = ref.clone();
    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    // Grouped tiles count groups (mini-container holds group means in
    // group order); point tiles count pixels.
    const int grouped = (groups && groups->ng > 0) ? 1 : 0;
    const int m = grouped ? groups->ng : (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    // --pad-radix (or env CS_PAD_RADIX=1 for experiments): solve the tile on
    // a power-of-two canvas so cv::dct hits its fast radix path (48x56 -> 64x64
    // is ~1.5x cheaper per transform). Sample coordinates stay in the top-left;
    // the pad is left unconstrained so the prior extrapolates it, and the tail
    // crops back. Default off => bit-identical legacy behavior. The canvas is
    // larger, so decodes run slightly softer (~-0.003 SSIM uniform, ~-0.005
    // seg); that gap is intrinsic and does not close under coef/tv retuning.
    static const bool pad_radix_env = []{
        const char* e = std::getenv("CS_PAD_RADIX"); return e && *e && *e != '0';
    }();
    const bool pad_radix = cs_default_pad_radix() != 0 || pad_radix_env;
    int R = rows, C = cols;
    if (pad_radix && !grouped && basis == CS_BASIS_DCT &&
        cs_default_dct_block() <= 0 && !cs_gpu::enabled()) {
        R = 1; while (R < rows) R <<= 1;
        C = 1; while (C < cols) C <<= 1;
    }
    const bool padded = (R != rows || C != cols);
    const int n = R * C;
    std::vector<float> canvas;
    if (padded) {
        canvas.assign((size_t)n, 0.0f);
        const cv::Mat refm(rows, cols, CV_32F, (float*)ref.data);
        for (int r = 0; r < rows; ++r)
            std::memcpy(canvas.data() + (size_t)r * C, refm.ptr<float>(r), sizeof(float) * cols);
    }
    cs_fista_basis bx = cs_make_basis(R, C, basis, wscale);
    if (grouped) {
        // Averaged rows are shorter than point rows (true-average weights),
        // so the smooth-part Lipschitz 2*||A||^2 drops below lip. Groups are
        // disjoint within a tile, hence ||A||^2 = max row norm^2: rescale
        // the fixed step to match instead of stepping 4x conservatively.
        float maxn2 = 0.0f;
        for (int k = 0; k < groups->ng; ++k) {
            float rn2 = 0.0f;
            for (int e = groups->off[k]; e < groups->off[k + 1]; ++e)
                rn2 += groups->w[e] * groups->w[e];
            if (rn2 > maxn2) maxn2 = rn2;
        }
        if (maxn2 > 0.0f) bx.lip = 2.0f * maxn2;
    }

    std::vector<float> b;
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);

    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> y((size_t)n), x_prev((size_t)n), grad((size_t)n), pix((size_t)n), sc((size_t)n);
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    float* x = padded ? canvas.data() : (float*)ref.data;
    if (sp) g_solveprof.wrap_setup_ns += sp_ns_since(sp_t0);
    // Grouped (averaged) measurements have no HIP kernel: CPU fallback.
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97 && cs_default_dct_block() <= 0 && !grouped) {
        // GPU path: batched with concurrent callers inside cs_gpu; the tail
        // (ref chain copy + IDCT + 255) stays on the CPU below, unchanged.
        // tv rides per-problem: 0 takes the sampled fast path, >0 the TV
        // branch (same cs_fista_grad math as the CPU core). The HIP kernel
        // has no Onsager term: warn once and run it plain.
        if (use_onsager) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "Note: --onsager is CPU-only; ignoring on the GPU path"
                          << std::endl;
            }
        }
        cs_gpu::FistaProblem pb;
        pb.tv = tv;
        pb.b = b.data();
        pb.rix = ri_x.data();
        pb.riy = ri_y.data();
        pb.m = m;
        pb.rows = R;
        pb.cols = C;
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
        cs_fista_core_single(x, b.data(), ri_x.data(), ri_y.data(), m, R, C,
            w.data(), param_c, tv, inner, bx, y, x_prev, grad, pix, sc, use_onsager,
            grouped ? groups : nullptr);
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
    if (padded) {
        if (copy_next_ref && !next_ref.empty()) {
            if (!next_ref.isContinuous()) next_ref = next_ref.clone();
            for (int r = 0; r < rows; ++r)
                std::memcpy((float*)next_ref.data + (size_t)r * cols, x + (size_t)r * C, sizeof(float) * cols);
        }
        cv::Mat Cm(R, C, CV_32F, x);
        cv::dct(Cm, Cm, cv::DCT_INVERSE);
        Cm = Cm * 255.0f;
        cv::Mat refm(rows, cols, CV_32F, (float*)ref.data);
        for (int r = 0; r < rows; ++r)
            std::memcpy(refm.ptr<float>(r), Cm.ptr<float>(r), sizeof(float) * cols);
    } else {
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
            cs_dct_inverse(AtAxb2, bx.dct_block);
            AtAxb2 = AtAxb2 * 255.0f;
        }
    }
    if (sp) {
        g_solveprof.wrap_tail_ns += sp_ns_since(sp_t1);
        g_solveprof.wrap_calls += 1;
        g_solveprof.samples += m;
    }
}

void cs_refine_y_confidence(const cv::Mat& measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& plane, float strength, float coef, float tv,
    int iterations, int fista_iters, int basis, float wscale)
{
    if (!(strength > 0.0f) || basis != CS_BASIS_DCT ||
        plane.empty() || plane.type() != CV_32F ||
        ri_x.size() != ri_y.size() || ri_x.empty()) return;
    const int rows = plane.rows, cols = plane.cols, n = rows * cols;
    const int m = (int)ri_x.size();
    if (!plane.isContinuous() || rows < 8 || cols < 8 ||
        measurements.total() < (size_t)CS_HEADER_PIXELS + m) return;

    cv::Mat observed = cv::Mat::zeros(rows, cols, CV_32F);
    for (int k = 0; k < m; ++k) {
        if ((unsigned)ri_x[k] >= (unsigned)rows || (unsigned)ri_y[k] >= (unsigned)cols) return;
        observed.at<float>(ri_x[k], ri_y[k]) = 1.0f;
    }
    // Blur the binary sample mask on the solve grid. Normalize by the tile's
    // mean density, so a uniform draw has little/no extra prior. The clamp
    // prevents sparsely sampled tiles from getting an excessively strong pull.
    cv::Mat density, guide, target, weight(rows, cols, CV_32F);
    cv::GaussianBlur(observed, density, cv::Size(), 5.0, 5.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(plane, guide, cv::Size(), 1.2, 1.2, cv::BORDER_REPLICATE);
    guide.convertTo(target, CV_32F, 1.0 / 255.0);
    const float mean_density = (float)m / n;
    const float scale = (std::min)(strength, 1.0f) * 0.25f;
    for (int r = 0; r < rows; ++r) {
        const float* d = density.ptr<float>(r);
        const float* obs = observed.ptr<float>(r);
        float* dst = weight.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            const float low = (std::max)(0.0f, 1.0f - d[c] / (mean_density + 1e-6f));
            dst[c] = obs[c] > 0.0f ? 0.0f : scale * low;
        }
    }
    if (cv::countNonZero(weight) == 0) return;

    // The first-pass Y solve is a full-scale warm start. Penalize only sparse
    // coefficient *corrections* from it; measured pixels keep their original
    // quadratic data term. The target is a lightly smoothed first-pass image,
    // not a potentially misleading neighbour tile or a zero/flat reference.
    cv::Mat initial;
    plane.convertTo(initial, CV_32F, 1.0 / 255.0);
    const cs_fista_basis bx = cs_make_basis(rows, cols, basis, wscale);
    std::vector<float> x((size_t)n), y((size_t)n), prev((size_t)n), z((size_t)n),
        grad((size_t)n), pix((size_t)n), scratch((size_t)n), anchor((size_t)n), b;
    cs_extract_channel_measurements(measurements, 0, m, b);
    std::memcpy(x.data(), initial.ptr<float>(), sizeof(float) * (size_t)n);
    cs_synth_adj(x.data(), rows, cols, bx);
    anchor = x;
    y = x;
    prev = x;
    const float lambda = coef * 0.1f;
    float L = bx.lip + 2.0f * scale + 8.0f * tv;
    const bool bt = tv > 0.0f;
    const int iters = (std::min)(24, cs_fista_inner_iters(iterations, fista_iters));
    std::vector<float> tvgrad(tv > 0.0f ? (size_t)n : 0);
    auto smooth = [&](const float* coeff, float* gradient) -> double {
        std::memcpy(pix.data(), coeff, sizeof(float) * (size_t)n);
        cs_synth(pix.data(), rows, cols, bx);
        if (gradient) std::fill(scratch.begin(), scratch.end(), 0.0f);
        double fx = 0.0;
        for (int k = 0; k < m; ++k) {
            const int i = ri_x[k] * cols + ri_y[k];
            const float diff = pix[(size_t)i] - b[(size_t)k];
            fx += (double)diff * diff;
            if (gradient) scratch[(size_t)i] = 2.0f * diff;
        }
        const float* w = weight.ptr<float>();
        const float* t = target.ptr<float>();
        for (int i = 0; i < n; ++i) {
            const float diff = pix[(size_t)i] - t[i];
            fx += (double)w[i] * diff * diff;
            if (gradient) scratch[(size_t)i] += 2.0f * w[i] * diff;
        }
        if (tv > 0.0f) {
            fx += tv * cs_tv_grad_phi(pix.data(), tvgrad.data(), rows, cols);
            if (gradient)
                for (int i = 0; i < n; ++i) scratch[(size_t)i] += tv * tvgrad[(size_t)i];
        }
        if (gradient) {
            cs_synth_adj(scratch.data(), rows, cols, bx);
            std::memcpy(gradient, scratch.data(), sizeof(float) * (size_t)n);
        }
        return fx;
    };
    float momentum = 1.0f;
    for (int k = 0; k < iters; ++k) {
        const double fy = smooth(y.data(), grad.data());
        for (int trial = 0; ; ++trial) {
            const float step = 1.0f / L;
            for (int i = 0; i < n; ++i) {
                const float v = y[(size_t)i] - step * grad[(size_t)i] - anchor[(size_t)i];
                const float thr = lambda * step * bx.wscale[(size_t)i];
                z[(size_t)i] = anchor[(size_t)i] + (std::fabs(v) > thr ?
                    std::copysign(std::fabs(v) - thr, v) : 0.0f);
            }
            if (!bt) break;
            const double fz = smooth(z.data(), nullptr);
            double dot = 0.0, dz2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double dz = (double)z[(size_t)i] - y[(size_t)i];
                dot += (double)grad[(size_t)i] * dz;
                dz2 += dz * dz;
            }
            if (fz <= fy + dot + 0.5 * L * dz2 + 1e-7 * (1.0 + std::fabs(fy)) || trial >= 24) break;
            L *= 2.0f;
        }
        const float new_momentum = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * momentum * momentum));
        const float factor = (momentum - 1.0f) / new_momentum;
        for (int i = 0; i < n; ++i)
            y[(size_t)i] = z[(size_t)i] + factor * (z[(size_t)i] - prev[(size_t)i]);
        prev.swap(z);
        momentum = new_momentum;
    }
    cs_synth(prev.data(), rows, cols, bx);
    for (int i = 0; i < n; ++i) plane.ptr<float>()[i] = prev[(size_t)i] * 255.0f;
}

bool cs_y_mixed_density(const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int rows, int cols)
{
    if (rows < 16 || cols < 16 || ri_x.size() != ri_y.size() || ri_x.empty()) return false;
    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_32F);
    for (size_t k = 0; k < ri_x.size(); ++k) {
        if ((unsigned)ri_x[k] >= (unsigned)rows || (unsigned)ri_y[k] >= (unsigned)cols) return false;
        mask.at<float>(ri_x[k], ri_y[k]) = 1.0f;
    }
    cv::Mat density;
    cv::GaussianBlur(mask, density, cv::Size(), 5.0, 5.0, cv::BORDER_REPLICATE);
    const double avg = cv::mean(mask)[0];
    double low = 1.0, high = 0.0;
    for (int r = 0; r < rows; r += 4)
        for (int c = 0; c < cols; c += 4) {
            const double v = density.at<float>(r, c);
            low = (std::min)(low, v);
            high = (std::max)(high, v);
        }
    return low < 0.7 * avg && high > 1.3 * avg;
}

void cs_blend_y_dual_tv(const cv::Mat& low_bgr, const cv::Mat& high_bgr,
    const cv::Mat& sample_mask, cv::Mat& blended, cv::Mat* weight_out)
{
    blended.release();
    if (low_bgr.empty() || high_bgr.empty() || low_bgr.type() != CV_8UC3 ||
        high_bgr.type() != CV_8UC3 || low_bgr.size() != high_bgr.size() ||
        sample_mask.empty() || sample_mask.type() != CV_8U) return;

    // Edge detection uses the stable high-TV solution, so low-TV noise
    // cannot vote itself in. Selection below copies all three channels from
    // the chosen composite, which matches the saved binary mask exactly.
    cv::Mat high_ycc;
    cv::cvtColor(high_bgr, high_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> hch;
    cv::split(high_ycc, hch);
    cv::Mat high_y;
    hch[0].convertTo(high_y, CV_32F);
    cv::Mat edge_source, gx, gy, magnitude;
    cv::GaussianBlur(high_y, edge_source, cv::Size(), 1.0, 1.0);
    cv::Sobel(edge_source, gx, CV_32F, 1, 0, 3);
    cv::Sobel(edge_source, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, magnitude);

    cv::Mat mask;
    if (sample_mask.size() != high_bgr.size())
        cv::resize(sample_mask, mask, high_bgr.size(), 0, 0, cv::INTER_NEAREST);
    else mask = sample_mask;
    cv::Mat sampled, density;
    mask.convertTo(sampled, CV_32F);
    cv::GaussianBlur(sampled, density, cv::Size(), 5.0, 5.0, cv::BORDER_REPLICATE);
    const float mean = (float)cv::mean(sampled)[0];

    // Build solid detail segments: close strong edges and fill enclosed
    // interiors. The flood must start outside the image: (0,0) can itself
    // be an edge, in which case flooding 255 into it is a no-op and
    // bitwise_not(flood) makes the entire image a detail segment.
    cv::Mat mag_smooth, edge_bin;
    cv::GaussianBlur(magnitude, mag_smooth, cv::Size(), 1.5, 1.5,
        cv::BORDER_REPLICATE);
    cv::threshold(mag_smooth, edge_bin, 35.0, 255.0, cv::THRESH_BINARY);
    edge_bin.convertTo(edge_bin, CV_8U);
    const int kclose =
        (std::max)(7, (std::min)(25, (std::min)(high_bgr.rows, high_bgr.cols) / 200)) | 1;
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kclose, kclose));
    cv::Mat closed;
    cv::morphologyEx(edge_bin, closed, cv::MORPH_CLOSE, kernel);
    cv::Mat flood;
    cv::copyMakeBorder(closed, flood, 1, 1, 1, 1,
        cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
    cv::Mat holes, segments;
    cv::bitwise_not(flood(cv::Rect(1, 1, closed.cols, closed.rows)), holes);
    cv::bitwise_or(closed, holes, segments);
    // Keep only substantiated segments: real area AND enough strong-edge
    // pixels inside to justify low-TV detail. A closed loop around smooth
    // sky has a tiny edge fraction and stays high-TV. Cap component area so
    // merged reconstruction texture cannot turn the whole image white.
    cv::Mat labels, stats, centroids;
    const int ncomp =
        cv::connectedComponentsWithStats(segments, labels, stats, centroids, 8);
    const int min_area = (std::max)(32, high_bgr.rows * high_bgr.cols / 100000);
    // A single filled component occupying almost the whole image cannot
    // identify a meaningful detail boundary. Do not let it select low-TV
    // everywhere, even if an edge-rich background passes the edge fraction.
    const int max_area = (int)(0.80 * (double)segments.total());
    constexpr double kEdgeSubstantiation = 0.03;
    std::vector<int> edge_px((size_t)(std::max)(ncomp, 1), 0);
    for (int r = 0; r < labels.rows; ++r) {
        const int* L = labels.ptr<int>(r);
        const uchar* E = edge_bin.ptr<uchar>(r);
        for (int c = 0; c < labels.cols; ++c) {
            if (E[c]) edge_px[(size_t)L[c]]++;
        }
    }
    cv::Mat kept = cv::Mat::zeros(segments.size(), CV_8U);
    for (int i = 1; i < ncomp; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area && area <= max_area &&
            edge_px[(size_t)i] >= kEdgeSubstantiation * area)
            kept.setTo(255, labels == i);
    }
    // Coverage veto: segments without samples keep the high-TV base.
    cv::Mat coverage, selected;
    cv::threshold(density, coverage, 0.2f * mean, 255.0, cv::THRESH_BINARY);
    coverage.convertTo(coverage, CV_8U);
    cv::bitwise_and(kept, coverage, selected);
    // Closed contours miss flat portions of a colorful object such as the
    // crown of a bird. At half resolution, connect colorful regions of
    // similar hue and accept only components substantially supported by the
    // existing detail segments. Unsaturated cage bars and distant background
    // colors do not bridge into the object.
    if (cv::countNonZero(selected) > 0) {
        const cv::Size small_size((selected.cols + 1) / 2, (selected.rows + 1) / 2);
        cv::Mat small_seed, small_color, hsv;
        cv::resize(selected, small_seed, small_size, 0, 0, cv::INTER_AREA);
        cv::threshold(small_seed, small_seed, 128, 255, cv::THRESH_BINARY);
        cv::resize(high_bgr, small_color, small_size, 0, 0, cv::INTER_AREA);
        cv::GaussianBlur(small_color, small_color, cv::Size(), 1.5, 1.5);
        cv::cvtColor(small_color, hsv, cv::COLOR_BGR2HSV);
        cv::Mat grown = cv::Mat::zeros(small_size, CV_8U);
        // Hue bins keep adjacent shades of one object connected without
        // connecting unrelated saturated regions (blue sky vs green bird).
        for (int h0 = 0; h0 < 180; h0 += 15) {
            cv::Mat hue_bin;
            cv::inRange(hsv, cv::Scalar(h0, 90, 25),
                cv::Scalar(h0 + 14, 255, 255), hue_bin);
            cv::morphologyEx(hue_bin, hue_bin, cv::MORPH_CLOSE,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
            cv::Mat color_labels, color_stats, color_centers;
            const int count = cv::connectedComponentsWithStats(hue_bin,
                color_labels, color_stats, color_centers, 8);
            std::vector<int> seed_count((size_t)count, 0);
            for (int r = 0; r < small_seed.rows; ++r) {
                const uchar* s = small_seed.ptr<uchar>(r);
                const int* l = color_labels.ptr<int>(r);
                for (int c = 0; c < small_seed.cols; ++c)
                    if (s[c] && l[c]) ++seed_count[(size_t)l[c]];
            }
            std::vector<uchar> accept((size_t)count, 0);
            for (int i = 1; i < count; ++i) {
                const int area = color_stats.at<int>(i, cv::CC_STAT_AREA);
                if (area >= 64 && area <= (int)(0.30 * hue_bin.total()) &&
                    seed_count[(size_t)i] >= (std::max)(16, area / 40))
                    accept[(size_t)i] = 255;
            }
            for (int r = 0; r < grown.rows; ++r) {
                const int* l = color_labels.ptr<int>(r);
                uchar* g = grown.ptr<uchar>(r);
                for (int c = 0; c < grown.cols; ++c)
                    if (l[c]) g[c] |= accept[(size_t)l[c]];
            }
        }
        cv::resize(grown, grown, selected.size(), 0, 0, cv::INTER_NEAREST);
        cv::bitwise_or(selected, grown, selected);
        cv::bitwise_and(selected, coverage, selected);
    }
    // The saved segmentation is the actual hard compositing decision: white
    // takes the low-TV (well-sampled detail) tile composite, black keeps the
    // high-TV (sparse-region) composite. No second per-pixel edge weighting,
    // chroma substitution, or feathering can make the dump disagree with the
    // pixels used. Both candidates were overlap-feathered before this step.
    if (weight_out) selected.convertTo(*weight_out, CV_32F, 1.0 / 255.0);
    blended = high_bgr.clone();
    low_bgr.copyTo(blended, selected);
}

bool cs_build_dual_tv_class_views(const cv::Mat& composite, const cv::Mat& weight,
    const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    cv::Mat& lowtv, cv::Mat& hightv)
{
    lowtv.release();
    hightv.release();
    if (composite.empty() || composite.type() != CV_8UC3) return false;
    if (weight.empty() || weight.type() != CV_32F || weight.size() != composite.size())
        return false;
    if (tiles.empty() || tiles.size() != coordinates.size()) return false;
    // per-tile white fraction of the binary weight (white = low-TV solve
    // chosen for those pixels), split against the mean tile fraction.
    struct Cell { cv::Rect rect; double frac; };
    std::vector<Cell> cells;
    double sum = 0.0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].size() != coordinates[i].size()) return false;
        for (size_t j = 0; j < tiles[i].size(); ++j) {
            const cv::Mat& t = tiles[i][j];
            if (t.empty()) continue;
            cv::Rect r(coordinates[i][j].x, coordinates[i][j].y, t.cols, t.rows);
            r &= cv::Rect(0, 0, composite.cols, composite.rows);
            if (r.empty()) continue;
            const cv::Mat w = weight(r);
            const double frac = cv::mean(w)[0];  // 0/1 mask: mean == fraction
            cells.push_back({ r, frac });
            sum += frac;
        }
    }
    if (cells.empty()) return false;
    const double mean = sum / (double)cells.size();
    lowtv = cv::Mat::zeros(composite.size(), CV_8UC3);
    hightv = cv::Mat::zeros(composite.size(), CV_8UC3);
    long long n_low = 0, n_high = 0;
    for (const Cell& c : cells) {
        cv::Mat* dst = nullptr;
        if (c.frac > mean) { dst = &lowtv; ++n_low; }
        else if (c.frac < mean) { dst = &hightv; ++n_high; }
        else continue;  // ties go in neither canvas
        composite(c.rect).copyTo((*dst)(c.rect));
    }
    if (n_low == 0 && n_high == 0) {
        lowtv.release();
        hightv.release();
        return false;
    }
    auto label = [](cv::Mat& panel, const char* t) {
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    char info[96];
    std::snprintf(info, sizeof(info), "low-TV tiles: %lld", n_low);
    label(lowtv, info);
    std::snprintf(info, sizeof(info), "high-TV tiles: %lld", n_high);
    label(hightv, info);
    return true;
}

bool cs_build_dual_tv_cutouts(const cv::Mat& blended, const cv::Mat& low,
    const cv::Mat& weight, cv::Mat& lowcut, cv::Mat& highcut)
{
    lowcut.release();
    highcut.release();
    if (blended.empty() || blended.type() != CV_8UC3) return false;
    if (low.empty() || low.type() != CV_8UC3 || low.size() != blended.size())
        return false;
    if (weight.empty() || weight.size() != blended.size()) return false;
    cv::Mat sel;
    if (weight.type() == CV_32F)
        weight.convertTo(sel, CV_8U, 255.0);
    else if (weight.type() == CV_8U)
        sel = weight;
    else
        return false;
    cv::Mat inv;
    cv::bitwise_not(sel, inv);
    lowcut = cv::Mat::zeros(blended.size(), CV_8UC3);
    highcut = cv::Mat::zeros(blended.size(), CV_8UC3);
    low.copyTo(lowcut, sel);
    blended.copyTo(highcut, inv);
    return true;
}

void fista_channel_begin(fista_channel_task& t, const cv::Mat& pixel_measurements,
    const int& channel, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations,
    cv::Mat& ref, float tv, int reweights, int fista_iters, int basis, float wscale,
    bool use_onsager, const cs_meas_groups* groups)
{
    if (!ref.isContinuous()) ref = ref.clone();
    t.basis = basis;
    t.wscale = wscale;
    t.groups = (groups && groups->ng > 0) ? groups : nullptr;
    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    // Grouped (averaged) measurements have no HIP kernel: CPU fallback.
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97 && cs_default_dct_block() <= 0 && !t.groups) {
        const int m = (int)ri_x.size();
        const int inner = cs_fista_inner_iters(iterations, fista_iters);
        if (reweights < 1) reweights = 1;
        if (reweights > 5) reweights = 5;
        cs_extract_channel_measurements(pixel_measurements, channel, m, t.b);
        cs_gpu::FistaProblem& pb = t.pb;
        pb.tv = tv;
        pb.b = t.b.data();
        pb.rix = ri_x.data();
        pb.riy = ri_y.data();
        pb.m = m;
        pb.rows = rows;
        pb.cols = cols;
        pb.x = (float*)ref.data;
        pb.lambda = param_c;
        pb.inner = inner;
        pb.reweights = reweights;
        if (sp) g_solveprof.wrap_setup_ns += sp_ns_since(sp_t0);
        // enqueue only: the caller submits every independent channel of the
        // tile/wave before any wait, so the worker's queue never drains
        t.h = cs_gpu::fista_solve_submit(pb);
        if (sp) { g_solveprof.rw_passes += (reweights > 1) ? (reweights - 1) : 0; }
    } else {
        // CPU (or CDF97): unchanged full blocking solve + tail inline
        reconstruct_color_channel_fista(pixel_measurements, channel, param_c, rows, cols,
            ri_x, ri_y, iterations, ref, false, cs_null_mat(), tv, reweights, fista_iters,
            basis, wscale, use_onsager, t.groups);
        t.h = nullptr;
    }
}

void fista_channel_end(fista_channel_task& t, cv::Mat& ref)
{
    if (!t.h) return;  // CPU path or device fallback: begin() ran it all
    const bool sp = g_solveprof.on;
    auto sp_g = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    cs_gpu::fista_solve_wait(t.h);
    t.h = nullptr;
    auto sp_t1 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (sp) {
        g_solveprof.wrap_core_ns += sp_ns_since(sp_g);
        g_solveprof.wrap_calls += 1;
        g_solveprof.samples += t.pb.m;
    }
    if (t.basis == CS_BASIS_CDF97) {
        // never on the GPU path (begin routes CDF97 to the CPU), kept for
        // parity with the single-phase tail
        cs_fista_basis bx = cs_make_basis(ref.rows, ref.cols, t.basis, t.wscale);
        cs_dwt_inverse((float*)ref.data, ref.rows, ref.cols, bx.levels);
        cv::Mat plane(ref.rows, ref.cols, CV_32F, ref.data);
        plane = plane * 255.0f;
    } else {
        cv::Mat AtAxb2(ref.rows, ref.cols, CV_32F, (float*)ref.data);
        cs_dct_inverse(AtAxb2, cs_default_dct_block());
        AtAxb2 = AtAxb2 * 255.0f;
    }
    if (sp) g_solveprof.wrap_tail_ns += sp_ns_since(sp_t1);
}

void reconstruct_image_fista_joint(const cv::Mat& pixel_measurements, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat refs[3], float tv, int reweights, int fista_iters, int basis, float wscale,
    const cs_meas_groups* groups)
{
    for (int ch = 0; ch < 3; ++ch) {
        if (!refs[ch].isContinuous()) refs[ch] = refs[ch].clone();
    }
    const int n = rows * cols;
    const int grouped = (groups && groups->ng > 0) ? 1 : 0;
    const int m = grouped ? groups->ng : (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    cs_fista_basis bx = cs_make_basis(rows, cols, basis, wscale);
    if (grouped) {
        // Same Lipschitz rescaling as the single-channel path: disjoint
        // averaged rows shorten ||A||^2 to the max row norm^2.
        float maxn2 = 0.0f;
        for (int k = 0; k < groups->ng; ++k) {
            float rn2 = 0.0f;
            for (int e = groups->off[k]; e < groups->off[k + 1]; ++e)
                rn2 += groups->w[e] * groups->w[e];
            if (rn2 > maxn2) maxn2 = rn2;
        }
        if (maxn2 > 0.0f) bx.lip = 2.0f * maxn2;
    }

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
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97 && cs_default_dct_block() <= 0 && !grouped) {
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
            y, xp, grad, z, pix, sc, grouped ? groups : nullptr);
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
            cs_dct_inverse(plane, bx.dct_block);
            plane = plane * 255.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// CS super-resolution 2x refinement (DCT + TV, CPU)
// ---------------------------------------------------------------------------
// Replaces the AVIR 2x interpolation of a solved LR tile with a small sparse
// recovery: the HR tile (2H x 2W DCT coefficients) is solved so that its
// 2x2-box downsample matches the original LR random samples, regularized by
// reweighted-L1 on the DCT correction + optional smoothed TV on the HR grid.
// The AVIR upscale of the solved LR tile is the anchor: LR measurements
// cannot determine the missing HR frequencies, so preserve its nullspace
// estimate rather than shrinking the whole HR image toward zero.
// Forward operator A = P*D*IDCT with D = 2x2 average (||D|| = 0.5, so the
// data-term Lipschitz is 0.25*lip, not lip): the fixed step 1/L stays exact
// at tv = 0 and backtracking covers tv > 0 — same policy as
// cs_fista_core_single. DCT-only; other bases fall back to AVIR.

bool cs_is_superres_backend(const std::string& backend) {
    std::string s;
    s.reserve(backend.size());
    for (char c : backend) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        s.push_back(c);
    }
    return s == "cs" || s == "cs-sr" || s == "cssr" || s == "cs_sr";
}

bool cs_is_srdirect_backend(const std::string& backend) {
    std::string s;
    s.reserve(backend.size());
    for (char c : backend) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        s.push_back(c);
    }
    return s == "sr-direct" || s == "srdirect" || s == "sr_direct";
}

bool cs_is_dip_backend(const std::string& backend) {
    std::string s;
    s.reserve(backend.size());
    for (char c : backend) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        s.push_back(c);
    }
    return s == "dip";
}

// 2x2-box downsample HR (Hr x Wc) -> LR (Lr x Lc), Hr = 2*Lr, Wc = 2*Lc.
static void cs_sr_downsample(const float* hr, float* lr, int Lr, int Lc) {
    const int Wc = Lc * 2;
    for (int i = 0; i < Lr; ++i) {
        for (int j = 0; j < Lc; ++j) {
            const int h0 = (i * 2) * Wc + (j * 2);
            lr[i * Lc + j] = 0.25f * (hr[h0] + hr[h0 + 1] + hr[h0 + Wc] + hr[h0 + Wc + 1]);
        }
    }
}

// Gradient of the smooth part in the HR coefficient domain:
//   g = 2*IDCT^T(D^T(scatter(D(IDCT(y)) - b))) [+ tv*IDCT^T(tvgrad(IDCT(y)))].
// pix_hr/up_hr are n_hr scratch, lr_buf/sc_lr are n_lr scratch. Returns the
// smooth-objective value f(y) = ||Ax-b||^2 (+ tv*TV) for backtracking.
static float cs_sr_fista_grad(const float* y, float* pix_hr, float* lr_buf, float* sc_lr,
    float* up_hr, float* g, const float* b, const float* anchor, float anchor_weight,
    const int* rix, const int* riy, int m, int Hr, int Wc, int Lr, int Lc, float tv_lambda,
    const int* hr_idx, const float* hr_val, int n_hr_anchors, float hr_weight)
{
    const int n_hr = Hr * Wc;
    const int n_lr = Lr * Lc;
    std::memcpy(pix_hr, y, sizeof(float) * (size_t)n_hr);
    cv::Mat P(Hr, Wc, CV_32F, pix_hr);
    cs_dct_inverse(P, cs_default_dct_block());
    cs_sr_downsample(pix_hr, lr_buf, Lr, Lc);
    std::memset(sc_lr, 0, sizeof(float) * (size_t)n_lr);
    float fx = 0.0f;
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * Lc + riy[k];
        if ((unsigned)idx >= (unsigned)n_lr) continue;
        const float diff = lr_buf[idx] - b[k];
        sc_lr[idx] = diff;
        fx += diff * diff;
    }
    // Adjoint of the 2x2 average: replicate the LR residual into each of
    // the 4 HR children, scaled by d(avg)/d(child) = 0.25.
    for (int i = 0; i < Lr; ++i) {
        for (int j = 0; j < Lc; ++j) {
            const float v = 0.25f * sc_lr[i * Lc + j];
            const int h0 = (i * 2) * Wc + (j * 2);
            up_hr[h0] = v; up_hr[h0 + 1] = v;
            up_hr[h0 + Wc] = v; up_hr[h0 + Wc + 1] = v;
        }
    }
    cv::Mat G(Hr, Wc, CV_32F, up_hr);
    cs_dct_forward(G, cs_default_dct_block());
    for (int i = 0; i < n_hr; ++i) g[i] = 2.0f * up_hr[i];

    // Exact-HR data term: sum hr_w*(pix[a]-v_a)^2 over anchor pixels.
    // Point sampling is orthonormal, so each anchor pulls its own pixel at
    // full weight; the sparse residual is DCT'd exactly like the LR term.
    // (up_hr is free scratch here: the data grad is already folded into g,
    // and the TV branch below only reads pix_hr.)
    if (n_hr_anchors > 0 && hr_weight > 0.0f && hr_idx && hr_val) {
        std::memset(up_hr, 0, sizeof(float) * (size_t)n_hr);
        for (int k = 0; k < n_hr_anchors; ++k) {
            const int idx = hr_idx[k];
            if ((unsigned)idx >= (unsigned)n_hr) continue;
            const float diff = pix_hr[idx] - hr_val[k];
            up_hr[idx] += diff;
            fx += hr_weight * diff * diff;
        }
        cv::Mat S(Hr, Wc, CV_32F, up_hr);
        cs_dct_forward(S, cs_default_dct_block());
        for (int i = 0; i < n_hr; ++i) g[i] += 2.0f * hr_weight * up_hr[i];
    }

    if (tv_lambda > 0.0f) {
        float* tvgrad = up_hr; // data grad already folded into g
        const float phi = cs_tv_grad_phi(pix_hr, tvgrad, Hr, Wc);
        cv::Mat T(Hr, Wc, CV_32F, tvgrad);
        cs_dct_forward(T, cs_default_dct_block());
        for (int i = 0; i < n_hr; ++i) g[i] += tv_lambda * tvgrad[i];
        fx += tv_lambda * phi;
    }
    for (int i = 0; i < n_hr; ++i) {
        const float diff = y[i] - anchor[i];
        g[i] += 2.0f * anchor_weight * diff;
        fx += anchor_weight * diff * diff;
    }
    return fx;
}

// Smooth-objective value f(xc) for a backtracking candidate.
static float cs_sr_smooth_fx(const float* xc, float* pix_hr, float* lr_buf,
    const float* b, const float* anchor, float anchor_weight,
    const int* rix, const int* riy, int m,
    int Hr, int Wc, int Lr, int Lc, float tv_lambda, float* tv_scratch,
    const int* hr_idx, const float* hr_val, int n_hr_anchors, float hr_weight)
{
    const int n_hr = Hr * Wc;
    const int n_lr = Lr * Lc;
    std::memcpy(pix_hr, xc, sizeof(float) * (size_t)n_hr);
    cv::Mat P(Hr, Wc, CV_32F, pix_hr);
    cs_dct_inverse(P, cs_default_dct_block());
    cs_sr_downsample(pix_hr, lr_buf, Lr, Lc);
    float fx = 0.0f;
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * Lc + riy[k];
        if ((unsigned)idx >= (unsigned)n_lr) continue;
        const float diff = lr_buf[idx] - b[k];
        fx += diff * diff;
    }
    if (tv_lambda > 0.0f) {
        fx += tv_lambda * cs_tv_grad_phi(pix_hr, tv_scratch, Hr, Wc);
    }
    if (n_hr_anchors > 0 && hr_weight > 0.0f && hr_idx && hr_val) {
        for (int k = 0; k < n_hr_anchors; ++k) {
            const int idx = hr_idx[k];
            if ((unsigned)idx >= (unsigned)n_hr) continue;
            const float diff = pix_hr[idx] - hr_val[k];
            fx += hr_weight * diff * diff;
        }
    }
    for (int i = 0; i < n_hr; ++i) {
        const float diff = xc[i] - anchor[i];
        fx += anchor_weight * diff * diff;
    }
    return fx;
}

// Anchor-proximity RED map (see hpp): exact distance to the nearest anchor
// via distanceTransform (anchors = zero pixels), mapped to a [0,1] blend
// factor. Empty when there is nothing to modulate (legacy uniform RED).
cv::Mat cs_hr_red_map(int Hr, int Wc, const std::vector<int>& anchor_idx, float radius)
{
    cv::Mat empty;
    if (Hr <= 0 || Wc <= 0 || anchor_idx.empty() || !(radius > 0.0f)) return empty;
    cv::Mat mask(Hr, Wc, CV_8U, cv::Scalar(255));
    bool any = false;
    for (int idx : anchor_idx) {
        if (idx < 0 || idx >= Hr * Wc) continue;
        mask.at<uint8_t>(idx / Wc, idx % Wc) = 0;
        any = true;
    }
    if (!any) return empty;
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);
    cv::Mat factor(Hr, Wc, CV_32F);
    const float inv2s2 = 1.0f / (2.0f * radius * radius);
    for (int y = 0; y < Hr; ++y) {
        const float* dp = dist.ptr<float>(y);
        float* fp = factor.ptr<float>(y);
        for (int x = 0; x < Wc; ++x) {
            const float d = dp[x];
            fp[x] = 1.0f - std::exp(-d * d * inv2s2);
        }
    }
    return factor;
}

// RED-lite proximal step (Romano et al., "Regularization by Denoising"):
// x <- x - red*(x - D(x)) in the pixel domain, with D = fastNlMeans.
// Applied once per outer FISTA pass (not per inner iteration: a full
// denoiser call per gradient step is infeasible on CPU at auto single-tile
// 12MP geometries; per-pass application is the standard practical PnP/RED
// cadence for expensive denoisers). red in [0,1], 0 = no-op. pix is CV_32F
// single-channel, full scale; x stays in the coefficient domain for the
// caller (synth/analy round-trip around the blend).
// NOTE: D is swappable — BM3D (xphoto) fits this exact slot but costs
// minutes per call on CPU at these sizes, so NLM (photo, already linked)
// is the CPU-feasible choice.
static void cs_sr_red_proximal(float* x, int Hr, int Wc, float red)
{
    if (!(red > 0.0f)) return;
    if (red > 1.0f) red = 1.0f;
    cv::Mat pix(Hr, Wc, CV_32F, x);
    cs_dct_inverse(pix, cs_default_dct_block());
    cv::Mat u8, dn, dnf;
    // full scale here is [0,1]: expand to 8U for the denoiser (a bare
    // convertTo would quantize everything to 0/1 and darken the tile).
    pix.convertTo(u8, CV_8U, 255.0);
    cv::fastNlMeansDenoising(u8, dn, 5.0f, 7, 15);
    dn.convertTo(dnf, CV_32F, 1.0 / 255.0);
    pix -= red * (pix - dnf);
    cs_dct_forward(pix, cs_default_dct_block());
}

// Anchor-weighted RED-lite proximal: same NLM pull, but the blend strength
// varies per pixel (wmap in [0,1], e.g. from cs_hr_red_map): exact anchors
// take no denoising pull while poorly-anchored pixels take the full one.
// Empty wmap (or wrong geometry) degrades to the uniform path exactly.
static void cs_sr_red_proximal_weighted(float* x, int Hr, int Wc, float red,
    const cv::Mat& wmap)
{
    if (!(red > 0.0f)) return;
    if (red > 1.0f) red = 1.0f;
    if (wmap.empty() || wmap.rows != Hr || wmap.cols != Wc || wmap.type() != CV_32F) {
        cs_sr_red_proximal(x, Hr, Wc, red);
        return;
    }
    cv::Mat pix(Hr, Wc, CV_32F, x);
    cs_dct_inverse(pix, cs_default_dct_block());
    cv::Mat u8, dn, dnf;
    pix.convertTo(u8, CV_8U, 255.0);
    cv::fastNlMeansDenoising(u8, dn, 5.0f, 7, 15);
    dn.convertTo(dnf, CV_32F, 1.0 / 255.0);
    cv::Mat step = (pix - dnf).mul(wmap);
    pix -= red * step;
    cs_dct_forward(pix, cs_default_dct_block());
}

double cs_estimate_noise_sigma(const cv::Mat& bgr)
{    if (bgr.empty() || bgr.type() != CV_8UC3) return 0.0;
    if (bgr.rows < 4 || bgr.cols < 4) return 0.0;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    cv::Mat yf;
    chs[0].convertTo(yf, CV_32F);
    // Immerkaer Laplacian (valid interior only: 1px border excluded), then
    // median block sigma so texture does not pose as grain (see header).
    static const float kdata[9] = { 1, -2, 1, -2, 4, -2, 1, -2, 1 };
    const cv::Mat kernel(3, 3, CV_32F, (void*)kdata);
    cv::Mat lo;
    cv::filter2D(yf, lo, CV_32F, kernel, cv::Point(-1, -1), 0.0,
        cv::BORDER_REPLICATE);
    cv::Mat ab;
    cv::absdiff(lo, cv::Scalar::all(0), ab);
    static const double kScale = 1.2533141373155001 / 6.0;  // sqrt(pi/2)/6
    constexpr int BS = 32;
    std::vector<double> blocks;
    blocks.reserve((size_t)((yf.rows + BS - 1) / BS) * ((yf.cols + BS - 1) / BS));
    for (int y0 = 1; y0 + 1 < yf.rows; y0 += BS) {
        for (int x0 = 1; x0 + 1 < yf.cols; x0 += BS) {
            const int w = (std::min)(BS, yf.cols - 1 - x0);
            const int h = (std::min)(BS, yf.rows - 1 - y0);
            if (w < 8 || h < 8) continue;
            blocks.push_back(kScale * cv::mean(ab(cv::Rect(x0, y0, w, h)))[0]);
        }
    }
    if (blocks.empty()) return 0.0;
    std::nth_element(blocks.begin(), blocks.begin() + blocks.size() / 2, blocks.end());
    return blocks[blocks.size() / 2];
}

void cs_tv_denoise_tile(cv::Mat& bgr, float lambda, int iters)
{
    if (!(lambda > 0.0f) || iters < 1) return;
    if (bgr.empty() || bgr.type() != CV_8UC3) return;
    if (bgr.rows < 4 || bgr.cols < 4) return;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    cv::Mat f;
    chs[0].convertTo(f, CV_32F);
    const int H = f.rows, W = f.cols;
    const int n = H * W;
    // Chambolle projection: u = f - lambda*div(p), p updated with step tau
    // (tau <= 1/8 for the unit-weight gradient). Serial per tile; tiles run
    // in parallel upstream.
    std::vector<float> px((size_t)n, 0.0f), py((size_t)n, 0.0f);
    std::vector<float> div((size_t)n, 0.0f), gx((size_t)n, 0.0f),
        gy((size_t)n, 0.0f);
    const float* fp = (const float*)f.data;
    constexpr float tau = 0.125f;
    std::vector<float> u((size_t)n);
    for (int it = 0; it < iters; ++it) {
        // div(p): backward differences, Neumann (zero flux at borders)
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const size_t o = (size_t)r * W + c;
                const float pxc = px[o];
                const float pxm = (c > 0) ? px[o - 1] : 0.0f;
                const float pyc = py[o];
                const float pym = (r > 0) ? py[o - (size_t)W] : 0.0f;
                div[o] = (pxc - pxm) + (pyc - pym);
            }
        }
        // g = grad(div(p) - f/lambda), forward differences
        const float inv = 1.0f / lambda;
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const size_t o = (size_t)r * W + c;
                const float v = div[o] - fp[o] * inv;
                const float vrx = (c + 1 < W) ? div[o + 1] - fp[o + 1] * inv : v;
                const float vdn = (r + 1 < H) ? div[o + (size_t)W] - fp[o + (size_t)W] * inv : v;
                gx[o] = vrx - v;
                gy[o] = vdn - v;
            }
        }
        for (int i = 0; i < n; ++i) {
            const float nx = px[(size_t)i] + tau * gx[(size_t)i];
            const float ny = py[(size_t)i] + tau * gy[(size_t)i];
            const float norm = std::sqrt(nx * nx + ny * ny);
            const float s = 1.0f + tau * norm;
            px[(size_t)i] = nx / s;
            py[(size_t)i] = ny / s;
        }
    }
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const size_t o = (size_t)r * W + c;
            const float pxc = px[o];
            const float pxm = (c > 0) ? px[o - 1] : 0.0f;
            const float pyc = py[o];
            const float pym = (r > 0) ? py[o - (size_t)W] : 0.0f;
            u[o] = fp[o] - lambda * ((pxc - pxm) + (pyc - pym));
        }
    }
    cv::Mat um(H, W, CV_32F, u.data());
    um.convertTo(chs[0], CV_8U);
    cv::Mat merged, out;
    cv::merge(chs, merged);
    cv::cvtColor(merged, out, cv::COLOR_YCrCb2BGR);
    if (!out.empty() && out.size() == bgr.size() && out.type() == CV_8UC3)
        bgr = std::move(out);
}

#ifdef _WIN32
// In-process DnCNN color denoiser (repo-vendored ONNX model) for the dncnn
// RED branch. Sessions are cached per path (load once, warn once); Run is
// thread-safe so fused tile workers may share one session. Model contract
// (verified by unit test, not assumed): RGB planar [0,1] in, CLEAN image
// out (not residual) — a clean input comes back nearly unchanged, so the
// first implementation's residual assumption failed loudly here and was
// flipped.
struct CsDncnnEntry {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cs-dncnn" };
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
};
static std::mutex g_dncnn_mutex;
static std::map<std::string, std::unique_ptr<CsDncnnEntry>> g_dncnn_cache;
static std::map<std::string, bool> g_dncnn_failed;
// Tile geometry per model kind: DnCNN is tiny (20 layers); SCUNet/Restormer
// carry far more activations, so they run smaller tiles to bound memory.
static int cs_onnx_tile_for_kind(int kind) {
    return kind == CS_ONNX_DNCNN ? 512 : 256;
}

// The ORT DLL is delay-loaded and resolved from PATH: a stale system DLL
// older than the vendored headers makes GetApi(ORT_API_VERSION) return null,
// and constructing Ort::Env on it AV-crashes (not an Ort::Exception, so the
// loader below cannot catch it). Probe the API table first so a
// version-skewed runtime degrades to "model unavailable" instead of taking
// the process down.
static bool cs_ort_api_available() {
    try {
        const OrtApiBase* base = OrtGetApiBase();
        if (!base || !base->GetApi) return false;
        return base->GetApi(ORT_API_VERSION) != nullptr;
    }
    catch (...) {
        return false;
    }
}

static CsDncnnEntry* cs_dncnn_entry(const std::string& path, const char* tag) {
    std::lock_guard<std::mutex> lk(g_dncnn_mutex);
    auto it = g_dncnn_cache.find(path);
    if (it != g_dncnn_cache.end()) return it->second.get();
    if (g_dncnn_failed.find(path) != g_dncnn_failed.end()) return nullptr;
    if (!cs_ort_api_available()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                "Warning: onnxruntime.dll is older than the build headers "
                "(API %d unavailable); DnCNN denoiser disabled\n",
                ORT_API_VERSION);
        }
        g_dncnn_failed[path] = true;
        return nullptr;
    }
    try {
        auto e = std::make_unique<CsDncnnEntry>();
        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const std::wstring wpath(path.begin(), path.end());
        e->session = std::make_unique<Ort::Session>(e->env, wpath.c_str(), opt);
        Ort::AllocatorWithDefaultOptions alloc;
        e->in_name = e->session->GetInputNameAllocated(0, alloc).get();
        e->out_name = e->session->GetOutputNameAllocated(0, alloc).get();
        auto ti = e->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            ti.GetDimensionsCount() != 4) {
            throw Ort::Exception("dncnn: expected float NCHW input", ORT_INVALID_ARGUMENT);
        }
        auto r = g_dncnn_cache.emplace(path, std::move(e));
        return r.first->second.get();
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: %s model '%s' failed to load: %s\n",
            tag, path.c_str(), ex.what());
        g_dncnn_failed[path] = true;
        return nullptr;
    }
}

static bool cs_dncnn_run_tile(CsDncnnEntry* e, const float* rgb01, float* clean,
    int h, int w)
{
    int64_t shape[4] = { 1, 3, h, w };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem,
        const_cast<float*>(rgb01), (size_t)3 * h * w, shape, 4);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    std::vector<Ort::Value> out;
    try {
        out = e->session->Run(Ort::RunOptions{ nullptr }, inn, &in, 1, outn, 1);
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: onnx-denoiser run failed: %s\n", ex.what());
        return false;
    }
    if (out.size() != 1 || !out[0].IsTensor()) return false;
    const std::vector<int64_t> dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] != h || dims[3] != w) return false;
    const float* src = out[0].GetTensorData<float>();
    std::memcpy(clean, src, sizeof(float) * (size_t)3 * h * w);
    return true;
}

bool cs_dncnn_available(const std::string& model_path) {
    if (model_path.empty()) return false;
    return cs_dncnn_entry(model_path, "DnCNN") != nullptr;
}

// Shared tiled RGB-NCHW clean-output runner (see header contract): BGR in,
// BGR out, feathered overlap compositing, per-path session cache. tile/ov
// bound activation memory per model size (DnCNN 512/48; SCUNet/Restormer
// 256/32).
static bool cs_onnx_denoise_tiled(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, const char* tag, int tile, int ov, bool pad128)
{
    dst_bgr.release();
    if (src_bgr.empty() || src_bgr.type() != CV_8UC3) return false;
    if (tile < 32 || ov < 8 || ov * 2 >= tile) return false;
    CsDncnnEntry* e = cs_dncnn_entry(model_path, tag);
    if (!e) return false;
    // Swin-window models (SCUNet/Restormer) reshape downsampled grids into
    // fixed 8-wide windows down to the /16 bottleneck: every model run is
    // padded to multiples of 128 below (replicate edge), so odd edge tiles
    // run instead of failing Reshape/Gather nodes. DnCNN is fully
    // convolutional: unpadded (bit-identical legacy path).
    const cv::Mat& src = src_bgr;
    const int H0 = src_bgr.rows, W0 = src_bgr.cols;
    const int H = src.rows, W = src.cols;
    (void)pad128;
    cv::Mat rgb;
    cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chs;
    cv::split(rgb, chs);
    const int T = tile, OV = ov;
    std::vector<int> ys, xs;
    for (int y = 0; y < H; y += T - OV) ys.push_back(y);
    for (int x = 0; x < W; x += T - OV) xs.push_back(x);
    if (ys.back() + T < H) ys.push_back(H - T);
    if (xs.back() + T < W) xs.push_back(W - T);
    cv::Mat acc(H, W, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight(H, W, CV_32F, cv::Scalar(0));
    std::vector<float> in, rs;
    for (size_t t = 0; t < ys.size() * xs.size(); ++t) {
        const int y0 = ys[t / xs.size()], x0 = xs[t % xs.size()];
        const int th = (std::min)(T, H - y0), tw = (std::min)(T, W - x0);
        // Swin-window models need multiples of 128 per run: replicate-pad
        // the run tile (DnCNN runs unpadded). Only the valid th x tw
        // region feeds the feathered accumulation below.
        const int ph = pad128 ? ((th + 127) / 128) * 128 : th;
        const int pw = pad128 ? ((tw + 127) / 128) * 128 : tw;
        in.assign((size_t)3 * ph * pw, 0.0f);
        rs.assign((size_t)3 * ph * pw, 0.0f);
        for (int c = 0; c < 3; ++c) {
            float* dst = &in[(size_t)c * ph * pw];
            for (int r = 0; r < ph; ++r) {
                const int sr = y0 + ((r < th) ? r : th - 1);
                const uint8_t* srow = chs[(size_t)c].ptr<uint8_t>(sr) + x0;
                for (int k = 0; k < pw; ++k) {
                    const int sk = (k < tw) ? k : tw - 1;
                    dst[(size_t)r * pw + k] = srow[sk] * (1.0f / 255.0f);
                }
            }
        }
        if (!cs_dncnn_run_tile(e, in.data(), rs.data(), ph, pw)) return false;
        for (int r = 0; r < th; ++r) {
            const int gy = y0 + r;
            cv::Vec3f* arow = acc.ptr<cv::Vec3f>(gy);
            float* wrow = weight.ptr<float>(gy);
            // linear feather over OV at interior tile borders; full weight
            // where the tile touches the image edge (no neighbor there)
            const int dt = (y0 == 0) ? OV : r;
            const int db = (y0 + th == H) ? OV : th - 1 - r;
            const float wy = (std::min)(1.0f, (std::min)(dt, db) / (float)OV);
            for (int k = 0; k < tw; ++k) {
                const int gx = x0 + k;
                const int dl = (x0 == 0) ? OV : k;
                const int dr = (x0 + tw == W) ? OV : tw - 1 - k;
                const float wx = (std::min)(1.0f, (std::min)(dl, dr) / (float)OV);
                const float wt = wx * wy;
                const size_t o = (size_t)r * pw + k;
                // clean-output convention: the model output IS the denoised
                // estimate (convertTo saturates to [0,255] at the end).
                const cv::Vec3f clean_rgb(
                    rs[o],
                    rs[(size_t)ph * pw + o],
                    rs[(size_t)2 * ph * pw + o]);
                arow[gx] += clean_rgb * wt;
                wrow[gx] += wt;
            }
        }
    }
    cv::Mat clean_rgb(H, W, CV_32FC3);
    for (int i = 0; i < H; ++i) {
        const cv::Vec3f* arow = acc.ptr<cv::Vec3f>(i);
        const float* wrow = weight.ptr<float>(i);
        cv::Vec3f* crow = clean_rgb.ptr<cv::Vec3f>(i);
        for (int j = 0; j < W; ++j) {
            const float w = wrow[j] > 0.0f ? wrow[j] : 1.0f;
            crow[j] = arow[j] / w;
        }
    }
    cv::Mat clean_bgr;
    cv::cvtColor(clean_rgb, clean_bgr, cv::COLOR_RGB2BGR);
    clean_bgr.convertTo(dst_bgr, CV_8U, 255.0);
    if (dst_bgr.rows != H0 || dst_bgr.cols != W0 || dst_bgr.type() != CV_8UC3) {
        dst_bgr.release();
        return false;
    }
    return true;
}

static void cs_onnx_geom_for_kind(int kind, const char*& tag, int& tile, int& ov) {
    if (kind == CS_ONNX_SCUNET) { tag = "SCUNet"; tile = 256; ov = 32; return; }
    if (kind == CS_ONNX_RESTORMER) { tag = "Restormer"; tile = 256; ov = 32; return; }
    tag = "DnCNN"; tile = 512; ov = 48;
}

bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path)
{
    return cs_onnx_denoise_tiled(src_bgr, dst_bgr, model_path, "DnCNN", 512, 48, false);
}

bool cs_onnx_available(const std::string& model_path, int kind) {
    if (model_path.empty()) return false;
    const char* tag; int tile, ov;
    cs_onnx_geom_for_kind(kind, tag, tile, ov);
    return cs_dncnn_entry(model_path, tag) != nullptr;
}

bool cs_onnx_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int kind)
{
    const char* tag; int tile, ov;
    cs_onnx_geom_for_kind(kind, tag, tile, ov);
    return cs_onnx_denoise_tiled(src_bgr, dst_bgr, model_path, tag, tile, ov,
        kind != CS_ONNX_DNCNN);
}
#else
bool cs_dncnn_available(const std::string& model_path) {
    (void)model_path;
    return false;
}
bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    return false;
}
bool cs_onnx_available(const std::string& model_path, int kind) {
    (void)model_path;
    (void)kind;
    return false;
}
bool cs_onnx_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int kind)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    (void)kind;
    return false;
}
#endif

// One weighted-L1 FISTA run over the HR coefficients (DCT-only). The composed
// operator A = P*D*IDCT has ||A|| <= ||D|| = 0.5 (2x2 averaging is a
// contraction by 2: ||D||_2 = 0.5), so the data-term Lipschitz is
// 2*||A||^2 <= 0.25*lip (lip = 2 for the P*IDCT path) — NOT lip itself.
// Using lip would step 4x too small and strand coefficient growth.
static void cs_sr_core_single(float* x, const float* b, const float* anchor,
    float anchor_weight, const int* rix, const int* riy, int m,
    int Hr, int Wc, int Lr, int Lc, const float* w, float lambda, float tv_lambda, int iters,
    const cs_fista_basis& bx,
    std::vector<float>& y, std::vector<float>& x_prev,
    std::vector<float>& grad, std::vector<float>& pix_hr,
    std::vector<float>& lr_buf, std::vector<float>& sc_lr, std::vector<float>& up_hr,
    const int* hr_idx, const float* hr_val, int n_hr_anchors, float hr_weight)
{
    const int n_hr = Hr * Wc;
    const bool bt = (tv_lambda > 0.0f);
    // Data-term Lipschitz for P*D*IDCT: 2*||A||^2 <= 0.25*lip. TV keeps the
    // same 8*tv curvature fold + backtracking policy as the LR core.
    // Exact-HR anchors add 2*hr_w (orthonormal point sampling).
    // (Always-on backtracking was trialed here after a white-output scare
    // that turned out to be a white-input test fixture, not instability:
    // biased samples solve fine at 32.47 dB. Kept off: the fixed step is
    // exact-safe and ~40% cheaper per iteration.)
    const float hr_lip = (n_hr_anchors > 0 && hr_weight > 0.0f) ? 2.0f * hr_weight : 0.0f;
    float L = 0.25f * bx.lip + 2.0f * anchor_weight + 8.0f * tv_lambda + hr_lip;
    std::memcpy(y.data(), x, sizeof(float) * (size_t)n_hr);
    std::memcpy(x_prev.data(), x, sizeof(float) * (size_t)n_hr);
    std::vector<float> z((size_t)n_hr);
    float* yy = y.data();
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        const float fy = cs_sr_fista_grad(yy, pix_hr.data(), lr_buf.data(), sc_lr.data(),
            up_hr.data(), grad.data(), b, anchor, anchor_weight,
            rix, riy, m, Hr, Wc, Lr, Lc, tv_lambda,
            hr_idx, hr_val, n_hr_anchors, hr_weight);
        float step = 1.0f / L;
        if (bt) {
            for (int trial = 0; ; ++trial) {
                step = 1.0f / L;
                const float base = lambda * step;
                #pragma omp simd
                for (int i = 0; i < n_hr; ++i) {
                    const float v = yy[i] - step * grad[i] - anchor[i];
                    const float thr = base * w[i] * bx.wscale[i];
                    const float av = std::fabs(v);
                    z[(size_t)i] = anchor[i] + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
                }
                const float fz = cs_sr_smooth_fx(z.data(), pix_hr.data(), lr_buf.data(),
                    b, anchor, anchor_weight, rix, riy, m, Hr, Wc, Lr, Lc, tv_lambda, up_hr.data(),
                    hr_idx, hr_val, n_hr_anchors, hr_weight);
                double dot = 0.0, dz2 = 0.0;
                for (int i = 0; i < n_hr; ++i) {
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
            #pragma omp simd
            for (int i = 0; i < n_hr; ++i) {
                const float v = yy[i] - step * grad[i] - anchor[i];
                const float thr = base * w[i] * bx.wscale[i];
                const float av = std::fabs(v);
                z[(size_t)i] = anchor[i] + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
            }
        }
        const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float mom = (t - 1.0f) / t_new;
        #pragma omp simd
        for (int i = 0; i < n_hr; ++i) yy[i] = z[(size_t)i] + mom * (z[(size_t)i] - x_prev[(size_t)i]);
        std::memcpy(x_prev.data(), z.data(), sizeof(float) * (size_t)n_hr);
        t = t_new;
    }
    std::memcpy(x, z.data(), sizeof(float) * (size_t)n_hr);
}

void cs_sr_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int reweights, int fista_iters, int basis, float wscale,
    cv::Mat& hr_out, float red, const std::string& red_denoiser, const std::string& dncnn_model,
    bool sr_cascade, const std::string& red_model, float red_decay,
    const CsHrSrCtx& hr_sr)
{
    hr_out.release();
    if (lr_tile.empty() || lr_tile.type() != CV_8UC3) return;
    const int Lr = lr_tile.rows, Lc = lr_tile.cols;
    if (Lr < 8 || Lc < 8) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    if (basis != CS_BASIS_DCT || ri_x.size() != ri_y.size() || ri_x.empty()) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    const int Hr = Lr * 2, Wc = Lc * 2;
    const int n_hr = Hr * Wc;
    const int n_lr = Lr * Lc;
    const int m = (int)ri_x.size();
    if (pixel_measurements.total() < (size_t)CS_HEADER_PIXELS + (size_t)m) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    if (reweights < 1) reweights = 1;
    if (reweights > 2) reweights = 2;
    // HQS-style strength schedule: decay in [0,1] scales the RED blend per
    // outer pass (strong early, gentle late, DPIR practice); 1.0 (default)
    // keeps every pass identical to the legacy path.
    if (!(red_decay >= 0.0f && red_decay <= 1.0f)) red_decay = 1.0f;
    const int red_kind = cs_onnx_kind_for(red_denoiser);
    const std::string red_path = cs_red_model_for(red_denoiser, dncnn_model, red_model);
    // Full LR per-pass budget: the warm start below is already at full scale
    // (not the /10 generic-ref convention — the solved LR tile IS a good
    // estimate), so every step refines rather than regrows coefficients.
    // An explicit --fista-iters override is honored as-is.
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    const cs_fista_basis bx = cs_make_basis(Hr, Wc, CS_BASIS_DCT, wscale);

    // Y measurements: OpenCV BGR2YCrCb luma weights over the sampled BGR
    // triplets (same weights the Y plane below was built with).
    std::vector<float> bB, bG, bR, b((size_t)m);
    cs_extract_channel_measurements(pixel_measurements, 0, m, bB);
    cs_extract_channel_measurements(pixel_measurements, 1, m, bG);
    cs_extract_channel_measurements(pixel_measurements, 2, m, bR);
    for (int k = 0; k < m; ++k) {
        b[(size_t)k] = 0.114f * bB[(size_t)k] + 0.587f * bG[(size_t)k] + 0.299f * bR[(size_t)k];
    }
    // AVIR image: always needed for the final Cr/Cb planes (luminance-only
    // solve: SR lives in Y, chroma is smooth enough to interpolate, so only
    // one HR FISTA solve runs instead of three). It also supplies the
    // default Y warm start; --sr-init cascade replaces the warm start below
    // with the multiscale binned-FISTA chain (same anchor weight).
    cv::Mat init_hr;
    cs_upscale_2x_avir(lr_tile, init_hr);
    if (init_hr.empty() || init_hr.rows != Hr || init_hr.cols != Wc) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    cv::Mat init_ycc;
    cv::cvtColor(init_hr, init_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> init_ycc_chs;
    cv::split(init_ycc, init_ycc_chs);

    // Scratch for the single luma solve.
    std::vector<float> y((size_t)n_hr), x_prev((size_t)n_hr), grad((size_t)n_hr),
        pix_hr((size_t)n_hr), up_hr((size_t)n_hr);
    std::vector<float> lr_buf((size_t)n_lr), sc_lr((size_t)n_lr);
    std::vector<float> w((size_t)n_hr, 1.0f);
    cv::Mat xf;
    if (sr_cascade) {
        // Multiscale warm start: binned coarse-to-fine FISTA chain lands HR
        // pixels; a non-finite/degenerate cascade falls back to AVIR below.
        std::vector<float> cas((size_t)n_hr, 0.0f);
        cs_sr_cascade_init(b.data(), ri_x.data(), ri_y.data(), m,
            Lr, Lc, coef, fista_iters, wscale, cas.data());
        bool ok = false;
        for (float v : cas) if (std::isfinite(v)) { ok = true; break; }
        if (ok) {
            cv::Mat casM(Hr, Wc, CV_32F, cas.data());
            casM.convertTo(xf, CV_32F);
            cs_dct_forward(xf, cs_default_dct_block());
        }
        else {
            init_ycc_chs[0].convertTo(xf, CV_32F, 1.0 / 255.0);
            cs_dct_forward(xf, cs_default_dct_block());
        }
    }
    else {
        init_ycc_chs[0].convertTo(xf, CV_32F, 1.0 / 255.0);
        cs_dct_forward(xf, cs_default_dct_block());
    }
    // Full-scale warm start (no /10): the AVIR upscale of the solved tile
    // is already close to the truth, so FISTA refines in place. The /10
    // convention is only for generic gray references that must regrow.
    if (!xf.isContinuous()) xf = xf.clone();
    float* x = (float*)xf.data;
    const std::vector<float> anchor(x, x + n_hr);
    // The LR samples constrain only one quarter of the HR dimensions.
    // A modest anchor protects unmeasured detail; sparse corrections
    // remove sample inconsistencies without re-solving the entire image.
    constexpr float anchor_weight = 0.25f;
    const float correction_coef = coef * 0.1f;
    std::fill(w.begin(), w.end(), 1.0f);
    // HR-detail anchors as in-solve data terms: global output-geometry
    // indices map to tile-local HR pixels via the tile origin; anchors
    // outside this tile are skipped (neighbor tiles own them). The list is
    // also the RED-modulation support (see hr_red_map below), so it fills
    // whenever anchors are present; the core guards the data-term weight
    // and the map builder guards the RED radius independently.
    std::vector<int> tile_hr_idx;
    std::vector<float> tile_hr_val;
    if (hr_sr.anchors && !hr_sr.anchors->empty() && hr_sr.stride > 0) {
        tile_hr_idx.reserve(hr_sr.anchors->size());
        tile_hr_val.reserve(hr_sr.anchors->size());
        for (const CsHrSample& s : *hr_sr.anchors) {
            const int gx = (int)(s.idx % (uint32_t)hr_sr.stride);
            const int gy = (int)(s.idx / (uint32_t)hr_sr.stride);
            const int lx = gx - hr_sr.ox, ly = gy - hr_sr.oy;
            if (lx < 0 || ly < 0 || lx >= Wc || ly >= Hr) continue;
            tile_hr_idx.push_back(ly * Wc + lx);
            tile_hr_val.push_back(s.y * (1.0f / 255.0f));
        }
    }
    // Anchor-weighted RED: per-pixel blend map from the same tile-local
    // anchors (empty = legacy uniform RED). Built once; the per-pass
    // decayed strength multiplies it at each application.
    const cv::Mat hr_red_map = (red > 0.0f) ?
        cs_hr_red_map(Hr, Wc, tile_hr_idx, hr_sr.red_radius) : cv::Mat();
    for (int r = 0; r < reweights; ++r) {
        cs_sr_core_single(x, b.data(), anchor.data(), anchor_weight,
            ri_x.data(), ri_y.data(), m, Hr, Wc, Lr, Lc,
            w.data(), correction_coef, tv, inner, bx,
            y, x_prev, grad, pix_hr, lr_buf, sc_lr, up_hr,
            tile_hr_idx.data(), tile_hr_val.data(), (int)tile_hr_idx.size(),
            hr_sr.weight);
        // RED-lite: pull the pass solution toward the denoiser manifold
        // before reweighting, so the reweights protect denoised structure.
        // red = 0 skips exactly (bit-identical to the unregularized path).
        // ONNX denoisers (dncnn/scunet/restormer) denoise the merged BGR
        // tile (the color models need all three channels); the Y plane is
        // re-extracted for the next pass. A failed run keeps the
        // un-denoised pass (graceful).
        const float red_r = (red_decay >= 1.0f || r == 0)
            ? red : red * std::pow(red_decay, (float)r);
        if (red_r > 0.0f && red_kind >= 0) {
            cv::Mat px(Hr, Wc, CV_32F, x);
            cs_dct_inverse(px, cs_default_dct_block());
            cv::Mat y8;
            {
                cv::Mat tmp = px * 255.0f;
                tmp.convertTo(y8, CV_8U);
            }
            std::vector<cv::Mat> bgr_ycc{ y8, init_ycc_chs[1], init_ycc_chs[2] };
            cv::Mat bgr_hr, dn_hr;
            cv::Mat merged_ycc;
            cv::merge(bgr_ycc, merged_ycc);
            cv::cvtColor(merged_ycc, bgr_hr, cv::COLOR_YCrCb2BGR);
            const float rl = red_r > 1.0f ? 1.0f : red_r;
            if (cs_onnx_denoise_bgr(bgr_hr, dn_hr, red_path, red_kind)) {
                cv::Mat dn_ycc;
                cv::cvtColor(dn_hr, dn_ycc, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> dn_chs;
                cv::split(dn_ycc, dn_chs);
                cv::Mat dnf;
                dn_chs[0].convertTo(dnf, CV_32F, 1.0 / 255.0);
                if (hr_red_map.empty()) {
                    px -= rl * (px - dnf);
                } else {
                    px -= rl * (px - dnf).mul(hr_red_map);
                }
            }
            cs_dct_forward(px, cs_default_dct_block());
        }
        else {
            if (hr_red_map.empty()) {
                cs_sr_red_proximal(x, Hr, Wc, red_r);
            } else {
                cs_sr_red_proximal_weighted(x, Hr, Wc, red_r, hr_red_map);
            }
        }
        if (r + 1 < reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n_hr; ++i) {
                const float av = std::fabs(x[i] - anchor[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n_hr; ++i)
                w[(size_t)i] = eps / (std::fabs(x[i] - anchor[i]) + eps);
        }
    }
    cv::Mat plane(Hr, Wc, CV_32F, x);
    cs_dct_inverse(plane, cs_default_dct_block());
    plane = plane * 255.0f;
    cv::Mat out_y;
    plane.convertTo(out_y, CV_8U);
    std::vector<cv::Mat> out_ycc{ out_y, init_ycc_chs[1], init_ycc_chs[2] };
    cv::Mat merged;
    cv::merge(out_ycc, merged);
    cv::cvtColor(merged, hr_out, cv::COLOR_YCrCb2BGR);
}

// Single-stage SR tile wrapper for the fused decrypt path: same I/O
// contract as cs_sr_upscale_tile_2x (solved CV_8UC3 tile in, 2x CV_8UC3 out)
// but the HR luma is solved directly from the LR samples (no LR-solve warm
// start inside the coefficients). Chroma comes from the AVIR upscale, so one
// HR solve runs instead of three. Degenerate inputs AVIR-fallback inline so
// the caller never loses a tile.
void cs_sr_direct_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int reweights, int fista_iters, int basis, float wscale,
    cv::Mat& hr_out, bool sr_cascade)
{
    hr_out.release();
    if (lr_tile.empty() || lr_tile.type() != CV_8UC3) return;
    const int Lr = lr_tile.rows, Lc = lr_tile.cols;
    if (Lr < 8 || Lc < 8) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    if (basis != CS_BASIS_DCT || ri_x.size() != ri_y.size() || ri_x.empty()) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    const int Hr = Lr * 2, Wc = Lc * 2;
    const int n_hr = Hr * Wc;
    const int m = (int)ri_x.size();
    if (pixel_measurements.total() < (size_t)CS_HEADER_PIXELS + (size_t)m) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    // AVIR supplies the chroma planes (luma-only direct solve).
    cv::Mat avir_hr;
    cs_upscale_2x_avir(lr_tile, avir_hr);
    if (avir_hr.empty() || avir_hr.rows != Hr || avir_hr.cols != Wc) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    cv::Mat avir_ycc;
    cv::cvtColor(avir_hr, avir_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> avir_chs;
    cv::split(avir_ycc, avir_chs);
    // Luma samples under the BGR2YCrCb weights (same as the two-stage path).
    std::vector<float> bB, bG, bR;
    std::vector<float> b((size_t)m);
    cs_extract_channel_measurements(pixel_measurements, 0, m, bB);
    cs_extract_channel_measurements(pixel_measurements, 1, m, bG);
    cs_extract_channel_measurements(pixel_measurements, 2, m, bR);
    for (int k = 0; k < m; ++k)
        b[(size_t)k] = 0.114f * bB[(size_t)k] + 0.587f * bG[(size_t)k] + 0.299f * bR[(size_t)k];
    std::vector<float> hr((size_t)n_hr, 0.0f);
    if (sr_cascade) {
        std::vector<float> cas((size_t)n_hr, 0.0f);
        cs_sr_cascade_init(b.data(), ri_x.data(), ri_y.data(), m,
            Lr, Lc, coef, fista_iters, wscale, cas.data());
        bool ok = false;
        for (float v : cas) if (std::isfinite(v)) { ok = true; break; }
        if (ok)
            cs_sr_direct_luma(b.data(), ri_x.data(), ri_y.data(), m,
                Lr, Lc, coef, tv, iterations, reweights, fista_iters,
                wscale, 2, 0.25f, hr.data(), cas.data());
        else
            cs_sr_direct_luma(b.data(), ri_x.data(), ri_y.data(), m,
                Lr, Lc, coef, tv, iterations, reweights, fista_iters,
                wscale, 0, 0.0f, hr.data());
    }
    else {
        // Cold start from zeros with no anchor (measured: 25.27 dB fixture;
        // the demosaic scatter + anchor converges dark: 15.98 dB there,
        // worse in production since the anchor drags unobserved pixels).
        cs_sr_direct_luma(b.data(), ri_x.data(), ri_y.data(), m,
            Lr, Lc, coef, tv, iterations, reweights, fista_iters,
            wscale, 0, 0.0f, hr.data());
    }
    bool finite = false;
    for (float v : hr) if (std::isfinite(v)) { finite = true; break; }
    if (!finite) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    cv::Mat plane(Hr, Wc, CV_32F, hr.data());
    cv::Mat y8 = plane * 255.0f;
    y8.convertTo(y8, CV_8U);
    std::vector<cv::Mat> out_ycc{ y8, avir_chs[1], avir_chs[2] };
    cv::Mat merged, out;
    cv::merge(out_ycc, merged);
    cv::cvtColor(merged, out, cv::COLOR_YCrCb2BGR);
    if (out.empty() || out.rows != Hr || out.cols != Wc)
        cs_upscale_2x_avir(lr_tile, hr_out);
    else
        hr_out = std::move(out);
}

// ---------------------------------------------------------------------------
// Untrained prior (Deep Decoder style) for single-tile 2x refinement.
// Tiny generator, learned per tile from a fixed seed against that tile's
// own samples: latent tensor -> [nearest-upsample to explicit level size,
// per-pixel channel mix + bias, ReLU] x d -> linear + sigmoid luma.
// Nearest (not bilinear) upsampling has an exact transpose (block sums),
// so backprop is exact by construction; Adam optimizes data (2x2-box
// samples) + AVIR-anchor loss. Underparameterized vs the HR pixels on
// production tiles (overfit barrier); serial per tile, tiles parallelize
// upstream. Deterministic (fixed seed, no RNG elsewhere).
static void cs_bilinear_up(const float* src, int sh, int sw,
    float* dst, int dh, int dw);
namespace {
struct CsDipLevel {
    int h = 0, w = 0;             // activation spatial dims (post-upsample)
    std::vector<float> up;        // upsampled input (h*w*k, kept for backward)
    std::vector<float> pre;       // pre-activation (h*w*k)
    std::vector<float> post;      // post-activation (h*w*k)
};
// Bilinear upsample of a pixel-major multi-channel plane via cs_bilinear_up.
static void cs_dip_upsample(const float* src, int sh, int sw,
    float* dst, int dh, int dw, int k)
{
    std::vector<float> plane((size_t)sh * sw), plane_up((size_t)dh * dw);
    for (int c = 0; c < k; ++c) {
        for (int i = 0; i < sh * sw; ++i) plane[(size_t)i] = src[(size_t)i * k + c];
        cs_bilinear_up(plane.data(), sh, sw, plane_up.data(), dh, dw);
        for (int i = 0; i < dh * dw; ++i) dst[(size_t)i * k + c] = plane_up[(size_t)i];
    }
}
// Exact transpose of cs_bilinear_up's map (same index/weight computation,
// scattering): accumulates (+=) into dlo, which the caller zeroes.
static void cs_dip_upsample_adj(const float* dhi, int dh, int dw,
    float* dlo, int sh, int sw, int k)
{
    const double sy = (double)sh / dh, sx = (double)sw / dw;
    for (int i = 0; i < dh; ++i) {
        double fy = (i + 0.5) * sy - 0.5;
        if (fy < 0.0) fy = 0.0;
        if (fy > sh - 1.0) fy = (double)(sh - 1);
        int y0 = (int)fy;
        double wy = fy - y0;
        if (y0 >= sh - 1) { y0 = sh - 2; wy = 1.0; }
        for (int j = 0; j < dw; ++j) {
            double fx = (j + 0.5) * sx - 0.5;
            if (fx < 0.0) fx = 0.0;
            if (fx > sw - 1.0) fx = (double)(sw - 1);
            int x0 = (int)fx;
            double wx = fx - x0;
            if (x0 >= sw - 1) { x0 = sw - 2; wx = 1.0; }
            const float* g = dhi + ((size_t)i * dw + j) * (size_t)k;
            float* a = dlo + (size_t)y0 * sw * k + (size_t)x0 * k;
            float* b2 = dlo + (size_t)y0 * sw * k + (size_t)(x0 + 1) * k;
            float* c = dlo + ((size_t)(y0 + 1) * sw + x0) * k;
            float* d = dlo + ((size_t)(y0 + 1) * sw + x0 + 1) * k;
            const float w00 = (float)((1 - wy) * (1 - wx));
            const float w01 = (float)((1 - wy) * wx);
            const float w10 = (float)(wy * (1 - wx));
            const float w11 = (float)(wy * wx);
            for (int ch = 0; ch < k; ++ch) {
                const float gv = g[ch];
                a[ch] += gv * w00;
                b2[ch] += gv * w01;
                c[ch] += gv * w10;
                d[ch] += gv * w11;
            }
        }
    }
}
} // namespace

void cs_dip_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& hr_out, float anchor_w, int iters, float step)
{
    hr_out.release();
    if (lr_tile.empty() || lr_tile.type() != CV_8UC3) return;
    const int Lr = lr_tile.rows, Lc = lr_tile.cols;
    if (Lr < 8 || Lc < 8) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    if (ri_x.size() != ri_y.size() || ri_x.empty()) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    const int Hr = Lr * 2, Wc = Lc * 2;
    const int n_hr = Hr * Wc;
    const int m = (int)ri_x.size();
    if (pixel_measurements.total() < (size_t)CS_HEADER_PIXELS + (size_t)m) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    if (!(anchor_w >= 0.0f) || iters < 1 || !(step > 0.0f)) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    // AVIR supplies the chroma planes (luma-only generator solve) and the
    // anchor image.
    cv::Mat avir_hr;
    cs_upscale_2x_avir(lr_tile, avir_hr);
    if (avir_hr.empty() || avir_hr.rows != Hr || avir_hr.cols != Wc) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    cv::Mat avir_ycc;
    cv::cvtColor(avir_hr, avir_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> avir_chs;
    cv::split(avir_ycc, avir_chs);
    cv::Mat avir_yf;
    avir_chs[0].convertTo(avir_yf, CV_32F, 1.0 / 255.0);
    if (!avir_yf.isContinuous()) avir_yf = avir_yf.clone();
    // Luma samples under the BGR2YCrCb weights (same as the FISTA paths).
    std::vector<float> bB, bG, bR, b((size_t)m);
    cs_extract_channel_measurements(pixel_measurements, 0, m, bB);
    cs_extract_channel_measurements(pixel_measurements, 1, m, bG);
    cs_extract_channel_measurements(pixel_measurements, 2, m, bR);
    for (int k = 0; k < m; ++k)
        b[(size_t)k] = 0.114f * bB[(size_t)k] + 0.587f * bG[(size_t)k] + 0.299f * bR[(size_t)k];

    // Net geometry: d=3 hidden blocks from h0xw0, k channels; explicit
    // level sizes so non-multiple-of-8 tiles still land exactly on Hr x Wc.
    constexpr int DK = 24, DD = 3;
    const int h0 = (std::max)(2, Hr / 8), w0 = (std::max)(2, Wc / 8);
    std::vector<int> lh(DD + 1), lw(DD + 1);
    lh[0] = h0; lw[0] = w0;
    for (int l = 1; l <= DD; ++l) {
        lh[l] = (l == DD) ? Hr : (std::min)(Hr, lh[l - 1] * 2);
        lw[l] = (l == DD) ? Wc : (std::min)(Wc, lw[l - 1] * 2);
    }
    // Flat params: z0 | per-level (W kxk, b k) | final (wf k, bf 1).
    const size_t n_z0 = (size_t)h0 * w0 * DK;
    const size_t n_lvl = (size_t)DK * DK + DK;
    std::vector<size_t> off(DD + 2, 0);
    off[0] = 0;
    for (int l = 0; l < DD; ++l) off[l + 1] = off[l] + ((l == 0) ? n_z0 : n_lvl);
    const size_t n_par = off[DD] + n_lvl + (size_t)DK + 1;
    std::vector<float> P(n_par), M(n_par, 0.0f), V(n_par, 0.0f), G(n_par);
    {
        std::mt19937 rng(0xD1A5u);
        const float wsd = std::sqrt(2.0f / (float)DK);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t i = 0; i < n_z0; ++i) P[off[0] + i] = 0.5f * gauss(rng);
        for (int l = 0; l < DD; ++l) {
            float* W = &P[off[l + 1]];
            for (int j = 0; j < DK * DK; ++j) W[j] = wsd * gauss(rng);
            // biases stay 0
        }
        float* wf = &P[off[DD] + n_lvl];
        // Residual head: zero output weights, so the generator starts
        // exactly at the AVIR luma and only learns what the samples add.
        for (int j = 0; j < DK; ++j) { (void)gauss; wf[j] = 0.0f; }
        // final bias stays 0 (sigmoid(0) = mid-gray start)
    }
    std::vector<CsDipLevel> lv(DD);
    for (int l = 0; l < DD; ++l) {
        lv[l].h = lh[l + 1]; lv[l].w = lw[l + 1];
        lv[l].up.assign((size_t)lh[l + 1] * lw[l + 1] * DK, 0.0f);
        lv[l].pre.assign((size_t)lh[l + 1] * lw[l + 1] * DK, 0.0f);
        lv[l].post.assign((size_t)lh[l + 1] * lw[l + 1] * DK, 0.0f);
    }
    std::vector<float> img((size_t)n_hr), dimg((size_t)n_hr);
    const float* avir_p = (const float*)avir_yf.data;
    auto forward = [&]() {
        // level inputs: z0 for l=0, previous post otherwise; bilinearly
        // upsampled to the level size, then per-pixel mix + leaky ReLU
        // (slope 0.2: no dead units).
        for (int l = 0; l < DD; ++l) {
            const int ph = lh[l], pw = lw[l], ch = lh[l + 1], cw = lw[l + 1];
            const float* src = (l == 0) ? P.data() + off[0] : lv[l - 1].post.data();
            cs_dip_upsample(src, ph, pw, lv[l].up.data(), ch, cw, DK);
            const float* W = P.data() + off[l + 1];
            const float* bb = W + (size_t)DK * DK;
            float* pre = lv[l].pre.data();
            float* post = lv[l].post.data();
            for (int i = 0; i < ch * cw; ++i) {
                const float* xs = lv[l].up.data() + (size_t)i * DK;
                float* pr = pre + (size_t)i * DK;
                float* po = post + (size_t)i * DK;
                for (int j = 0; j < DK; ++j) {
                    const float* Wj = W + (size_t)j * DK;
                    float a = bb[j];
                    for (int ii = 0; ii < DK; ++ii) a += Wj[ii] * xs[ii];
                    pr[j] = a;
                    po[j] = a > 0.0f ? a : 0.2f * a;
                }
            }
        }
        // final linear + sigmoid luma
        const float* top = lv[DD - 1].post.data();
        const float* wf = P.data() + off[DD] + n_lvl;
        const float bf = P[off[DD] + n_lvl + DK];
        for (int i = 0; i < n_hr; ++i) {
            const float* x = top + (size_t)i * DK;
            float a = bf;
            for (int j = 0; j < DK; ++j) a += wf[j] * x[j];
            img[(size_t)i] = avir_p[i] + a;  // residual over AVIR luma
        }
    };
    // Adam fit against samples + anchor.
    constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float b1t = b1, b2t = b2;
    const bool dbg = std::getenv("CS_DIP_DEBUG") != nullptr;
    // Deterministic hold-out: every fifth measured sample is validation
    // only (never enters the gradient). Early stopping watches it.
    auto is_val = [](int k) { return (k % 5) == 4; };
    auto loss_now = [&](bool val) {
        double dl = 0.0;
        for (int k = 0; k < m; ++k) {
            if (is_val(k) != val) continue;
            const int lr_r = ri_x[k], lr_c = ri_y[k];
            if ((unsigned)lr_r >= (unsigned)Lr || (unsigned)lr_c >= (unsigned)Lc) continue;
            float pred = 0.0f;
            for (int dr = 0; dr < 2; ++dr)
                for (int dc = 0; dc < 2; ++dc) {
                    const int hr_r = lr_r * 2 + dr, hr_c = lr_c * 2 + dc;
                    if ((unsigned)hr_r >= (unsigned)Hr || (unsigned)hr_c >= (unsigned)Wc) continue;
                    pred += img[(size_t)hr_r * Wc + hr_c];
                }
            pred *= 0.25f;
            const double e = pred - b[(size_t)k];
            dl += e * e;
        }
        return dl / (m > 0 ? m : 1);
    };
    std::vector<float> gimg((size_t)n_hr);
    if (dbg) { forward(); std::fprintf(stderr, "[dip] loss0 train=%.6f val=%.6f\n", loss_now(false), loss_now(true)); }
    std::vector<float> bestP;
    double best_val = 1e30;
    int since_best = 0;
    for (int it = 0; it < iters; ++it) {
        forward();
        // dL/dG: sample term (2x2-mean adjoint: replicate/4 at measured
        // blocks, mean over m) + anchor term (mean over n). Training
        // samples only; validation samples are excluded from the gradient.
        std::fill(gimg.begin(), gimg.end(), 0.0f);
        for (int k = 0; k < m; ++k) {
            if (is_val(k)) continue;
            const int lr_r = ri_x[k], lr_c = ri_y[k];
            if ((unsigned)lr_r >= (unsigned)Lr || (unsigned)lr_c >= (unsigned)Lc) continue;
            float pred = 0.0f;
            for (int dr = 0; dr < 2; ++dr)
                for (int dc = 0; dc < 2; ++dc) {
                    const int hr_r = lr_r * 2 + dr, hr_c = lr_c * 2 + dc;
                    if ((unsigned)hr_r >= (unsigned)Hr || (unsigned)hr_c >= (unsigned)Wc) continue;
                    pred += img[(size_t)hr_r * Wc + hr_c];
                }
            pred *= 0.25f;
            const float e = 2.0f * (pred - b[(size_t)k]) / (float)(m > 0 ? m : 1) * 0.25f;
            for (int dr = 0; dr < 2; ++dr)
                for (int dc = 0; dc < 2; ++dc) {
                    const int hr_r = lr_r * 2 + dr, hr_c = lr_c * 2 + dc;
                    if ((unsigned)hr_r >= (unsigned)Hr || (unsigned)hr_c >= (unsigned)Wc) continue;
                    gimg[(size_t)hr_r * Wc + hr_c] += e;
                }
        }
        if (anchor_w > 0.0f) {
            const float aw = 2.0f * anchor_w / (float)n_hr;
            for (int i = 0; i < n_hr; ++i) gimg[(size_t)i] += aw * (img[(size_t)i] - avir_p[i]);
        }
        // backward: sigmoid -> final linear -> levels (top-down), with
        // exact-transpose nearest downsampling between levels. dcur holds
        // the delta at the current (high) resolution; each level consumes
        // it, accumulates W/b grads, and scatters into a fresh low-res
        // accumulator for the next level down.
        std::fill(G.begin(), G.end(), 0.0f);
        std::vector<float> dcur((size_t)n_hr * DK);
        for (int i = 0; i < n_hr; ++i) {
            // identity output (residual head): no sigmoid derivative
            const float ds = gimg[(size_t)i];
            const float* x = lv[DD - 1].post.data() + (size_t)i * DK;
            float* dlt = dcur.data() + (size_t)i * DK;
            float* gwf = G.data() + off[DD] + n_lvl;
            for (int j = 0; j < DK; ++j) {
                gwf[j] += ds * x[j];
                dlt[j] = ds * (P[off[DD] + n_lvl + (size_t)j]);
            }
            G[off[DD] + n_lvl + DK] += ds;
        }
        for (int l = DD - 1; l >= 0; --l) {
            const int ch = lh[l + 1], cw = lw[l + 1];
            const int ph = lh[l], pw = lw[l];
            const float* pre = lv[l].pre.data();
            const float* xs = lv[l].up.data();
            // W stays frozen for the whole level: grads accumulate into G
            // (separate buffer); Adam below is the only writer of P.
            const float* W = P.data() + off[l + 1];
            float* GW = G.data() + off[l + 1];
            float* GBl = GW + (size_t)DK * DK;
            // dup: delta pushed through the masked mix, still at level size;
            // transposed below into the low-res accumulator.
            std::vector<float> dup((size_t)ch * cw * DK);
            for (int i = 0; i < ch * cw; ++i) {
                const float* dlt = dcur.data() + (size_t)i * DK;
                const float* pr = pre + (size_t)i * DK;
                const float* xv = xs + (size_t)i * DK;
                float* du = dup.data() + (size_t)i * DK;
                for (int ii = 0; ii < DK; ++ii) {
                    float s = 0.0f;
                    for (int j = 0; j < DK; ++j) {
                        const float g = (pr[j] > 0.0f) ? dlt[j] : 0.2f * dlt[j];
                        s += W[(size_t)j * DK + ii] * g;
                    }
                    du[ii] = s;
                }
                for (int j = 0; j < DK; ++j) {
                    const float g = (pr[j] > 0.0f) ? dlt[j] : 0.2f * dlt[j];
                    GBl[j] += g;
                    for (int ii = 0; ii < DK; ++ii) GW[(size_t)j * DK + ii] += g * xv[ii];
                }
            }
            std::vector<float> dlow((size_t)ph * pw * DK, 0.0f);
            cs_dip_upsample_adj(dup.data(), ch, cw, dlow.data(), ph, pw, DK);
            dcur.swap(dlow);
        }
        // Latent tensor gradient: the level-0 accumulator left in dcur.
        std::memcpy(G.data() + off[0], dcur.data(), n_z0 * sizeof(float));
        // Adam step over all params.
        for (size_t i = 0; i < n_par; ++i) {
            M[i] = b1 * M[i] + (1.0f - b1) * G[i];
            V[i] = b2 * V[i] + (1.0f - b2) * G[i] * G[i];
            const float mh = M[i] / (1.0f - b1t);
            const float vh = V[i] / (1.0f - b2t);
            P[i] -= step * mh / (std::sqrt(vh) + eps);
        }
        b1t *= b1;
        b2t *= b2;
        // Early stopping on held-out samples (the DIP regularizer): the
        // fit keeps shrinking the training loss well past the point where
        // the unseen-sample error starts rising (it fits sample noise).
        // Keep the weights of the best validation step; stop after 100
        // steps without improvement.
        if ((it + 1) % 25 == 0) {
            const double vl = loss_now(true);
            if (dbg) std::fprintf(stderr, "[dip] it=%d train=%.6f val=%.6f\n",
                it + 1, loss_now(false), vl);
            if (vl < best_val * (1.0 - 1e-3)) {
                best_val = vl;
                bestP = P;
                since_best = 0;
            } else {
                since_best += 25;
                if (it + 1 >= 50 && since_best >= 100) {
                    if (dbg) std::fprintf(stderr, "[dip] early stop at it=%d\n", it + 1);
                    break;
                }
            }
        }
    }
    if (!bestP.empty()) {
        P = bestP;
        forward();  // img is now the best-validation generator output
    }
    if (dbg) std::fprintf(stderr, "[dip] best val=%.6f\n", best_val);
    bool finite = false;
    for (float v : img) if (std::isfinite(v)) { finite = true; break; }
    if (!finite) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    cv::Mat plane(Hr, Wc, CV_32F, img.data());
    cv::Mat y8 = plane * 255.0f;
    y8.convertTo(y8, CV_8U);
    std::vector<cv::Mat> out_ycc{ y8, avir_chs[1], avir_chs[2] };
    cv::Mat merged, out;
    cv::merge(out_ycc, merged);
    cv::cvtColor(merged, out, cv::COLOR_YCrCb2BGR);
    if (out.empty() || out.rows != Hr || out.cols != Wc)
        cs_upscale_2x_avir(lr_tile, hr_out);
    else
        hr_out = std::move(out);
}

// ---------------------------------------------------------------------------
// Single-stage (integrated) HR solve: full HR-coefficient FISTA against the
// LR samples with NO LR-solve warm start. init_mode 0 = zeros (pure
// single-stage); 1 = AVIR-upscaled zero-filled scatter (cheap demosaic,
// also used as the anchor when anchor_w > 0). Exists to measure whether
// collapsing the two-stage pipeline (LR solve -> 8U quantize -> HR refine)
// into one float-consistent solve wins on quality/time. Caller sizes inner
// via iterations/fista_iters exactly like the two-stage path, so budgets
// are comparable by construction.
void cs_sr_direct_luma(const float* b, const int* rix, const int* riy, int m,
    int Lr, int Lc, float coef, float tv, int iterations, int reweights,
    int fista_iters, float wscale, int init_mode, float anchor_w,
    float* hr_out, const float* warm_px)
{
    const int Hr = Lr * 2, Wc = Lc * 2;
    const int n_hr = Hr * Wc;
    if (!b || !rix || !riy || m <= 0 || !hr_out) return;
    if (reweights < 1) reweights = 1;
    if (reweights > 2) reweights = 2;
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    const cs_fista_basis bx = cs_make_basis(Hr, Wc, CS_BASIS_DCT, wscale);

    std::vector<float> y((size_t)n_hr), x_prev((size_t)n_hr), grad((size_t)n_hr),
        pix_hr((size_t)n_hr), up_hr((size_t)n_hr);
    std::vector<float> lr_buf((size_t)Lr * Lc), sc_lr((size_t)Lr * Lc);
    std::vector<float> w((size_t)n_hr, 1.0f);
    std::vector<float> x((size_t)n_hr, 0.0f), anchor((size_t)n_hr, 0.0f);
    if (init_mode == 1) {
        // scatter samples into a zero LR tile, upscale, DCT: cheap demosaic
        std::vector<float> lr((size_t)Lr * Lc, 0.0f);
        const int n_lr = Lr * Lc;
        for (int k = 0; k < m; ++k) {
            const int idx = rix[k] * Lc + riy[k];
            if ((unsigned)idx < (unsigned)n_lr) lr[(size_t)idx] = b[k];
        }
        cv::Mat lrm(Lr, Lc, CV_32F, lr.data());
        cv::Mat hrm;
        cv::resize(lrm, hrm, cv::Size(Wc, Hr), 0, 0, cv::INTER_LINEAR);
        cs_dct_forward(hrm, cs_default_dct_block());
        std::memcpy(x.data(), hrm.data, sizeof(float) * (size_t)n_hr);
        anchor = x;
    }
    else if (init_mode == 2 && warm_px) {
        // external float warm start (Hr x Wc pixels, [0,1]); also anchored
        cv::Mat wm(Hr, Wc, CV_32F, (void*)warm_px);
        cv::Mat xm;
        wm.convertTo(xm, CV_32F);
        cs_dct_forward(xm, cs_default_dct_block());
        std::memcpy(x.data(), xm.data, sizeof(float) * (size_t)n_hr);
        anchor = x;
    }
    // correction-scale lambda mirrors the two-stage path (coef*0.1)
    const float lambda = coef * 0.1f;
    for (int r = 0; r < reweights; ++r) {
        cs_sr_core_single(x.data(), b, anchor.data(), anchor_w,
            rix, riy, m, Hr, Wc, Lr, Lc, w.data(), lambda, tv, inner, bx,
            y, x_prev, grad, pix_hr, lr_buf, sc_lr, up_hr,
            nullptr, nullptr, 0, 0.0f);
        if (r + 1 < reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n_hr; ++i) {
                const float av = std::fabs(x[(size_t)i] - anchor[(size_t)i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n_hr; ++i)
                w[(size_t)i] = eps / (std::fabs(x[(size_t)i] - anchor[(size_t)i]) + eps);
        }
    }
    cv::Mat plane(Hr, Wc, CV_32F, x.data());
    cs_dct_inverse(plane, cs_default_dct_block());
    std::memcpy(hr_out, plane.data, sizeof(float) * (size_t)n_hr);
}

// ---------------------------------------------------------------------------
// Multiscale cascade warm start for single-stage SR. The LR samples are
// binned into progressively coarser cells (cell value = sample mean,
// unobserved cells absent from the data term); each level runs a short
// unweighted DCT-FISTA solve (reusing cs_fista_core_single: at coarse
// scales the problem is tiny and well-conditioned, so a handful of steps
// from any start converges); the solution is bilinearly upsampled to seed
// the next finer level. The chain ends with HR pixels for a direct solve.
// This converts one impossible cold start into a chain of easy warm starts
// (LapSRN/pyramid philosophy applied to the solver, not the network); the
// thumbnail some containers already ship (HF-focus) is the same idea at
// image scale and could seed level 0 in the future.
// Cost is negligible next to the final solve (grids shrink 4x per level;
// coarse iters fixed at 12).
static void cs_bilinear_up(const float* src, int sh, int sw,
    float* dst, int dh, int dw)
{
    // Forward-only bilinear upsample (no adjoint needed: warm-start
    // propagation only; mapping matches INTER_LINEAR convention).
    if (!src || !dst || sh < 2 || sw < 2 || dh <= 0 || dw <= 0) return;
    const double sy = (double)sh / dh, sx = (double)sw / dw;
    for (int i = 0; i < dh; ++i) {
        double fy = (i + 0.5) * sy - 0.5;
        if (fy < 0.0) fy = 0.0;
        if (fy > sh - 1.0) fy = (double)(sh - 1);
        int y0 = (int)fy;
        double wy = fy - y0;
        if (y0 >= sh - 1) { y0 = sh - 2; wy = 1.0; }
        for (int j = 0; j < dw; ++j) {
            double fx = (j + 0.5) * sx - 0.5;
            if (fx < 0.0) fx = 0.0;
            if (fx > sw - 1.0) fx = (double)(sw - 1);
            int x0 = (int)fx;
            double wx = fx - x0;
            if (x0 >= sw - 1) { x0 = sw - 2; wx = 1.0; }
            const float a = src[(size_t)y0 * sw + x0];
            const float b2 = src[(size_t)y0 * sw + x0 + 1];
            const float c = src[(size_t)(y0 + 1) * sw + x0];
            const float d = src[(size_t)(y0 + 1) * sw + x0 + 1];
            dst[(size_t)i * dw + j] = (float)(
                a * (1 - wy) * (1 - wx) + b2 * (1 - wy) * wx +
                c * wy * (1 - wx) + d * wy * wx);
        }
    }
}

void cs_sr_cascade_init(const float* b, const int* rix, const int* riy, int m,
    int Lr, int Lc, float coef, int fista_iters, float wscale, float* hr_init)
{
    const int Hr = Lr * 2, Wc = Lc * 2;
    const int n_hr = Hr * Wc;
    if (!b || !rix || !riy || m <= 0 || !hr_init || Lr < 8 || Lc < 8) {
        if (hr_init && Lr >= 8 && Lc >= 8) std::memset(hr_init, 0, sizeof(float) * (size_t)n_hr);
        return;
    }
    // coarse FISTA budget: explicit override or 12 fixed steps (coarse
    // levels converge in a handful from any start; fixed keeps the
    // cascade overhead negligible next to the final solve).
    const int coarse_iters = fista_iters > 0
        ? ((std::min)(64, (std::max)(4, fista_iters))) : 12;
    // level grids, coarse-first: halve (floor) while >= 16px, then the LR
    // grid itself. Each level bins the samples into its cells and runs one
    // short unweighted FISTA pass; the solution upscales to seed
    // the next level. A final bilinear step lands HR pixels (no solve).
    struct Lv { int h, w; };
    std::vector<Lv> levels;
    for (int h = Lr / 2, w = Lc / 2; h >= 16 && w >= 16; h /= 2, w /= 2)
        levels.push_back({ h, w });
    std::reverse(levels.begin(), levels.end());
    levels.push_back({ Lr, Lc });

    std::vector<float> cur;
    int ch = 0, cw = 0;
    for (size_t li = 0; li < levels.size(); ++li) {
        const int gh = levels[li].h, gw = levels[li].w;
        const int n = gh * gw;
        // bin samples into level cells (mean per observed cell)
        std::vector<double> sum((size_t)n, 0.0);
        std::vector<int> cnt((size_t)n, 0);
        for (int k = 0; k < m; ++k) {
            int i = (int)((long long)rix[k] * gh / Lr);
            int j = (int)((long long)riy[k] * gw / Lc);
            if ((unsigned)i >= (unsigned)gh || (unsigned)j >= (unsigned)gw) continue;
            sum[(size_t)i * gw + j] += b[k];
            cnt[(size_t)i * gw + j] += 1;
        }
        std::vector<int> ox, oy;
        std::vector<float> ob;
        ox.reserve((size_t)n);
        oy.reserve((size_t)n);
        ob.reserve((size_t)n);
        for (int i = 0; i < gh; ++i) {
            for (int j = 0; j < gw; ++j) {
                const int c = cnt[(size_t)i * gw + j];
                if (c > 0) {
                    ox.push_back(i);
                    oy.push_back(j);
                    ob.push_back((float)(sum[(size_t)i * gw + j] / c));
                }
            }
        }
        const int mo = (int)ob.size();
        std::vector<float> xc((size_t)n, 0.0f);
        if (mo > 0) {
            // warm start: previous level upsampled (zeros for level 0)
            if (ch > 0) {
                std::vector<float> up((size_t)n);
                cs_bilinear_up(cur.data(), ch, cw, up.data(), gh, gw);
                cv::Mat wm(gh, gw, CV_32F, up.data());
                cs_dct_forward(wm, cs_default_dct_block());
                std::memcpy(xc.data(), wm.data, sizeof(float) * (size_t)n);
            }
            const cs_fista_basis bx = cs_make_basis(gh, gw, CS_BASIS_DCT, wscale);
            std::vector<float> w((size_t)n, 1.0f), y((size_t)n), xp((size_t)n),
                grad((size_t)n), pix((size_t)n), sc((size_t)n);
            cs_fista_core_single(xc.data(), ob.data(), ox.data(), oy.data(), mo,
                gh, gw, w.data(), coef, 0.0f, coarse_iters, bx,
                y, xp, grad, pix, sc);
            cv::Mat pm(gh, gw, CV_32F, xc.data());
            cs_dct_inverse(pm, bx.dct_block);
            cur.assign((size_t)n, 0.0f);
            std::memcpy(cur.data(), pm.data, sizeof(float) * (size_t)n);
        }
        else if (ch > 0) {
            // unobserved level: propagate the upsampled previous solution
            std::vector<float> up((size_t)n, 0.0f);
            cs_bilinear_up(cur.data(), ch, cw, up.data(), gh, gw);
            cur.swap(up);
        }
        else {
            cur.assign((size_t)n, 0.0f);
        }
        ch = gh;
        cw = gw;
    }
    // final landing: LR pixels -> HR pixels (no solve at HR here)
    if (ch > 0 && !cur.empty()) {
        cs_bilinear_up(cur.data(), ch, cw, hr_init, Hr, Wc);
    }
    else {
        std::memset(hr_init, 0, sizeof(float) * (size_t)n_hr);
    }
}

// (Consensus-ADMM core cs_admm_core and its CPU fallback removed; FISTA is
// the only solver.)

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
                // simd-safe: w is local; p.x[012] are caller inputs.
                #pragma omp simd
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
            // simd-safe: w is local; p.x is a caller input.
            #pragma omp simd
            for (int i = 0; i < n; ++i)
                w[(size_t)i] = eps / (std::fabs(p.x[i]) + eps);
        }
    }
}

static const struct CsGpuFallbackRegFista {
    CsGpuFallbackRegFista() { cs_gpu::register_cpu_fallback_fista(&cs_gpu_cpu_fallback_fista); }
} cs_gpu_fallback_reg_fista;

// (ADMM channel wrapper removed with the solver.)

// (OWL-QN subsampled-chroma solve evaluate_coarse +
// reconstruct_color_channel_subchroma removed with the solver.)

std::vector<CsHrSample> cs_select_hr_detail(const cv::Mat& native_bgr,
    int out_w, int out_h, int count) {
    std::vector<CsHrSample> out;
    if (count <= 0 || native_bgr.empty() || native_bgr.type() != CV_8UC3)
        return out;
    if (out_w <= 0 || out_h <= 0) return out;
    if (count > CS_HRDETAIL_MAX_SAMPLES) count = CS_HRDETAIL_MAX_SAMPLES;
    cv::Mat ref = native_bgr;
    cv::Mat resized;
    if (native_bgr.cols != out_w || native_bgr.rows != out_h) {
        cv::resize(native_bgr, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        ref = resized;
    }
    cv::Mat gray, gx, gy, mag;
    cv::cvtColor(ref, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::addWeighted(cv::abs(gx), 1.0, cv::abs(gy), 1.0, 0.0, mag);
    // Per-cell top-2 bucketing (16x16) spreads anchors along edges instead
    // of clustering them on the single strongest contour.
    constexpr int CELL = 16, PER_CELL = 2;
    struct Cand { float g; int idx; };
    std::vector<Cand> pool;
    pool.reserve((size_t)out_w * out_h / 32);
    for (int cy0 = 0; cy0 < out_h; cy0 += CELL) {
        for (int cx0 = 0; cx0 < out_w; cx0 += CELL) {
            Cand best[PER_CELL] = { {-1.0f, -1}, {-1.0f, -1} };
            const int cy1 = (std::min)(cy0 + CELL, out_h);
            const int cx1 = (std::min)(cx0 + CELL, out_w);
            for (int y = cy0; y < cy1; ++y) {
                const float* mp = mag.ptr<float>(y);
                for (int x = cx0; x < cx1; ++x) {
                    const float g = mp[x];
                    if (g > best[0].g) {
                        best[1] = best[0];
                        best[0] = { g, y * out_w + x };
                    }
                    else if (g > best[1].g) {
                        best[1] = { g, y * out_w + x };
                    }
                }
            }
            for (int k = 0; k < PER_CELL; ++k)
                if (best[k].idx >= 0 && best[k].g > 0.0f) pool.push_back(best[k]);
        }
    }
    if (pool.empty()) return out;
    if ((int)pool.size() > count) {
        std::nth_element(pool.begin(), pool.begin() + count, pool.end(),
            [](const Cand& a, const Cand& b) { return a.g > b.g; });
        pool.resize((size_t)count);
    }
    // Deterministic order: strength descending, index ascending on ties.
    std::sort(pool.begin(), pool.end(), [](const Cand& a, const Cand& b) {
        if (a.g != b.g) return a.g > b.g;
        return a.idx < b.idx;
    });
    out.reserve(pool.size());
    for (const Cand& c : pool) {
        CsHrSample s;
        s.idx = (uint32_t)c.idx;
        s.y = gray.at<uint8_t>(c.idx / out_w, c.idx % out_w);
        out.push_back(s);
    }
    return out;
}

void cs_write_hr_payload(uint8_t* dst, const std::vector<CsHrSample>& s,
    const uint8_t key[32], const uint8_t iv[16]) {
    uint8_t* p = dst;
    for (const CsHrSample& sm : s) {
        p[0] = (uint8_t)(sm.idx & 0xff);
        p[1] = (uint8_t)((sm.idx >> 8) & 0xff);
        p[2] = (uint8_t)((sm.idx >> 16) & 0xff);
        p[3] = (uint8_t)((sm.idx >> 24) & 0xff);
        p[4] = sm.y;
        p[5] = 0;
        p += CS_HRDETAIL_SAMPLE_BYTES;
    }
    const size_t len = s.size() * CS_HRDETAIL_SAMPLE_BYTES;
    if (len == 0) return;
    uint8_t counter[CS_IV_BYTES];
    std::memcpy(counter, iv, CS_IV_BYTES);
    counter[0] ^= CS_HRDETAIL_TAG;
    if (!cs_aes256_ctr_xor(dst, len, key, counter)) {
        cs_wipe(counter, sizeof(counter));
        throw std::runtime_error("hr-detail payload encryption failed");
    }
    cs_wipe(counter, sizeof(counter));
}

bool cs_read_hr_payload(const uint8_t* src, size_t len, const uint8_t key[32],
    const uint8_t iv[16], std::vector<CsHrSample>& out) {
    out.clear();
    if (len == 0 || len % CS_HRDETAIL_SAMPLE_BYTES != 0) return false;
    const size_t n = len / CS_HRDETAIL_SAMPLE_BYTES;
    if (n > CS_HRDETAIL_MAX_SAMPLES) return false;
    std::vector<uint8_t> buf(src, src + len);
    uint8_t counter[CS_IV_BYTES];
    std::memcpy(counter, iv, CS_IV_BYTES);
    counter[0] ^= CS_HRDETAIL_TAG;
    if (!cs_aes256_ctr_xor(buf.data(), buf.size(), key, counter)) {
        cs_wipe(counter, sizeof(counter));
        return false;
    }
    cs_wipe(counter, sizeof(counter));
    out.reserve(n);
    for (size_t k = 0; k < n; ++k) {
        const uint8_t* p = buf.data() + k * CS_HRDETAIL_SAMPLE_BYTES;
        CsHrSample s;
        s.idx = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        s.y = p[4];
        out.push_back(s);
    }
    cs_wipe(buf.data(), buf.size());
    return true;
}

size_t cs_apply_hr_detail(cv::Mat& bgr, const std::vector<CsHrSample>& s) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || s.empty()) return 0;
    const int W = bgr.cols, H = bgr.rows;
    const size_t total = (size_t)W * H;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    const cv::Mat orig_y = chs[0].clone();
    size_t in_range = 0;
    for (const CsHrSample& sm : s) {
        if ((size_t)sm.idx >= total) continue;
        chs[0].at<uint8_t>((int)(sm.idx / (uint32_t)W), (int)(sm.idx % (uint32_t)W)) = sm.y;
        ++in_range;
    }
    if (in_range == 0) return 0;
    cv::merge(chs, ycc);
    cv::cvtColor(ycc, bgr, cv::COLOR_YCrCb2BGR);
    // Gamut guard: exact Y plus solver chroma can fall outside BGR gamut
    // (clipped on the way back, Y drifts). Revert anchors that did not
    // survive the roundtrip instead of injecting clip artifacts.
    cv::Mat chk;
    cv::cvtColor(bgr, chk, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chk_chs;
    cv::split(chk, chk_chs);
    size_t applied = 0;
    for (const CsHrSample& sm : s) {
        if ((size_t)sm.idx >= total) continue;
        const int y = (int)(sm.idx / (uint32_t)W), x = (int)(sm.idx % (uint32_t)W);
        if (std::abs((int)chk_chs[0].at<uint8_t>(y, x) - (int)sm.y) <= 1) {
            ++applied;
        } else {
            chs[0].at<uint8_t>(y, x) = orig_y.at<uint8_t>(y, x);
        }
    }
    if (applied < in_range) {
        cv::merge(chs, ycc);
        cv::cvtColor(ycc, bgr, cv::COLOR_YCrCb2BGR);
    }
    return applied;
}



