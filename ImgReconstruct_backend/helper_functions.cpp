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
        cv::dct(c[i], c[i], 0);
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
        const float fy = cs_fista_grad(yy, pix.data(), sc.data(), grad.data(), b, rix, riy, m, rows, cols, tv_lambda, bx);
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
    std::vector<float>& pix, std::vector<float>& sc)
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
            fy += cs_fista_grad(yc[c], pix.data(), sc.data(), gc[c], bb[c], rix, riy, m, rows, cols, tv_lambda, bx);
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

void fista_channel_begin(fista_channel_task& t, const cv::Mat& pixel_measurements,
    const int& channel, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations,
    cv::Mat& ref, float tv, int reweights, int fista_iters, int basis, float wscale)
{
    if (!ref.isContinuous()) ref = ref.clone();
    t.basis = basis;
    t.wscale = wscale;
    const bool sp = g_solveprof.on;
    auto sp_t0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (cs_gpu::enabled() && basis != CS_BASIS_CDF97) {
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
            basis, wscale);
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
        cv::dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
        AtAxb2 = AtAxb2 * 255.0f;
    }
    if (sp) g_solveprof.wrap_tail_ns += sp_ns_since(sp_t1);
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
    const int* rix, const int* riy, int m, int Hr, int Wc, int Lr, int Lc, float tv_lambda)
{
    const int n_hr = Hr * Wc;
    const int n_lr = Lr * Lc;
    std::memcpy(pix_hr, y, sizeof(float) * (size_t)n_hr);
    cv::Mat P(Hr, Wc, CV_32F, pix_hr);
    cv::dct(P, P, cv::DCT_INVERSE);
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
    cv::dct(G, G, 0);
    for (int i = 0; i < n_hr; ++i) g[i] = 2.0f * up_hr[i];

    if (tv_lambda > 0.0f) {
        float* tvgrad = up_hr; // data grad already folded into g
        const float phi = cs_tv_grad_phi(pix_hr, tvgrad, Hr, Wc);
        cv::Mat T(Hr, Wc, CV_32F, tvgrad);
        cv::dct(T, T, 0);
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
    int Hr, int Wc, int Lr, int Lc, float tv_lambda, float* tv_scratch)
{
    const int n_hr = Hr * Wc;
    const int n_lr = Lr * Lc;
    std::memcpy(pix_hr, xc, sizeof(float) * (size_t)n_hr);
    cv::Mat P(Hr, Wc, CV_32F, pix_hr);
    cv::dct(P, P, cv::DCT_INVERSE);
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
    for (int i = 0; i < n_hr; ++i) {
        const float diff = xc[i] - anchor[i];
        fx += anchor_weight * diff * diff;
    }
    return fx;
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
    cv::dct(pix, pix, cv::DCT_INVERSE);
    cv::Mat u8, dn, dnf;
    // full scale here is [0,1]: expand to 8U for the denoiser (a bare
    // convertTo would quantize everything to 0/1 and darken the tile).
    pix.convertTo(u8, CV_8U, 255.0);
    cv::fastNlMeansDenoising(u8, dn, 5.0f, 7, 15);
    dn.convertTo(dnf, CV_32F, 1.0 / 255.0);
    pix -= red * (pix - dnf);
    cv::dct(pix, pix, 0);
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

static CsDncnnEntry* cs_dncnn_entry(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_dncnn_mutex);
    auto it = g_dncnn_cache.find(path);
    if (it != g_dncnn_cache.end()) return it->second.get();
    if (g_dncnn_failed.find(path) != g_dncnn_failed.end()) return nullptr;
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
        std::fprintf(stderr, "Warning: DnCNN model '%s' failed to load: %s\n",
            path.c_str(), ex.what());
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
        std::fprintf(stderr, "Warning: DnCNN run failed: %s\n", ex.what());
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
    return cs_dncnn_entry(model_path) != nullptr;
}

bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path)
{
    dst_bgr.release();
    if (src_bgr.empty() || src_bgr.type() != CV_8UC3) return false;
    CsDncnnEntry* e = cs_dncnn_entry(model_path);
    if (!e) return false;
    const int H = src_bgr.rows, W = src_bgr.cols;
    cv::Mat rgb;
    cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chs;
    cv::split(rgb, chs);
    // tiled runner: 512px tiles, 48px feathered overlap (activation memory
    // of a 20-layer 64ch net is ~70MB per 512 tile; a 12MP single run
    // would need gigabytes).
    constexpr int T = 512, OV = 48;
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
        in.assign((size_t)3 * th * tw, 0.0f);
        rs.assign((size_t)3 * th * tw, 0.0f);
        for (int c = 0; c < 3; ++c) {
            float* dst = &in[(size_t)c * th * tw];
            for (int r = 0; r < th; ++r) {
                const uint8_t* srow = chs[(size_t)c].ptr<uint8_t>(y0 + r) + x0;
                for (int k = 0; k < tw; ++k) dst[(size_t)r * tw + k] = srow[k] * (1.0f / 255.0f);
            }
        }
        if (!cs_dncnn_run_tile(e, in.data(), rs.data(), th, tw)) return false;
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
                const size_t o = (size_t)r * tw + k;
                // clean-output convention: the model output IS the denoised
                // estimate (convertTo saturates to [0,255] at the end).
                const cv::Vec3f clean_rgb(
                    rs[o],
                    rs[(size_t)th * tw + o],
                    rs[(size_t)2 * th * tw + o]);
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
    return true;
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
    std::vector<float>& lr_buf, std::vector<float>& sc_lr, std::vector<float>& up_hr)
{
    const int n_hr = Hr * Wc;
    const bool bt = (tv_lambda > 0.0f);
    // Data-term Lipschitz for P*D*IDCT: 2*||A||^2 <= 0.25*lip. TV keeps the
    // same 8*tv curvature fold + backtracking policy as the LR core.
    // (Always-on backtracking was trialed here after a white-output scare
    // that turned out to be a white-input test fixture, not instability:
    // biased samples solve fine at 32.47 dB. Kept off: the fixed step is
    // exact-safe and ~40% cheaper per iteration.)
    float L = 0.25f * bx.lip + 2.0f * anchor_weight + 8.0f * tv_lambda;
    std::memcpy(y.data(), x, sizeof(float) * (size_t)n_hr);
    std::memcpy(x_prev.data(), x, sizeof(float) * (size_t)n_hr);
    std::vector<float> z((size_t)n_hr);
    float* yy = y.data();
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        const float fy = cs_sr_fista_grad(yy, pix_hr.data(), lr_buf.data(), sc_lr.data(),
            up_hr.data(), grad.data(), b, anchor, anchor_weight,
            rix, riy, m, Hr, Wc, Lr, Lc, tv_lambda);
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
                    b, anchor, anchor_weight, rix, riy, m, Hr, Wc, Lr, Lc, tv_lambda, up_hr.data());
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
    cv::Mat& hr_out, float red, const std::string& red_denoiser, const std::string& dncnn_model)
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
    // Full LR per-pass budget: the warm start below is already at full scale
    // (not the /10 generic-ref convention — the solved LR tile IS a good
    // estimate), so every step refines rather than regrows coefficients.
    // An explicit --fista-iters override is honored as-is.
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    const cs_fista_basis bx = cs_make_basis(Hr, Wc, CS_BASIS_DCT, wscale);

    // AVIR warm start: solved LR content carries the low frequencies, so
    // the SR passes only need to fill in the sparse high frequencies.
    // Luminance-only solve (benchmark convention: SR lives in Y, chroma is
    // smooth enough to interpolate): the AVIR HR image below supplies both
    // the Y warm start and the final Cr/Cb planes, so only one HR FISTA
    // solve runs instead of three.
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

    // Y measurements: OpenCV BGR2YCrCb luma weights over the sampled BGR
    // triplets (same weights the Y plane below was built with).
    std::vector<float> bB, bG, bR, b((size_t)m);
    cs_extract_channel_measurements(pixel_measurements, 0, m, bB);
    cs_extract_channel_measurements(pixel_measurements, 1, m, bG);
    cs_extract_channel_measurements(pixel_measurements, 2, m, bR);
    for (int k = 0; k < m; ++k) {
        b[(size_t)k] = 0.114f * bB[(size_t)k] + 0.587f * bG[(size_t)k] + 0.299f * bR[(size_t)k];
    }

    // Scratch for the single luma solve.
    std::vector<float> y((size_t)n_hr), x_prev((size_t)n_hr), grad((size_t)n_hr),
        pix_hr((size_t)n_hr), up_hr((size_t)n_hr);
    std::vector<float> lr_buf((size_t)n_lr), sc_lr((size_t)n_lr);
    std::vector<float> w((size_t)n_hr, 1.0f);
    cv::Mat xf;
    init_ycc_chs[0].convertTo(xf, CV_32F, 1.0 / 255.0);
    cv::dct(xf, xf, 0);
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
    for (int r = 0; r < reweights; ++r) {
        cs_sr_core_single(x, b.data(), anchor.data(), anchor_weight,
            ri_x.data(), ri_y.data(), m, Hr, Wc, Lr, Lc,
            w.data(), correction_coef, tv, inner, bx,
            y, x_prev, grad, pix_hr, lr_buf, sc_lr, up_hr);
        // RED-lite: pull the pass solution toward the denoiser manifold
        // before reweighting, so the reweights protect denoised structure.
        // red = 0 skips exactly (bit-identical to the unregularized path).
        // "dncnn" denoises the merged BGR tile (the color model needs all
        // three channels); the Y plane is re-extracted for the next pass.
        // A failed DnCNN run keeps the un-denoised pass (graceful).
        if (red > 0.0f && red_denoiser == "dncnn") {
            cv::Mat px(Hr, Wc, CV_32F, x);
            cv::dct(px, px, cv::DCT_INVERSE);
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
            const float rl = red > 1.0f ? 1.0f : red;
            if (cs_dncnn_denoise_bgr(bgr_hr, dn_hr, dncnn_model)) {
                cv::Mat dn_ycc;
                cv::cvtColor(dn_hr, dn_ycc, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> dn_chs;
                cv::split(dn_ycc, dn_chs);
                cv::Mat dnf;
                dn_chs[0].convertTo(dnf, CV_32F, 1.0 / 255.0);
                px -= rl * (px - dnf);
            }
            cv::dct(px, px, 0);
        }
        else {
            cs_sr_red_proximal(x, Hr, Wc, red);
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
    cv::dct(plane, plane, cv::DCT_INVERSE);
    plane = plane * 255.0f;
    cv::Mat out_y;
    plane.convertTo(out_y, CV_8U);
    std::vector<cv::Mat> out_ycc{ out_y, init_ycc_chs[1], init_ycc_chs[2] };
    cv::Mat merged;
    cv::merge(out_ycc, merged);
    cv::cvtColor(merged, hr_out, cv::COLOR_YCrCb2BGR);
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
        cv::dct(hrm, hrm, 0);
        std::memcpy(x.data(), hrm.data, sizeof(float) * (size_t)n_hr);
        anchor = x;
    }
    else if (init_mode == 2 && warm_px) {
        // external float warm start (Hr x Wc pixels, [0,1]); also anchored
        cv::Mat wm(Hr, Wc, CV_32F, (void*)warm_px);
        cv::Mat xm;
        wm.convertTo(xm, CV_32F);
        cv::dct(xm, xm, 0);
        std::memcpy(x.data(), xm.data, sizeof(float) * (size_t)n_hr);
        anchor = x;
    }
    // correction-scale lambda mirrors the two-stage path (coef*0.1)
    const float lambda = coef * 0.1f;
    for (int r = 0; r < reweights; ++r) {
        cs_sr_core_single(x.data(), b, anchor.data(), anchor_w,
            rix, riy, m, Hr, Wc, Lr, Lc, w.data(), lambda, tv, inner, bx,
            y, x_prev, grad, pix_hr, lr_buf, sc_lr, up_hr);
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
    cv::dct(plane, plane, cv::DCT_INVERSE);
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
                cv::dct(wm, wm, 0);
                std::memcpy(xc.data(), wm.data, sizeof(float) * (size_t)n);
            }
            const cs_fista_basis bx = cs_make_basis(gh, gw, CS_BASIS_DCT, wscale);
            std::vector<float> w((size_t)n, 1.0f), y((size_t)n), xp((size_t)n),
                grad((size_t)n), pix((size_t)n), sc((size_t)n);
            cs_fista_core_single(xc.data(), ob.data(), ox.data(), oy.data(), mo,
                gh, gw, w.data(), coef, 0.0f, coarse_iters, bx,
                y, xp, grad, pix, sc);
            cv::Mat pm(gh, gw, CV_32F, xc.data());
            cv::dct(pm, pm, cv::DCT_INVERSE);
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



