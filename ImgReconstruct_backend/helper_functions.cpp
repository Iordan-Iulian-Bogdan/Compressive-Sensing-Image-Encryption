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
    const float eps = 1e-3f;
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



