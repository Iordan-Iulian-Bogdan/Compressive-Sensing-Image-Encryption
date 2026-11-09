// The cs-sr 2x stage of the CS decoder: PnP-ADMM over the HR luma's DCT coefficients against the tile's LR
// samples and the HR anchors, AVIR warm start and chroma (CPU per tile; batched on the GPU in cs_sr_gpu_batch).
// The public entry points are declared in helper_functions.hpp.
#include "helper_functions.hpp"
#include "fista_kernels.hpp"
#include "cs_gpu.h"
#include "crypto_utils.hpp"
#ifdef CS_GPU_TILES
#include "gpu_admm_broker.hpp"
#include "gpu_sr.h"
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

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

// CS_ADMM_PROX_OFF=1: identity prox in the ADMM loop (no NLM), same switch as gpu_admm.
static bool cs_sr_admm_prox_off()
{
    static const bool on = [] {
        const char* e = std::getenv("CS_ADMM_PROX_OFF");
        const bool v = e && e[0] == '1';
        if (v) std::fprintf(stderr, "cs-sr admm: prox off (identity)\n");
        return v;
    }();
    return on;
}

// Sigma-matched fastNlMeans strength (h) for the PnP prox modes: map the
// tile's own Immerkaer luma noise estimate to the denoiser's noise parameter
// (clamped to the useful range; a zero estimate falls back to a mild 3).
static float cs_sr_prox_nlm_h(const cv::Mat& lr_tile)
{
    double s = cs_estimate_noise_sigma(lr_tile);
    if (!(s > 0.0)) s = 3.0;
    float h = (float)(1.6 * s);
    if (h < 3.0f) h = 3.0f;
    if (h > 30.0f) h = 30.0f;
    return h;
}

// Per-round schedule of the PnP-ADMM penalty rho and NLM strength (DPIR-style
// continuation). rho goes geometrically from the base rho to CS_SR_RHO_END and the
// NLM strength is multiplied by a factor going geometrically from CS_SR_H0 to
// CS_SR_H1 over the rounds; CS_SR_URESCALE=1 rescales the scaled dual u by
// rho_old/rho_new when rho changes. Defaults (unset) keep the constant behaviour.
struct CsSrAdmmSched { float rho_end = 0.0f, h0 = 1.0f, h1 = 1.0f; bool rescale_u = false; };
static const CsSrAdmmSched& cs_sr_admm_sched() {
    static const CsSrAdmmSched s = [] {
        CsSrAdmmSched r;
        if (const char* v = std::getenv("CS_SR_RHO_END")) r.rho_end = (float)std::atof(v);
        if (const char* v = std::getenv("CS_SR_H0")) r.h0 = (float)std::atof(v);
        if (const char* v = std::getenv("CS_SR_H1")) r.h1 = (float)std::atof(v);
        if (const char* v = std::getenv("CS_SR_URESCALE")) r.rescale_u = v[0] == '1';
        if (!(r.h0 > 0.0f)) r.h0 = 1.0f;
        if (!(r.h1 > 0.0f)) r.h1 = 1.0f;
        return r;
    }();
    return s;
}
static float cs_sr_geo(float a, float b, int k, int n) {
    if (n <= 1 || a == b) return a;
    return a * std::pow(b / a, (float)k / (float)(n - 1));
}

// Y measurements: OpenCV BGR2YCrCb luma weights over the sampled BGR triplets (same weights
// the Y plane of the warm start is built with).
static void cs_sr_luma_samples(const cv::Mat& pixel_measurements, int m, std::vector<float>& b)
{
    std::vector<float> bB, bG, bR;
    cs_extract_channel_measurements(pixel_measurements, 0, m, bB);
    cs_extract_channel_measurements(pixel_measurements, 1, m, bG);
    cs_extract_channel_measurements(pixel_measurements, 2, m, bR);
    b.resize((size_t)m);
    for (int k = 0; k < m; ++k)
        b[(size_t)k] = 0.114f * bB[(size_t)k] + 0.587f * bG[(size_t)k] + 0.299f * bR[(size_t)k];
}

// HR-detail anchors as in-solve data terms: global output-geometry indices map to tile-local
// HR pixels via the tile origin; anchors outside this tile are skipped (neighbor tiles own them).
static void cs_sr_tile_anchors(const CsHrSrCtx& hr_sr, int Hr, int Wc,
    std::vector<int>& idx, std::vector<float>& val)
{
    if (!hr_sr.anchors || hr_sr.anchors->empty() || hr_sr.stride <= 0) return;
    idx.reserve(hr_sr.anchors->size());
    val.reserve(hr_sr.anchors->size());
    for (const CsHrSample& s : *hr_sr.anchors) {
        const int gx = (int)(s.idx % (uint32_t)hr_sr.stride);
        const int gy = (int)(s.idx / (uint32_t)hr_sr.stride);
        const int lx = gx - hr_sr.ox, ly = gy - hr_sr.oy;
        if (lx < 0 || ly < 0 || lx >= Wc || ly >= Hr) continue;
        idx.push_back(ly * Wc + lx);
        val.push_back(s.y * (1.0f / 255.0f));
    }
}

// cs_sr_tile_anchors over anchors stable-sorted by global index: the tile's anchors are found row
// by row with two binary searches instead of a scan of the whole list. Same anchors, and anchors
// sharing an index keep their list order, so the last one still wins when they are scattered.
static void cs_sr_tile_anchors_sorted(const std::vector<CsHrSample>& sorted, const CsHrSrCtx& hr_sr,
    int Hr, int Wc, std::vector<int>& idx, std::vector<float>& val)
{
    if (sorted.empty() || hr_sr.stride <= 0) return;
    const long long S = hr_sr.stride;
    const long long x0 = (std::max)(0, hr_sr.ox), x1 = (std::min)((long long)hr_sr.ox + Wc, S);
    if (x0 >= x1) return;
    auto less_idx = [](const CsHrSample& s, long long v) { return (long long)s.idx < v; };
    for (int ly = (std::max)(0, -hr_sr.oy); ly < Hr; ++ly) {
        const long long row = (long long)(hr_sr.oy + ly) * S;
        auto lo = std::lower_bound(sorted.begin(), sorted.end(), row + x0, less_idx);
        for (; lo != sorted.end() && (long long)lo->idx < row + x1; ++lo) {
            idx.push_back(ly * Wc + (int)((long long)lo->idx - row - hr_sr.ox));
            val.push_back(lo->y * (1.0f / 255.0f));
        }
    }
}

void cs_sr_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int fista_iters, float wscale, int admm_iters,
    cv::Mat& hr_out, const CsHrSrCtx& hr_sr)
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
    const int n_lr = Lr * Lc;
    const int m = (int)ri_x.size();
    if (pixel_measurements.total() < (size_t)CS_HEADER_PIXELS + (size_t)m) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    // Full LR per-pass budget: the warm start below is already at full scale
    // (not the /10 generic-ref convention — the solved LR tile IS a good
    // estimate), so every step refines rather than regrows coefficients.
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    const cs_fista_basis bx = cs_make_basis(Hr, Wc, CS_BASIS_DCT, wscale);

    std::vector<float> b;
    cs_sr_luma_samples(pixel_measurements, m, b);
    // AVIR 2x of the solved tile: the luma start of the ADMM and the final Cr/Cb planes
    // (luminance-only solve: SR lives in Y, chroma is smooth enough to interpolate).
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
    const std::vector<float> w((size_t)n_hr, 1.0f);
    std::vector<float> xv((size_t)n_hr);
    float* x = xv.data();
    const float correction_coef = coef * 0.1f;
    // HR-detail anchors as in-solve data terms: global output-geometry
    // indices map to tile-local HR pixels via the tile origin; anchors
    // outside this tile are skipped (neighbor tiles own them).
    std::vector<int> tile_hr_idx;
    std::vector<float> tile_hr_val;
    cs_sr_tile_anchors(hr_sr, Hr, Wc, tile_hr_idx, tile_hr_val);
    // Sigma-matched NLM strength for the prox.
    const float nlm_prox_h = cs_sr_prox_nlm_h(lr_tile);
    // PnP-ADMM: x-update solves the data term plus rho/2||x-(z-u)||^2 in
    // the anchor slot, then z = D(x+u) is the denoiser prox of the
    // implicit prior and u += x - z (scaled dual). The measurement model
    // enforces each pass; the prior proposes through the prox.
    int admm = admm_iters;
    if (admm < 1) admm = 1;
    if (admm > 50) admm = 50;
    const float rho = 0.1f;
    std::vector<float> zpx((size_t)n_hr, 0.0f), upx((size_t)n_hr, 0.0f),
        anch((size_t)n_hr, 0.0f);
    {
        cv::Mat zm;
        init_ycc_chs[0].convertTo(zm, CV_32F, 1.0f / 255.0f);
        std::memcpy(zpx.data(), zm.ptr<float>(0), sizeof(float) * (size_t)n_hr);
    }
    // GPU path (env CS_GPU_ADMM=1): same loop on the device (exact-HR anchors and tv > 0 included);
    // anything else runs below.
    bool gpu_done = false;
#ifdef CS_GPU_TILES
    if (tv >= 0.0f && cs_default_dct_block() <= 0 && cs_gpu_admm_enabled()) {
        const bool hr_on = !tile_hr_idx.empty() && hr_sr.weight > 0.0f;
        std::vector<float> hr_map;
        if (hr_on) {
            hr_map.assign((size_t)n_hr, -1e30f);
            for (size_t k = 0; k < tile_hr_idx.size(); ++k) hr_map[(size_t)tile_hr_idx[k]] = tile_hr_val[k];
        }
        gpu_admm::Problem gp{ Hr, Wc, b.data(), ri_x.data(), ri_y.data(), m,
            zpx.data(), correction_coef, nlm_prox_h };
        gp.hr_val = hr_on ? hr_map.data() : nullptr;
        gp.tv = tv;
        gp.ox = hr_sr.ox; gp.oy = hr_sr.oy;
        const CsSrAdmmSched& sc = cs_sr_admm_sched();
        gpu_admm::Params gprm{ admm, inner, rho, bx.lip, wscale,
            sc.rho_end, sc.h0, sc.h1, sc.rescale_u ? 1 : 0 };
        gprm.hr_weight = hr_on ? hr_sr.weight : 0.0f;
        gprm.bt = tv > 0.0f ? 1 : 0;
        gpu_done = cs_gpu_admm_run(gp, gprm);
    }
#endif
    const CsSrAdmmSched& sched = cs_sr_admm_sched();
    const float rho_end = sched.rho_end > 0.0f ? sched.rho_end : rho;
    float rho_prev = rho;
    for (int k = 0; !gpu_done && k < admm; ++k) {
        const float rho_k = cs_sr_geo(rho, rho_end, k, admm);
        const float h_k = nlm_prox_h * cs_sr_geo(sched.h0, sched.h1, k, admm);
        if (sched.rescale_u && rho_k != rho_prev) {
            const float f = rho_prev / rho_k;
            for (float& uu : upx) uu *= f;
        }
        rho_prev = rho_k;
        for (int i = 0; i < n_hr; ++i)
            anch[(size_t)i] = zpx[(size_t)i] - upx[(size_t)i];
        {
            cv::Mat am(Hr, Wc, CV_32F, anch.data());
            cs_dct_forward(am, cs_default_dct_block());
        }
        {
            cv::Mat sm_(Hr, Wc, CV_32F);
            std::memcpy(sm_.ptr<float>(0), zpx.data(), sizeof(float) * (size_t)n_hr);
            cs_dct_forward(sm_, cs_default_dct_block());
            std::memcpy(x, sm_.ptr<float>(0), sizeof(float) * (size_t)n_hr);
        }
        cs_sr_core_single(x, b.data(), anch.data(), rho_k,
            ri_x.data(), ri_y.data(), m, Hr, Wc, Lr, Lc,
            w.data(), correction_coef, tv, inner, bx,
            y, x_prev, grad, pix_hr, lr_buf, sc_lr, up_hr,
            tile_hr_idx.data(), tile_hr_val.data(), (int)tile_hr_idx.size(),
            hr_sr.weight);
        cv::Mat px(Hr, Wc, CV_32F, x);
        cs_dct_inverse(px, cs_default_dct_block());
        cv::Mat zp(Hr, Wc, CV_32F);
        {
            const float* p = px.ptr<float>(0);
            float* vv = zp.ptr<float>(0);
            for (int i = 0; i < n_hr; ++i) vv[i] = p[i] + upx[(size_t)i];
        }
        {
            // prox: fastNlMeans on the 8-bit plane, or (CS_ADMM_PROX_OFF=1) identity, keeping only
            // the 8-bit rounding of the NLM input, as gpu_admm's k_unpad_u8
            cv::Mat u8, dn;
            zp.convertTo(u8, CV_8U, 255.0);
            if (cs_sr_admm_prox_off()) dn = u8;
            else cv::fastNlMeansDenoising(u8, dn, h_k, 7, 15);
            dn.convertTo(zp, CV_32F, 1.0 / 255.0);
        }
        {
            const float* pp = px.ptr<float>(0);
            const float* zz = zp.ptr<float>(0);
            for (int i = 0; i < n_hr; ++i) {
                zpx[(size_t)i] = zz[i];
                upx[(size_t)i] += pp[i] - zz[i];
            }
        }
    }
    {
        cv::Mat zm(Hr, Wc, CV_32F);
        std::memcpy(zm.ptr<float>(0), zpx.data(), sizeof(float) * (size_t)n_hr);
        cs_dct_forward(zm, cs_default_dct_block());
        std::memcpy(x, zm.ptr<float>(0), sizeof(float) * (size_t)n_hr);
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

#ifdef CS_GPU_TILES
// Per-tile host data of cs_sr_gpu_batch (kept alive until the GPU pass returns) and its ADMM parameter groups.
struct CsSrBatchPrep {
    struct Tile {
        int rows = 0, cols = 0;   // LR tile size the tile was prepared for
        cv::Mat lr; std::vector<float> b, hval; std::vector<int> hidx; gpu_admm::Params prm; bool ok = false;
        std::vector<std::uint8_t> dbg_avir; float dbg_h = 0.0f;
    };
    using Key = std::tuple<int, int, float, float, float, float, float, float, int, float, int>;
    std::vector<Tile> pp;
    std::map<Key, std::vector<size_t>> groups;
};

std::shared_ptr<CsSrBatchPrep> cs_sr_gpu_batch_prepare(const std::vector<CsSrBatchTile>& tiles, int admm_iters)
{
    // Same gate as the per-tile GPU ADMM path of cs_sr_upscale_tile_2x.
    if (tiles.empty() || cs_default_dct_block() > 0 || !cs_gpu_admm_enabled())
        return nullptr;
    auto S = std::make_shared<CsSrBatchPrep>();
    const int admm = (std::min)(50, (std::max)(1, admm_iters));
    const float rho = 0.1f;
    const CsSrAdmmSched& sc = cs_sr_admm_sched();

    std::vector<CsSrBatchPrep::Tile>& pp = S->pp;
    pp.resize(tiles.size());
    // Anchors sorted once (every tile shares the decode's anchor list).
    const std::vector<CsHrSample>* anchors = tiles.front().hr_sr.anchors;
    std::vector<CsHrSample> sorted_anchors;
    if (anchors && tiles.front().hr_sr.stride > 0) {
        sorted_anchors = *anchors;
        std::stable_sort(sorted_anchors.begin(), sorted_anchors.end(),
            [](const CsHrSample& a, const CsHrSample& b) { return a.idx < b.idx; });
    }
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long kk = 0; kk < (long long)tiles.size(); ++kk) {
        const size_t k = (size_t)kk;
        const CsSrBatchTile& T = tiles[k];
        CsSrBatchPrep::Tile& P = pp[k];
        if (!T.lr_tile || T.lr_tile->empty() || T.lr_tile->type() != CV_8UC3) continue;
        const int Lr = T.lr_tile->rows, Lc = T.lr_tile->cols, Hr = 2 * Lr, Wc = 2 * Lc;
        const int m = (int)T.ri_x->size();
        if (Lr < 8 || Lc < 8 || !(T.tv >= 0.0f) || !gpu_admm::supported(Hr, Wc) ||
            T.meas->total() < (size_t)CS_HEADER_PIXELS + (size_t)m)
            continue;
        P.rows = Lr; P.cols = Lc;   // the pixels are taken in cs_sr_gpu_batch (the LR solve may still be running)
        cs_sr_luma_samples(*T.meas, m, P.b);
        std::vector<int> hidx; std::vector<float> hval;
        if (T.hr_sr.anchors == anchors && !sorted_anchors.empty())
            cs_sr_tile_anchors_sorted(sorted_anchors, T.hr_sr, Hr, Wc, hidx, hval);
        else
            cs_sr_tile_anchors(T.hr_sr, Hr, Wc, hidx, hval);
        const bool hr_on = !hidx.empty() && T.hr_sr.weight > 0.0f;
        if (hr_on) {
            // sparse for the device scatter: increasing indices, the last value of a repeated pixel (as a dense
            // map filled in list order)
            std::vector<int> order(hidx.size());
            for (size_t q = 0; q < order.size(); ++q) order[q] = (int)q;
            std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return hidx[(size_t)a] < hidx[(size_t)c]; });
            for (size_t q = 0; q < order.size(); ++q) {
                const int i = hidx[(size_t)order[q]];
                if (q + 1 < order.size() && hidx[(size_t)order[q + 1]] == i) continue;
                P.hidx.push_back(i);
                P.hval.push_back(hval[(size_t)order[q]]);
            }
        }
        const cs_fista_basis bx = cs_make_basis(Hr, Wc, CS_BASIS_DCT, T.wscale);
        P.prm = gpu_admm::Params{ admm, cs_fista_inner_iters(T.iterations, T.fista_iters), rho, bx.lip, T.wscale,
            sc.rho_end, sc.h0, sc.h1, sc.rescale_u ? 1 : 0 };
        P.prm.bt = T.tv > 0.0f ? 1 : 0;
        // Fixed-step batches take a per-tile anchor weight (tiles without anchors get none), so
        // anchored and anchor-free tiles share one batch; backtracking batches keep them apart.
        P.prm.hr_weight = (hr_on || (!P.prm.bt && T.hr_sr.weight > 0.0f)) ? T.hr_sr.weight : 0.0f;
        P.ok = true;
    }
    for (size_t k = 0; k < tiles.size(); ++k) {
        const CsSrBatchPrep::Tile& P = pp[k];
        if (!P.ok) continue;
        S->groups[CsSrBatchPrep::Key(P.prm.admm, P.prm.inner, P.prm.rho, P.prm.lip, P.prm.wscale, P.prm.rho_end,
            P.prm.h_start, P.prm.h_end, P.prm.rescale_u, P.prm.hr_weight, P.prm.bt)].push_back(k);
    }
    return S;
}

void cs_sr_gpu_batch(std::vector<CsSrBatchTile>& tiles, int admm_iters, std::vector<char>& done, CsSrBatchPrep* prep)
{
    done.assign(tiles.size(), 0);
    std::shared_ptr<CsSrBatchPrep> own;
    if (!prep) { own = cs_sr_gpu_batch_prepare(tiles, admm_iters); prep = own.get(); }
    if (!prep || prep->pp.size() != tiles.size()) return;
    std::vector<CsSrBatchPrep::Tile>& pp = prep->pp;
    int verify_every = 0;
    if (const char* v = std::getenv("CS_GPU_SR_VERIFY")) verify_every = (std::max)(0, std::atoi(v));
    // The solved LR tiles, which must have kept the size they were prepared for.
    for (size_t k = 0; k < tiles.size(); ++k) {
        CsSrBatchPrep::Tile& P = pp[k];
        const cv::Mat* L = tiles[k].lr_tile;
        if (!P.ok) continue;
        if (!L || L->type() != CV_8UC3 || L->rows != P.rows || L->cols != P.cols) { P.ok = false; continue; }
        P.lr = L->isContinuous() ? *L : L->clone();
    }

    for (auto& g0 : prep->groups) {
        std::pair<const CsSrBatchPrep::Key, std::vector<size_t>> g{ g0.first, {} };
        for (size_t k : g0.second) if (pp[k].ok) g.second.push_back(k);
        if (g.second.empty()) continue;
        std::vector<gpu_sr::Tile> jobs;
        jobs.reserve(g.second.size());
        // The group's output tiles are views into one buffer: one allocation (and one free)
        // instead of thousands of mid-sized ones.
        size_t pool_bytes = 0;
        for (size_t k : g.second) pool_bytes += (size_t)12 * pp[k].lr.rows * pp[k].lr.cols;
        const bool pooled = pool_bytes > 0 && pool_bytes <= (size_t)INT_MAX;
        cv::Mat pool;
        if (pooled) pool.create(1, (int)pool_bytes, CV_8U);
        size_t pool_off = 0;
        for (size_t k : g.second) {
            const CsSrBatchTile& T = tiles[k];
            CsSrBatchPrep::Tile& P = pp[k];
            const int n = 12 * P.lr.rows * P.lr.cols;
            if (pooled) *T.hr_out = pool.colRange((int)pool_off, (int)pool_off + n).reshape(3, 2 * P.lr.rows);
            else T.hr_out->create(2 * P.lr.rows, 2 * P.lr.cols, CV_8UC3);
            pool_off += (size_t)n;
            gpu_sr::Tile J{ P.lr.rows, P.lr.cols, P.lr.data, P.b.data(), T.ri_x->data(), T.ri_y->data(),
                (int)T.ri_x->size(), T.coef * 0.1f, T.tv, nullptr, T.hr_out->data };
            J.ox = T.hr_sr.ox; J.oy = T.hr_sr.oy;
            J.hr_idx = P.hidx.data(); J.hr_v = P.hval.data(); J.n_hr = (int)P.hidx.size();
            if (verify_every > 0 && k % (size_t)verify_every == 0) {
                P.dbg_avir.resize((size_t)12 * P.lr.rows * P.lr.cols);
                J.dbg_avir = P.dbg_avir.data();
                J.dbg_nlm_h = &P.dbg_h;
            }
            jobs.push_back(J);
        }
        const auto t_grp = std::chrono::steady_clock::now();
        const bool grp_ok = gpu_sr::run(jobs, g.second.empty() ? gpu_admm::Params{} : pp[g.second.front()].prm);
        if (const char* pr = std::getenv("CS_GPU_PROF"); pr && pr[0] == '1') {   // one line per ADMM parameter group
            const gpu_admm::Params& q = pp[g.second.front()].prm;
            long long px = 0;
            for (size_t k : g.second) px += (long long)pp[k].lr.rows * pp[k].lr.cols;
            std::fprintf(stderr, "[gpu-sr group] %zu tiles, %lld LR px, admm %d inner %d lip %.3f hr_w %g bt %d: %.1f ms\n",
                g.second.size(), px, q.admm, q.inner, q.lip, q.hr_weight, q.bt,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_grp).count());
        }
        if (!grp_ok) {
            for (size_t k : g.second) tiles[k].hr_out->release();
            continue;
        }
        for (size_t k : g.second) done[k] = 1;
    }

    if (verify_every > 0) {
        // GPU vs CPU: AVIR warm start, NLM strength, and the whole tile (CPU cs_sr_upscale_tile_2x,
        // whose ADMM runs on the GPU broker, so the difference isolates the batched pipeline).
        long long n = 0, a1 = 0, a2 = 0, f2 = 0; int amax = 0; double hmax = 0.0, fsum = 0.0; long long fn = 0; int nt = 0;
        for (size_t k = 0; k < tiles.size(); ++k) {
            if (!done[k] || pp[k].dbg_avir.empty()) continue;
            const CsSrBatchTile& T = tiles[k];
            cv::Mat cpu_avir;
            cs_upscale_2x_avir(pp[k].lr, cpu_avir);
            if (cpu_avir.total() * 3 != pp[k].dbg_avir.size()) continue;
            for (size_t q = 0; q < pp[k].dbg_avir.size(); ++q) {
                const int d = std::abs((int)cpu_avir.data[q] - (int)pp[k].dbg_avir[q]);
                ++n; a1 += d >= 1; a2 += d >= 2; amax = (std::max)(amax, d);
            }
            hmax = (std::max)(hmax, (double)std::fabs(cs_sr_prox_nlm_h(pp[k].lr) - pp[k].dbg_h));
            cv::Mat cpu_hr;
            cs_sr_upscale_tile_2x(pp[k].lr, *T.meas, *T.ri_x, *T.ri_y, T.coef, T.tv, T.iterations, T.fista_iters,
                T.wscale, admm_iters, cpu_hr, T.hr_sr);
            if (cpu_hr.size() == T.hr_out->size()) {
                for (size_t q = 0; q < cpu_hr.total() * 3; ++q) {
                    const int d = std::abs((int)cpu_hr.data[q] - (int)T.hr_out->data[q]);
                    fsum += d; ++fn; f2 += d > 2;
                }
            }
            ++nt;
        }
        std::fprintf(stderr, "[gpu-sr verify] %d tiles: %s warm start %.3f%% values differ by 1, %.4f%% by >=2 (max %d); "
            "NLM h max diff %.4f; final tile vs CPU path mean |diff| %.3f, %.3f%% > 2 levels\n",
            nt, "AVIR", n ? 100.0 * a1 / n : 0.0, n ? 100.0 * a2 / n : 0.0, amax, hmax, fn ? fsum / fn : 0.0, fn ? 100.0 * f2 / fn : 0.0);
    }
}
#else
struct CsSrBatchPrep {};

std::shared_ptr<CsSrBatchPrep> cs_sr_gpu_batch_prepare(const std::vector<CsSrBatchTile>&, int)
{
    return nullptr;
}

void cs_sr_gpu_batch(std::vector<CsSrBatchTile>& tiles, int, std::vector<char>& done, CsSrBatchPrep*)
{
    done.assign(tiles.size(), 0);
}
#endif
