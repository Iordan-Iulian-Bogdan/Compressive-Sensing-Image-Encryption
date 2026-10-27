// Super-resolution backends for the CS decoder: cs-sr (FISTA with HR anchors and
// RED), sr-direct, the DIP prior and the CS cascade init. Moved out of
// helper_functions.cpp; the public entry points are declared in helper_functions.hpp.
#include "helper_functions.hpp"
#include "fista_kernels.hpp"
#include "cs_gpu.h"
#include "crypto_utils.hpp"
#ifdef CS_GPU_TILES
#include "gpu_admm_broker.hpp"
#include "gpu_sr.h"
#include "gpu_swin.h"
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

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

// PnP proximal step for one HR luma plane in PIXEL domain (CV_32F [0,1], in
// place): replace the plane with the denoiser image D(v) — the proximal
// operator of the implicit prior. red_kind >= 0 uses the BGR ONNX color model
// (needs the HR chroma planes for its input); red_kind < 0 uses
// fastNlMeansDenoising on luma with strength nlm_h (sigma-matched). Returns
// false when an ONNX run fails (plane left untouched).
static bool cs_sr_prox_luma_plane(cv::Mat& plane, const cv::Mat& chroma_cr,
    const cv::Mat& chroma_cb, int red_kind, const std::string& red_path, float nlm_h)
{
    if (plane.empty() || plane.type() != CV_32F) return false;
    if (red_kind >= 0) {
        cv::Mat y8;
        {
            cv::Mat tmp = plane * 255.0f;
            tmp.convertTo(y8, CV_8U);
        }
        std::vector<cv::Mat> bgr_ycc{ y8, chroma_cr, chroma_cb };
        cv::Mat merged_ycc, bgr_hr, dn_hr;
        cv::merge(bgr_ycc, merged_ycc);
        cv::cvtColor(merged_ycc, bgr_hr, cv::COLOR_YCrCb2BGR);
        if (!cs_onnx_denoise_bgr(bgr_hr, dn_hr, red_path, red_kind)) return false;
        cv::Mat dn_ycc;
        cv::cvtColor(dn_hr, dn_ycc, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> dn_chs;
        cv::split(dn_ycc, dn_chs);
        dn_chs[0].convertTo(plane, CV_32F, 1.0 / 255.0);
        return true;
    }
    cv::Mat u8, dn;
    plane.convertTo(u8, CV_8U, 255.0);
    cv::fastNlMeansDenoising(u8, dn, nlm_h, 7, 15);
    dn.convertTo(plane, CV_32F, 1.0 / 255.0);
    return true;
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
    float coef, float tv, int iterations, int reweights, int fista_iters, int basis, float wscale,
    cv::Mat& hr_out, float red, const std::string& red_denoiser, const std::string& dncnn_model,
    bool sr_cascade, const std::string& red_model, float red_decay,
    const CsHrSrCtx& hr_sr, const CsSrPrior& sr_prior)
{
    hr_out.release();
    if (lr_tile.empty() || lr_tile.type() != CV_8UC3) return;
    const int Lr = lr_tile.rows, Lc = lr_tile.cols;
    if (Lr < 8 || Lc < 8) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    // Diagnostic (opt-in): CS_DUMP_LRTILES=<file> appends a sample of the solved LR tiles that
    // feed the 2x prior/AVIR as records [int32 ox, oy, rows, cols][BGR bytes], ox/oy being the
    // tile origin in the 2x output geometry. CS_DUMP_LR_EVERY=<n> (default 20) keeps ~1 in n
    // tiles, picked by a hash of the origin so the choice is the same on every run.
    if (const char* dump_path = std::getenv("CS_DUMP_LRTILES"); dump_path && *dump_path) {
        static std::mutex dump_mu;
        int every = 20;
        if (const char* e = std::getenv("CS_DUMP_LR_EVERY")) every = (std::max)(1, std::atoi(e));
        const unsigned long long key = ((unsigned long long)(unsigned)hr_sr.ox * 73856093ULL) ^
            ((unsigned long long)(unsigned)hr_sr.oy * 19349663ULL);
        if (every <= 1 || ((key >> 4) % (unsigned long long)every) == 0ULL) {
            const cv::Mat c = lr_tile.isContinuous() ? lr_tile : lr_tile.clone();
            const int32_t hdr[4] = { hr_sr.ox, hr_sr.oy, c.rows, c.cols };
            std::lock_guard<std::mutex> lk(dump_mu);
            if (FILE* f = std::fopen(dump_path, "ab")) {
                std::fwrite(hdr, sizeof(int32_t), 4, f);
                std::fwrite(c.data, 1, c.total() * c.elemSize(), f);
                std::fclose(f);
            }
        }
    }
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

    std::vector<float> b;
    cs_sr_luma_samples(pixel_measurements, m, b);
    // Prior image. Default is the AVIR resampler (always needed for the final
    // Cr/Cb planes — luminance-only solve: SR lives in Y, chroma is smooth
    // enough to interpolate). --sr-prior-model instead asks a learned ONNX 2x
    // photo-SR net to propose the HR image (luma anchor + chroma), which the
    // LR measurements below then enforce; a missing model or a non-2x result
    // falls back to AVIR per tile. --sr-init cascade replaces the warm start
    // below with the multiscale binned-FISTA chain (same anchor weight).
    cv::Mat init_hr;
    if (!sr_prior.model.empty()) {
        // Segment gate: crop the container's LR-grid labels to this tile. Any
        // size or origin mismatch leaves seg_roi empty, so the prior runs
        // everywhere (the legacy behavior).
        cv::Mat seg_roi;
        if (sr_prior.seg_lr && !sr_prior.seg_lr->empty()) {
            const int lx = hr_sr.ox / 2, ly = hr_sr.oy / 2;
            const cv::Mat& sl = *sr_prior.seg_lr;
            if (sl.channels() == 1 && lx >= 0 && ly >= 0 &&
                lx + lr_tile.cols <= sl.cols && ly + lr_tile.rows <= sl.rows)
                seg_roi = sl(cv::Rect(lx, ly, lr_tile.cols, lr_tile.rows));
        }
        // Single-segment sub-tiles start from cs-sr on their own samples (prior
        // off) instead of AVIR. Samples are picked by LR position and shifted
        // to the sub-tile origin; the nested call has no prior, so it cannot
        // recurse. Without samples the nested call falls back to AVIR itself.
        const auto single_patch = [&](int y0, int x0, int th, int tw, cv::Mat& hr_bgr) -> bool {
            std::vector<int> sx, sy, keep;
            for (int k = 0; k < m; ++k) {
                const int r = ri_x[(size_t)k], c = ri_y[(size_t)k];
                if (r >= y0 && r < y0 + th && c >= x0 && c < x0 + tw) {
                    keep.push_back(k);
                    sx.push_back(r - y0);
                    sy.push_back(c - x0);
                }
            }
            const int ms = (int)keep.size();
            cv::Mat sub_meas(1, CS_HEADER_PIXELS + ms, CV_8UC3);
            for (int i = 0; i < CS_HEADER_PIXELS; ++i)
                sub_meas.at<cv::Vec3b>(i) = pixel_measurements.at<cv::Vec3b>(i);
            for (int j = 0; j < ms; ++j)
                sub_meas.at<cv::Vec3b>(CS_HEADER_PIXELS + j) =
                    pixel_measurements.at<cv::Vec3b>(CS_HEADER_PIXELS + keep[(size_t)j]);
            const cv::Mat sub_lr = lr_tile(cv::Rect(x0, y0, tw, th)).clone();
            cs_sr_upscale_tile_2x(sub_lr, sub_meas, sx, sy, coef, tv, iterations, reweights,
                fista_iters, basis, wscale, hr_bgr, red, red_denoiser, dncnn_model,
                sr_cascade, red_model, red_decay, CsHrSrCtx(), CsSrPrior());
            return !hr_bgr.empty();
        };
        int sc = 0;
        if (cs_onnx_sr_upscale_bgr(lr_tile, init_hr, sr_prior.model, sc, seg_roi, single_patch) &&
            sc == 2 && init_hr.rows == Hr && init_hr.cols == Wc) {
            // strong SR proposal accepted
        } else {
            init_hr.release();
        }
    }
    if (init_hr.empty()) cs_upscale_2x_avir(lr_tile, init_hr);
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
    cs_sr_tile_anchors(hr_sr, Hr, Wc, tile_hr_idx, tile_hr_val);
    // Anchor-weighted RED: per-pixel blend map from the same tile-local
    // anchors (empty = legacy uniform RED). Built once; the per-pass
    // decayed strength multiplies it at each application.
    const cv::Mat hr_red_map = (red > 0.0f) ?
        cs_hr_red_map(Hr, Wc, tile_hr_idx, hr_sr.red_radius) : cv::Mat();
    // Strong-prior + PnP controls (see CsSrPrior). pnp_mode 0 = legacy RED
    // blend (bit-identical); 1 = denoiser-as-prox (per-pass replacement);
    // 2 = PnP-ADMM (dual splitting).
    const int pnp_mode = (sr_prior.pnp_mode >= 1 && sr_prior.pnp_mode <= 2)
        ? sr_prior.pnp_mode : 0;
    // Sigma-matched NLM strength for the prox modes; only computed when a prox
    // actually runs (the legacy path stays bit-identical and cost-free).
    const float nlm_prox_h = (pnp_mode >= 1) ? cs_sr_prox_nlm_h(lr_tile) : 5.0f;
    if (pnp_mode == 2) {
        // PnP-ADMM: x-update solves the data term plus rho/2||x-(z-u)||^2 in
        // the anchor slot, then z = D(x+u) is the denoiser prox of the
        // implicit prior and u += x - z (scaled dual). The measurement model
        // enforces each pass; the strong prior proposes through the prox.
        int admm = sr_prior.pnp_iters;
        if (admm < 1) admm = 1;
        if (admm > 50) admm = 50;
        const float rho = sr_prior.pnp_rho > 0.0f ? sr_prior.pnp_rho : 0.1f;
        std::vector<float> zpx((size_t)n_hr, 0.0f), upx((size_t)n_hr, 0.0f),
            anch((size_t)n_hr, 0.0f);
        {
            cv::Mat zm;
            init_ycc_chs[0].convertTo(zm, CV_32F, 1.0f / 255.0f);
            std::memcpy(zpx.data(), zm.ptr<float>(0), sizeof(float) * (size_t)n_hr);
        }
        std::fill(w.begin(), w.end(), 1.0f);
        // GPU batch path (env CS_GPU_ADMM=1): same loop on the device for the
        // NLM-prox case (exact-HR anchors and tv > 0 included); anything else runs below.
        bool gpu_done = false;
#ifdef CS_GPU_TILES
        if (red_kind < 0 && tv >= 0.0f && cs_default_dct_block() <= 0 && cs_gpu_admm_enabled()) {
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
            cv::Mat v(Hr, Wc, CV_32F);
            {
                const float* p = px.ptr<float>(0);
                float* vv = v.ptr<float>(0);
                for (int i = 0; i < n_hr; ++i) vv[i] = p[i] + upx[(size_t)i];
            }
            cv::Mat zp = v.clone();
            bool zok = true;
            if (red_kind >= 0)
                zok = cs_sr_prox_luma_plane(zp, init_ycc_chs[1], init_ycc_chs[2],
                    red_kind, red_path, h_k);
            else
                cs_sr_prox_luma_plane(zp, cv::Mat(), cv::Mat(), -1, std::string(),
                    h_k);
            if (!zok) zp = v;
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
    }
    else {
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
        if (pnp_mode == 1) {
            // PnP prox (see CsSrPrior): replace the pass estimate with the
            // denoiser image — the proximal operator of the implicit prior,
            // sigma-matched for NLM — with no blend. The strong prior
            // proposes; the LR samples (enforced by the core above) hold the
            // data term.
            cv::Mat px(Hr, Wc, CV_32F, x);
            cs_dct_inverse(px, cs_default_dct_block());
            if (red_kind >= 0) {
                cs_sr_prox_luma_plane(px, init_ycc_chs[1], init_ycc_chs[2],
                    red_kind, red_path, nlm_prox_h);
            } else {
                cs_sr_prox_luma_plane(px, cv::Mat(), cv::Mat(), -1, std::string(),
                    nlm_prox_h);
            }
            cs_dct_forward(px, cs_default_dct_block());
        }
        else if (red_r > 0.0f && red_kind >= 0) {
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

// Solved float luma (0..1, Hr x Wc) + the AVIR chroma planes -> BGR tile in
// hr_out; a non-finite or malformed result falls back to the plain AVIR upscale.
static void cs_sr_finish_luma_tile(const std::vector<float>& luma, int Hr, int Wc,
    const std::vector<cv::Mat>& avir_chs, const cv::Mat& lr_tile, cv::Mat& hr_out)
{
    bool finite = false;
    for (float v : luma) if (std::isfinite(v)) { finite = true; break; }
    if (!finite) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    cv::Mat plane(Hr, Wc, CV_32F, const_cast<float*>(luma.data()));
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
    cs_sr_finish_luma_tile(hr, Hr, Wc, avir_chs, lr_tile, hr_out);
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
    cs_sr_finish_luma_tile(img, Hr, Wc, avir_chs, lr_tile, hr_out);
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

#ifdef CS_GPU_TILES
void cs_sr_gpu_batch(std::vector<CsSrBatchTile>& tiles, const std::string& red_denoiser,
    bool sr_cascade, const CsSrPrior& sr_prior, std::vector<char>& done)
{
    done.assign(tiles.size(), 0);
    // Same gate as the per-tile GPU ADMM path of cs_sr_upscale_tile_2x (pnp_mode 2, NLM prox).
    if (tiles.empty() || sr_prior.pnp_mode != 2 || sr_cascade ||
        cs_onnx_kind_for(red_denoiser) >= 0 || cs_default_dct_block() > 0 || !cs_gpu_admm_enabled())
        return;
    // --sr-prior-model: the HIP Swin2SR (gpu_swin) replaces ONNX Runtime when the model's exported
    // weights are there and no segment gate is set (seg-gated tiles mix prior and cs-sr patches).
    // A tile at most 256 px a side is one ONNX patch, so the HIP output is the same warm start.
    const bool swin = !sr_prior.model.empty();
    if (swin) {
        if ((sr_prior.seg_lr && !sr_prior.seg_lr->empty()) || !cs_sr_gpu_prior_ready(sr_prior.model) ||
            !gpu_swin::load(gpu_swin::weights_path(sr_prior.model)))
            return;
        for (const CsSrBatchTile& T : tiles)
            if (!T.lr_tile || T.lr_tile->rows > 256 || T.lr_tile->cols > 256 ||
                !gpu_swin::supported(T.lr_tile->rows, T.lr_tile->cols))
                return;
    }
    const int admm = (std::min)(50, (std::max)(1, sr_prior.pnp_iters));
    const float rho = sr_prior.pnp_rho > 0.0f ? sr_prior.pnp_rho : 0.1f;
    const CsSrAdmmSched& sc = cs_sr_admm_sched();
    int verify_every = 0;
    if (const char* v = std::getenv("CS_GPU_SR_VERIFY")) verify_every = (std::max)(0, std::atoi(v));

    // Per-tile host data (kept alive until the GPU pass returns) and ADMM parameter groups.
    struct Prep {
        cv::Mat lr; std::vector<float> b, hr_map; gpu_admm::Params prm; bool ok = false;
        std::vector<std::uint8_t> dbg_avir; float dbg_h = 0.0f;
    };
    std::vector<Prep> pp(tiles.size());
    using Key = std::tuple<int, int, float, float, float, float, float, float, int, float, int>;
    std::map<Key, std::vector<size_t>> groups;
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
        Prep& P = pp[k];
        if (!T.lr_tile || T.lr_tile->empty() || T.lr_tile->type() != CV_8UC3) continue;
        const int Lr = T.lr_tile->rows, Lc = T.lr_tile->cols, Hr = 2 * Lr, Wc = 2 * Lc;
        const int m = (int)T.ri_x->size();
        if (Lr < 8 || Lc < 8 || !(T.tv >= 0.0f) || !gpu_admm::supported(Hr, Wc) ||
            T.meas->total() < (size_t)CS_HEADER_PIXELS + (size_t)m)
            continue;
        P.lr = T.lr_tile->isContinuous() ? *T.lr_tile : T.lr_tile->clone();
        cs_sr_luma_samples(*T.meas, m, P.b);
        std::vector<int> hidx; std::vector<float> hval;
        if (T.hr_sr.anchors == anchors && !sorted_anchors.empty())
            cs_sr_tile_anchors_sorted(sorted_anchors, T.hr_sr, Hr, Wc, hidx, hval);
        else
            cs_sr_tile_anchors(T.hr_sr, Hr, Wc, hidx, hval);
        const bool hr_on = !hidx.empty() && T.hr_sr.weight > 0.0f;
        if (hr_on) {
            P.hr_map.assign((size_t)Hr * Wc, -1e30f);
            for (size_t q = 0; q < hidx.size(); ++q) P.hr_map[(size_t)hidx[q]] = hval[q];
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
        const Prep& P = pp[k];
        if (!P.ok) continue;
        groups[Key(P.prm.admm, P.prm.inner, P.prm.rho, P.prm.lip, P.prm.wscale, P.prm.rho_end, P.prm.h_start,
            P.prm.h_end, P.prm.rescale_u, P.prm.hr_weight, P.prm.bt)].push_back(k);
    }

    for (auto& g : groups) {
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
            Prep& P = pp[k];
            const int n = 12 * P.lr.rows * P.lr.cols;
            if (pooled) *T.hr_out = pool.colRange((int)pool_off, (int)pool_off + n).reshape(3, 2 * P.lr.rows);
            else T.hr_out->create(2 * P.lr.rows, 2 * P.lr.cols, CV_8UC3);
            pool_off += (size_t)n;
            gpu_sr::Tile J{ P.lr.rows, P.lr.cols, P.lr.data, P.b.data(), T.ri_x->data(), T.ri_y->data(),
                (int)T.ri_x->size(), T.coef * 0.1f, T.tv, P.hr_map.empty() ? nullptr : P.hr_map.data(), T.hr_out->data };
            if (verify_every > 0 && k % (size_t)verify_every == 0) {
                P.dbg_avir.resize((size_t)12 * P.lr.rows * P.lr.cols);
                J.dbg_avir = P.dbg_avir.data();
                J.dbg_nlm_h = &P.dbg_h;
            }
            jobs.push_back(J);
        }
        if (!gpu_sr::run(jobs, g.second.empty() ? gpu_admm::Params{} : pp[g.second.front()].prm, swin)) {
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
            int sc2 = 0;
            if (swin) cs_onnx_sr_upscale_bgr(pp[k].lr, cpu_avir, sr_prior.model, sc2, cv::Mat(), {});
            else cs_upscale_2x_avir(pp[k].lr, cpu_avir);
            if (cpu_avir.total() * 3 != pp[k].dbg_avir.size()) continue;
            for (size_t q = 0; q < pp[k].dbg_avir.size(); ++q) {
                const int d = std::abs((int)cpu_avir.data[q] - (int)pp[k].dbg_avir[q]);
                ++n; a1 += d >= 1; a2 += d >= 2; amax = (std::max)(amax, d);
            }
            hmax = (std::max)(hmax, (double)std::fabs(cs_sr_prox_nlm_h(pp[k].lr) - pp[k].dbg_h));
            cv::Mat cpu_hr;
            cs_sr_upscale_tile_2x(pp[k].lr, *T.meas, *T.ri_x, *T.ri_y, T.coef, T.tv, T.iterations, 1, T.fista_iters,
                CS_BASIS_DCT, T.wscale, cpu_hr, 0.0f, red_denoiser, CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, T.hr_sr, sr_prior);
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
            nt, swin ? "prior" : "AVIR", n ? 100.0 * a1 / n : 0.0, n ? 100.0 * a2 / n : 0.0, amax, hmax, fn ? fsum / fn : 0.0, fn ? 100.0 * f2 / fn : 0.0);
    }
}

bool cs_sr_gpu_prior_ready(const std::string& model)
{
    // Only the weight file is checked here (no HIP call, so the device warm-up keeps running in
    // the background); cs_sr_gpu_batch loads it.
    const char* e = std::getenv("CS_GPU_SWIN");
    const char* a = std::getenv("CS_GPU_ADMM");   // as cs_gpu_admm_enabled, without its device query
    if (model.empty() || (e && std::strcmp(e, "0") == 0) || !(a && a[0] == '1')) return false;
    std::FILE* f = std::fopen(gpu_swin::weights_path(model).c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}
#else
void cs_sr_gpu_batch(std::vector<CsSrBatchTile>& tiles, const std::string&, bool, const CsSrPrior&,
    std::vector<char>& done)
{
    done.assign(tiles.size(), 0);
}

bool cs_sr_gpu_prior_ready(const std::string&) { return false; }
#endif
