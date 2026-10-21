// Standalone check of gpu_admm against the CPU reference algorithm:
//  1. the NLM kernel vs cv::fastNlMeansDenoising(h, 7, 15) on 8-bit planes;
//  2. the full PnP-ADMM loop vs a CPU re-implementation built from cv::dct,
//     cv::fastNlMeansDenoising and plain loops (same math as superres.cpp).
// Build: tests/build_gpu_admm_test.bat
#include "gpu_admm.h"

#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

cv::Mat make_truth(std::mt19937& rng, int R, int C) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float f1 = 0.03f + 0.15f * u(rng), f2 = 0.03f + 0.15f * u(rng), ph = 6.28f * u(rng);
    const float bx = u(rng) * C, by = u(rng) * R;
    cv::Mat t(R, C, CV_32F);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            float v = 0.5f + 0.25f * std::sin(f1 * c + ph) * std::cos(f2 * r)
                + 0.3f * std::exp(-((c - bx) * (c - bx) + (r - by) * (r - by)) / 200.0f)
                + (((c / 9 + r / 7) & 1) ? 0.08f : -0.08f);
            t.at<float>(r, c) = std::min(1.0f, std::max(0.0f, v));
        }
    return t;
}

struct Case {
    int R, C, m;
    cv::Mat truth, z0;
    std::vector<float> b;
    std::vector<int> rix, riy;
    float lambda, h;
};

Case make_case(std::mt19937& rng, int R, int C, float ratio) {
    Case k; k.R = R; k.C = C;
    k.truth = make_truth(rng, R, C);
    const int Lr = R / 2, Lc = C / 2;
    std::vector<int> idx((size_t)Lr * Lc);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::shuffle(idx.begin(), idx.end(), rng);
    k.m = std::max(1, (int)(ratio * Lr * Lc));
    for (int s = 0; s < k.m; ++s) {
        const int i = idx[(size_t)s] / Lc, j = idx[(size_t)s] % Lc;
        k.rix.push_back(i); k.riy.push_back(j);
        k.b.push_back(0.25f * (k.truth.at<float>(2 * i, 2 * j) + k.truth.at<float>(2 * i, 2 * j + 1) +
            k.truth.at<float>(2 * i + 1, 2 * j) + k.truth.at<float>(2 * i + 1, 2 * j + 1)));
    }
    cv::GaussianBlur(k.truth, k.z0, cv::Size(0, 0), 1.2);
    cv::Mat noise(R, C, CV_32F);
    cv::randn(noise, 0.0, 0.02);
    k.z0 += noise;
    k.lambda = 0.003f;
    k.h = 4.5f;
    return k;
}

void ref_admm(const Case& k, int admm, int inner, float rho, float lip, float wscale, cv::Mat& zout) {
    const int R = k.R, C = k.C;
    cv::Mat z = k.z0.clone(), u = cv::Mat::zeros(R, C, CV_32F);
    cv::Mat ws(R, C, CV_32F, cv::Scalar(1.0f));
    if (wscale > 1.0f) {
        const double lw = std::log((double)wscale);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                ws.at<float>(r, c) = (float)std::exp(lw * 0.5 * ((double)r / (R - 1) + (double)c / (C - 1)));
    }
    const float L = 0.25f * lip + 2.0f * rho, step = 1.0f / L;
    for (int it = 0; it < admm; ++it) {
        cv::Mat anch, x, y, xp, zc, a = z - u;
        cv::dct(a, anch);
        cv::dct(z, x);
        y = x.clone(); xp = x.clone(); zc = x.clone();
        float t = 1.0f;
        for (int kk = 0; kk < inner; ++kk) {
            cv::Mat pix, sc = cv::Mat::zeros(R, C, CV_32F), G;
            cv::dct(y, pix, cv::DCT_INVERSE);
            for (int s = 0; s < k.m; ++s) {
                const int i = k.rix[(size_t)s], j = k.riy[(size_t)s];
                const float avg = 0.25f * (pix.at<float>(2 * i, 2 * j) + pix.at<float>(2 * i, 2 * j + 1) +
                    pix.at<float>(2 * i + 1, 2 * j) + pix.at<float>(2 * i + 1, 2 * j + 1));
                const float v = 0.25f * (avg - k.b[(size_t)s]);
                sc.at<float>(2 * i, 2 * j) = v; sc.at<float>(2 * i, 2 * j + 1) = v;
                sc.at<float>(2 * i + 1, 2 * j) = v; sc.at<float>(2 * i + 1, 2 * j + 1) = v;
            }
            cv::dct(sc, G);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) {
                    const float yy = y.at<float>(r, c), aa = anch.at<float>(r, c);
                    const float g = 2.0f * G.at<float>(r, c) + 2.0f * rho * (yy - aa);
                    const float v = yy - step * g - aa;
                    const float thr = k.lambda * step * ws.at<float>(r, c);
                    const float av = std::fabs(v);
                    const float zz = aa + (av > thr ? (v > 0.0f ? 1.0f : -1.0f) * (av - thr) : 0.0f);
                    y.at<float>(r, c) = zz + mom * (zz - xp.at<float>(r, c));
                    xp.at<float>(r, c) = zz;
                    zc.at<float>(r, c) = zz;
                }
            t = t_new;
        }
        cv::Mat px, v, u8, dn, zp;
        cv::dct(zc, px, cv::DCT_INVERSE);
        v = px + u;
        v.convertTo(u8, CV_8U, 255.0);
        cv::fastNlMeansDenoising(u8, dn, k.h, 7, 15);
        dn.convertTo(zp, CV_32F, 1.0 / 255.0);
        u += px - zp;
        z = zp;
    }
    zout = z;
}

double psnr(const cv::Mat& a, const cv::Mat& b) {
    const double mse = cv::norm(a, b, cv::NORM_L2SQR) / (double)a.total();
    return mse <= 1e-20 ? 200.0 : 10.0 * std::log10(1.0 / mse);
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    // 1. NLM kernel vs OpenCV
    {
        const int sizes[][2] = { {112, 92}, {64, 64}, {96, 130}, {12, 12}, {40, 24} };
        std::vector<cv::Mat> src, ref, got;
        std::vector<gpu_admm::NlmPlane> planes;
        const float hs[] = { 3.0f, 4.5f, 8.0f, 14.0f, 30.0f };
        for (int i = 0; i < 25; ++i) {
            const auto& s = sizes[i % 5];
            cv::Mat f = make_truth(rng, s[0], s[1]), n(s[0], s[1], CV_32F);
            cv::randn(n, 0.0, 0.04 + 0.01 * (i % 4));
            cv::Mat u8;
            cv::Mat fn = f + n;
            fn.convertTo(u8, CV_8U, 255.0);
            src.push_back(u8);
            got.push_back(cv::Mat(s[0], s[1], CV_8U));
            cv::Mat r;
            cv::fastNlMeansDenoising(u8, r, hs[i % 5], 7, 15);
            ref.push_back(r);
        }
        for (size_t i = 0; i < src.size(); ++i)
            planes.push_back({ src[i].rows, src[i].cols, src[i].data, got[i].data, hs[i % 5] });
        if (!gpu_admm::nlm_denoise_u8(planes)) { std::printf("nlm_denoise_u8 FAILED\n"); return 1; }
        long long diff_px = 0, tot = 0; int maxd = 0;
        for (size_t i = 0; i < src.size(); ++i) {
            cv::Mat d; cv::absdiff(got[i], ref[i], d);
            for (int r = 0; r < d.rows; ++r)
                for (int c = 0; c < d.cols; ++c) {
                    const int v = d.at<unsigned char>(r, c);
                    if (v) ++diff_px;
                    maxd = std::max(maxd, v);
                }
            tot += (long long)d.total();
        }
        std::printf("NLM vs OpenCV: %zu planes, %lld of %lld pixels differ, max |diff| = %d\n",
            src.size(), diff_px, tot, maxd);
    }
    // 2. ADMM loop vs CPU reference
    {
        const int NT = 48;
        const int sizes[][2] = { {112, 92}, {96, 96}, {120, 88}, {100, 100} };
        std::vector<Case> cases;
        for (int i = 0; i < NT; ++i) {
            const auto& s = sizes[i % 4];
            cases.push_back(make_case(rng, s[0], s[1], 0.35f + 0.5f * ((i % 5) / 4.0f)));
        }
        const gpu_admm::Params prm{ 8, 40, 0.1f, 2.0f, 2.0f };
        std::vector<cv::Mat> gz;
        std::vector<gpu_admm::Problem> probs;
        for (auto& k : cases) gz.push_back(k.z0.clone());
        for (size_t i = 0; i < cases.size(); ++i) {
            const Case& k = cases[i];
            probs.push_back({ k.R, k.C, k.b.data(), k.rix.data(), k.riy.data(), k.m, (float*)gz[i].data, k.lambda, k.h });
        }
        for (int rep = 0; rep < 2; ++rep) {
            for (size_t i = 0; i < cases.size(); ++i) cases[i].z0.copyTo(gz[i]);
            const auto t0 = std::chrono::steady_clock::now();
            if (!gpu_admm::solve_pnp_admm(probs, prm)) { std::printf("solve_pnp_admm FAILED\n"); return 1; }
            std::printf("gpu solve_pnp_admm(%d tiles): %.1f ms\n", NT, ms_since(t0));
        }
        double min_psnr = 1e9, sum_gpu = 0, sum_ref = 0, sum_z0 = 0;
        const auto c0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < cases.size(); ++i) {
            cv::Mat r;
            ref_admm(cases[i], prm.admm, prm.inner, prm.rho, prm.lip, prm.wscale, r);
            min_psnr = std::min(min_psnr, psnr(gz[i], r));
            sum_gpu += psnr(gz[i], cases[i].truth);
            sum_ref += psnr(r, cases[i].truth);
            sum_z0 += psnr(cases[i].z0, cases[i].truth);
        }
        // Batch-size scaling of the whole ADMM call (latency vs throughput).
        for (int n : { 1, 2, 4, 8, 16, 32, 48 }) {
            std::vector<gpu_admm::Problem> sub(probs.begin(), probs.begin() + n);
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                gpu_admm::solve_pnp_admm(sub, prm);
                best = std::min(best, ms_since(t0));
            }
            std::printf("  batch %2d tiles: %7.1f ms  (%.2f ms/tile)\n", n, best, best / n);
        }
        std::printf("checked %d tiles against the CPU reference:\n", NT);
        std::printf("  min PSNR(gpu vs reference) = %.1f dB\n", min_psnr);
        std::printf("  mean PSNR vs ground truth: start %.2f dB, gpu %.2f dB, reference %.2f dB\n",
            sum_z0 / NT, sum_gpu / NT, sum_ref / NT);
        std::printf("  CPU reference: %.1f ms for %d tiles (%.1f ms/tile, single thread)\n",
            ms_since(c0), NT, ms_since(c0) / NT);
    }
    return 0;
}
