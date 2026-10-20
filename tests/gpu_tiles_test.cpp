// Standalone check of gpu_tiles::solve_fista_dct against a plain CPU reference
// of the same algorithm (dense orthonormal DCT, reweighted-L1 FISTA).
// Build: tests/build_gpu_tiles_test.bat
#include "gpu_tiles.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace {

const double kPi = 3.14159265358979323846;

std::vector<float> dct_table(int N) {
    std::vector<float> C((size_t)N * N);
    for (int k = 0; k < N; ++k) {
        const double a = k == 0 ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
        for (int n = 0; n < N; ++n)
            C[(size_t)k * N + n] = (float)(a * std::cos(kPi * (2.0 * n + 1.0) * k / (2.0 * N)));
    }
    return C;
}

// out = Cr * in * Cc^T
void dct2(const float* in, float* out, int R, int Cn, const std::vector<float>& Cr, const std::vector<float>& Cc) {
    std::vector<float> tmp((size_t)R * Cn);
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < Cn; ++k) {
            float s = 0.0f;
            for (int n = 0; n < Cn; ++n) s += in[r * Cn + n] * Cc[(size_t)k * Cn + n];
            tmp[(size_t)r * Cn + k] = s;
        }
    for (int u = 0; u < R; ++u)
        for (int k = 0; k < Cn; ++k) {
            float s = 0.0f;
            for (int r = 0; r < R; ++r) s += Cr[(size_t)u * R + r] * tmp[(size_t)r * Cn + k];
            out[u * Cn + k] = s;
        }
}

// out = Cr^T * in * Cc
void idct2(const float* in, float* out, int R, int Cn, const std::vector<float>& Cr, const std::vector<float>& Cc) {
    std::vector<float> tmp((size_t)R * Cn);
    for (int r = 0; r < R; ++r)
        for (int n = 0; n < Cn; ++n) {
            float s = 0.0f;
            for (int k = 0; k < Cn; ++k) s += in[r * Cn + k] * Cc[(size_t)k * Cn + n];
            tmp[(size_t)r * Cn + n] = s;
        }
    for (int m = 0; m < R; ++m)
        for (int n = 0; n < Cn; ++n) {
            float s = 0.0f;
            for (int r = 0; r < R; ++r) s += Cr[(size_t)r * R + m] * tmp[(size_t)r * Cn + n];
            out[m * Cn + n] = s;
        }
}

struct Tile {
    int R, C, m;
    std::vector<float> truth, b, x0;
    std::vector<int> rix, riy;
};

Tile make_tile(std::mt19937& rng, int R, int C, float ratio) {
    Tile t; t.R = R; t.C = C;
    t.truth.resize((size_t)R * C);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float f1 = 0.05f + 0.2f * u(rng), f2 = 0.05f + 0.2f * u(rng), ph = 6.28f * u(rng);
    const float bx = u(rng) * C, by = u(rng) * R;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            float v = 0.5f + 0.25f * std::sin(f1 * c + ph) * std::cos(f2 * r)
                + 0.3f * std::exp(-((c - bx) * (c - bx) + (r - by) * (r - by)) / 60.0f)
                + 0.03f * (u(rng) - 0.5f);
            t.truth[(size_t)r * C + c] = std::min(1.0f, std::max(0.0f, v));
        }
    std::vector<int> idx((size_t)R * C);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::shuffle(idx.begin(), idx.end(), rng);
    t.m = std::max(1, (int)(ratio * R * C));
    double mean = 0.0;
    for (int k = 0; k < t.m; ++k) {
        t.rix.push_back(idx[(size_t)k] / C);
        t.riy.push_back(idx[(size_t)k] % C);
        t.b.push_back(t.truth[(size_t)idx[(size_t)k]]);
        mean += t.b.back();
    }
    mean /= t.m;
    t.x0.assign((size_t)R * C, 0.0f);
    t.x0[0] = (float)(mean * std::sqrt((double)R * C));   // orthonormal DC of a constant image
    return t;
}

// Same algorithm as the GPU kernels / CPU cs_fista_core_single (DCT, tv = 0).
void ref_solve(Tile& t, float lambda, int inner, int reweights, float wscale, std::vector<float>& x) {
    const int R = t.R, C = t.C, n = R * C;
    const auto Cr = dct_table(R), Cc = dct_table(C);
    x = t.x0;
    std::vector<float> w((size_t)n, 1.0f), ws((size_t)n, 1.0f), y, xp, z((size_t)n), pix((size_t)n), sc((size_t)n), g((size_t)n);
    if (wscale > 1.0f) {
        const double logw = std::log((double)wscale);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                const double fr = R > 1 ? (double)r / (R - 1) : 0.0, fc = C > 1 ? (double)c / (C - 1) : 0.0;
                ws[(size_t)r * C + c] = (float)std::exp(logw * 0.5 * (fr + fc));
            }
    }
    for (int rw = 0; rw < reweights; ++rw) {
        y = x; xp = x;
        float tt = 1.0f;
        for (int k = 0; k < inner; ++k) {
            idct2(y.data(), pix.data(), R, C, Cr, Cc);
            std::fill(sc.begin(), sc.end(), 0.0f);
            for (int s = 0; s < t.m; ++s) {
                const int i = t.rix[(size_t)s] * C + t.riy[(size_t)s];
                sc[(size_t)i] = pix[(size_t)i] - t.b[(size_t)s];
            }
            dct2(sc.data(), g.data(), R, C, Cr, Cc);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * tt * tt));
            const float mom = (tt - 1.0f) / t_new;
            for (int i = 0; i < n; ++i) {
                const float v = y[(size_t)i] - g[(size_t)i];
                const float thr = 0.5f * lambda * w[(size_t)i] * ws[(size_t)i];
                const float a = std::fabs(v);
                const float zz = a > thr ? (v > 0.0f ? a - thr : thr - a) : 0.0f;
                z[(size_t)i] = zz;
                y[(size_t)i] = zz + mom * (zz - xp[(size_t)i]);
                xp[(size_t)i] = zz;
            }
            tt = t_new;
        }
        x = z;
        if (rw + 1 < reweights) {
            float mx = 0.0f;
            for (float v : x) mx = std::max(mx, std::fabs(v));
            float eps = 0.02f * mx; if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i) w[(size_t)i] = eps / (std::fabs(x[(size_t)i]) + eps);
        }
    }
}

double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double)a[i] - b[i]; se += d * d; }
    se /= a.size();
    return se <= 1e-20 ? 200.0 : 10.0 * std::log10(1.0 / se);
}

}  // namespace

int main() {
    if (!gpu_tiles::available()) { std::printf("no HIP device\n"); return 2; }
    const int NT = 4096, NCHECK = 96;
    const int sizes[][2] = { {32, 24}, {32, 26}, {34, 24}, {30, 22}, {28, 24}, {64, 48}, {20, 16} };
    std::mt19937 rng(12345);
    std::vector<Tile> tiles;
    for (int i = 0; i < NT; ++i) {
        const auto& s = sizes[i % 7];
        tiles.push_back(make_tile(rng, s[0], s[1], 0.3f + 0.6f * ((i % 5) / 4.0f)));
    }
    const float lambda = 0.03f;
    const gpu_tiles::Params prm{ 20, 2, 2.0f };

    std::vector<std::vector<float>> gx(tiles.size());
    std::vector<gpu_tiles::TileProblem> probs;
    for (size_t i = 0; i < tiles.size(); ++i) {
        gx[i] = tiles[i].x0;
        probs.push_back({ tiles[i].R, tiles[i].C, tiles[i].b.data(), tiles[i].rix.data(), tiles[i].riy.data(), tiles[i].m, gx[i].data(), lambda });
    }
    auto time_gpu = [&]() {
        for (size_t i = 0; i < tiles.size(); ++i) gx[i] = tiles[i].x0;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = gpu_tiles::solve_fista_dct(probs, prm);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("gpu solve_fista_dct(%d tiles): %s, %.1f ms\n", NT, ok ? "ok" : "FAILED",
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        return ok;
    };
    if (!time_gpu()) return 1;   // first call (device init)
    if (!time_gpu()) return 1;   // steady state

    double max_abs = 0.0, worst_rel = 0.0, min_psnr_vs_ref = 1e9, sum_gpu_t = 0.0, sum_ref_t = 0.0;
    const auto c0 = std::chrono::steady_clock::now();
    for (int i = 0; i < NCHECK; ++i) {
        std::vector<float> rx;
        ref_solve(tiles[(size_t)i], lambda, prm.inner, prm.reweights, prm.wscale, rx);
        double num = 0.0, den = 0.0;
        for (size_t k = 0; k < rx.size(); ++k) {
            const double d = (double)gx[(size_t)i][k] - rx[k];
            max_abs = std::max(max_abs, std::fabs(d));
            num += d * d; den += (double)rx[k] * rx[k];
        }
        worst_rel = std::max(worst_rel, std::sqrt(num / std::max(den, 1e-30)));
        const Tile& T = tiles[(size_t)i];
        const auto Cr = dct_table(T.R), Cc = dct_table(T.C);
        std::vector<float> pg((size_t)T.R * T.C), pr((size_t)T.R * T.C);
        idct2(gx[(size_t)i].data(), pg.data(), T.R, T.C, Cr, Cc);
        idct2(rx.data(), pr.data(), T.R, T.C, Cr, Cc);
        min_psnr_vs_ref = std::min(min_psnr_vs_ref, psnr(pg, pr));
        sum_gpu_t += psnr(pg, T.truth); sum_ref_t += psnr(pr, T.truth);
    }
    const auto c1 = std::chrono::steady_clock::now();
    const double ref_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    std::printf("checked %d tiles against the CPU reference:\n", NCHECK);
    std::printf("  max |coef diff| = %.3e   worst relative L2 = %.3e\n", max_abs, worst_rel);
    std::printf("  min PSNR(gpu pixels vs reference pixels) = %.1f dB\n", min_psnr_vs_ref);
    std::printf("  mean PSNR vs ground truth: gpu %.2f dB, reference %.2f dB\n", sum_gpu_t / NCHECK, sum_ref_t / NCHECK);
    std::printf("  single-thread CPU reference: %.1f ms for %d tiles (%.2f ms/tile)\n", ref_ms, NCHECK, ref_ms / NCHECK);
    return 0;
}
