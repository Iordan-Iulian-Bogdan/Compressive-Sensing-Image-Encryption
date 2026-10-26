// Batched GPU solve of decrypt tiles (see gpu_decrypt.hpp).
#include "gpu_decrypt.hpp"
#include "gpu_tiles.h"
#include "gap_tv.hpp"
#include "fista_kernels.hpp"
#include "helper_functions.hpp"
#include "cs_wavelet.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>

namespace {
// One plane problem: inputs, and the plane itself (DCT warm start -> solution -> pixels x 255).
struct Plane {
    const cv::Mat* meas = nullptr;
    int ch = 0, rows = 0, cols = 0;
    const std::vector<int>* rix = nullptr;
    const std::vector<int>* riy = nullptr;
    int inner = 0, reweights = 1;
    float wscale = 1.0f, coef = 0.0f, tv = 0.0f;
    std::vector<float> b;   // samples in [0,1]
    std::vector<float> x;   // DCT coefficients: warm start in, solution out; pixel values x 255 after the tail
    bool prepared = false;
    bool ok = false;
};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Smooth-fill DCT warm start, batched FISTA grouped by (inner steps, reweights, wscale), inverse DCT x 255.
// Planes whose GPU batch failed keep ok = false. Returns false without touching anything when no device.
bool solve_planes(std::vector<Plane>& ps, const char* tag) {
    if (ps.empty() || !gpu_tiles::available()) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const int np = (int)ps.size();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < np; ++k) {
        Plane& P = ps[(size_t)k];
        if (P.rix->empty() || P.rows < 1 || P.cols < 1) continue;
        cs_point_sample_fill(*P.meas, P.ch, P.rows, P.cols, *P.rix, *P.riy, P.b, P.x);
        cv::Mat X(P.rows, P.cols, CV_32F, P.x.data());
        cs_dct_forward(X, 0);   // P.x now holds the coefficients
        P.prepared = true;
    }
    std::map<std::tuple<int, int, float>, std::vector<int>> groups;
    for (int k = 0; k < np; ++k)
        if (ps[(size_t)k].prepared)
            groups[std::make_tuple(ps[(size_t)k].inner, ps[(size_t)k].reweights, ps[(size_t)k].wscale)].push_back(k);
    const double t_prep = ms_since(t0);

    const auto t1 = std::chrono::steady_clock::now();
    for (auto& g : groups) {
        std::vector<gpu_tiles::TileProblem> probs;
        probs.reserve(g.second.size());
        for (int k : g.second) {
            Plane& P = ps[(size_t)k];
            probs.push_back({ P.rows, P.cols, P.b.data(), P.rix->data(), P.riy->data(),
                (int)P.rix->size(), P.x.data(), P.coef, P.tv });
        }
        const gpu_tiles::Params prm{ std::get<0>(g.first), std::get<1>(g.first), std::get<2>(g.first) };
        if (!gpu_tiles::solve_fista_dct(probs, prm)) continue;
        for (int k : g.second) ps[(size_t)k].ok = true;
    }
    const double t_gpu = ms_since(t1);

    // Tail identical to the CPU core's: inverse DCT, scale to 0..255.
    const auto t2 = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < np; ++k) {
        Plane& P = ps[(size_t)k];
        if (!P.ok) continue;
        cv::Mat X(P.rows, P.cols, CV_32F, P.x.data());
        cs_dct_inverse(X, 0);
        X = X * 255.0f;
    }
    std::fprintf(stderr, "[%s] phases: prep %.0f ms, gpu %.0f ms (%zu batches), tail %.0f ms\n",
        tag, t_prep, t_gpu, groups.size(), ms_since(t2));
    return true;
}
}  // namespace

void cs_gpu_solve_tiles(std::vector<GpuTileJob>& jobs, std::vector<char>& solved) {
    solved.assign(jobs.size(), 0);
    std::vector<Plane> ps(jobs.size() * 3);
    for (size_t j = 0; j < jobs.size(); ++j) {
        const GpuTileJob& J = jobs[j];
        for (int ch = 0; ch < 3; ++ch) {
            Plane& P = ps[j * 3 + (size_t)ch];
            const int it = ch == 0 ? J.iterations : J.iterations - J.iterations / 2;
            P.meas = J.measurements; P.ch = ch; P.rows = J.rows; P.cols = J.cols;
            P.rix = J.rix; P.riy = J.riy;
            P.inner = cs_fista_inner_iters(it, J.fista_iters); P.reweights = J.reweights; P.wscale = J.wscale;
            P.coef = J.coef; P.tv = J.tv;
        }
    }
    if (!solve_planes(ps, "gpu-tiles")) return;

    // Merge to 8-bit BGR, as the CPU path.
    const int nj = (int)jobs.size();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < nj; ++j) {
        Plane* P = &ps[(size_t)j * 3];
        if (!(P[0].ok && P[1].ok && P[2].ok)) continue;
        const GpuTileJob& J = jobs[(size_t)j];
        cv::Mat planes[3];
        for (int ch = 0; ch < 3; ++ch) planes[ch] = cv::Mat(J.rows, J.cols, CV_32F, P[ch].x.data());
        cv::merge(planes, 3, *J.out);
        J.out->convertTo(*J.out, CV_8UC3);
        solved[(size_t)j] = 1;
    }
}

void cs_gpu_solve_tiles_ycc(std::vector<GpuYccTileJob>& jobs, std::vector<char>& solved) {
    solved.assign(jobs.size(), 0);
    std::vector<Plane> ps(jobs.size() * 3);
    for (size_t j = 0; j < jobs.size(); ++j) {
        const GpuYccTileJob& J = jobs[j];
        const int rw = J.reweights < 1 ? 1 : (J.reweights > 5 ? 5 : J.reweights);   // CPU core's clamp
        for (int c = 0; c < 3; ++c) {
            const GpuPlane& G = J.plane[c];
            Plane& P = ps[j * 3 + (size_t)c];
            P.meas = G.measurements; P.ch = 0; P.rows = G.rows; P.cols = G.cols;
            P.rix = G.rix; P.riy = G.riy;
            P.inner = cs_fista_inner_iters(G.iterations, J.fista_iters); P.reweights = rw; P.wscale = J.wscale;
            P.coef = G.coef; P.tv = G.tv;
        }
    }
    if (!solve_planes(ps, "gpu-tiles-ycc")) return;

    // Same tail as decrypt_tiles_ycc420: chroma bilinear to the luma size, merge YCrCb, 8 bit, BGR.
    const int nj = (int)jobs.size();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < nj; ++j) {
        Plane* P = &ps[(size_t)j * 3];
        if (!(P[0].ok && P[1].ok && P[2].ok)) continue;
        const GpuYccTileJob& J = jobs[(size_t)j];
        const int Th = P[0].rows, Tw = P[0].cols;
        cv::Mat y(Th, Tw, CV_32F, P[0].x.data()), cr_full, cb_full;
        cv::resize(cv::Mat(P[1].rows, P[1].cols, CV_32F, P[1].x.data()), cr_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
        cv::resize(cv::Mat(P[2].rows, P[2].cols, CV_32F, P[2].x.data()), cb_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
        cv::Mat chs[3] = { y, cr_full, cb_full }, ycc;
        cv::merge(chs, 3, ycc);
        ycc.convertTo(ycc, CV_8UC3);
        cv::cvtColor(ycc, *J.out, cv::COLOR_YCrCb2BGR);
        solved[(size_t)j] = 1;
    }
}
