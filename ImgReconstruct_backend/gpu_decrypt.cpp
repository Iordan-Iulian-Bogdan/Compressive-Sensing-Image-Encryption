// Batched GPU solve of decrypt tiles (see gpu_decrypt.hpp).
#include "gpu_decrypt.hpp"
#include "gpu_tiles.h"
#include "gap_tv.hpp"
#include "fista_kernels.hpp"
#include "helper_functions.hpp"
#include "cs_wavelet.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>

namespace {
struct Channel {
    std::vector<float> b;   // samples in [0,1]
    std::vector<float> x;   // DCT coefficients: warm start in, solution out
    int inner = 0;          // FISTA steps for this channel's batch key
    bool prepared = false;
    bool ok = false;
};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

void cs_gpu_solve_tiles(std::vector<GpuTileJob>& jobs, std::vector<char>& solved) {
    solved.assign(jobs.size(), 0);
    if (jobs.empty() || !gpu_tiles::available()) return;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::array<Channel, 3>> st(jobs.size());

    // Per tile, independent: samples, smooth fill, DCT warm start.
    const int nj = (int)jobs.size();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < nj; ++j) {
        const GpuTileJob& J = jobs[(size_t)j];
        const int m = (int)J.rix->size();
        if (m <= 0 || J.rows < 1 || J.cols < 1) continue;
        for (int ch = 0; ch < 3; ++ch) {
            Channel& C = st[(size_t)j][(size_t)ch];
            const int it = ch == 0 ? J.iterations : J.iterations - J.iterations / 2;
            C.inner = cs_fista_inner_iters(it, J.fista_iters);
            cs_point_sample_fill(*J.measurements, ch, J.rows, J.cols, *J.rix, *J.riy, C.b, C.x);
            cv::Mat X(J.rows, J.cols, CV_32F, C.x.data());
            cs_dct_forward(X, 0);   // C.x now holds the coefficients
            C.prepared = true;
        }
    }
    // (inner steps, reweights, wscale) -> (job, channel) members
    std::map<std::tuple<int, int, float>, std::vector<std::pair<int, int>>> groups;
    for (size_t j = 0; j < jobs.size(); ++j)
        for (int ch = 0; ch < 3; ++ch)
            if (st[j][(size_t)ch].prepared)
                groups[std::make_tuple(st[j][(size_t)ch].inner, jobs[j].reweights, jobs[j].wscale)]
                    .push_back({ (int)j, ch });
    const double t_prep = ms_since(t0);

    const auto t1 = std::chrono::steady_clock::now();
    for (auto& g : groups) {
        std::vector<gpu_tiles::TileProblem> probs;
        probs.reserve(g.second.size());
        for (const auto& jc : g.second) {
            const GpuTileJob& J = jobs[(size_t)jc.first];
            Channel& C = st[(size_t)jc.first][(size_t)jc.second];
            probs.push_back({ J.rows, J.cols, C.b.data(), J.rix->data(), J.riy->data(),
                (int)J.rix->size(), C.x.data(), J.coef, J.tv });
        }
        const gpu_tiles::Params prm{ std::get<0>(g.first), std::get<1>(g.first), std::get<2>(g.first) };
        if (!gpu_tiles::solve_fista_dct(probs, prm)) continue;
        for (const auto& jc : g.second) st[(size_t)jc.first][(size_t)jc.second].ok = true;
    }
    const double t_gpu = ms_since(t1);

    // Tail identical to the CPU path: inverse DCT, scale to 0..255, merge to 8-bit BGR.
    const auto t2 = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < nj; ++j) {
        if (!(st[(size_t)j][0].ok && st[(size_t)j][1].ok && st[(size_t)j][2].ok)) continue;
        const GpuTileJob& J = jobs[(size_t)j];
        cv::Mat planes[3];
        for (int ch = 0; ch < 3; ++ch) {
            planes[ch] = cv::Mat(J.rows, J.cols, CV_32F, st[(size_t)j][(size_t)ch].x.data());
            cs_dct_inverse(planes[ch], 0);
            planes[ch] = planes[ch] * 255.0f;
        }
        cv::merge(planes, 3, *J.out);
        J.out->convertTo(*J.out, CV_8UC3);
        solved[(size_t)j] = 1;
    }
    std::fprintf(stderr, "[gpu-tiles] phases: prep %.0f ms, gpu %.0f ms (%zu batches), tail %.0f ms\n",
        t_prep, t_gpu, groups.size(), ms_since(t2));
}
