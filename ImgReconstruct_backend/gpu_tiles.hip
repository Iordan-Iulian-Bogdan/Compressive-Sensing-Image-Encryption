// Batched GPU tile solver (see gpu_tiles.h).
//
// Layout: every tile owns a contiguous slice [off, off + rows*cols) of each
// state array. DCT work is two small dense matrix products per transform
// (T = X * Cc^T, then C_r * T for the forward transform), which is cheap for
// tiles up to ~100 px and needs no FFT library. Kernels use blockIdx.y as the
// tile index and blockIdx.x * kThreads + threadIdx.x as the element index.
#include "gpu_tiles.h"

#include "gpu_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace gpu_tiles {
namespace {

using namespace gpu_common;

// Residual at the samples, zero elsewhere (SC is zeroed first).
__global__ void k_scatter(const TileDev* td, const int* stile, const int* rix, const int* riy,
    const float* b, const float* PIX, float* SC, int total) {
    const int s = blockIdx.x * kThreads + threadIdx.x;
    if (s >= total) return;
    const TileDev T = td[stile[s]];
    const int idx = T.off + rix[s] * T.cols + riy[s];
    SC[idx] = PIX[idx] - b[s];
}

// Soft threshold + FISTA momentum. Lipschitz constant 2 => step 1/2, and the
// gradient 2*DCT(sc) turns the update into v = y - G with threshold lambda*w*ws/2.
__global__ void k_step(const TileDev* td, const float* G, const float* W, const float* WS,
    float* Y, float* XP, float* Z, const float* LAM, float mom) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float v = Y[o] - G[o];
    const float thr = 0.5f * LAM[blockIdx.y] * W[o] * WS[o];
    const float a = fabsf(v);
    const float z = a > thr ? (v > 0.0f ? a - thr : thr - a) : 0.0f;
    Y[o] = z + mom * (z - XP[o]);
    XP[o] = z;
    Z[o] = z;
}

// eps = max(0.02 * max|z|, 1e-3) per tile (one block per tile).
__global__ void k_absmax(const TileDev* td, const float* Z, float* EPS) {
    __shared__ float sh[kThreads];
    const TileDev T = td[blockIdx.x];
    const int n = T.rows * T.cols;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) m = fmaxf(m, fabsf(Z[T.off + i]));
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float eps = 0.02f * sh[0];
        EPS[blockIdx.x] = eps < 1e-3f ? 1e-3f : eps;
    }
}

__global__ void k_wupdate(const TileDev* td, const float* Z, const float* EPS, float* W) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float e = EPS[blockIdx.y];
    W[o] = e / (fabsf(Z[o]) + e);
}

}  // namespace

bool available() {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

bool solve_fista_dct(const std::vector<TileProblem>& tiles, const Params& prm) {
    if (tiles.empty()) return true;
    if (!available()) return false;
    const int nt = (int)tiles.size();

    // Host-side layout.
    std::vector<TileDev> td((size_t)nt);
    std::map<int, int> table_off;           // N -> offset in the table array
    std::vector<float> table;
    auto table_for = [&](int N) {
        auto it = table_off.find(N);
        if (it != table_off.end()) return it->second;
        const int off = (int)table.size();
        table.resize(table.size() + (size_t)N * N);
        fill_dct_table(table.data() + off, N);
        table_off[N] = off;
        return off;
    };
    long long total = 0;
    int max_n = 0, total_samples = 0;
    for (int t = 0; t < nt; ++t) {
        const TileProblem& P = tiles[(size_t)t];
        TileDev d{};
        d.rows = P.rows; d.cols = P.cols;
        d.off = (int)total;
        d.rtab = table_for(P.rows);
        d.ctab = table_for(P.cols);
        td[(size_t)t] = d;
        total += (long long)P.rows * P.cols;
        max_n = std::max(max_n, P.rows * P.cols);
        total_samples += P.m;
    }
    if (total > 2000000000LL) return false;

    std::vector<float> hX((size_t)total), hb((size_t)total_samples);
    std::vector<int> hrix((size_t)total_samples), hriy((size_t)total_samples), hst((size_t)total_samples);
    {
        int so = 0;
        for (int t = 0; t < nt; ++t) {
            const TileProblem& P = tiles[(size_t)t];
            std::copy(P.x, P.x + (size_t)P.rows * P.cols, hX.begin() + td[(size_t)t].off);
            for (int k = 0; k < P.m; ++k) {
                hb[(size_t)so + k] = P.b[k];
                hrix[(size_t)so + k] = P.rix[k];
                hriy[(size_t)so + k] = P.riy[k];
                hst[(size_t)so + k] = t;
            }
            so += P.m;
        }
    }

    DevBuf<TileDev> dtd; DevBuf<float> dtab, dX, dY, dXP, dZ, dW, dWS, dTMP, dPIX, dSC, dG, dEPS, dLAM, db;
    DevBuf<int> drix, driy, dst;
    if (!dtd.alloc((size_t)nt) || !dtab.alloc(table.size()) || !dX.alloc((size_t)total) ||
        !dY.alloc((size_t)total) || !dXP.alloc((size_t)total) || !dZ.alloc((size_t)total) ||
        !dW.alloc((size_t)total) || !dWS.alloc((size_t)total) || !dTMP.alloc((size_t)total) ||
        !dPIX.alloc((size_t)total) || !dSC.alloc((size_t)total) || !dG.alloc((size_t)total) ||
        !dEPS.alloc((size_t)nt) || !dLAM.alloc((size_t)nt) || !db.alloc((size_t)std::max(1, total_samples)) ||
        !drix.alloc((size_t)std::max(1, total_samples)) || !driy.alloc((size_t)std::max(1, total_samples)) ||
        !dst.alloc((size_t)std::max(1, total_samples))) {
        std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
        return false;
    }
    {
        std::vector<float> lam((size_t)nt);
        for (int t = 0; t < nt; ++t) lam[(size_t)t] = tiles[(size_t)t].lambda;
        HIP_TRY(hipMemcpy(dLAM.p, lam.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    }
    HIP_TRY(hipMemcpy(dtd.p, td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dtab.p, table.data(), table.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dX.p, hX.data(), (size_t)total * sizeof(float), hipMemcpyHostToDevice));
    if (total_samples > 0) {
        HIP_TRY(hipMemcpy(db.p, hb.data(), (size_t)total_samples * sizeof(float), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(drix.p, hrix.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(driy.p, hriy.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dst.p, hst.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
    }

    const dim3 blk(kThreads);
    const dim3 grid((unsigned)((max_n + kThreads - 1) / kThreads), (unsigned)nt);
    const dim3 sgrid((unsigned)((total_samples + kThreads - 1) / kThreads));
    const size_t bytes = (size_t)total * sizeof(float);
    const float logw = prm.wscale > 1.0f ? std::log(prm.wscale) : 0.0f;

    k_ws<<<grid, blk>>>(dtd.p, dWS.p, logw);
    // W = 1 initially.
    {
        std::vector<float> ones((size_t)total, 1.0f);
        HIP_TRY(hipMemcpy(dW.p, ones.data(), bytes, hipMemcpyHostToDevice));
    }

    for (int r = 0; r < prm.reweights; ++r) {
        HIP_TRY(hipMemcpyAsync(dY.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        float t = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            k_synth_a<<<grid, blk>>>(dtd.p, dtab.p, dY.p, dTMP.p);
            k_synth_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p);
            HIP_TRY(hipMemsetAsync(dSC.p, 0, bytes));
            if (total_samples > 0)
                k_scatter<<<sgrid, blk>>>(dtd.p, dst.p, drix.p, driy.p, db.p, dPIX.p, dSC.p, total_samples);
            k_adj_a<<<grid, blk>>>(dtd.p, dtab.p, dSC.p, dTMP.p);
            k_adj_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dG.p);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            k_step<<<grid, blk>>>(dtd.p, dG.p, dW.p, dWS.p, dY.p, dXP.p, dZ.p, dLAM.p, mom);
            t = t_new;
        }
        HIP_TRY(hipMemcpyAsync(dX.p, dZ.p, bytes, hipMemcpyDeviceToDevice));
        if (r + 1 < prm.reweights) {
            k_absmax<<<dim3((unsigned)nt), blk>>>(dtd.p, dZ.p, dEPS.p);
            k_wupdate<<<grid, blk>>>(dtd.p, dZ.p, dEPS.p, dW.p);
        }
    }
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    HIP_TRY(hipMemcpy(hX.data(), dX.p, bytes, hipMemcpyDeviceToHost));

    for (int t = 0; t < nt; ++t) {
        const TileProblem& P = tiles[(size_t)t];
        std::copy(hX.begin() + td[(size_t)t].off, hX.begin() + td[(size_t)t].off + (size_t)P.rows * P.cols, P.x);
    }
    return true;
}

}  // namespace gpu_tiles
