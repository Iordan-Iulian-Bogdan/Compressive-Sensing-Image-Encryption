// Batched GPU tile solver (see gpu_tiles.h).
//
// Layout: every tile owns a contiguous slice [off, off + rows*cols) of each
// state array. DCT work is two small dense matrix products per transform
// (T = X * Cc^T, then C_r * T for the forward transform), which is cheap for
// tiles up to ~100 px and needs no FFT library. Kernels use blockIdx.y as the
// tile index and blockIdx.x * kThreads + threadIdx.x as the element index.
#include "gpu_tiles.h"

#include "gpu_common.hpp"
#include "gpu_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
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

// ---- tv > 0 (backtracking) branch: mirrors cs_fista_grad / cs_smooth_fx /
// cs_fista_core_single of helper_functions.cpp for the DCT basis ----

// f = sum SC^2 + tv * sum PHI per tile (SC is zero away from the samples).
__global__ void k_obj(const TileDev* td, const float* SC, const float* PHI, const float* TV, double* F) {
    const TileDev T = td[blockIdx.x];
    const int n = T.rows * T.cols;
    double d = 0.0, p = 0.0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float s = SC[T.off + i];
        d += (double)(s * s);
        p += (double)PHI[T.off + i];
    }
    const double ds = block_sum(d), ps = block_sum(p);
    if (threadIdx.x == 0) F[blockIdx.x] = (double)((float)ds + TV[blockIdx.x] * (float)ps);
}

// S = 2*SC + tv*TVG, so one forward DCT gives g = 2*DCT(sc) + tv*DCT(tvgrad).
__global__ void k_comb(const TileDev* td, const float* SC, const float* TVG, const float* TV, float* S) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    S[o] = 2.0f * SC[o] + TV[blockIdx.y] * TVG[o];
}

// Backtracking candidate z = soft(y - g/L, lambda/L * w * ws) for tiles still searching.
__global__ void k_trial(const TileDev* td, const float* Y, const float* G, const float* W, const float* WS,
    const float* LAM, const float* LT, const int* ACT, float* Z) {
    if (!ACT[blockIdx.y]) return;
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float step = 1.0f / LT[blockIdx.y];
    const float v = Y[o] - step * G[o];
    const float thr = LAM[blockIdx.y] * step * W[o] * WS[o];
    const float av = fabsf(v);
    Z[o] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
}

// Host layout + device copies of a batch's tiles, samples and warm starts.
struct Batch {
    int nt = 0, max_n = 0, max_rows = 0, max_cols = 0, total_samples = 0;
    long long total = 0;
    std::vector<TileDev> td;
    std::vector<float> table, hX;
    DevBuf<TileDev> dtd; DevBuf<float> dtab, dX, db, dLAM, dWS, dW; DevBuf<int> drix, driy, dst;

    bool build(const std::vector<TileProblem>& tiles, float wscale) {
        nt = (int)tiles.size();
        TileLayout lay;
        for (const TileProblem& P : tiles) { lay.add(P.rows, P.cols); total_samples += P.m; }
        td = lay.td; table = lay.table; total = lay.total; max_n = lay.max_n;
        max_rows = lay.max_rows; max_cols = lay.max_cols;
        if (total > 2000000000LL) return false;
        hX.resize((size_t)total);
        std::vector<float> hb((size_t)total_samples), lam((size_t)nt);
        std::vector<int> hrix((size_t)total_samples), hriy((size_t)total_samples), hst((size_t)total_samples);
        std::vector<int> so((size_t)nt + 1, 0);
        for (int t = 0; t < nt; ++t) so[(size_t)t + 1] = so[(size_t)t] + tiles[(size_t)t].m;
        parallel_for(nt, [&](int t) {
            const TileProblem& P = tiles[(size_t)t];
            std::copy(P.x, P.x + (size_t)P.rows * P.cols, hX.begin() + td[(size_t)t].off);
            const size_t o = (size_t)so[(size_t)t];
            for (int k = 0; k < P.m; ++k) {
                hb[o + k] = P.b[k];
                hrix[o + k] = P.rix[k];
                hriy[o + k] = P.riy[k];
                hst[o + k] = t;
            }
            lam[(size_t)t] = P.lambda;
        });
        const size_t ns = (size_t)std::max(1, total_samples);
        if (!dtd.alloc((size_t)nt) || !dtab.alloc(table.size()) || !dX.alloc((size_t)total) || !db.alloc(ns) ||
            !drix.alloc(ns) || !driy.alloc(ns) || !dst.alloc(ns) || !dLAM.alloc((size_t)nt) ||
            !dWS.alloc((size_t)total) || !dW.alloc((size_t)total)) {
            std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
            return false;
        }
        const size_t bytes = (size_t)total * sizeof(float);
        HIP_TRY(hipMemcpy(dtd.p, td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dtab.p, table.data(), table.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dX.p, hX.data(), bytes, hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dLAM.p, lam.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
        if (total_samples > 0) {
            HIP_TRY(hipMemcpy(db.p, hb.data(), (size_t)total_samples * sizeof(float), hipMemcpyHostToDevice));
            HIP_TRY(hipMemcpy(drix.p, hrix.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
            HIP_TRY(hipMemcpy(driy.p, hriy.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
            HIP_TRY(hipMemcpy(dst.p, hst.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
        }
        const float logw = wscale > 1.0f ? std::log(wscale) : 0.0f;
        k_ws<<<grid(), dim3(kThreads)>>>(dtd.p, dWS.p, logw);
        HIP_TRY(hipMemsetD32((hipDeviceptr_t)dW.p, 0x3F800000, (size_t)total));   // W = 1.0f initially
        return true;
    }
    dim3 grid() const { return dim3((unsigned)((max_n + kThreads - 1) / kThreads), (unsigned)nt); }
    dim3 sgrid() const { return dim3((unsigned)((total_samples + kThreads - 1) / kThreads)); }

    // Dense DCT passes on the LDS-tiled GEMM (same k-ordered sums as k_synth_* / k_adj_*).
    // PIX = IDCT(Y) and G = DCT(SC); TMP is scratch.
    void synth(const float* Y, float* TMP, float* PIX) const {
        const int nbn = (max_cols + kBN - 1) / kBN;
        const dim3 g((unsigned)(((max_rows + kBM - 1) / kBM) * nbn), (unsigned)nt);
        k_gemm<OP_DATA, OP_TAB, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, Y, TMP, nbn, Epi{});
        k_gemm<OP_TABT, OP_DATA, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, TMP, PIX, nbn, Epi{});
    }
    void adj(const float* SC, float* TMP, float* G) const {
        const int nbn = (max_cols + kBN - 1) / kBN;
        const dim3 g((unsigned)(((max_rows + kBM - 1) / kBM) * nbn), (unsigned)nt);
        k_gemm<OP_DATA, OP_TABT, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, SC, TMP, nbn, Epi{});
        k_gemm<OP_TAB, OP_DATA, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, TMP, G, nbn, Epi{});
    }

    // reweighting: W = eps / (|Z| + eps), eps from the tile's max |Z|
    void reweight(const float* dZ, float* dEPS) {
        k_absmax<<<dim3((unsigned)nt), dim3(kThreads)>>>(dtd.p, dZ, dEPS);
        k_wupdate<<<grid(), dim3(kThreads)>>>(dtd.p, dZ, dEPS, dW.p);
    }
    bool download(const std::vector<TileProblem>& tiles) {
        HIP_TRY(hipGetLastError());
        HIP_TRY(hipDeviceSynchronize());
        HIP_TRY(hipMemcpy(hX.data(), dX.p, (size_t)total * sizeof(float), hipMemcpyDeviceToHost));
        parallel_for(nt, [&](int t) {
            const TileProblem& P = tiles[(size_t)t];
            std::copy(hX.begin() + td[(size_t)t].off, hX.begin() + td[(size_t)t].off + (size_t)P.rows * P.cols, P.x);
        });
        return true;
    }
};

// tv = 0: exact fixed step 1/2 (previous behaviour, unchanged).
bool solve_fixed(const std::vector<TileProblem>& tiles, const Params& prm) {
    Batch B;
    if (!B.build(tiles, prm.wscale)) return false;
    const size_t N = (size_t)B.total, bytes = N * sizeof(float);
    DevBuf<float> dY, dXP, dZ, dTMP, dPIX, dSC, dG, dEPS;
    if (!dY.alloc(N) || !dXP.alloc(N) || !dZ.alloc(N) || !dTMP.alloc(N) || !dPIX.alloc(N) ||
        !dSC.alloc(N) || !dG.alloc(N) || !dEPS.alloc((size_t)B.nt)) {
        std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
        return false;
    }
    const dim3 blk(kThreads), grid = B.grid(), sgrid = B.sgrid();
    for (int r = 0; r < prm.reweights; ++r) {
        HIP_TRY(hipMemcpyAsync(dY.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        float t = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            B.synth(dY.p, dTMP.p, dPIX.p);
            HIP_TRY(hipMemsetAsync(dSC.p, 0, bytes));
            if (B.total_samples > 0)
                k_scatter<<<sgrid, blk>>>(B.dtd.p, B.dst.p, B.drix.p, B.driy.p, B.db.p, dPIX.p, dSC.p, B.total_samples);
            B.adj(dSC.p, dTMP.p, dG.p);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            k_step<<<grid, blk>>>(B.dtd.p, dG.p, B.dW.p, B.dWS.p, dY.p, dXP.p, dZ.p, B.dLAM.p, mom);
            t = t_new;
        }
        HIP_TRY(hipMemcpyAsync(B.dX.p, dZ.p, bytes, hipMemcpyDeviceToDevice));
        if (r + 1 < prm.reweights) B.reweight(dZ.p, dEPS.p);
    }
    return B.download(tiles);
}

// tv > 0: smoothed-TV term + per-tile backtracking, as cs_fista_core_single (bt branch).
bool solve_backtrack(const std::vector<TileProblem>& tiles, const Params& prm) {
    Batch B;
    if (!B.build(tiles, prm.wscale)) return false;
    const int nt = B.nt;
    const size_t N = (size_t)B.total, bytes = N * sizeof(float);
    DevBuf<float> dY, dXP, dZ, dTMP, dPIX, dSC, dG, dEPS, dTVG, dPHI, dS, dTV, dLT;
    DevBuf<double> dF, dD2; DevBuf<int> dACT;
    if (!dY.alloc(N) || !dXP.alloc(N) || !dZ.alloc(N) || !dTMP.alloc(N) || !dPIX.alloc(N) || !dSC.alloc(N) ||
        !dG.alloc(N) || !dEPS.alloc((size_t)nt) || !dTVG.alloc(N) || !dPHI.alloc(N) || !dS.alloc(N) ||
        !dTV.alloc((size_t)nt) || !dLT.alloc((size_t)nt) || !dF.alloc((size_t)nt) || !dD2.alloc(2 * (size_t)nt) ||
        !dACT.alloc((size_t)nt)) {
        std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
        return false;
    }
    std::vector<float> tv((size_t)nt), L((size_t)nt);
    for (int t = 0; t < nt; ++t) tv[(size_t)t] = tiles[(size_t)t].tv;
    HIP_TRY(hipMemcpy(dTV.p, tv.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    std::vector<double> fy((size_t)nt), fz((size_t)nt), d2(2 * (size_t)nt);
    std::vector<int> act((size_t)nt);
    const dim3 blk(kThreads), grid = B.grid(), sgrid = B.sgrid(), tgrid((unsigned)nt);
    // smooth objective of the pixels in PIX: residual into SC, TV gradient into TVG, f into F
    auto smooth = [&](const float* coef) {
        B.synth(coef, dTMP.p, dPIX.p);
        (void)hipMemsetAsync(dSC.p, 0, bytes);
        if (B.total_samples > 0)
            k_scatter<<<sgrid, blk>>>(B.dtd.p, B.dst.p, B.drix.p, B.driy.p, B.db.p, dPIX.p, dSC.p, B.total_samples);
        k_tv<<<grid, blk>>>(B.dtd.p, dPIX.p, dTVG.p, dPHI.p);
        k_obj<<<tgrid, blk>>>(B.dtd.p, dSC.p, dPHI.p, dTV.p, dF.p);
    };
    for (int r = 0; r < prm.reweights; ++r) {
        HIP_TRY(hipMemcpyAsync(dY.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        for (int t = 0; t < nt; ++t) L[(size_t)t] = 2.0f + 8.0f * tv[(size_t)t];   // bx.lip (DCT) + 8*tv, per core call
        float tm = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            // gradient at y: g = 2*DCT(sc) + tv*DCT(tvgrad); fy = ||res||^2 + tv*phi
            smooth(dY.p);
            k_comb<<<grid, blk>>>(B.dtd.p, dSC.p, dTVG.p, dTV.p, dS.p);
            B.adj(dS.p, dTMP.p, dG.p);
            HIP_TRY(hipMemcpy(fy.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
            std::fill(act.begin(), act.end(), 1);
            for (int trial = 0; trial < 25; ++trial) {
                HIP_TRY(hipMemcpy(dLT.p, L.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
                HIP_TRY(hipMemcpy(dACT.p, act.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
                k_trial<<<grid, blk>>>(B.dtd.p, dY.p, dG.p, B.dW.p, B.dWS.p, B.dLAM.p, dLT.p, dACT.p, dZ.p);
                smooth(dZ.p);
                k_dots<<<tgrid, blk>>>(B.dtd.p, dG.p, dZ.p, dY.p, dD2.p);
                HIP_TRY(hipMemcpy(fz.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                HIP_TRY(hipMemcpy(d2.data(), dD2.p, 2 * (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                const int left = bt_accept(nt, fy, fz, d2, L, act, trial);
                if (left == 0) break;
            }
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * tm * tm));
            k_mom<<<grid, blk>>>(B.dtd.p, dZ.p, dY.p, dXP.p, (tm - 1.0f) / t_new);
            tm = t_new;
        }
        HIP_TRY(hipMemcpyAsync(B.dX.p, dZ.p, bytes, hipMemcpyDeviceToDevice));
        if (r + 1 < prm.reweights) B.reweight(dZ.p, dEPS.p);
    }
    return B.download(tiles);
}

}  // namespace

}  // namespace gpu_tiles

namespace gpu_common {
namespace {
struct Arena {
    std::mutex mu;
    char* base = nullptr;
    size_t cap = 0, top = 0;
    struct Piece { size_t off; bool live; };
    std::vector<Piece> pieces;   // in address order
} g_arena;
}  // namespace

void arena_reserve(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    if (g_arena.base || bytes == 0) return;
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) return;
    g_arena.base = (char*)p;
    g_arena.cap = bytes;
}

void* arena_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    const size_t n = (bytes + 255) & ~(size_t)255;
    if (!g_arena.base || g_arena.cap - g_arena.top < n) return nullptr;
    g_arena.pieces.push_back({ g_arena.top, true });
    void* p = g_arena.base + g_arena.top;
    g_arena.top += n;
    return p;
}

bool arena_free(void* p) {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    char* c = (char*)p;
    if (!g_arena.base || c < g_arena.base || c >= g_arena.base + g_arena.cap) return false;
    const size_t off = (size_t)(c - g_arena.base);
    for (auto it = g_arena.pieces.rbegin(); it != g_arena.pieces.rend(); ++it)
        if (it->off == off) { it->live = false; break; }
    while (!g_arena.pieces.empty() && !g_arena.pieces.back().live) {
        g_arena.top = g_arena.pieces.back().off;
        g_arena.pieces.pop_back();
    }
    return true;
}
}  // namespace gpu_common

namespace gpu_tiles {

bool available() {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

void warmup() {
    if (!available()) return;
    // The device arena: half the free memory, at most 2 GiB (reserving it must finish while the CPU
    // still unpacks; a larger decode takes its overflow from hipMalloc).
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return;
    gpu_common::arena_reserve(std::min<size_t>(free_b / 2, (size_t)2 << 30));
}

bool solve_fista_dct(const std::vector<TileProblem>& tiles, const Params& prm) {
    if (tiles.empty()) return true;
    if (!available()) return false;
    std::vector<TileProblem> fixed, bt;
    for (const TileProblem& P : tiles) (P.tv > 0.0f ? bt : fixed).push_back(P);
    if (!fixed.empty() && !solve_fixed(fixed, prm)) return false;
    if (!bt.empty() && !solve_backtrack(bt, prm)) return false;
    return true;
}

}  // namespace gpu_tiles
