// Batched GPU PnP-ADMM super-resolution stage (see gpu_admm.h).
//
// Mirrors the pnp_mode == 2 loop of cs_sr_upscale_tile_2x (superres.cpp):
//   anch = DCT(z - u); x = DCT(z)
//   x <- cs-sr FISTA core(x, anchor = anch, weight rho)       [inner steps]
//   px = IDCT(x); v = px + u; z = NLM(v) (8-bit); u += px - z
// State stays on the device for the whole loop. The NLM kernel reproduces
// OpenCV's fastNlMeansDenoising(h, 7, 15) for 8-bit planes (integer SSD,
// fixed-point weights, reflect-101 border, rounded integer division).
#include "gpu_admm.h"
#include "gpu_common.hpp"

#include <cstdint>

namespace gpu_admm {
namespace {

using namespace gpu_common;

constexpr int kTpl = 7, kSearch = 15;
constexpr int kTplHalf = kTpl / 2, kSearchHalf = kSearch / 2;
constexpr int kBorder = kTplHalf + kSearchHalf;         // 10
constexpr int kFixedMult = 37426;                       // INT_MAX / (15*15*255), as OpenCV
constexpr float kAlmostMul = 64.0f / 49.0f;             // 2^6 / template area
constexpr int kLdsBytes = 49152;                        // dynamic LDS budget of the NLM kernel

__device__ inline int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

// SC = Z - U
__global__ void k_sub(const TileDev* td, const float* A, const float* B, float* OUT) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    OUT[T.off + i] = A[T.off + i] - B[T.off + i];
}

// Data-term residual of the 2x2-average operator at the LR samples, replicated
// (x0.25) into the four HR children. SC is zeroed first.
__global__ void k_down_scatter(const TileDev* td, const int* stile, const int* rix, const int* riy,
    const float* b, const float* PIX, float* SC, int total) {
    const int s = blockIdx.x * kThreads + threadIdx.x;
    if (s >= total) return;
    const TileDev T = td[stile[s]];
    const int i = rix[s], j = riy[s];
    if (i < 0 || j < 0 || 2 * i + 1 >= T.rows || 2 * j + 1 >= T.cols) return;
    const int h0 = T.off + 2 * i * T.cols + 2 * j;
    const float avg = 0.25f * (PIX[h0] + PIX[h0 + 1] + PIX[h0 + T.cols] + PIX[h0 + T.cols + 1]);
    const float v = 0.25f * (avg - b[s]);
    SC[h0] = v; SC[h0 + 1] = v; SC[h0 + T.cols] = v; SC[h0 + T.cols + 1] = v;
}

// Soft threshold around the anchor + FISTA momentum (cs_sr_core_single, tv = 0).
__global__ void k_step(const TileDev* td, const float* G, const float* ANCH, const float* WS,
    float* Y, float* XP, float* ZC, const float* LAM, float aw, float step, float mom) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float y = Y[o], a = ANCH[o];
    const float g = 2.0f * G[o] + 2.0f * aw * (y - a);
    const float v = y - step * g - a;
    const float thr = LAM[blockIdx.y] * step * WS[o];
    const float av = fabsf(v);
    const float z = a + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
    Y[o] = z + mom * (z - XP[o]);
    XP[o] = z;
    ZC[o] = z;
}

// Padded 8-bit image for the NLM kernel: Q = quantize(PIX + U) with a
// reflect-101 border (uchar = clamp(round(v * 255))).
__global__ void k_quant(const TileDev* td, const int* qoff, const float* PIX, const float* U, std::uint8_t* Q) {
    const TileDev T = td[blockIdx.y];
    const int PW = T.cols + 2 * kBorder, PH = T.rows + 2 * kBorder;
    const int p = blockIdx.x * kThreads + threadIdx.x;
    if (p >= PW * PH) return;
    const int sy = reflect101(p / PW - kBorder, T.rows), sx = reflect101(p % PW - kBorder, T.cols);
    const int o = T.off + sy * T.cols + sx;
    int q = __float2int_rn((PIX[o] + U[o]) * 255.0f);
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    Q[qoff[blockIdx.y] + p] = (std::uint8_t)q;
}

// Padded copy of an unpadded 8-bit plane (validation entry point).
__global__ void k_pad_u8(const TileDev* td, const int* qoff, const std::uint8_t* SRC, std::uint8_t* Q) {
    const TileDev T = td[blockIdx.y];
    const int PW = T.cols + 2 * kBorder, PH = T.rows + 2 * kBorder;
    const int p = blockIdx.x * kThreads + threadIdx.x;
    if (p >= PW * PH) return;
    const int sy = reflect101(p / PW - kBorder, T.rows), sx = reflect101(p % PW - kBorder, T.cols);
    Q[qoff[blockIdx.y] + p] = SRC[T.off + sy * T.cols + sx];
}

// fastNlMeansDenoising(h, 7, 15), 8-bit. One thread per output pixel; the
// padded tile is staged in LDS. DN has the unpadded layout (offset T.off).
__global__ void k_nlm(const TileDev* td, const int* qoff, const float* HH,
    const std::uint8_t* Q, std::uint8_t* DN) {
    extern __shared__ std::uint8_t s[];
    const TileDev T = td[blockIdx.y];
    const int PW = T.cols + 2 * kBorder, PH = T.rows + 2 * kBorder;
    const std::uint8_t* q = Q + qoff[blockIdx.y];
    for (int p = threadIdx.x; p < PW * PH; p += kThreads) s[p] = q[p];
    __syncthreads();
    const int gi = blockIdx.x * kThreads + threadIdx.x;
    if (gi >= T.rows * T.cols) return;
    const int y = gi / T.cols, x = gi % T.cols;
    const float h = HH[blockIdx.y];
    const float h2 = h * h;
    int est = 0, wsum = 0;
    const int cy = y + kBorder, cx = x + kBorder;
    for (int dy = -kSearchHalf; dy <= kSearchHalf; ++dy) {
        for (int dx = -kSearchHalf; dx <= kSearchHalf; ++dx) {
            int ssd = 0;
            for (int ty = -kTplHalf; ty <= kTplHalf; ++ty) {
                const std::uint8_t* a = s + (cy + ty) * PW + (cx - kTplHalf);
                const std::uint8_t* c = s + (cy + dy + ty) * PW + (cx + dx - kTplHalf);
                #pragma unroll
                for (int tx = 0; tx < kTpl; ++tx) {
                    const int d = (int)a[tx] - (int)c[tx];
                    ssd += d * d;
                }
            }
            const float dist = (float)(ssd >> 6) * kAlmostMul;
            int w = __float2int_rn((float)kFixedMult * expf(-dist / h2));
            if (w < 38) w = 0;   // weight < 0.001 * fixed-point multiplier (37.4) is dropped
            est += w * (int)s[(cy + dy) * PW + (cx + dx)];
            wsum += w;
        }
    }
    DN[T.off + gi] = (std::uint8_t)(((unsigned)est + (unsigned)(wsum / 2)) / (unsigned)wsum);
}

// z = NLM output / 255; u += px - z.
__global__ void k_dual(const TileDev* td, const float* PIX, const std::uint8_t* DN, float* U, float* Z) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float zp = (float)DN[o] * 0.003921569f;   // (float)(1.0 / 255.0), as convertTo
    U[o] += PIX[o] - zp;
    Z[o] = zp;
}

int padded_size(int rows, int cols) { return (rows + 2 * kBorder) * (cols + 2 * kBorder); }

// Byte offsets of every tile's padded plane; returns the total.
long long layout_padded(const TileLayout& L, std::vector<int>& qoff) {
    qoff.resize(L.td.size());
    long long tot = 0;
    for (size_t t = 0; t < L.td.size(); ++t) {
        qoff[t] = (int)tot;
        tot += padded_size(L.td[t].rows, L.td[t].cols);
    }
    return tot;
}

}  // namespace

bool available() {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

bool supported(int rows, int cols) {
    return rows >= 12 && cols >= 12 && rows % 2 == 0 && cols % 2 == 0 &&
        padded_size(rows, cols) <= kLdsBytes;
}

bool nlm_denoise_u8(const std::vector<NlmPlane>& planes) {
    if (planes.empty()) return true;
    TileLayout L;
    for (const NlmPlane& P : planes) {
        if (!supported(P.rows, P.cols)) return false;
        L.add(P.rows, P.cols);
    }
    const int nt = (int)planes.size();
    std::vector<int> qoff;
    const long long qtot = layout_padded(L, qoff);
    std::vector<std::uint8_t> hsrc((size_t)L.total);
    std::vector<float> hh((size_t)nt);
    int max_pad = 0;
    for (int t = 0; t < nt; ++t) {
        std::copy(planes[(size_t)t].src, planes[(size_t)t].src + (size_t)planes[(size_t)t].rows * planes[(size_t)t].cols,
            hsrc.begin() + L.td[(size_t)t].off);
        hh[(size_t)t] = planes[(size_t)t].h;
        max_pad = std::max(max_pad, padded_size(planes[(size_t)t].rows, planes[(size_t)t].cols));
    }
    DevBuf<TileDev> dtd; DevBuf<int> dq; DevBuf<float> dh; DevBuf<std::uint8_t> dsrc, dQ, dDN;
    if (!dtd.alloc((size_t)nt) || !dq.alloc((size_t)nt) || !dh.alloc((size_t)nt) || !dsrc.alloc((size_t)L.total) ||
        !dQ.alloc((size_t)qtot) || !dDN.alloc((size_t)L.total)) return false;
    HIP_TRY(hipMemcpy(dtd.p, L.td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dq.p, qoff.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dh.p, hh.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dsrc.p, hsrc.data(), (size_t)L.total, hipMemcpyHostToDevice));
    const dim3 blk(kThreads);
    k_pad_u8<<<dim3((unsigned)((max_pad + kThreads - 1) / kThreads), (unsigned)nt), blk>>>(dtd.p, dq.p, dsrc.p, dQ.p);
    k_nlm<<<dim3((unsigned)((L.max_n + kThreads - 1) / kThreads), (unsigned)nt), blk, (size_t)max_pad>>>(
        dtd.p, dq.p, dh.p, dQ.p, dDN.p);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    std::vector<std::uint8_t> hdn((size_t)L.total);
    HIP_TRY(hipMemcpy(hdn.data(), dDN.p, (size_t)L.total, hipMemcpyDeviceToHost));
    for (int t = 0; t < nt; ++t)
        std::copy(hdn.begin() + L.td[(size_t)t].off,
            hdn.begin() + L.td[(size_t)t].off + (size_t)planes[(size_t)t].rows * planes[(size_t)t].cols, planes[(size_t)t].dst);
    return true;
}

bool solve_pnp_admm(const std::vector<Problem>& tiles, const Params& prm) {
    if (tiles.empty()) return true;
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev <= 0) return false;
    const int nt = (int)tiles.size();

    TileLayout L;
    int total_samples = 0, max_pad = 0;
    for (const Problem& P : tiles) {
        if (!supported(P.rows, P.cols)) return false;
        L.add(P.rows, P.cols);
        total_samples += P.m;
        max_pad = std::max(max_pad, padded_size(P.rows, P.cols));
    }
    if (L.total > 2000000000LL) return false;
    std::vector<int> qoff;
    const long long qtot = layout_padded(L, qoff);

    std::vector<float> hZ((size_t)L.total), hb((size_t)total_samples), hlam((size_t)nt), hh((size_t)nt);
    std::vector<int> hrix((size_t)total_samples), hriy((size_t)total_samples), hst((size_t)total_samples);
    {
        int so = 0;
        for (int t = 0; t < nt; ++t) {
            const Problem& P = tiles[(size_t)t];
            std::copy(P.z, P.z + (size_t)P.rows * P.cols, hZ.begin() + L.td[(size_t)t].off);
            for (int k = 0; k < P.m; ++k) {
                hb[(size_t)so + k] = P.b[k];
                hrix[(size_t)so + k] = P.rix[k];
                hriy[(size_t)so + k] = P.riy[k];
                hst[(size_t)so + k] = t;
            }
            so += P.m;
            hlam[(size_t)t] = P.lambda;
            hh[(size_t)t] = P.nlm_h;
        }
    }

    DevBuf<TileDev> dtd; DevBuf<int> dq, drix, driy, dst;
    DevBuf<float> dtab, dZ, dU, dANCH, dX, dY, dXP, dZC, dTMP, dPIX, dSC, dG, dWS, dLAM, dH, db;
    DevBuf<std::uint8_t> dQ, dDN;
    const size_t N = (size_t)L.total, S = (size_t)total_samples;
    if (!dtd.alloc((size_t)nt) || !dq.alloc((size_t)nt) || !dtab.alloc(L.table.size()) || !dZ.alloc(N) ||
        !dU.alloc(N) || !dANCH.alloc(N) || !dX.alloc(N) || !dY.alloc(N) || !dXP.alloc(N) || !dZC.alloc(N) ||
        !dTMP.alloc(N) || !dPIX.alloc(N) || !dSC.alloc(N) || !dG.alloc(N) || !dWS.alloc(N) ||
        !dLAM.alloc((size_t)nt) || !dH.alloc((size_t)nt) || !db.alloc(S) || !drix.alloc(S) || !driy.alloc(S) ||
        !dst.alloc(S) || !dQ.alloc((size_t)qtot) || !dDN.alloc(N)) {
        std::fprintf(stderr, "gpu_admm: device allocation failed\n");
        return false;
    }
    HIP_TRY(hipMemcpy(dtd.p, L.td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dq.p, qoff.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dtab.p, L.table.data(), L.table.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dZ.p, hZ.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dLAM.p, hlam.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dH.p, hh.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    if (total_samples > 0) {
        HIP_TRY(hipMemcpy(db.p, hb.data(), S * sizeof(float), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(drix.p, hrix.data(), S * sizeof(int), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(driy.p, hriy.data(), S * sizeof(int), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dst.p, hst.data(), S * sizeof(int), hipMemcpyHostToDevice));
    }
    HIP_TRY(hipMemset(dU.p, 0, N * sizeof(float)));

    const dim3 blk(kThreads);
    const dim3 grid((unsigned)((L.max_n + kThreads - 1) / kThreads), (unsigned)nt);
    const dim3 pgrid((unsigned)((max_pad + kThreads - 1) / kThreads), (unsigned)nt);
    const dim3 sgrid((unsigned)((total_samples + kThreads - 1) / kThreads));
    const size_t bytes = N * sizeof(float);
    const float logw = prm.wscale > 1.0f ? std::log(prm.wscale) : 0.0f;
    const float Lc = 0.25f * prm.lip + 2.0f * prm.rho;
    const float step = 1.0f / Lc;
    k_ws<<<grid, blk>>>(dtd.p, dWS.p, logw);

    for (int it = 0; it < prm.admm; ++it) {
        // anch = DCT(z - u); x = DCT(z)
        k_sub<<<grid, blk>>>(dtd.p, dZ.p, dU.p, dSC.p);
        k_adj_a<<<grid, blk>>>(dtd.p, dtab.p, dSC.p, dTMP.p);
        k_adj_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dANCH.p);
        k_adj_a<<<grid, blk>>>(dtd.p, dtab.p, dZ.p, dTMP.p);
        k_adj_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dX.p);
        HIP_TRY(hipMemcpyAsync(dY.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        float t = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            k_synth_a<<<grid, blk>>>(dtd.p, dtab.p, dY.p, dTMP.p);
            k_synth_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p);
            HIP_TRY(hipMemsetAsync(dSC.p, 0, bytes));
            if (total_samples > 0)
                k_down_scatter<<<sgrid, blk>>>(dtd.p, dst.p, drix.p, driy.p, db.p, dPIX.p, dSC.p, total_samples);
            k_adj_a<<<grid, blk>>>(dtd.p, dtab.p, dSC.p, dTMP.p);
            k_adj_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dG.p);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            k_step<<<grid, blk>>>(dtd.p, dG.p, dANCH.p, dWS.p, dY.p, dXP.p, dZC.p, dLAM.p, prm.rho, step, mom);
            t = t_new;
        }
        // px = IDCT(x); v = px + u -> 8-bit; z = NLM(v); u += px - z
        k_synth_a<<<grid, blk>>>(dtd.p, dtab.p, dZC.p, dTMP.p);
        k_synth_b<<<grid, blk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p);
        k_quant<<<pgrid, blk>>>(dtd.p, dq.p, dPIX.p, dU.p, dQ.p);
        k_nlm<<<grid, blk, (size_t)max_pad>>>(dtd.p, dq.p, dH.p, dQ.p, dDN.p);
        k_dual<<<grid, blk>>>(dtd.p, dPIX.p, dDN.p, dU.p, dZ.p);
    }
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    HIP_TRY(hipMemcpy(hZ.data(), dZ.p, bytes, hipMemcpyDeviceToHost));
    for (int t = 0; t < nt; ++t) {
        const Problem& P = tiles[(size_t)t];
        std::copy(hZ.begin() + L.td[(size_t)t].off, hZ.begin() + L.td[(size_t)t].off + (size_t)P.rows * P.cols, P.z);
    }
    return true;
}

}  // namespace gpu_admm
