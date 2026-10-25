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
#include "gpu_gemm.hpp"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>

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
__global__ void k_nlm(const TileDev* td, const int* qoff, const float* HH, float hmul,
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
    const float h = HH[blockIdx.y] * hmul;
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

// u *= f (dual rescaling when rho changes).
__global__ void k_scale(const TileDev* td, float* U, float f) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    U[T.off + i] *= f;
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

// ---- tv > 0 (backtracking) inner loop: mirrors cs_sr_fista_grad / cs_sr_smooth_fx /
// cs_sr_core_single (bt branch) of superres.cpp ----

// Data residual of the 2x2 average at the samples (0.25*(avg - b) on each child)
// plus the exact-HR anchor term hw*(pix - v), one thread per HR pixel.
__global__ void k_resid_hr(const TileDev* td, const float* PIX, const float* BLR, const float* HAV, float hw, float* SC) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int N = T.cols, m = i / N, n = i % N, m0 = m & ~1, n0 = n & ~1;
    const float* p = PIX + T.off;
    const float bs = BLR[T.off / 4 + (m / 2) * (N / 2) + n / 2];
    float v = 0.0f;
    if (bs > -1e29f && m0 + 1 < T.rows && n0 + 1 < N)
        v = 0.25f * (0.25f * (p[m0 * N + n0] + p[m0 * N + n0 + 1] + p[(m0 + 1) * N + n0] + p[(m0 + 1) * N + n0 + 1]) - bs);
    if (HAV) { const float h = HAV[T.off + i]; if (h > -1e29f) v += hw * (p[i] - h); }
    SC[T.off + i] = v;
}

// Smooth objective per tile (one block per tile): ||avg - b||^2 over the samples
// + hw*||pix - v||^2 over the anchors + tv*sum phi + aw*||y - anch||^2.
__global__ void k_obj_hr(const TileDev* td, const float* PIX, const float* BLR, const float* HAV, float hw,
    const float* PHI, const float* TV, const float* Y, const float* ANCH, float aw, double* F) {
    const TileDev T = td[blockIdx.x];
    const int N = T.cols, n = T.rows * N, nl = (T.rows / 2) * (N / 2);
    const float* p = PIX + T.off;
    double dat = 0.0, hr = 0.0, phi = 0.0, an = 0.0;
    for (int c = threadIdx.x; c < nl; c += kThreads) {
        const float bs = BLR[T.off / 4 + c];
        if (bs <= -1e29f) continue;
        const int m0 = (c / (N / 2)) * 2, n0 = (c % (N / 2)) * 2;
        const float d = 0.25f * (p[m0 * N + n0] + p[m0 * N + n0 + 1] + p[(m0 + 1) * N + n0] + p[(m0 + 1) * N + n0 + 1]) - bs;
        dat += (double)(d * d);
    }
    for (int i = threadIdx.x; i < n; i += kThreads) {
        if (HAV) { const float h = HAV[T.off + i]; if (h > -1e29f) { const float d = p[i] - h; hr += (double)(d * d); } }
        phi += (double)PHI[T.off + i];
        const float d = Y[T.off + i] - ANCH[T.off + i];
        an += (double)(d * d);
    }
    const double a = block_sum(dat), b = block_sum(hr), c2 = block_sum(phi), e = block_sum(an);
    if (threadIdx.x == 0) F[blockIdx.x] = (double)((float)a + hw * (float)b + TV[blockIdx.x] * (float)c2 + aw * (float)e);
}

// S = 2*SC + tv*TVG (one forward DCT then gives the data + anchor + TV gradient).
__global__ void k_comb_hr(const TileDev* td, const float* SC, const float* TVG, const float* TV, float* S) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    S[T.off + i] = 2.0f * SC[T.off + i] + TV[blockIdx.y] * TVG[T.off + i];
}

// G += 2*aw*(y - anch): the ADMM coupling term (coefficient domain).
__global__ void k_gcouple(const TileDev* td, const float* Y, const float* ANCH, float aw, float* G) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    G[o] += 2.0f * aw * (Y[o] - ANCH[o]);
}

// Candidate z = anch + soft(y - g/L - anch, lambda/L * ws) for tiles still searching.
__global__ void k_trial_hr(const TileDev* td, const float* Y, const float* G, const float* WS, const float* LAM,
    const float* LT, const int* ACT, const float* ANCH, float* Z) {
    if (!ACT[blockIdx.y]) return;
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float step = 1.0f / LT[blockIdx.y];
    const float v = Y[o] - step * G[o] - ANCH[o];
    const float thr = LAM[blockIdx.y] * step * WS[o];
    const float av = fabsf(v);
    Z[o] = ANCH[o] + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
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
        dtd.p, dq.p, dh.p, 1.0f, dQ.p, dDN.p);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    std::vector<std::uint8_t> hdn((size_t)L.total);
    HIP_TRY(hipMemcpy(hdn.data(), dDN.p, (size_t)L.total, hipMemcpyDeviceToHost));
    for (int t = 0; t < nt; ++t)
        std::copy(hdn.begin() + L.td[(size_t)t].off,
            hdn.begin() + L.td[(size_t)t].off + (size_t)planes[(size_t)t].rows * planes[(size_t)t].cols, planes[(size_t)t].dst);
    return true;
}

// Opt-in per-kernel timing (env CS_GPU_PROF=1): hipEvent pairs around every
// launch of solve_pnp_admm, accumulated per kernel name and printed at exit.
struct Prof {
    bool on = false;
    std::vector<std::pair<const char*, std::pair<hipEvent_t, hipEvent_t>>> ev;
    hipEvent_t t_begin{}, t_end{};
    Prof() { const char* e = std::getenv("CS_GPU_PROF"); on = e && e[0] == '1'; }
    void begin_all() { if (!on) return; hipEventCreate(&t_begin); hipEventRecord(t_begin); }
    void b(const char* n) {
        if (!on) return;
        hipEvent_t a, c; hipEventCreate(&a); hipEventCreate(&c);
        hipEventRecord(a);
        ev.push_back({ n, { a, c } });
    }
    void e() { if (on) hipEventRecord(ev.back().second.second); }
    void end_all(long long tiles, long long px);
};
struct ProfAgg {
    std::mutex mu;
    std::map<std::string, std::pair<double, long long>> k;   // name -> (ms, launches)
    double span_ms = 0.0; long long calls = 0, tiles = 0, px = 0;
    ~ProfAgg() {
        if (calls == 0) return;
        double sum = 0; for (auto& kv : k) sum += kv.second.first;
        std::fprintf(stderr, "[gpu-prof] gpu_admm: %lld calls, %lld tiles, %.1f Mpx/iter-plane; kernel sum %.1f ms, wall span %.1f ms (idle gaps %.1f ms)\n",
            calls, tiles, px * 1e-6, sum, span_ms, span_ms - sum);
        std::vector<std::pair<std::string, std::pair<double, long long>>> v(k.begin(), k.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
        for (auto& kv : v)
            std::fprintf(stderr, "[gpu-prof]   %-16s %9.1f ms  %5.1f%%  %7lld launches  %8.1f us/launch\n",
                kv.first.c_str(), kv.second.first, 100.0 * kv.second.first / (sum > 0 ? sum : 1), kv.second.second,
                1000.0 * kv.second.first / (double)kv.second.second);
    }
};
ProfAgg g_prof_agg;
void Prof::end_all(long long tiles, long long px) {
    if (!on) return;
    hipEventCreate(&t_end); hipEventRecord(t_end); hipEventSynchronize(t_end);
    float span = 0; hipEventElapsedTime(&span, t_begin, t_end);
    std::lock_guard<std::mutex> lk(g_prof_agg.mu);
    for (auto& e : ev) {
        float ms = 0; hipEventElapsedTime(&ms, e.second.first, e.second.second);
        auto& a = g_prof_agg.k[e.first]; a.first += ms; a.second += 1;
        hipEventDestroy(e.second.first); hipEventDestroy(e.second.second);
    }
    g_prof_agg.span_ms += span; ++g_prof_agg.calls; g_prof_agg.tiles += tiles; g_prof_agg.px += px;
    ev.clear();
}
#define TIMED(name, ...) do { prof.b(name); __VA_ARGS__; prof.e(); } while (0)

// The whole ADMM loop. dZext: device z in TileLayout order (add(rows, cols) per tile, in order),
// updated in place; null = take z from / return it to the host Problem::z arrays.
static bool run_admm(const std::vector<Problem>& tiles, const Params& prm, float* dZext) {
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

    // Samples live on a per-tile LR grid (offset off/4, rows/2 x cols/2): the
    // sample value, or a large negative sentinel where there is none.
    std::vector<float> hZ(dZext ? 0 : (size_t)L.total), hblr((size_t)L.total / 4, -1e30f), hlam((size_t)nt), hh((size_t)nt);
    {
        for (int t = 0; t < nt; ++t) {
            const Problem& P = tiles[(size_t)t];
            if (!dZext) std::copy(P.z, P.z + (size_t)P.rows * P.cols, hZ.begin() + L.td[(size_t)t].off);
            float* blr = hblr.data() + L.td[(size_t)t].off / 4;
            for (int k = 0; k < P.m; ++k) {
                if (P.rix[k] < 0 || P.riy[k] < 0 || P.rix[k] >= P.rows / 2 || P.riy[k] >= P.cols / 2) continue;
                blr[P.rix[k] * (P.cols / 2) + P.riy[k]] = P.b[k];
            }
            hlam[(size_t)t] = P.lambda;
            hh[(size_t)t] = P.nlm_h;
        }
    }

    // Exact-HR anchors (tile layout, sentinel where none), only when the batch carries a weight.
    const bool use_hr = prm.hr_weight > 0.0f;
    std::vector<float> hhav;
    if (use_hr) {
        hhav.assign((size_t)L.total, -1e30f);
        for (int t = 0; t < nt; ++t) {
            const Problem& P = tiles[(size_t)t];
            if (P.hr_val) std::copy(P.hr_val, P.hr_val + (size_t)P.rows * P.cols, hhav.begin() + L.td[(size_t)t].off);
        }
    }

    DevBuf<TileDev> dtd; DevBuf<int> dq;
    DevBuf<float> dtab, dZ, dU, dANCH, dX, dY, dXP, dZC, dTMP, dPIX, dSC, dWS, dLAM, dH, dBLR, dHAV;
    DevBuf<std::uint8_t> dQ, dDN;
    const size_t N = (size_t)L.total;
    if (!dtd.alloc((size_t)nt) || !dq.alloc((size_t)nt) || !dtab.alloc(L.table.size()) || (!dZext && !dZ.alloc(N)) ||
        !dU.alloc(N) || !dANCH.alloc(N) || !dX.alloc(N) || !dY.alloc(N) || !dXP.alloc(N) || !dZC.alloc(N) ||
        !dTMP.alloc(N) || !dPIX.alloc(N) || !dSC.alloc(N) || !dWS.alloc(N) ||
        !dLAM.alloc((size_t)nt) || !dH.alloc((size_t)nt) || !dBLR.alloc(N / 4) ||
        !dQ.alloc((size_t)qtot) || !dDN.alloc(N) || (use_hr && !dHAV.alloc(N))) {
        std::fprintf(stderr, "gpu_admm: device allocation failed\n");
        return false;
    }
    float* const Zp = dZext ? dZext : dZ.p;   // z lives here for the whole loop
    if (use_hr) HIP_TRY(hipMemcpy(dHAV.p, hhav.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // tv > 0 batches: explicit gradient, smooth objective and per-tile backtracking state.
    const bool bt = prm.bt != 0;
    DevBuf<float> dTVA, dTVG, dPHI, dS, dG, dLT; DevBuf<double> dF, dD2; DevBuf<int> dACT;
    std::vector<float> tva((size_t)nt), Lt((size_t)nt);
    std::vector<double> fy((size_t)nt), fz((size_t)nt), d2(2 * (size_t)nt);
    std::vector<int> act((size_t)nt);
    if (bt) {
        for (int t = 0; t < nt; ++t) tva[(size_t)t] = tiles[(size_t)t].tv;
        if (!dTVA.alloc((size_t)nt) || !dTVG.alloc(N) || !dPHI.alloc(N) || !dS.alloc(N) || !dG.alloc(N) ||
            !dLT.alloc((size_t)nt) || !dF.alloc((size_t)nt) || !dD2.alloc(2 * (size_t)nt) || !dACT.alloc((size_t)nt)) {
            std::fprintf(stderr, "gpu_admm: device allocation failed\n");
            return false;
        }
        HIP_TRY(hipMemcpy(dTVA.p, tva.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    }
    HIP_TRY(hipMemcpy(dtd.p, L.td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dq.p, qoff.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dtab.p, L.table.data(), L.table.size() * sizeof(float), hipMemcpyHostToDevice));
    if (!dZext) HIP_TRY(hipMemcpy(dZ.p, hZ.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dLAM.p, hlam.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dH.p, hh.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dBLR.p, hblr.data(), (N / 4) * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemset(dU.p, 0, N * sizeof(float)));

    const dim3 blk(kThreads);
    const dim3 grid((unsigned)((L.max_n + kThreads - 1) / kThreads), (unsigned)nt);
    const dim3 pgrid((unsigned)((max_pad + kThreads - 1) / kThreads), (unsigned)nt);
    const int nbn = (L.max_cols + kBN - 1) / kBN;
    const dim3 ggrid((unsigned)(((L.max_rows + kBM - 1) / kBM) * nbn), (unsigned)nt);
    const dim3 gblk(kGemmThreads);
    const Epi none{};
    const size_t bytes = N * sizeof(float);
    const float logw = prm.wscale > 1.0f ? std::log(prm.wscale) : 0.0f;
    const float rho_end = prm.rho_end > 0.0f ? prm.rho_end : prm.rho;
    float rho_prev = prm.rho;
    Prof prof;
    prof.begin_all();
#define FWD_DCT(src, dst) do {         TIMED("gemm_fwd_a", k_gemm<OP_DATA, OP_TABT, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, (src), dTMP.p, nbn, none));         TIMED("gemm_fwd_b", k_gemm<OP_TAB, OP_DATA, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dTMP.p, (dst), nbn, none));     } while (0)
    k_ws<<<grid, blk>>>(dtd.p, dWS.p, logw);

    for (int it = 0; it < prm.admm; ++it) {
        // per-round schedule of rho and of the NLM strength
        const float frac = prm.admm > 1 ? (float)it / (float)(prm.admm - 1) : 0.0f;
        const float rho_k = prm.rho == rho_end ? prm.rho : prm.rho * std::pow(rho_end / prm.rho, frac);
        const float hmul_k = prm.h_start == prm.h_end ? prm.h_start : prm.h_start * std::pow(prm.h_end / prm.h_start, frac);
        const float Lc = 0.25f * prm.lip + 2.0f * rho_k + (use_hr ? 2.0f * prm.hr_weight : 0.0f);   // fixed-step branch
        const float step = 1.0f / Lc;
        if (prm.rescale_u && rho_k != rho_prev) TIMED("k_scale", k_scale<<<grid, blk>>>(dtd.p, dU.p, rho_prev / rho_k));
        rho_prev = rho_k;
        // anch = DCT(z - u); x = DCT(z)
        TIMED("k_sub", k_sub<<<grid, blk>>>(dtd.p, Zp, dU.p, dSC.p));
        FWD_DCT(dSC.p, dANCH.p);
        FWD_DCT(Zp, dX.p);
        HIP_TRY(hipMemcpyAsync(dY.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, dX.p, bytes, hipMemcpyDeviceToDevice));
        float t = 1.0f;
        if (bt) {
            // smooth objective of coefficients C (PIX = IDCT(C), SC/TVG/PHI refreshed), into dF
            auto smooth = [&](const float* C) {
                k_gemm<OP_DATA, OP_TAB, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, C, dTMP.p, nbn, none);
                k_gemm<OP_TABT, OP_DATA, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p, nbn, none);
                k_resid_hr<<<grid, blk>>>(dtd.p, dPIX.p, dBLR.p, use_hr ? dHAV.p : nullptr, prm.hr_weight, dSC.p);
                k_tv<<<grid, blk>>>(dtd.p, dPIX.p, dTVG.p, dPHI.p);
                k_obj_hr<<<dim3((unsigned)nt), blk>>>(dtd.p, dPIX.p, dBLR.p, use_hr ? dHAV.p : nullptr, prm.hr_weight,
                    dPHI.p, dTVA.p, C, dANCH.p, rho_k, dF.p);
            };
            for (int q = 0; q < nt; ++q)   // per core call, as the CPU: 0.25*lip + 2*rho + 8*tv + 2*hw
                Lt[(size_t)q] = 0.25f * prm.lip + 2.0f * rho_k + 8.0f * tva[(size_t)q] + (use_hr ? 2.0f * prm.hr_weight : 0.0f);
            for (int k = 0; k < prm.inner; ++k) {
                smooth(dY.p);
                k_comb_hr<<<grid, blk>>>(dtd.p, dSC.p, dTVG.p, dTVA.p, dS.p);
                FWD_DCT(dS.p, dG.p);
                k_gcouple<<<grid, blk>>>(dtd.p, dY.p, dANCH.p, rho_k, dG.p);
                HIP_TRY(hipMemcpy(fy.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                std::fill(act.begin(), act.end(), 1);
                for (int trial = 0; trial < 25; ++trial) {
                    HIP_TRY(hipMemcpy(dLT.p, Lt.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
                    HIP_TRY(hipMemcpy(dACT.p, act.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
                    k_trial_hr<<<grid, blk>>>(dtd.p, dY.p, dG.p, dWS.p, dLAM.p, dLT.p, dACT.p, dANCH.p, dZC.p);
                    smooth(dZC.p);
                    k_dots<<<dim3((unsigned)nt), blk>>>(dtd.p, dG.p, dZC.p, dY.p, dD2.p);
                    HIP_TRY(hipMemcpy(fz.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                    HIP_TRY(hipMemcpy(d2.data(), dD2.p, 2 * (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                    if (bt_accept(nt, fy, fz, d2, Lt, act, trial) == 0) break;
                }
                const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
                k_mom<<<grid, blk>>>(dtd.p, dZC.p, dY.p, dXP.p, (t - 1.0f) / t_new);
                t = t_new;
            }
        }
        for (int k = 0; !bt && k < prm.inner; ++k) {
            // PIX = IDCT(y); SC = residual of the 2x2 average at the samples (fused epilogue)
            TIMED("gemm_inv_a", k_gemm<OP_DATA, OP_TAB, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dY.p, dTMP.p, nbn, none));
            Epi er{}; er.blr = dBLR.p;
            if (use_hr) { er.hav = dHAV.p; er.hw = prm.hr_weight; }
            TIMED("gemm_inv_b+resid", k_gemm<OP_TABT, OP_DATA, EPI_RESID><<<ggrid, gblk>>>(dtd.p, dtab.p, dTMP.p, dSC.p, nbn, er));
            // gradient = 2 DCT(SC); soft threshold + momentum (fused epilogue)
            TIMED("gemm_fwd_a", k_gemm<OP_DATA, OP_TABT, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dSC.p, dTMP.p, nbn, none));
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            Epi es{}; es.anch = dANCH.p; es.ws = dWS.p; es.lam = dLAM.p; es.y = dY.p; es.xp = dXP.p; es.zc = dZC.p;
            es.aw = rho_k; es.step = step; es.mom = (t - 1.0f) / t_new;
            TIMED("gemm_fwd_b+step", k_gemm<OP_TAB, OP_DATA, EPI_STEP><<<ggrid, gblk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p, nbn, es));
            t = t_new;
        }
        // px = IDCT(x); v = px + u -> 8-bit; z = NLM(v); u += px - z
        TIMED("gemm_inv_a", k_gemm<OP_DATA, OP_TAB, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dZC.p, dTMP.p, nbn, none));
        TIMED("gemm_inv_b", k_gemm<OP_TABT, OP_DATA, EPI_STORE><<<ggrid, gblk>>>(dtd.p, dtab.p, dTMP.p, dPIX.p, nbn, none));
        TIMED("k_quant", k_quant<<<pgrid, blk>>>(dtd.p, dq.p, dPIX.p, dU.p, dQ.p));
        TIMED("k_nlm", k_nlm<<<grid, blk, (size_t)max_pad>>>(dtd.p, dq.p, dH.p, hmul_k, dQ.p, dDN.p));
        TIMED("k_dual", k_dual<<<grid, blk>>>(dtd.p, dPIX.p, dDN.p, dU.p, Zp));
    }
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    prof.end_all(nt, L.total);
    if (dZext) return true;
    HIP_TRY(hipMemcpy(hZ.data(), dZ.p, bytes, hipMemcpyDeviceToHost));
    for (int t = 0; t < nt; ++t) {
        const Problem& P = tiles[(size_t)t];
        std::copy(hZ.begin() + L.td[(size_t)t].off, hZ.begin() + L.td[(size_t)t].off + (size_t)P.rows * P.cols, P.z);
    }
    return true;
}

bool solve_pnp_admm(const std::vector<Problem>& tiles, const Params& prm) {
    return run_admm(tiles, prm, nullptr);
}

bool solve_pnp_admm_device(const std::vector<Problem>& tiles, const Params& prm, float* dZ) {
    return dZ != nullptr && run_admm(tiles, prm, dZ);
}

}  // namespace gpu_admm
