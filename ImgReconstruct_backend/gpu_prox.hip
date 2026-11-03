// Learned ADMM prox (see gpu_prox.h) on the WMMA units, after gpu_enh.hip's convs: every 3x3 conv is an implicit GEMM
// over an fp16 NHWC halo in LDS (v_wmma_f32_16x16x16_f16, fp32 accumulation), one launch per layer, a block of 4
// waves computing an 8 x 8 output block of one tile for every output channel; zero padding at the tile border. Channels
// are zero padded to 16 / 32 (inputs 2 or 6 -> 16, hidden widths up to 32). Bias and ReLU are fused; the last layer adds
// q (read back from the 8-bit input, so the fp16 input plane cannot move the rounding) and stores the 8-bit output.
//
// Tiles run in chunks of at most kChunkPx pixels through one set of scratch buffers (allocated once), so the memory
// does not grow with the image: (16 + 2 * 32) halves per chunk pixel.
//
// Weight file (admm_prox_export.py): "PRX1", int32 layers, then per layer int32 cin, cout, float w[cout][cin][3][3],
// float b[cout].
#include "gpu_prox.h"
#include "gpu_common.hpp"
#include "gpu_wmma.hpp"

#include <cstdio>
#include <mutex>
#include <vector>

namespace gpu_prox {
namespace {

using namespace gpu_common;
using namespace gpu_wmma;

constexpr int kT = 128;                                 // four waves of 32; a wave owns 16 pixels of an 8 x 8 block
constexpr int kB = 8;                                   // output block side
constexpr int kXC = 16;                                 // input planes, zero padded
constexpr int kHC = 32;                                 // widest hidden layer (padded)
constexpr long long kChunkPx = 1 << 20;                 // tile pixels per chunk (scratch: 160 MB)
constexpr float kResScale = 255.0f / 64.0f, kSresScale = 255.0f / 8.0f;   // enh_inputs.RES_SCALE / SRES_SCALE

struct Layer { int cin, cout, cin_p, nt; size_t w; size_t b; };   // offsets into the fragment / bias buffers
std::vector<Layer> g_net;
half_t* g_w = nullptr;                                  // packed B fragments, every layer
float* g_b = nullptr;                                   // biases, padded to nt * 16 per layer
half_t* g_scr = nullptr;                                // X (kXC) + two hidden buffers (kHC) per chunk pixel
std::string g_path;
std::mutex g_mu;

__device__ inline h16 ldw(const half_t* W, int KT, int nt, int kt, int r16) { return ld16(W + ((size_t)(nt * KT + kt) * 16 + r16) * 16); }
__device__ inline float clip1(float v) { return fminf(1.0f, fmaxf(-1.0f, v)); }

// Layer-0 input planes (gpu_prox.h) into X ([chunk pixel][kXC] fp16; channels past cin are 0).
__global__ void k_feat(const TileDev* td, int o0, const int* qoff, int border, const std::uint8_t* Q, const float* BLR,
                       const float* HAV, const float* H, float hmul, int cin, half_t* X) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int y = i / T.cols, x = i % T.cols, PW = T.cols + 2 * border;
    const std::uint8_t* q8 = Q + qoff[blockIdx.y] + border * PW + border;
    const float q = q8[y * PW + x] * (1.0f / 255.0f);
    float o[kXC] = {};
    o[0] = q - 0.5f;
    o[1] = H[blockIdx.y] * hmul * (1.0f / 16.0f);
    if (cin >= 6) {
        const float a = HAV ? HAV[T.off + i] : -1e30f;
        if (a > -1e29f) { o[2] = 1.0f; o[3] = clip1((a - q) * kResScale); }
        const int ly = y >> 1, lx = x >> 1;
        const float b = (ly < T.rows / 2 && lx < T.cols / 2) ? BLR[T.off / 4 + ly * (T.cols / 2) + lx] : -1e30f;
        if (b > -1e29f) {
            const int y0 = ly * 2, x0 = lx * 2;
            const float box = (q8[y0 * PW + x0] + q8[y0 * PW + x0 + 1] + q8[(y0 + 1) * PW + x0] + q8[(y0 + 1) * PW + x0 + 1]) *
                              (0.25f / 255.0f);
            o[4] = 1.0f; o[5] = clip1((b - box) * kSresScale);
        }
    }
    h8 v0, v1;
#pragma unroll
    for (int k = 0; k < 8; ++k) { v0[k] = (half_t)o[k]; v1[k] = (half_t)o[8 + k]; }
    half_t* d = X + (size_t)(T.off - o0 + i) * kXC;
    *(h8*)d = v0; *(h8*)(d + 8) = v1;
}

struct ConvP {
    const TileDev* td; int o0;                          // the chunk's tiles; o0 = first tile's offset
    const half_t* IN; half_t* OUT;                      // [chunk pixel][CIN] -> [chunk pixel][NT * 16]
    const half_t* Wt; const float* bias;
    const int* qoff; int border; const std::uint8_t* Q; std::uint8_t* DN;   // last layer only
};

// One 3x3 conv on an 8 x 8 output block of tile blockIdx.y: CIN input channels, NT x 16 outputs. LAST: z = q + out[0]
// as 8-bit DN; otherwise ReLU into OUT.
template <int CIN, int NT, bool LAST>
__global__ void __launch_bounds__(kT) k_conv(const ConvP P) {
    constexpr int HS = kB + 2, RS = CIN + 8;            // halo side; LDS row stride in halves (aligned, staggers banks)
    constexpr int V8 = CIN / 8, KC = CIN / 16, KT = 9 * KC;
    __shared__ __attribute__((aligned(16))) half_t L[HS * HS * RS];
    const TileDev T = P.td[blockIdx.y];
    const int bpr = (T.cols + kB - 1) / kB;
    if ((int)blockIdx.x >= bpr * ((T.rows + kB - 1) / kB)) return;   // uniform per block: no WMMA lane is left out
    const int by0 = (blockIdx.x / bpr) * kB, bx0 = (blockIdx.x % bpr) * kB, base = T.off - P.o0;
    for (int e = threadIdx.x; e < HS * HS * V8; e += kT) {
        const int q = e / V8, c0 = (e % V8) * 8, y = by0 - 1 + q / HS, x = bx0 - 1 + q % HS;
        h8 v = (half_t)0.0f;
        if (y >= 0 && y < T.rows && x >= 0 && x < T.cols) v = *(const h8*)(P.IN + (size_t)(base + y * T.cols + x) * CIN + c0);
        *(h8*)(L + q * RS + c0) = v;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int p = 16 * w + r16, py = p / kB, px = p % kB;
    f8 acc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = splat(P.bias[nt * 16 + r16]);
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
        const half_t* a = L + ((py + tap / 3) * HS + px + tap % 3) * RS;
#pragma unroll
        for (int ch = 0; ch < KC; ++ch) {
            const h16 av = ld16(a + ch * 16);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[nt] = mma(av, ldw(P.Wt, KT, nt, tap * KC + ch, r16), acc[nt]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) pin(acc[nt]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = 16 * w + 2 * i + hi, y = by0 + q / kB, x = bx0 + q % kB;
        if (y >= T.rows || x >= T.cols) continue;
        if (LAST) {
            if (r16 != 0) continue;
            const int PW = T.cols + 2 * P.border;
            const float z = P.Q[P.qoff[blockIdx.y] + (y + P.border) * PW + x + P.border] * (1.0f / 255.0f) + acc[0][i];
            P.DN[T.off + y * T.cols + x] = (std::uint8_t)__float2int_rn(fminf(fmaxf(z, 0.0f), 1.0f) * 255.0f);
        } else {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                P.OUT[(size_t)(base + y * T.cols + x) * (NT * 16) + nt * 16 + r16] = (half_t)fmaxf(acc[nt][i], 0.0f);
        }
    }
}

bool launch(const Layer& L, bool last, unsigned gx, unsigned gy, const ConvP& P) {
#define CS_PROX_CONV(CI, N, LA) if (L.cin_p == CI && L.nt == N && last == LA) { k_conv<CI, N, LA><<<dim3(gx, gy), kT>>>(P); return true; }
    CS_PROX_CONV(16, 1, false) CS_PROX_CONV(16, 2, false) CS_PROX_CONV(32, 1, false) CS_PROX_CONV(32, 2, false)
    CS_PROX_CONV(16, 1, true) CS_PROX_CONV(32, 1, true)
#undef CS_PROX_CONV
    return false;
}

int up16(int v) { return (v + 15) / 16 * 16; }

}  // namespace

bool load(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_net.empty()) return path == g_path;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "gpu_prox: cannot open %s\n", path.c_str()); return false; }
    char magic[4] = {};
    int nl = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::string(magic, 4) == "PRX1" && std::fread(&nl, 4, 1, f) == 1 && nl >= 2;
    std::vector<Layer> net;
    std::vector<std::uint16_t> hw;
    std::vector<float> hb;
    for (int l = 0; ok && l < nl; ++l) {
        int dims[2];
        ok = std::fread(dims, 4, 2, f) == 2;
        if (!ok) break;
        const int cin = dims[0], cout = dims[1];
        // layer 0 reads the input planes (2 or 6); hidden widths up to kHC; the last layer has one output
        ok = (l == 0 ? (cin == 2 || cin == 6) : cin == net.back().cout) && cout >= 1 && cout <= kHC && (l < nl - 1 || cout == 1);
        if (!ok) break;
        std::vector<float> w((size_t)cout * cin * 9), b((size_t)cout);
        ok = std::fread(w.data(), 4, w.size(), f) == w.size() && std::fread(b.data(), 4, b.size(), f) == b.size();
        Layer L{ cin, cout, l == 0 ? kXC : up16(cin), up16(cout) / 16, hw.size(), hb.size() };
        // B fragments of W(k, n), k = tap * cin_p + c: tile nt, k-step kt, lane column j, 16 k values (ldw())
        for (int nt = 0; nt < L.nt; ++nt)
            for (int kt = 0; kt < 9 * L.cin_p / 16; ++kt)
                for (int j = 0; j < 16; ++j)
                    for (int e = 0; e < 16; ++e) {
                        const int k = kt * 16 + e, n = nt * 16 + j, tap = k / L.cin_p, c = k % L.cin_p;
                        hw.push_back(to_half(c < cin && n < cout ? w[((size_t)n * cin + c) * 9 + tap] : 0.0f));
                    }
        for (int n = 0; n < L.nt * 16; ++n) hb.push_back(n < cout ? b[(size_t)n] : 0.0f);
        net.push_back(L);
    }
    std::fclose(f);
    ok = ok && hipMalloc((void**)&g_w, hw.size() * 2) == hipSuccess && hipMalloc((void**)&g_b, hb.size() * 4) == hipSuccess &&
         hipMalloc((void**)&g_scr, (size_t)kChunkPx * (kXC + 2 * kHC) * sizeof(half_t)) == hipSuccess &&
         hipMemcpy(g_w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(g_b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice) == hipSuccess;
    if (!ok) {
        std::fprintf(stderr, "gpu_prox: %s is not a supported prox weight file (or device allocation failed)\n", path.c_str());
        (void)hipFree(g_w); (void)hipFree(g_b); (void)hipFree(g_scr); g_w = nullptr; g_b = nullptr; g_scr = nullptr;
        return false;
    }
    g_net = std::move(net);
    g_path = path;
    return true;
}

bool run(int nt, int max_rows, int max_cols, const TileDev* td, const TileDev* td_host, const int* qoff, int border,
         const std::uint8_t* Q, const float* BLR, const float* HAV, const float* H, float hmul, std::uint8_t* DN) {
    if (g_net.empty() || nt <= 0) return false;
    if ((long long)max_rows * max_cols > kChunkPx) return false;
    half_t* X = g_scr;
    half_t* bufs[2] = { g_scr + (size_t)kChunkPx * kXC, g_scr + (size_t)kChunkPx * (kXC + kHC) };
    const unsigned gfeat = (unsigned)((max_rows * max_cols + kThreads - 1) / kThreads);
    const unsigned gconv = (unsigned)(((max_rows + kB - 1) / kB) * ((max_cols + kB - 1) / kB));
    for (int t0 = 0; t0 < nt;) {
        int t1 = t0;
        long long px = 0;
        for (; t1 < nt; ++t1) {
            const long long tpx = (long long)td_host[t1].rows * td_host[t1].cols;
            if (px + tpx > kChunkPx) break;
            px += tpx;
        }
        const unsigned n = (unsigned)(t1 - t0);
        const int o0 = td_host[t0].off;
        k_feat<<<dim3(gfeat, n), dim3(kThreads)>>>(td + t0, o0, qoff + t0, border, Q, BLR, HAV, H + t0, hmul, g_net.front().cin, X);
        const half_t* in = X;
        for (size_t l = 0; l < g_net.size(); ++l) {
            const Layer& L = g_net[l];
            const bool last = l + 1 == g_net.size();
            ConvP P{ td + t0, o0, in, bufs[l % 2], g_w + L.w, g_b + L.b, qoff + t0, border, Q, DN };
            if (!launch(L, last, gconv, n, P)) {
                std::fprintf(stderr, "gpu_prox: no conv kernel for %d -> %d channels\n", L.cin, L.cout);
                return false;
            }
            in = bufs[l % 2];
        }
        t0 = t1;
    }
    return hipGetLastError() == hipSuccess;
}

}  // namespace gpu_prox
