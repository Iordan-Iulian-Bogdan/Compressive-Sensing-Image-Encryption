// Restoration U-Net of --enhance on HIP (see gpu_enh.h).
//
// Network (UNetX of x64/GpuBuild/train_enh_fresh.py, c = 16), NHWC fp16 activations, channels padded to 16:
//   X  = [rgb/255, planes]                 T  = relu(head(X))           S0 = e0(T)               full res, 16 ch
//   S1 = e1(relu(down1(S0)))               half res, 32 ch              M  = mid(relu(down2(S1))) quarter res, 64 ch
//   U2 = relu(ps(up2(M)))                  D  = d1(relu(fuse1([U2, S1])))                        half res, 32 ch
//   U1 = relu(ps(up1(D)))                  V  = fuse0([U1, S0])         out = rgb/255 + tail(V)   full res
// with ResBlock(x) = x + b(relu(a(x))) and ps = PixelShuffle(2). Every conv is 3x3 and zero padded (stride 2 for
// down1 / down2) and runs as an implicit GEMM over an fp16 halo in LDS on the WMMA units (fp32 accumulation); bias,
// ReLU, residual add, pixel shuffle, concatenated inputs and the final clamped 8-bit store are fused into it. The
// whole image runs at once. Sides that are not multiples of 4 are reflect padded (as UNetX.forward) and cropped back.
//
// Planes (enh_inputs.py): density = bilinear(Gaussian(mask * 255, sigma 4 on the LR grid)) / 255 - 0.5, kept in float
// (the CPU / Python versions round the two 8-bit intermediates); anchor indicator and (y - luma) / 64; LR sample mask
// and (sample - 2x2 mean of the decoded luma) / 8 on each sample's 2x2 block. Residuals clipped to [-1, 1].
#include "gpu_enh.h"
#include "gpu_common.hpp"
#include "gpu_wmma.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace gpu_enh {
namespace {

using namespace gpu_common;
using namespace gpu_wmma;

constexpr int kT = 128;          // four waves of 32; a wave owns 16 pixels of an 8 x 8 output block
constexpr int kB = 8;            // output block side
constexpr int kE = 256;          // threads of the elementwise kernels
constexpr int kXC = 16;          // channels of the input tensor X (3 + up to 5 planes, zero padded)
constexpr int kBlurR = 12;       // Gaussian radius on the LR grid (OpenCV's 25 taps for sigma 4)
constexpr float kBlurSigma = 4.0f;

struct Conv { int cin, cout, cin_p, cout_p; size_t w, b; };   // fp16 fragment / fp32 bias offsets
struct Model {
    DevBuf<float> fbuf;
    DevBuf<half_t> hbuf;
    int c = 0, nx = 0, blocks[3] = {};
    std::vector<Conv> conv;
};
std::mutex g_mu;
std::map<std::string, Model*> g_models;
Model* g_cur = nullptr;

__device__ inline h16 ldw(const half_t* W, int KT, int nt, int kt, int r16) { return ld16(W + ((size_t)(nt * KT + kt) * 16 + r16) * 16); }
__device__ inline float lum(const std::uint8_t* p) { return 0.299f * p[2] + 0.587f * p[1] + 0.114f * p[0]; }   // BGR bytes
__device__ inline float clip1(float v) { return fminf(1.0f, fmaxf(-1.0f, v)); }

struct ConvP {
    const half_t* A; const half_t* B;    // input = [A (CA channels), B (CB channels)], NHWC; B may be null (CB = 0)
    int CA, CB, IH, IW, OH, OW;
    const half_t* Wt; const float* bias;
    const half_t* RES;                   // mode 0: added before the ReLU (same layout as OUT), may be null
    half_t* OUT; int COUT;               // output channels (padded); mode 1 stores COUT / 4 channels at 2x
    int mode, relu;                      // 0 store, 1 pixel shuffle x2, 2 final 8-bit BGR
    const std::uint8_t* img; std::uint8_t* out8; int H, W;   // mode 2: the real image and its size
};

// One 3x3 conv on an 8 x 8 output block: CIN input channels (A then B), NT x 16 output channels, stride S.
template <int CIN, int NT, int S>
__global__ void __launch_bounds__(kT) k_conv(const ConvP P) {
    constexpr int HS = (kB - 1) * S + 3;   // halo side: 10 (stride 1) or 17 (stride 2)
    constexpr int RS = CIN + 8;            // LDS row stride in halves (16-byte aligned, staggers banks)
    constexpr int V8 = CIN / 8, KC = CIN / 16, KT = 9 * KC;
    __shared__ __attribute__((aligned(16))) half_t L[HS * HS * RS];
    const int bpr = (P.OW + kB - 1) / kB;
    const int by0 = (blockIdx.x / bpr) * kB, bx0 = (blockIdx.x % bpr) * kB;
    const int iy0 = by0 * S - 1, ix0 = bx0 * S - 1;
    for (int e = threadIdx.x; e < HS * HS * V8; e += kT) {
        const int q = e / V8, c0 = (e % V8) * 8, y = iy0 + q / HS, x = ix0 + q % HS;
        h8 v = (half_t)0.0f;
        if (y >= 0 && y < P.IH && x >= 0 && x < P.IW) {
            const size_t px = (size_t)y * P.IW + x;
            v = c0 < P.CA ? *(const h8*)(P.A + px * P.CA + c0) : *(const h8*)(P.B + px * P.CB + (c0 - P.CA));
        }
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
        const half_t* a = L + ((py * S + tap / 3) * HS + px * S + tap % 3) * RS;
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
        if (y >= P.OH || x >= P.OW) continue;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 16 + r16;
            float v = acc[nt][i];
            if (P.mode == 0) {
                const size_t o = ((size_t)y * P.OW + x) * P.COUT + c;
                if (P.RES) v += (float)P.RES[o];
                if (P.relu) v = fmaxf(v, 0.0f);
                P.OUT[o] = (half_t)v;
            } else if (P.mode == 1) {   // PixelShuffle: out[c, 2y + i, 2x + j] = in[4c + 2i + j, y, x]
                const int co = c >> 2, ii = (c >> 1) & 1, jj = c & 1, cs = P.COUT >> 2;
                if (P.relu) v = fmaxf(v, 0.0f);
                P.OUT[((size_t)(2 * y + ii) * (2 * P.OW) + 2 * x + jj) * cs + co] = (half_t)v;
            } else if (c < 3 && y < P.H && x < P.W) {   // enhanced = input rgb + residual, clamped, 8-bit BGR
                const size_t o = ((size_t)y * P.W + x) * 3 + (2 - c);
                v = fminf(1.0f, fmaxf(0.0f, v + P.img[o] * (1.0f / 255.0f)));
                P.out8[o] = (std::uint8_t)__float2int_rn(v * 255.0f);   // round half to even, as torch / nearbyint
            }
        }
    }
}

// Density, step 1: horizontal Gaussian of (mask ? 255 : 0) on the LR grid (replicated edges).
__global__ void k_blur_h(const std::uint8_t* M, float* T, int h, int w, const float* g) {
    const int i = blockIdx.x * kE + threadIdx.x;
    if (i >= h * w) return;
    const int y = i / w, x = i % w;
    float s = 0.0f;
    for (int k = -kBlurR; k <= kBlurR; ++k) s += g[k + kBlurR] * (M[(size_t)y * w + min(max(x + k, 0), w - 1)] ? 255.0f : 0.0f);
    T[i] = s;
}
// Density, step 2: vertical Gaussian.
__global__ void k_blur_v(const float* T, float* Dn, int h, int w, const float* g) {
    const int i = blockIdx.x * kE + threadIdx.x;
    if (i >= h * w) return;
    const int y = i / w, x = i % w;
    float s = 0.0f;
    for (int k = -kBlurR; k <= kBlurR; ++k) s += g[k + kBlurR] * T[(size_t)min(max(y + k, 0), h - 1) * w + x];
    Dn[i] = s;
}

// Input tensor X (Hp x Wp x 16, reflect padded past H x W): rgb, density, anchors (0 here, see k_anchors), LR samples.
__global__ void k_input(const std::uint8_t* img, int H, int W, int Hp, int Wp, const float* Dn, const std::uint8_t* M,
    const std::uint8_t* Sm, int h, int w, int nx, half_t* X) {
    const int i = blockIdx.x * kE + threadIdx.x;
    if (i >= Hp * Wp) return;
    const int yp = i / Wp, xp = i % Wp;
    const int y = yp < H ? yp : 2 * (H - 1) - yp, x = xp < W ? xp : 2 * (W - 1) - xp;
    const std::uint8_t* b = img + ((size_t)y * W + x) * 3;
    float o[kXC] = {};
    o[0] = b[2] * (1.0f / 255.0f); o[1] = b[1] * (1.0f / 255.0f); o[2] = b[0] * (1.0f / 255.0f);
    o[3] = -0.5f;
    if (M) {   // bilinear, as cv::resize INTER_LINEAR (half-pixel centres, clamped)
        const float sy = fmaxf((y + 0.5f) * ((float)h / H) - 0.5f, 0.0f), sx = fmaxf((x + 0.5f) * ((float)w / W) - 0.5f, 0.0f);
        int y0 = (int)sy, x0 = (int)sx;
        float fy = sy - y0, fx = sx - x0;
        if (y0 >= h - 1) { y0 = h - 1; fy = 0.0f; }
        if (x0 >= w - 1) { x0 = w - 1; fx = 0.0f; }
        const int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
        const float d = (1 - fy) * ((1 - fx) * Dn[(size_t)y0 * w + x0] + fx * Dn[(size_t)y0 * w + x1]) +
                        fy * ((1 - fx) * Dn[(size_t)y1 * w + x0] + fx * Dn[(size_t)y1 * w + x1]);
        o[3] = d * (1.0f / 255.0f) - 0.5f;
        const int li = y / 2, lj = x / 2;
        if (nx >= 5 && Sm && li < min(h, H / 2) && lj < min(w, W / 2) && M[(size_t)li * w + lj]) {
            const std::uint8_t* q = img + ((size_t)(2 * li) * W + 2 * lj) * 3;
            const float box = (lum(q) + lum(q + 3) + lum(q + (size_t)W * 3) + lum(q + (size_t)W * 3 + 3)) * 0.25f;
            o[6] = 1.0f;
            o[7] = clip1((Sm[(size_t)li * w + lj] - box) / 8.0f);
        }
    }
    h8 v0, v1;
    #pragma unroll
    for (int k = 0; k < 8; ++k) { v0[k] = (half_t)o[k]; v1[k] = (half_t)o[8 + k]; }
    *(h8*)(X + (size_t)i * kXC) = v0;
    *(h8*)(X + (size_t)i * kXC + 8) = v1;
}

// Anchor indicator / residual into channels 4 / 5 of X (real pixels only).
__global__ void k_anchors(const Anchor* A, int n, const std::uint8_t* img, int H, int W, int Wp, half_t* X) {
    const int t = blockIdx.x * kE + threadIdx.x;
    if (t >= n) return;
    const std::uint32_t idx = A[t].idx;
    if (idx >= (std::uint32_t)H * (std::uint32_t)W) return;
    const int y = (int)(idx / (std::uint32_t)W), x = (int)(idx % (std::uint32_t)W);
    half_t* o = X + ((size_t)y * Wp + x) * kXC;
    o[4] = (half_t)1.0f;
    o[5] = (half_t)clip1((A[t].y - lum(img + (size_t)idx * 3)) / 64.0f);
}

bool launch_conv(int cin, int nt, int s, const ConvP& P) {
    const unsigned g = (unsigned)(((P.OH + kB - 1) / kB) * ((P.OW + kB - 1) / kB));
#define CS_ENH_CONV(CI, N, S_) if (cin == CI && nt == N && s == S_) { k_conv<CI, N, S_><<<g, kT>>>(P); return true; }
    CS_ENH_CONV(16, 1, 1) CS_ENH_CONV(16, 2, 2) CS_ENH_CONV(32, 2, 1) CS_ENH_CONV(32, 4, 2) CS_ENH_CONV(64, 4, 1)
    CS_ENH_CONV(64, 8, 1) CS_ENH_CONV(64, 2, 1) CS_ENH_CONV(32, 4, 1) CS_ENH_CONV(32, 1, 1)
#undef CS_ENH_CONV
    std::fprintf(stderr, "gpu_enh: no conv kernel for %d input channels, %d output tiles, stride %d\n", cin, nt, s);
    return false;
}

// Packs W(k, n) (K x N, zero outside) as WMMA B fragments: tile nt, k-step kt, lane column j, 16 k values (ldw()).
template <class F>
size_t pack(std::vector<std::uint16_t>& out, int K, int N, F Wf) {
    const size_t at = out.size();
    for (int nt = 0; nt < N / 16; ++nt)
        for (int kt = 0; kt < K / 16; ++kt)
            for (int j = 0; j < 16; ++j)
                for (int e = 0; e < 16; ++e) out.push_back(to_half(Wf(kt * 16 + e, nt * 16 + j)));
    return at;
}

int up16(int v) { return (v + 15) / 16 * 16; }

}  // namespace

std::string weights_path(const std::string& onnx_model) {
    const size_t dot = onnx_model.find_last_of('.'), slash = onnx_model.find_last_of("/\\");
    const std::string stem = (dot != std::string::npos && (slash == std::string::npos || dot > slash)) ? onnx_model.substr(0, dot) : onnx_model;
    return stem + ".enh.bin";
}

int extra_planes() { return g_cur ? g_cur->nx : 0; }

bool load(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto it = g_models.find(path); it != g_models.end()) { g_cur = it->second; return g_cur != nullptr; }
    g_models[path] = nullptr;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    // header: magic "ENH1", version 1, c, blocks[3], extra planes, conv count; then per conv: cout, cin, w, b (fp32)
    std::uint32_t hdr[8] = {};
    std::vector<float> fv;
    std::vector<std::uint16_t> hv;
    auto* m = new Model();
    bool ok = std::fread(hdr, 4, 8, f) == 8 && hdr[0] == 0x31484E45u && hdr[1] == 1 && hdr[2] == 16 && (hdr[6] == 3 || hdr[6] == 5);
    if (ok) {
        m->c = (int)hdr[2]; m->nx = (int)hdr[6];
        for (int k = 0; k < 3; ++k) m->blocks[k] = (int)hdr[3 + k];
        const int c = m->c, *bk = m->blocks;
        // expected (cin, cout) of every conv, in file order
        std::vector<std::pair<int, int>> want{ { 3 + m->nx, c } };
        for (int i = 0; i < 2 * bk[0]; ++i) want.push_back({ c, c });
        want.push_back({ c, 2 * c });
        for (int i = 0; i < 2 * bk[1]; ++i) want.push_back({ 2 * c, 2 * c });
        want.push_back({ 2 * c, 4 * c });
        for (int i = 0; i < 2 * bk[2]; ++i) want.push_back({ 4 * c, 4 * c });
        want.push_back({ 4 * c, 8 * c });       // up2 (pixel shuffle -> 2c)
        want.push_back({ 4 * c, 2 * c });       // fuse1 ([U2, S1])
        for (int i = 0; i < 2 * bk[1]; ++i) want.push_back({ 2 * c, 2 * c });
        want.push_back({ 2 * c, 4 * c });       // up1 (pixel shuffle -> c)
        want.push_back({ 2 * c, c });           // fuse0 ([U1, S0])
        want.push_back({ c, 3 });               // tail
        ok = hdr[7] == want.size();
        for (size_t k = 0; ok && k < want.size(); ++k) {
            std::uint32_t co_ci[2];
            ok = std::fread(co_ci, 4, 2, f) == 2 && (int)co_ci[0] == want[k].second && (int)co_ci[1] == want[k].first;
            if (!ok) break;
            Conv cv{ (int)co_ci[1], (int)co_ci[0], k == 0 ? kXC : up16((int)co_ci[1]), up16((int)co_ci[0]), 0, 0 };
            std::vector<float> w((size_t)cv.cout * cv.cin * 9), b((size_t)cv.cout);
            ok = std::fread(w.data(), 4, w.size(), f) == w.size() && std::fread(b.data(), 4, b.size(), f) == b.size();
            if (!ok) break;
            cv.w = pack(hv, 9 * cv.cin_p, cv.cout_p, [&](int kk, int n) {   // K = tap * cin_p + input channel
                const int tap = kk / cv.cin_p, ic = kk % cv.cin_p;
                return (ic < cv.cin && n < cv.cout) ? w[((size_t)n * cv.cin + ic) * 9 + tap] : 0.0f;
            });
            cv.b = fv.size();
            for (int n = 0; n < cv.cout_p; ++n) fv.push_back(n < cv.cout ? b[(size_t)n] : 0.0f);
            m->conv.push_back(cv);
        }
        float extra;
        ok = ok && std::fread(&extra, 4, 1, f) == 0;   // nothing after the last conv
    }
    std::fclose(f);
    // Gaussian taps of the density blur (OpenCV getGaussianKernel(25, 4)), after the last bias (see run())
    double gs = 0.0;
    for (int k = -kBlurR; k <= kBlurR; ++k) gs += std::exp(-(double)k * k / (2.0 * kBlurSigma * kBlurSigma));
    for (int k = -kBlurR; k <= kBlurR; ++k) fv.push_back((float)(std::exp(-(double)k * k / (2.0 * kBlurSigma * kBlurSigma)) / gs));
    if (!ok || !m->fbuf.alloc(fv.size()) || !m->hbuf.alloc(hv.size()) ||
        hipMemcpy(m->fbuf.p, fv.data(), fv.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(m->hbuf.p, hv.data(), hv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "gpu_enh: %s is not a restoration-net weight file (re-run x64/GpuBuild/export_enh_onnx.py)\n", path.c_str());
        delete m;
        return false;
    }
    g_models[path] = m;
    g_cur = m;
    return true;
}

bool run(const Inputs& in, std::uint8_t* out_bgr, double* ms) {
    Model* m = g_cur;
    if (!m || !in.bgr || in.H < 2 || in.W < 2) return false;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const int H = in.H, W = in.W, Hp = (H + 3) / 4 * 4, Wp = (W + 3) / 4 * 4, Hh = Hp / 2, Wh = Wp / 2, Hq = Hp / 4, Wq = Wp / 4;
    const int c = m->c;
    const size_t npx = (size_t)H * W, nf = (size_t)Hp * Wp, nh = (size_t)Hh * Wh, nq = (size_t)Hq * Wq;
    const bool lr = in.y_mask && in.h > 0 && in.w > 0;
    const size_t nlr = lr ? (size_t)in.h * in.w : 1;
    DevBuf<std::uint8_t> dimg, dout, dM, dS;
    DevBuf<float> dT, dD;
    DevBuf<Anchor> dA;
    DevBuf<half_t> X, T0, T1, S0, Hb[4], Qb[3];
    bool ok = dimg.alloc(npx * 3) && dout.alloc(npx * 3) && dM.alloc(nlr) && dS.alloc(nlr) && dT.alloc(nlr) && dD.alloc(nlr) &&
        dA.alloc(in.n_anchors > 0 ? (size_t)in.n_anchors : 1) && X.alloc(nf * kXC) && T0.alloc(nf * c) && T1.alloc(nf * c) && S0.alloc(nf * c);
    for (auto& b : Hb) ok = ok && b.alloc(nh * 2 * c);
    for (auto& b : Qb) ok = ok && b.alloc(nq * 4 * c);
    if (!ok) { std::fprintf(stderr, "gpu_enh: device allocation failed\n"); return false; }
    HIP_TRY(hipMemcpy(dimg.p, in.bgr, npx * 3, hipMemcpyHostToDevice));
    if (lr) {
        HIP_TRY(hipMemcpy(dM.p, in.y_mask, nlr, hipMemcpyHostToDevice));
        if (in.y_samp) HIP_TRY(hipMemcpy(dS.p, in.y_samp, nlr, hipMemcpyHostToDevice));
    }
    if (in.n_anchors > 0) HIP_TRY(hipMemcpy(dA.p, in.anchors, (size_t)in.n_anchors * sizeof(Anchor), hipMemcpyHostToDevice));
    const float* gk = m->fbuf.p + m->conv.back().b + m->conv.back().cout_p;   // Gaussian taps (after the last bias)
    auto grid = [](size_t n) { return dim3((unsigned)((n + kE - 1) / kE)); };
    if (lr) {
        k_blur_h<<<grid(nlr), kE>>>(dM.p, dT.p, in.h, in.w, gk);
        k_blur_v<<<grid(nlr), kE>>>(dT.p, dD.p, in.h, in.w, gk);
    }
    k_input<<<grid(nf), kE>>>(dimg.p, H, W, Hp, Wp, dD.p, lr ? dM.p : nullptr, lr && in.y_samp ? dS.p : nullptr,
        in.h, in.w, m->nx, X.p);
    if (in.n_anchors > 0) k_anchors<<<grid((size_t)in.n_anchors), kE>>>(dA.p, in.n_anchors, dimg.p, H, W, Wp, X.p);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    const auto t1 = clk::now();

    size_t k = 0;
    auto conv = [&](const half_t* A, int CA, const half_t* B, int CB, int IH, int IW, int S, half_t* OUT,
                    const half_t* RES, int mode, int relu) {
        const Conv& cv = m->conv[k++];
        ConvP P{};
        P.A = A; P.B = B; P.CA = CA; P.CB = CB; P.IH = IH; P.IW = IW; P.OH = IH / S; P.OW = IW / S;
        P.Wt = m->hbuf.p + cv.w; P.bias = m->fbuf.p + cv.b; P.RES = RES; P.OUT = OUT; P.COUT = cv.cout_p;
        P.mode = mode; P.relu = relu; P.img = dimg.p; P.out8 = dout.p; P.H = H; P.W = W;
        return launch_conv(CA + CB, cv.cout_p / 16, S, P);
    };
    // n residual blocks at (IH, IW, C): cur -> result pointer; tmp holds a(x), alt receives x + b(...)
    auto resblocks = [&](int n, half_t* cur, half_t* tmp, half_t* alt, int C, int IH, int IW) -> half_t* {
        for (int i = 0; i < n; ++i) {
            if (!conv(cur, C, nullptr, 0, IH, IW, 1, tmp, nullptr, 0, 1)) return nullptr;
            if (!conv(tmp, C, nullptr, 0, IH, IW, 1, alt, cur, 0, 0)) return nullptr;
            std::swap(cur, alt);
        }
        return cur;
    };
    const int* bk = m->blocks;
    ok = conv(X.p, kXC, nullptr, 0, Hp, Wp, 1, T0.p, nullptr, 0, 1);                               // head
    half_t* s0 = ok ? resblocks(bk[0], T0.p, T1.p, S0.p, c, Hp, Wp) : nullptr;                     // e0
    ok = s0 && conv(s0, c, nullptr, 0, Hp, Wp, 2, Hb[0].p, nullptr, 0, 1);                          // down1
    half_t* s1 = ok ? resblocks(bk[1], Hb[0].p, Hb[1].p, Hb[2].p, 2 * c, Hh, Wh) : nullptr;         // e1
    ok = s1 && conv(s1, 2 * c, nullptr, 0, Hh, Wh, 2, Qb[0].p, nullptr, 0, 1);                      // down2
    half_t* mm = ok ? resblocks(bk[2], Qb[0].p, Qb[1].p, Qb[2].p, 4 * c, Hq, Wq) : nullptr;        // mid
    std::vector<half_t*> freeh;                                                                     // half-res buffers not holding s1
    for (auto& b : Hb) if (b.p != s1) freeh.push_back(b.p);
    ok = mm && conv(mm, 4 * c, nullptr, 0, Hq, Wq, 1, freeh[0], nullptr, 1, 1);                     // up2 -> U2
    ok = ok && conv(freeh[0], 2 * c, s1, 2 * c, Hh, Wh, 1, freeh[1], nullptr, 0, 1);                 // fuse1 -> D
    half_t* d = ok ? resblocks(bk[1], freeh[1], freeh[0], freeh[2], 2 * c, Hh, Wh) : nullptr;      // d1
    half_t* u1 = s0 == T0.p ? S0.p : T0.p;                                                         // full-res buffer free after e0
    ok = d && conv(d, 2 * c, nullptr, 0, Hh, Wh, 1, u1, nullptr, 1, 1);                             // up1 -> U1
    ok = ok && conv(u1, c, s0, c, Hp, Wp, 1, T1.p, nullptr, 0, 0);                                  // fuse0 -> V (linear)
    ok = ok && conv(T1.p, c, nullptr, 0, Hp, Wp, 1, nullptr, nullptr, 2, 0);                        // tail -> 8-bit
    if (!ok || k != m->conv.size()) return false;
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    const auto t2 = clk::now();
    HIP_TRY(hipMemcpy(out_bgr, dout.p, npx * 3, hipMemcpyDeviceToHost));
    if (ms) {
        ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms[1] = std::chrono::duration<double, std::milli>(t2 - t1).count();
        ms[2] = std::chrono::duration<double, std::milli>(clk::now() - t2).count();
    }
    return true;
}

}  // namespace gpu_enh
