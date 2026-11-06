// Restoration U-Net of --enhance on HIP (see gpu_enh.h).
//
// Network (UNetX of x64/GpuBuild/train_enh_fresh.py, c = 16 or 32), NHWC fp16 activations, channels padded to 16:
//   X  = [rgb/255, planes]                 T  = relu(head(X))           S0 = e0(T)               full res, c ch
//   S1 = e1(relu(down1(S0)))               half res, 2c ch              M  = mid(relu(down2(S1))) quarter res, 4c ch
//   U2 = relu(ps(up2(M)))                  D  = d1(relu(fuse1([U2, S1])))                        half res, 2c ch
//   U1 = relu(ps(up1(D)))                  V  = fuse0([U1, S0])         out = rgb/255 + tail(V)   full res
//   (version-2 weights: V = d0(relu(fuse0([U1, S0]))), d0 full-res residual blocks of c channels)
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
constexpr int kXC = 16;          // channels of the input tensor X (3 + up to 9 planes, zero padded)
constexpr int kBlurR = 12;       // Gaussian radius on the LR grid (OpenCV's 25 taps for sigma 4)
constexpr float kBlurSigma = 4.0f;

struct Conv { int cin, cout, cin_p, cout_p; size_t w, b; };   // fp16 fragment / fp32 bias offsets
struct Model {
    DevBuf<float> fbuf;
    DevBuf<half_t> hbuf;
    int c = 0, nx = 0, blocks[3] = {}, d0 = 0;   // d0: full-res residual blocks after relu(fuse0) (version-2 files)
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
    // Row windows (banded full-res layers): A / B / OUT / RES hold image rows from ia / ib / ob / rb on (OUT rows at
    // 2x in mode 1); output rows y0 .. y0 + ny - 1 are computed. All 0 and ny = OH: the whole image.
    int ia, ib, ob, rb, y0, ny;
    const std::uint8_t* act;             // ceil(OH / 8) x ceil(OW / 8) output block flags, null = all (gpu_enh::Inputs::active)
};

// One 3x3 conv on an 8 x 8 output block: CIN input channels (A then B), NT x 16 output channels from tile
// blockIdx.y * NT on (wide convs split their output tiles over blockIdx.y), stride S.
template <int CIN, int NT, int S>
__global__ void __launch_bounds__(kT) k_conv(const ConvP P) {
    constexpr int HS = (kB - 1) * S + 3;   // halo side: 10 (stride 1) or 17 (stride 2)
    constexpr int RS = CIN + 8;            // LDS row stride in halves (16-byte aligned, staggers banks)
    constexpr int V8 = CIN / 8, KC = CIN / 16, KT = 9 * KC;
    __shared__ __attribute__((aligned(16))) half_t L[HS * HS * RS];
    const int bpr = (P.OW + kB - 1) / kB;
    const int by0 = P.y0 + (blockIdx.x / bpr) * kB, bx0 = (blockIdx.x % bpr) * kB, yend = min(P.OH, P.y0 + P.ny);
    if (P.act) {   // band rows need not be 8-aligned: the block spans flag rows by0 / 8 and (by0 + 7) / 8
        const int nr = (P.OH + kB - 1) / kB, ry = by0 / kB, rz = min(nr - 1, (by0 + kB - 1) / kB);
        if (!P.act[ry * bpr + bx0 / kB] && !P.act[rz * bpr + bx0 / kB]) return;
    }
    const int iy0 = by0 * S - 1, ix0 = bx0 * S - 1;
    for (int e = threadIdx.x; e < HS * HS * V8; e += kT) {
        const int q = e / V8, c0 = (e % V8) * 8, y = iy0 + q / HS, x = ix0 + q % HS;
        h8 v = (half_t)0.0f;
        if (y >= 0 && y < P.IH && x >= 0 && x < P.IW)
            v = c0 < P.CA ? *(const h8*)(P.A + ((size_t)(y - P.ia) * P.IW + x) * P.CA + c0)
                          : *(const h8*)(P.B + ((size_t)(y - P.ib) * P.IW + x) * P.CB + (c0 - P.CA));
        *(h8*)(L + q * RS + c0) = v;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int p = 16 * w + r16, py = p / kB, px = p % kB, t0 = blockIdx.y * NT;
    f8 acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = splat(P.bias[(t0 + nt) * 16 + r16]);
    #pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
        const half_t* a = L + ((py * S + tap / 3) * HS + px * S + tap % 3) * RS;
        #pragma unroll
        for (int ch = 0; ch < KC; ++ch) {
            const h16 av = ld16(a + ch * 16);
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[nt] = mma(av, ldw(P.Wt, KT, t0 + nt, tap * KC + ch, r16), acc[nt]);
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) pin(acc[nt]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = 16 * w + 2 * i + hi, y = by0 + q / kB, x = bx0 + q % kB;
        if (y >= yend || x >= P.OW) continue;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = (t0 + nt) * 16 + r16;
            float v = acc[nt][i];
            if (P.mode == 0) {
                if (P.RES) v += (float)P.RES[((size_t)(y - P.rb) * P.OW + x) * P.COUT + c];
                if (P.relu) v = fmaxf(v, 0.0f);
                P.OUT[((size_t)(y - P.ob) * P.OW + x) * P.COUT + c] = (half_t)v;
            } else if (P.mode == 1) {   // PixelShuffle: out[c, 2y + i, 2x + j] = in[4c + 2i + j, y, x]
                const int co = c >> 2, ii = (c >> 1) & 1, jj = c & 1, cs = P.COUT >> 2;
                if (P.relu) v = fmaxf(v, 0.0f);
                P.OUT[((size_t)(2 * y + ii - P.ob) * (2 * P.OW) + 2 * x + jj) * cs + co] = (half_t)v;
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

// Chroma sample planes (9-plane nets): masks / values of Cr and Cb on the chroma grid (output / 4).
struct ChromaIn { const std::uint8_t *crm, *crs, *cbm, *cbs; int ch, cw; };

// Input tensor X (Hp x Wp x 16, reflect padded past H x W): rgb, density, anchors (0 here, see k_anchors), LR samples,
// chroma samples. X holds padded rows r0 .. r0 + nr - 1.
__global__ void k_input(const std::uint8_t* img, int H, int W, int Hp, int Wp, const float* Dn, const std::uint8_t* M,
    const std::uint8_t* Sm, int h, int w, int nx, ChromaIn C, half_t* X, int r0, int nr) {
    const int i = blockIdx.x * kE + threadIdx.x;
    if (i >= nr * Wp) return;
    const int yp = r0 + i / Wp, xp = i % Wp;
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
    // Cr / Cb sample mask and residual (sample - 4x4 mean of the decode's Cr / Cb, as cv::COLOR_BGR2YCrCb) / 4
    const int ci = y / 4, cj = x / 4;
    if (nx >= 9 && C.crm && ci < min(C.ch, H / 4) && cj < min(C.cw, W / 4)) {
        const size_t k = (size_t)ci * C.cw + cj;
        if (C.crm[k] || C.cbm[k]) {
            float R = 0.0f, G = 0.0f, B = 0.0f;
            for (int dy = 0; dy < 4; ++dy) {
                const std::uint8_t* q = img + ((size_t)(4 * ci + dy) * W + 4 * cj) * 3;
                for (int dx = 0; dx < 4; ++dx) { B += q[3 * dx]; G += q[3 * dx + 1]; R += q[3 * dx + 2]; }
            }
            R *= 1.0f / 16.0f; G *= 1.0f / 16.0f; B *= 1.0f / 16.0f;
            const float Yv = 0.299f * R + 0.587f * G + 0.114f * B;
            if (C.crm[k]) { o[8] = 1.0f; o[9] = clip1((C.crs[k] - ((R - Yv) * 0.713f + 128.0f)) / 4.0f); }
            if (C.cbm[k]) { o[10] = 1.0f; o[11] = clip1((C.cbs[k] - ((B - Yv) * 0.564f + 128.0f)) / 4.0f); }
        }
    }
    h8 v0, v1;
    #pragma unroll
    for (int k = 0; k < 8; ++k) { v0[k] = (half_t)o[k]; v1[k] = (half_t)o[8 + k]; }
    *(h8*)(X + (size_t)i * kXC) = v0;
    *(h8*)(X + (size_t)i * kXC + 8) = v1;
}

// Anchor indicator / residual into channels 4 / 5 of X (real pixels only).
__global__ void k_anchors(const Anchor* A, int n, const std::uint8_t* img, int H, int W, int Wp, half_t* X, int r0, int nr) {
    const int t = blockIdx.x * kE + threadIdx.x;
    if (t >= n) return;
    const std::uint32_t idx = A[t].idx;
    if (idx >= (std::uint32_t)H * (std::uint32_t)W) return;
    const int y = (int)(idx / (std::uint32_t)W), x = (int)(idx % (std::uint32_t)W);
    if (y < r0 || y >= r0 + nr) return;
    half_t* o = X + ((size_t)(y - r0) * Wp + x) * kXC;
    o[4] = (half_t)1.0f;
    o[5] = (half_t)clip1((A[t].y - lum(img + (size_t)idx * 3)) / 64.0f);
}

// nt output tiles go up to 8 per block (register budget), the rest over blockIdx.y.
bool launch_conv(int cin, int nt, int s, const ConvP& P) {
    const int ntb = std::min(nt, 8);
    const dim3 g((unsigned)(((P.ny + kB - 1) / kB) * ((P.OW + kB - 1) / kB)), (unsigned)(nt / ntb));
#define CS_ENH_CONV(CI, N, S_) if (cin == CI && ntb == N && s == S_) { k_conv<CI, N, S_><<<g, kT>>>(P); return true; }
    if (nt % ntb == 0) {
        // c = 16
        CS_ENH_CONV(16, 1, 1) CS_ENH_CONV(16, 2, 2) CS_ENH_CONV(32, 2, 1) CS_ENH_CONV(32, 4, 2) CS_ENH_CONV(64, 4, 1)
        CS_ENH_CONV(64, 8, 1) CS_ENH_CONV(64, 2, 1) CS_ENH_CONV(32, 4, 1) CS_ENH_CONV(32, 1, 1)
        // c = 32 (on top of the shared ones): head, down2, mid / up2, fuse1
        CS_ENH_CONV(16, 2, 1) CS_ENH_CONV(64, 8, 2) CS_ENH_CONV(128, 8, 1) CS_ENH_CONV(128, 4, 1)
    }
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

// full-res convs, half res x2, quarter x4, + slack for the strides / shuffles and the planes' 2x2 / 4x4 blocks
static int rf_of(const Model& m) {
    const int* bk = m.blocks;
    return (1 + 2 * bk[0]) + 2 + 2 * (2 * bk[1] + 1) + 4 * (2 * bk[2] + 1) + 2 * (1 + 2 * bk[1] + 1) + (1 + 2 * m.d0 + 1) + 8;
}
int receptive_field() { return g_cur ? rf_of(*g_cur) : 0; }

bool load(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto it = g_models.find(path); it != g_models.end()) { g_cur = it->second; return g_cur != nullptr; }
    g_models[path] = nullptr;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    // header: magic "ENH1", version (1, or 2 = one more u32: d0 blocks), c, blocks[3], extra planes, conv count; then per
    // conv: cout, cin, w, b (fp32)
    std::uint32_t hdr[8] = {}, nd0 = 0;
    std::vector<float> fv;
    std::vector<std::uint16_t> hv;
    auto* m = new Model();
    bool ok = std::fread(hdr, 4, 8, f) == 8 && hdr[0] == 0x31484E45u && (hdr[1] == 1 || hdr[1] == 2) &&
        (hdr[2] == 16 || hdr[2] == 32) && (hdr[6] == 3 || hdr[6] == 5 || hdr[6] == 9);
    if (ok && hdr[1] == 2) ok = std::fread(&nd0, 4, 1, f) == 1 && nd0 >= 1 && nd0 <= 8;
    if (ok) {
        m->c = (int)hdr[2]; m->nx = (int)hdr[6]; m->d0 = (int)nd0;
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
        for (int i = 0; i < 2 * m->d0; ++i) want.push_back({ c, c });   // d0
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
    // Plain hipMalloc, not the arena: the weights may load while a stage holds arena pieces (cs_enhance_prefetch), and a
    // long-lived piece above those would keep their space from being reused.
    if (!ok || hipMalloc((void**)&m->fbuf.p, fv.size() * sizeof(float)) != hipSuccess ||
        hipMalloc((void**)&m->hbuf.p, hv.size() * sizeof(half_t)) != hipSuccess ||
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
    // The full-res layers (X, head, e0 before down1; up1, fuse0, tail after d1) run in bands of BR output rows, so
    // their activations stay cache sized and only S0 (fuse0's skip input) is kept at full size. Each band recomputes
    // the halo rows its 3x3 convs need (one per conv), so the output is the same as the whole image at once.
    // CS_ENH_BAND=<rows> sets BR (default 128; 0 = one band).
    const int* bk = m->blocks;
    static const int band = [] { const char* e = std::getenv("CS_ENH_BAND"); return e ? std::max(0, std::atoi(e)) : 128; }();
    const int BR = band > 0 ? std::min(Hp, (band + 3) / 4 * 4) : Hp;
    const int hA = 1 + 2 * bk[0];                                   // X -> S0: head and e0's convs
    const int nd0 = m->d0;
    const int rows_f = std::min(Hp, BR + std::max(2 * hA, 6 + 4 * nd0));   // band buffer rows (stage B: U1 +-(2 + 2 d0), even ends)
    const size_t nb = (size_t)rows_f * Wp;
    DevBuf<std::uint8_t> dimg, dout, dM, dS;
    DevBuf<float> dT, dD;
    DevBuf<Anchor> dA;
    DevBuf<half_t> X, F[3], S0, Hb[4];
    bool ok = dimg.alloc(npx * 3) && dout.alloc(npx * 3) && dM.alloc(nlr) && dS.alloc(nlr) && dT.alloc(nlr) && dD.alloc(nlr) &&
        dA.alloc(in.n_anchors > 0 ? (size_t)in.n_anchors : 1) && X.alloc(nb * kXC) && S0.alloc(nf * c);
    for (auto& b : F) ok = ok && b.alloc(nb * c);
    for (auto& b : Hb) ok = ok && b.alloc(nh * 2 * c);
    if (!ok) { std::fprintf(stderr, "gpu_enh: device allocation failed\n"); return false; }
    if (const char* e = std::getenv("CS_GPU_PROF"); e && e[0] == '1') {
        (void)hipDeviceSynchronize();
        std::fprintf(stderr, "[gpu-prof] gpu_enh: device allocation %.1f ms (bands of %d rows)\n",
            std::chrono::duration<double, std::milli>(clk::now() - t0).count(), BR);
    }
    HIP_TRY(hipMemcpy(dimg.p, in.bgr, npx * 3, hipMemcpyHostToDevice));
    if (lr) {
        HIP_TRY(hipMemcpy(dM.p, in.y_mask, nlr, hipMemcpyHostToDevice));
        if (in.y_samp) HIP_TRY(hipMemcpy(dS.p, in.y_samp, nlr, hipMemcpyHostToDevice));
    }
    if (in.n_anchors > 0) HIP_TRY(hipMemcpy(dA.p, in.anchors, (size_t)in.n_anchors * sizeof(Anchor), hipMemcpyHostToDevice));
    ChromaIn chroma{};
    DevBuf<std::uint8_t> dC;
    if (m->nx >= 9 && in.cr_mask && in.cr_samp && in.cb_mask && in.cb_samp && in.ch > 0 && in.cw > 0) {
        const size_t nc = (size_t)in.ch * in.cw;
        if (!dC.alloc(4 * nc)) { std::fprintf(stderr, "gpu_enh: device allocation failed\n"); return false; }
        const std::uint8_t* src[4] = { in.cr_mask, in.cr_samp, in.cb_mask, in.cb_samp };
        for (int k = 0; k < 4; ++k) HIP_TRY(hipMemcpy(dC.p + k * nc, src[k], nc, hipMemcpyHostToDevice));
        chroma = { dC.p, dC.p + nc, dC.p + 2 * nc, dC.p + 3 * nc, in.ch, in.cw };
    }
    // Active blocks (in.active): the caller's blocks dilated by the receptive field, at full res; the half / quarter
    // res flags OR their 2 x 2 / 4 x 4 full-res blocks. Every conv runs on (at least) that region, so a conv's output
    // is exact wherever its 3x3 inputs are, and the stale values outside the region reach in by at most the receptive
    // field. Pixels outside it keep the input image.
    const std::uint8_t* act[3] = {};   // full, half, quarter res
    DevBuf<std::uint8_t> dAct;
    if (in.active && in.ah == (H + kB - 1) / kB && in.aw == (W + kB - 1) / kB) {
        const int rf = rf_of(*m);
        const int db = (rf + kB - 1) / kB;
        const int fh = (Hp + kB - 1) / kB, fw = (Wp + kB - 1) / kB, hh = (Hh + kB - 1) / kB, hw = (Wh + kB - 1) / kB,
                  qh = (Hq + kB - 1) / kB, qw = (Wq + kB - 1) / kB;
        std::vector<std::uint8_t> a((size_t)fh * fw + (size_t)hh * hw + (size_t)qh * qw, 0), t((size_t)fh * fw, 0);
        for (int y = 0; y < fh; ++y)   // horizontal then vertical max over +-db blocks
            for (int x = 0; x < fw; ++x) {
                std::uint8_t v = 0;
                for (int k = std::max(0, x - db); !v && k <= std::min(in.aw - 1, x + db); ++k) v = y < in.ah && in.active[(size_t)y * in.aw + k];
                t[(size_t)y * fw + x] = v;
            }
        std::size_t nact = 0;
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x) {
                std::uint8_t v = 0;
                for (int k = std::max(0, y - db); !v && k <= std::min(fh - 1, y + db); ++k) v = t[(size_t)k * fw + x];
                a[(size_t)y * fw + x] = v;
                nact += v;
                if (v) {
                    a[(size_t)fh * fw + (size_t)std::min(hh - 1, y / 2) * hw + std::min(hw - 1, x / 2)] = 1;
                    a[(size_t)fh * fw + (size_t)hh * hw + (size_t)std::min(qh - 1, y / 4) * qw + std::min(qw - 1, x / 4)] = 1;
                }
            }
        if (nact < (size_t)fh * fw) {
            if (!dAct.alloc(a.size())) { std::fprintf(stderr, "gpu_enh: device allocation failed\n"); return false; }
            HIP_TRY(hipMemcpy(dAct.p, a.data(), a.size(), hipMemcpyHostToDevice));
            HIP_TRY(hipMemcpy(dout.p, dimg.p, npx * 3, hipMemcpyDeviceToDevice));
            act[0] = dAct.p; act[1] = dAct.p + (size_t)fh * fw; act[2] = act[1] + (size_t)hh * hw;
        }
        if (const char* e = std::getenv("CS_GPU_PROF"); e && e[0] == '1')
            std::fprintf(stderr, "[gpu-prof] gpu_enh: %.0f%% of the blocks active (receptive field %d px)\n",
                100.0 * nact / ((double)fh * fw), rf);
    }
    const float* gk = m->fbuf.p + m->conv.back().b + m->conv.back().cout_p;   // Gaussian taps (after the last bias)
    auto grid = [](size_t n) { return dim3((unsigned)((n + kE - 1) / kE)); };
    if (lr) {
        k_blur_h<<<grid(nlr), kE>>>(dM.p, dT.p, in.h, in.w, gk);
        k_blur_v<<<grid(nlr), kE>>>(dT.p, dD.p, in.h, in.w, gk);
    }
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    const auto t1 = clk::now();

    // Conv number li (file order) on output rows y0 .. y0 + ny - 1; ia / ib / ob / rb: first image row held by A / B /
    // OUT / RES (OUT rows at 2x for mode 1).
    auto conv = [&](int li, const half_t* A, int CA, int ia, const half_t* B, int CB, int ib, int IH, int IW, int S,
                    half_t* OUT, int ob, const half_t* RES, int rb, int mode, int relu, int y0, int ny) {
        const Conv& cv = m->conv[(size_t)li];
        ConvP P{};
        P.A = A; P.B = B; P.CA = CA; P.CB = CB; P.IH = IH; P.IW = IW; P.OH = IH / S; P.OW = IW / S;
        P.Wt = m->hbuf.p + cv.w; P.bias = m->fbuf.p + cv.b; P.RES = RES; P.OUT = OUT; P.COUT = cv.cout_p;
        P.mode = mode; P.relu = relu; P.img = dimg.p; P.out8 = dout.p; P.H = H; P.W = W;
        P.ia = ia; P.ib = ib; P.ob = ob; P.rb = rb; P.y0 = y0; P.ny = ny;
        P.act = P.OH == Hp ? act[0] : P.OH == Hh ? act[1] : act[2];
        return launch_conv(CA + CB, cv.cout_p / 16, S, P);
    };
    auto full = [&](int li, const half_t* A, int CA, const half_t* B, int CB, int IH, int IW, int S, half_t* OUT,
                    const half_t* RES, int mode, int relu) {
        return conv(li, A, CA, 0, B, CB, 0, IH, IW, S, OUT, 0, RES, 0, mode, relu, 0, IH / S);
    };
    // n residual blocks from conv li at (IH, IW, C), whole image: cur -> result pointer
    auto resblocks = [&](int li, int n, half_t* cur, half_t* tmp, half_t* alt, int C, int IH, int IW) -> half_t* {
        for (int i = 0; i < n; ++i) {
            if (!full(li + 2 * i, cur, C, nullptr, 0, IH, IW, 1, tmp, nullptr, 0, 1)) return nullptr;
            if (!full(li + 2 * i + 1, tmp, C, nullptr, 0, IH, IW, 1, alt, cur, 0, 0)) return nullptr;
            std::swap(cur, alt);
        }
        return cur;
    };
    const int L_e0 = 1, L_down1 = L_e0 + 2 * bk[0], L_e1 = L_down1 + 1, L_down2 = L_e1 + 2 * bk[1], L_mid = L_down2 + 1,
              L_up2 = L_mid + 2 * bk[2], L_fuse1 = L_up2 + 1, L_d1 = L_fuse1 + 1, L_up1 = L_d1 + 2 * bk[1],
              L_fuse0 = L_up1 + 1, L_d0 = L_fuse0 + 1, L_tail = L_d0 + 2 * nd0;
    if (L_tail + 1 != (int)m->conv.size()) return false;
    // rows b0 - h .. b1 + h - 1 of the padded image, clipped
    auto win = [&](int b0, int b1, int h, int& r0, int& r1) { r0 = std::max(0, b0 - h); r1 = std::min(Hp, b1 + h); };

    // Stage A: X, head and e0 per band, into S0
    for (int b0 = 0; ok && b0 < Hp; b0 += BR) {
        const int b1 = std::min(Hp, b0 + BR);
        int x0, x1;
        win(b0, b1, hA, x0, x1);
        k_input<<<grid((size_t)(x1 - x0) * Wp), kE>>>(dimg.p, H, W, Hp, Wp, dD.p, lr ? dM.p : nullptr,
            lr && in.y_samp ? dS.p : nullptr, in.h, in.w, m->nx, chroma, X.p, x0, x1 - x0);
        if (in.n_anchors > 0)
            k_anchors<<<grid((size_t)in.n_anchors), kE>>>(dA.p, in.n_anchors, dimg.p, H, W, Wp, X.p, x0, x1 - x0);
        // conv l of the chain (head = 1) leaves halo hA - l; the last one writes rows b0 .. b1 - 1 of S0
        int h = hA - 1, r0, r1;
        win(b0, b1, h, r0, r1);
        half_t* cur = h == 0 ? S0.p : F[0].p;
        int cb = h == 0 ? 0 : r0;
        ok = conv(0, X.p, kXC, x0, nullptr, 0, 0, Hp, Wp, 1, cur, cb, nullptr, 0, 0, 1, r0, r1 - r0);   // head
        half_t* tmp = F[1].p; half_t* alt = F[2].p;
        for (int i = 0; ok && i < bk[0]; ++i) {                                                      // e0
            int t0_, t1_;
            win(b0, b1, --h, t0_, t1_);
            ok = conv(L_e0 + 2 * i, cur, c, cb, nullptr, 0, 0, Hp, Wp, 1, tmp, t0_, nullptr, 0, 0, 1, t0_, t1_ - t0_);
            win(b0, b1, --h, r0, r1);
            half_t* o = h == 0 ? S0.p : alt;
            const int ob = h == 0 ? 0 : r0;
            ok = ok && conv(L_e0 + 2 * i + 1, tmp, c, t0_, nullptr, 0, 0, Hp, Wp, 1, o, ob, cur, cb, 0, 0, r0, r1 - r0);
            alt = cur; cur = o; cb = ob;
        }
    }
    ok = ok && full(L_down1, S0.p, c, nullptr, 0, Hp, Wp, 2, Hb[0].p, nullptr, 0, 1);                       // down1
    half_t* s1 = ok ? resblocks(L_e1, bk[1], Hb[0].p, Hb[1].p, Hb[2].p, 2 * c, Hh, Wh) : nullptr;           // e1
    // The quarter-res buffers (each half a half-res one) live in two of the half-res buffers not holding s1 (fA, fB);
    // the third (fC) takes U2, so fuse1 and d1 can reuse fA / fB once the quarter-res data is dead.
    std::vector<half_t*> fr;                                                                                // half-res buffers not holding s1
    for (auto& b : Hb) if (b.p != s1) fr.push_back(b.p);
    half_t* Qb[3] = { fr[0], fr[0] + nq * 4 * c, fr[1] };
    ok = s1 && full(L_down2, s1, 2 * c, nullptr, 0, Hh, Wh, 2, Qb[0], nullptr, 0, 1);                       // down2
    half_t* mm = ok ? resblocks(L_mid, bk[2], Qb[0], Qb[1], Qb[2], 4 * c, Hq, Wq) : nullptr;                // mid
    half_t* freeh[3] = { fr[2], fr[0], fr[1] };
    ok = mm && full(L_up2, mm, 4 * c, nullptr, 0, Hq, Wq, 1, freeh[0], nullptr, 1, 1);                      // up2 -> U2
    ok = ok && full(L_fuse1, freeh[0], 2 * c, s1, 2 * c, Hh, Wh, 1, freeh[1], nullptr, 0, 1);                // fuse1 -> D
    half_t* d = ok ? resblocks(L_d1, bk[1], freeh[1], freeh[0], freeh[2], 2 * c, Hh, Wh) : nullptr;         // d1
    ok = d != nullptr;
    // Stage B: up1 (half-res rows -> U1 row pairs), fuse0 with S0, d0's blocks, tail per band. Each d0 conv eats one
    // halo row, so fuse0 writes rows b0 - 1 - 2 d0 .. b1 + 2 d0 (ReLU'd when d0 follows) and up1 one more.
    for (int b0 = 0; ok && b0 < Hp; b0 += BR) {
        const int b1 = std::min(Hp, b0 + BR);
        int v0, v1, u0, u1;
        int h = 1 + 2 * nd0;
        win(b0, b1, h, v0, v1);
        win(b0, b1, h + 1, u0, u1);
        u0 &= ~1; u1 = std::min(Hp, (u1 + 1) & ~1);
        ok = conv(L_up1, d, 2 * c, 0, nullptr, 0, 0, Hh, Wh, 1, F[0].p, u0, nullptr, 0, 1, 1, u0 / 2, (u1 - u0) / 2);     // up1 -> U1
        ok = ok && conv(L_fuse0, F[0].p, c, u0, S0.p, c, 0, Hp, Wp, 1, F[1].p, v0, nullptr, 0, 0, nd0 > 0, v0, v1 - v0); // fuse0 -> V
        half_t* cur = F[1].p; half_t* tmp = F[2].p; half_t* alt = F[0].p;   // U1 is dead once V exists
        int cb = v0;
        for (int i = 0; ok && i < nd0; ++i) {                                                        // d0
            int t0_, t1_, r0, r1;
            win(b0, b1, --h, t0_, t1_);
            ok = conv(L_d0 + 2 * i, cur, c, cb, nullptr, 0, 0, Hp, Wp, 1, tmp, t0_, nullptr, 0, 0, 1, t0_, t1_ - t0_);
            win(b0, b1, --h, r0, r1);
            ok = ok && conv(L_d0 + 2 * i + 1, tmp, c, t0_, nullptr, 0, 0, Hp, Wp, 1, alt, r0, cur, cb, 0, 0, r0, r1 - r0);
            std::swap(cur, alt); cb = r0;
        }
        ok = ok && conv(L_tail, cur, c, cb, nullptr, 0, 0, Hp, Wp, 1, nullptr, 0, nullptr, 0, 2, 0, b0, b1 - b0);     // tail -> 8-bit
    }
    if (!ok) return false;
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
