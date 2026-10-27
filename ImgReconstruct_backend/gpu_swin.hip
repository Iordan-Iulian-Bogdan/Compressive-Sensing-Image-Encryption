// Swin2SR-lightweight x2 on HIP (see gpu_swin.h).
//
// Same math as Swin2SRForImageSuperResolution (transformers, upsampler "pixelshuffledirect"),
// in the form scripts/export_swin2sr_weights.py checks against ONNX Runtime:
//   x = rgb/255 - mean; emb = conv3x3(x); h = LN(emb @ Wp + bp)
//   4 stages x 6 layers (odd layers shifted by 4): window attention with cosine scores,
//     scale exp(min(logit_scale, log 100)), 16*sigmoid(cpb-MLP) bias, shift mask -100 added twice
//     (as the reference does); h += LN(proj(attn)); h += LN(fc2(gelu(fc1(h))))
//   after each stage: h = conv1x1(conv3x3(h)) + stage input
//   out = pixel_shuffle(conv3x3(conv3x3(LN(h)) + emb)) + mean
// Tokens are stored per pixel, 60 channels contiguous (NHWC, fp32). The matrix products run on
// the RDNA3 WMMA units (16x16x16, fp16 inputs, fp32 accumulation); everything else (residual
// stream, LayerNorm, softmax, GELU) stays fp32 (with the hardware reciprocal / rsqrt and the tanh
// form of GELU). Against ONNX Runtime about 0.5% of the 8-bit outputs move by one level, none by more.
// Channels are padded to 64, head dims to 16 and the MLP to 128 with zero weights.
#include "gpu_swin.h"
#include "gpu_common.hpp"
#include "gpu_wmma.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace gpu_swin {
namespace {

using namespace gpu_common;
using namespace gpu_wmma;

constexpr int kC = 60, kHeads = 6, kHd = 10, kWin = 8, kTok = 64, kHid = 120, kStages = 4, kDepth = 6;
constexpr float kLnEps = 1e-5f;
constexpr int kRel = 2 * kWin - 1;    // relative offsets per axis (15)
constexpr int kXS = 72;               // LDS row stride (halves) of 64-channel rows
constexpr int kVS = 72;               // LDS row stride of V^T rows (64 tokens)
constexpr int kHS = 136;              // LDS row stride of MLP hidden rows (128)
constexpr int kThreads = 128;         // four waves of 32; a wave owns 16 tokens / pixels

struct Img {
    int R, C, wpr, nwin;   // size, 8x8 blocks per row, blocks
    int tok;               // token (pixel) offset of the image in the chunk buffers
    int pad;
    long long lr, hr;      // pixel offsets in the caller's LR / HR byte buffers
};

// Weights of one layer: fp16 B fragments (see pack()) and fp32 vectors, padded.
struct LayerW {
    const half_t *qkv, *o, *f1, *f2;
    const float *qkvb, *ob, *f1b, *f2b, *scale, *relt, *lnbw, *lnbb, *lnaw, *lnab;
};

struct Model {
    DevBuf<float> fbuf;
    DevBuf<half_t> hbuf;
    const float *mean, *fw, *fb, *eb, *elw, *elb;
    const half_t* ew;
    LayerW layer[kStages * kDepth];
    const half_t *cw[kStages], *pw[kStages], *abw, *uw;
    const float *cb[kStages], *pb[kStages], *lnw, *lnb, *abb, *ub;
};

std::mutex g_mu;
std::map<std::string, Model*> g_models;
Model* g_cur = nullptr;

// ---- device helpers ----

// B fragment kt of output tile nt from a packed weight matrix with KT k-steps.
__device__ inline h16 ldw(const half_t* W, int KT, int nt, int kt, int r16) { return ld16(W + ((size_t)(nt * KT + kt) * 16 + r16) * 16); }
// Sum / max over the 16 lanes of a half wave (one WMMA output row).
__device__ inline float sum16(float v) {
    v += dpp_xor<1>(v); v += dpp_xor<2>(v); v += dpp_xor<4>(v); v += dpp_xor<8>(v);
    return v;
}
__device__ inline float max16(float v) {
    v = fmaxf(v, dpp_xor<1>(v)); v = fmaxf(v, dpp_xor<2>(v)); v = fmaxf(v, dpp_xor<4>(v)); v = fmaxf(v, dpp_xor<8>(v));
    return v;
}
// GELU, tanh form written as x * sigmoid(2u) (within 5e-4 of the erf form; the result is
// rounded to fp16 for the next product anyway).
__device__ inline float gelu(float x) {
    const float u = 1.5957691216057308f * (x + 0.044715f * x * x * x);
    return x * __builtin_amdgcn_rcpf(1.0f + __expf(-u));
}
// x += LN(o) on the 60 real channels of 16 rows held in WMMA output layout (four 16-column
// tiles; lane column r16, rows 2i + hi).
__device__ inline void ln_add(const f8 (&o)[4], float (&x)[4][8], const float* w, const float* b, int r16) {
    const bool last = r16 < kC - 48;   // tile 3 holds columns 48..63
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float m = sum16(o[0][i] + o[1][i] + o[2][i] + (last ? o[3][i] : 0.0f)) * (1.0f / kC);
        float v = 0.0f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) { const float d = o[t][i] - m; if (t < 3 || last) v += d * d; }
        const float r = __builtin_amdgcn_rsqf(sum16(v) * (1.0f / kC) + kLnEps);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int c = 16 * t + r16;
            if (t < 3 || last) x[t][i] += (o[t][i] - m) * r * w[c] + b[c];
        }
    }
}

// ---- kernels ----

// Stem of one 8x8 pixel block: EMB = conv3x3(rgb/255 - mean) (zero padded per image, fp32),
// OUT = LN(EMB @ We + be) (the patch embedding, on the WMMA units).
__global__ void __launch_bounds__(kThreads) k_stem(const Img* im, const std::uint8_t* LR, const float* mean, const float* fw,
    const float* fb, const half_t* We, const float* eb, const float* lnw, const float* lnb, float* EMB, float* OUT) {
    const Img I = im[blockIdx.y];
    if ((int)blockIdx.x >= I.nwin) return;
    __shared__ float X[3 * 100];             // rgb/255 - mean on the 10x10 halo
    __shared__ float Wf[kC * 27 + kC];       // first conv weights [oc][c][3][3], then its bias
    __shared__ __attribute__((aligned(16))) half_t Eh[kTok * kXS];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int by0 = (blockIdx.x / I.wpr) * kWin, bx0 = (blockIdx.x % I.wpr) * kWin;
    for (int e = threadIdx.x; e < 300; e += kThreads) {
        const int c = e / 100, q = e % 100, py = by0 - 1 + q / 10, px = bx0 - 1 + q % 10;
        X[e] = (py >= 0 && py < I.R && px >= 0 && px < I.C)
            ? (float)LR[3 * (I.lr + (long long)py * I.C + px) + (2 - c)] * (1.0f / 255.0f) - mean[c] : 0.0f;
    }
    for (int e = threadIdx.x; e < kC * 28; e += kThreads) Wf[e] = e < kC * 27 ? fw[e] : fb[e - kC * 27];
    __syncthreads();
    {
        // two threads per pixel, 30 channels each
        const int p = threadIdx.x >> 1, c0 = (threadIdx.x & 1) * 30, py = p / kWin, px = p % kWin;
        float in[27];
        #pragma unroll
        for (int j = 0; j < 27; ++j) in[j] = X[(j / 9) * 100 + (py + (j % 9) / 3) * 10 + px + j % 3];
        float* e = EMB + (size_t)(I.tok + (by0 + py) * I.C + bx0 + px) * kC + c0;
        for (int oc = 0; oc < 30; ++oc) {
            float s = Wf[kC * 27 + c0 + oc];
            #pragma unroll
            for (int j = 0; j < 27; ++j) s += Wf[(c0 + oc) * 27 + j] * in[j];
            e[oc] = s;
            Eh[p * kXS + c0 + oc] = (half_t)s;
        }
        if (c0) for (int c = kC; c < 64; ++c) Eh[p * kXS + c] = (half_t)0.0f;
    }
    __syncthreads();
    f8 o[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) o[t] = splat(eb[t * 16 + r16]);
    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        const h16 a = ld16(Eh + (16 * w + r16) * kXS + kt * 16);
        #pragma unroll
        for (int t = 0; t < 4; ++t) o[t] = mma(a, ldw(We, 4, t, kt, r16), o[t]);
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) pin(o[t]);
    float x[4][8] = {};
    ln_add(o, x, lnw, lnb, r16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = 16 * w + 2 * i + hi;
        float* hp = OUT + (size_t)(I.tok + (by0 + q / kWin) * I.C + bx0 + q % kWin) * kC;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int c = 16 * t + r16;
            if (c < kC) hp[c] = x[t][i];
        }
    }
}

__device__ inline void layer_norm60(float* v, const float* w, const float* b) {
    float m = 0.0f;
    for (int c = 0; c < kC; ++c) m += v[c];
    m *= (1.0f / kC);
    float var = 0.0f;
    for (int c = 0; c < kC; ++c) { const float d = v[c] - m; var += d * d; }
    var *= (1.0f / kC);
    const float r = 1.0f / sqrtf(var + kLnEps);
    for (int c = 0; c < kC; ++c) v[c] = (v[c] - m) * r * w[c] + b[c];
}

// Shift-mask region (0, 1, 2) of a coordinate in the shifted frame of an n-wide axis.
__device__ inline int region(int v, int n) { return v < n - kWin ? 0 : (v < n - kWin / 2 ? 1 : 2); }

// Visibility of a wave's own LDS writes to its other lanes (no workgroup barrier needed).
__device__ inline void wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// One Swin layer on one window (blockIdx.x = window, blockIdx.y = image): reads the tokens from
// IN, writes them to OUT (may be IN: a window touches only its own pixels). Wave w owns tokens
// 16w..16w+15. (Splitting the heads into groups to cut LDS and raise occupancy measured slower:
// the extra barriers cost more.)
__global__ void __launch_bounds__(kThreads) k_layer(const Img* im, const float* IN, float* OUT, LayerW L, int shift) {
    const Img I = im[blockIdx.y];
    if ((int)blockIdx.x >= I.nwin) return;
    // XP: fp16 h (A operand, own rows per wave), aliased by the softmax rows of each wave
    // QKV: q | k | v^T of every head, later the MLP hidden rows of each wave
    __shared__ __attribute__((aligned(16))) half_t XP[kTok * kXS];
    __shared__ __attribute__((aligned(16))) half_t QKV[2 * kHeads * kTok * 16 + kHeads * 16 * kVS];
    __shared__ float RT[kHeads * kRel * kRel];
    __shared__ float NRM[2 * kHeads * kTok];   // 1 / |q| * scale, then 1 / |k|, per head and token
    half_t* Qs = QKV;
    half_t* Ks = QKV + kHeads * kTok * 16;
    half_t* Vt = QKV + 2 * kHeads * kTok * 16;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int wy0 = (blockIdx.x / I.wpr) * kWin, wx0 = (blockIdx.x % I.wpr) * kWin;
    auto src_pixel = [&](int tok) {
        const int y = (wy0 + tok / kWin + shift) % I.R, x = (wx0 + tok % kWin + shift) % I.C;
        return (size_t)(I.tok + y * I.C + x);
    };
    const int arow = 16 * w + r16;   // token of this lane's A-operand row
    // only windows in the last row / column of the shifted frame mix mask regions
    const bool masked = shift && (wy0 + kWin >= I.R || wx0 + kWin >= I.C);
    for (int e = threadIdx.x; e < kHeads * kRel * kRel; e += kThreads) RT[e] = L.relt[e];

    // residual stream of this wave's 16 tokens, in output layout
    float x[4][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 16 * w + 2 * i + hi;
        const float* hp = IN + src_pixel(row) * kC;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int c = 16 * t + r16;
            x[t][i] = c < kC ? hp[c] : 0.0f;
            XP[row * kXS + c] = (half_t)x[t][i];
        }
    }
    wave_sync();
    h16 a[4];
    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = ld16(XP + arow * kXS + kt * 16);

    f8 o[4];   // attention projection, accumulated head by head
    #pragma unroll
    for (int t = 0; t < 4; ++t) o[t] = splat(L.ob[t * 16 + r16]);
    half_t* P = XP + w * 16 * kXS;   // the wave's softmax rows (its h rows are in a[] now)
    // q | k | v as 12 column tiles (64 columns each, head h at columns 10h..10h+9), scattered into
    // the per-head layouts padded to 16 (the pads zeroed here); q and k are stored raw and their
    // norms applied to the scores
    for (int e = threadIdx.x; e < 2 * kHeads * kTok; e += kThreads)
        for (int d = kHd; d < 16; ++d) QKV[e * 16 + d] = (half_t)0.0f;
    for (int e = threadIdx.x; e < kHeads * (16 - kHd) * kTok; e += kThreads)
        Vt[((e / ((16 - kHd) * kTok)) * 16 + kHd + (e / kTok) % (16 - kHd)) * kVS + e % kTok] = (half_t)0.0f;
    for (int nt = 0; nt < 12; ++nt) {
        f8 c = splat(L.qkvb[nt * 16 + r16]);
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) c = mma(a[kt], ldw(L.qkv, 4, nt, kt, r16), c);
        pin(c);
        const int which = nt / 4, cc = (nt % 4) * 16 + r16, h = cc / kHd, d = cc % kHd;
        if (cc < kC) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 16 * w + 2 * i + hi;
                if (which == 2) Vt[(h * 16 + d) * kVS + row] = (half_t)c[i];
                else (which ? Ks : Qs)[(h * kTok + row) * 16 + d] = (half_t)c[i];
            }
        }
    }
    __syncthreads();
    // 1 / |q| (times the head's logit scale) and 1 / |k| per head and token
    for (int e = threadIdx.x; e < 2 * kHeads * kTok; e += kThreads) {
        float n = 0.0f;
        for (int d = 0; d < kHd; ++d) { const float v = (float)QKV[e * 16 + d]; n += v * v; }
        NRM[e] = __builtin_amdgcn_rsqf(fmaxf(n, 1e-24f)) * (e < kHeads * kTok ? L.scale[e / kTok] : 1.0f);
    }
    __syncthreads();
    // attention per head; the context goes through the wave's own q rows into the projection
    for (int h = 0; h < kHeads; ++h) {
        const h16 aq = ld16(Qs + (h * kTok + arow) * 16);
        f8 s[4];
        #pragma unroll
        for (int jt = 0; jt < 4; ++jt) { s[jt] = mma(aq, ld16(Ks + (h * kTok + jt * 16 + r16) * 16), splat(0.0f)); pin(s[jt]); }
        const float* T = RT + h * kRel * kRel;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = 16 * w + 2 * i + hi, qy = q / kWin, qx = q % kWin;
            const int rq = masked ? region(wy0 + qy, I.R) * 3 + region(wx0 + qx, I.C) : 0;
            const float nq = NRM[h * kTok + q];
            float m = -3.0e38f;
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const int k = jt * 16 + r16, ky = k / kWin, kx = k % kWin;
                float v = s[jt][i] * nq * NRM[(kHeads + h) * kTok + k] + T[(qy - ky + kWin - 1) * kRel + (qx - kx + kWin - 1)];
                if (masked && rq != region(wy0 + ky, I.R) * 3 + region(wx0 + kx, I.C)) v -= 200.0f;   // the reference adds its -100 mask twice
                s[jt][i] = v;
                m = fmaxf(m, v);
            }
            m = max16(m);
            float sum = 0.0f;
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) { s[jt][i] = __expf(s[jt][i] - m); sum += s[jt][i]; }
            const float inv = __builtin_amdgcn_rcpf(sum16(sum));
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) P[(2 * i + hi) * kXS + jt * 16 + r16] = (half_t)(s[jt][i] * inv);
        }
        wave_sync();
        f8 cx = splat(0.0f);
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) cx = mma(ld16(P + r16 * kXS + kt * 16), ld16(Vt + (h * 16 + r16) * kVS + kt * 16), cx);
        pin(cx);
        #pragma unroll
        for (int i = 0; i < 8; ++i) Qs[(h * kTok + 16 * w + 2 * i + hi) * 16 + r16] = (half_t)cx[i];
        wave_sync();
        const h16 ac = ld16(Qs + (h * kTok + arow) * 16);
        #pragma unroll
        for (int t = 0; t < 4; ++t) o[t] = mma(ac, ldw(L.o, kHeads, t, h, r16), o[t]);
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) pin(o[t]);
    ln_add(o, x, L.lnbw, L.lnbb, r16);   // h += LN(proj)
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int t = 0; t < 4; ++t) XP[(16 * w + 2 * i + hi) * kXS + 16 * t + r16] = (half_t)x[t][i];
    __syncthreads();   // every wave is done with k / v^T before the hidden rows reuse them

    // MLP: hidden = gelu(h W1 + b1); h += LN(hidden W2 + b2)
    half_t* HID = QKV + w * 16 * kHS;
    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = ld16(XP + arow * kXS + kt * 16);
    for (int nt = 0; nt < 8; nt += 2) {
        f8 c[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            c[u] = splat(L.f1b[(nt + u) * 16 + r16]);
            #pragma unroll
            for (int kt = 0; kt < 4; ++kt) c[u] = mma(a[kt], ldw(L.f1, 4, nt + u, kt, r16), c[u]);
            pin(c[u]);
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                HID[(2 * i + hi) * kHS + (nt + u) * 16 + r16] = (half_t)gelu(c[u][i]);
    }
    wave_sync();
    #pragma unroll
    for (int t = 0; t < 4; ++t) o[t] = splat(L.f2b[t * 16 + r16]);
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        const h16 ah = ld16(HID + r16 * kHS + kt * 16);
        #pragma unroll
        for (int t = 0; t < 4; ++t) o[t] = mma(ah, ldw(L.f2, 8, t, kt, r16), o[t]);
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) pin(o[t]);
    ln_add(o, x, L.lnaw, L.lnab, r16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float* hp = OUT + src_pixel(16 * w + 2 * i + hi) * kC;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int c = 16 * t + r16;
            if (c < kC) hp[c] = x[t][i];
        }
    }
}

// conv3x3 (60 -> 16*NT channels, zero padded per image) of one 8x8 pixel block, as an implicit
// GEMM over a 10x10 fp16 halo in LDS (K = 9 taps x 64 channels). Input optionally layer-normed
// first (lnw). Epilogues:
//   MODE 0: OUT = conv + RES (RES may be null)
//   MODE 1: OUT = conv1x1(conv) + RES (the stage end; OUT may be RES)
//   MODE 2: pixel shuffle x2 + mean, to 8-bit BGR at the image's HR offset (NT = 1, 12 channels)
template <int NT, int MODE>
__global__ void __launch_bounds__(kThreads) k_conv(const Img* im, const float* IN, const float* lnw, const float* lnb,
    const half_t* Wc, const float* bc, const half_t* Wp, const float* bp, const float* RES, float* OUT,
    const float* mean, std::uint8_t* HR) {
    const Img I = im[blockIdx.y];
    if ((int)blockIdx.x >= I.nwin) return;
    __shared__ __attribute__((aligned(16))) half_t Pt[100 * kXS];
    __shared__ __attribute__((aligned(16))) half_t Ct[MODE == 1 ? 4 * 16 * kXS : 8];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int by0 = (blockIdx.x / I.wpr) * kWin, bx0 = (blockIdx.x % I.wpr) * kWin;
    if (lnw) {
        if (threadIdx.x < 100) {
            const int py = by0 - 1 + threadIdx.x / 10, px = bx0 - 1 + threadIdx.x % 10;
            half_t* d = Pt + threadIdx.x * kXS;
            if (py >= 0 && py < I.R && px >= 0 && px < I.C) {
                float v[kC];
                const float* s = IN + (size_t)(I.tok + py * I.C + px) * kC;
                for (int c = 0; c < kC; ++c) v[c] = s[c];
                layer_norm60(v, lnw, lnb);
                for (int c = 0; c < kC; ++c) d[c] = (half_t)v[c];
            } else {
                for (int c = 0; c < kC; ++c) d[c] = (half_t)0.0f;
            }
            for (int c = kC; c < 64; ++c) d[c] = (half_t)0.0f;
        }
    } else {
        for (int e = threadIdx.x; e < 100 * 64; e += kThreads) {
            const int q = e / 64, c = e % 64, py = by0 - 1 + q / 10, px = bx0 - 1 + q % 10;
            float v = 0.0f;
            if (c < kC && py >= 0 && py < I.R && px >= 0 && px < I.C) v = IN[(size_t)(I.tok + py * I.C + px) * kC + c];
            Pt[q * kXS + c] = (half_t)v;
        }
    }
    __syncthreads();
    const int p = 16 * w + r16, py = p / kWin, px = p % kWin;
    f8 acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = splat(bc[nt * 16 + r16]);
    for (int tap = 0; tap < 9; ++tap) {
        const half_t* a = Pt + ((py + tap / 3) * 10 + px + tap % 3) * kXS;
        #pragma unroll
        for (int ch = 0; ch < 4; ++ch) {
            const h16 av = ld16(a + ch * 16);
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[nt] = mma(av, ldw(Wc, 36, nt, tap * 4 + ch, r16), acc[nt]);
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) pin(acc[nt]);
    if constexpr (MODE == 1) {
        half_t* C = Ct + w * 16 * kXS;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) C[(2 * i + hi) * kXS + nt * 16 + r16] = (half_t)acc[nt][i];
        __syncthreads();
        f8 o[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) o[t] = splat(bp[t * 16 + r16]);
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const h16 av = ld16(C + r16 * kXS + kt * 16);
            #pragma unroll
            for (int t = 0; t < 4; ++t) o[t] = mma(av, ldw(Wp, 4, t, kt, r16), o[t]);
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) pin(o[t]);
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc[t] = o[t];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = 16 * w + 2 * i + hi, y = by0 + q / kWin, x = bx0 + q % kWin;
        if constexpr (MODE == 2) {
            if (r16 < 12) {
                const int c = r16 / 4, ii = (r16 / 2) % 2, jj = r16 % 2;
                int v = __float2int_rn((acc[0][i] + mean[c]) * 255.0f);
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                HR[3 * (I.hr + (long long)(2 * y + ii) * (2 * I.C) + 2 * x + jj) + (2 - c)] = (std::uint8_t)v;
            }
        } else {
            const size_t o = (size_t)(I.tok + y * I.C + x) * kC;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int c = nt * 16 + r16;
                if (c < kC) OUT[o + c] = acc[nt][i] + (RES ? RES[o + c] : 0.0f);
            }
        }
    }
}

// ---- host ----

// Packs W(k, n) (K x N, zero outside) as WMMA B fragments: tile nt, k-step kt, lane column j,
// 16 k values; matches ldw().
template <class F>
size_t pack(std::vector<std::uint16_t>& out, int K, int N, F W) {
    const size_t at = out.size();
    for (int nt = 0; nt < N / 16; ++nt)
        for (int kt = 0; kt < K / 16; ++kt)
            for (int j = 0; j < 16; ++j)
                for (int e = 0; e < 16; ++e) out.push_back(to_half(W(kt * 16 + e, nt * 16 + j)));
    return at;
}

}  // namespace

std::string weights_path(const std::string& onnx_model) {
    const size_t dot = onnx_model.find_last_of('.'), slash = onnx_model.find_last_of("/\\");
    const std::string stem = (dot != std::string::npos && (slash == std::string::npos || dot > slash)) ? onnx_model.substr(0, dot) : onnx_model;
    return stem + ".swin2sr.bin";
}

bool supported(int rows, int cols) { return rows >= kWin && cols >= kWin && rows % kWin == 0 && cols % kWin == 0; }

bool load(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_models.find(path);
    if (it != g_models.end()) { g_cur = it->second; return g_cur != nullptr; }
    g_models[path] = nullptr;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::uint32_t hdr[8] = {};
    std::vector<float> w;
    bool ok = std::fread(hdr, 4, 8, f) == 8 && hdr[0] == 0x53325753u && hdr[1] == 1 && hdr[2] == (unsigned)kC &&
        hdr[3] == (unsigned)kStages && hdr[4] == (unsigned)kDepth && hdr[5] == (unsigned)kHeads && hdr[6] == (unsigned)kWin &&
        hdr[7] == (unsigned)kHid;
    if (ok) {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 32, SEEK_SET);
        w.resize((size_t)(sz - 32) / 4);
        ok = std::fread(w.data(), 4, w.size(), f) == w.size();
    }
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "gpu_swin: %s is not a Swin2SR weight file (re-run scripts/export_swin2sr_weights.py)\n", path.c_str()); return false; }

    // Walk the file (order of export_swin2sr_weights.py) and build the device layout: fp32
    // vectors (padded) in fv, fp16 fragments in hv. Pointers are fixed up after the upload.
    size_t at = 0;
    auto next = [&](size_t n) { const float* p = w.data() + at; at += n; return p; };
    std::vector<float> fv;
    std::vector<std::uint16_t> hv;
    auto vec = [&](const float* p, int n, int padded) {
        const size_t o = fv.size();
        for (int i = 0; i < padded; ++i) fv.push_back(i < n ? p[i] : 0.0f);
        return o;
    };
    struct LayerO { size_t qkv, o, f1, f2, qkvb, ob, f1b, f2b, scale, relt, lnbw, lnbb, lnaw, lnab; } lo[kStages * kDepth];
    size_t cw[kStages], cb[kStages], pw[kStages], pb[kStages];
    const size_t mean = vec(next(3), 3, 3), fw = vec(next(kC * 27), kC * 27, kC * 27), fb = vec(next(kC), kC, kC);
    const float* ewp = next(kC * kC);   // [in][out]
    const size_t ew = pack(hv, 64, 64, [&](int k, int n) { return (k < kC && n < kC) ? ewp[(size_t)k * kC + n] : 0.0f; });
    const size_t eb = vec(next(kC), kC, 64);
    const size_t elw = vec(next(kC), kC, kC), elb = vec(next(kC), kC, kC);
    auto conv_frag = [&](const float* cwp, int nout, int npad) {   // [oc][ic][3][3] -> K = tap*64 + ic
        return pack(hv, 9 * 64, npad, [&](int k, int n) {
            const int tap = k / 64, ic = k % 64;
            return (ic < kC && n < nout) ? cwp[((size_t)n * kC + ic) * 9 + tap] : 0.0f;
        });
    };
    for (int s = 0; s < kStages; ++s) {
        for (int l = 0; l < kDepth; ++l) {
            LayerO& L = lo[s * kDepth + l];
            const float *qw = next(kC * kC), *qb = next(kC), *kw = next(kC * kC), *vw = next(kC * kC), *vb = next(kC);
            const float *scale = next(kHeads), *relb = next((size_t)kHeads * kTok * kTok);
            const float *ow = next(kC * kC), *ob = next(kC), *lnbw = next(kC), *lnbb = next(kC);
            const float *f1w = next(kC * kHid), *f1b = next(kHid), *f2w = next(kHid * kC), *f2b = next(kC);
            const float *lnaw = next(kC), *lnab = next(kC);
            // q | k | v columns: 64 each (60 used, head h at 10h..10h+9)
            L.qkv = pack(hv, 64, 3 * 64, [&](int k, int n) {
                const int which = n / 64, c = n % 64;
                const float* m = which == 0 ? qw : (which == 1 ? kw : vw);
                return (k < kC && c < kC) ? m[(size_t)k * kC + c] : 0.0f;
            });
            L.qkvb = fv.size();
            for (int n = 0; n < 3 * 64; ++n) {
                const int which = n / 64, c = n % 64;
                fv.push_back(c < kC && which != 1 ? (which == 0 ? qb : vb)[c] : 0.0f);
            }
            L.o = pack(hv, kHeads * 16, 64, [&](int k, int n) {
                const int h = k / 16, d = k % 16;
                return (d < kHd && n < kC) ? ow[(size_t)(h * kHd + d) * kC + n] : 0.0f;
            });
            L.f1 = pack(hv, 64, 128, [&](int k, int n) { return (k < kC && n < kHid) ? f1w[(size_t)k * kHid + n] : 0.0f; });
            L.f2 = pack(hv, 128, 64, [&](int k, int n) { return (k < kHid && n < kC) ? f2w[(size_t)k * kC + n] : 0.0f; });
            L.ob = vec(ob, kC, 64); L.f1b = vec(f1b, kHid, 128); L.f2b = vec(f2b, kC, 64); L.scale = vec(scale, kHeads, kHeads);
            // relative-position bias as a 15 x 15 table per head (relb[h][q][k] depends on q - k only)
            L.relt = fv.size();
            fv.resize(fv.size() + (size_t)kHeads * kRel * kRel, 0.0f);
            for (int h = 0; h < kHeads; ++h)
                for (int q = 0; q < kTok; ++q)
                    for (int k = 0; k < kTok; ++k)
                        fv[L.relt + (size_t)h * kRel * kRel + (q / kWin - k / kWin + kWin - 1) * kRel + (q % kWin - k % kWin + kWin - 1)] =
                            relb[((size_t)h * kTok + q) * kTok + k];
            L.lnbw = vec(lnbw, kC, kC); L.lnbb = vec(lnbb, kC, kC); L.lnaw = vec(lnaw, kC, kC); L.lnab = vec(lnab, kC, kC);
        }
        const float *cwp = next((size_t)kC * kC * 9), *cbp = next(kC), *pwp = next(kC * kC), *pbp = next(kC);
        cw[s] = conv_frag(cwp, kC, 64); cb[s] = vec(cbp, kC, 64);
        pw[s] = pack(hv, 64, 64, [&](int k, int n) { return (k < kC && n < kC) ? pwp[(size_t)k * kC + n] : 0.0f; });
        pb[s] = vec(pbp, kC, 64);
    }
    const size_t lnw = vec(next(kC), kC, kC), lnb = vec(next(kC), kC, kC);
    const float *abwp = next((size_t)kC * kC * 9), *abbp = next(kC), *uwp = next((size_t)12 * kC * 9), *ubp = next(12);
    const size_t abw = conv_frag(abwp, kC, 64), abb = vec(abbp, kC, 64), uw = conv_frag(uwp, 12, 16), ub = vec(ubp, 12, 16);
    auto* m = new Model();
    if (at != w.size() || !m->fbuf.alloc(fv.size()) || !m->hbuf.alloc(hv.size()) ||
        hipMemcpy(m->fbuf.p, fv.data(), fv.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(m->hbuf.p, hv.data(), hv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "gpu_swin: %s has an unexpected size\n", path.c_str());
        delete m;
        return false;
    }
    const float* F = m->fbuf.p;
    const half_t* Hh = m->hbuf.p;
    m->mean = F + mean; m->fw = F + fw; m->fb = F + fb; m->ew = Hh + ew; m->eb = F + eb; m->elw = F + elw; m->elb = F + elb;
    for (int l = 0; l < kStages * kDepth; ++l) {
        const LayerO& o = lo[l];
        m->layer[l] = LayerW{ Hh + o.qkv, Hh + o.o, Hh + o.f1, Hh + o.f2, F + o.qkvb, F + o.ob, F + o.f1b, F + o.f2b,
            F + o.scale, F + o.relt, F + o.lnbw, F + o.lnbb, F + o.lnaw, F + o.lnab };
    }
    for (int s = 0; s < kStages; ++s) { m->cw[s] = Hh + cw[s]; m->cb[s] = F + cb[s]; m->pw[s] = Hh + pw[s]; m->pb[s] = F + pb[s]; }
    m->lnw = F + lnw; m->lnb = F + lnb; m->abw = Hh + abw; m->abb = F + abb; m->uw = Hh + uw; m->ub = F + ub;
    g_models[path] = m;
    g_cur = m;
    return true;
}

bool run(const std::vector<Image>& images, const std::uint8_t* d_lr, std::uint8_t* d_hr) {
    Model* m = g_cur;
    if (!m) return false;
    for (const Image& im : images) if (!supported(im.rows, im.cols)) return false;
    // Chunks of up to ~2 M pixels: three 60-channel float buffers of 480 MB each.
    const long long kChunkPx = 2000000;
    size_t first = 0;
    while (first < images.size()) {
        std::vector<Img> ch;
        long long px = 0;
        int max_win = 0;
        for (size_t i = first; i < images.size(); ++i) {
            const Image& s = images[i];
            const long long n = (long long)s.rows * s.cols;
            if (!ch.empty() && px + n > kChunkPx) break;
            Img I{};
            I.R = s.rows; I.C = s.cols; I.wpr = s.cols / kWin; I.nwin = (s.rows / kWin) * I.wpr;
            I.tok = (int)px; I.lr = s.lr_off; I.hr = s.hr_off;
            ch.push_back(I);
            px += n;
            max_win = std::max(max_win, I.nwin);
        }
        first += ch.size();
        const unsigned ni = (unsigned)ch.size();
        DevBuf<Img> dim; DevBuf<float> dH, dR, dE;
        if (!dim.alloc(ch.size()) || !dH.alloc((size_t)px * kC) || !dR.alloc((size_t)px * kC) || !dE.alloc((size_t)px * kC)) {
            std::fprintf(stderr, "gpu_swin: device allocation failed\n");
            return false;
        }
        HIP_TRY(hipMemcpy(dim.p, ch.data(), ch.size() * sizeof(Img), hipMemcpyHostToDevice));
        const dim3 gwin((unsigned)max_win, ni);
        k_stem<<<gwin, dim3(kThreads)>>>(dim.p, d_lr, m->mean, m->fw, m->fb, m->ew, m->eb, m->elw, m->elb, dE.p, dR.p);
        for (int s = 0; s < kStages; ++s) {
            // R holds the stage input; the layers run on H; the stage end writes R again
            for (int l = 0; l < kDepth; ++l)
                k_layer<<<gwin, dim3(kThreads)>>>(dim.p, l == 0 ? dR.p : dH.p, dH.p, m->layer[s * kDepth + l], (l % 2) ? kWin / 2 : 0);
            k_conv<4, 1><<<gwin, dim3(kThreads)>>>(dim.p, dH.p, nullptr, nullptr, m->cw[s], m->cb[s], m->pw[s], m->pb[s],
                dR.p, dR.p, nullptr, nullptr);
        }
        k_conv<4, 0><<<gwin, dim3(kThreads)>>>(dim.p, dR.p, m->lnw, m->lnb, m->abw, m->abb, nullptr, nullptr, dE.p, dH.p,
            nullptr, nullptr);
        k_conv<1, 2><<<gwin, dim3(kThreads)>>>(dim.p, dH.p, nullptr, nullptr, m->uw, m->ub, nullptr, nullptr, nullptr, nullptr,
            m->mean, d_hr);
        HIP_TRY(hipGetLastError());
        HIP_TRY(hipDeviceSynchronize());
    }
    return true;
}

}  // namespace gpu_swin
