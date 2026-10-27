// Segment boundary calm on HIP (see gpu_calm.h).
//
// Per image pixel p with label c (labels nearest-upscaled from the LR grid):
//   gate(p) = clamp(1 - d(p) / overlap, 0, 1) if c is a low-density class, else 0, where d(p) is the Euclidean distance
//             from p to the nearest image pixel with another label (the image border is not a frontier);
//   out(p)  = f(p) + gate(p) * (bilateral(f)(p) - f(p)), rounded to 8 bit (round half to even, as convertTo).
// bilateral = cv::bilateralFilter(f, 13, 18, 4, BORDER_REPLICATE) on float BGR: circular window of radius 6, spatial
// weight exp(-r^2 / (2 * 4^2)), colour weight exp(-s^2 / (2 * 18^2)) with s the sum of the three channel differences.
// d(p) is searched over the LR cells within `overlap` of p: each cell covers a block of image pixels, the nearest of
// which is a clamp, so a 2x image costs about a quarter of a per-pixel search.
#include "gpu_calm.h"
#include "gpu_common.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>

namespace gpu_calm {
namespace {

using namespace gpu_common;

constexpr int kE = 256;
constexpr int kBR = 6;                         // bilateral radius (diameter 13)
constexpr float kSigS = 4.0f, kSigC = 18.0f;

struct CalmP {
    const std::uint8_t* in; std::uint8_t* out; int H, W;
    const std::uint8_t* lab; int h, w;
    const std::uint8_t* low;                    // 256 flags
    int overlap;
};

__device__ inline int lab_row(int y, const CalmP& P) { return (int)(((long long)y * P.h) / P.H); }
__device__ inline int lab_col(int x, const CalmP& P) { return (int)(((long long)x * P.w) / P.W); }
// First image row / col that maps to LR row / col i (inverse of the floor mapping above).
__device__ inline int first_row(int i, const CalmP& P) { return (int)(((long long)i * P.H + P.h - 1) / P.h); }
__device__ inline int first_col(int j, const CalmP& P) { return (int)(((long long)j * P.W + P.w - 1) / P.w); }

__global__ void k_calm(const CalmP P) {
    const int t = blockIdx.x * kE + threadIdx.x;
    if (t >= P.H * P.W) return;
    const int y = t / P.W, x = t % P.W;
    const std::uint8_t* src = P.in + (size_t)t * 3;
    std::uint8_t* dst = P.out + (size_t)t * 3;
    const int ly = lab_row(y, P), lx = lab_col(x, P);
    const int c = P.lab[(size_t)ly * P.w + lx];
    if (!P.low[c]) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; return; }
    // nearest other-label pixel within `overlap`
    const float R = (float)P.overlap;
    float best2 = R * R;
    const int i0 = max(0, lab_row(max(0, y - P.overlap), P)), i1 = min(P.h - 1, lab_row(min(P.H - 1, y + P.overlap), P));
    const int j0 = max(0, lab_col(max(0, x - P.overlap), P)), j1 = min(P.w - 1, lab_col(min(P.W - 1, x + P.overlap), P));
    for (int i = i0; i <= i1; ++i) {
        const int r0 = first_row(i, P), r1 = first_row(i + 1, P) - 1;
        const float dy = (float)(y < r0 ? r0 - y : (y > r1 ? y - r1 : 0));
        if (dy * dy >= best2) continue;
        const std::uint8_t* lr = P.lab + (size_t)i * P.w;
        for (int j = j0; j <= j1; ++j) {
            if (lr[j] == c) continue;
            const int c0 = first_col(j, P), c1 = first_col(j + 1, P) - 1;
            const float dx = (float)(x < c0 ? c0 - x : (x > c1 ? x - c1 : 0));
            best2 = fminf(best2, dy * dy + dx * dx);
        }
    }
    const float gate = fminf(1.0f, fmaxf(0.0f, 1.0f - sqrtf(best2) / R));
    if (gate <= 0.0f) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; return; }
    // bilateral at p only
    const float b0 = src[0], g0 = src[1], r0 = src[2];
    float sw = 0.0f, sb = 0.0f, sg = 0.0f, sr = 0.0f;
    const float cs = -0.5f / (kSigS * kSigS), cc = -0.5f / (kSigC * kSigC);
    for (int dy = -kBR; dy <= kBR; ++dy) {
        const int yy = min(max(y + dy, 0), P.H - 1);
        for (int dx = -kBR; dx <= kBR; ++dx) {
            const int rr2 = dy * dy + dx * dx;
            if (rr2 > kBR * kBR) continue;
            const std::uint8_t* q = P.in + ((size_t)yy * P.W + min(max(x + dx, 0), P.W - 1)) * 3;
            const float s = fabsf(q[0] - b0) + fabsf(q[1] - g0) + fabsf(q[2] - r0);
            const float wgt = __expf(rr2 * cs + s * s * cc);
            sw += wgt; sb += wgt * q[0]; sg += wgt * q[1]; sr += wgt * q[2];
        }
    }
    const float o[3] = { b0 + gate * (sb / sw - b0), g0 + gate * (sg / sw - g0), r0 + gate * (sr / sw - r0) };
    for (int k = 0; k < 3; ++k) dst[k] = (std::uint8_t)min(255, max(0, __float2int_rn(o[k])));
}

}  // namespace

bool run(std::uint8_t* img, int H, int W, const std::uint8_t* labels, int h, int w, const bool low[256], int overlap,
    double* ms) {
    if (!img || !labels || H < 1 || W < 1 || h < 1 || w < 1 || overlap <= 0) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = (size_t)H * W;
    DevBuf<std::uint8_t> din, dout, dlab, dlow;
    if (!din.alloc(n * 3) || !dout.alloc(n * 3) || !dlab.alloc((size_t)h * w) || !dlow.alloc(256)) {
        std::fprintf(stderr, "gpu_calm: device allocation failed\n");
        return false;
    }
    std::uint8_t lf[256];
    for (int c = 0; c < 256; ++c) lf[c] = low[c] ? 1 : 0;
    HIP_TRY(hipMemcpy(din.p, img, n * 3, hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dlab.p, labels, (size_t)h * w, hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dlow.p, lf, 256, hipMemcpyHostToDevice));
    const CalmP P{ din.p, dout.p, H, W, dlab.p, h, w, dlow.p, overlap };
    k_calm<<<dim3((unsigned)((n + kE - 1) / kE)), kE>>>(P);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipMemcpy(img, dout.p, n * 3, hipMemcpyDeviceToHost));
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace gpu_calm
