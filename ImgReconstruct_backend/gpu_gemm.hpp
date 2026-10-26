#pragma once
// LDS-tiled batched GEMM for the dense per-tile DCTs (HIP; include from .hip
// files after gpu_common.hpp).
//
// One launch computes OUT[M x N] = A[M x K] * B[K x N] for every tile of a
// batch; blockIdx.y is the tile, blockIdx.x enumerates 32x32 output blocks.
// Operands are either tile data (row-major, tile width = cols) or the tile's
// orthonormal DCT table (rows x rows for the left side, cols x cols for the
// right side), read normally or transposed, which covers all four transforms:
//   inverse pass A  TMP = Y   * Cc        A=DATA B=TAB
//   inverse pass B  PIX = Cr^T * TMP      A=TABT B=DATA
//   forward pass A  TMP = SC  * Cc^T      A=DATA B=TABT
//   forward pass B  G   = Cr  * TMP       A=TAB  B=DATA
// Each of the 64 threads of a block owns a 4x4 register block, so a 2x2
// neighbourhood of the output never straddles two threads; the epilogues use
// that to fuse the neighbouring elementwise steps of the FISTA iteration.
#include "gpu_common.hpp"

namespace gpu_common {
namespace {

enum Operand { OP_DATA = 0, OP_TAB = 1, OP_TABT = 2 };
enum Epilogue {
    EPI_STORE = 0,   // OUT[m][n] = acc
    EPI_RESID = 1,   // acc is the HR image: write the 2x2-average data residual (cs-sr)
    EPI_STEP = 2     // acc is the gradient DCT: soft-threshold + FISTA momentum (cs-sr)
};

struct Epi {
    const float* blr;     // EPI_RESID: LR sample grid per tile (rows/2 x cols/2), < -1e29 = no sample
    const float* hav;     // EPI_RESID: exact-HR anchor values (tile layout, < -1e29 = none), or null
    float hw;             // EPI_RESID: anchor data-term weight
    const float* anch;    // EPI_STEP
    const float* ws;      // frequency weights, or null: computed in place from logw (as k_ws)
    float logw;
    const float* lam;     // per-tile L1 weight
    float* y; float* xp; float* zc;   // zc may be null (the new iterate is also in xp)
    float aw, step, mom;
    const int* hrt;       // EPI_STEP: per-tile 1 = tile has HR anchors (takes step_hr), or null
    float step_hr;
};

constexpr int kBM = 32, kBN = 32, kBK = 16, kGemmThreads = 64;

// Epilogue arithmetic shared by k_gemm and k_gemm2, so fused and unfused passes compute
// identical values.
// EPI_RESID on one 2x2 block p00..p11 of the HR image: the 2x2-average residual at the LR
// sample bs (0 where there is none), the same on every child.
__device__ inline float epi_resid(float p00, float p01, float p10, float p11, float bs) {
    const float avg = 0.25f * (p00 + p01 + p10 + p11);
    return bs > -1e29f ? 0.25f * (avg - bs) : 0.0f;
}
// Exact-HR anchor term hw * (pix - h) added to a residual (h < -1e29: no anchor).
__device__ inline float epi_anchor(float v, float pix, float h, float hw) {
    return h > -1e29f ? v + hw * (pix - h) : v;
}
// Frequency weight of coefficient (r, c) (as k_ws), or 1 when logw == 0.
__device__ inline float epi_ws(int r, int c, int M, int N, float logw) {
    if (logw == 0.0f) return 1.0f;
    const float fr = M > 1 ? (float)r / (float)(M - 1) : 0.0f;
    const float fc = N > 1 ? (float)c / (float)(N - 1) : 0.0f;
    return expf(logw * 0.5f * (fr + fc));
}
// EPI_STEP at one coefficient: gradient 2*acc + coupling, soft threshold, FISTA momentum.
// Returns z; y and xp are updated in place.
__device__ inline float epi_step(float acc, float& y, float& xp, float a, float aw, float step, float lam,
    float ws, float mom) {
    const float g = 2.0f * acc + 2.0f * aw * (y - a);
    const float v = y - step * g - a;
    const float thr = lam * step * ws;
    const float av = fabsf(v);
    const float z = a + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
    y = z + mom * (z - xp);
    xp = z;
    return z;
}

template <int AM, int BM, int EPI>
__global__ void __launch_bounds__(kGemmThreads) k_gemm(const TileDev* td, const float* tab,
    const float* in, float* out, int nbn, Epi ep) {
    const TileDev T = td[blockIdx.y];
    const int M = T.rows, N = T.cols;
    const int bm = blockIdx.x / nbn, bn = blockIdx.x % nbn;
    const int m0 = bm * kBM, n0 = bn * kBN;
    if (m0 >= M || n0 >= N) return;
    const int K = (AM == OP_DATA) ? T.cols : T.rows;
    const float* data = in + T.off;
    const float* rtab = tab + T.rtab;
    const float* ctab = tab + T.ctab;

    __shared__ float As[kBK][kBM + 4];
    __shared__ float Bs[kBK][kBN + 4];
    const int t = threadIdx.x;
    const int ty = t / 8, tx = t % 8;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    constexpr int kPerA = kBM * kBK / kGemmThreads, kPerB = kBK * kBN / kGemmThreads;   // 8 each
    float ra[kPerA], rb[kPerB];
    // Global -> registers for one K chunk. All loads are issued back to back so
    // their latencies overlap; the LDS stores happen after.
    auto load_chunk = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < kPerA; ++q) {
            const int e = t + q * kGemmThreads;
            int mi, ki;
            if (AM == OP_TABT) { mi = e % kBM; ki = e / kBM; } else { mi = e / kBK; ki = e % kBK; }
            const int m = m0 + mi, k = k0 + ki;
            float v = 0.0f;
            if (m < M && k < K) {
                if (AM == OP_DATA) v = data[m * T.cols + k];
                else if (AM == OP_TAB) v = rtab[m * T.rows + k];
                else v = rtab[k * T.rows + m];
            }
            ra[q] = v;
        }
        #pragma unroll
        for (int q = 0; q < kPerB; ++q) {
            const int e = t + q * kGemmThreads;
            int ki, ni;
            if (BM == OP_TABT) { ki = e % kBK; ni = e / kBK; } else { ki = e / kBN; ni = e % kBN; }
            const int k = k0 + ki, n = n0 + ni;
            float v = 0.0f;
            if (k < K && n < N) {
                if (BM == OP_DATA) v = data[k * T.cols + n];
                else if (BM == OP_TAB) v = ctab[k * T.cols + n];
                else v = ctab[n * T.cols + k];
            }
            rb[q] = v;
        }
    };
    load_chunk(0);
    for (int k0 = 0; k0 < K; k0 += kBK) {
        #pragma unroll
        for (int q = 0; q < kPerA; ++q) {
            const int e = t + q * kGemmThreads;
            int mi, ki;
            if (AM == OP_TABT) { mi = e % kBM; ki = e / kBM; } else { mi = e / kBK; ki = e % kBK; }
            As[ki][mi] = ra[q];
        }
        #pragma unroll
        for (int q = 0; q < kPerB; ++q) {
            const int e = t + q * kGemmThreads;
            int ki, ni;
            if (BM == OP_TABT) { ki = e % kBK; ni = e / kBK; } else { ki = e / kBN; ni = e % kBN; }
            Bs[ki][ni] = rb[q];
        }
        __syncthreads();
        if (k0 + kBK < K) load_chunk(k0 + kBK);   // next chunk's loads overlap the FMAs below
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    const int mb = m0 + ty * 4, nb = n0 + tx * 4;
    if (EPI == EPI_STORE) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (mb + i < M && nb + j < N) out[T.off + (mb + i) * N + nb + j] = acc[i][j];
    } else if (EPI == EPI_RESID) {
        const float* blr = ep.blr + T.off / 4;
        #pragma unroll
        for (int ii = 0; ii < 4; ii += 2)
            #pragma unroll
            for (int jj = 0; jj < 4; jj += 2) {
                const int m = mb + ii, n = nb + jj;
                if (m + 1 < M && n + 1 < N) {
                    const float v = epi_resid(acc[ii][jj], acc[ii][jj + 1], acc[ii + 1][jj], acc[ii + 1][jj + 1],
                        blr[(m / 2) * (N / 2) + n / 2]);
                    float* o = out + T.off + m * N + n;
                    o[0] = v; o[1] = v; o[N] = v; o[N + 1] = v;
                    if (ep.hav) {
                        // exact-HR anchors: hw * (pix - value) joins the same forward DCT
                        // (CPU: 2*hw*DCT(scatter(pix[a] - v_a)) added to the gradient)
                        const float* h = ep.hav + T.off + m * N + n;
                        o[0] = epi_anchor(v, acc[ii][jj], h[0], ep.hw);
                        o[1] = epi_anchor(v, acc[ii][jj + 1], h[1], ep.hw);
                        o[N] = epi_anchor(v, acc[ii + 1][jj], h[N], ep.hw);
                        o[N + 1] = epi_anchor(v, acc[ii + 1][jj + 1], h[N + 1], ep.hw);
                    }
                }
            }
    } else {
        const float lam = ep.lam[blockIdx.y];
        const float step = (ep.hrt && ep.hrt[blockIdx.y]) ? ep.step_hr : ep.step;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (mb + i < M && nb + j < N) {
                    const int o = T.off + (mb + i) * N + nb + j;
                    float y = ep.y[o], xp = ep.xp[o];
                    const float ws = ep.ws ? ep.ws[o] : epi_ws(mb + i, nb + j, M, N, ep.logw);
                    const float z = epi_step(acc[i][j], y, xp, ep.anch[o], ep.aw, step, lam, ws, ep.mom);
                    ep.y[o] = y;
                    ep.xp[o] = xp;
                    if (ep.zc) ep.zc[o] = z;
                }
    }
}

// Two chained per-tile products on a stripe of kFM output rows, the intermediate kept in LDS:
//   S   = A1[m0 : m0+kFM, :] * IN        A1 = Cr (OP_TAB) or Cr^T (OP_TABT), K = rows
//   S'  = epilogue(S)                    EPI_RESID: data residual (+ anchors)
//                                        EPI_STEP: FISTA step (y, xp updated); S' = y, or z when feed_z
//   OUT = S'[m0 : m0+kFM, :] * B2        B2 = Cc (OP_TAB) or Cc^T (OP_TABT), K = cols
// A column transform followed by a row transform needs only the block's own rows, so a FISTA
// step is two launches instead of four and its intermediates never reach memory. Both products
// accumulate in ascending k with zero padding to whole kBK chunks, as k_gemm, and the epilogues
// share its arithmetic, so results are bit-identical to the unfused passes.
// Tiles must have cols <= kFN. 128 threads; a thread owns 8 rows x 4 columns.
constexpr int kFM = 32, kFN = 128, kFThreads = 128;
template <int A1, int EPI1, int B2>
__global__ void __launch_bounds__(kFThreads) k_gemm2(const TileDev* td, const float* tab, const float* in,
    float* out, Epi ep, int feed_z) {
    const TileDev T = td[blockIdx.y];
    const int M = T.rows, N = T.cols;
    const int m0 = blockIdx.x * kFM;
    if (m0 >= M) return;
    const float* data = in + T.off;
    const float* rtab = tab + T.rtab;
    const float* ctab = tab + T.ctab;

    __shared__ float As[kBK][kFM + 4];
    __shared__ float Bs[kBK][kFN + 4];
    __shared__ float Ss[kFM][kFN + 4];
    const int t = threadIdx.x, ty = t / 32, tx = t % 32;
    const int rb = ty * 8, cb = tx * 4;   // block-local rows / columns of this thread
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    // ---- product 1, K = M ----
    constexpr int kPA = kFM * kBK / kFThreads, kPB = kBK * kFN / kFThreads;   // 4, 16
    float ra[kPA], rb1[kPB];
    auto load1 = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < kPA; ++q) {
            const int e = t + q * kFThreads;
            int mi, ki;
            if (A1 == OP_TABT) { mi = e % kFM; ki = e / kFM; } else { mi = e / kBK; ki = e % kBK; }
            const int m = m0 + mi, k = k0 + ki;
            float v = 0.0f;
            if (m < M && k < M) v = (A1 == OP_TABT) ? rtab[k * M + m] : rtab[m * M + k];
            ra[q] = v;
        }
        #pragma unroll
        for (int q = 0; q < kPB; ++q) {
            const int e = t + q * kFThreads;
            const int ki = e / kFN, ni = e % kFN, k = k0 + ki;
            rb1[q] = (k < M && ni < N) ? data[k * N + ni] : 0.0f;
        }
    };
    load1(0);
    for (int k0 = 0; k0 < M; k0 += kBK) {
        #pragma unroll
        for (int q = 0; q < kPA; ++q) {
            const int e = t + q * kFThreads;
            int mi, ki;
            if (A1 == OP_TABT) { mi = e % kFM; ki = e / kFM; } else { mi = e / kBK; ki = e % kBK; }
            As[ki][mi] = ra[q];
        }
        #pragma unroll
        for (int q = 0; q < kPB; ++q) {
            const int e = t + q * kFThreads;
            Bs[e / kFN][e % kFN] = rb1[q];
        }
        __syncthreads();
        if (k0 + kBK < M) load1(k0 + kBK);
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float a[8], b[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[k][rb + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][cb + j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // ---- epilogue 1 into the LDS stripe (zero outside the tile) ----
    if (EPI1 == EPI_RESID) {
        const float* blr = ep.blr + T.off / 4;
        #pragma unroll
        for (int ii = 0; ii < 8; ii += 2)
            #pragma unroll
            for (int jj = 0; jj < 4; jj += 2) {
                const int m = m0 + rb + ii, n = cb + jj;
                float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
                if (m + 1 < M && n + 1 < N) {
                    const float v = epi_resid(acc[ii][jj], acc[ii][jj + 1], acc[ii + 1][jj], acc[ii + 1][jj + 1],
                        blr[(m / 2) * (N / 2) + n / 2]);
                    v00 = v; v01 = v; v10 = v; v11 = v;
                    if (ep.hav) {
                        const float* h = ep.hav + T.off + m * N + n;
                        v00 = epi_anchor(v, acc[ii][jj], h[0], ep.hw);
                        v01 = epi_anchor(v, acc[ii][jj + 1], h[1], ep.hw);
                        v10 = epi_anchor(v, acc[ii + 1][jj], h[N], ep.hw);
                        v11 = epi_anchor(v, acc[ii + 1][jj + 1], h[N + 1], ep.hw);
                    }
                }
                Ss[rb + ii][n] = v00; Ss[rb + ii][n + 1] = v01;
                Ss[rb + ii + 1][n] = v10; Ss[rb + ii + 1][n + 1] = v11;
            }
    } else {
        const float lam = ep.lam[blockIdx.y];
        const float step = (ep.hrt && ep.hrt[blockIdx.y]) ? ep.step_hr : ep.step;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = m0 + rb + i, n = cb + j;
                float s = 0.0f;
                if (m < M && n < N) {
                    const int o = T.off + m * N + n;
                    float y = ep.y[o], xp = ep.xp[o];
                    const float z = epi_step(acc[i][j], y, xp, ep.anch[o], ep.aw, step, lam,
                        epi_ws(m, n, M, N, ep.logw), ep.mom);
                    ep.y[o] = y;
                    ep.xp[o] = xp;
                    s = feed_z ? z : y;
                }
                Ss[rb + i][n] = s;
            }
    }
    __syncthreads();

    // ---- product 2, OUT = Ss * B2, K = N ----
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    float rb2[kPB];
    auto load2 = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < kPB; ++q) {
            const int e = t + q * kFThreads;
            int ki, ni;
            if (B2 == OP_TABT) { ki = e % kBK; ni = e / kBK; } else { ki = e / kFN; ni = e % kFN; }
            const int k = k0 + ki;
            float v = 0.0f;
            if (k < N && ni < N) v = (B2 == OP_TABT) ? ctab[ni * N + k] : ctab[k * N + ni];
            rb2[q] = v;
        }
    };
    load2(0);
    for (int k0 = 0; k0 < N; k0 += kBK) {
        #pragma unroll
        for (int q = 0; q < kPB; ++q) {
            const int e = t + q * kFThreads;
            int ki, ni;
            if (B2 == OP_TABT) { ki = e % kBK; ni = e / kBK; } else { ki = e / kFN; ni = e % kFN; }
            Bs[ki][ni] = rb2[q];
        }
        __syncthreads();
        if (k0 + kBK < N) load2(k0 + kBK);
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float a[8], b[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = Ss[rb + i][k0 + k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][cb + j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + rb + i, n = cb + j;
            if (m < M && n < N) out[T.off + m * N + n] = acc[i][j];
        }
}

}  // namespace
}  // namespace gpu_common
