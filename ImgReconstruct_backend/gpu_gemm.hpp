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
    const float* anch;    // EPI_STEP
    const float* ws;
    const float* lam;     // per-tile L1 weight
    float* y; float* xp; float* zc;
    float aw, step, mom;
};

constexpr int kBM = 32, kBN = 32, kBK = 16, kGemmThreads = 64;

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
                    const float avg = 0.25f * (acc[ii][jj] + acc[ii][jj + 1] + acc[ii + 1][jj] + acc[ii + 1][jj + 1]);
                    const float bs = blr[(m / 2) * (N / 2) + n / 2];
                    const float v = bs > -1e29f ? 0.25f * (avg - bs) : 0.0f;
                    float* o = out + T.off + m * N + n;
                    o[0] = v; o[1] = v; o[N] = v; o[N + 1] = v;
                }
            }
    } else {
        const float lam = ep.lam[blockIdx.y];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (mb + i < M && nb + j < N) {
                    const int o = T.off + (mb + i) * N + nb + j;
                    const float y = ep.y[o], a = ep.anch[o];
                    const float g = 2.0f * acc[i][j] + 2.0f * ep.aw * (y - a);
                    const float v = y - ep.step * g - a;
                    const float thr = lam * ep.step * ep.ws[o];
                    const float av = fabsf(v);
                    const float z = a + ((av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f);
                    ep.y[o] = z + ep.mom * (z - ep.xp[o]);
                    ep.xp[o] = z;
                    ep.zc[o] = z;
                }
    }
}

}  // namespace
}  // namespace gpu_common
