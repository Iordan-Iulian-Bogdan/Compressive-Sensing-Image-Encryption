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
#include "gpu_wmma.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

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

// ---- one ADMM round's FISTA loop per tile, on the WMMA units ----
// k_fista_w runs every fixed-step FISTA iteration of one tile in one block and writes the pixel
// image of the final iterate (what the fused k_gemm2 loop + gemm_inv_b produce), so nothing but
// the round's inputs and that image touches memory. Per step:
//   P1 = y Cc, P2 = Cr^T P1 (pixels), residual r (2x2 average vs the samples, + anchors),
//   P3 = r Cc^T, P4 = Cr P3 (gradient DCT), soft threshold + momentum (epi_step).
// The products take fp16 operands (tables pre-packed by pack_gemm_frags, data rounded when it is
// staged in LDS) and accumulate in fp32; y, xp, the anchor term and the HR anchors stay in
// registers in WMMA output layout. Wave w owns rows 16 (w / kWPR) .. +15 and column tiles
// kTPW (w % kWPR) .. +kTPW-1, the same elements in every product, so the epilogues need no data movement
// but the 2x2 neighbour exchange (lane ^ 1, lane ^ 16).
enum FragVariant { FR_A_TAB, FR_A_TABT, FR_B_TAB, FR_B_TABT };
constexpr int kFragSets = 4;
constexpr int kTPW = 2, kWPR = 4;   // column tiles per wave, waves per 16-row block (Np <= 128)
constexpr int kFistaThreads = 768;   // 32 * kWPR * Mp / 16 for Mp <= 96 (more threads would cap the registers)

__host__ __device__ inline int pad16(int n) { return (n + 15) & ~15; }
// LDS bytes of k_fista_w for padded tile sizes Mp x Np: rows (A operand) + columns (B operand).
inline size_t fista_w_lds(int Mp, int Np) { return (size_t)(Mp * (Np + 8) + Np * (Mp + 8)) * sizeof(std::uint16_t); }

// Appends the WMMA fragments of the N x N DCT table C to out (fp16 bits) and returns their
// offset: A fragments [mt][kt][row][k] of C (TAB) and C^T (TABT), then B fragments
// [nt][kt][col][k] of C (TAB) and C^T (TABT), Np x Np halves each.
inline size_t pack_gemm_frags(std::vector<std::uint16_t>& out, const float* C, int N) {
    const int Np = pad16(N), KT = Np / 16;
    const size_t at = out.size();
    out.resize(at + (size_t)kFragSets * Np * Np);
    for (int t = 0; t < KT; ++t)
        for (int kt = 0; kt < KT; ++kt)
            for (int l = 0; l < 16; ++l)
                for (int e = 0; e < 16; ++e) {
                    const int r = t * 16 + l, k = kt * 16 + e;   // A: row r, column k; B: row k, column r
                    const bool in = r < N && k < N;
                    const size_t i = at + ((size_t)(t * KT + kt) * 16 + l) * 16 + e, v = (size_t)Np * Np;
                    out[i + FR_A_TAB * v] = gpu_wmma::to_half(in ? C[r * N + k] : 0.0f);
                    out[i + FR_A_TABT * v] = gpu_wmma::to_half(in ? C[k * N + r] : 0.0f);
                    out[i + FR_B_TAB * v] = gpu_wmma::to_half(in ? C[k * N + r] : 0.0f);
                    out[i + FR_B_TABT * v] = gpu_wmma::to_half(in ? C[r * N + k] : 0.0f);
                }
    return at;
}

struct FistaW {
    const float *y, *anch, *blr, *hav, *lam;   // round start (y = DCT(z)), samples, anchors (or null), L1 weights
    const int* hrt;                            // per tile: takes step_hr (or null)
    float hw, aw, step, step_hr, logw;
    int inner;
};

template <bool HR>
__global__ void __launch_bounds__(kFistaThreads) k_fista_w(const TileDev* td, const gpu_wmma::half_t* frag, FistaW p, float* pix_out) {
    using namespace gpu_wmma;
    extern __shared__ __attribute__((aligned(16))) half_t lds[];
    const TileDev T = td[blockIdx.x];
    const int M = T.rows, N = T.cols, Mp = pad16(M), Np = pad16(N), MT = Mp / 16, NT = Np / 16;
    const int SA = Np + 8, SB = Mp + 8;
    half_t* Ra = lds;             // rows: A operand of P1 / P3 (m, k)
    half_t* Bt = lds + Mp * SA;   // columns: B operand of P2 / P4 (n, k)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r16 = lane & 15, hi = lane >> 4;
    const int mt = w / kWPR, t0 = (w % kWPR) * kTPW, t1 = min(NT, t0 + kTPW);
    const bool active = mt < MT;   // wave-uniform
    const half_t* fCc = frag + T.pad[1] + (size_t)FR_B_TAB * Np * Np;
    const half_t* fCcT = frag + T.pad[1] + (size_t)FR_B_TABT * Np * Np;
    const half_t* fCr = frag + T.pad[0] + (size_t)FR_A_TAB * Mp * Mp;
    const half_t* fCrT = frag + T.pad[0] + (size_t)FR_A_TABT * Mp * Mp;
    const float lam = p.lam[blockIdx.x];
    const float step = (p.hrt && p.hrt[blockIdx.x]) ? p.step_hr : p.step;
    const float* blr = p.blr + T.off / 4;

    // element (j, i) of this lane: m = 16 mt + 2 i + hi, n = 16 (t0 + j) + r16
    float y[kTPW][8], xp[kTPW][8], an[kTPW][8], hv[kTPW][8], ws[kTPW][8];
    #pragma unroll
    for (int j = 0; j < kTPW; ++j)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = mt * 16 + 2 * i + hi, n = (t0 + j) * 16 + r16;
            const bool in = active && t0 + j < t1 && m < M && n < N;
            const int o = T.off + m * N + n;
            y[j][i] = in ? p.y[o] : 0.0f;
            xp[j][i] = y[j][i];
            an[j][i] = in ? p.anch[o] : 0.0f;
            hv[j][i] = (HR && in) ? p.hav[o] : -1e30f;
            ws[j][i] = epi_ws(m, n, M, N, p.logw);
        }
    auto put_rows = [&](const float (&v)[kTPW][8]) {
        #pragma unroll
        for (int j = 0; j < kTPW; ++j)
            if (t0 + j < t1)
                #pragma unroll
                for (int i = 0; i < 8; ++i) Ra[(mt * 16 + 2 * i + hi) * SA + (t0 + j) * 16 + r16] = (half_t)v[j][i];
    };
    auto put_cols = [&](const f8 (&a)[kTPW]) {
        #pragma unroll
        for (int j = 0; j < kTPW; ++j)
            if (t0 + j < t1)
                #pragma unroll
                for (int i = 0; i < 8; ++i) Bt[((t0 + j) * 16 + r16) * SB + mt * 16 + 2 * i + hi] = (half_t)a[j][i];
    };
    // acc = (own rows of Ra) * B fragments, K = N
    auto prod_rows = [&](const half_t* fB, f8 (&a)[kTPW]) {
        #pragma unroll
        for (int j = 0; j < kTPW; ++j) a[j] = splat(0.0f);
        for (int kt = 0; kt < NT; ++kt) {
            const h16 av = ld16(Ra + (mt * 16 + r16) * SA + kt * 16);
            #pragma unroll
            for (int j = 0; j < kTPW; ++j)
                if (t0 + j < t1) a[j] = mma(av, ld16(fB + ((size_t)((t0 + j) * NT + kt) * 16 + r16) * 16), a[j]);
        }
        #pragma unroll
        for (int j = 0; j < kTPW; ++j) pin(a[j]);
    };
    // acc = A fragments (row tile mt) * Bt, K = M
    auto prod_cols = [&](const half_t* fA, f8 (&a)[kTPW]) {
        #pragma unroll
        for (int j = 0; j < kTPW; ++j) a[j] = splat(0.0f);
        for (int kt = 0; kt < MT; ++kt) {
            const h16 av = ld16(fA + ((size_t)(mt * MT + kt) * 16 + r16) * 16);
            #pragma unroll
            for (int j = 0; j < kTPW; ++j)
                if (t0 + j < t1) a[j] = mma(av, ld16(Bt + ((t0 + j) * 16 + r16) * SB + kt * 16), a[j]);
        }
        #pragma unroll
        for (int j = 0; j < kTPW; ++j) pin(a[j]);
    };

    f8 acc[kTPW];
    float t = 1.0f;   // FISTA momentum sequence, as the host loop
    for (int k = 0; k <= p.inner; ++k) {
        const bool last = k == p.inner;   // last pass: the pixels of the final iterate (z = xp) only
        if (last)
            #pragma unroll
            for (int j = 0; j < kTPW; ++j)
                #pragma unroll
                for (int i = 0; i < 8; ++i) y[j][i] = xp[j][i];
        if (active) put_rows(y);
        __syncthreads();
        if (active) { prod_rows(fCc, acc); put_cols(acc); }
        __syncthreads();
        if (active) prod_cols(fCrT, acc);   // pixels
        if (last) break;
        if (active) {
            #pragma unroll
            for (int j = 0; j < kTPW; ++j) {
                if (t0 + j >= t1) continue;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    // the pixel's 2x2 block: column neighbour in lane ^ 1, row neighbour in lane ^ 16
                    const float px = acc[j][i], pc = dpp_xor<1>(px);
                    const float pr = __shfl_xor(px, 16, 32), prc = __shfl_xor(pc, 16, 32);
                    const bool co = r16 & 1;
                    const float e0 = hi ? pr : px, e1 = hi ? prc : pc, o0 = hi ? px : pr, o1 = hi ? pc : prc;
                    const int m = mt * 16 + 2 * i + hi, n = (t0 + j) * 16 + r16;
                    float v = 0.0f;
                    if (m < M && n < N) {
                        v = epi_resid(co ? e1 : e0, co ? e0 : e1, co ? o1 : o0, co ? o0 : o1, blr[(m / 2) * (N / 2) + n / 2]);
                        if (HR) v = epi_anchor(v, px, hv[j][i], p.hw);
                    }
                    Ra[m * SA + n] = (half_t)v;
                }
            }
        }
        __syncthreads();
        if (active) { prod_rows(fCcT, acc); put_cols(acc); }
        __syncthreads();
        const float t_new = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t * t)), mom = (t - 1.0f) / t_new;
        t = t_new;
        if (active) {
            prod_cols(fCr, acc);   // gradient DCT
            #pragma unroll
            for (int j = 0; j < kTPW; ++j)
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = mt * 16 + 2 * i + hi, n = (t0 + j) * 16 + r16;
                    if (t0 + j < t1 && m < M && n < N)
                        epi_step(acc[j][i], y[j][i], xp[j][i], an[j][i], p.aw, step, lam, ws[j][i], mom);
                }
        }
    }
    if (active)
        #pragma unroll
        for (int j = 0; j < kTPW; ++j)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = mt * 16 + 2 * i + hi, n = (t0 + j) * 16 + r16;
                if (t0 + j < t1 && m < M && n < N) pix_out[T.off + m * N + n] = acc[j][i];
            }
}

}  // namespace
}  // namespace gpu_common
