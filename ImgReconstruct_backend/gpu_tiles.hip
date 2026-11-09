// Batched GPU tile solver (see gpu_tiles.h).
//
// Layout: every tile owns a contiguous slice [off, off + rows*cols) of each
// state array. DCT work is two small dense matrix products per transform
// (T = X * Cc^T, then C_r * T for the forward transform), which is cheap for
// tiles up to ~100 px and needs no FFT library. Kernels use blockIdx.y as the
// tile index and blockIdx.x * kThreads + threadIdx.x as the element index.
#include "gpu_tiles.h"

#include "gpu_common.hpp"
#include "gpu_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace gpu_tiles {
namespace {

using namespace gpu_common;

// Residual at the samples, zero elsewhere (SC is zeroed first).
__global__ void k_scatter(const TileDev* td, const int* stile, const int* rix, const int* riy,
    const float* b, const float* PIX, float* SC, int total) {
    const int s = blockIdx.x * kThreads + threadIdx.x;
    if (s >= total) return;
    const TileDev T = td[stile[s]];
    const int idx = T.off + rix[s] * T.cols + riy[s];
    SC[idx] = PIX[idx] - b[s];
}

// Soft threshold + FISTA momentum. Lipschitz constant 2 => step 1/2, and the
// gradient 2*DCT(sc) turns the update into v = y - G with threshold lambda*w*ws/2.
__global__ void k_step(const TileDev* td, const float* G, const float* W, const float* WS,
    float* Y, float* XP, float* Z, const float* LAM, float mom) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float v = Y[o] - G[o];
    const float thr = 0.5f * LAM[blockIdx.y] * W[o] * WS[o];
    const float a = fabsf(v);
    const float z = a > thr ? (v > 0.0f ? a - thr : thr - a) : 0.0f;
    Y[o] = z + mom * (z - XP[o]);
    XP[o] = z;
    Z[o] = z;
}

// eps = max(0.02 * max|z|, 1e-3) per tile (one block per tile).
__global__ void k_absmax(const TileDev* td, const float* Z, float* EPS) {
    __shared__ float sh[kThreads];
    const TileDev T = td[blockIdx.x];
    const int n = T.rows * T.cols;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) m = fmaxf(m, fabsf(Z[T.off + i]));
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float eps = 0.02f * sh[0];
        EPS[blockIdx.x] = eps < 1e-3f ? 1e-3f : eps;
    }
}

__global__ void k_wupdate(const TileDev* td, const float* Z, const float* EPS, float* W) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float e = EPS[blockIdx.y];
    W[o] = e / (fabsf(Z[o]) + e);
}

// ---- tv > 0 (backtracking) branch: mirrors cs_fista_grad / cs_smooth_fx /
// cs_fista_core_single of helper_functions.cpp for the DCT basis ----

// f = sum SC^2 + tv * sum PHI per tile (SC is zero away from the samples).
__global__ void k_obj(const TileDev* td, const float* SC, const float* PHI, const float* TV, double* F) {
    const TileDev T = td[blockIdx.x];
    const int n = T.rows * T.cols;
    double d = 0.0, p = 0.0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float s = SC[T.off + i];
        d += (double)(s * s);
        p += (double)PHI[T.off + i];
    }
    const double ds = block_sum(d), ps = block_sum(p);
    if (threadIdx.x == 0) F[blockIdx.x] = (double)((float)ds + TV[blockIdx.x] * (float)ps);
}

// S = 2*SC + tv*TVG, so one forward DCT gives g = 2*DCT(sc) + tv*DCT(tvgrad).
__global__ void k_comb(const TileDev* td, const float* SC, const float* TVG, const float* TV, float* S) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    S[o] = 2.0f * SC[o] + TV[blockIdx.y] * TVG[o];
}

// Backtracking candidate z = soft(y - g/L, lambda/L * w * ws) for tiles still searching.
__global__ void k_trial(const TileDev* td, const float* Y, const float* G, const float* W, const float* WS,
    const float* LAM, const float* LT, const int* ACT, float* Z) {
    if (!ACT[blockIdx.y]) return;
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    const float step = 1.0f / LT[blockIdx.y];
    const float v = Y[o] - step * G[o];
    const float thr = LAM[blockIdx.y] * step * W[o] * WS[o];
    const float av = fabsf(v);
    Z[o] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
}

// Opt-in phase timing (env CS_GPU_PROF=1): device-synced marks around build (host layout + upload),
// the FISTA loop and the download of each solve, printed per call.
struct PhaseProf {
    bool on = false;
    std::chrono::steady_clock::time_point t;
    double ms[3] = {}, sub[2] = {};   // ms: build (rest = upload), loop, download; sub: host layout, device allocation
    PhaseProf() { const char* e = std::getenv("CS_GPU_PROF"); on = e && e[0] == '1'; t = std::chrono::steady_clock::now(); }
    void lap(double& out) {
        if (!on) return;
        (void)hipDeviceSynchronize();
        const auto n = std::chrono::steady_clock::now();
        out = std::chrono::duration<double, std::milli>(n - t).count(); t = n;
    }
    void mark(int i) { lap(ms[i]); }
    void mark_sub(int i) { lap(sub[i]); }
    void print(const char* kind, int nt, int max_rows, int max_cols, long long total, int samples, int inner, int rw) const {
        if (!on) return;
        std::fprintf(stderr, "[gpu-prof] gpu_tiles %s: %d tiles (max %dx%d, %.2f Mpx, %d samples), %d steps x %d reweights: "
            "build %.1f ms (layout %.1f, alloc %.1f, upload %.1f), loop %.1f ms, download %.1f ms\n", kind, nt, max_rows,
            max_cols, total * 1e-6, samples, inner, rw, sub[0] + sub[1] + ms[0], sub[0], sub[1], ms[0], ms[1], ms[2]);
    }
};

// ---- tv = 0 tiles, fused: the whole reweighted FISTA loop of one tile in one block ----
// k_lr_fista replaces solve_fixed's launches (two GEMM pairs, memset, scatter and k_step per step;
// absmax / wupdate per reweight) with one block per tile that keeps everything in LDS and registers.
// Thread t owns the 4x4 output block t of the tile in every product (rows 4 (t / nbn).., columns
// 4 (t % nbn)..), so each element's sum runs over k in ascending order as in k_gemm and the
// elementwise steps are the same expressions: the results match the unfused path.
// LDS: A (Y, then PIX / SC) and B (TMP); the DCT tables are read through the cache (LDS for them halved the
// blocks per CU).
constexpr int kLrThreads = 256;

// One 4x4 output block of a per-tile product, by k_gemm's operand conventions:
//   P 0: TMP = Y  * Cc     P 1: PIX = Cr^T * TMP     P 2: TMP = SC * Cc^T     P 3: G = Cr * TMP
// A (Y, SC) is stored column-major (A[k * M + m]) and B (TMP) row-major; CrT / CcT are the transposed tables, so every
// operand is read along its contiguous side. V: M and N are multiples of 4 (and the tables 16-byte aligned), so the
// four operands of a k step are one 16-byte load each.
template <int P, bool V>
__device__ inline void lr_prod(const float* A, const float* B, const float* Cr, const float* CrT, const float* Cc,
    const float* CcT, int M, int N, int mb, int nb, float acc[4][4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    int mi[4], ni[4];   // clamped rows / columns (outputs past the tile are computed but not used)
    #pragma unroll
    for (int i = 0; i < 4; ++i) { mi[i] = min(mb + i, M - 1); ni[i] = min(nb + i, N - 1); }
    // a(m, k) = SA[k * M + m], b(k, n) = SB[k * N + n]
    const float* SA = (P == 0 || P == 2) ? A : (P == 1 ? Cr : CrT);
    const float* SB = P == 0 ? Cc : (P == 2 ? CcT : B);
    const int K = (P == 0 || P == 2) ? N : M;
    for (int k = 0; k < K; ++k) {
        float a[4], b[4];
        if (V) {
            const float4 va = *(const float4*)(SA + k * M + mb), vb = *(const float4*)(SB + k * N + nb);
            a[0] = va.x; a[1] = va.y; a[2] = va.z; a[3] = va.w;
            b[0] = vb.x; b[1] = vb.y; b[2] = vb.z; b[3] = vb.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = SA[k * M + mi[i]];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = SB[k * N + ni[j]];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
}

// soff: per-tile sample offsets (nt + 1); sidx / sb: tile-local sample pixel indices (row * cols + col), values.
// MOM: the inner momentum factors (the same in every reweighting pass). X: warm start in, solution out.
__global__ void __launch_bounds__(kLrThreads) k_lr_fista(const TileDev* td, const float* tab, const int* soff,
    const int* sidx, const float* sb, const float* LAM, const float* MOM, float* X, int inner,
    int reweights, float logw) {
    extern __shared__ float lds[];
    const TileDev T = td[blockIdx.x];
    const int M = T.rows, N = T.cols, n = M * N;
    float* A = lds;
    float* B = lds + n;
    const float* Cr = tab + T.rtab;
    const float* Cc = tab + T.ctab;
    const float* CrT = tab + T.pad[0];   // transposed tables (Batch::build, lean)
    const float* CcT = tab + T.pad[1];
    const int t = threadIdx.x;
    const bool vec = ((M | N) & 3) == 0 && ((T.rtab | T.ctab | T.pad[0] | T.pad[1]) & 3) == 0;
    // samples as a dense map (sentinel where none), read into the owned blocks' registers
    for (int i = t; i < n; i += kLrThreads) A[i] = -1e30f;
    __syncthreads();
    for (int s = soff[blockIdx.x] + t; s < soff[blockIdx.x + 1]; s += kLrThreads) A[sidx[s]] = sb[s];
    __syncthreads();
    const int nbn = (N + 3) / 4;
    const bool own = t < ((M + 3) / 4) * nbn;
    const int mb = own ? (t / nbn) * 4 : 0, nb = own ? (t % nbn) * 4 : 0;
    bool in[4][4];
    float bs[4][4], y[4][4], xp[4][4], w[4][4], ws[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = mb + i, c = nb + j;
            in[i][j] = own && m < M && c < N;
            bs[i][j] = in[i][j] ? A[m * N + c] : -1e30f;
            y[i][j] = xp[i][j] = in[i][j] ? X[T.off + m * N + c] : 0.0f;
            w[i][j] = 1.0f;
            // frequency weight, as k_ws
            const float fr = M > 1 ? (float)m / (float)(M - 1) : 0.0f;
            const float fc = N > 1 ? (float)c / (float)(N - 1) : 0.0f;
            ws[i][j] = expf(logw * 0.5f * (fr + fc));
        }
    const float lam = LAM[blockIdx.x];
    __shared__ float red[kLrThreads];
    for (int r = 0; r < reweights; ++r) {
        for (int k = 0; k < inner; ++k) {
            __syncthreads();   // the previous step's last product (reads B) is done everywhere
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (in[i][j]) A[(nb + j) * M + mb + i] = y[i][j];
            __syncthreads();
            float acc[4][4];
            if (vec) lr_prod<0, true>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            else lr_prod<0, false>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            __syncthreads();   // A free (Y was only this product's operand)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (in[i][j]) B[(mb + i) * N + nb + j] = acc[i][j];
            __syncthreads();
            if (vec) lr_prod<1, true>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            else lr_prod<1, false>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            // residual at the samples, zero elsewhere (k_scatter)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (in[i][j]) A[(nb + j) * M + mb + i] = bs[i][j] > -1e29f ? acc[i][j] - bs[i][j] : 0.0f;
            __syncthreads();
            if (vec) lr_prod<2, true>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            else lr_prod<2, false>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            __syncthreads();   // B free
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (in[i][j]) B[(mb + i) * N + nb + j] = acc[i][j];
            __syncthreads();
            if (vec) lr_prod<3, true>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            else lr_prod<3, false>(A, B, Cr, CrT, Cc, CcT, M, N, mb, nb, acc);
            // soft threshold + momentum (k_step)
            const float mom = MOM[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float v = y[i][j] - acc[i][j];
                    const float thr = 0.5f * lam * w[i][j] * ws[i][j];
                    const float a = fabsf(v);
                    const float z = a > thr ? (v > 0.0f ? a - thr : thr - a) : 0.0f;
                    y[i][j] = z + mom * (z - xp[i][j]);
                    xp[i][j] = z;
                }
        }
        if (r + 1 < reweights) {
            // eps = max(0.02 * max|z|, 1e-3) over the tile, W = eps / (|z| + eps) (k_absmax, k_wupdate)
            float m = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (in[i][j]) m = fmaxf(m, fabsf(xp[i][j]));
            red[t] = m;
            __syncthreads();
            for (int s = kLrThreads / 2; s > 0; s >>= 1) {
                if (t < s) red[t] = fmaxf(red[t], red[t + s]);
                __syncthreads();
            }
            const float e0 = 0.02f * red[0];
            const float e = e0 < 1e-3f ? 1e-3f : e0;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    w[i][j] = e / (fabsf(xp[i][j]) + e);
                    y[i][j] = xp[i][j];   // the next pass starts from z
                }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            if (in[i][j]) X[T.off + (mb + i) * N + nb + j] = xp[i][j];
}

// ---- solve_ycc_bgr's merge: the CPU tail of cs_gpu_solve_tiles_ycc, one block row per job ----
struct MergeDev {
    const float *y, *cr, *cb;   // inverse-DCT planes (pixels / 255)
    int rows, cols, crows, ccols;
    long long out;              // byte offset of the job's BGR tile
};

// cv::resize INTER_LINEAR source coordinate of destination index d (scale = source / destination size).
__device__ inline float lin_src(int d, double scale, int n, int& s0, int& s1) {
    float f = (float)((d + 0.5) * scale - 0.5);
    int s = (int)floorf(f);
    f -= (float)s;
    if (s < 0) { f = 0.f; s = 0; }
    if (s >= n - 1) { f = 0.f; s = n - 1; }
    s0 = s; s1 = min(s + 1, n - 1);
    return f;
}

__device__ inline int sat8i(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// x 255, chroma bilinear (horizontal, then vertical), convertTo CV_8U (round half even, saturate), then
// YCrCb -> BGR in OpenCV's 8-bit fixed point (as gpu_sr's k_finish).
__global__ void k_ycc_merge(const MergeDev* md, std::uint8_t* OUT) {
    const MergeDev M = md[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= M.rows * M.cols) return;
    const int r = i / M.cols, c = i % M.cols;
    int x0, x1, y0, y1;
    const float fx = lin_src(c, (double)M.ccols / M.cols, M.ccols, x0, x1);
    const float fy = lin_src(r, (double)M.crows / M.rows, M.crows, y0, y1);
    auto bil = [&](const float* P) {
        const float* p0 = P + y0 * M.ccols;
        const float* p1 = P + y1 * M.ccols;
        const float h0 = p0[x0] * 255.0f * (1.0f - fx) + p0[x1] * 255.0f * fx;
        const float h1 = p1[x0] * 255.0f * (1.0f - fx) + p1[x1] * 255.0f * fx;
        return h0 * (1.0f - fy) + h1 * fy;
    };
    const int y = sat8i((int)rintf(M.y[i] * 255.0f));
    const int cr = sat8i((int)rintf(bil(M.cr))) - 128, cb = sat8i((int)rintf(bil(M.cb))) - 128;
    std::uint8_t* q = OUT + M.out + 3 * (long long)i;
    q[0] = (std::uint8_t)sat8i(y + ((cb * 29049 + (1 << 13)) >> 14));
    q[1] = (std::uint8_t)sat8i(y + ((cb * -5636 + cr * -11698 + (1 << 13)) >> 14));
    q[2] = (std::uint8_t)sat8i(y + ((cr * 22987 + (1 << 13)) >> 14));
}

// Host layout + device copies of a batch's tiles, samples and warm starts.
struct Batch {
    int nt = 0, max_n = 0, max_rows = 0, max_cols = 0, total_samples = 0;
    long long total = 0;
    std::vector<TileDev> td;
    std::vector<float> table, hX;
    DevBuf<TileDev> dtd; DevBuf<float> dtab, dX, db, dLAM, dWS, dW; DevBuf<int> drix, driy, dst, dso;
    // host arrays build_host fills and upload copies to the device
    std::vector<float> hb, lam;
    std::vector<int> hrix, hriy, hst, so;
    bool lean = false;
    double layout_ms = 0.0;   // build_host's wall time

    // build_host + upload (the two halves solve_ycc_bgr runs apart: the host half off the GPU lock)
    bool build(const std::vector<TileProblem>& tiles, float wscale, PhaseProf& pp, bool lean_) {
        if (!build_host(tiles, lean_)) return false;
        pp.mark_sub(0);
        return upload(wscale, pp);
    }

    // Host half: layout, tables, packed samples. Touches no device state.
    // lean (fused kernel): samples as one packed index each (in drix), no per-sample tile ids and no W / WS arrays
    bool build_host(const std::vector<TileProblem>& tiles, bool lean_) {
        const auto h0 = std::chrono::steady_clock::now();
        lean = lean_;
        nt = (int)tiles.size();
        TileLayout lay;
        for (const TileProblem& P : tiles) { lay.add(P.rows, P.cols); total_samples += P.m; }
        td = lay.td; table = lay.table; total = lay.total; max_n = lay.max_n;
        max_rows = lay.max_rows; max_cols = lay.max_cols;
        if (lean) {   // the fused kernel also reads every table transposed (pad[0]: rows table, pad[1]: columns table)
            std::map<int, int> toff;
            for (const auto& kv : lay.table_off) {
                const int N = kv.first, off = kv.second;
                table.resize((table.size() + 3) / 4 * 4);   // 16-byte aligned
                toff[N] = (int)table.size();
                for (int r = 0; r < N; ++r)
                    for (int c = 0; c < N; ++c) table.push_back(lay.table[(size_t)off + (size_t)c * N + r]);
            }
            for (TileDev& d : td) { d.pad[0] = toff[d.rows]; d.pad[1] = toff[d.cols]; }
        }
        if (total > 2000000000LL) return false;
        hX.resize((size_t)total);
        hb.resize((size_t)total_samples); lam.resize((size_t)nt);
        hrix.resize((size_t)total_samples); hriy.resize(lean ? 0 : (size_t)total_samples); hst.resize(lean ? 0 : (size_t)total_samples);
        so.assign((size_t)nt + 1, 0);
        for (int t = 0; t < nt; ++t) so[(size_t)t + 1] = so[(size_t)t] + tiles[(size_t)t].m;
        parallel_for(nt, [&](int t) {
            const TileProblem& P = tiles[(size_t)t];
            std::copy(P.x, P.x + (size_t)P.rows * P.cols, hX.begin() + td[(size_t)t].off);
            const size_t o = (size_t)so[(size_t)t];
            if (lean)
                for (int k = 0; k < P.m; ++k) {
                    hb[o + k] = P.b[k];
                    hrix[o + k] = P.rix[k] * P.cols + P.riy[k];
                }
            else
                for (int k = 0; k < P.m; ++k) {
                    hb[o + k] = P.b[k];
                    hrix[o + k] = P.rix[k];
                    hriy[o + k] = P.riy[k];
                    hst[o + k] = t;
                }
            lam[(size_t)t] = P.lambda;
        });
        layout_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
        return true;
    }

    // Device half: allocation, copies, and (not lean) the W / WS initialisation.
    bool upload(float wscale, PhaseProf& pp) {
        const size_t ns = (size_t)std::max(1, total_samples);
        if (!dtd.alloc((size_t)nt) || !dtab.alloc(table.size()) || !dX.alloc((size_t)total) || !db.alloc(ns) ||
            !drix.alloc(ns) || !dLAM.alloc((size_t)nt) || !dso.alloc((size_t)nt + 1) ||
            (!lean && (!driy.alloc(ns) || !dst.alloc(ns) || !dWS.alloc((size_t)total) || !dW.alloc((size_t)total)))) {
            std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
            return false;
        }
        pp.mark_sub(1);
        const size_t bytes = (size_t)total * sizeof(float);
        HIP_TRY(hipMemcpy(dtd.p, td.data(), (size_t)nt * sizeof(TileDev), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dtab.p, table.data(), table.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dX.p, hX.data(), bytes, hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dLAM.p, lam.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy(dso.p, so.data(), ((size_t)nt + 1) * sizeof(int), hipMemcpyHostToDevice));
        if (total_samples > 0) {
            HIP_TRY(hipMemcpy(db.p, hb.data(), (size_t)total_samples * sizeof(float), hipMemcpyHostToDevice));
            HIP_TRY(hipMemcpy(drix.p, hrix.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
            if (!lean) {
                HIP_TRY(hipMemcpy(driy.p, hriy.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
                HIP_TRY(hipMemcpy(dst.p, hst.data(), (size_t)total_samples * sizeof(int), hipMemcpyHostToDevice));
            }
        }
        if (lean) return true;
        const float logw = wscale > 1.0f ? std::log(wscale) : 0.0f;
        k_ws<<<grid(), dim3(kThreads)>>>(dtd.p, dWS.p, logw);
        HIP_TRY(hipMemsetD32((hipDeviceptr_t)dW.p, 0x3F800000, (size_t)total));   // W = 1.0f initially
        return true;
    }
    dim3 grid() const { return dim3((unsigned)((max_n + kThreads - 1) / kThreads), (unsigned)nt); }
    dim3 sgrid() const { return dim3((unsigned)((total_samples + kThreads - 1) / kThreads)); }

    // Dense DCT passes on the LDS-tiled GEMM (same k-ordered sums as k_synth_* / k_adj_*).
    // PIX = IDCT(Y) and G = DCT(SC); TMP is scratch.
    void synth(const float* Y, float* TMP, float* PIX) const {
        const int nbn = (max_cols + kBN - 1) / kBN;
        const dim3 g((unsigned)(((max_rows + kBM - 1) / kBM) * nbn), (unsigned)nt);
        k_gemm<OP_DATA, OP_TAB, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, Y, TMP, nbn, Epi{});
        k_gemm<OP_TABT, OP_DATA, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, TMP, PIX, nbn, Epi{});
    }
    void adj(const float* SC, float* TMP, float* G) const {
        const int nbn = (max_cols + kBN - 1) / kBN;
        const dim3 g((unsigned)(((max_rows + kBM - 1) / kBM) * nbn), (unsigned)nt);
        k_gemm<OP_DATA, OP_TABT, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, SC, TMP, nbn, Epi{});
        k_gemm<OP_TAB, OP_DATA, EPI_STORE><<<g, dim3(kGemmThreads)>>>(dtd.p, dtab.p, TMP, G, nbn, Epi{});
    }

    // reweighting: W = eps / (|Z| + eps), eps from the tile's max |Z|
    void reweight(const float* dZ, float* dEPS) {
        k_absmax<<<dim3((unsigned)nt), dim3(kThreads)>>>(dtd.p, dZ, dEPS);
        k_wupdate<<<grid(), dim3(kThreads)>>>(dtd.p, dZ, dEPS, dW.p);
    }
    bool download(const std::vector<TileProblem>& tiles) {
        HIP_TRY(hipGetLastError());
        HIP_TRY(hipDeviceSynchronize());
        HIP_TRY(hipMemcpy(hX.data(), dX.p, (size_t)total * sizeof(float), hipMemcpyDeviceToHost));
        parallel_for(nt, [&](int t) {
            const TileProblem& P = tiles[(size_t)t];
            std::copy(hX.begin() + td[(size_t)t].off, hX.begin() + td[(size_t)t].off + (size_t)P.rows * P.cols, P.x);
        });
        return true;
    }
};

// solve_ycc_bgr's view of one solve: pix (allocated by the caller, total floats) receives the inverse DCT of the
// solution instead of the download, tile t at pix + td[t].off.
struct Kept { DevBuf<float> pix; std::vector<TileDev> td; };

// tv = 0: exact fixed step 1/2 (previous behaviour, unchanged).
// The host half of solve_fixed (prepare_fixed) kept for its device half (run_fixed).
struct FixedPrep {
    std::unique_ptr<Batch> B;
    bool fused = false;
    size_t lds = 0;
};

// Host half: the fused / unfused choice and the batch's host layout. No device state is touched.
bool prepare_fixed(const std::vector<TileProblem>& tiles, const Params& prm, FixedPrep& fp, bool keep) {
    // Fused per-tile loop (k_lr_fista; CS_GPU_LR_FUSED=0 keeps the launch-per-step loop) when every tile fits one
    // block: at most kLrThreads 4x4 output blocks, and A + B in 64 KB of LDS.
    size_t lds = 0;
    bool fit = true;
    for (const TileProblem& P : tiles) {
        lds = std::max(lds, (size_t)(2 * P.rows * P.cols) * sizeof(float));
        fit = fit && ((P.rows + 3) / 4) * ((P.cols + 3) / 4) <= kLrThreads;
    }
    const char* ef = std::getenv("CS_GPU_LR_FUSED");
    const bool fused = fit && lds <= 64 * 1024 && !(ef && ef[0] == '0');
    if (keep && !fused) return false;
    fp.fused = fused;
    fp.lds = lds;
    fp.B = std::make_unique<Batch>();
    return fp.B->build_host(tiles, fused);
}

// Device half; the batch's device buffers are released on return (as when solve_fixed owned the Batch).
bool run_fixed(const std::vector<TileProblem>& tiles, const Params& prm, FixedPrep& fp, Kept* keep = nullptr) {
    PhaseProf pp;
    pp.sub[0] = fp.B->layout_ms;
    Batch& B = *fp.B;
    const bool fused = fp.fused;
    const size_t lds = fp.lds;
    if (!B.upload(prm.wscale, pp)) return false;
    const size_t N = (size_t)B.total, bytes = N * sizeof(float);
    DevBuf<float> dY, dXP, dZ, dTMP, dPIX, dSC, dG, dEPS;
    if (!fused && (!dY.alloc(N) || !dXP.alloc(N) || !dZ.alloc(N) || !dTMP.alloc(N) || !dPIX.alloc(N) ||
        !dSC.alloc(N) || !dG.alloc(N) || !dEPS.alloc((size_t)B.nt))) {
        std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
        return false;
    }
    const dim3 blk(kThreads), grid = B.grid(), sgrid = B.sgrid();
    pp.mark(0);
    if (fused) {
        std::vector<float> mom((size_t)std::max(1, prm.inner));
        float t = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {   // the momentum sequence of the loop below
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            mom[(size_t)k] = (t - 1.0f) / t_new;
            t = t_new;
        }
        DevBuf<float> dMOM;
        if (!dMOM.alloc(mom.size())) return false;
        HIP_TRY(hipMemcpy(dMOM.p, mom.data(), mom.size() * sizeof(float), hipMemcpyHostToDevice));
        const float logw = prm.wscale > 1.0f ? std::log(prm.wscale) : 0.0f;
        k_lr_fista<<<dim3((unsigned)B.nt), dim3(kLrThreads), lds>>>(B.dtd.p, B.dtab.p, B.dso.p, B.drix.p,
            B.db.p, B.dLAM.p, dMOM.p, B.dX.p, prm.inner, prm.reweights, logw);
        HIP_TRY(hipGetLastError());
    }
    for (int r = 0; r < (fused ? 0 : prm.reweights); ++r) {
        HIP_TRY(hipMemcpyAsync(dY.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        float t = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            B.synth(dY.p, dTMP.p, dPIX.p);
            HIP_TRY(hipMemsetAsync(dSC.p, 0, bytes));
            if (B.total_samples > 0)
                k_scatter<<<sgrid, blk>>>(B.dtd.p, B.dst.p, B.drix.p, B.driy.p, B.db.p, dPIX.p, dSC.p, B.total_samples);
            B.adj(dSC.p, dTMP.p, dG.p);
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            k_step<<<grid, blk>>>(B.dtd.p, dG.p, B.dW.p, B.dWS.p, dY.p, dXP.p, dZ.p, B.dLAM.p, mom);
            t = t_new;
        }
        HIP_TRY(hipMemcpyAsync(B.dX.p, dZ.p, bytes, hipMemcpyDeviceToDevice));
        if (r + 1 < prm.reweights) B.reweight(dZ.p, dEPS.p);
    }
    pp.mark(1);
    bool ok = true;
    if (keep) {   // inverse DCT on the device; the solution stays there
        DevBuf<float> dTMP1;
        if (!dTMP1.alloc(N)) return false;
        B.synth(B.dX.p, dTMP1.p, keep->pix.p);
        HIP_TRY(hipGetLastError());
        keep->td = B.td;
    } else {
        ok = B.download(tiles);
    }
    pp.mark(2);
    pp.print(fused ? "fixed (fused)" : "fixed", B.nt, B.max_rows, B.max_cols, B.total, B.total_samples, prm.inner, prm.reweights);
    fp.B.reset();
    return ok;
}

bool solve_fixed(const std::vector<TileProblem>& tiles, const Params& prm, Kept* keep = nullptr) {
    FixedPrep fp;
    return prepare_fixed(tiles, prm, fp, keep != nullptr) && run_fixed(tiles, prm, fp, keep);
}

// tv > 0: smoothed-TV term + per-tile backtracking, as cs_fista_core_single (bt branch).
bool solve_backtrack(const std::vector<TileProblem>& tiles, const Params& prm) {
    PhaseProf pp;
    Batch B;
    if (!B.build(tiles, prm.wscale, pp, false)) return false;
    const int nt = B.nt;
    const size_t N = (size_t)B.total, bytes = N * sizeof(float);
    DevBuf<float> dY, dXP, dZ, dTMP, dPIX, dSC, dG, dEPS, dTVG, dPHI, dS, dTV, dLT;
    DevBuf<double> dF, dD2; DevBuf<int> dACT;
    if (!dY.alloc(N) || !dXP.alloc(N) || !dZ.alloc(N) || !dTMP.alloc(N) || !dPIX.alloc(N) || !dSC.alloc(N) ||
        !dG.alloc(N) || !dEPS.alloc((size_t)nt) || !dTVG.alloc(N) || !dPHI.alloc(N) || !dS.alloc(N) ||
        !dTV.alloc((size_t)nt) || !dLT.alloc((size_t)nt) || !dF.alloc((size_t)nt) || !dD2.alloc(2 * (size_t)nt) ||
        !dACT.alloc((size_t)nt)) {
        std::fprintf(stderr, "gpu_tiles: device allocation failed\n");
        return false;
    }
    std::vector<float> tv((size_t)nt), L((size_t)nt);
    for (int t = 0; t < nt; ++t) tv[(size_t)t] = tiles[(size_t)t].tv;
    HIP_TRY(hipMemcpy(dTV.p, tv.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
    std::vector<double> fy((size_t)nt), fz((size_t)nt), d2(2 * (size_t)nt);
    std::vector<int> act((size_t)nt);
    const dim3 blk(kThreads), grid = B.grid(), sgrid = B.sgrid(), tgrid((unsigned)nt);
    pp.mark(0);
    // smooth objective of the pixels in PIX: residual into SC, TV gradient into TVG, f into F
    auto smooth = [&](const float* coef) {
        B.synth(coef, dTMP.p, dPIX.p);
        (void)hipMemsetAsync(dSC.p, 0, bytes);
        if (B.total_samples > 0)
            k_scatter<<<sgrid, blk>>>(B.dtd.p, B.dst.p, B.drix.p, B.driy.p, B.db.p, dPIX.p, dSC.p, B.total_samples);
        k_tv<<<grid, blk>>>(B.dtd.p, dPIX.p, dTVG.p, dPHI.p);
        k_obj<<<tgrid, blk>>>(B.dtd.p, dSC.p, dPHI.p, dTV.p, dF.p);
    };
    for (int r = 0; r < prm.reweights; ++r) {
        HIP_TRY(hipMemcpyAsync(dY.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        HIP_TRY(hipMemcpyAsync(dXP.p, B.dX.p, bytes, hipMemcpyDeviceToDevice));
        for (int t = 0; t < nt; ++t) L[(size_t)t] = 2.0f + 8.0f * tv[(size_t)t];   // bx.lip (DCT) + 8*tv, per core call
        float tm = 1.0f;
        for (int k = 0; k < prm.inner; ++k) {
            // gradient at y: g = 2*DCT(sc) + tv*DCT(tvgrad); fy = ||res||^2 + tv*phi
            smooth(dY.p);
            k_comb<<<grid, blk>>>(B.dtd.p, dSC.p, dTVG.p, dTV.p, dS.p);
            B.adj(dS.p, dTMP.p, dG.p);
            HIP_TRY(hipMemcpy(fy.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
            std::fill(act.begin(), act.end(), 1);
            for (int trial = 0; trial < 25; ++trial) {
                HIP_TRY(hipMemcpy(dLT.p, L.data(), (size_t)nt * sizeof(float), hipMemcpyHostToDevice));
                HIP_TRY(hipMemcpy(dACT.p, act.data(), (size_t)nt * sizeof(int), hipMemcpyHostToDevice));
                k_trial<<<grid, blk>>>(B.dtd.p, dY.p, dG.p, B.dW.p, B.dWS.p, B.dLAM.p, dLT.p, dACT.p, dZ.p);
                smooth(dZ.p);
                k_dots<<<tgrid, blk>>>(B.dtd.p, dG.p, dZ.p, dY.p, dD2.p);
                HIP_TRY(hipMemcpy(fz.data(), dF.p, (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                HIP_TRY(hipMemcpy(d2.data(), dD2.p, 2 * (size_t)nt * sizeof(double), hipMemcpyDeviceToHost));
                const int left = bt_accept(nt, fy, fz, d2, L, act, trial);
                if (left == 0) break;
            }
            const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * tm * tm));
            k_mom<<<grid, blk>>>(B.dtd.p, dZ.p, dY.p, dXP.p, (tm - 1.0f) / t_new);
            tm = t_new;
        }
        HIP_TRY(hipMemcpyAsync(B.dX.p, dZ.p, bytes, hipMemcpyDeviceToDevice));
        if (r + 1 < prm.reweights) B.reweight(dZ.p, dEPS.p);
    }
    pp.mark(1);
    const bool ok = B.download(tiles);
    pp.mark(2);
    pp.print("backtrack", B.nt, B.max_rows, B.max_cols, B.total, B.total_samples, prm.inner, prm.reweights);
    return ok;
}

}  // namespace

}  // namespace gpu_tiles

namespace gpu_common {
namespace {
struct Block {
    char* base = nullptr;
    size_t cap = 0, top = 0;
    struct Piece { size_t off; bool live; };
    std::vector<Piece> pieces;   // in address order
};
struct Arena {
    std::mutex mu;
    std::vector<Block> blocks;
} g_arena;
}  // namespace

void arena_reserve(size_t bytes) {
    if (bytes == 0) return;
    void* p = nullptr;   // allocated outside the lock: stages keep using the blocks already there meanwhile
    if (hipMalloc(&p, bytes) != hipSuccess) return;
    std::lock_guard<std::mutex> lk(g_arena.mu);
    Block b;
    b.base = (char*)p;
    b.cap = bytes;
    g_arena.blocks.push_back(std::move(b));
}

void* arena_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    const size_t n = (bytes + 255) & ~(size_t)255;
    for (Block& b : g_arena.blocks) {
        if (b.cap - b.top < n) continue;
        b.pieces.push_back({ b.top, true });
        void* p = b.base + b.top;
        b.top += n;
        return p;
    }
    return nullptr;
}

bool arena_free(void* p) {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    char* c = (char*)p;
    for (Block& b : g_arena.blocks) {
        if (c < b.base || c >= b.base + b.cap) continue;
        const size_t off = (size_t)(c - b.base);
        for (auto it = b.pieces.rbegin(); it != b.pieces.rend(); ++it)
            if (it->off == off) { it->live = false; break; }
        while (!b.pieces.empty() && !b.pieces.back().live) {
            b.top = b.pieces.back().off;
            b.pieces.pop_back();
        }
        return true;
    }
    return false;
}

namespace {
struct Resident {
    std::mutex mu;
    const void* host = nullptr;
    size_t bytes = 0;
    void* dev = nullptr;
} g_res;
void dev_free(void* p) { if (p && !arena_free(p)) (void)hipFree(p); }
}  // namespace

void resident_put(const void* host, size_t bytes, void* dev) {
    std::lock_guard<std::mutex> lk(g_res.mu);
    dev_free(g_res.dev);
    g_res.host = host; g_res.bytes = bytes; g_res.dev = dev;
}

void resident_range(const void** host, size_t* bytes) {
    std::lock_guard<std::mutex> lk(g_res.mu);
    *host = g_res.dev ? g_res.host : nullptr;
    *bytes = g_res.dev ? g_res.bytes : 0;
}

void* resident_take(const void* host, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_res.mu);
    void* d = g_res.dev;
    if (!(d && host == g_res.host && bytes == g_res.bytes)) { dev_free(d); d = nullptr; }
    g_res.host = nullptr; g_res.bytes = 0; g_res.dev = nullptr;
    return d;
}
}  // namespace gpu_common

namespace gpu_tiles {

bool available() {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

void warmup() {
    const auto t0 = std::chrono::steady_clock::now();
    if (!available()) return;
    // The device arena: half the free memory, at most 3 GiB (a 12 MP decode peaks at ~2.7 GiB in the cs-sr
    // ADMM; a larger one takes its overflow from hipMalloc). A 1 GiB first block, enough for the LR solve, is
    // usable at once; the rest is reserved while that solve runs (below).
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return;
    const auto t1 = std::chrono::steady_clock::now();
    const size_t total = std::min<size_t>(free_b / 2, (size_t)3 << 30), first = std::min<size_t>(total, (size_t)1 << 30);
    gpu_common::arena_reserve(first);
    // The rest in 256 MiB blocks with a 0.5 ms pause after each: a hipMalloc holds the HIP runtime's lock (~8 ms per
    // 256 MiB), so an upload the LR solve issues meanwhile waits for one block, not for the whole reserve (~60 ms).
    for (size_t left = total - first; left > 0;) {
        const size_t b = std::min<size_t>(left, (size_t)256 << 20);
        gpu_common::arena_reserve(b);
        left -= b;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    if (const char* e = std::getenv("CS_GPU_PROF"); e && e[0] == '1')
        std::fprintf(stderr, "[gpu-prof] warmup: runtime %.1f ms, arena reserve %.1f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
}

bool solve_fista_dct(const std::vector<TileProblem>& tiles, const Params& prm) {
    if (tiles.empty()) return true;
    if (!available()) return false;
    std::vector<TileProblem> fixed, bt;
    for (const TileProblem& P : tiles) (P.tv > 0.0f ? bt : fixed).push_back(P);
    if (!fixed.empty() && !solve_fixed(fixed, prm)) return false;
    if (!bt.empty() && !solve_backtrack(bt, prm)) return false;
    return true;
}

bool solve_ycc_bgr(const std::vector<TileProblem>& planes, const std::vector<PlaneGroup>& groups, unsigned char* out,
    const std::function<void()>& gpu_begin) {
    if (planes.empty() || planes.size() % 3 || !available()) return false;
    const int nj = (int)(planes.size() / 3);
    std::vector<MergeDev> md((size_t)nj);
    long long out_bytes = 0;
    int max_n = 0;
    for (int j = 0; j < nj; ++j) {
        const TileProblem& Y = planes[3 * (size_t)j];
        const TileProblem& Cr = planes[3 * (size_t)j + 1];
        if (Cr.rows != planes[3 * (size_t)j + 2].rows || Cr.cols != planes[3 * (size_t)j + 2].cols) return false;
        md[(size_t)j] = { nullptr, nullptr, nullptr, Y.rows, Y.cols, Cr.rows, Cr.cols, out_bytes };
        out_bytes += 3LL * Y.rows * Y.cols;
        max_n = std::max(max_n, Y.rows * Y.cols);
    }
    for (const TileProblem& P : planes) if (P.tv > 0.0f) return false;
    if (out_bytes > 2000000000LL) return false;
    // Host layouts of every group first: they need no device, so the caller can take the GPU lock (gpu_begin) after them.
    std::vector<std::vector<TileProblem>> gtiles(groups.size());
    std::vector<long long> gn(groups.size(), 0);
    std::vector<FixedPrep> prep(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        for (int k : groups[g].planes) { gtiles[g].push_back(planes[(size_t)k]); gn[g] += (long long)planes[(size_t)k].rows * planes[(size_t)k].cols; }
        if (!prepare_fixed(gtiles[g], groups[g].prm, prep[g], true)) return false;
    }
    if (gpu_begin) gpu_begin();
    DevBuf<std::uint8_t> dOUT;   // first, so the arena reclaims the solves' buffers above it while it stays resident
    if (!dOUT.alloc((size_t)out_bytes)) return false;
    std::vector<Kept> kept(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        const std::vector<TileProblem>& tiles = gtiles[g];
        if (!kept[g].pix.alloc((size_t)gn[g]) || !run_fixed(tiles, groups[g].prm, prep[g], &kept[g])) return false;
        (void)hipStreamQuery(nullptr);   // submits this group's launches before the next group's upload
        for (size_t t = 0; t < tiles.size(); ++t) {
            const int k = groups[g].planes[t];
            MergeDev& M = md[(size_t)k / 3];
            (k % 3 == 0 ? M.y : k % 3 == 1 ? M.cr : M.cb) = kept[g].pix.p + kept[g].td[t].off;
        }
    }
    for (const MergeDev& M : md) if (!M.y || !M.cr || !M.cb) return false;
    DevBuf<MergeDev> dmd;
    if (!dmd.alloc(md.size())) return false;
    HIP_TRY(hipMemcpy(dmd.p, md.data(), md.size() * sizeof(MergeDev), hipMemcpyHostToDevice));
    k_ycc_merge<<<dim3((unsigned)((max_n + kThreads - 1) / kThreads), (unsigned)nj), dim3(kThreads)>>>(dmd.p, dOUT.p);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipMemcpy(out, dOUT.p, (size_t)out_bytes, hipMemcpyDeviceToHost));
    gpu_common::resident_put(out, (size_t)out_bytes, dOUT.release());
    return true;
}

}  // namespace gpu_tiles
