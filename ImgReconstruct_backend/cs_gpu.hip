// HIP implementation of the batched consensus-ADMM solve offload.
//
// Contract (cs_gpu.h): admm_solve_blocking(p) must leave p.x holding the same
// solution the CPU cs_admm_core wrapper would produce (DCT basis, tv = 0).
// Concurrent callers enqueue into a shared queue; one worker thread coalesces
// arrivals (short grace window) into batches keyed by (rows, cols, inner,
// reweights) -- m varies per tile, packed at the group max -- and runs them
// as a single GPU session:
//
//   setup   scatter b/mask2, one FDCT for Atb, z=x0/u=0, rho = 2m/n
//   pass    for each IRLS pass: [reweight kernel], then inner steps of
//             rhs = Atb + rho*(z-u)                    (kernel)
//             IDCT:  tmp = rhs*Ac;  rhs = Ar^T*tmp     (2 batched GEMMs)
//             x = rhs / (mask2 + rho)                  (kernel)
//             FDCT:  tmp = rhs*Ac^T; x = Ar*tmp        (2 batched GEMMs)
//             z = soft(x+u, lambda*w/rho); Boyd rho balance; u update
//   down   copy each problem's x back and wake its caller
//
// Transforms use a cosine-matrix GEMM (no FFT dependency); the DCT matrices
// are the orthonormal DCT-II that OpenCV's cv::dct uses, so synth/adj match
// cs_synth/cs_synth_adj. On any device error the batch is recomputed on the
// CPU through the registered fallback and the module stays dead (queue still
// serves every later call, straight to the CPU).

#include "cs_gpu.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// FFT fast-DCT (hipFFT, always on): the GEMM cosine-matrix transforms cost
// O(n*(R+C)) per tile while a separable FFT-DCT costs O(n*log n) -- at
// 504x378 that is 336 MFLOP vs ~33 MFLOP per iteration. Each 1D DCT-II/III
// of length N runs as one 2N-point real FFT (symmetric extension /
// Hermitian pre-twiddle, Makhoul) plus O(N) twiddle kernels, exact for odd
// and even N. Batched plans are cached per (length, batch); any hipFFT
// failure permanently falls back to the GEMM kernels below (g_fft_dead).

// Library GEMM experiment (rocBLAS strided-batched sgemm): measured SLOWER
// than the native tiled kernels on our shapes (86x107, batch <= 24): +234 ms
// device-side solve, +~150 ms Tensile warmup per process. Correct (53.53 dB
// vs CPU) but not worth it. Kept for reference, compiled out by default;
// flip to 1 to re-enable (needs the rocblas.lib link + rocblas.dll on PATH).
#define CS_GPU_USE_ROCBLAS 0
#if CS_GPU_USE_ROCBLAS
#include <rocblas/rocblas.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cs_gpu {
namespace {

constexpr int kBlock = 256;       // threads for elementwise / reduce kernels
constexpr int kTile = 16;         // GEMM tile edge
constexpr int kGraceUs = 800;     // straggler wait before flushing a batch
constexpr int kBatchTarget = 24;  // flush early once this many are pending

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_dead{false};  // device failed: route everything to CPU
std::atomic<bool> g_probed{false};
bool g_has_device = false;
CpuFallbackFn g_fb = nullptr;
FistaFallbackFn g_fb_fista = nullptr;

// rocBLAS library GEMM path: handle created once in set_enabled (main
// thread), used only by the worker thread afterwards. g_rb_dead flips on the
// first library error and that call (and all later ones) falls back to the
// native kernels in gemm().
#if CS_GPU_USE_ROCBLAS
rocblas_handle g_rb = nullptr;
bool g_rb_tried = false;
std::atomic<bool> g_rb_dead{false};

void rb_init()
{
    if (g_rb_tried) return;
    g_rb_tried = true;
    if (rocblas_create_handle(&g_rb) != rocblas_status_success) {
        g_rb = nullptr;
        fprintf(stderr, "[gpu] rocblas unavailable, using native GEMM kernels\n");
        return;
    }
    if (rocblas_set_stream(g_rb, 0) != rocblas_status_success) {
        rocblas_destroy_handle(g_rb);
        g_rb = nullptr;
        fprintf(stderr, "[gpu] rocblas unavailable, using native GEMM kernels\n");
        return;
    }
    // warmup: one tiny single-batch sgemm on scratch device memory so Tensile
    // init lands here, not on the first flush. Values are irrelevant.
    float *wa = nullptr, *wb = nullptr, *wc = nullptr;
    const size_t wB = 16 * 16 * 4;
    bool wok = hipMalloc((void**)&wa, wB) == hipSuccess &&
        hipMalloc((void**)&wb, wB) == hipSuccess &&
        hipMalloc((void**)&wc, wB) == hipSuccess &&
        hipMemset(wa, 0, wB) == hipSuccess &&
        hipMemset(wb, 0, wB) == hipSuccess;
    if (wok) {
        const float alpha = 1.0f, beta = 0.0f;
        wok = rocblas_sgemm_strided_batched(g_rb, rocblas_operation_none,
                  rocblas_operation_none, 16, 16, 16, &alpha, wb, 16, 0, wa,
                  16, 0, &beta, wc, 16, 256, 1) == rocblas_status_success &&
            hipStreamSynchronize(0) == hipSuccess;
    }
    if (wa) (void)hipFree(wa);
    if (wb) (void)hipFree(wb);
    if (wc) (void)hipFree(wc);
    if (!wok) {
        rocblas_destroy_handle(g_rb);
        g_rb = nullptr;
        fprintf(stderr,
            "[gpu] rocblas warmup failed, using native GEMM kernels\n");
        return;
    }
    fprintf(stderr, "[gpu] rocblas library GEMM path ready\n");
}
#else
void rb_init() {}  // library GEMM compiled out: native kernels only
#endif

// FFT fast-DCT kill switch: any hipFFT plan/exec failure sets this and all
// row/col halves permanently use the GEMM kernels. Worker thread only.
std::atomic<bool> g_fft_dead{false};


bool gpu_ok(const char* what, hipError_t e) {
    if (e == hipSuccess) return true;
    fprintf(stderr, "[gpu] %s: %s\n", what, hipGetErrorString(e));
    g_dead.store(true, std::memory_order_relaxed);
    return false;
}
#define GPU_CHECK(call)                       \
    do {                                      \
        if (!gpu_ok(#call, (call))) return false; \
    } while (0)
#define GPU_LAUNCH_CHECK() GPU_CHECK(hipGetLastError())

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// C[b] = opA(A) * opB(B) for batched matrices; A/B stride 0 broadcasts the
// same matrix across the batch (used for the shared DCT matrices).
template <int TA, int TB>
__global__ void gemm_k(float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, int M, int N, int K,
    size_t sA, size_t sB, size_t sC)
{
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    const size_t ab = (size_t)blockIdx.z * sA;
    const size_t bb = (size_t)blockIdx.z * sB;
    const size_t cb = (size_t)blockIdx.z * sC;
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += kTile) {
        const int kk = k0 + tx;
        const int kr = k0 + ty;
        As[ty][tx] = (row < M && kk < K)
            ? A[(TA ? (size_t)kk * M + row : (size_t)row * K + kk) + ab]
            : 0.0f;
        Bs[ty][tx] = (col < N && kr < K)
            ? B[(TB ? (size_t)col * K + kr : (size_t)kr * N + col) + bb]
            : 0.0f;
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kTile; ++t) acc += As[ty][t] * Bs[t][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col + cb] = acc;
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K,
    int ta, int tb, size_t sA, size_t sB, size_t sC, int batch)
{
    // Library path: row-major C(MxN) = opA(A)(MxK) * opB(B)(KxN) is issued as
    // column-major C^T(NxM) = opB(B)^T * opA(A)^T, i.e. rocBLAS operands are
    // swapped (A-arg = our B) with the transpose flags folded into lda/ldb:
    //   m=N, n=M, k=K, transA = tb?'T':'N', lda = tb?K:N (stride sB),
    //                    transB = ta?'T':'N', ldb = ta?M:K (stride sA).
    // Stride 0 broadcasts the shared DCT matrix across the batch.
#if CS_GPU_USE_ROCBLAS
    if (g_rb && !g_rb_dead.load(std::memory_order_relaxed)) {
        const float alpha = 1.0f, beta = 0.0f;
        const rocblas_operation opA =
            tb ? rocblas_operation_transpose : rocblas_operation_none;
        const rocblas_operation opB =
            ta ? rocblas_operation_transpose : rocblas_operation_none;
        const rocblas_status st = rocblas_sgemm_strided_batched(g_rb, opA, opB,
            N, M, K, &alpha, B, tb ? K : N, (rocblas_stride)sB, A,
            ta ? M : K, (rocblas_stride)sA, &beta, C, N, (rocblas_stride)sC,
            batch);
        if (st == rocblas_status_success) return;
        if (!g_rb_dead.exchange(true))
            fprintf(stderr,
                "[gpu] rocblas sgemm failed (%d), falling back to native kernels\n",
                (int)st);
    }
#endif
    // Native fallback (also the path when rocBLAS is unavailable).
    const dim3 blk(kTile, kTile, 1);
    const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile, batch);
    if (ta == 0 && tb == 0)
        hipLaunchKernelGGL((gemm_k<0, 0>), grid, blk, 0, 0, C, A, B, M, N, K, sA, sB, sC);
    else if (ta == 1 && tb == 0)
        hipLaunchKernelGGL((gemm_k<1, 0>), grid, blk, 0, 0, C, A, B, M, N, K, sA, sB, sC);
    else if (ta == 0 && tb == 1)
        hipLaunchKernelGGL((gemm_k<0, 1>), grid, blk, 0, 0, C, A, B, M, N, K, sA, sB, sC);
    else
        hipLaunchKernelGGL((gemm_k<1, 1>), grid, blk, 0, 0, C, A, B, M, N, K, sA, sB, sC);
}

// 1D DCT halves over R x C planes (batch P, stride n): synthesis (IDCT
// direction: row = src*Ac, col = Ar^T*src) and analysis (FDCT direction:
// row = src*Ac^T, col = Ar*src). Each tries the hipFFT fast path first and
// falls back to the GEMM kernel on any FFT failure. Defined after the pool
// (they need FFT scratch); prototypes here for idct/fdct_batch below.
void row_synth(const float* src, float* dst, const float* ac,
    int R, int C, int P, size_t n);
void col_synth(const float* src, float* dst, const float* ar,
    int R, int C, int P, size_t n);
void row_anlys(const float* src, float* dst, const float* ac,
    int R, int C, int P, size_t n);
void col_anlys(const float* src, float* dst, const float* ar,
    int R, int C, int P, size_t n);

// dst = Ar^T * (src * Ac): coefficient -> pixel, batched, strided n per problem.
void idct_batch(const float* ar, const float* ac, const float* src, float* tmp,
    float* dst, int R, int C, int P, size_t n)
{
    row_synth(src, tmp, ac, R, C, P, n);  // tmp = src * Ac
    col_synth(tmp, dst, ar, R, C, P, n);  // dst = Ar^T * tmp
}

// dst = Ar * (src * Ac^T): pixel -> coefficient, batched.
void fdct_batch(const float* ar, const float* ac, const float* src, float* tmp,
    float* dst, int R, int C, int P, size_t n)
{
    row_anlys(src, tmp, ac, R, C, P, n);  // tmp = src * Ac^T
    col_anlys(tmp, dst, ar, R, C, P, n);  // dst = Ar * tmp
}

// pix[idx] += 2b, mask2[idx] += 2 at every measured coordinate. One block per
// problem: m varies per tile (random sampling count), so indexing is per
// problem with the b/rix/riy arrays packed at an mmax stride.
__global__ void scatter_k(float* pix, float* mask2, const float* b2,
    const int* rix, const int* riy, const int* ms, int mmax, int cols, int n)
{
    const int p = blockIdx.x;
    const int mp = ms[p];
    const long long base = (long long)p * n;
    const long long ib = (long long)p * mmax;
    for (int k = threadIdx.x; k < mp; k += blockDim.x) {
        const int idx = (int)rix[ib + k] * cols + (int)riy[ib + k];
        atomicAdd(&pix[base + idx], b2[ib + k]);
        atomicAdd(&mask2[base + idx], 2.0f);
    }
}

__global__ void init_xu_k(float* z, float* u, const float* x, long long total)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        z[i] = x[i];
        u[i] = 0.0f;
    }
}

// Fresh-consensus penalty per reweight pass (mirrors cs_admm_core entry):
// rho = max(2*m/n, 0.05). One thread per problem.
__global__ void rho_init_k(float* rho, const int* ms, int n, int P)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float r = 2.0f * (float)ms[p] / (float)n;
    if (r < 0.05f) r = 0.05f;
    rho[p] = r;
}

// rhs = Atb + rho*(z - u)  (tv = 0, so the tau*x and TV terms vanish)
__global__ void rhs_k(float* rhs, const float* atb, const float* z,
    const float* u, const float* rho, int n)
{
    const int p = blockIdx.x;
    const float r = rho[p];
    const long long base = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        rhs[base + i] = atb[base + i] + r * (z[base + i] - u[base + i]);
}

// closed-form x-update division: v /= mask2 + sigma, sigma = rho (tv = 0)
__global__ void div_k(float* v, const float* mask2, const float* rho, int n)
{
    const int p = blockIdx.x;
    const float r = rho[p];
    const long long base = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        v[base + i] /= (mask2[base + i] + r);
}

// exact soft-threshold z, scaled-dual u update, Boyd residual balancing
// (mu = 10, tau = 2, rho clamped to [1e-3, 100] on the down branch) -- the
// same sequence and rounding pattern as cs_admm_core.
__global__ void z_dual_k(float* z, const float* x, float* u, const float* w,
    float* rho, const float* lambda, int n)
{
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_rho, s_scale;
    __shared__ double s_r[kBlock], s_s[kBlock];
    const float r0 = rho[p];
    const float lam = lambda[p];
    const long long base = (long long)p * n;
    double pr = 0.0, ps = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        const long long g = base + i;
        const float zold = z[g];
        const float v = x[g] + u[g];
        const float thr = (lam * w[g]) / r0;
        const float av = fabsf(v);
        const float znew = (av > thr) ? copysignf(av - thr, v) : 0.0f;
        z[g] = znew;
        const double rr = (double)x[g] - (double)znew;
        const double ss = (double)r0 * ((double)znew - (double)zold);
        pr += rr * rr;
        ps += ss * ss;
    }
    s_r[tid] = pr;
    s_s[tid] = ps;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_r[tid] += s_r[tid + off];
            s_s[tid] += s_s[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double rn = sqrt(s_r[0]);
        const double sn = sqrt(s_s[0]);
        float nr = r0;
        if (rn > 10.0 * sn && r0 < 100.0f)
            nr = r0 * 2.0f;
        else if (sn > 10.0 * rn && r0 > 1e-3f)
            nr = r0 * 0.5f;
        s_scale = r0 / nr;
        s_rho = nr;
    }
    __syncthreads();
    const float sc = s_scale;
    for (int i = tid; i < n; i += blockDim.x) {
        const long long g = base + i;
        u[g] = (u[g] + (x[g] - z[g])) * sc;
    }
    if (tid == 0) rho[p] = s_rho;
}

// Sampled synthesis (fused IDCT tail + residual): r[k] = synth(y)[pos_k] -
// b[k] WITHOUT materializing the full image. tmp = y*Ac is still a full
// GEMM (all columns feed some position); the Ar^T stage collapses to one
// dot product per sampled position (m*R mults vs R*R*C dense). Reads y, tmp
// (L2-resident at tile sizes), Ar; writes compact r. Deterministic, no
// atomics (positions distinct by construction).
__global__ void synth_sample_k(float* r, const float* tmp, const float* ar,
    const float* b, const int* rix, const int* riy, const int* ms, int mmax,
    int R, int C)
{
    const int p = blockIdx.x;
    const int mp = ms[p];
    const long long tb = (long long)p * R * C;
    const long long ib = (long long)p * mmax;
    for (int k = threadIdx.x; k < mp; k += blockDim.x) {
        const int ix = (int)rix[ib + k], iy = (int)riy[ib + k];
        float acc = 0.0f;
        for (int j = 0; j < R; ++j) acc += ar[(size_t)j * R + ix] * tmp[tb + (size_t)j * C + iy];
        r[ib + k] = acc - b[ib + k];
    }
}

// Sparse adjoint head: tmp2 = sc*Ac^T where sc is nonzero only at measured
// positions (row-grouped: row_ptr segments the measurement order by row).
// tmp2[i,j] = sum over row-i measurements of r[k]*Ac[j,col_k]. Writes all
// n entries (rows without measurements get 0), so no memset is needed.
// GEMM: m*C mults vs R*C*C dense. Reads compact r + grouped indices.
__global__ void sadjoint_k(float* tmp2, const float* r,
    const int* riy, const int* row_ptr, const float* ac,
    const int* ms, int mmax, int R, int C)
{
    const int p = blockIdx.x;
    const long long tb = (long long)p * R * C;
    const long long ib = (long long)p * mmax;
    const long long rb = (long long)p * (R + 1);
    const int n = R * C;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const int i = t / C, j = t % C;
        float acc = 0.0f;
        for (int k = row_ptr[rb + i]; k < row_ptr[rb + i + 1]; ++k)
            acc += r[ib + k] * ac[(size_t)j * C + riy[ib + k]];
        tmp2[tb + t] = acc;
    }
}

// Isotropic TV gradient on the full pixel image (gather formulation of
// cs_tv_grad_phi, race-free: each thread computes its own pixel from the
// forward-difference cells that touch it). For pixel (i,j):
//   own cell (i,j):      a = pix[i+1,j]-pix[i,j] (0 at bottom edge),
//                        b = pix[i,j+1]-pix[i,j] (0 at right edge),
//                        acc -= a/d + b/d
//   cell above (i-1,j):  its down-edge add aT/dT  (i > 0)
//   cell left (i,j-1):   its right-edge add bL/dL (j > 0)
// d = sqrt(a^2 + b^2 + eps^2), eps = 1e-3, same as the CPU.
__global__ void tvgrad_k(float* tvg, const float* pix, int R, int C)
{
    const int p = blockIdx.x;
    const long long tb = (long long)p * R * C;
    const int n = R * C;
    constexpr float eps = 1e-3f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const int i = t / C, j = t % C;
        const float c = pix[tb + t];
        float acc = 0.0f;
        {
            const float a = (i + 1 < R) ? pix[tb + t + C] - c : 0.0f;
            const float b = (j + 1 < C) ? pix[tb + t + 1] - c : 0.0f;
            const float d = sqrtf(a * a + b * b + eps * eps);
            acc -= a / d + b / d;
        }
        if (i > 0) {
            const float up = pix[tb + t - C];
            const float aT = c - up;
            const float bT = (j + 1 < C) ? pix[tb + t - C + 1] - up : 0.0f;
            const float dT = sqrtf(aT * aT + bT * bT + eps * eps);
            acc += aT / dT;
        }
        if (j > 0) {
            const float lf = pix[tb + t - 1];
            const float aL = (i + 1 < R) ? pix[tb + t - 1 + C] - lf : 0.0f;
            const float bL = c - lf;
            const float dL = sqrtf(aL * aL + bL * bL + eps * eps);
            acc += bL / dL;
        }
        tvg[tb + t] = acc;
    }
}

// Fused TV shrink + momentum: same formulas as the CPU DCT core with
// tv > 0 (L = 2 + 8*tv, step = 1/L):
//   v = y - step*(2*gd + tv*gt); z = soft(v, lam*step*w); y = z+mom*(z-xp).
// gd is the sampled data gradient (Fdct of the scattered residual, without
// the x2 the CPU folds into g), gt the TV adjoint. tv = 0 reproduces
// shrink_mom_k up to one reassociated multiply.
__global__ void shrink_tvmom_k(float* z, float* y, const float* gd,
    const float* gt, const float* w, const float* lam, const float* tv,
    float* xp, float mom, int n)
{
    const int p = blockIdx.x;
    const float tvp = tv[p];
    const float step = 1.0f / (2.0f + 8.0f * tvp);
    const float bs = lam[p] * step;
    const long long bse = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const long long g = bse + i;
        const float v = y[g] - step * (2.0f * gd[g] + tvp * gt[g]);
        const float thr = bs * w[g];
        const float av = fabsf(v);
        const float zv = (av > thr) ? copysignf(av - thr, v) : 0.0f;
        z[g] = zv;
        y[g] = zv + mom * (zv - xp[g]);
        xp[g] = zv;
    }
}

// TV twin of shrink_joint_mom_k: row-coupled group shrink with the TV
// gradient folded per plane (same cs_fista_core_joint math, DCT).
__global__ void shrink_joint_tvmom_k(float* z0, float* z1, float* z2,
    float* y0, float* y1, float* y2,
    const float* gd0, const float* gd1, const float* gd2,
    const float* gt0, const float* gt1, const float* gt2,
    const float* w, const float* lam, const float* tv,
    float* xp0, float* xp1, float* xp2, float mom, int n)
{
    const int p = blockIdx.x;
    const float tvp = tv[p];
    const float step = 1.0f / (2.0f + 8.0f * tvp);
    const float bs = lam[p] * step;
    const long long bse = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const long long g = bse + i;
        const float v0 = y0[g] - step * (2.0f * gd0[g] + tvp * gt0[g]);
        const float v1 = y1[g] - step * (2.0f * gd1[g] + tvp * gt1[g]);
        const float v2 = y2[g] - step * (2.0f * gd2[g] + tvp * gt2[g]);
        const float rn = sqrtf(v0 * v0 + v1 * v1 + v2 * v2);
        const float thr = bs * w[g];
        const float s = (rn > thr && rn > 0.0f) ? (1.0f - thr / rn) : 0.0f;
        const float z0v = v0 * s, z1v = v1 * s, z2v = v2 * s;
        z0[g] = z0v; z1[g] = z1v; z2[g] = z2v;
        y0[g] = z0v + mom * (z0v - xp0[g]); xp0[g] = z0v;
        y1[g] = z1v + mom * (z1v - xp1[g]); xp1[g] = z1v;
        y2[g] = z2v + mom * (z2v - xp2[g]); xp2[g] = z2v;
    }
}

// Fused shrink + momentum (one launch, one pass instead of two): same ops
// as shrink_k followed by momentum_k, with z kept in registers. v = y - g
// and base = lam/2 fold step = 1/L = 1/2 exactly (DCT/tv=0); wscale is ones.
__global__ void shrink_mom_k(float* z, float* y, const float* grad,
    const float* w, const float* lam, float* xp, float mom, int n)
{
    const int p = blockIdx.x;
    const float bs = 0.5f * lam[p];
    const long long bse = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const long long g = bse + i;
        const float v = y[g] - grad[g];
        const float thr = bs * w[g];
        const float av = fabsf(v);
        const float zv = (av > thr) ? copysignf(av - thr, v) : 0.0f;
        z[g] = zv;
        y[g] = zv + mom * (zv - xp[g]);
        xp[g] = zv;
    }
}

// Fused row-coupled group shrink + per-plane momentum (single launch).
// Same formulas as shrink_joint_k + 3x momentum_k with zc in registers.
__global__ void shrink_joint_mom_k(float* z0, float* z1, float* z2,
    float* y0, float* y1, float* y2,
    const float* g0, const float* g1, const float* g2,
    const float* w, const float* lam,
    float* xp0, float* xp1, float* xp2, float mom, int n)
{
    const int p = blockIdx.x;
    const float bs = 0.5f * lam[p];
    const long long bse = (long long)p * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const long long g = bse + i;
        const float v0 = y0[g] - g0[g];
        const float v1 = y1[g] - g1[g];
        const float v2 = y2[g] - g2[g];
        const float rn = sqrtf(v0 * v0 + v1 * v1 + v2 * v2);
        const float thr = bs * w[g];
        const float s = (rn > thr && rn > 0.0f) ? (1.0f - thr / rn) : 0.0f;
        const float z0v = v0 * s, z1v = v1 * s, z2v = v2 * s;
        z0[g] = z0v; z1[g] = z1v; z2[g] = z2v;
        y0[g] = z0v + mom * (z0v - xp0[g]); xp0[g] = z0v;
        y1[g] = z1v + mom * (z1v - xp1[g]); xp1[g] = z1v;
        y2[g] = z2v + mom * (z2v - xp2[g]); xp2[g] = z2v;
    }
}

// Joint IRLS pass over row norms: eps = max(0.02 * max||row||_2, 1e-3);
// w_i = eps / (rn_i + eps), shared across the 3 planes.
__global__ void reweight_joint_k(float* w, const float* x0, const float* x1,
    const float* x2, int n)
{
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_red[kBlock], s_eps;
    const long long base = (long long)p * n;
    float mx = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const long long g = base + i;
        const float rn = sqrtf(x0[g] * x0[g] + x1[g] * x1[g] + x2[g] * x2[g]);
        if (rn > mx) mx = rn;
    }
    s_red[tid] = mx;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (tid < off) s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) {
        float eps = 0.02f * s_red[0];
        if (eps < 1e-3f) eps = 1e-3f;
        s_eps = eps;
    }
    __syncthreads();
    const float eps = s_eps;
    for (int i = tid; i < n; i += blockDim.x) {
        const long long g = base + i;
        const float rn = sqrtf(x0[g] * x0[g] + x1[g] * x1[g] + x2[g] * x2[g]);
        w[g] = eps / (rn + eps);
    }
}
__global__ void reweight_k(float* w, const float* x, int n)
{
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_red[kBlock], s_eps;
    const long long base = (long long)p * n;
    float mx = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float a = fabsf(x[base + i]);
        if (a > mx) mx = a;
    }
    s_red[tid] = mx;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (tid < off) s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) {
        float eps = 0.02f * s_red[0];
        if (eps < 1e-3f) eps = 1e-3f;
        s_eps = eps;
    }
    __syncthreads();
    const float eps = s_eps;
    for (int i = tid; i < n; i += blockDim.x) {
        const long long g = base + i;
        w[g] = eps / (fabsf(x[g]) + eps);
    }
}

// ---------------------------------------------------------------------------
// DCT matrices (cached per tile shape)
// ---------------------------------------------------------------------------

void build_dct_host(std::vector<float>& A, int N)
{
    A.resize((size_t)N * (size_t)N);
    const double pi = 3.14159265358979323846;
    const double s0 = sqrt(1.0 / N);
    const double s = sqrt(2.0 / N);
    for (int k = 0; k < N; ++k) {
        const double ck = (k == 0) ? s0 : s;
        for (int j = 0; j < N; ++j)
            A[(size_t)k * N + j] =
                (float)(ck * cos(pi * (2 * j + 1) * k / (2.0 * N)));
    }
}

struct DctPair {
    float* ar = nullptr;
    float* ac = nullptr;
};
std::unordered_map<long long, DctPair> g_dct;  // worker thread only

bool dct_get(int rows, int cols, float*& ar, float*& ac)
{
    const long long key = ((long long)rows << 32) | (unsigned)cols;
    auto it = g_dct.find(key);
    if (it == g_dct.end()) {
        DctPair p;
        std::vector<float> hr, hc;
        build_dct_host(hr, rows);
        build_dct_host(hc, cols);
        if (!gpu_ok("hipMalloc Ar", hipMalloc((void**)&p.ar, hr.size() * 4)))
            return false;
        if (!gpu_ok("hipMalloc Ac", hipMalloc((void**)&p.ac, hc.size() * 4))) {
            (void)hipFree(p.ar);
            return false;
        }
        if (!gpu_ok("upload Ar",
                hipMemcpy(p.ar, hr.data(), hr.size() * 4, hipMemcpyHostToDevice)) ||
            !gpu_ok("upload Ac",
                hipMemcpy(p.ac, hc.data(), hc.size() * 4, hipMemcpyHostToDevice))) {
            (void)hipFree(p.ar);
            (void)hipFree(p.ac);
            return false;
        }
        it = g_dct.emplace(key, p).first;
    }
    ar = it->second.ar;
    ac = it->second.ac;
    return true;
}

// ---------------------------------------------------------------------------
// queue + worker (state intentionally never destroyed: the worker outlives
// static destructors at process exit)
// ---------------------------------------------------------------------------

struct Completion {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
};

struct Job {
    int kind;  // 0 = ADMM (p), 1 = FISTA (fp); same device layout either way
    AdmmProblem p;
    FistaProblem fp;
    std::shared_ptr<Completion> c;
};

struct QState {
    std::mutex m;
    std::condition_variable cv;
    std::deque<Job> q;
    bool started = false;
    // lock-free queue-size mirror: the worker's grace window spins on this so
    // enqueuers are never blocked and timing is not at the mercy of the
    // ~15.6 ms Windows timer granularity (cv.wait_until overshoots 0.8ms->15ms)
    std::atomic<int> pend{0};
};
QState& Q()
{
    static QState* s = new QState;
    return *s;
}

void finish(Job& j, bool use_fb)
{
    std::lock_guard<std::mutex> lk(j.c->m);
    if (j.c->done) return;
    if (use_fb) {
        if (j.kind == 1) {
            if (g_fb_fista) g_fb_fista(j.fp);
        } else if (g_fb) {
            g_fb(j.p);
        }
    }
    j.c->done = true;
    j.c->cv.notify_all();
}

struct Key {
    int kind, joint, rows, cols, inner, rews;
    explicit Key(const AdmmProblem& p)
        : kind(0), joint(0), rows(p.rows), cols(p.cols), inner(p.inner), rews(p.reweights)
    {
    }
    explicit Key(const FistaProblem& p)
        : kind(1), joint(p.joint), rows(p.rows), cols(p.cols), inner(p.inner), rews(p.reweights)
    {
    }
    bool operator<(const Key& o) const
    {
        if (kind != o.kind) return kind < o.kind;
        if (joint != o.joint) return joint < o.joint;
        if (rows != o.rows) return rows < o.rows;
        if (cols != o.cols) return cols < o.cols;
        if (inner != o.inner) return inner < o.inner;
        return rews < o.rews;
    }
};

bool dbg_flag()
{
    static int dbg = -1;
    if (dbg < 0) {
        const char* e = std::getenv("CS_GPU_DEBUG");
        dbg = (e && *e && *e != '0') ? 1 : 0;
    }
    return dbg == 1;
}

// grow-only pooled device buffers: steady state does zero alloc/free per flush
template <typename T> struct Buf {
    T* p = nullptr;
    size_t cap = 0;
    bool ensure(size_t bytes)
    {
        if (cap >= bytes) return true;
        if (p) (void)hipFree(p);
        p = nullptr;
        cap = 0;
        if (hipMalloc((void**)&p, bytes) != hipSuccess) {
            p = nullptr;
            return false;
        }
        cap = bytes;
        return true;
    }
};
struct Pool {
    Buf<float> din;  // packed H2D: rho,lam,ms,b,rix,riy,x,w (byte layout below)
    Buf<float> z, u, pix, rhs, atb, mask2;
    Buf<float> tmp;  // FISTA GEMM scratch (ADMM reuses rhs/atb in place)
    Buf<int> rowptr;  // FISTA row-grouped measurement index (P*(R+1) ints)
    Buf<float> tv0, tv1;  // TV branch: full-pix image + TV grad/adjoint (PnB)
    Buf<float> tg0, tg1, tg2;  // joint TV branch: per-plane TV adjoints (PnB)
    Buf<float> ffts;  // FFT-DCT scratch: lines*(2N+2) floats per phase
    // joint stacked planes (one PnB each); sequential per-plane use shares
    // pix/sc/tmp with the single path
    Buf<float> y1, y2, xp1, xp2, z1, z2, g1, g2;
};
Pool& pool()
{
    static Pool* q = new Pool;  // worker-lifetime; never freed at exit
    return *q;
}
struct Pin {
    void* p = nullptr;
    size_t cap = 0;
    bool ensure(size_t bytes)
    {
        if (cap >= bytes) return true;
        if (p) (void)hipHostFree(p);
        p = nullptr;
        cap = 0;
        if (hipHostMalloc(&p, bytes) != hipSuccess) {
            p = nullptr;
            return false;
        }
        cap = bytes;
        return true;
    }
};
Pin& pin()
{
    static Pin* q = new Pin;
    return *q;
}

// Forward: row-grouped counting sort of one problem's measurements, defined
// near run_jobs (worker thread only). Reorders (b,rix,riy), fills row_ptr.
void sort_measurements_by_row(float* b, int* rix, int* riy, int m, int R,
    int* row_ptr);

// Forward: row-grouped counting sort of one problem's measurements, defined
// near run_jobs (worker thread only). Reorders (b,rix,riy), fills row_ptr.
void sort_measurements_by_row(float* b, int* rix, int* riy, int m, int R,
    int* row_ptr);

// ---------------------------------------------------------------------------
// FFT fast-DCT: one batched 1D DCT-II (analysis, fwd) or DCT-III (synthesis,
// inv) over `lines` lines of length N via a single 2N-point real FFT
// (Makhoul symmetric extension / Hermitian pre-twiddle). Row lines are the
// R contiguous rows of each problem (split=R); column lines the C strided
// columns (split=C). Scratch holds lines*(2N+2) floats (unpadded R2C needs
// the +2). All worker-thread state, grow-only like the pool.
// ---------------------------------------------------------------------------

struct FftCaches {
    std::map<std::pair<int, int>, hipfftHandle> fwd;  // (M, batch) R2C plans
    std::map<std::pair<int, int>, hipfftHandle> inv;  // (M, batch) C2R plans
    struct Tw { float2* f = nullptr; float2* g = nullptr; };
    std::unordered_map<int, Tw> tw;  // twiddle pairs by N
};
FftCaches& fftc()
{
    static FftCaches* c = new FftCaches;  // worker-lifetime
    return *c;
}

// FFT suitability: rocFFT flies on 2/3/5/7-smooth lengths and falls back to
// slow Bluestein/chirp paths on large prime factors (M=518=2*7*37 measured
// ~20x per-exec vs smooth sizes). Only route smooth lengths to FFT; the
// GEMM halves stay competitive on awkward sizes. M is always even (M=2N).
bool fft_smooth(int M)
{
    int m = M;
    for (int p = 2; p * p <= m; p += (p == 2 ? 1 : 2)) {
        if (m % p == 0) {
            if (p > 7) return false;
            do {
                m /= p;
            } while (m % p == 0);
        }
    }
    return m <= 7;
}

// Escape hatch: CS_GPU_FFT=0 forces the GEMM kernels (A/B benchmarking,
// awkward-shape insurance). Checked once per process.
bool fft_disabled()
{
    static int off = -1;
    if (off < 0) {
        const char* e = std::getenv("CS_GPU_FFT");
        off = (e && *e == '0') ? 1 : 0;
    }
    return off == 1;
}

bool fft_twiddle(int N, float2*& fwd, float2*& inv){
    auto& tc = fftc().tw;
    auto it = tc.find(N);
    if (it != tc.end()) {
        fwd = it->second.f;
        inv = it->second.g;
        return true;
    }
    const double pi = 3.14159265358979323846;
    const double c0 = sqrt(1.0 / N), c1 = sqrt(2.0 / N);
    std::vector<float2> hf((size_t)N), hg((size_t)N);
    for (int k = 0; k < N; ++k) {
        const double th = pi * (double)k / (2.0 * N);
        const double co = cos(th), si = sin(th);
        const double ck = (k == 0) ? c0 : c1;
        hf[(size_t)k] = { (float)(ck * co * 0.5), (float)(ck * si * 0.5) };
        hg[(size_t)k] = { (float)(2.0 * co / ck), (float)(2.0 * si / ck) };
    }
    float2* df = nullptr;
    float2* dg = nullptr;
    if (hipMalloc((void**)&df, (size_t)N * 8) != hipSuccess ||
        hipMalloc((void**)&dg, (size_t)N * 8) != hipSuccess ||
        hipMemcpy(df, hf.data(), (size_t)N * 8, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(dg, hg.data(), (size_t)N * 8, hipMemcpyHostToDevice) != hipSuccess) {
        if (df) (void)hipFree(df);
        if (dg) (void)hipFree(dg);
        fprintf(stderr, "[gpu] fft twiddle alloc failed, using GEMM kernels\n");
        g_fft_dead.store(true, std::memory_order_relaxed);
        return false;
    }
    tc[N] = { df, dg };
    fwd = df;
    inv = dg;
    return true;
}

bool fft_plan(int M, int batch, bool fwd, hipfftHandle& h)
{
    auto& pc = fwd ? fftc().fwd : fftc().inv;
    const auto key = std::make_pair(M, batch);
    auto it = pc.find(key);
    if (it != pc.end()) {
        h = it->second;
        return true;
    }
    const int N = M / 2;
    int n[1] = { M };
    hipfftResult rc;
    const auto pq0 = std::chrono::steady_clock::now();
    if (fwd) {
        int inem[1] = { M + 2 }, onem[1] = { N + 1 };
        rc = hipfftPlanMany(&h, 1, n, inem, 1, M + 2, onem, 1, N + 1,
            HIPFFT_R2C, batch);
    } else {
        int inem[1] = { N + 1 }, onem[1] = { M + 2 };
        rc = hipfftPlanMany(&h, 1, n, inem, 1, N + 1, onem, 1, M + 2,
            HIPFFT_C2R, batch);
    }
    if (rc != HIPFFT_SUCCESS) {
        fprintf(stderr, "[gpu] hipfft plan failed (%d), using GEMM kernels\n",
            (int)rc);
        g_fft_dead.store(true, std::memory_order_relaxed);
        return false;
    }
    pc[key] = h;
    {
        const auto pq1 = std::chrono::steady_clock::now();
        const double qms =
            std::chrono::duration<double, std::milli>(pq1 - pq0).count();
        fprintf(stderr,
            "[gpu] hipfft plan M=%d batch=%d %s took %.0f ms\n",
            M, batch, fwd ? "R2C" : "C2R", qms);
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[gpu] fft fast-DCT path ready (M=%d batch=%d)\n",
                M, batch);
        }
    }
    return true;
}

// Symmetric extension into padded FFT lines: buf[l*(2N+2)+i] = in[g(i)] with
// g(i) = i < N ? i : 2N-1-i. One block per line.
__global__ void fft_ext_k(float* buf, const float* in, int lines, int N,
    int split, size_t n, int C, bool isRow)
{
    const int l = blockIdx.x;
    if (l >= lines) return;
    const int M = 2 * N, LP = M + 2;
    const int p = l / split, e = l - p * split;
    const long long ib =
        (long long)p * n + (isRow ? (long long)e * C : e);
    const int ie = isRow ? 1 : C;
    const long long ob = (long long)l * LP;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        const int g = (i < N) ? i : M - 1 - i;
        buf[ob + i] = in[ib + (long long)g * ie];
    }
}

// Analysis twiddle: dst = Re-scaled spectrum, X[k] = a[k]*Yr[k]+b[k]*Yi[k].
__global__ void fft_twf_k(float* dst, const float* buf, const float2* tw,
    int lines, int N, int split, size_t n, int C, bool isRow)
{
    const int l = blockIdx.x;
    if (l >= lines) return;
    const int LP = 2 * N + 2;
    const int p = l / split, e = l - p * split;
    const long long ob =
        (long long)p * n + (isRow ? (long long)e * C : e);
    const int oe = isRow ? 1 : C;
    const float2* S = (const float2*)(buf + (long long)l * LP);
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        const float2 t = tw[k];
        dst[ob + (long long)k * oe] = t.x * S[k].x + t.y * S[k].y;
    }
}

// Synthesis pre-twiddle: Hermitian spectrum Z[k] = X[k]*(a[k]+j*b[k]), Z[N]=0.
__global__ void fft_prei_k(float* buf, const float* X, const float2* tw,
    int lines, int N, int split, size_t n, int C, bool isRow)
{
    const int l = blockIdx.x;
    if (l >= lines) return;
    const int LP = 2 * N + 2;
    const int p = l / split, e = l - p * split;
    const long long ib =
        (long long)p * n + (isRow ? (long long)e * C : e);
    const int ie = isRow ? 1 : C;
    float2* Z = (float2*)(buf + (long long)l * LP);
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        const float xv = X[ib + (long long)k * ie];
        const float2 t = tw[k];
        Z[k] = { xv * t.x, xv * t.y };
    }
    if (threadIdx.x == 0) Z[N] = { 0.0f, 0.0f };
}

// Synthesis extract: dst = time-domain first-N samples scaled by 1/(2N).
__global__ void fft_exti_k(float* dst, const float* buf,
    int lines, int N, int split, size_t n, int C, bool isRow)
{
    const int l = blockIdx.x;
    if (l >= lines) return;
    const int LP = 2 * N + 2;
    const int p = l / split, e = l - p * split;
    const long long ob =
        (long long)p * n + (isRow ? (long long)e * C : e);
    const int oe = isRow ? 1 : C;
    const float sc = 1.0f / (float)(2 * N);
    const long long ib = (long long)l * LP;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        dst[ob + (long long)i * oe] = buf[ib + i] * sc;
}

// One batched 1D stage over R x C planes (batch P, stride n): fwd selects
// analysis (DCT-II), inv synthesis (DCT-III); isRow selects contiguous rows
// (split=R) vs strided columns (split=C). False = use the GEMM fallback.
bool fft_lines(const float* src, float* dst, int lines, int N, int split,
    size_t n, int C, bool isRow, bool fwd)
{
    if (lines <= 0) return true;
    const int M = 2 * N, LP = M + 2;
    // Batch quantum: plan cache is keyed by batch, and P churns every flush.
    // Rounding up bounds the plan count (~batchQ/1024 distinct); the FFT
    // harmlessly transforms the padding lines (kernels only touch `lines`).
    const int batchQ = ((lines + 1023) / 1024) * 1024;
    float2 *twf = nullptr, *twi = nullptr;
    if (!fft_twiddle(N, twf, twi)) return false;
    hipfftHandle plan = nullptr;
    if (!fft_plan(M, batchQ, fwd, plan)) return false;
    Pool& pl = pool();
    if (!pl.ffts.ensure((size_t)batchQ * LP * 4)) {
        g_fft_dead.store(true, std::memory_order_relaxed);
        return false;
    }
    float* s = pl.ffts.p;
    if (fwd) {
        hipLaunchKernelGGL(fft_ext_k, dim3(lines), dim3(kBlock), 0, 0,
            s, src, lines, N, split, n, C, isRow);
        if (hipGetLastError() != hipSuccess) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
        if (hipfftExecR2C(plan, (hipfftReal*)s, (hipfftComplex*)s) !=
            HIPFFT_SUCCESS) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
        hipLaunchKernelGGL(fft_twf_k, dim3(lines), dim3(kBlock), 0, 0,
            dst, s, twf, lines, N, split, n, C, isRow);
        if (hipGetLastError() != hipSuccess) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
    } else {
        hipLaunchKernelGGL(fft_prei_k, dim3(lines), dim3(kBlock), 0, 0,
            s, src, twi, lines, N, split, n, C, isRow);
        if (hipGetLastError() != hipSuccess) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
        if (hipfftExecC2R(plan, (hipfftComplex*)s, (hipfftReal*)s) !=
            HIPFFT_SUCCESS) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
        hipLaunchKernelGGL(fft_exti_k, dim3(lines), dim3(kBlock), 0, 0,
            dst, s, lines, N, split, n, C, isRow);
        if (hipGetLastError() != hipSuccess) {
            g_fft_dead.store(true, std::memory_order_relaxed);
            return false;
        }
    }
    return true;
}

// Row/col halves: synthesis (IDCT direction: row = src*Ac, col = Ar^T*src)
// and analysis (FDCT direction: row = src*Ac^T, col = Ar*src). FFT first,
// GEMM fallback (exact previous behavior) on any FFT failure.
void row_synth(const float* src, float* dst, const float* ac,
    int R, int C, int P, size_t n)
{
    if (!g_fft_dead.load(std::memory_order_relaxed) && !fft_disabled() &&
        fft_smooth(2 * C) &&
        fft_lines(src, dst, P * R, C, R, n, C, true, false))
        return;
    gemm(src, ac, dst, R, C, C, 0, 0, n, 0, n, P);  // tmp = src * Ac
}

void col_synth(const float* src, float* dst, const float* ar,
    int R, int C, int P, size_t n)
{
    if (!g_fft_dead.load(std::memory_order_relaxed) && !fft_disabled() &&
        fft_smooth(2 * R) &&
        fft_lines(src, dst, P * C, R, C, n, C, false, false))
        return;
    gemm(ar, src, dst, R, C, R, 1, 0, 0, n, n, P);  // dst = Ar^T * tmp
}

void row_anlys(const float* src, float* dst, const float* ac,
    int R, int C, int P, size_t n)
{
    if (!g_fft_dead.load(std::memory_order_relaxed) && !fft_disabled() &&
        fft_smooth(2 * C) &&
        fft_lines(src, dst, P * R, C, R, n, C, true, true))
        return;
    gemm(src, ac, dst, R, C, C, 0, 1, n, 0, n, P);  // tmp = src * Ac^T
}

void col_anlys(const float* src, float* dst, const float* ar,
    int R, int C, int P, size_t n)
{
    if (!g_fft_dead.load(std::memory_order_relaxed) && !fft_disabled() &&
        fft_smooth(2 * R) &&
        fft_lines(src, dst, P * C, R, C, n, C, false, true))
        return;
    gemm(ar, src, dst, R, C, R, 0, 0, 0, n, n, P);  // dst = Ar * tmp
}

bool run_group(std::vector<Job*>& jobs, const Key& key)
{
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const bool dbg = dbg_flag();
    const int P = (int)jobs.size();
    const int n = key.rows * key.cols;
    // m varies per tile; pack measurements at the group max so one stride works
    int mmax = 0;
    for (int q = 0; q < P; ++q)
        if (jobs[q]->p.m > mmax) mmax = jobs[q]->p.m;
    if (P <= 0 || n <= 0 || mmax <= 0 || key.inner <= 0 || key.rews <= 0) return false;
    const size_t nB = (size_t)n * 4, mB = (size_t)mmax * 4;
    const size_t PnB = (size_t)P * nB, PmB = (size_t)P * mB;
    // one packed buffer, host and device share this layout (bytes):
    // [rho P][lam P][ms P][b P*mmax][rix P*mmax][riy P*mmax][x P*n][w P*n]
    const size_t offLam = (size_t)P * 4;
    const size_t offMs = (size_t)P * 8;
    const size_t offB = (size_t)P * 12;
    const size_t offRix = offB + PmB;
    const size_t offRiy = offRix + PmB;
    const size_t offX = offRiy + PmB;
    const size_t offW = offX + PnB;
    const size_t total = offW + PnB;
    const clk::time_point t0 = clk::now();

    Pool& pl = pool();
    if (!pl.din.ensure(total) || !pl.z.ensure(PnB) || !pl.u.ensure(PnB) ||
        !pl.pix.ensure(PnB) || !pl.rhs.ensure(PnB) || !pl.atb.ensure(PnB) ||
        !pl.mask2.ensure(PnB))
        return false;
    if (!pin().ensure(total)) return false;

    char* hst = (char*)pin().p;
    float* h_rho = (float*)(hst + 0);
    float* h_lam = (float*)(hst + offLam);
    int* h_ms = (int*)(hst + offMs);
    float* h_b = (float*)(hst + offB);
    int* h_rix = (int*)(hst + offRix);
    int* h_riy = (int*)(hst + offRiy);
    float* h_x = (float*)(hst + offX);
    float* h_w = (float*)(hst + offW);
    // fill host staging; b pre-scaled x2 so Atb = 2*S^T P^T b without a pass.
    // bytes beyond p.m in b/rix/riy are never read (scatter loops k < ms[p]).
    for (int q = 0; q < P; ++q) {
        const AdmmProblem& p = jobs[q]->p;
        float* bd = h_b + (size_t)q * mmax;
        for (int k = 0; k < p.m; ++k) bd[k] = 2.0f * p.b[k];
        std::memcpy(h_rix + (size_t)q * mmax, p.rix, (size_t)p.m * 4);
        std::memcpy(h_riy + (size_t)q * mmax, p.riy, (size_t)p.m * 4);
        std::memcpy(h_x + (size_t)q * n, p.x, nB);
        float rho = 2.0f * (float)p.m / (float)n;
        h_rho[q] = (rho < 0.05f) ? 0.05f : rho;
        h_lam[q] = p.lambda;
        h_ms[q] = p.m;
    }
    for (size_t i = 0; i < PnB / 4; ++i) h_w[i] = 1.0f;

    char* dst = (char*)pl.din.p;
    float* d_rho = (float*)(dst + 0);
    float* d_lam = (float*)(dst + offLam);
    int* d_ms = (int*)(dst + offMs);
    float* d_b = (float*)(dst + offB);
    int* d_rix = (int*)(dst + offRix);
    int* d_riy = (int*)(dst + offRiy);
    float* d_x = (float*)(dst + offX);
    float* d_w = (float*)(dst + offW);

    float* ar = nullptr;
    float* ac = nullptr;
    if (!dct_get(key.rows, key.cols, ar, ac)) return false;

    GPU_CHECK(hipMemcpyAsync(pl.din.p, hst, total, hipMemcpyHostToDevice, 0));

    // setup: scatter into pix/mask2, Atb = FDCT(scatter), z = x0, u = 0
    GPU_CHECK(hipMemsetAsync(pl.pix.p, 0, PnB, 0));
    GPU_CHECK(hipMemsetAsync(pl.mask2.p, 0, PnB, 0));
    hipLaunchKernelGGL(scatter_k, dim3(P), dim3(kBlock), 0, 0,
        pl.pix.p, pl.mask2.p, d_b, d_rix, d_riy, d_ms, mmax, key.cols, n);
    {
        const long long totalX = (long long)P * n;
        const int grid = (int)((totalX + kBlock - 1) / kBlock);
        hipLaunchKernelGGL(init_xu_k, dim3(grid > 0 ? grid : 1), dim3(kBlock),
            0, 0, pl.z.p, pl.u.p, d_x, totalX);
    }
    GPU_LAUNCH_CHECK();
    fdct_batch(ar, ac, pl.pix.p, pl.rhs.p, pl.atb.p, key.rows, key.cols, P, n);
    GPU_LAUNCH_CHECK();
    const clk::time_point t1 = dbg ? clk::now() : t0;
    if (dbg && !gpu_ok("sync setup", hipStreamSynchronize(0))) return false;

    for (int r = 0; r < key.rews; ++r) {
        if (r > 0) {
            hipLaunchKernelGGL(reweight_k, dim3(P), dim3(kBlock), 0, 0,
                d_w, d_x, n);
            GPU_LAUNCH_CHECK();
            // fresh consensus per pass, mirroring cs_admm_core entry:
            // z = x (d_x holds the previous pass solution), u = 0, rho reset
            {
                const long long totalX = (long long)P * n;
                const int grid = (int)((totalX + kBlock - 1) / kBlock);
                hipLaunchKernelGGL(init_xu_k, dim3(grid > 0 ? grid : 1),
                    dim3(kBlock), 0, 0, pl.z.p, pl.u.p, d_x, totalX);
                const int rg = (P + kBlock - 1) / kBlock;
                hipLaunchKernelGGL(rho_init_k, dim3(rg > 0 ? rg : 1),
                    dim3(kBlock), 0, 0, d_rho, d_ms, n, P);
                GPU_LAUNCH_CHECK();
            }
        }
        for (int k = 0; k < key.inner; ++k) {
            hipLaunchKernelGGL(rhs_k, dim3(P), dim3(kBlock), 0, 0,
                pl.rhs.p, pl.atb.p, pl.z.p, pl.u.p, d_rho, n);
            idct_batch(ar, ac, pl.rhs.p, pl.pix.p, pl.rhs.p, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(div_k, dim3(P), dim3(kBlock), 0, 0,
                pl.rhs.p, pl.mask2.p, d_rho, n);
            fdct_batch(ar, ac, pl.rhs.p, pl.pix.p, d_x, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(z_dual_k, dim3(P), dim3(kBlock), 0, 0,
                pl.z.p, d_x, pl.u.p, d_w, d_rho, d_lam, n);
        }
    }
    GPU_LAUNCH_CHECK();  // sticky launch errors from the loop above
    const clk::time_point t2 = dbg ? clk::now() : t0;
    if (dbg && !gpu_ok("sync solve", hipStreamSynchronize(0))) return false;

    GPU_CHECK(hipGetLastError());   // promote any execution error
    GPU_CHECK(hipMemcpyAsync(h_x, d_x, PnB, hipMemcpyDeviceToHost, 0));
    GPU_CHECK(hipStreamSynchronize(0));  // single drain: kernels + D2H
    const clk::time_point t3 = clk::now();
    for (int q = 0; q < P; ++q)
        std::memcpy(jobs[q]->p.x, h_x + (size_t)q * n, nB);
    if (dbg)
        fprintf(stderr,
            "[gpu] flush P=%d %dx%d mmax=%d inner=%d rw=%d | up=%.1f solve=%.1f "
            "down=%.1f tot=%.1f ms\n",
            P, key.rows, key.cols, mmax, key.inner, key.rews, ms(t0, t1),
            ms(t1, t2), ms(t2, t3), ms(t0, t3));
    return true;
}

// FISTA twin of run_group: reweighted-L1 proximal gradient with Nesterov
// momentum, mirroring cs_fista_core_single (init y = xp = x0, per-pass
// reweight, x = z out). DCT only; per-problem tv selects the branch:
// tv = 0 uses the sampled gradient with the exact fixed step 1/L, L = 2;
// tv > 0 adds the full-pix TV chain with step 1/(2 + 8*tv), as on the CPU.
// Per-iteration sequence per problem, all in the packed PnB buffers
// (sampled gradient: only measured positions ever materialize):
//
//   tmp  = y * Ac                     (1 batched GEMM via d_tmp)
//   r    = sampled synth - b          (synth_sample_k, m*R, no full pix)
//   tmp2 = sparse sc * Ac^T           (sadjoint_k, m*C, no memset)
//   grad = Ar * tmp2                  (1 batched GEMM)
//   prox : z = shrink(y - grad, lam/2 * w)   [step=1/2 folded: v = y - g]
//   mom  : y = z + mom*(z-xp); xp = z (t updated on the host, as on CPU)
bool run_group_fista(std::vector<Job*>& jobs, const Key& key)
{
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const bool dbg = dbg_flag();
    const int P = (int)jobs.size();
    const int n = key.rows * key.cols;
    int mmax = 0;
    for (int q = 0; q < P; ++q)
        if (jobs[q]->fp.m > mmax) mmax = jobs[q]->fp.m;
    if (P <= 0 || n <= 0 || mmax <= 0 || key.inner <= 0 || key.rews <= 0) return false;
    const size_t nB = (size_t)n * 4, mB = (size_t)mmax * 4;
    const size_t PnB = (size_t)P * nB, PmB = (size_t)P * mB;
    // packed H2D layout (bytes): [lam P][tv P][ms P][b P*mmax][rix][riy][x P*n][w P*n]
    const size_t offTv = (size_t)P * 4;
    const size_t offMs = (size_t)P * 8;
    const size_t offB = (size_t)P * 12;
    const size_t offRix = offB + PmB;
    const size_t offRiy = offRix + PmB;
    const size_t offX = offRiy + PmB;
    const size_t offW = offX + PnB;
    const size_t total = offW + PnB;
    const clk::time_point t0 = clk::now();

    Pool& pl = pool();
    if (!pl.din.ensure(total) || !pl.z.ensure(PnB) || !pl.u.ensure(PnB) ||
        !pl.pix.ensure(PnB) || !pl.rhs.ensure(PnB) || !pl.atb.ensure(PnB) ||
        !pl.mask2.ensure(PnB) || !pl.tmp.ensure(PnB) ||
        !pl.tv0.ensure(PnB) || !pl.tv1.ensure(PnB) ||
        !pl.rowptr.ensure((size_t)P * (key.rows + 1) * 4))
        return false;
    if (!pin().ensure(total)) return false;

    char* hst = (char*)pin().p;
    float* h_lam = (float*)(hst + 0);
    float* h_tv = (float*)(hst + offTv);
    int* h_ms = (int*)(hst + offMs);
    float* h_b = (float*)(hst + offB);
    int* h_rix = (int*)(hst + offRix);
    int* h_riy = (int*)(hst + offRiy);
    float* h_x = (float*)(hst + offX);
    float* h_w = (float*)(hst + offW);
    // row-grouped order for the sparse adjoint: reorders (b,rix,riy) per
    // problem and fills h_rowptr[q*(R+1)..]; the compact residual shares
    // this order everywhere downstream.
    std::vector<int> h_rowptr((size_t)P * (key.rows + 1));
    bool anyTv = false;
    for (int q = 0; q < P; ++q) {
        const FistaProblem& p = jobs[q]->fp;
        float* bd = h_b + (size_t)q * mmax;
        for (int k = 0; k < p.m; ++k) bd[k] = p.b[k];
        std::memcpy(h_rix + (size_t)q * mmax, p.rix, (size_t)p.m * 4);
        std::memcpy(h_riy + (size_t)q * mmax, p.riy, (size_t)p.m * 4);
        std::memcpy(h_x + (size_t)q * n, p.x, nB);
        h_lam[q] = p.lambda;
        h_tv[q] = p.tv;
        if (p.tv != 0.0f) anyTv = true;
        h_ms[q] = p.m;
        sort_measurements_by_row(bd, h_rix + (size_t)q * mmax,
            h_riy + (size_t)q * mmax, p.m, key.rows,
            h_rowptr.data() + (size_t)q * (key.rows + 1));
    }
    for (size_t i = 0; i < PnB / 4; ++i) h_w[i] = 1.0f;

    char* dst = (char*)pl.din.p;
    float* d_lam = (float*)(dst + 0);
    float* d_tv = (float*)(dst + offTv);
    int* d_ms = (int*)(dst + offMs);
    float* d_b = (float*)(dst + offB);
    int* d_rix = (int*)(dst + offRix);
    int* d_riy = (int*)(dst + offRiy);
    float* d_x = (float*)(dst + offX);
    float* d_w = (float*)(dst + offW);

    // Sampled-gradient buffer roles (ADMM-named pool, FISTA use): y=pl.z,
    // xp=pl.u, zc=pl.rhs, grad=pl.atb, compact resid r=pl.pix (PmB<=PnB),
    // sparse tmp2=pl.mask2, gemm scratch=pl.tmp. No full pix/sc image, no
    // memset: synth_sample + sadjoint replace 2 GEMMs + memset + residual.
    // TV branch additionally uses tv0 (full pix, then TV adjoint) + tv1.
    float *d_y = pl.z.p, *d_xp = pl.u.p, *d_z = pl.rhs.p, *d_g = pl.atb.p;
    float *d_r = pl.pix.p, *d_tmp2 = pl.mask2.p, *d_tmp = pl.tmp.p;
    float *d_tv0 = pl.tv0.p, *d_tv1 = pl.tv1.p;
    int* d_rowptr = pl.rowptr.p;

    float* ar = nullptr;
    float* ac = nullptr;
    if (!dct_get(key.rows, key.cols, ar, ac)) return false;

    GPU_CHECK(hipMemcpyAsync(pl.din.p, hst, total, hipMemcpyHostToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_rowptr, h_rowptr.data(),
        (size_t)P * (key.rows + 1) * 4, hipMemcpyHostToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_y, d_x, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_xp, d_x, PnB, hipMemcpyDeviceToDevice, 0));

    for (int r = 0; r < key.rews; ++r) {
        if (r > 0) {
            hipLaunchKernelGGL(reweight_k, dim3(P), dim3(kBlock), 0, 0,
                d_w, d_x, n);
            GPU_LAUNCH_CHECK();
            // per-pass restart, mirroring cs_fista_core_single: d_x already
            // holds the previous pass solution (copied at the loop end).
            GPU_CHECK(hipMemcpyAsync(d_y, d_x, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_xp, d_x, PnB, hipMemcpyDeviceToDevice, 0));
        }
        float t = 1.0f;
        for (int k = 0; k < key.inner; ++k) {
            // sampled gradient: tmp = y*Ac (full GEMM), r = sampled
            // synth - b (m*R), tmp2 = sparse sc*Ac^T (m*C), grad = Ar*tmp2.
            // Replaces idct + memset + resid + fdct (4 GEMMs -> 2 GEMMs).
            row_synth(d_y, d_tmp, ac, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(synth_sample_k, dim3(P), dim3(kBlock), 0, 0,
                d_r, d_tmp, ar, d_b, d_rix, d_riy, d_ms, mmax,
                key.rows, key.cols);
            hipLaunchKernelGGL(sadjoint_k, dim3(P), dim3(kBlock), 0, 0,
                d_tmp2, d_r, d_riy, d_rowptr, ac, d_ms, mmax,
                key.rows, key.cols);
            col_anlys(d_tmp2, d_g, ar, key.rows, key.cols, P, n);
            const float t_new = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            if (!anyTv) {
                hipLaunchKernelGGL(shrink_mom_k, dim3(P), dim3(kBlock), 0, 0,
                    d_z, d_y, d_g, d_w, d_lam, d_xp, mom, n);
            } else {
                // TV branch: full pix = IDCT(y), TV grad + adjoint, then
                // v = y - step*(2*gd + tv*gt) with step = 1/(2+8*tv).
                idct_batch(ar, ac, d_y, d_tmp, d_tv0, key.rows, key.cols, P, n);
                hipLaunchKernelGGL(tvgrad_k, dim3(P), dim3(kBlock), 0, 0,
                    d_tv1, d_tv0, key.rows, key.cols);
                fdct_batch(ar, ac, d_tv1, d_tmp, d_tv0, key.rows, key.cols, P, n);
                hipLaunchKernelGGL(shrink_tvmom_k, dim3(P), dim3(kBlock), 0, 0,
                    d_z, d_y, d_g, d_tv0, d_w, d_lam, d_tv, d_xp, mom, n);
            }
            t = t_new;
        }
        if (r + 1 < key.rews)
            GPU_CHECK(hipMemcpyAsync(d_x, d_z, PnB, hipMemcpyDeviceToDevice, 0));
    }
    GPU_LAUNCH_CHECK();
    if (dbg && !gpu_ok("sync solve", hipStreamSynchronize(0))) return false;

    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipMemcpyAsync(d_x, d_z, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(h_x, d_x, PnB, hipMemcpyDeviceToHost, 0));
    GPU_CHECK(hipStreamSynchronize(0));  // single drain: kernels + D2H
    const clk::time_point t3 = clk::now();
    for (int q = 0; q < P; ++q)
        std::memcpy(jobs[q]->fp.x, h_x + (size_t)q * n, nB);
    if (dbg)
        fprintf(stderr,
            "[gpu] fista flush P=%d %dx%d mmax=%d inner=%d rw=%d tot=%.1f ms\n",
            P, key.rows, key.cols, mmax, key.inner, key.rews, ms(t0, t3));
    return true;
}

// Joint twin: the same loop over stacked [x0|x1|x2] with row-coupled
// reweight + group shrink (mirrors cs_fista_core_joint). Per-plane grads
// reuse the batched GEMM calls (batch=P per plane); only the two coupled
// steps need the stacked kernels. Buffers: plane c lives in
// (y,xp,z,grad)[c] with shared pix/sc/tmp across the sequential planes.
bool run_group_fista_joint(std::vector<Job*>& jobs, const Key& key)
{
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const bool dbg = dbg_flag();
    const int P = (int)jobs.size();
    const int n = key.rows * key.cols;
    int mmax = 0;
    for (int q = 0; q < P; ++q)
        if (jobs[q]->fp.m > mmax) mmax = jobs[q]->fp.m;
    if (P <= 0 || n <= 0 || mmax <= 0 || key.inner <= 0 || key.rews <= 0) return false;
    const size_t nB = (size_t)n * 4, mB = (size_t)mmax * 4;
    const size_t PnB = (size_t)P * nB, PmB = (size_t)P * mB;
    // H2D layout (bytes): [lam P][tv P][ms P][b0][b1][b2 P*mmax each][rix][riy]
    //                     [x0][x1][x2 P*n each][w P*n]
    const size_t offTv = (size_t)P * 4;
    const size_t offMs = (size_t)P * 8;
    const size_t offB0 = (size_t)P * 12;
    const size_t offB1 = offB0 + PmB;
    const size_t offB2 = offB1 + PmB;
    const size_t offRix = offB2 + PmB;
    const size_t offRiy = offRix + PmB;
    const size_t offX0 = offRiy + PmB;
    const size_t offX1 = offX0 + PnB;
    const size_t offX2 = offX1 + PnB;
    const size_t offW = offX2 + PnB;
    const size_t total = offW + PnB;
    const clk::time_point t0 = clk::now();

    Pool& pl = pool();
    if (!pl.din.ensure(total) || !pl.z.ensure(PnB) || !pl.u.ensure(PnB) ||
        !pl.pix.ensure(PnB) || !pl.rhs.ensure(PnB) || !pl.atb.ensure(PnB) ||
        !pl.mask2.ensure(PnB) || !pl.tmp.ensure(PnB) ||
        !pl.tv0.ensure(PnB) || !pl.tv1.ensure(PnB) ||
        !pl.tg0.ensure(PnB) || !pl.tg1.ensure(PnB) || !pl.tg2.ensure(PnB) ||
        !pl.rowptr.ensure((size_t)P * (key.rows + 1) * 4) ||
        !pl.y1.ensure(PnB) || !pl.y2.ensure(PnB) ||
        !pl.xp1.ensure(PnB) || !pl.xp2.ensure(PnB) ||
        !pl.z1.ensure(PnB) || !pl.z2.ensure(PnB) ||
        !pl.g1.ensure(PnB) || !pl.g2.ensure(PnB))
        return false;
    if (!pin().ensure(total)) return false;

    char* hst = (char*)pin().p;
    float* h_lam = (float*)(hst + 0);
    float* h_tv = (float*)(hst + offTv);
    int* h_ms = (int*)(hst + offMs);
    float* h_b0 = (float*)(hst + offB0);
    float* h_b1 = (float*)(hst + offB1);
    float* h_b2 = (float*)(hst + offB2);
    int* h_rix = (int*)(hst + offRix);
    int* h_riy = (int*)(hst + offRiy);
    float* h_x0 = (float*)(hst + offX0);
    float* h_x1 = (float*)(hst + offX1);
    float* h_x2 = (float*)(hst + offX2);
    float* h_w = (float*)(hst + offW);
    // Joint row-grouped order: one shared permutation for (b0,b1,b2,rix,riy)
    // per problem; the compact residuals share it downstream.
    std::vector<int> h_rowptr((size_t)P * (key.rows + 1));
    bool anyTv = false;
    for (int q = 0; q < P; ++q) {
        const FistaProblem& p = jobs[q]->fp;
        if (!p.b1 || !p.b2 || !p.x1 || !p.x2) return false;
        float* b0d = h_b0 + (size_t)q * mmax;
        float* b1d = h_b1 + (size_t)q * mmax;
        float* b2d = h_b2 + (size_t)q * mmax;
        for (int k = 0; k < p.m; ++k) {
            b0d[k] = p.b[k];
            b1d[k] = p.b1[k];
            b2d[k] = p.b2[k];
        }
        std::memcpy(h_rix + (size_t)q * mmax, p.rix, (size_t)p.m * 4);
        std::memcpy(h_riy + (size_t)q * mmax, p.riy, (size_t)p.m * 4);
        std::memcpy(h_x0 + (size_t)q * n, p.x, nB);
        std::memcpy(h_x1 + (size_t)q * n, p.x1, nB);
        std::memcpy(h_x2 + (size_t)q * n, p.x2, nB);
        h_lam[q] = p.lambda;
        h_tv[q] = p.tv;
        if (p.tv != 0.0f) anyTv = true;
        h_ms[q] = p.m;
        // row-group with the 3 planes permuted together (shared rix/riy)
        int* rx = h_rix + (size_t)q * mmax;
        int* ry = h_riy + (size_t)q * mmax;
        int* rp = h_rowptr.data() + (size_t)q * (key.rows + 1);
        {
            // counting sort by row, stable, permuting b0/b1/b2 together
            const int R = key.rows, m = p.m;
            static std::vector<int> cnt, pos;
            static std::vector<float> t0, t1, t2;
            static std::vector<int> tx, ty;
            if ((int)cnt.size() < R + 1) {
                cnt.assign((size_t)R + 1, 0);
                pos.assign((size_t)R + 1, 0);
            } else {
                std::fill(cnt.begin(), cnt.begin() + R + 1, 0);
            }
            if ((int)t0.size() < m) {
                t0.assign(m, 0.0f); t1.assign(m, 0.0f); t2.assign(m, 0.0f);
                tx.assign(m, 0); ty.assign(m, 0);
            }
            for (int k = 0; k < m; ++k) cnt[(size_t)rx[k]]++;
            rp[0] = 0;
            for (int i = 0; i < R; ++i) rp[i + 1] = rp[i] + cnt[(size_t)i];
            for (int i = 0; i <= R; ++i) pos[(size_t)i] = rp[i];
            for (int k = 0; k < m; ++k) {
                const int d = pos[(size_t)rx[k]]++;
                t0[(size_t)d] = b0d[k];
                t1[(size_t)d] = b1d[k];
                t2[(size_t)d] = b2d[k];
                tx[(size_t)d] = rx[k];
                ty[(size_t)d] = ry[k];
            }
            std::memcpy(b0d, t0.data(), (size_t)m * 4);
            std::memcpy(b1d, t1.data(), (size_t)m * 4);
            std::memcpy(b2d, t2.data(), (size_t)m * 4);
            std::memcpy(rx, tx.data(), (size_t)m * 4);
            std::memcpy(ry, ty.data(), (size_t)m * 4);
        }
    }
    for (size_t i = 0; i < PnB / 4; ++i) h_w[i] = 1.0f;

    char* dst = (char*)pl.din.p;
    float* d_lam = (float*)(dst + 0);
    float* d_tv = (float*)(dst + offTv);
    int* d_ms = (int*)(dst + offMs);
    float* d_b0 = (float*)(dst + offB0);
    float* d_b1 = (float*)(dst + offB1);
    float* d_b2 = (float*)(dst + offB2);
    int* d_rix = (int*)(dst + offRix);
    int* d_riy = (int*)(dst + offRiy);
    float* d_x0 = (float*)(dst + offX0);
    float* d_x1 = (float*)(dst + offX1);
    float* d_x2 = (float*)(dst + offX2);
    float* d_w = (float*)(dst + offW);

    // plane buffers: c0 reuses the single-path fields, c1/c2 the joint ones.
    // r/tmp2/tmp + tv0/tv1 are shared sequentially (each plane's chain
    // completes before the next reuses the scratch); data grads persist in
    // g0/g1/g2 and TV adjoints in tg0/tg1/tg2 for the joint shrink.
    float *d_y0 = pl.z.p, *d_y1 = pl.y1.p, *d_y2 = pl.y2.p;
    float *d_xp0 = pl.u.p, *d_xp1 = pl.xp1.p, *d_xp2 = pl.xp2.p;
    float *d_z0 = pl.rhs.p, *d_z1 = pl.z1.p, *d_z2 = pl.z2.p;
    float *d_g0 = pl.atb.p, *d_g1 = pl.g1.p, *d_g2 = pl.g2.p;
    float *d_r = pl.pix.p, *d_tmp2 = pl.mask2.p, *d_tmp = pl.tmp.p;
    float *d_tv0 = pl.tv0.p, *d_tv1 = pl.tv1.p;
    float *d_tg0 = pl.tg0.p, *d_tg1 = pl.tg1.p, *d_tg2 = pl.tg2.p;
    int* d_rowptr = pl.rowptr.p;

    float* ar = nullptr;
    float* ac = nullptr;
    if (!dct_get(key.rows, key.cols, ar, ac)) return false;

    GPU_CHECK(hipMemcpyAsync(pl.din.p, hst, total, hipMemcpyHostToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_rowptr, h_rowptr.data(),
        (size_t)P * (key.rows + 1) * 4, hipMemcpyHostToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_y0, d_x0, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_y1, d_x1, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_y2, d_x2, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_xp0, d_x0, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_xp1, d_x1, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_xp2, d_x2, PnB, hipMemcpyDeviceToDevice, 0));

    for (int r = 0; r < key.rews; ++r) {
        if (r > 0) {
            hipLaunchKernelGGL(reweight_joint_k, dim3(P), dim3(kBlock), 0, 0,
                d_w, d_x0, d_x1, d_x2, n);
            GPU_LAUNCH_CHECK();
            // per-pass restart, mirroring cs_fista_core_joint: d_x[012]
            // already hold the previous pass solutions (copied at loop end).
            GPU_CHECK(hipMemcpyAsync(d_y0, d_x0, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_y1, d_x1, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_y2, d_x2, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_xp0, d_x0, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_xp1, d_x1, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_xp2, d_x2, PnB, hipMemcpyDeviceToDevice, 0));
        }
        float t = 1.0f;
        for (int k = 0; k < key.inner; ++k) {
            row_synth(d_y0, d_tmp, ac, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(synth_sample_k, dim3(P), dim3(kBlock), 0, 0,
                d_r, d_tmp, ar, d_b0, d_rix, d_riy, d_ms, mmax,
                key.rows, key.cols);
            hipLaunchKernelGGL(sadjoint_k, dim3(P), dim3(kBlock), 0, 0,
                d_tmp2, d_r, d_riy, d_rowptr, ac, d_ms, mmax,
                key.rows, key.cols);
            col_anlys(d_tmp2, d_g0, ar, key.rows, key.cols, P, n);
            if (anyTv) {
                idct_batch(ar, ac, d_y0, d_tmp, d_tv0, key.rows, key.cols, P, n);
                hipLaunchKernelGGL(tvgrad_k, dim3(P), dim3(kBlock), 0, 0,
                    d_tv1, d_tv0, key.rows, key.cols);
                fdct_batch(ar, ac, d_tv1, d_tmp, d_tg0, key.rows, key.cols, P, n);
            }
            row_synth(d_y1, d_tmp, ac, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(synth_sample_k, dim3(P), dim3(kBlock), 0, 0,
                d_r, d_tmp, ar, d_b1, d_rix, d_riy, d_ms, mmax,
                key.rows, key.cols);
            hipLaunchKernelGGL(sadjoint_k, dim3(P), dim3(kBlock), 0, 0,
                d_tmp2, d_r, d_riy, d_rowptr, ac, d_ms, mmax,
                key.rows, key.cols);
            col_anlys(d_tmp2, d_g1, ar, key.rows, key.cols, P, n);
            if (anyTv) {
                idct_batch(ar, ac, d_y1, d_tmp, d_tv0, key.rows, key.cols, P, n);
                hipLaunchKernelGGL(tvgrad_k, dim3(P), dim3(kBlock), 0, 0,
                    d_tv1, d_tv0, key.rows, key.cols);
                fdct_batch(ar, ac, d_tv1, d_tmp, d_tg1, key.rows, key.cols, P, n);
            }
            row_synth(d_y2, d_tmp, ac, key.rows, key.cols, P, n);
            hipLaunchKernelGGL(synth_sample_k, dim3(P), dim3(kBlock), 0, 0,
                d_r, d_tmp, ar, d_b2, d_rix, d_riy, d_ms, mmax,
                key.rows, key.cols);
            hipLaunchKernelGGL(sadjoint_k, dim3(P), dim3(kBlock), 0, 0,
                d_tmp2, d_r, d_riy, d_rowptr, ac, d_ms, mmax,
                key.rows, key.cols);
            col_anlys(d_tmp2, d_g2, ar, key.rows, key.cols, P, n);
            if (anyTv) {
                idct_batch(ar, ac, d_y2, d_tmp, d_tv0, key.rows, key.cols, P, n);
                hipLaunchKernelGGL(tvgrad_k, dim3(P), dim3(kBlock), 0, 0,
                    d_tv1, d_tv0, key.rows, key.cols);
                fdct_batch(ar, ac, d_tv1, d_tmp, d_tg2, key.rows, key.cols, P, n);
            }
            const float t_new = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t * t));
            const float mom = (t - 1.0f) / t_new;
            if (!anyTv) {
                hipLaunchKernelGGL(shrink_joint_mom_k, dim3(P), dim3(kBlock), 0, 0,
                    d_z0, d_z1, d_z2, d_y0, d_y1, d_y2,
                    d_g0, d_g1, d_g2, d_w, d_lam, d_xp0, d_xp1, d_xp2, mom, n);
            } else {
                hipLaunchKernelGGL(shrink_joint_tvmom_k, dim3(P), dim3(kBlock), 0, 0,
                    d_z0, d_z1, d_z2, d_y0, d_y1, d_y2,
                    d_g0, d_g1, d_g2, d_tg0, d_tg1, d_tg2,
                    d_w, d_lam, d_tv, d_xp0, d_xp1, d_xp2, mom, n);
            }
            t = t_new;
        }
        if (r + 1 < key.rews) {
            GPU_CHECK(hipMemcpyAsync(d_x0, d_z0, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_x1, d_z1, PnB, hipMemcpyDeviceToDevice, 0));
            GPU_CHECK(hipMemcpyAsync(d_x2, d_z2, PnB, hipMemcpyDeviceToDevice, 0));
        }
    }
    GPU_LAUNCH_CHECK();
    if (dbg && !gpu_ok("sync solve", hipStreamSynchronize(0))) return false;

    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipMemcpyAsync(d_x0, d_z0, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_x1, d_z1, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(d_x2, d_z2, PnB, hipMemcpyDeviceToDevice, 0));
    GPU_CHECK(hipMemcpyAsync(h_x0, d_x0, PnB, hipMemcpyDeviceToHost, 0));
    GPU_CHECK(hipMemcpyAsync(h_x1, d_x1, PnB, hipMemcpyDeviceToHost, 0));
    GPU_CHECK(hipMemcpyAsync(h_x2, d_x2, PnB, hipMemcpyDeviceToHost, 0));
    GPU_CHECK(hipStreamSynchronize(0));  // single drain: kernels + D2H
    const clk::time_point t3 = clk::now();
    for (int q = 0; q < P; ++q) {
        std::memcpy(jobs[q]->fp.x, h_x0 + (size_t)q * n, nB);
        std::memcpy(jobs[q]->fp.x1, h_x1 + (size_t)q * n, nB);
        std::memcpy(jobs[q]->fp.x2, h_x2 + (size_t)q * n, nB);
    }
    if (dbg)
        fprintf(stderr,
            "[gpu] fista-joint flush P=%d %dx%d mmax=%d inner=%d rw=%d tot=%.1f ms\n",
            P, key.rows, key.cols, mmax, key.inner, key.rews, ms(t0, t3));
    return true;
}

void run_jobs(std::vector<Job>& batch)
{
    std::map<Key, std::vector<Job*>> groups;
    for (auto& j : batch) {
        if (j.kind == 1) groups[Key(j.fp)].push_back(&j);
        else groups[Key(j.p)].push_back(&j);
    }
    for (auto& kv : groups) {
        bool ok = false;
        if (!g_dead.load(std::memory_order_relaxed)) {
            if (kv.first.kind == 1 && kv.first.joint == 1) ok = run_group_fista_joint(kv.second, kv.first);
            else if (kv.first.kind == 1) ok = run_group_fista(kv.second, kv.first);
            else ok = run_group(kv.second, kv.first);
        }
        for (auto* j : kv.second) finish(*j, !ok);
    }
}

// Stable counting sort of one problem's measurements by row, in place over
// the staging slices: reorders (b,rix,riy) and fills row_ptr[0..R]
// (exclusive ends). O(m + R). The residual vector then shares this order
// everywhere downstream, so no permutation array ever leaves the host.
// Scratch grows once (worker thread only).
void sort_measurements_by_row(float* b, int* rix, int* riy, int m, int R, int* row_ptr)
{
    static std::vector<int> cnt, pos;
    static std::vector<float> tb;
    static std::vector<int> tx, ty;
    if ((int)cnt.size() < R + 1) {
        cnt.assign((size_t)R + 1, 0);
        pos.assign((size_t)R + 1, 0);
    } else {
        std::fill(cnt.begin(), cnt.begin() + R + 1, 0);
    }
    if ((int)tb.size() < m) {
        tb.assign(m, 0.0f);
        tx.assign(m, 0);
        ty.assign(m, 0);
    }
    for (int k = 0; k < m; ++k) cnt[(size_t)rix[k]]++;
    row_ptr[0] = 0;
    for (int i = 0; i < R; ++i) row_ptr[i + 1] = row_ptr[i] + cnt[(size_t)i];
    for (int i = 0; i <= R; ++i) pos[(size_t)i] = row_ptr[i];
    for (int k = 0; k < m; ++k) {
        const int d = pos[(size_t)rix[k]]++;
        tb[(size_t)d] = b[k];
        tx[(size_t)d] = rix[k];
        ty[(size_t)d] = riy[k];
    }
    std::memcpy(b, tb.data(), (size_t)m * 4);
    std::memcpy(rix, tx.data(), (size_t)m * 4);
    std::memcpy(riy, ty.data(), (size_t)m * 4);
}

void worker_main()
{
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const bool dbg = dbg_flag();
    QState& s = Q();
    std::unique_lock<std::mutex> lk(s.m);
    bool have_prev = false;
    clk::time_point t_prev_end{};
    for (;;) {
        s.cv.wait(lk, [] { return !Q().q.empty(); });
        // grace window: drop the lock, spin until the batch fills or arrivals
        // stall (keeps exact timing -- cv.wait_until would oversleep ~15ms)
        lk.unlock();
        const clk::time_point t_arrive = clk::now();
        const auto soft = t_arrive + std::chrono::microseconds(kGraceUs);
        const auto hard = t_arrive + std::chrono::milliseconds(3);
        int last_p = s.pend.load(std::memory_order_acquire);
        clk::time_point t_grow = t_arrive;
        for (;;) {
            const int p = s.pend.load(std::memory_order_acquire);
            if (p != last_p) {
                last_p = p;
                t_grow = clk::now();
            }
            if (p >= kBatchTarget) break;
            const clk::time_point now = clk::now();
            if (now >= hard) break;
            if (now >= soft && now - t_grow > std::chrono::microseconds(150))
                break;
            std::this_thread::yield();
        }
        lk.lock();
        const int pend = (int)s.q.size();
        std::vector<Job> batch;
        batch.reserve(s.q.size());
        for (auto& j : s.q) batch.push_back(std::move(j));
        s.q.clear();
        s.pend.store(0, std::memory_order_release);
        lk.unlock();
        const clk::time_point t_flush = clk::now();
        if (dbg)
            fprintf(stderr, "[gpu] sched gap=%.1f grace=%.1f pend=%d\n",
                have_prev ? ms(t_prev_end, t_flush) : -1.0,
                ms(t_arrive, t_flush), pend);
        try {
            run_jobs(batch);
        } catch (const std::exception& e) {
            fprintf(stderr, "[gpu] worker: %s\n", e.what());
            for (auto& j : batch) finish(j, true);
        } catch (...) {
            fprintf(stderr, "[gpu] worker: unknown exception\n");
            for (auto& j : batch) finish(j, true);
        }
        t_prev_end = clk::now();
        have_prev = true;
        lk.lock();
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void register_cpu_fallback(CpuFallbackFn fn)
{
    g_fb = fn;
}

void register_cpu_fallback_fista(FistaFallbackFn fn)
{
    g_fb_fista = fn;
}

bool available()
{
    if (!g_probed.exchange(true, std::memory_order_acq_rel)) {
        int n = 0;
        const hipError_t e = hipGetDeviceCount(&n);
        g_has_device = (e == hipSuccess && n > 0);
        if (!g_has_device)
            fprintf(stderr, "[gpu] no HIP device: %s\n",
                e == hipSuccess ? "device count is 0" : hipGetErrorString(e));
    }
    return g_has_device;
}

bool enabled()
{
    return g_enabled.load(std::memory_order_relaxed);
}

void set_enabled(bool on)
{
    if (!on) {
        g_enabled.store(false, std::memory_order_relaxed);
        return;
    }
    if (!available()) {
        fprintf(stderr, "[gpu] --device gpu unavailable, staying on CPU\n");
        g_enabled.store(false, std::memory_order_relaxed);
        return;
    }
    hipDeviceProp_t prop;
    std::memset(&prop, 0, sizeof(prop));
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
        fprintf(stderr, "[gpu] fista/admm solve offload enabled: %s\n", prop.name);
    else
        fprintf(stderr, "[gpu] fista/admm solve offload enabled\n");
    // pay hip module/context/stream init now -- it lands on the first flush
    // otherwise (~165 ms) and hides better under PNG load + KDF here
    {
        float* wb = nullptr;
        if (hipMalloc((void**)&wb, 4096) == hipSuccess) {
            hipLaunchKernelGGL(init_xu_k, dim3(1), dim3(kBlock), 0, 0, wb, wb, wb, 16);
            (void)hipStreamSynchronize(0);
            (void)hipFree(wb);
        }
        (void)hipGetLastError();  // consume any warmup error
    }
    rb_init();  // rocBLAS handle + Tensile warmup; silent native fallback
    g_enabled.store(true, std::memory_order_relaxed);
}

void admm_solve_blocking(const AdmmProblem& p)
{
    if (!enabled() || g_dead.load(std::memory_order_relaxed)) {
        if (g_fb) g_fb(p);
        return;
    }
    auto c = std::make_shared<Completion>();
    QState& s = Q();
    {
        std::lock_guard<std::mutex> lk(s.m);
        if (!s.started) {
            s.started = true;
            new std::thread(worker_main);  // blocks on s.m until we unlock
        }
        Job j;
        j.kind = 0;
        j.p = p;
        j.c = c;
        s.q.push_back(std::move(j));
        s.pend.store((int)s.q.size(), std::memory_order_release);
    }
    s.cv.notify_one();
    std::unique_lock<std::mutex> lk(c->m);
    c->cv.wait(lk, [&] { return c->done; });
}

void fista_solve_blocking(const FistaProblem& p)
{
    if (!enabled() || g_dead.load(std::memory_order_relaxed)) {
        if (g_fb_fista) g_fb_fista(p);
        return;
    }
    auto c = std::make_shared<Completion>();
    QState& s = Q();
    {
        std::lock_guard<std::mutex> lk(s.m);
        if (!s.started) {
            s.started = true;
            new std::thread(worker_main);  // blocks on s.m until we unlock
        }
        Job j;
        j.kind = 1;
        j.fp = p;
        j.c = c;
        s.q.push_back(std::move(j));
        s.pend.store((int)s.q.size(), std::memory_order_release);
    }
    s.cv.notify_one();
    std::unique_lock<std::mutex> lk(c->m);
    c->cv.wait(lk, [&] { return c->done; });
}

}  // namespace cs_gpu

