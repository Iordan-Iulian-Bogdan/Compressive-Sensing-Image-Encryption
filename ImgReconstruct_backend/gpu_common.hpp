#pragma once
// Pieces shared by the HIP tile backends (gpu_tiles.hip, gpu_admm.hip): the
// per-tile layout record, dense orthonormal 2D DCT kernels over a batch of
// tiles, and small host helpers. Include from .hip files only.
//
// Layout: every tile owns a contiguous slice [off, off + rows*cols) of each
// state array. A 2D DCT is two small dense matrix products (cheap for tiles up
// to ~150 px, no FFT library). Kernels use blockIdx.y as the tile index and
// blockIdx.x * kThreads + threadIdx.x as the element index.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace gpu_common {

constexpr int kThreads = 256;

struct TileDev {
    int rows, cols;
    int off;       // element offset of this tile in the state arrays
    int rtab;      // offset of the row DCT table (rows x rows) in the table array
    int ctab;      // offset of the column DCT table (cols x cols)
    int pad[3];
};

#define HIP_TRY(call)                                                              \
    do {                                                                           \
        hipError_t e_ = (call);                                                    \
        if (e_ != hipSuccess) {                                                    \
            std::fprintf(stderr, "gpu: %s failed: %s\n", #call, hipGetErrorString(e_)); \
            return false;                                                          \
        }                                                                          \
    } while (0)

namespace {

// Inverse 2D DCT, pass A: TMP[r][n] = sum_k Y[r][k] * Cc[k][n]
__global__ void k_synth_a(const TileDev* td, const float* tab, const float* Y, float* TMP) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int r = i / T.cols, n = i % T.cols;
    const float* Cc = tab + T.ctab;
    const float* y = Y + T.off + r * T.cols;
    float s = 0.0f;
    for (int k = 0; k < T.cols; ++k) s += y[k] * Cc[k * T.cols + n];
    TMP[T.off + i] = s;
}

// Inverse 2D DCT, pass B: PIX[m][n] = sum_r Cr[r][m] * TMP[r][n]
__global__ void k_synth_b(const TileDev* td, const float* tab, const float* TMP, float* PIX) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int m = i / T.cols, n = i % T.cols;
    const float* Cr = tab + T.rtab;
    const float* tmp = TMP + T.off;
    float s = 0.0f;
    for (int r = 0; r < T.rows; ++r) s += Cr[r * T.rows + m] * tmp[r * T.cols + n];
    PIX[T.off + i] = s;
}

// Forward 2D DCT, pass A: TMP[r][k] = sum_n SC[r][n] * Cc[k][n]
__global__ void k_adj_a(const TileDev* td, const float* tab, const float* SC, float* TMP) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int r = i / T.cols, k = i % T.cols;
    const float* Cc = tab + T.ctab;
    const float* sc = SC + T.off + r * T.cols;
    float s = 0.0f;
    for (int n = 0; n < T.cols; ++n) s += sc[n] * Cc[k * T.cols + n];
    TMP[T.off + i] = s;
}

// Forward 2D DCT, pass B: G[u][k] = sum_r Cr[u][r] * TMP[r][k]
__global__ void k_adj_b(const TileDev* td, const float* tab, const float* TMP, float* G) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int u = i / T.cols, k = i % T.cols;
    const float* Cr = tab + T.rtab;
    const float* tmp = TMP + T.off;
    float s = 0.0f;
    for (int r = 0; r < T.rows; ++r) s += Cr[u * T.rows + r] * tmp[r * T.cols + k];
    G[T.off + i] = s;
}

// Frequency weight ramp, identical to the CPU FillDctWeights for block <= 0.
__global__ void k_ws(const TileDev* td, float* WS, float logw) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int r = i / T.cols, c = i % T.cols;
    const float fr = T.rows > 1 ? (float)r / (float)(T.rows - 1) : 0.0f;
    const float fc = T.cols > 1 ? (float)c / (float)(T.cols - 1) : 0.0f;
    WS[T.off + i] = expf(logw * 0.5f * (fr + fc));
}

// ---- shared by the tv > 0 (backtracking) branches of gpu_tiles and gpu_admm ----

// Smoothed-TV gradient and per-pixel TV energy (d0 - eps) of a tile, as the CPU
// cs_tv_grad_phi (eps = 1e-2; keep in sync with helper_functions.cpp).
__global__ void k_tv(const TileDev* td, const float* PIX, float* TVG, float* PHI) {
    const TileDev T = td[blockIdx.y];
    const int t = blockIdx.x * kThreads + threadIdx.x;
    if (t >= T.rows * T.cols) return;
    const int rows = T.rows, cols = T.cols, i = t / cols, j = t % cols;
    const float* pix = PIX + T.off;
    const float eps = 1e-2f;
    const bool has_down = i + 1 < rows;
    const float c = pix[t];
    const float a0 = has_down ? pix[t + cols] - c : 0.0f;
    const float b0 = (j + 1 < cols) ? pix[t + 1] - c : 0.0f;
    const float d0 = sqrtf(a0 * a0 + b0 * b0 + eps * eps);
    float acc = -(a0 / d0 + b0 / d0);
    if (i > 0) {
        const float up = pix[t - cols];
        const float aT = c - up;
        const float bT = (j + 1 < cols) ? pix[t - cols + 1] - up : 0.0f;
        acc += aT / sqrtf(aT * aT + bT * bT + eps * eps);
    }
    if (j > 0) {
        const float lf = pix[t - 1];
        const float aL = has_down ? pix[t - 1 + cols] - lf : 0.0f;
        const float bL = c - lf;
        acc += bL / sqrtf(aL * aL + bL * bL + eps * eps);
    }
    TVG[T.off + t] = acc;
    PHI[T.off + t] = d0 - eps;
}

// Sum of v over a block of kThreads threads (double; every thread gets the result).
__device__ double block_sum(double v) {
    __shared__ double sh[kThreads];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}

// Backtracking test terms per tile (one block per tile): dot = sum g*(z - y),
// dz2 = sum (z - y)^2, in double as the CPU.
__global__ void k_dots(const TileDev* td, const float* G, const float* Z, const float* Y, double* D2) {
    const TileDev T = td[blockIdx.x];
    const int n = T.rows * T.cols;
    double dot = 0.0, dz2 = 0.0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const double dz = (double)Z[T.off + i] - (double)Y[T.off + i];
        dot += (double)G[T.off + i] * dz;
        dz2 += dz * dz;
    }
    const double a = block_sum(dot), b = block_sum(dz2);
    if (threadIdx.x == 0) { D2[2 * blockIdx.x] = a; D2[2 * blockIdx.x + 1] = b; }
}

// FISTA momentum after an accepted step: y = z + mom*(z - xp), xp = z.
__global__ void k_mom(const TileDev* td, const float* Z, float* Y, float* XP, float mom) {
    const TileDev T = td[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= T.rows * T.cols) return;
    const int o = T.off + i;
    Y[o] = Z[o] + mom * (Z[o] - XP[o]);
    XP[o] = Z[o];
}

// Host side of one backtracking step for a batch: given f(y), f(z), dot and dz2
// per tile, accept (act = 0) or double L, exactly the CPU rule; returns how many
// tiles still search.
inline int bt_accept(int nt, const std::vector<double>& fy, const std::vector<double>& fz,
    const std::vector<double>& d2, std::vector<float>& L, std::vector<int>& act, int trial) {
    int left = 0;
    for (int t = 0; t < nt; ++t) {
        if (!act[(size_t)t]) continue;
        const double Q = fy[(size_t)t] + d2[2 * (size_t)t] + 0.5 * (double)L[(size_t)t] * d2[2 * (size_t)t + 1];
        if (fz[(size_t)t] <= Q + 1e-7 * (1.0 + std::fabs(fy[(size_t)t])) || trial >= 24) act[(size_t)t] = 0;
        else { L[(size_t)t] *= 2.0f; ++left; }
    }
    return left;
}

// Orthonormal DCT-II table C[k][n] = a_k cos(pi (2n+1) k / 2N), N x N.
void fill_dct_table(float* C, int N) {
    for (int k = 0; k < N; ++k) {
        const double a = k == 0 ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
        for (int n = 0; n < N; ++n)
            C[k * N + n] = (float)(a * std::cos(3.14159265358979323846 * (2.0 * n + 1.0) * k / (2.0 * N)));
    }
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    ~DevBuf() { if (p) (void)hipFree(p); }
    bool alloc(size_t n) { return hipMalloc((void**)&p, (n ? n : 1) * sizeof(T)) == hipSuccess; }
};

}  // namespace

// Host-side tile layout: per-tile records plus the shared DCT table array.
struct TileLayout {
    std::vector<TileDev> td;
    std::vector<float> table;
    std::map<int, int> table_off;
    long long total = 0;
    int max_n = 0, max_rows = 0, max_cols = 0;

    int table_for(int N) {
        auto it = table_off.find(N);
        if (it != table_off.end()) return it->second;
        const int off = (int)table.size();
        table.resize(table.size() + (size_t)N * N);
        fill_dct_table(table.data() + off, N);
        table_off[N] = off;
        return off;
    }
    void add(int rows, int cols) {
        TileDev d{};
        d.rows = rows; d.cols = cols;
        d.off = (int)total;
        d.rtab = table_for(rows);
        d.ctab = table_for(cols);
        td.push_back(d);
        total += (long long)rows * cols;
        max_n = std::max(max_n, rows * cols);
        max_rows = std::max(max_rows, rows);
        max_cols = std::max(max_cols, cols);
    }
};

}  // namespace gpu_common
