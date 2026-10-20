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
    int max_n = 0;

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
    }
};

}  // namespace gpu_common
