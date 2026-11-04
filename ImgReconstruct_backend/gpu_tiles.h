#pragma once
// Batched GPU (HIP) tile solver: new backend, independent of cs_gpu.h.
//
// Plain C++ header: callers see no HIP headers. The implementation is in
// gpu_tiles.hip. All tiles of a call are solved together, one thread block row
// per tile, so thousands of small tiles keep the device busy.
#include <vector>

namespace gpu_tiles {

// True when a HIP device is present and usable.
bool available();

// Starts the HIP runtime and device context (one small allocation) so the first solve does not
// pay for it; meant for a background thread at decode start. No-op without a device.
void warmup();

// One tile problem: orthonormal DCT basis, point sampling.
struct TileProblem {
    int rows, cols;      // tile size (any size; the GPU does not require even dims)
    const float* b;      // m samples in [0,1]
    const int* rix;      // m sample row indices (tile-local)
    const int* riy;      // m sample column indices (tile-local)
    int m;
    float* x;            // rows*cols floats: DCT-coefficient warm start in, solution out
    float lambda;        // L1 weight of this tile (varies with --per-tile-coef)
    float tv = 0.0f;     // smoothed-TV weight; > 0 runs the backtracking branch (CPU core contract)
};

struct Params {          // shared by every tile of one call
    int inner;           // FISTA steps per reweighting pass
    int reweights;       // IRLS passes
    float wscale;        // highest-frequency weight multiplier (DC x1); <= 1 = uniform
};

// Reweighted-L1 FISTA on every tile, matching the CPU cs_fista_core_single
// contract (DCT basis, no Onsager): tv = 0 tiles take the exact fixed step 1/2;
// tv > 0 tiles add the smoothed-TV term and backtrack per tile (L from 2 + 8*tv,
// doubled until f(z) <= Q_L(z, y), at most 25 trials), as the CPU core does.
// Returns false when the device is missing or a HIP call fails; then no x is modified.
bool solve_fista_dct(const std::vector<TileProblem>& tiles, const Params& p);

// The YCC decode's LR stage with its tail on the device. planes holds Y, Cr, Cb of job j at 3j, 3j+1, 3j+2 (their x
// are only read); each group is one solve_fista_dct call (tv = 0 tiles, fused kernel). Each job's planes then take the
// CPU tail's steps on the device: inverse DCT x 255, chroma bilinear to the luma size (cv::resize INTER_LINEAR), 8 bit
// (convertTo), YCrCb -> BGR (OpenCV fixed point). out receives the jobs' BGR tiles back to back; its device copy is
// left in gpu_common's resident slot for the cs-sr stage. Returns false (out untouched) when a plane needs another
// solver, or on a HIP failure.
struct PlaneGroup { Params prm; std::vector<int> planes; };
bool solve_ycc_bgr(const std::vector<TileProblem>& planes, const std::vector<PlaneGroup>& groups, unsigned char* out);

}  // namespace gpu_tiles
