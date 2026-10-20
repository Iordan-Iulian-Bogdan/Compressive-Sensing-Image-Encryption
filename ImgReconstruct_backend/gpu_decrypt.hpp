#pragma once
// Batched GPU solve of decrypt tiles (plain C++, no HIP headers). Built only in
// the GPU build (-DCS_GPU_TILES, see build_gpu.bat).
#include <opencv2/core.hpp>
#include <vector>

struct GpuTileJob {
    const cv::Mat* measurements;   // tile container: header + BGR samples
    int rows, cols;
    const std::vector<int>* rix;   // sample row indices (tile-local)
    const std::vector<int>* riy;   // sample column indices (tile-local)
    int iterations;                // luma budget; chroma gets iterations - iterations/2
    int fista_iters, reweights;
    float coef, wscale;
    cv::Mat* out;                  // receives the solved CV_8UC3 tile
};

/** @brief Solves the three colour channels of every job with reweighted-L1
FISTA (DCT, tv = 0) on the GPU, in batches that share (steps, reweights,
wscale). Unlike the CPU path there is no neighbour warm start: each tile starts
from a smooth fill of its own samples (transformed to DCT coefficients), so all
tiles can be solved at once. solved[k] = 1 when job k finished (its out is set);
jobs with odd dimensions, no samples, or a failed GPU call are left for the CPU
path. */
void cs_gpu_solve_tiles(std::vector<GpuTileJob>& jobs, std::vector<char>& solved);
