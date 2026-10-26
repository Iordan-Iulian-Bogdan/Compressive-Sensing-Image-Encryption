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
    float tv = 0.0f;               // smoothed-TV weight (> 0: backtracking branch, as the CPU core)
};

/** @brief Solves the three colour channels of every job with reweighted-L1
FISTA (DCT, any tv) on the GPU, in batches that share (steps, reweights,
wscale). Unlike the CPU path there is no neighbour warm start: each tile starts
from a smooth fill of its own samples (transformed to DCT coefficients), so all
tiles can be solved at once. solved[k] = 1 when job k finished (its out is set);
jobs with odd dimensions, no samples, or a failed GPU call are left for the CPU
path. */
void cs_gpu_solve_tiles(std::vector<GpuTileJob>& jobs, std::vector<char>& solved);

/** @brief One plane of a YCC tile: the arguments reconstruct_color_channel_fista takes (channel 0). */
struct GpuPlane {
    const cv::Mat* measurements;   // plane container: header + samples in channel 0
    int rows, cols;
    const std::vector<int>* rix;
    const std::vector<int>* riy;
    int iterations;
    float coef, tv;
};

struct GpuYccTileJob {
    GpuPlane plane[3];             // Y at tile resolution, Cr and Cb on the coarse grid
    int fista_iters, reweights;
    float wscale;
    cv::Mat* out;                  // receives the merged CV_8UC3 BGR tile (Y's size)
};

/** @brief YCC form of cs_gpu_solve_tiles: every plane of every job is solved on the GPU (smooth-fill
warm start, no neighbours), then each tile is merged exactly like decrypt_tiles_ycc420's CPU tail
(chroma bilinear to the luma size, YCrCb -> BGR, 8 bit). solved[k] = 1 when job k's out is set. */
void cs_gpu_solve_tiles_ycc(std::vector<GpuYccTileJob>& jobs, std::vector<char>& solved);
