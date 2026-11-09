#pragma once
// Batched GPU solve of decrypt tiles (plain C++, no HIP headers). Built only in
// the GPU build (-DCS_GPU_TILES, see build_gpu.bat).
#include <opencv2/core.hpp>
#include <functional>
#include <vector>

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
warm start, no neighbours), then each tile is merged as decrypt_tiles_ycc420's CPU tail (chroma bilinear to
the luma size, YCrCb -> BGR, 8 bit), on the device: float rounding moves a few pixels by a level against the host
tail (CS_GPU_LR_HOST_TAIL=1 keeps that one), and the outs are views into one buffer, whose device copy gpu_sr
takes. solved[k] = 1 when job k's out is set. on_gpu, when set, is
called once the host prep is done and the GPU batches start (the caller's own host work can overlap them). */
void cs_gpu_solve_tiles_ycc(std::vector<GpuYccTileJob>& jobs, std::vector<char>& solved,
    const std::function<void()>& on_gpu = {});
