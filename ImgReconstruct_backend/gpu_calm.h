#pragma once
// Segment boundary calm (cs_seg_calm_boundaries in image_decryption.cpp) on HIP. Plain C++ header; implementation in
// gpu_calm.hip.
//
// Same result as the CPU pass up to two approximations: the distance to the class frontier is exact Euclidean (the CPU
// uses OpenCV's 3x3 chamfer DIST_L2) and the bilateral colour weight is evaluated directly (OpenCV interpolates a table).
// Only pixels whose gate is non-zero are computed: inside a low-density class, closer than `overlap` to another label.
#include <cstdint>

namespace gpu_calm {

// img: H x W x 3 8-bit BGR (host, updated in place). labels: h x w segment ids (host), nearest-upscaled to the image
// (label of image pixel (y, x) = labels[floor(y h / H)][floor(x w / W)]). low[c] marks the low-density classes.
// overlap: gate width in image pixels. ms, when given, receives the GPU time (upload, kernel, download) in milliseconds.
// Returns false on a HIP failure (img untouched).
bool run(std::uint8_t* img, int H, int W, const std::uint8_t* labels, int h, int w, const bool low[256], int overlap,
    double* ms = nullptr);

}  // namespace gpu_calm
