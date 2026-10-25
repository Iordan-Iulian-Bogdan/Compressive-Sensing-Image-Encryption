#pragma once
// GPU compositing of the solved HR tiles (plain C++ header; implementation in gpu_composite.hip).
//
// blend_tiles reproduces blendTilesWithImage's cosine-feathered path (image_tiles.cpp, feather
// > 0) bit for bit: per pixel sum over the covering tiles in (i, j) order of tile * w, with the
// weights computed by the CPU's own cs_feather_weights, divide by max(weight sum, 1e-6), round
// half to even to 8 bits. Float operations are issued without contraction.
#include <cstdint>
#include <vector>

namespace gpu_composite {

struct TileRef {
    int i, j;                    // grid position (blend order is i-major, then j)
    int x, y, w, h;              // placement and size in the output canvas (inside it)
    const std::uint8_t* bgr;     // w*h*3 bytes, continuous
    const float* weights;        // w*h feather weights (cs_feather_weights of this placement)
};

// Returns false (out untouched) when the device is missing, a HIP call fails, or the tiles do
// not form a regular grid (x and width depending only on j, y and height only on i).
bool blend_tiles(const std::vector<TileRef>& tiles, int out_w, int out_h, std::uint8_t* out_bgr);

}  // namespace gpu_composite
