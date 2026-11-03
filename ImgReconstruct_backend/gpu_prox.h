#pragma once
// Learned proximal step for the GPU PnP-ADMM loop (gpu_admm.hip), in place of the NLM prox when CS_ADMM_PROX names a
// weight file (x64/GpuBuild/admm_prox_export.py). A small residual CNN, z = q + f(q, h, planes): 3x3 convs with zero
// padding at each tile's border, ReLU between layers, 8-bit output like the NLM step. Inputs per HR pixel, as
// admm_prox_train.py builds them (cin = 2: the first two only; cin = 6: all):
//   q - 0.5, h / 16, anchor mask, anchor residual clamp((a - q) * 255 / 64, -1, 1),
//   LR sample mask (2x2 block), LR sample residual clamp((b - 2x2 mean of q) * 255 / 8, -1, 1)
// q is the 8-bit NLM input (px + u), h that round's NLM strength. Include from .hip files only.
#include <cstdint>
#include <string>

namespace gpu_common { struct TileDev; }

namespace gpu_prox {

// Loads the weight file once (later calls with the same path reuse it) and allocates the fixed scratch. False when it
// cannot be read, its layout is not one run() supports or the allocation fails.
bool load(const std::string& path);

// One prox step on every tile (device pointers in the gpu_admm layout; td_host the same records on the host, for the
// chunking): Q is the padded 8-bit input (offsets qoff, `border` pixels each side), BLR the per-tile LR samples
// (offset off / 4, rows/2 x cols/2, < -1e29 where none), HAV the per-pixel anchors (< -1e29 where none; null = no
// anchors), H the per-tile NLM strengths (times hmul this round). DN receives the 8-bit output.
bool run(int nt, int max_rows, int max_cols, const gpu_common::TileDev* td, const gpu_common::TileDev* td_host,
    const int* qoff, int border, const std::uint8_t* Q, const float* BLR, const float* HAV, const float* H, float hmul,
    std::uint8_t* DN);

}  // namespace gpu_prox
