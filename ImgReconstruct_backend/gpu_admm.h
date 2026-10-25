#pragma once
// Batched GPU (HIP) PnP-ADMM super-resolution stage: the cs-sr FISTA core, the
// NLM proximal step, the DCTs and the dual update of cs_sr_upscale_tile_2x
// (pnp_mode 2, NLM prox, luma only), for many tiles at once. Plain C++ header;
// implementation in gpu_admm.hip.
#include <cstdint>
#include <vector>

namespace gpu_admm {

// One HR luma tile. rows/cols are the HR size (even, >= 12); the LR grid is
// rows/2 x cols/2 and rix/riy index it.
struct Problem {
    int rows, cols;
    const float* b;       // m LR samples in [0,1] (luma)
    const int* rix;       // m LR row indices
    const int* riy;       // m LR column indices
    int m;
    float* z;             // rows*cols HR pixels in [0,1]: initial z in, final z out
    float lambda;         // L1 weight of the cs-sr core (CPU: coef * 0.1)
    float nlm_h;          // fastNlMeansDenoising strength of the prox
    // Exact-HR anchors (--hr-detail / sidecar): rows*cols values in [0,1], < -1e29 where
    // the pixel has none; null = no anchors. Weight comes from Params::hr_weight.
    const float* hr_val = nullptr;
    float tv = 0.0f;      // smoothed-TV weight of the core (needs Params::bt = 1 when > 0)
};

struct Params {           // shared by every tile of one call
    int admm;             // ADMM iterations
    int inner;            // FISTA steps per ADMM iteration
    float rho;            // ADMM penalty (the core's anchor weight)
    float lip;            // basis Lipschitz (cs_fista_basis::lip)
    float wscale;         // highest-frequency weight multiplier; <= 1 = uniform
    // Per-round schedule (see cs_sr_admm_sched in superres.cpp): rho runs geometrically
    // from 'rho' to rho_end (<= 0: constant), the NLM strength is scaled by a factor
    // running from h_start to h_end, and rescale_u != 0 rescales the dual by
    // rho_old / rho_new when rho changes. Defaults keep every round identical.
    float rho_end = 0.0f;
    float h_start = 1.0f, h_end = 1.0f;
    int rescale_u = 0;
    // Anchor data-term weight (hr_w): adds 2*hr_weight to the step Lipschitz like the CPU
    // core, so anchored and anchor-free tiles are batched apart (it is part of the key).
    float hr_weight = 0.0f;
    // 1 = tv > 0 batch: explicit gradient + smoothed TV + per-tile backtracking (CPU
    // cs_sr_core_single bt branch) instead of the fused fixed-step epilogues.
    int bt = 0;
};

// True when a HIP device is present and usable.
bool available();

// True when a tile is within what the kernels support.
bool supported(int rows, int cols);

// Runs the whole ADMM loop on every tile. Returns false (no z modified) when
// the device is missing, a tile is unsupported or a HIP call fails.
bool solve_pnp_admm(const std::vector<Problem>& tiles, const Params& p);

// Same loop with z resident on the device (Problem::z unused): dZ holds every tile's HR
// pixels in [0,1], tiles laid out back to back in order (gpu_common TileLayout::add order),
// initial z in, final z out. Used by the batched tile pipeline (gpu_sr.hip).
bool solve_pnp_admm_device(const std::vector<Problem>& tiles, const Params& p, float* dZ);

// OpenCV-compatible fastNlMeansDenoising(src, dst, h, 7, 15) on 8-bit planes,
// exposed for validation. src/dst are rows*cols bytes.
struct NlmPlane { int rows, cols; const std::uint8_t* src; std::uint8_t* dst; float h; };
bool nlm_denoise_u8(const std::vector<NlmPlane>& planes);

}  // namespace gpu_admm
