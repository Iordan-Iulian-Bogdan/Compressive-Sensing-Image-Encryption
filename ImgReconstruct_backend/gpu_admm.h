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
};

struct Params {           // shared by every tile of one call
    int admm;             // ADMM iterations
    int inner;            // FISTA steps per ADMM iteration
    float rho;            // ADMM penalty (the core's anchor weight)
    float lip;            // basis Lipschitz (cs_fista_basis::lip)
    float wscale;         // highest-frequency weight multiplier; <= 1 = uniform
};

// True when a HIP device is present and usable.
bool available();

// True when a tile is within what the kernels support.
bool supported(int rows, int cols);

// Runs the whole ADMM loop on every tile. Returns false (no z modified) when
// the device is missing, a tile is unsupported or a HIP call fails.
bool solve_pnp_admm(const std::vector<Problem>& tiles, const Params& p);

// OpenCV-compatible fastNlMeansDenoising(src, dst, h, 7, 15) on 8-bit planes,
// exposed for validation. src/dst are rows*cols bytes.
struct NlmPlane { int rows, cols; const std::uint8_t* src; std::uint8_t* dst; float h; };
bool nlm_denoise_u8(const std::vector<NlmPlane>& planes);

}  // namespace gpu_admm
