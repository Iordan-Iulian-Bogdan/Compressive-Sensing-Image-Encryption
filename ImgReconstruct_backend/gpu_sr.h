#pragma once
// Batched GPU cs-sr tile stage (plain C++ header; implementation in gpu_sr.hip).
//
// For many solved LR tiles at once, the whole PnP-ADMM path of cs_sr_upscale_tile_2x on the
// device, in one pass:
//   NLM strength from the tile's Immerkaer noise estimate (cs_sr_prox_nlm_h)
//   AVIR 2x warm start (separable matrices, gpu_avir_tables.hpp, + AVIR's exact dither)
//   BGR -> YCrCb (OpenCV 8-bit fixed point), luma / 255 as the ADMM start
//   luma PnP-ADMM (gpu_admm::solve_pnp_admm_device, z stays on the device)
//   round to 8 bits, merge with the AVIR Cr/Cb, YCrCb -> BGR
// Results match the CPU path up to AVIR's dither (within 1 level on a few percent of the
// warm-start pixels) and float summation order.
#include <cstdint>
#include <vector>

#include "gpu_admm.h"

namespace gpu_sr {

struct Tile {
    int lr_rows, lr_cols;          // LR tile; the HR tile is (2*lr_rows) x (2*lr_cols)
    const std::uint8_t* lr_bgr;    // lr_rows*lr_cols*3 bytes, continuous BGR
    const float* b;                // m luma samples in [0,1] on the LR grid
    const int* rix;                // m LR row indices
    const int* riy;                // m LR column indices
    int m;
    float lambda;                  // cs-sr L1 weight (coef * 0.1)
    float tv;                      // core TV weight (the batch's Params::bt must match tv > 0)
    const float* hr_val;           // HR anchor values (< -1e29 = none) or null
    std::uint8_t* hr_bgr;          // out: HR BGR, (2*lr_rows)*(2*lr_cols)*3 bytes
    std::uint8_t* dbg_avir = nullptr;   // optional out: the AVIR 2x BGR warm start (verification)
    float* dbg_nlm_h = nullptr;         // optional out: the NLM strength used (verification)
};

// Runs every tile through the pipeline with one shared ADMM parameter set (the caller groups
// tiles by Params). Every tile must satisfy gpu_admm::supported(2*lr_rows, 2*lr_cols).
// Returns false (outputs untouched) when the device is missing or a HIP call fails.
bool run(std::vector<Tile>& tiles, const gpu_admm::Params& prm);

}  // namespace gpu_sr
