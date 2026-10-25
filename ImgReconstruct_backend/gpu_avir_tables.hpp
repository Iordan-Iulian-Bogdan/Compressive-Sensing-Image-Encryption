#pragma once
// AVIR 2x (cs_upscale_2x_avir) as separable per-channel matrices, for the GPU tile pipeline.
//
// AVIR's resampling is linear and separable per channel: for an R x C tile, before its 8-bit
// error-diffusion dither, out = A_r (2R x R) * X * A_c^T (A_c: 2C x C) on the 0..255 scale.
// The matrices are extracted from AVIR itself (3-channel interleaved runs, the production
// layout; channels do not mix) and cached per size. Validated by x64/GpuBuild/tools/
// avir_probe.cpp: fit error ~1e-4 levels; with the exact dither replica the 8-bit result
// matches AVIR within 1 level (the dither amplifies the 1e-4 float differences into +-1 on a
// few percent of pixels, never 2).
#include <vector>

// Row-major A_r (2R x R) and A_c (2C x C); pointers stay valid for the process lifetime.
// Thread-safe. Returns false for sizes AVIR cannot handle (R or C < 1).
bool cs_avir_2x_tables(int R, int C, const std::vector<float>*& Ar, const std::vector<float>*& Ac);
