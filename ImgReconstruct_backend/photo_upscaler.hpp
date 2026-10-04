#ifndef IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
#define IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_

#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Optional per-tile 2x photo-upscaler backend for the decrypt pipeline.
//
// The decrypt solver works in a compact geometry and every solved tile is
// upscaled 2x before blending. Which upscaler does that is selected here:
//
// - "avir" (default): the vendored AVIR resampler (cs_upscale_2x_avir).
//   Zero dependencies, millisecond-scale per tile, always available.
// - "waifu2x": one batched nunif subprocess over the whole tile grid:
//   tiles are written to a temp dir, a single
//     <waifu2x_cmd> -i <in_dir> -o <out_dir> <waifu2x_args>
//   run upscales them (default args select the photo model, pure 2x
//   upscale, CPU), and the results are read back. Any tile that fails or
//   comes back at the wrong size falls back to AVIR with a warning, so a
//   missing Python/nunif install degrades to AVIR instead of failing the
//   decrypt.
//
// Env defaults (read at startup): CS_WAIFU2X_CMD / CS_WAIFU2X_ARGS.
// They let a machine without `python` on PATH (e.g. a portable embedded
// interpreter) keep working with plain `--photo-upscaler waifu2x`.
// Explicit --waifu2x-cmd/--waifu2x-args flags always win over the env.
//
// Requires nunif importable for the waifu2x backend:
//   pip install torch --index-url https://download.pytorch.org/whl/cpu
//   pip install nunif (or a local clone on PYTHONPATH / .pth)
//   python -m waifu2x.download_models
inline std::string cs_waifu2x_cmd_default() {
  if (const char* env = std::getenv("CS_WAIFU2X_CMD"))
    if (env[0]) return std::string(env);
  return "python -m waifu2x.cli";
}

inline std::string cs_waifu2x_args_default() {
  if (const char* env = std::getenv("CS_WAIFU2X_ARGS"))
    if (env[0]) return std::string(env);
  return "";
}

struct CsPhotoUpscalerOptions {
  std::string backend = "avir";
  std::string waifu2x_cmd = cs_waifu2x_cmd_default();
  // Full-arg override (explicit --waifu2x-args flag or CS_WAIFU2X_ARGS).
  // Empty (default) = compose from the fields below.
  std::string waifu2x_args = cs_waifu2x_args_default();
  // Denoising control: "scale" (pure 2x upscale, default) or
  // "noise_scale" (waifu2x denoise + 2x upscale) with strength
  // waifu2x_noise in [0, 3]. ("noise" alone is rejected: it does not
  // upscale, so it cannot satisfy the 2x tile contract.)
  std::string waifu2x_method = "scale";
  int waifu2x_noise = 0;
};

// Effective args for the subprocess: the explicit override when set,
// otherwise `--style photo --method <method> -n <noise> -g -1`.
std::string cs_waifu2x_effective_args(const CsPhotoUpscalerOptions& opt);

// Upscale every non-empty CV_8UC3 tile of the grid exactly 2x, in place.
// The caller then scales the tile origins by 2 (all of them; empty tiles
// are skipped by reconstruct/blend regardless of their coordinates).
// Postcondition: every non-empty CV_8UC3 tile is 2x its input size.
void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt);

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
