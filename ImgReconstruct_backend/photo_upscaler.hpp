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
// - "waifu2x-ncnn": same harness around the native waifu2x-ncnn-vulkan
//   binary (Vulkan, runs on AMD/NVIDIA/Intel GPUs): same models and
//   quality as nunif waifu2x, no Python. Args are `-n <noise|-1> -s 2`
//   (`scale` method maps to `-n -1`, i.e. no denoise).
// - "realcugan": same harness around realcugan-ncnn-vulkan (Vulkan):
//   `-n <noise|-1> -s 2 -m <model>` with `models-se` default; the `-n`
//   levels are the direct analog of waifu2x `noise_scale` denoising.
//
// Env defaults (read at startup): CS_WAIFU2X_CMD / CS_WAIFU2X_ARGS,
// CS_WAIFU2X_NCNN_CMD / CS_WAIFU2X_NCNN_ARGS, CS_REALCUGAN_CMD /
// CS_REALCUGAN_ARGS / CS_REALCUGAN_MODEL. Explicit --*-cmd/--*-args flags
// always win over the env.
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

inline std::string cs_env_or(const char* var, const std::string& fallback) {
  if (const char* env = std::getenv(var))
    if (env[0]) return std::string(env);
  return fallback;
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
  // Native ncnn backends share the method/noise fields above.
  std::string waifu2x_ncnn_cmd = cs_env_or("CS_WAIFU2X_NCNN_CMD", "waifu2x-ncnn-vulkan");
  std::string waifu2x_ncnn_args = cs_env_or("CS_WAIFU2X_NCNN_ARGS", "");
  std::string realcugan_cmd = cs_env_or("CS_REALCUGAN_CMD", "realcugan-ncnn-vulkan");
  std::string realcugan_args = cs_env_or("CS_REALCUGAN_ARGS", "");
  std::string realcugan_model = cs_env_or("CS_REALCUGAN_MODEL", "models-se");
};

// Effective args for the subprocess: the explicit override when set,
// otherwise `--style photo --method <method> -n <noise> -g -1`.
std::string cs_waifu2x_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective args for waifu2x-ncnn-vulkan: `-n <noise> -s 2`, or `-n -1 -s 2`
// for pure upscale (`scale` method). Returns the explicit override when set.
std::string cs_waifu2x_ncnn_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective args for realcugan-ncnn-vulkan: `-n <noise> -s 2 -m <model>`,
// `-n -1` for pure upscale. Returns the explicit override when set.
std::string cs_realcugan_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective -m directory for realcugan: accepts short names (se/pro/nose,
// with or without the models- prefix) or an explicit path. Short names
// resolve against the realcugan binary's directory first (models ship next
// to the exe), then against the current directory; explicit paths (absolute
// or containing a separator) pass through untouched.
std::string cs_realcugan_model_dir(const CsPhotoUpscalerOptions& opt);

// Upscale every non-empty CV_8UC3 tile of the grid exactly 2x, in place.
// The caller then scales the tile origins by 2 (all of them; empty tiles
// are skipped by reconstruct/blend regardless of their coordinates).
// Postcondition: every non-empty CV_8UC3 tile is 2x its input size.
void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt);

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
