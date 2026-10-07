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
// - "fsrcnn": FSRCNN 2x neural upscale via OpenCV dnn_superres
//   (cs_fsrcnn_upscale_2x, in-process CPU, ~10-20 ms/tile). Needs the
//   FSRCNN_x2.pb weights (opt.fsrcnn_model); missing model or a build
//   without opencv2/dnn_superres.hpp falls back to AVIR per tile.
// - "cs" / "cs-sr": CS super-resolution refinement
//   (cs_sr_upscale_tile_2x in helper_functions.hpp): the HR tile's DCT
//   coefficients are re-solved against the tile's original LR samples
//   (DCT + TV, CPU). Handled as a fused 2x path in the decrypt pipeline,
//   not through the batch helpers below; falls back to AVIR per tile.
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
//   quality as nunif waifu2x, no Python. Args are `-n <noise|-1> -s 2
//   -m <model>` (`scale` method maps to `-n -1`, i.e. no denoise; the
//   default model is the photo one, models-upconv_7_photo).
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
  // RED-lite NLM strength for the cs-sr backend (0 = off): blends each
  // outer FISTA pass toward its fastNlMeans-denoised self. Ignored by all
  // other backends (including AVIR fallback tiles).
  float sr_red = 0.0f;
  // RED denoiser select for cs-sr ("nlmeans" luma fastNlMeans, "dncnn"
  // color DnCNN model below). Ignored unless sr_red > 0.
  std::string sr_red_denoiser = "nlmeans";
  // DnCNN color model path (used only with sr_red_denoiser == "dncnn").
  std::string sr_dncnn_model = "models/dncnn/dncnn_color.onnx";
  // Coupled-dictionary SR model (CSD2) for --sr-dict: when non-empty, the
  // fused 2x path refines tiles with patch-OMP synthesis (Dh) from LR
  // gradient-feature codes (Dl) instead of the FISTA HR solve, in every
  // container mode (it operates on solved tiles). --red is ignored there.
  std::string sr_dict;
  // FSRCNN 2x model for the "fsrcnn" backend (OpenCV dnn_superres,
  // in-process CPU; default resolves against the working directory).
  // A missing/unreadable model falls back to AVIR per tile.
  std::string fsrcnn_model = "FSRCNN_x2.pb";
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
  // Model dir (or short name resolved against the binary's directory):
  // photo default; use models-cunet for anime-style content.
  std::string waifu2x_ncnn_model =
      cs_env_or("CS_WAIFU2X_NCNN_MODEL", "models-upconv_7_photo");
  std::string realcugan_cmd = cs_env_or("CS_REALCUGAN_CMD", "realcugan-ncnn-vulkan");
  std::string realcugan_args = cs_env_or("CS_REALCUGAN_ARGS", "");
  std::string realcugan_model = cs_env_or("CS_REALCUGAN_MODEL", "models-se");
};

// Effective args for the subprocess: the explicit override when set,
// otherwise `--style photo --method <method> -n <noise> -g -1`.
std::string cs_waifu2x_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective args for waifu2x-ncnn-vulkan: `-n <noise> -s 2 -m <model>`, or
// `-n -1 -s 2 -m <model>` for pure upscale (`scale` method). Returns the
// explicit override when set.
std::string cs_waifu2x_ncnn_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective -m directory for waifu2x-ncnn: a model-dir name (default
// models-upconv_7_photo) or an explicit path. Plain names resolve against
// the ncnn binary's directory first (models ship next to the exe), then
// against the current directory; explicit paths pass through untouched.
std::string cs_waifu2x_ncnn_model_dir(const CsPhotoUpscalerOptions& opt);

// Effective args for realcugan-ncnn-vulkan: `-n <noise> -s 2 -m <model>`,
// `-n -1` for pure upscale. Returns the explicit override when set.
std::string cs_realcugan_effective_args(const CsPhotoUpscalerOptions& opt);

// Effective -m directory for realcugan: accepts short names (se/pro/nose,
// with or without the models- prefix) or an explicit path. Short names
// resolve against the realcugan binary's directory first (models ship next
// to the exe), then against the current directory; explicit paths (absolute
// or containing a separator) pass through untouched.
std::string cs_realcugan_model_dir(const CsPhotoUpscalerOptions& opt);

// Upscale one LR tile to 2x HR with the selected backend: AVIR runs
// in-process; waifu2x / waifu2x-ncnn / realcugan run one subprocess on the
// single tile (same -i/-o dir contract as the batch path, one temp dir per
// call). Any backend failure (bad output size, nonzero exit, missing
// binary) falls back to AVIR for the tile. Returns true when the selected
// backend produced dst_hr, false when the AVIR fallback (or nothing) did.
// Used by the streaming upscale pipeline: one serialized subprocess per
// tile while solving continues, instead of one post-join batch.
bool cs_upscale_one_tile_2x(const cv::Mat& src_lr, cv::Mat& dst_hr,
                            const CsPhotoUpscalerOptions& opt);

// Upscale one LR tile exactly 2x with FSRCNN (OpenCV dnn_superres, CPU).
// True on success; false when the model is missing/unreadable, the output
// violates the 2x CV_8UC3 contract, or the build lacks
// opencv2/dnn_superres.hpp (caller falls back to AVIR per tile).
bool cs_fsrcnn_upscale_2x(const cv::Mat& src, cv::Mat& dst,
                          const std::string& model);

// Upscale every non-empty CV_8UC3 tile of the grid exactly 2x, in place.
// The caller then scales the tile origins by 2 (all of them; empty tiles
// are skipped by reconstruct/blend regardless of their coordinates).
// Postcondition: every non-empty CV_8UC3 tile is 2x its input size.
void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt);

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
