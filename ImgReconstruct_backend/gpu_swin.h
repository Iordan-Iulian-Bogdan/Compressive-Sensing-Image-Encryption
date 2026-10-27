#pragma once
// Swin2SR-lightweight x2 on HIP (the --sr-prior-model network), for batches of small images.
// Plain C++ header; implementation in gpu_swin.hip.
//
// The weights come from <model>.swin2sr.bin, written by scripts/export_swin2sr_weights.py from the
// ONNX model; that script also checks its PyTorch reference of this math against ONNX Runtime.
// Results match ONNX Runtime to float rounding (not bit for bit), so an 8-bit output can differ
// by one level where a value sits on a rounding edge.
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_swin {

// Path of the weight file that belongs to an ONNX model ("<dir>/<stem>.swin2sr.bin").
std::string weights_path(const std::string& onnx_model);

// Loads (once per path) and uploads the weights; false when the file is missing or malformed.
bool load(const std::string& weights_file);

// One input image: rows x cols 8-bit BGR pixels at pixel offset lr_off of the input buffer, output
// 2 rows x 2 cols 8-bit BGR at pixel offset hr_off of the output buffer. rows and cols must be
// multiples of 8 (the attention window); supported() says whether an image can run.
struct Image { int rows, cols; long long lr_off, hr_off; };
bool supported(int rows, int cols);

// Runs the network on every image (device buffers, 3 bytes per pixel, BGR) and writes
// round(255 * output) clamped to 0..255, as OpenCV's convertTo(CV_8U, 255) does. Images run in
// chunks that bound the working memory. Returns false on a HIP failure or if load() did not succeed.
bool run(const std::vector<Image>& images, const std::uint8_t* d_lr_bgr, std::uint8_t* d_hr_bgr);

}  // namespace gpu_swin
