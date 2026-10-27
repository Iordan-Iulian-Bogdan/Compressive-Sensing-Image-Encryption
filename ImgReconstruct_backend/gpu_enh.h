#pragma once
// Restoration U-Net of --enhance on HIP: input planes, the whole network and the 8-bit output on the GPU, one image at
// a time (no tiling). Plain C++ header; implementation in gpu_enh.hip.
//
// The weights come from <model>.enh.bin, written next to the ONNX model by x64/GpuBuild/export_enh_hip.py from the
// PyTorch checkpoint (UNetX in train_enh_fresh.py). The planes follow enh_inputs.py / cs_enh_planes (enh_net.hpp); the
// convolutions run on the RDNA3 WMMA units (fp16 inputs, fp32 accumulation, fp16 activations).
#include <cstdint>
#include <string>

namespace gpu_enh {

// Weight file that belongs to an ONNX model ("<dir>/<stem>.enh.bin").
std::string weights_path(const std::string& onnx_model);

// Loads (once per path) and uploads the weights; false when the file is missing or malformed.
bool load(const std::string& weights_file);

// Extra input planes of the loaded model (3 or 5), 0 when none is loaded.
int extra_planes();

// One full-res luma anchor; the layout of CsHrSample (crypto_utils.hpp).
struct Anchor { std::uint32_t idx; std::uint8_t y; };

struct Inputs {
    const std::uint8_t* bgr;            // H x W x 3 decoded image (host)
    int H, W;
    const std::uint8_t* y_mask;         // h x w LR luma sample mask (nonzero = sampled), null = none
    const std::uint8_t* y_samp;         // h x w LR luma sample values
    int h, w;
    const Anchor* anchors;              // full-res anchors (idx = row * W + col), may be null
    int n_anchors;
};

// Runs the loaded model on one image and writes the enhanced H x W x 3 BGR bytes to out_bgr (host; may alias
// in.bgr). ms, when given, receives {upload + planes, network, download} in milliseconds. False on a HIP failure or
// when nothing is loaded; out_bgr is then untouched.
bool run(const Inputs& in, std::uint8_t* out_bgr, double* ms = nullptr);

}  // namespace gpu_enh
