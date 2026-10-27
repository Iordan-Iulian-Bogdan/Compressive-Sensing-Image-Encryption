#pragma once
// Restoration CNN (--enhance): a same-size U-Net run on the finished decode, fed the decoder-side planes it was trained
// with. Model and planes: x64/GpuBuild/train_enh_fresh.py (UNetX), enh_inputs.py, exported by export_enh_onnx.py.
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "crypto_utils.hpp"   // CsHrSample

/** @brief What the decoder knows besides the decoded image (all optional: a missing one leaves its planes at zero). */
struct CsEnhInputs {
    const cv::Mat* y_mask = nullptr;                    // LR luma sample mask (CV_8U, nonzero = sampled), output / 2
    const cv::Mat* y_samp = nullptr;                    // LR luma sample values (CV_8U), same grid
    const std::vector<CsHrSample>* anchors = nullptr;   // full-res luma anchors (idx = row * W + col)
};

/** @brief The extra input planes (CV_32F, the image's size) in the model's channel order: density - 0.5, anchor
indicator, anchor residual / 64, LR sample mask, LR sample residual / 8 (residuals clipped to [-1, 1]); `n` = 3 or 5
first planes. Matches enh_inputs.extra_planes up to the density blur (OpenCV Gaussian vs PIL's box approximation). */
void cs_enh_planes(const cv::Mat& bgr, const CsEnhInputs& in, int n, std::vector<cv::Mat>& planes);

/** @brief Runs the restoration net on `bgr` (CV_8UC3) in place: RGB [0, 1] + extra planes in, enhanced RGB out (clamped,
rounded to 8 bit). GPU build: on HIP (gpu_enh.hip: planes, whole-image network and output on the GPU) when the weight
file exported next to the model (<stem>.enh.bin) loads, unless CS_GPU_ENH=0. Otherwise ONNX Runtime on `model_path`:
1024 px tiles with 64 px context, reflect-padded to multiples of 4, DirectML when CS_GPU_ONNX=1 in the GPU build, else
CPU; the plane count comes from the model's input channels (3 + 3 or 3 + 5). Returns false (image untouched, warning
printed) when no path can run. CS_ENH_DUMP=<prefix> (ONNX path) writes the input image (<prefix>_in.png) and each
plane (<prefix>_p<k>.f32, raw float32 rows) for checking against the Python side. */
bool cs_enhance_net(cv::Mat& bgr, const std::string& model_path, const CsEnhInputs& in);
