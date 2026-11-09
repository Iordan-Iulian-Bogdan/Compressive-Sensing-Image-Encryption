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
    const cv::Mat* cr_mask = nullptr;                   // chroma sample masks / values (CV_8U), output / 4 (4:2:0 of LR)
    const cv::Mat* cr_samp = nullptr;
    const cv::Mat* cb_mask = nullptr;
    const cv::Mat* cb_samp = nullptr;
    int y_bits = 8, c_bits = 8;                         // sample depths (cs_enh_project's quantization bins)
    const cv::Mat* active = nullptr;                    // HIP path: 8x8 output blocks read (CV_8U, ceil(H / 8) x ceil(W / 8)),
                                                        // the rest may stay unenhanced; null = all
};

/** @brief The extra input planes (CV_32F, the image's size) in the model's channel order: density - 0.5, anchor
indicator, anchor residual / 64, LR sample mask, LR sample residual / 8, then (n = 9) Cr sample mask, Cr sample
residual / 4, Cb sample mask, Cb sample residual / 4 (chroma samples on their 4x4 output blocks; residuals clipped to
[-1, 1]); `n` = 3, 5 or 9 first planes. Matches enh_inputs.extra_planes up to the density blur (OpenCV Gaussian vs PIL's
box approximation). */
void cs_enh_planes(const cv::Mat& bgr, const CsEnhInputs& in, int n, std::vector<cv::Mat>& planes);

/** @brief Runs the restoration net on `bgr` (CV_8UC3) in place: RGB [0, 1] + extra planes in, enhanced RGB out (clamped,
rounded to 8 bit). GPU build: on HIP (gpu_enh.hip: planes, whole-image network and output on the GPU) when the weight
file exported next to the model (<stem>.enh.bin) loads, unless CS_GPU_ENH=0. Otherwise ONNX Runtime on `model_path`:
1024 px tiles with 64 px context, reflect-padded to multiples of 4, DirectML when CS_GPU_ONNX=1 in the GPU build, else
CPU; the plane count comes from the model's input channels (3 + 3, 3 + 5 or 3 + 9). Returns false (image untouched, warning
printed) when no path can run. CS_ENH_DUMP=<prefix> (ONNX path) writes the input image (<prefix>_in.png) and each
plane (<prefix>_p<k>.f32, raw float32 rows) for checking against the Python side. */
bool cs_enhance_net(cv::Mat& bgr, const std::string& model_path, const CsEnhInputs& in);

/** @brief Data consistency after the net: moves `bgr` (CV_8UC3, in place) onto the container's measurements. Each
sampled 4x4 block's mean Cr / Cb into its sample's quantization bin (+- pad; luma kept), then each sampled 2x2 block's
mean luma into its bin (same shift on B, G, R), then each anchor pixel's luma to the anchor. Only the measured
components change; pixels and blocks without samples are untouched. CS_ENH_DC=0 skips it. */
void cs_enh_project(cv::Mat& bgr, const CsEnhInputs& in);

/** @brief GPU build: loads and uploads the HIP weights of `model_path` (as cs_enhance_net would on first use), so a
decode can do it on a side thread while it unpacks and solves. No-op in the CPU build or with CS_GPU_ENH=0. */
void cs_enhance_prefetch(const std::string& model_path);

/** @brief GPU build: loads `model_path` and reserves device-arena room for its run on an H x W image (large images
only), so the multi-scale decode's biggest layer finds its activations in the arena instead of paying hipMalloc under
the GPU lock. Meant for a side thread. No-op in the CPU build or with CS_GPU_ENH=0. */
void cs_enhance_reserve(const std::string& model_path, int H, int W);

/** @brief Receptive-field radius (full-res px, upper bound) of the HIP weights of `model_path`; 0 when the HIP path
is off or they do not load (CsEnhInputs::active is then ignored). */
int cs_enhance_receptive_field(const std::string& model_path);
