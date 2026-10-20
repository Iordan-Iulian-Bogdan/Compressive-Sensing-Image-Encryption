#pragma once
// Plug-and-play NLM denoiser inside the main reconstruction (prototype).
#include <opencv2/core.hpp>
#include <vector>

/** @brief Consensus ADMM for point-sampled tiles: the sampled pixels are pinned
to their measured values, one branch applies a DCT soft-threshold (lambda_l1,
skipped when 0), the other applies fastNlMeansDenoising with strength h (gray
levels, 8-bit scale) as the implicit image prior. Same contract as
cs_gaptv_channel: writes the 0..255 pixel result into ref (rows x cols, CV_32F)
and ignores the warm start already in ref. With lambda_l1 > 0, rows and cols
must be even (cv::dct). */
void cs_pnp_nlm_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float h, float lambda_l1, cv::Mat& ref);
