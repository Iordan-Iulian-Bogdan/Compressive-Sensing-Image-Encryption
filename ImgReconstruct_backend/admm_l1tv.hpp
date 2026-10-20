#pragma once
// ADMM with DCT-L1 + TV priors for point-sampled tiles (prototype).
#include <opencv2/core.hpp>
#include <vector>

/** @brief Consensus ADMM for  min  lambda_l1*||DCT(x)||_1 + lambda_tv*TV(x)
subject to x[sample] = measured value. Each iteration soft-thresholds in the
DCT domain, applies a Chambolle TV prox, averages the two, and pins the
samples. Same contract as cs_gaptv_channel: writes the 0..255 pixel result into
ref (rows x cols, CV_32F); the existing warm start in ref is ignored. rows and
cols must be even (cv::dct). Weights are on [0,1] intensities. */
void cs_admm_l1tv_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float lambda_l1, float lambda_tv, cv::Mat& ref);
