#pragma once
// Smooth fill of a point-sampled plane (the warm start of the GPU tile solves).
#include <opencv2/core.hpp>
#include <vector>

/** @brief Samples of one channel in [0,1] (b) and a smooth starting image (fill)
built from them by normalized Gaussian convolution. */
void cs_point_sample_fill(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    std::vector<float>& b, std::vector<float>& fill);
