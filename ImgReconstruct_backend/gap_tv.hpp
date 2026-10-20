#pragma once
// GAP-TV reconstruction of one colour channel from point samples (prototype).
#include <opencv2/core.hpp>
#include <vector>

/** @brief Chambolle (2004) TV prox: v = argmin 0.5*||v - x||^2 + lam*TV(v).
px/py (dual field, n floats each, zero-initialised by the caller) persist across
calls to warm-start; d is n floats of scratch. */
void cs_tv_chambolle(const float* x, float* v, float* px, float* py, float* d,
    int rows, int cols, float lam, int iters);

/** @brief Samples of one channel in [0,1] (b) and a smooth starting image (fill)
built from them by normalized Gaussian convolution. Shared by the TV solvers. */
void cs_point_sample_fill(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    std::vector<float>& b, std::vector<float>& fill);

/** @brief Generalized alternating projection with a total-variation denoiser.
For point sampling the projection step pins each sampled pixel to its measured
value; the denoiser step is a Chambolle TV prox. pixel_measurements holds the
container header followed by one BGR sample per (ri_x[k] = row, ri_y[k] = col).
Writes the pixel-domain result (0..255) into ref (rows x cols, CV_32F). The
warm start already in ref is ignored: the solver starts from a smooth fill of
the samples. lambda is the TV weight on [0,1] intensities. */
void cs_gaptv_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float lambda, cv::Mat& ref);
