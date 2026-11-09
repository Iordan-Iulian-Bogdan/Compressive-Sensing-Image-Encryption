// Smooth fill of a point-sampled plane (see gap_tv.hpp).
#include "gap_tv.hpp"
#include "fista_kernels.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>

void cs_point_sample_fill(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    std::vector<float>& b, std::vector<float>& fill)
{
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);
    // Gaussian-weighted average of the samples (normalized convolution), width
    // scaled to the sample spacing; the global mean where no sample is near.
    cv::Mat val = cv::Mat::zeros(rows, cols, CV_32F), msk = cv::Mat::zeros(rows, cols, CV_32F);
    double mean = 0.0;
    for (int k = 0; k < m; ++k) {
        const int idx = ri_x[(size_t)k] * cols + ri_y[(size_t)k];
        val.ptr<float>(0)[idx] += b[(size_t)k];
        msk.ptr<float>(0)[idx] += 1.0f;
        mean += b[(size_t)k];
    }
    mean = m > 0 ? mean / m : 0.5;
    const float rho = (std::max)(1e-3f, (float)m / (float)n);
    const double sigma = (std::min)(12.0, (std::max)(1.0, 1.5 / std::sqrt((double)rho)));
    cv::Mat num, den;
    cv::GaussianBlur(val, num, cv::Size(0, 0), sigma);
    cv::GaussianBlur(msk, den, cv::Size(0, 0), sigma);
    fill.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        const float dn = den.ptr<float>(0)[i];
        fill[(size_t)i] = dn > 1e-4f ? num.ptr<float>(0)[i] / dn : (float)mean;
    }
}
