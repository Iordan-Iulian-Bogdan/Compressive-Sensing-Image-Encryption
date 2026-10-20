// GAP-TV solver for point-sampled tiles (prototype, see gap_tv.hpp).
#include "gap_tv.hpp"
#include "fista_kernels.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Chambolle (2004) dual projection for min_v 0.5*||v - x||^2 + lam*TV(v),
// isotropic TV, forward differences with Neumann borders. px/py are the dual
// field (zeros on the last column / row) and persist across calls, which
// warm-starts the next denoise. d is n floats of scratch.
void cs_tv_chambolle(const float* x, float* v, float* px, float* py, float* d,
    int rows, int cols, float lam, int iters)
{
    const int n = rows * cols;
    const float tau = 0.125f;
    const float inv = 1.0f / lam;
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const int t = i * cols + j;
                const float dx = px[t] - (j > 0 ? px[t - 1] : 0.0f);
                const float dy = py[t] - (i > 0 ? py[t - cols] : 0.0f);
                d[t] = dx + dy - x[t] * inv;
            }
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const int t = i * cols + j;
                const float gx = (j + 1 < cols) ? d[t + 1] - d[t] : 0.0f;
                const float gy = (i + 1 < rows) ? d[t + cols] - d[t] : 0.0f;
                const float nrm = 1.0f + tau * std::sqrt(gx * gx + gy * gy);
                px[t] = (px[t] + tau * gx) / nrm;
                py[t] = (py[t] + tau * gy) / nrm;
            }
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const int t = i * cols + j;
            const float dx = px[t] - (j > 0 ? px[t - 1] : 0.0f);
            const float dy = py[t] - (i > 0 ? py[t - cols] : 0.0f);
            v[t] = x[t] - lam * (dx + dy);
        }
    }
    (void)n;
}

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

void cs_gaptv_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float lambda, cv::Mat& ref)
{
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    std::vector<float> b, v, x((size_t)n);
    cs_point_sample_fill(pixel_measurements, channel, rows, cols, ri_x, ri_y, b, v);

    std::vector<float> px((size_t)n, 0.0f), py((size_t)n, 0.0f), d((size_t)n);
    const int inner = 4;
    for (int k = 0; k < iterations; ++k) {
        x = v;
        for (int s = 0; s < m; ++s)
            x[(size_t)(ri_x[(size_t)s] * cols + ri_y[(size_t)s])] = b[(size_t)s];
        cs_tv_chambolle(x.data(), v.data(), px.data(), py.data(), d.data(), rows, cols, lambda, inner);
    }
    // Output keeps the measured pixels exactly.
    for (int s = 0; s < m; ++s)
        v[(size_t)(ri_x[(size_t)s] * cols + ri_y[(size_t)s])] = b[(size_t)s];

    if (!ref.isContinuous()) ref = ref.clone();
    float* out = ref.ptr<float>(0);
    for (int i = 0; i < n; ++i) out[i] = v[(size_t)i] * 255.0f;
}
