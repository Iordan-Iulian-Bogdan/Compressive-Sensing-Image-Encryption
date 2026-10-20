// Plug-and-play NLM prior inside the main solve (prototype, see pnp_nlm.hpp).
#include "pnp_nlm.hpp"
#include "gap_tv.hpp"
#include "helper_functions.hpp"
#include "cs_wavelet.hpp"

#include <opencv2/photo.hpp>
#include <algorithm>
#include <cmath>

// Variables: x (sample-consistent image), a (DCT-L1 branch), b (denoiser
// branch), scaled duals ua/ub, penalty 1. Per iteration:
//   a = softDCT(x + ua)   b = NLM(x + ub)
//   x = pin_samples(((a - ua) + (b - ub)) / 2)
//   ua += x - a           ub += x - b
// With lambda_l1 == 0 the a-branch is the identity and this reduces to a
// damped plug-and-play iteration.
void cs_pnp_nlm_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float h, float lambda_l1, cv::Mat& ref)
{
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    std::vector<float> samples, x;
    cs_point_sample_fill(pixel_measurements, channel, rows, cols, ri_x, ri_y, samples, x);

    std::vector<float> a((size_t)n), b((size_t)n), ua((size_t)n, 0.0f), ub((size_t)n, 0.0f),
        t((size_t)n);
    cv::Mat u8(rows, cols, CV_8U), dn;
    const int dct_block = cs_default_dct_block();
    for (int k = 0; k < iterations; ++k) {
        for (int i = 0; i < n; ++i) t[(size_t)i] = x[(size_t)i] + ua[(size_t)i];
        if (lambda_l1 > 0.0f) {
            cv::Mat T(rows, cols, CV_32F, t.data());
            cs_dct_forward(T, dct_block);
            const float dc = t[0]; // keep the mean
            for (int i = 0; i < n; ++i) {
                const float v = t[(size_t)i], av = std::fabs(v);
                t[(size_t)i] = av > lambda_l1 ? (v > 0.0f ? av - lambda_l1 : lambda_l1 - av) : 0.0f;
            }
            t[0] = dc;
            cs_dct_inverse(T, dct_block);
        }
        a = t;
        uchar* up = u8.ptr<uchar>(0);
        for (int i = 0; i < n; ++i)
            up[i] = cv::saturate_cast<uchar>((x[(size_t)i] + ub[(size_t)i]) * 255.0f);
        cv::fastNlMeansDenoising(u8, dn, h, 7, 21);
        const uchar* dp = dn.ptr<uchar>(0);
        for (int i = 0; i < n; ++i) b[(size_t)i] = dp[i] * (1.0f / 255.0f);
        for (int i = 0; i < n; ++i)
            x[(size_t)i] = 0.5f * ((a[(size_t)i] - ua[(size_t)i]) + (b[(size_t)i] - ub[(size_t)i]));
        for (int s = 0; s < m; ++s)
            x[(size_t)(ri_x[(size_t)s] * cols + ri_y[(size_t)s])] = samples[(size_t)s];
        for (int i = 0; i < n; ++i) {
            ua[(size_t)i] += x[(size_t)i] - a[(size_t)i];
            ub[(size_t)i] += x[(size_t)i] - b[(size_t)i];
        }
    }

    if (!ref.isContinuous()) ref = ref.clone();
    float* out = ref.ptr<float>(0);
    for (int i = 0; i < n; ++i) out[i] = x[(size_t)i] * 255.0f;
}
