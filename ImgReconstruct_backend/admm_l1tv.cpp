// Consensus ADMM with DCT-L1 and TV priors (prototype, see admm_l1tv.hpp).
#include "admm_l1tv.hpp"
#include "gap_tv.hpp"
#include "helper_functions.hpp"
#include "cs_wavelet.hpp"

#include <algorithm>
#include <cmath>

// Variables: x (data-consistent image), a (L1 branch), b (TV branch), with
// scaled duals ua, ub and penalty rho = 1, so the L1 threshold is lambda_l1 and
// the TV weight is lambda_tv. Per iteration:
//   a = prox_L1(x + ua)   b = prox_TV(x + ub)
//   x = pin_samples(((a - ua) + (b - ub)) / 2)
//   ua += x - a           ub += x - b
void cs_admm_l1tv_channel(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int iterations, float lambda_l1, float lambda_tv, cv::Mat& ref)
{
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    std::vector<float> samples, x;
    cs_point_sample_fill(pixel_measurements, channel, rows, cols, ri_x, ri_y, samples, x);

    std::vector<float> a((size_t)n), b((size_t)n), ua((size_t)n, 0.0f), ub((size_t)n, 0.0f),
        t((size_t)n), px((size_t)n, 0.0f), py((size_t)n, 0.0f), d((size_t)n);
    const int dct_block = cs_default_dct_block();
    const int tv_inner = 4;
    for (int k = 0; k < iterations; ++k) {
        for (int i = 0; i < n; ++i) t[(size_t)i] = x[(size_t)i] + ua[(size_t)i];
        {
            cv::Mat T(rows, cols, CV_32F, t.data());
            cs_dct_forward(T, dct_block);
            // Keep the DC coefficient: shrinking it would pull the mean toward 0.
            const float dc = t[0];
            for (int i = 0; i < n; ++i) {
                const float v = t[(size_t)i], av = std::fabs(v);
                t[(size_t)i] = av > lambda_l1 ? (v > 0.0f ? av - lambda_l1 : lambda_l1 - av) : 0.0f;
            }
            t[0] = dc;
            cs_dct_inverse(T, dct_block);
        }
        a = t;
        for (int i = 0; i < n; ++i) t[(size_t)i] = x[(size_t)i] + ub[(size_t)i];
        cs_tv_chambolle(t.data(), b.data(), px.data(), py.data(), d.data(), rows, cols,
            lambda_tv, tv_inner);
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
