// Shared reconstruction-quality helpers: PSNR, SSIM (OpenCV contrib quality
// module), and an amplified error map for visual artifact inspection.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

#if defined(__has_include)
#  if __has_include(<opencv2/quality.hpp>)
#    include <opencv2/quality.hpp>
#    define CS_HAVE_OPENCV_QUALITY 1
#  endif
#endif

namespace cs_quality {

// Mean-squared-error PSNR over all channels; 99 dB when images match.
inline double psnr(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return -1.0;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    const cv::Scalar s = cv::sum(diff);
    const double mse = (s[0] + s[1] + s[2]) / (double)(a.total() * 3);
    if (mse < 1e-9) return 99.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// Global SSIM via cv::quality::QualitySSIM. Returns the mean over channels
// (0 worst .. 1 best), or -1 when the quality module is unavailable or the
// inputs are incompatible.
inline double ssim(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || a.size() != b.size() || a.type() != b.type()) return -1.0;
#ifdef CS_HAVE_OPENCV_QUALITY
    cv::Mat qmap;
    const cv::Scalar q = cv::quality::QualitySSIM::compute(a, b, qmap);
    const double v = (q[0] + q[1] + q[2]) / 3.0;
    if (!std::isfinite(v)) return -1.0;
    return v;
#else
    (void)b;
    return -1.0;
#endif
}

// Mean absolute per-pixel error (0..255 scale).
inline double mean_abs_error(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || a.size() != b.size() || a.type() != b.type()) return -1.0;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    const cv::Scalar s = cv::sum(diff);
    return (s[0] + s[1] + s[2]) / (double)(a.total() * 3);
}

// Maximum absolute per-pixel error across all channels.
inline double max_abs_error(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || a.size() != b.size() || a.type() != b.type()) return -1.0;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    double maxv = 0.0;
    cv::minMaxLoc(diff.reshape(1), nullptr, &maxv);
    return maxv;
}

// Edge-region scores: Sobel gradient magnitude on the ORIGINAL gray plane,
// mask = pixels at/above the 90th percentile (top-10% edges, truth-defined
// so runs stay comparable). Reports PSNR and MAE over masked pixels across
// all channels, plus the actual mask share. Deterministic; -1 on
// incompatible inputs. Built for SR/HR-detail work, where global metrics
// saturate but edge neighborhoods move.
inline void edge_scores(const cv::Mat& original, const cv::Mat& decoded,
    double& edge_psnr, double& edge_mae, double& masked_fraction) {
    edge_psnr = -1.0; edge_mae = -1.0; masked_fraction = 0.0;
    if (original.empty() || original.size() != decoded.size() ||
        original.type() != decoded.type() || original.type() != CV_8UC3)
        return;
    cv::Mat gray, gx, gy, mag;
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    cv::Mat flat = mag.reshape(1, 1);
    std::vector<float> vals;
    flat.copyTo(vals);
    if (vals.empty()) return;
    const size_t k = vals.size() * 9 / 10;
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    const float thr = vals[k];
    cv::Mat mask;
    cv::compare(mag, thr, mask, cv::CMP_GE);
    const double cnt = (double)cv::countNonZero(mask);
    if (cnt < 1.0) return;
    masked_fraction = cnt / (double)original.total();
    cv::Mat diff;
    cv::absdiff(original, decoded, diff);
    diff.convertTo(diff, CV_32F);
    cv::Mat sq = diff.mul(diff);
    cv::Mat sqm, dm;
    sq.copyTo(sqm, mask);
    diff.copyTo(dm, mask);
    const cv::Scalar ss = cv::sum(sqm);
    const cv::Scalar as = cv::sum(dm);
    const double mse = (ss[0] + ss[1] + ss[2]) / (cnt * 3.0);
    edge_mae = (as[0] + as[1] + as[2]) / (cnt * 3.0);
    edge_psnr = (mse < 1e-9) ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / mse);
}

// Amplified false-color error map (JET).
// amplification <= 0 auto-scales so the peak error fills 0..255;
// amplification > 0 multiplies the mean-abs-difference grayscale by that factor
// before clipping (fixed gain, comparable across runs).
inline cv::Mat error_map(const cv::Mat& a, const cv::Mat& b, double amplification = 0.0) {
    if (a.empty() || a.size() != b.size() || a.type() != b.type()) return cv::Mat();
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    cv::Mat gray;
    if (diff.channels() == 3) {
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = diff;
    }
    // absdiff of 8U is already |a-b|; after CV_32F conversion values are 0..255
    double scale = 1.0;
    if (amplification > 0.0) {
        scale = amplification;
    }
    else {
        double maxv = 0.0;
        cv::minMaxLoc(gray, nullptr, &maxv);
        scale = maxv > 1e-9 ? 255.0 / maxv : 1.0;
    }
    cv::Mat scaled;
    gray.convertTo(scaled, CV_8U, scale);
    cv::Mat colored;
    cv::applyColorMap(scaled, colored, cv::COLORMAP_JET);
    return colored;
}

// Horizontal strip: original | decoded | JET error map (all resized to match).
inline cv::Mat side_by_side(const cv::Mat& original, const cv::Mat& decoded, double amplification = 0.0) {
    if (original.empty() || decoded.empty()) return cv::Mat();
    cv::Mat dec = decoded;
    if (dec.size() != original.size() || dec.type() != original.type()) {
        cv::resize(decoded, dec, original.size());
    }
    cv::Mat err = error_map(original, dec, amplification);
    cv::Mat strip;
    cv::hconcat(original, dec, strip);
    cv::hconcat(strip, err, strip);
    return strip;
}

} // namespace cs_quality
