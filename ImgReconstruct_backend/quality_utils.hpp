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
