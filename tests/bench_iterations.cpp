// Benchmark: how do L-BFGS iteration count and compression ratio affect
// reconstruction quality (PSNR vs the original) and wall time?
//
// For each compression ratio the container is encrypted once (fixed sample
// positions), then decrypted repeatedly with different iteration counts in
// manual parameter mode (AUTO_PARAM overrides iterations/coef, so the sweep
// must use MANUAL_PARAM to control the solver).
#include "image_encryption.hpp"
#include "image_decryption.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

double psnr(const cv::Mat& a, const cv::Mat& b) {
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

cv::Mat make_test_image(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)(128 + 60 * std::sin(x * 0.05)),
                (uchar)(80 + x * 80 / w),
                (uchar)(60 + y * 100 / h));
        }
    }
    cv::rectangle(img, cv::Rect(w / 4, h / 4, w / 2, h / 2), cv::Scalar(40, 30, 220), cv::FILLED);
    cv::circle(img, cv::Point(w / 2, h / 2), (w < h ? w : h) / 6, cv::Scalar(200, 220, 60), cv::FILLED);
    return img;
}

// AUTO-equivalent solver coefficient for a compression ratio
float coef_for(float ratio) {
    return ratio < 0.5f ? 0.045f : (1.0f / ratio) * 0.01875f;
}

void sweep_ratio(const cv::Mat& original, const std::string& tmp_in, const std::string& password) {
    const std::string tmp_out = ".bench_output.png";
    const float ratios[] = { 1.0f, 0.75f, 0.5f, 0.25f };
    const int its[] = { 1, 2, 3, 5, 8, 10, 20, 30 };

    for (float ratio : ratios) {
        cv::Mat encrypted;
        if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, ratio, &encrypted) != 0) {
            std::printf("ratio %.2f: encrypt failed\n", ratio);
            continue;
        }
        const float coef = coef_for(ratio);
        std::printf("\nratio %.2f (coef %.5f):\n%-12s %10s %10s\n", ratio, coef, "iterations", "PSNR dB", "time ms");
        for (int it : its) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            CSencryption::params = MANUAL_PARAM;
            const int rc = decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, it, 8, coef, false);
            CSencryption::params = AUTO_PARAM;
            const auto t1 = std::chrono::high_resolution_clock::now();
            if (rc != 0) {
                std::printf("%-12d DECRYPT FAILED (%d)\n", it, rc);
                continue;
            }
            cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = psnr(original, sized);
            const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::printf("%-12d %10.2f %10ld\n", it, p, ms);
        }
    }
    std::remove(tmp_out.c_str());
    std::remove(".bench_output.png");
}

} // namespace

int main() {
    CSencryption::params = AUTO_PARAM;
    const std::string password = "roundtrip-test-password";

    // --- synthetic -------------------------------------------------------
    {
        cv::Mat original = make_test_image(960, 720);
        const std::string tmp_in = ".bench_input.png";
        cv::imwrite(tmp_in, original);
        std::printf("synthetic 960x720 (encrypt halved to 480x360, 24 tiles):\n");
        sweep_ratio(original, tmp_in, password);
        std::remove(tmp_in.c_str());
    }

    // --- real photo ------------------------------------------------------
    {
        std::string path;
        if (const char* env = std::getenv("CS_TEST_PHOTO")) path = env;
        if (path.empty()) path = "IMG_3690.png";
        if (!std::ifstream(path.c_str(), std::ios::binary).good()) {
            path = "ImgReconstruct_backend/IMG_3690.png";
        }
        std::ifstream probe(path.c_str(), std::ios::binary);
        if (!probe.good()) {
            std::printf("\nphoto sweep skipped: no photo found (set CS_TEST_PHOTO=<path>)\n");
            return 0;
        }
        cv::Mat original = cv::imread(path, cv::IMREAD_COLOR);
        if (original.empty()) return 0;
        if (original.cols > 2400) {
            cv::resize(original, original, cv::Size(2400, original.rows * 2400 / original.cols));
        }
        const std::string tmp_in = ".bench_photo.png";
        cv::imwrite(tmp_in, original);
        std::printf("\nphoto %dx%d (encrypt halved):\n", original.cols, original.rows);
        sweep_ratio(original, tmp_in, password);
        std::remove(tmp_in.c_str());
    }

    return 0;
}
