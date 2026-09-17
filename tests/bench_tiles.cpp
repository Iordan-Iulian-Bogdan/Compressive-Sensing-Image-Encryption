// Benchmark: how does the tile grid count (N x N tiles) affect reconstruction
// quality (PSNR) and wall time? Fewer tiles = larger tiles = more context per
// solve (and less parallelism within wavefront waves); more tiles = cheaper
// solves but less information per solve.
//
// num_tiles is swept in manual parameter mode (AUTO_PARAM forces 24) with
// iterations at the measured saturation (5) and the AUTO-equivalent
// coefficient, at two compression ratios (1.0 and 0.25).
//
// Photo source (optional): $CS_TEST_PHOTO, IMG_3690.png, or the project
// sample; skipped if none is available.
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

float coef_for(float ratio) {
    return ratio < 0.5f ? 0.045f : (1.0f / ratio) * 0.01875f;
}

void sweep_tiles(const char* label, const cv::Mat& original, const std::string& tmp_in, const std::string& password, float ratio) {
    const std::string tmp_out = ".bench_tiles.png";
    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, ratio, &encrypted) != 0) {
        std::printf("\n%s ratio %.2f: encrypt failed\n", label, ratio);
        return;
    }
    const float coef = coef_for(ratio);
    std::printf("\n%s ratio %.2f (coef %.5f, 5 iterations, overlap 24):\n%-10s %10s %10s\n",
        label, ratio, coef, "tiles", "PSNR dB", "time ms");
    const int tiles_list[] = { 8, 12, 16, 20, 24, 32, 48, 64, 96, 128 };
    for (int n : tiles_list) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        CSencryption::params = MANUAL_PARAM;
        const int rc = decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, n, 24, 5, 8, coef, false);
        CSencryption::params = AUTO_PARAM;
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (rc != 0) {
            std::printf("%-10d DECRYPT FAILED (%d)\n", n, rc);
            continue;
        }
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        cv::Mat sized;
        cv::resize(dec, sized, original.size());
        const double p = psnr(original, sized);
        const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::printf("%-10d %10.2f %10ld\n", n, p, ms);
    }
    std::remove(tmp_out.c_str());
}

} // namespace

int main() {
    CSencryption::params = AUTO_PARAM;
    const std::string password = "roundtrip-test-password";

    {
        cv::Mat original = make_test_image(960, 720);
        const std::string tmp_in = ".bench_input.png";
        cv::imwrite(tmp_in, original);
        std::printf("synthetic 960x720 (encrypt halved to 480x360):\n");
        sweep_tiles("synthetic", original, tmp_in, password, 1.0f);
        sweep_tiles("synthetic", original, tmp_in, password, 0.25f);
        std::remove(tmp_in.c_str());
    }

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
        sweep_tiles("photo", original, tmp_in, password, 1.0f);
        sweep_tiles("photo", original, tmp_in, password, 0.25f);
        std::remove(tmp_in.c_str());
    }

    return 0;
}
