// Benchmark: learned K-SVD patch dictionary vs the fixed DCT basis across
// compression ratios. Same encrypted container, three decodes:
//   DCT  (baseline solver)
//   DICT with the auto-equivalent l1 coefficient
//   DICT with a smaller l1 coefficient
// The dictionary file must be trained beforehand (train_dictionary tool).
#include "image_encryption.hpp"
#include "image_decryption.hpp"
#include "quality_utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

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

void run_cell(const char* name, const cv::Mat& original, const cv::Mat& encrypted,
    const std::string& password, const cs_dictionary* dict, float coef, int iters)
{
    const std::string tmp_out = ".bench_dict.png";
    const auto t0 = std::chrono::high_resolution_clock::now();
    CSencryption::params = MANUAL_PARAM;
    const int rc = decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, iters, 8, coef, false, false, 0.0f,
        dict ? "cs_dict.dict" : "");
    CSencryption::params = AUTO_PARAM;
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        std::printf("%-24s DECRYPT FAILED (%d)\n", name, rc);
        return;
    }
    cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
    cv::Mat sized;
    cv::resize(dec, sized, original.size());
    const double p = cs_quality::psnr(original, sized);
    const double s = cs_quality::ssim(original, sized);
    const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("%-24s %10.2f %10.4f %10ld\n", name, p, s, ms);
}

void sweep_dict(const char* label, const cv::Mat& original, const std::string& tmp_in, const std::string& password, float ratio) {
    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, ratio, 0, &encrypted) != 0) {
        std::printf("\n%s ratio %.2f: encrypt failed\n", label, ratio);
        return;
    }
    std::printf("\n%s ratio %.2f (5 iterations, overlap 24):\n%-24s %10s %10s %10s\n",
        label, ratio, "mode", "PSNR dB", "SSIM", "time ms");
    char namebuf[64];
    std::snprintf(namebuf, sizeof(namebuf), "DCT (coef %.5f)", coef_for(ratio));
    run_cell(namebuf, original, encrypted, password, nullptr, coef_for(ratio), 8);
    std::snprintf(namebuf, sizeof(namebuf), "K-SVD dict (coef %.5f)", coef_for(ratio));
    run_cell(namebuf, original, encrypted, password, nullptr, coef_for(ratio), 8);
    run_cell("K-SVD dict (coef 0.005)", original, encrypted, password, nullptr, 0.005f, 8);
    std::remove(".bench_dict.png");
}

} // namespace

int main() {
    CSencryption::params = AUTO_PARAM;
    const std::string password = "roundtrip-test-password";

    // the dictionary must exist next to the binary
    cs_dictionary probe;
    if (!cs_load_dictionary("cs_dict.dict", probe)) {
        std::printf("no cs_dict.dict found -- run train_dictionary first\n");
        return 1;
    }
    std::printf("loaded dictionary: %d atoms, patch %d\n", probe.atoms, probe.patch);

    {
        cv::Mat original = make_test_image(960, 720);
        const std::string tmp_in = ".bench_input.png";
        cv::imwrite(tmp_in, original);
        std::printf("synthetic 960x720 (encrypt halved to 480x360, 24 tiles):\n");
        sweep_dict("synthetic", original, tmp_in, password, 1.0f);
        sweep_dict("synthetic", original, tmp_in, password, 0.5f);
        sweep_dict("synthetic", original, tmp_in, password, 0.25f);
        sweep_dict("synthetic", original, tmp_in, password, 0.15f);
        sweep_dict("synthetic", original, tmp_in, password, 0.1f);
        std::remove(tmp_in.c_str());
    }

    {
        std::string path;
        if (const char* env = std::getenv("CS_TEST_PHOTO")) path = env;
        if (path.empty()) path = "IMG_3690.png";
        if (!std::ifstream(path.c_str(), std::ios::binary).good()) {
            path = "ImgReconstruct_backend/IMG_3690.png";
        }
        std::ifstream probe_file(path.c_str(), std::ios::binary);
        if (!probe_file.good()) {
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
        std::printf("\nphoto %dx%d (encrypt halved, 24 tiles):\n", original.cols, original.rows);
        sweep_dict("photo", original, tmp_in, password, 1.0f);
        sweep_dict("photo", original, tmp_in, password, 0.5f);
        sweep_dict("photo", original, tmp_in, password, 0.25f);
        sweep_dict("photo", original, tmp_in, password, 0.15f);
        sweep_dict("photo", original, tmp_in, password, 0.1f);
        std::remove(tmp_in.c_str());
    }

    return 0;
}

