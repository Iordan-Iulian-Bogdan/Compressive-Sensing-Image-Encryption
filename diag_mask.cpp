// visualize adaptive sampling: binary mask + per-tile density heat map
#include "CS_encryption.hpp"
#include "crypto_utils.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// expose protected returnAdaptiveIndices
struct IdxGen : CSencryption {
    using CSencryption::returnAdaptiveIndices;
};

static cv::Mat make_test_image(int w, int h) {
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

static std::vector<uint8_t> compute_lod(const cv::Mat& img, int tile_size, int lod_min) {
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;
    cv::Mat lap, abs_lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::absdiff(lap, cv::Scalar(0), abs_lap);

    const int rows = img.rows, cols = img.cols;
    const int tiles_cols = (cols + tile_size - 1) / tile_size;
    const int tiles_rows = (rows + tile_size - 1) / tile_size;
    std::vector<uint8_t> lod((size_t)tiles_rows * tiles_cols, (uint8_t)lod_min);
    std::vector<float> energy(lod.size(), 0.f);
    float e_min = 1e30f, e_max = -1e30f;
    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            const int row0 = tr * tile_size, col0 = tc * tile_size;
            const int tw = (tile_size < cols - col0) ? tile_size : cols - col0;
            const int th = (tile_size < rows - row0) ? tile_size : rows - row0;
            const float e = (float)cv::mean(abs_lap(cv::Rect(col0, row0, tw, th)))[0];
            const size_t i = (size_t)tr * tiles_cols + tc;
            energy[i] = e;
            e_min = (std::min)(e_min, e);
            e_max = (std::max)(e_max, e);
        }
    }
    const float range = e_max - e_min;
    for (size_t i = 0; i < energy.size(); ++i) {
        float n = range > 1e-6f ? (energy[i] - e_min) / range : 0.f;
        int v = lod_min + (int)std::lround((double)n * (255 - lod_min));
        lod[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return lod;
}

static void derive_key(const std::string& password, uint8_t key[32], uint8_t salt[CS_SALT_BYTES]) {
    // deterministic salt so mask is reproducible for a given password
    std::memset(salt, 0xA5, CS_SALT_BYTES);
    salt[0] = (uint8_t)password.size();
    if (!cs_derive_key(password, salt, key)) {
        std::fprintf(stderr, "key derivation failed\n");
        std::exit(1);
    }
}

static void label(cv::Mat& panel, const std::string& t, cv::Point org = cv::Point(8, 22)) {
    cv::putText(panel, t, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(panel, t, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// Contact sheet: source + LOD columns, then one column per strength
// (mask / cyan overlay / density). strength uses the same mapping as the CLI.
static void visualize_strengths(const std::string& tag, const cv::Mat& img,
    int tile_size, float ratio, int lod_min, const std::vector<double>& strengths)
{
    const int rows = img.rows, cols = img.cols;
    const std::vector<uint8_t> lod = compute_lod(img, tile_size, lod_min);
    const int m = (int)((long long)rows * cols * ratio);

    uint8_t key[32], salt[CS_SALT_BYTES];
    derive_key("mask-visual-password", key, salt);

    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;
    cv::Mat gray3;
    cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);

    const int tiles_cols = (cols + tile_size - 1) / tile_size;
    const int tiles_rows = (rows + tile_size - 1) / tile_size;
    auto draw_grid = [&](cv::Mat& panel) {
        for (int x = 0; x < cols; x += tile_size)
            cv::line(panel, cv::Point(x, 0), cv::Point(x, rows - 1), cv::Scalar(30, 30, 30), 1);
        for (int y = 0; y < rows; y += tile_size)
            cv::line(panel, cv::Point(0, y), cv::Point(cols - 1, y), cv::Scalar(30, 30, 30), 1);
    };

    cv::Mat lod_m(tiles_rows, tiles_cols, CV_8UC1);
    for (size_t i = 0; i < lod.size(); i++)
        lod_m.at<uchar>((int)i / tiles_cols, (int)i % tiles_cols) = lod[i];
    cv::Mat lod_color;
    cv::applyColorMap(lod_m, lod_color, cv::COLORMAP_VIRIDIS);
    cv::resize(lod_color, lod_color, img.size(), 0, 0, cv::INTER_NEAREST);
    draw_grid(lod_color);
    label(lod_color, "LOD (viridis)");

    std::vector<cv::Mat> row0{ gray3, lod_color };
    std::vector<cv::Mat> row1{
        cv::Mat(img.size(), CV_8UC3, cv::Scalar(20, 20, 20)),
        cv::Mat(img.size(), CV_8UC3, cv::Scalar(20, 20, 20))
    };
    std::vector<cv::Mat> row2{
        cv::Mat(img.size(), CV_8UC3, cv::Scalar(20, 20, 20)),
        cv::Mat(img.size(), CV_8UC3, cv::Scalar(20, 20, 20))
    };
    label(row1[0], "overlay");
    label(row2[0], "density");
    label(row0[0], "source");

    std::printf("%s: %dx%d tile=%d ratio=%.2f m=%d strengths=%zu\n",
        tag.c_str(), cols, rows, tile_size, ratio, m, strengths.size());

    for (double s : strengths) {
        const int wbase = cs_adaptive_base_from_strength(s);
        std::vector<int> ri_x, ri_y;
        IdxGen gen;
        gen.returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, key, tile_size, wbase);

        cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
        for (size_t k = 0; k < ri_x.size(); ++k)
            mask.at<uchar>(ri_x[k], ri_y[k]) = 255;
        cv::Mat mask3;
        cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);

        cv::Mat ov = gray3.clone();
        ov.setTo(cv::Scalar(60, 60, 60));
        cv::Mat tint(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
        tint.setTo(cv::Scalar(255, 180, 0), mask);
        cv::addWeighted(ov, 0.55, gray3, 0.45, 0, ov);
        (cv::max)(ov, tint, ov);

        cv::Mat dens(tiles_rows, tiles_cols, CV_32FC1, cv::Scalar(0));
        std::vector<int> cnt((size_t)tiles_rows * tiles_cols, 0);
        for (size_t k = 0; k < ri_x.size(); ++k) {
            const int tr = ri_x[k] / tile_size;
            const int tc = ri_y[k] / tile_size;
            if (tr < tiles_rows && tc < tiles_cols) cnt[(size_t)tr * tiles_cols + tc]++;
        }
        double dmin = 1e9, dmax = 0, dsum = 0;
        for (int tr = 0; tr < tiles_rows; ++tr)
            for (int tc = 0; tc < tiles_cols; ++tc) {
                const int row0y = tr * tile_size, col0 = tc * tile_size;
                const int tw = (tile_size < cols - col0) ? tile_size : cols - col0;
                const int th = (tile_size < rows - row0y) ? tile_size : rows - row0y;
                const float d = (float)cnt[(size_t)tr * tiles_cols + tc] / (float)(tw * th);
                dens.at<float>(tr, tc) = d;
                dmin = (std::min)(dmin, (double)d);
                dmax = (std::max)(dmax, (double)d);
                dsum += d;
            }
        cv::Mat du8, dc;
        dens.convertTo(du8, CV_8U, 255.0);
        cv::applyColorMap(du8, dc, cv::COLORMAP_JET);
        cv::resize(dc, dc, img.size(), 0, 0, cv::INTER_NEAREST);
        draw_grid(dc);

        char t[112];
        std::snprintf(t, sizeof(t), "s=%.2f base=%d draw=%zu d=%.2f/%.2f/%.2f",
            s, wbase, ri_x.size(), dmin, dmax, dsum / (tiles_rows * tiles_cols));
        label(mask3, t);
        label(ov, t);
        label(dc, t);
        std::printf("  strength %.2f  base=%d  drawn=%zu  density min/max/mean = %.2f/%.2f/%.2f\n",
            s, wbase, ri_x.size(), dmin, dmax, dsum / (tiles_rows * tiles_cols));

        row0.push_back(mask3);
        row1.push_back(ov);
        row2.push_back(dc);
    }

    cv::Mat r0, r1, r2, sheet;
    cv::hconcat(row0, r0);
    cv::hconcat(row1, r1);
    cv::hconcat(row2, r2);
    cv::vconcat(std::vector<cv::Mat>{r0, r1, r2}, sheet);

    const std::string out = ".mask_" + tag + ".png";
    cv::imwrite(out, sheet);
    std::printf("  -> %s (%dx%d)\n\n", out.c_str(), sheet.cols, sheet.rows);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cv::ocl::setUseOpenCL(false);

    // strength ladder: near-uniform .. tuned .. aggressive
    const std::vector<double> strengths = { 0.0, 0.25, 0.5, 0.75, 1.0 };

    // synthetic: mask column layout is large; keep tile small enough to show pattern
    {
        cv::Mat img = make_test_image(360, 270);
        visualize_strengths("synthetic_t64_r50_strengths", img, 64, 0.5f, 32, strengths);
    }

    // photo half-res, t64 and t128
    {
        const std::string path = "ImgReconstruct_backend/IMG_3690.png";
        cv::Mat full = cv::imread(path, cv::IMREAD_COLOR);
        if (full.empty()) {
            std::printf("photo missing: %s\n", path.c_str());
            return 0;
        }
        cv::Mat half;
        cv::resize(full, half, cv::Size(full.cols / 2, full.rows / 2));
        visualize_strengths("photo_t64_r50_strengths", half, 64, 0.5f, 32, strengths);
        visualize_strengths("photo_t128_r50_strengths", half, 128, 0.5f, 32, strengths);
    }
    (void)argc; (void)argv;
    return 0;
}
