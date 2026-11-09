#pragma once
// Shared includes, check() and procedural test images for the backend test
// suite. Each tests/test_*.cpp file covers one topic; test_main.cpp runs them.
#include "crypto_utils.hpp"
#include "image_encryption.hpp"
#include "image_decryption.hpp"
#include "quality_utils.hpp"
#include "cs_gpu.h"

#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
extern int g_failures;

inline void check(bool ok, const char* name) {
    if (ok) {
        std::printf("[PASS] %s\n", name);
    }
    else {
        std::printf("[FAIL] %s\n", name);
        g_failures++;
    }
    std::fflush(stdout); // crash diagnostics: never lose the tail to buffering
}

// procedural test image: smooth gradient + shapes (structure, no noise)
inline cv::Mat make_test_image(int w, int h) {
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

// Production encode / decode as the CLI runs them, without the nets: the single-scale seg8-tile container
// (mode 19, --multiscale off) or the multi-scale one (mode 20), and the cs-sr PnP-ADMM decode (tv 0, 2 ADMM passes).
// sample_bits 0 = the default 6-bit luma. The encoder's anchor pass decodes its container in memory, as the CLI does.
inline int prod_encrypt(const std::string& in, const std::string& out, const std::string& password, float ratio,
    bool multiscale = false, int hr_detail = 2000, int sample_bits = 0, int chroma_bits = 0) {
    const std::function<cv::Mat(const cv::Mat&)> anchor_decode = [&](const cv::Mat& container) {
        CsPhotoUpscalerOptions up;
        up.auto_overlap = 8;
        cv::Mat dec;
        if (decrypt_image::decrypt_image_tiled(container, "", password, false, 0.0f, 12, up, &dec) != 0) dec.release();
        return dec;
    };
    return multiscale
        ? cs_encrypt_multiscale(in, out, password, hr_detail, anchor_decode, ratio)
        : encrypt_image::encrypt_image_tiled(in, out, password, ratio, 8, hr_detail, sample_bits, chroma_bits, false,
              anchor_decode);
}

inline int prod_decrypt(const cv::Mat& container, const std::string& out, const std::string& password) {
    CsPhotoUpscalerOptions up;
    up.sr_pnp_iters = 2;
    return decrypt_image::decrypt_image_tiled(container, out, password, false, 0.0f, 0, up);
}

// The container as the decoder reads it from disk.
inline cv::Mat read_container(const std::string& path) { return cv::imread(path, cv::IMREAD_COLOR); }

// Test entry points, called in order from main() in test_main.cpp.
void test_crypto();
void test_shuffle();
void test_tile_helpers();
void test_roundtrip();
void test_odd_size_roundtrip();
void test_dct_adjoint();
void test_block_dct();
void test_hr_detail();
void test_hr_sr_anchors();
void test_enh_planes();
void test_sample_bits();
void test_seg_auto();
void test_encoder_geometry();
void test_auto_reg_params();
void test_gpu_fista_agreement();
