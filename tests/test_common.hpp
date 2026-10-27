#pragma once
// Shared includes, check() and procedural test images for the backend test
// suite. Each tests/test_*.cpp file covers one topic; test_main.cpp runs them.
#include "crypto_utils.hpp"
#include "image_encryption.hpp"
#include "image_decryption.hpp"
#include "cs_dict.hpp"
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

// Test entry points, called in order from main() in test_main.cpp.
void test_crypto();
void test_shuffle();
void test_tile_helpers();
void test_roundtrip();
void test_fista_solvers();
void test_dct_adjoint();
void test_block_dct();
void test_wavelet_basis();
void test_adaptive_counts();
void test_sample_split();
void test_grain_tv();
void test_adaptive_roundtrip();
void test_spectral_lod();
void test_lod_full_roundtrip();
void test_ycc420_roundtrip();
void test_ycc422_roundtrip();
void test_twopass_roundtrip();
void test_tsampling();
void test_regions_roundtrip();
void test_photo_roundtrip();
void test_photo_upscaler();
void test_cs_super_resolution();
void test_sr_direct();
void test_coupled_dict();
void test_dncnn_red();
void test_onnx_denoisers();
void test_onsager();
void test_hr_detail();
void test_hr_sr_anchors();
void test_enh_planes();
void test_edge_scores();
void test_adaptive_gamma();
void test_adaptive_cutoff();
void test_anchor_weighted_red();
void test_hf_focus_roundtrip();
void test_hf_weighted_spread();
void test_adaptive_tile_grid_limit();
void test_sample_bits();
void test_y_confidence_prior();
void test_y_dual_tv_blend();
void test_y_dual_tv_color_segment();
void test_dual_tv_split();
void test_seg_auto();
void test_seg_edge();
void test_encoder_geometry();
void test_auto_reg_params();
void test_per_tile_coef();
void test_gpu_fista_agreement();
