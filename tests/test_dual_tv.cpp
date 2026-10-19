// Backend tests: luma confidence, dual-TV, grain TV, Onsager, anchor-weighted RED.
#include "test_common.hpp"

void test_grain_tv() {
    cv::Mat clean = make_test_image(128, 128);
    cv::Mat noisy;
    {
        cv::Mat nse;
        clean.convertTo(nse, CV_32F);
        cv::Mat gauss(nse.size(), CV_32FC3);
        cv::randn(gauss, 0.0, 25.0);
        cv::Mat noisyf = nse + gauss;
        noisyf.convertTo(noisy, CV_8U);
    }
    const double sig_clean = cs_estimate_noise_sigma(clean);
    const double sig_noisy = cs_estimate_noise_sigma(noisy);
    std::printf("       grain-tv sigma clean %.2f, noisy %.2f\n", sig_clean, sig_noisy);
    check(std::isfinite(sig_clean) && std::isfinite(sig_noisy),
        "grain-tv: estimator finite");
    check(sig_clean > 0.0 && sig_clean < 30.0,
        "grain-tv: clean content reads sane");
    check(sig_noisy > 2.0 * sig_clean,
        "grain-tv: noise reads well above content");
    check(cs_estimate_noise_sigma(cv::Mat()) == 0.0,
        "grain-tv: empty input rejected");
    {
        cv::Mat gray;
        cv::cvtColor(clean, gray, cv::COLOR_BGR2GRAY);
        check(cs_estimate_noise_sigma(gray) == 0.0,
            "grain-tv: non-BGR input rejected");
        check(cs_estimate_noise_sigma(cv::Mat(2, 2, CV_8UC3,
            cv::Scalar(9, 9, 9))) == 0.0,
            "grain-tv: tiny input rejected");
    }
    // ROF mapping used by the worker (lambda = excess over reference):
    // noisy improves clearly, clean barely moves, runs are deterministic.
    const double ref = 8.0;
    check(cs_grain_tv_lambda(sig_noisy, (float)ref) > 0.0f &&
        cs_grain_tv_lambda(sig_clean, (float)ref) == 0.0f &&
        cs_grain_tv_lambda(sig_noisy, 0.0f) == 0.0f,
        "grain-tv: strength mapping fires only above reference");
    const double p_noisy = cs_quality::psnr(clean, noisy);
    cv::Mat dn = noisy.clone();
    cs_tv_denoise_tile(dn, cs_grain_tv_lambda(sig_noisy, (float)ref));
    const double p_dn = cs_quality::psnr(clean, dn);
    std::printf("       grain-tv noisy %.2f dB -> denoised %.2f dB\n", p_noisy, p_dn);
    check(p_dn > p_noisy + 1.0,
        "grain-tv: excess-mapped ROF improves noisy input >= 1 dB");
    cv::Mat dn2 = noisy.clone();
    cs_tv_denoise_tile(dn2, cs_grain_tv_lambda(sig_noisy, (float)ref));
    {
        // absdiff-based (cv::norm INF flaked on identical mats in this
        // static OpenCV build; the transparent reduction never does).
        cv::Mat d;
        cv::absdiff(dn, dn2, d);
        double mn, mx;
        cv::minMaxLoc(d.reshape(1), &mn, &mx);
        check(mx == 0.0, "grain-tv: ROF deterministic");
    }
    {
        cv::Mat dc = clean.clone();
        cs_tv_denoise_tile(dc, 2.0f);
        check(cs_quality::psnr(clean, dc) > 35.0,
            "grain-tv: small lambda barely moves clean input");
        cv::Mat empty;
        cs_tv_denoise_tile(empty, 5.0f);
        check(empty.empty(), "grain-tv: empty tile no-op");
        cv::Mat gray;
        cv::cvtColor(clean, gray, cv::COLOR_BGR2GRAY);
        cs_tv_denoise_tile(gray, 5.0f);
        check(gray.type() == CV_8UC1, "grain-tv: non-BGR tile untouched");
        cv::Mat neg = noisy.clone();
        cs_tv_denoise_tile(neg, 0.0f);
        check(cv::norm(neg, noisy, cv::NORM_INF) == 0,
            "grain-tv: zero lambda no-op");
    }
}

void test_onsager() {
    // 50%-sampled 40x40 fixture through one channel solve, correction
    // on vs off: both finite/sane/deterministic, and the correction
    // engages (outputs differ) on content with partial support.
    cv::Mat truth_bgr = make_test_image(80, 80);
    cv::Mat truth_gray;
    cv::cvtColor(truth_bgr, truth_gray, cv::COLOR_BGR2GRAY);
    cv::Mat truth_lr;
    cv::resize(truth_gray, truth_lr, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
    cv::Mat lr;
    cv::resize(truth_bgr, lr, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
    std::mt19937 rng(99);
    std::vector<int> rx, ry;
    cv::Mat lr8;
    lr.convertTo(lr8, CV_8U);
    for (int r = 0; r < 40; ++r)
        for (int c = 0; c < 40; ++c)
            if ((rng() % 100) < 50) { rx.push_back(r); ry.push_back(c); }
    const int m = (int)rx.size();
    check(m > 400, "onsager: fixture holds enough samples");
    cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int k = 0; k < m; ++k)
        packed.at<cv::Vec3b>(CS_HEADER_PIXELS + k) = lr8.at<cv::Vec3b>(rx[k], ry[k]);
    auto solve_once = [&](bool oc) {
        cv::Mat ref = cv::Mat::zeros(40, 40, CV_32F);
        cv::dct(ref, ref, 0);
        ref /= 10.0f;
        cv::Mat dummy;
        reconstruct_color_channel_fista(packed, 0, 0.0375f, 40, 40, rx, ry,
            2, ref, false, dummy, 0.0f, 1, 8, CS_BASIS_DCT, 2.0f, oc);
        cv::Mat out;
        ref.convertTo(out, CV_8U);
        return out;
    };
    cv::Mat p8 = solve_once(false), c8 = solve_once(true);
    // finite pixels (8U saturate rules out inf/nan; guard anyway)
    bool fin = true;
    {
        cv::Mat pf, cf;
        p8.convertTo(pf, CV_32F);
        c8.convertTo(cf, CV_32F);
        for (int r = 0; r < 40 && fin; ++r)
            for (int c = 0; c < 40; ++c)
                if (!std::isfinite(pf.at<float>(r, c)) ||
                    !std::isfinite(cf.at<float>(r, c))) { fin = false; break; }
    }
    check(fin, "onsager: both paths finite");
    const double p_plain = cs_quality::psnr(truth_lr, p8);
    const double p_corr = cs_quality::psnr(truth_lr, c8);
    std::printf("       onsager off %.2f dB vs on %.2f dB\n", p_plain, p_corr);
    check(std::isfinite(p_plain) && std::isfinite(p_corr) &&
        p_plain > 8.0 && p_corr > 8.0, "onsager: both paths sane");
    {
        cv::Mat d;
        cv::absdiff(p8, c8, d);
        double mn, mx;
        cv::minMaxLoc(d.reshape(1), &mn, &mx);
        check(mx > 0.0, "onsager: correction engages (outputs differ)");
    }
    cv::Mat corr2 = solve_once(true);
    {
        cv::Mat d;
        cv::absdiff(c8, corr2, d);
        double mn, mx;
        cv::minMaxLoc(d.reshape(1), &mn, &mx);
        check(mx == 0.0, "onsager: deterministic");
    }
}

void test_anchor_weighted_red() {
    // Map unit pins: exact distance falloff on a 16x16 tile.
    {
        const std::vector<int> idx = { 5 * 16 + 5 };
        const cv::Mat map = cs_hr_red_map(16, 16, idx, 4.0f);
        check(!map.empty() && map.type() == CV_32F && map.size() == cv::Size(16, 16),
            "hr-red: map geometry and type");
        bool center_zero = false, far_full = false, monotone = true;
        float prev = -1.0f;
        if (!map.empty()) {
            center_zero = (map.at<float>(5, 5) == 0.0f);
            far_full = (map.at<float>(15, 15) > 0.99f);
            for (int x = 5; x < 16; ++x) {
                const float v = map.at<float>(5, x);
                if (v < prev || v < 0.0f || v > 1.0f) { monotone = false; break; }
                prev = v;
            }
        }
        check(center_zero, "hr-red: zero factor at anchor pixels");
        check(far_full, "hr-red: full factor far from anchors");
        check(monotone, "hr-red: factor rises monotonically with distance");
        check(cs_hr_red_map(16, 16, {}, 4.0f).empty(),
            "hr-red: no anchors means uniform (empty map)");
        check(cs_hr_red_map(16, 16, idx, 0.0f).empty(),
            "hr-red: zero radius means uniform (empty map)");
        check(cs_hr_red_map(16, 16, { 99999 }, 4.0f).empty(),
            "hr-red: out-of-tile anchors mean uniform (empty map)");
        check(cs_hr_red_map(0, 16, idx, 4.0f).empty(),
            "hr-red: degenerate geometry means uniform (empty map)");
    }
    // Solve level on the synthetic rig: modulated RED differs from uniform
    // RED, while no-anchors/radius-0 stay bit-identical to uniform.
    {
        const cv::Mat original = make_test_image(80, 80);
        cv::Mat measured, solved;
        cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
        cv::add(measured, cv::Scalar(8, 8, 8), solved);
        const int m = measured.rows * measured.cols;
        cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<int> rx, ry;
        rx.reserve(m); ry.reserve(m);
        for (int row = 0; row < measured.rows; ++row)
            for (int col = 0; col < measured.cols; ++col) {
                rx.push_back(row); ry.push_back(col);
                packed.at<cv::Vec3b>(CS_HEADER_PIXELS + row * measured.cols + col) =
                    measured.at<cv::Vec3b>(row, col);
            }
        cv::Mat orig_gray;
        cv::cvtColor(original, orig_gray, cv::COLOR_BGR2GRAY);
        const int pts[8][2] = {{10,10},{30,12},{55,20},{70,33},{15,45},{40,60},{62,70},{25,25}};
        std::vector<CsHrSample> anchors;
        for (int k = 0; k < 8; ++k) {
            CsHrSample s;
            s.idx = (uint32_t)(pts[k][1] * 80 + pts[k][0]);
            s.y = orig_gray.at<uint8_t>(pts[k][1], pts[k][0]);
            anchors.push_back(s);
        }
        cv::Mat uniform, modulated;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, uniform, 0.5f);
        CsHrSrCtx ctx;
        ctx.anchors = &anchors;
        ctx.stride = 80;
        ctx.weight = 0.0f; // isolate RED modulation from the data term
        ctx.red_radius = 8.0f;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, modulated, 0.5f, "nlmeans",
            CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, ctx);
        check(modulated.size() == original.size() && modulated.type() == CV_8UC3,
            "hr-red: modulated solve keeps 2x geometry and color type");
        if (!modulated.empty() && !uniform.empty()) {
            check(cv::norm(modulated, uniform, cv::NORM_INF) != 0,
                "hr-red: modulation changes the RED solve");
        }
        CsHrSrCtx noanchors;
        noanchors.red_radius = 8.0f;
        cv::Mat norad;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, norad, 0.5f, "nlmeans",
            CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, noanchors);
        check(cv::norm(norad, uniform, cv::NORM_INF) == 0,
            "hr-red: no anchors stays bit-identical to uniform RED");
        CsHrSrCtx zerorad = ctx;
        zerorad.red_radius = 0.0f;
        cv::Mat zrad;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, zrad, 0.5f, "nlmeans",
            CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, zerorad);
        check(cv::norm(zrad, uniform, cv::NORM_INF) == 0,
            "hr-red: zero radius stays bit-identical to uniform RED");
    }
}

// ---------------------------------------------------------------------------
// Per-tile coef (--per-tile-coef): scaling math + an adaptive end-to-end
// where tile counts differ (uniform containers would scale by exactly 1).
// ---------------------------------------------------------------------------
void test_y_confidence_prior() {
    const int rows = 48, cols = 64;
    cv::Mat first(rows, cols, CV_32F);
    std::vector<int> rx, ry;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const bool sparse = c >= cols / 2;
            const bool observed = sparse ? ((r % 8 == 0) && (c % 8 == 0))
                                         : ((r + c) % 4 != 0);
            first.at<float>(r, c) = 128.0f +
                (sparse && !observed ? ((r + c) % 2 ? 30.0f : -30.0f) : 0.0f);
            if (observed) { rx.push_back(r); ry.push_back(c); }
        }
    cv::Mat measurements(1, CS_HEADER_PIXELS + (int)rx.size(), CV_8UC3, cv::Scalar(0));
    for (size_t k = 0; k < rx.size(); ++k)
        measurements.at<cv::Vec3b>(CS_HEADER_PIXELS + (int)k) = cv::Vec3b(128, 128, 128);
    cv::Mat off = first.clone(), on = first.clone();
    cs_refine_y_confidence(measurements, rx, ry, off, 0.0f, 0.01f, 0.0f,
        6, 0, CS_BASIS_DCT, 2.0f);
    check(cv::norm(off, first, cv::NORM_INF) == 0.0,
        "y-confidence: zero strength leaves pixels unchanged");
    cs_refine_y_confidence(measurements, rx, ry, on, 1.0f, 0.01f, 0.0f,
        6, 0, CS_BASIS_DCT, 2.0f);
    double sparse_error = 0, dense_change = 0, sample_error = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (c >= cols / 2) sparse_error += std::fabs(on.at<float>(r, c) - 128.0f);
            else dense_change += std::fabs(on.at<float>(r, c) - first.at<float>(r, c));
        }
    for (size_t k = 0; k < rx.size(); ++k)
        sample_error += std::fabs(on.at<float>(rx[k], ry[k]) - 128.0f);
    std::printf("       y-confidence sparse MAE %.2f, dense change %.2f, measured MAE %.2f\n",
        sparse_error / (rows * cols / 2), dense_change / (rows * cols / 2), sample_error / rx.size());
    check(sparse_error < rows * cols / 2 * 20.0,
        "y-confidence: reduces unsampled sparse-area noise");
    check(dense_change / (rows * cols / 2) < 2.0,
        "y-confidence: preserves densely measured region");
    check(sample_error / rx.size() < 2.0,
        "y-confidence: measured data remains constrained");
}

void test_y_dual_tv_blend() {
    const int rows = 64, cols = 128;
    cv::Mat low(rows, cols, CV_8UC3, cv::Scalar(120, 120, 120));
    cv::Mat high(rows, cols, CV_8UC3, cv::Scalar(120, 120, 120));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (c < cols / 2) {
                const bool inside = r >= 12 && r < 52 && c >= 24 && c < 40;
                const uchar v = inside ? 200 : 40;
                low.at<cv::Vec3b>(r, c) = cv::Vec3b(v, v, v);
                const uchar h = inside ? 175 : 65;
                high.at<cv::Vec3b>(r, c) = cv::Vec3b(h, h, h);
            } else {
                const uchar v = (r + c) % 2 ? 80 : 160;
                low.at<cv::Vec3b>(r, c) = cv::Vec3b(v, v, v);
            }
        }
    std::vector<int> rx, ry;
    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8U);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const bool dense = c < cols / 2;
            if (dense ? ((r + c) % 4 != 0) : (r % 8 == 0 && c % 8 == 0)) {
                rx.push_back(r); ry.push_back(c);
                mask.at<uchar>(r, c) = 1;
            }
        }
    cv::Mat merged, weight;
    check(cs_y_mixed_density(rx, ry, rows, cols), "y-dual-tv: mixed tile detected");
    cs_blend_y_dual_tv(low, high, mask, merged, &weight);
    check(!merged.empty() && merged.size() == low.size(),
        "y-dual-tv: blend geometry");
    check(!weight.empty() && weight.size() == low.size() && weight.type() == CV_32F,
        "y-dual-tv: mask geometry/type");
    if (!weight.empty()) {
        double wmin, wmax;
        cv::minMaxLoc(weight, &wmin, &wmax);
        check(wmin >= 0.0 && wmax <= 1.0,
            "y-dual-tv: binary mask range");
        bool exact_source = true;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const float w = weight.at<float>(r, c);
                const cv::Vec3b expected = w == 1.0f
                    ? low.at<cv::Vec3b>(r, c) : high.at<cv::Vec3b>(r, c);
                if ((w != 0.0f && w != 1.0f) ||
                    merged.at<cv::Vec3b>(r, c) != expected) exact_source = false;
            }
        check(exact_source, "y-dual-tv: each output BGR pixel matches mask-selected candidate");
        check(weight.at<float>(32, 8) < 1e-6,
            "y-dual-tv: flat dense region has zero weight");
        check(weight.at<float>(32, 24) > 0.0,
            "y-dual-tv: edge region has positive weight");
        check(weight.at<float>(32, 32) > 0.9,
            "y-dual-tv: segment interior is filled, not just edges");
    }
    if (!merged.empty()) {
        check(weight.at<float>(32, 24) == 1.0f &&
            merged.at<cv::Vec3b>(32, 24) == low.at<cv::Vec3b>(32, 24),
            "y-dual-tv: image-wide segment copies low-TV detail");
        double wmax2 = 0; int argmax_r = 0, argmax_c = 0;
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            const float w = weight.at<float>(r, c);
            if (w > wmax2) { wmax2 = w; argmax_r = r; argmax_c = c; }
        }
        check(wmax2 > 0.9 &&
            merged.at<cv::Vec3b>(argmax_r, argmax_c)[0] ==
                low.at<cv::Vec3b>(argmax_r, argmax_c)[0],
            "y-dual-tv: full-weight pixels come from exactly one solve");
        check(weight.at<float>(32, 18) == 0.0f &&
            merged.at<cv::Vec3b>(32, 18) == high.at<cv::Vec3b>(32, 18),
            "y-dual-tv: exterior copies high-TV composite");
        cv::Mat sparse = merged(cv::Rect(85, 10, 32, 40));
        cv::Mat sparse_high = high(cv::Rect(85, 10, 32, 40));
        check(cv::norm(sparse, sparse_high, cv::NORM_L1) < 50,
            "y-dual-tv: sparse checkerboard noise remains suppressed");
        check(merged.at<cv::Vec3b>(32, 8)[0] == high.at<cv::Vec3b>(32, 8)[0],
            "y-dual-tv: flat dense region keeps high-TV base");

        // A strong edge on the top/left border used to make floodFill(0,0)
        // a no-op, and bitwise_not then selected the whole image. The
        // outside padding must leave unsupported areas black.
        cv::Mat corner = high.clone(), corner_weight;
        corner(cv::Rect(0, 0, 12, 12)).setTo(cv::Scalar(0, 0, 0));
        cs_blend_y_dual_tv(low, corner, mask, merged, &corner_weight);
        check(!corner_weight.empty() &&
            corner_weight.at<float>(32, 100) < 1e-6f,
            "y-dual-tv: border edge cannot fill unsupported image");

        // Dense, connected texture must never turn the entire image into
        // one low-TV segment, even when its component contains many edges.
        cv::Mat textured = high.clone(), textured_weight;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const uchar v = (c / 3) % 2 ? 210 : 30;
                textured.at<cv::Vec3b>(r, c) = cv::Vec3b(v, v, v);
            }
        cs_blend_y_dual_tv(low, textured, mask, merged, &textured_weight);
        check(!textured_weight.empty() && cv::mean(textured_weight)[0] < 0.8,
            "y-dual-tv: near-full-image component rejected");
    }
}

void test_y_dual_tv_color_segment() {
    const int rows = 160, cols = 192;
    cv::Mat high(rows, cols, CV_8UC3, cv::Scalar(180, 80, 25)); // blue background
    const cv::Scalar green(45, 150, 65), crown(55, 158, 70);
    cv::rectangle(high, cv::Rect(60, 68, 72, 75), green, cv::FILLED);
    cv::rectangle(high, cv::Rect(70, 27, 52, 54), crown, cv::FILLED);
    cv::Mat low = high.clone();
    cv::Mat sampled(rows, cols, CV_8U, cv::Scalar(0));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if ((r + c) % 3 != 0) sampled.at<uchar>(r, c) = 1;
    cv::Mat result, weight;
    cs_blend_y_dual_tv(low, high, sampled, result, &weight);
    check(!weight.empty() && weight.at<float>(38, 96) > 0.9f,
        "y-dual-tv: color segment fills crown beyond closed-edge seed");
    check(!weight.empty() && weight.at<float>(38, 25) < 1e-6f,
        "y-dual-tv: colorful background stays outside object segment");
}

void test_dual_tv_split() {
    // Checkerboard weight over a 2x2 solve-tile grid: white tiles (0,1) and
    // (1,0) go to lowtv, black tiles (0,0) and (1,1) to hightv (mean 0.5).
    // Row 1 carries the membership probes: both canvases are labeled inside
    // tile (0,0), so row 0 is label-polluted by design.
    const cv::Mat comp = make_test_image(64, 64);
    cv::Mat weight(64, 64, CV_32F, cv::Scalar(0.0f));
    weight(cv::Rect(32, 0, 32, 32)).setTo(1.0f);
    weight(cv::Rect(0, 32, 32, 32)).setTo(1.0f);
    std::vector<std::vector<cv::Mat>> tiles(2, std::vector<cv::Mat>(2));
    std::vector<std::vector<TileCoord>> coords(2, std::vector<TileCoord>(2));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            tiles[i][j] = cv::Mat(32, 32, CV_8UC3, cv::Scalar(1, 2, 3));
            coords[i][j] = TileCoord{ j * 32, i * 32 };
        }
    cv::Mat lowtv, hightv;
    check(cs_build_dual_tv_class_views(comp, weight, tiles, coords, lowtv, hightv),
        "dual-tv-split: builds on valid input");
    check(lowtv.size() == comp.size() && lowtv.type() == CV_8UC3 &&
        hightv.size() == comp.size() && hightv.type() == CV_8UC3,
        "dual-tv-split: full geometry and color type");
    auto tile_nonblack = [](const cv::Mat& m, int tr, int tc) {
        long long nz = 0;
        for (int r = 0; r < 32; ++r) {
            const cv::Vec3b* p = m.ptr<cv::Vec3b>(tr * 32 + r) + tc * 32;
            for (int k = 0; k < 32; ++k)
                if (p[k] != cv::Vec3b(0, 0, 0)) ++nz;
        }
        return nz;
    };
    check(tile_nonblack(lowtv, 1, 0) > 0 && tile_nonblack(hightv, 1, 0) == 0,
        "dual-tv-split: white tile lands in lowtv canvas only");
    check(tile_nonblack(hightv, 1, 1) > 0 && tile_nonblack(lowtv, 1, 1) == 0,
        "dual-tv-split: black tile lands in hightv canvas only");
    {
        cv::Mat diff;
        cv::absdiff(comp(cv::Rect(0, 32, 32, 32)), lowtv(cv::Rect(0, 32, 32, 32)), diff);
        cv::Mat gray;
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
        check(cv::countNonZero(gray) == 0, "dual-tv-split: lowtv pixels match composite");
    }
    // pixel-exact cutouts: low keeps the low candidate on white weight,
    // high keeps the blend on black weight, black elsewhere (no labels).
    {
        cv::Mat low = comp.clone();
        cv::Mat blended = comp.clone();
        blended(cv::Rect(32, 0, 32, 64)).setTo(cv::Scalar(7, 8, 9));
        cv::Mat lowcut, highcut;
        check(cs_build_dual_tv_cutouts(blended, low, weight, lowcut, highcut),
            "dual-tv-cutouts: builds on valid input");
        check(lowcut.size() == comp.size() && lowcut.type() == CV_8UC3 &&
            highcut.size() == comp.size() && highcut.type() == CV_8UC3,
            "dual-tv-cutouts: full geometry and color type");
        auto rect_eq = [](const cv::Mat& a, const cv::Mat& b, const cv::Rect& r) {
            cv::Mat diff;
            cv::absdiff(a(r), b(r), diff);
            cv::Mat gray;
            cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
            return cv::countNonZero(gray) == 0;
        };
        auto rect_black = [](const cv::Mat& a, const cv::Rect& r) {
            cv::Mat gray;
            cv::cvtColor(a(r), gray, cv::COLOR_BGR2GRAY);
            return cv::countNonZero(gray) == 0;
        };
        // weight white on (0,1)+(1,0): lowcut keeps low there, black elsewhere
        check(rect_eq(lowcut, low, cv::Rect(32, 0, 32, 32)) &&
            rect_eq(lowcut, low, cv::Rect(0, 32, 32, 32)),
            "dual-tv-cutouts: low keeps low-candidate on white weight");
        check(rect_black(lowcut, cv::Rect(0, 0, 32, 32)) &&
            rect_black(lowcut, cv::Rect(32, 32, 32, 32)),
            "dual-tv-cutouts: low black on black weight");
        // weight black on (0,0)+(1,1): highcut keeps blended there
        check(rect_eq(highcut, blended, cv::Rect(0, 0, 32, 32)) &&
            rect_eq(highcut, blended, cv::Rect(32, 32, 32, 32)),
            "dual-tv-cutouts: high keeps blended on black weight");
        check(rect_black(highcut, cv::Rect(32, 0, 32, 32)) &&
            rect_black(highcut, cv::Rect(0, 32, 32, 32)),
            "dual-tv-cutouts: high black on white weight");
        // 8U 0/255 weights accepted with identical result
        cv::Mat w8;
        weight.convertTo(w8, CV_8U, 255.0);
        cv::Mat lowcut8, highcut8;
        check(cs_build_dual_tv_cutouts(blended, low, w8, lowcut8, highcut8),
            "dual-tv-cutouts: 8U weights accepted");
        check(cv::norm(lowcut, lowcut8, cv::NORM_INF) == 0 &&
            cv::norm(highcut, highcut8, cv::NORM_INF) == 0,
            "dual-tv-cutouts: 8U weights match float weights");
        cv::Mat lc2, hc2;
        check(!cs_build_dual_tv_cutouts(cv::Mat(), low, weight, lc2, hc2),
            "dual-tv-cutouts: empty blend rejected");
        check(!cs_build_dual_tv_cutouts(blended, low, cv::Mat(), lc2, hc2),
            "dual-tv-cutouts: empty weight rejected");
    }
    // degenerate inputs rejected
    cv::Mat l2, h2;
    check(!cs_build_dual_tv_class_views(cv::Mat(), weight, tiles, coords, l2, h2),
        "dual-tv-split: empty composite rejected");
    check(!cs_build_dual_tv_class_views(comp, cv::Mat(), tiles, coords, l2, h2),
        "dual-tv-split: empty weight rejected");
    check(!cs_build_dual_tv_class_views(comp, weight,
        std::vector<std::vector<cv::Mat>>(), coords, l2, h2),
        "dual-tv-split: empty tile grid rejected");
    // all-tied fractions (uniform 0.5 weight): nothing to split
    {
        cv::Mat half(64, 64, CV_32F, cv::Scalar(0.5f));
        check(!cs_build_dual_tv_class_views(comp, half, tiles, coords, l2, h2),
            "dual-tv-split: tied fractions report nothing to split");
    }
}

