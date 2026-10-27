// Backend tests: photo roundtrip, photo upscalers, CS/direct super-resolution, HR detail, coupled dict, DnCNN/ONNX denoisers.
#include "test_common.hpp"
#include "enh_net.hpp"

// ---------------------------------------------------------------------------
// 6. photo-based roundtrip (second data point on real image statistics)
// ---------------------------------------------------------------------------
// Photos are gitignored (*.png), so this test is opt-in: it runs when a real
// photo is available, and skips (without failing) otherwise. Photo sources,
// in order: $CS_TEST_PHOTO, IMG_3690.png next to the test binary, or the
// project sample at ImgReconstruct_backend/IMG_3690.png.
void test_photo_roundtrip() {
    std::string path;
    if (const char* env = std::getenv("CS_TEST_PHOTO")) {
        path = env;
    }
    if (path.empty()) path = "IMG_3690.png";
    if (path.empty() || !std::ifstream(path.c_str()).good()) {
        path = "ImgReconstruct_backend/IMG_3690.png";
    }
    if (!std::ifstream(path.c_str(), std::ios::binary).good()) {
        std::printf("[SKIP] photo roundtrip: no photo found (set CS_TEST_PHOTO=<path>)\n");
        return;
    }

    cv::Mat original = cv::imread(path, cv::IMREAD_COLOR);
    if (original.empty()) {
        std::printf("[SKIP] photo roundtrip: '%s' did not load\n", path.c_str());
        return;
    }

    const std::string password = "roundtrip-test-password";
    const std::string tmp_out = ".cs_test_photo_output.png";
    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(path, "", password, 1.0f, 0, &encrypted) != 0) {
        check(false, "photo roundtrip: encrypt");
        return;
    }

    // cap the solve dimensions: a multi-tens-of-MP input would dominate the
    // suite's runtime; the auto parameters (24 tiles, overlap 24) still get
    // exercised at full tile statistics
    const int max_dim = 2400;
    if (original.cols > max_dim || original.rows > max_dim) {
        cv::resize(original, original, cv::Size(max_dim, original.rows * max_dim / original.cols));
    }
    const std::string tmp_in = ".cs_test_photo.png";
    cv::imwrite(tmp_in, original);

    if (decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, 5, 4, 0.01f, false) != 0) {
        check(false, "photo roundtrip: decrypt");
        std::remove(tmp_in.c_str());
        return;
    }

    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "photo roundtrip: decrypted file readable");
    if (!decrypted.empty()) {
        cv::Mat dec_sized;
        cv::resize(decrypted, dec_sized, original.size());
        const double p = cs_quality::psnr(original, dec_sized);
        const double s = cs_quality::ssim(original, dec_sized);
        std::printf("       photo roundtrip PSNR (%s, %dx%d): %.2f dB\n",
            path.c_str(), original.cols, original.rows, p);
        std::printf("       photo roundtrip SSIM: %.4f\n", s);
        // second data point on natural image statistics; with wavefront
        // neighbor warm-starts this measures ~30.4 dB (16.7 dB before).
        // Threshold keeps ~3.5 dB of headroom below the baseline.
        check(p > 27.0, "photo roundtrip: PSNR > 27 dB");
        // natural images score a bit lower on SSIM than the synthetic
        // gradient; baseline is ~0.95+, leave room for photo variance.
        if (s >= 0.0) check(s > 0.88, "photo roundtrip: SSIM > 0.88");
        else check(false, "photo roundtrip: SSIM unavailable");
    }

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// Photo upscaler backends: AVIR default, waifu2x opt-in with AVIR fallback.
// ---------------------------------------------------------------------------
void test_photo_upscaler() {
    auto make_grid = []() {
        std::vector<std::vector<cv::Mat>> grid(2, std::vector<cv::Mat>(2));
        grid[0][0] = cv::Mat(16, 24, CV_8UC3, cv::Scalar(10, 20, 30));
        grid[0][1] = cv::Mat(16, 24, CV_8UC3, cv::Scalar(40, 50, 60));
        grid[1][0] = cv::Mat(16, 24, CV_8UC3, cv::Scalar(70, 80, 90));
        // grid[1][1] stays empty: skipped tiles must survive untouched
        return grid;
    };
    auto all_2x = [](const std::vector<std::vector<cv::Mat>>& grid) {
        return !grid[0][0].empty() && grid[0][0].cols == 48 && grid[0][0].rows == 32 &&
            !grid[0][1].empty() && grid[0][1].cols == 48 && grid[0][1].rows == 32 &&
            !grid[1][0].empty() && grid[1][0].cols == 48 && grid[1][0].rows == 32 &&
            grid[1][1].empty();
    };
    {
        CsPhotoUpscalerOptions opt;  // AVIR default
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: avir upscales every tile exactly 2x");
    }
    {
        CsPhotoUpscalerOptions opt;
        opt.backend = "bogus-backend";
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: unknown backend warns and uses AVIR");
    }
    {
        // FSRCNN without weights: clean failure contract (no crash, AVIR
        // fallback owned by the caller).
        cv::Mat dst;
        check(!cs_fsrcnn_upscale_2x(cv::Mat(16, 16, CV_8UC3, cv::Scalar(1, 2, 3)), dst,
            ".cs_test_nope.pb"), "photo-upscaler: fsrcnn missing model fails cleanly");
        check(!cs_fsrcnn_upscale_2x(cv::Mat(), dst, ".cs_test_nope.pb"),
            "photo-upscaler: fsrcnn empty input rejected");
    }
    {
        // FSRCNN with weights (opt-in asset): exact 2x, deterministic.
        std::string model;
        if (const char* env = std::getenv("CS_TEST_FSRCNN")) model = env;
        if (model.empty()) model = "FSRCNN_x2.pb";
        if (!std::ifstream(model.c_str(), std::ios::binary).good()) model = "../FSRCNN_x2.pb";
        if (!std::ifstream(model.c_str(), std::ios::binary).good()) {
            std::printf("[SKIP] fsrcnn: no weights found (set CS_TEST_FSRCNN=<path>)\n");
        } else {
            cv::Mat tile = make_test_image(48, 32);
            cv::Mat up1, up2;
            check(cs_fsrcnn_upscale_2x(tile, up1, model), "photo-upscaler: fsrcnn upscales");
            check(!up1.empty() && up1.cols == 96 && up1.rows == 64 && up1.type() == CV_8UC3,
                "photo-upscaler: fsrcnn exact 2x geometry and type");
            check(cs_fsrcnn_upscale_2x(tile, up2, model), "photo-upscaler: fsrcnn second run");
            check(cv::norm(up1, up2, cv::NORM_INF) == 0, "photo-upscaler: fsrcnn deterministic");
            CsPhotoUpscalerOptions opt;
            opt.backend = "fsrcnn";
            opt.fsrcnn_model = model;
            auto grid = make_grid();
            cs_upscale_tiles_2x(grid, opt);
            check(all_2x(grid), "photo-upscaler: fsrcnn batch upscales every tile exactly 2x");
        }
    }
    {
        // No nunif here: a bogus command must degrade to AVIR, not fail.
        CsPhotoUpscalerOptions opt;
        opt.backend = "waifu2x";
        opt.waifu2x_cmd = "cs-nonexistent-waifu2x-cmd";
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: waifu2x failure falls back to AVIR");
    }
    {
        // Final polish up-then-down: empty/unknown selection is a no-op
        // (image untouched); a failing backend also leaves it untouched.
        CsPhotoUpscalerOptions opt;
        cv::Mat img = make_test_image(48, 32);
        cv::Mat before = img.clone();
        check(!cs_final_polish_updown(img, opt),
            "final-polish: empty selection is a no-op");
        check(cv::norm(img, before, cv::NORM_INF) == 0,
            "final-polish: empty selection leaves pixels untouched");
        opt.final_polish = "bogus-polish";
        check(!cs_final_polish_updown(img, opt),
            "final-polish: unknown backend is a no-op");
        check(cv::norm(img, before, cv::NORM_INF) == 0,
            "final-polish: unknown backend leaves pixels untouched");
        opt.final_polish = "waifu2x";
        opt.waifu2x_cmd = "cs-nonexistent-waifu2x-cmd";
        check(!cs_final_polish_updown(img, opt),
            "final-polish: backend failure leaves pixels untouched");
        check(cv::norm(img, before, cv::NORM_INF) == 0,
            "final-polish: failed run keeps geometry and content");
        cv::Mat empty;
        check(!cs_final_polish_updown(empty, opt),
            "final-polish: empty input rejected");
    }
    {
        // Denoising control composes into the waifu2x args.
        CsPhotoUpscalerOptions opt;
        check(cs_waifu2x_effective_args(opt) ==
            "--style photo --method scale -n 0 -g -1",
            "photo-upscaler: default args are pure 2x upscale");
        opt.waifu2x_method = "noise_scale";
        opt.waifu2x_noise = 2;
        check(cs_waifu2x_effective_args(opt) ==
            "--style photo --method noise_scale -n 2 -g -1",
            "photo-upscaler: method/noise compose into args");
        opt.waifu2x_args = "--custom";
        check(cs_waifu2x_effective_args(opt) == "--custom",
            "photo-upscaler: explicit args override method/noise");
    }
    {
        // Native ncnn backends share method/noise; -n -1 = pure upscale.
        CsPhotoUpscalerOptions opt;
        opt.backend = "waifu2x-ncnn";
        opt.waifu2x_ncnn_cmd = "cs-nonexistent-ncnn-cmd";
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: waifu2x-ncnn failure falls back to AVIR");
        check(cs_waifu2x_ncnn_effective_args(opt) ==
            "-n -1 -s 2 -m models-upconv_7_photo",
            "photo-upscaler: ncnn default args are pure 2x upscale");
        opt.waifu2x_method = "noise_scale";
        opt.waifu2x_noise = 2;
        check(cs_waifu2x_ncnn_effective_args(opt) ==
            "-n 2 -s 2 -m models-upconv_7_photo",
            "photo-upscaler: ncnn method/noise compose into args");
        opt.waifu2x_ncnn_args = "--custom";
        check(cs_waifu2x_ncnn_effective_args(opt) == "--custom",
            "photo-upscaler: ncnn explicit args override method/noise");
    }
    {
        CsPhotoUpscalerOptions opt;
        opt.backend = "realcugan";
        opt.realcugan_cmd = "cs-nonexistent-realcugan-cmd";
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: realcugan failure falls back to AVIR");
        check(cs_realcugan_effective_args(opt) == "-n -1 -s 2 -m models-se",
            "photo-upscaler: realcugan default args select models-se");
        opt.waifu2x_method = "noise_scale";
        opt.waifu2x_noise = 1;
        opt.realcugan_model = "models-pro";
        check(cs_realcugan_effective_args(opt) == "-n 1 -s 2 -m models-pro",
            "photo-upscaler: realcugan method/noise/model compose into args");
        opt.realcugan_args = "--custom";
        check(cs_realcugan_effective_args(opt) == "--custom",
            "photo-upscaler: realcugan explicit args override everything");
    }
    {
        // Short model names map to model dirs; explicit paths pass through.
        // Hermetic cmd: the resolver prefers model dirs shipped next to the
        // binary (SearchPathA), so a machine with realcugan installed would
        // otherwise return an install-qualified path here instead of the
        // bare passthrough below.
        CsPhotoUpscalerOptions opt;
        opt.realcugan_cmd = "cs-nonexistent-realcugan-cmd";
        opt.realcugan_model = "pro";
        check(cs_realcugan_model_dir(opt) == "models-pro",
            "photo-upscaler: pro maps to models-pro");
        opt.realcugan_model = "se";
        check(cs_realcugan_model_dir(opt) == "models-se",
            "photo-upscaler: se maps to models-se");
        opt.realcugan_model = "nose";
        check(cs_realcugan_model_dir(opt) == "models-nose",
            "photo-upscaler: nose maps to models-nose");
        opt.realcugan_model = "C:/m/models-se";
        check(cs_realcugan_model_dir(opt) == "C:/m/models-se",
            "photo-upscaler: explicit model path passes through");
        check(cs_realcugan_effective_args(opt) ==
            "-n -1 -s 2 -m C:/m/models-se",
            "photo-upscaler: explicit model path composes into args");
    }
}

void test_cs_super_resolution() {
    check(cs_is_superres_backend("cs-sr") && !cs_is_superres_backend("avir"),
        "cs-sr: backend selection");
    check(cs_is_srdirect_backend("sr-direct") && !cs_is_srdirect_backend("cs-sr") &&
        !cs_is_superres_backend("sr-direct"),
        "sr-direct: backend selection");
    // Simulate an imperfect LR reconstruction: measurements come from the
    // clean LR image, while the solved tile has a small brightness bias.
    const cv::Mat original = make_test_image(80, 80);
    cv::Mat measured, solved;
    cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
    cv::add(measured, cv::Scalar(8, 8, 8), solved);
    const int m = measured.rows * measured.cols;
    // NOTE: packed is (N x 1): at<>(k) addresses rows. A 1xN row mat with
    // the same pattern would read/write out of bounds (step mismatch).
    cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<int> rx, ry;
    rx.reserve(m); ry.reserve(m);
    for (int row = 0; row < measured.rows; ++row)
        for (int col = 0; col < measured.cols; ++col) {
            rx.push_back(row); ry.push_back(col);
            packed.at<cv::Vec3b>(CS_HEADER_PIXELS + row * measured.cols + col) =
                measured.at<cv::Vec3b>(row, col);
        }
    cv::Mat baseline, refined;
    cs_upscale_2x_avir(solved, baseline);
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, refined);
    check(refined.size() == original.size() && refined.type() == CV_8UC3,
        "cs-sr: 2x geometry and color type");
    if (!refined.empty()) {
        cv::Mat base_lr, sr_lr;
        cv::resize(baseline, base_lr, measured.size(), 0, 0, cv::INTER_AREA);
        cv::resize(refined, sr_lr, measured.size(), 0, 0, cv::INTER_AREA);
        check(cv::norm(sr_lr, measured, cv::NORM_L2) <
              cv::norm(base_lr, measured, cv::NORM_L2),
            "cs-sr: refinement reduces LR measurement residual");
    }
    cv::Mat fallback;
    cs_sr_upscale_tile_2x(solved, packed, {}, {}, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, fallback);
    check(cv::norm(fallback, baseline, cv::NORM_INF) == 0,
        "cs-sr: missing measurements fall back to AVIR");
    // RED-lite plumbing: explicit red=0 matches the default call exactly,
    // and a mid-strength RED pass completes with valid geometry.
    cv::Mat red0, red05;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, red0, 0.0f);
    check(cv::norm(red0, refined, cv::NORM_INF) == 0,
        "cs-sr: red=0 is bit-identical to default");
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, red05, 0.5f);
    check(red05.size() == original.size() && red05.type() == CV_8UC3,
        "cs-sr: red=0.5 keeps 2x geometry and color type");
    if (!red05.empty()) {
        cv::Mat red_lr;
        cv::resize(red05, red_lr, measured.size(), 0, 0, cv::INTER_AREA);
        const double n = cv::norm(red_lr, measured, cv::NORM_L2);
        check(std::isfinite(n), "cs-sr: red=0.5 residual is finite");
    }
    // HQS schedule: decay=1.0 is bit-identical to the default call;
    // decay=0 keeps the first pass and drops the second, staying sane.
    cv::Mat red10, red00;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, red10, 0.5f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f);
    check(cv::norm(red10, red05, cv::NORM_INF) == 0,
        "cs-sr: red-decay=1 is bit-identical to default");
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, red00, 0.5f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 0.0f);
    check(red00.size() == original.size() && red00.type() == CV_8UC3,
        "cs-sr: red-decay=0 keeps 2x geometry and color type");
    // --sr-init cascade warm start: same contract, valid geometry.
    cv::Mat casc;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, casc, 0.0f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, true);
    check(casc.size() == original.size() && casc.type() == CV_8UC3,
        "cs-sr: cascade init keeps 2x geometry and color type");
    // sr-direct fused wrapper: single-stage solve from the same samples.
    cv::Mat direct, direct_casc, direct_fb;
    cs_sr_direct_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, direct);
    check(direct.size() == original.size() && direct.type() == CV_8UC3,
        "sr-direct: 2x geometry and color type");
    if (!direct.empty()) {
        // Fully sampled fixture: the single-stage solve must fit the
        // samples at least as well as AVIR (catches dark/degenerate solves).
        cv::Mat direct_lr;
        cv::resize(direct, direct_lr, measured.size(), 0, 0, cv::INTER_AREA);
        cv::Mat base_lr;
        cv::resize(baseline, base_lr, measured.size(), 0, 0, cv::INTER_AREA);
        check(cv::norm(direct_lr, measured, cv::NORM_L2) <
              cv::norm(base_lr, measured, cv::NORM_L2),
            "sr-direct: single-stage solve reduces LR measurement residual");
    }
    cs_sr_direct_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, direct_casc, true);
    check(direct_casc.size() == original.size() && direct_casc.type() == CV_8UC3,
        "sr-direct: cascade init keeps 2x geometry and color type");
    cs_sr_direct_upscale_tile_2x(solved, packed, {}, {}, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, direct_fb);
    check(cv::norm(direct_fb, baseline, cv::NORM_INF) == 0,
        "sr-direct: missing measurements fall back to AVIR");
    check(cs_is_dip_backend("dip") && !cs_is_dip_backend("cs-sr") &&
        !cs_is_superres_backend("dip"),
        "dip: backend selection");
    // Untrained-prior wrapper on the same fully sampled fixture: valid
    // geometry, fits samples at least as well as AVIR (catches broken
    // adjoints: a wrong transpose cannot reduce the residual), and
    // deterministic across runs (fixed seed, serial math).
    cv::Mat dip, dip2, dip_fb;
    cs_dip_upscale_tile_2x(solved, packed, rx, ry, dip);
    check(dip.size() == original.size() && dip.type() == CV_8UC3,
        "dip: 2x geometry and color type");
    if (!dip.empty()) {
        cv::Mat dip_lr, dip_base_lr;
        cv::resize(dip, dip_lr, measured.size(), 0, 0, cv::INTER_AREA);
        cv::resize(baseline, dip_base_lr, measured.size(), 0, 0, cv::INTER_AREA);
        check(cv::norm(dip_lr, measured, cv::NORM_L2) <
              cv::norm(dip_base_lr, measured, cv::NORM_L2),
            "dip: generator fit reduces LR measurement residual");
    }
    cs_dip_upscale_tile_2x(solved, packed, rx, ry, dip2);
    if (!dip.empty() && !dip2.empty()) {
        cv::Mat d;
        cv::absdiff(dip, dip2, d);
        double mn, mx;
        cv::minMaxLoc(d.reshape(1), &mn, &mx);
        check(mx == 0.0, "dip: deterministic");
    }
    cs_dip_upscale_tile_2x(solved, packed, {}, {}, dip_fb);
    check(cv::norm(dip_fb, baseline, cv::NORM_INF) == 0,
        "dip: missing measurements fall back to AVIR");
}

// Single-stage vs two-stage SR experiment: the same LR samples feed (a)
// the production two-stage path (BGR LR solve with bias + HR refine) and
// (b) cs_sr_direct_luma (float-consistent HR solve, no LR stage), from
// zeros and from an AVIR-demosaic init. Prints PSNR/time for the
// integration decision; asserts only pipeline integrity (finite, sane).
void test_sr_direct() {
    cv::Mat truth_bgr = make_test_image(80, 80);
    cv::Mat truth_gray;
    cv::cvtColor(truth_bgr, truth_gray, cv::COLOR_BGR2GRAY);
    cv::Mat truth_f;
    truth_gray.convertTo(truth_f, CV_32F, 1.0 / 255.0);
    cv::Mat lr;
    cv::resize(truth_f, lr, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
    // random 50% sample positions (fixed seed: deterministic fixture)
    std::mt19937 rng(1234);
    std::vector<int> rx, ry;
    std::vector<float> b;
    for (int r = 0; r < 40; ++r) {
        for (int c = 0; c < 40; ++c) {
            if ((rng() % 100) < 50) {
                rx.push_back(r); ry.push_back(c);
                b.push_back(lr.at<float>(r, c));
            }
        }
    }
    const int m = (int)b.size();
    check(m > 400, "sr-direct: fixture holds enough samples");
    // (a) two-stage through the production wrapper (biased LR like a solver).
    // NOTE: packed is (N x 1) to mirror the container convention (header
    // pixels then body rows); at<>() flat-indexes continuous mats either
    // way, but Nx1 matches what cs_extract_channel_measurements reads.
    cv::Mat lr8, solved_bgr, packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    lr.convertTo(lr8, CV_8U, 255.0);
    cv::cvtColor(lr8, lr8, cv::COLOR_GRAY2BGR);
    cv::add(lr8, cv::Scalar(8, 8, 8), solved_bgr);
    for (int k = 0; k < m; ++k) {
        packed.at<cv::Vec3b>(CS_HEADER_PIXELS + k) = solved_bgr.at<cv::Vec3b>(rx[k], ry[k]);
    }
    cv::Mat two;
    const double t0 = (double)cv::getTickCount();
    cs_sr_upscale_tile_2x(solved_bgr, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, two);
    const double t_two = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    double p_two = -1.0;
    if (!two.empty()) {
        cv::Mat two_g;
        cv::cvtColor(two, two_g, cv::COLOR_BGR2GRAY);
        p_two = cs_quality::psnr(truth_gray, two_g);
    }
    // (b) direct HR solves: zeros init, then demosaic init + anchor
    std::vector<float> hr0(80 * 80, 0.0f), hr1(80 * 80, 0.0f);
    const double t1 = (double)cv::getTickCount();
    cs_sr_direct_luma(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
        8, 2, 0, 2.0f, 0, 0.0f, hr0.data());
    const double t_d0 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
    const double t2 = (double)cv::getTickCount();
    cs_sr_direct_luma(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
        8, 2, 0, 2.0f, 1, 0.25f, hr1.data());
    const double t_d1 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
    cv::Mat m0(80, 80, CV_32F, hr0.data()), m1(80, 80, CV_32F, hr1.data());
    cv::Mat m08, m18;
    m0.convertTo(m08, CV_8U, 255.0);
    m1.convertTo(m18, CV_8U, 255.0);
    const double p_d0 = cs_quality::psnr(truth_gray, m08);
    const double p_d1 = cs_quality::psnr(truth_gray, m18);
    std::printf("       sr-direct: two-stage %.2f dB / %.3fs | zero-init %.2f dB / %.3fs | demosaic %.2f dB / %.3fs\n",
        p_two, t_two, p_d0, t_d0, p_d1, t_d1);
    check(std::isfinite(p_d0) && std::isfinite(p_d1), "sr-direct: both inits finite");
    check(p_d0 > 8.0 && p_d1 > 8.0, "sr-direct: both inits sane");
    // (c) zeros init, 4x iteration budget: can the direct solve reach the
    // two-stage quality at all, and at what cost?
    std::vector<float> hrC(80 * 80, 0.0f);
    const double t3 = (double)cv::getTickCount();
    cs_sr_direct_luma(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
        8, 2, 128, 2.0f, 0, 0.0f, hrC.data());
    const double t_dC = ((double)cv::getTickCount() - t3) / cv::getTickFrequency();
    cv::Mat mC(80, 80, CV_32F, hrC.data()), mC8;
    mC.convertTo(mC8, CV_8U, 255.0);
    const double p_dC = cs_quality::psnr(truth_gray, mC8);
    // (d) cheap LR warm start in FLOAT (no 8U quantize): 1-pass LR FISTA on
    // B, AVIR-upscaled, then the standard HR solve + anchor. Isolates the
    // 8-bit quantization hypothesis from warm-start economics.
    double p_dD = -1.0, t_dD = 0.0;
    {
        const double t4 = (double)cv::getTickCount();
        cv::Mat lr_ref = cv::Mat::zeros(40, 40, CV_32F);
        cv::dct(lr_ref, lr_ref, 0);
        lr_ref /= 10.0f;
        cv::Mat dummy;
        reconstruct_color_channel_fista(packed, 0, 0.0375f, 40, 40, rx, ry,
            2, lr_ref, false, dummy, 0.0f, 1, 8, CS_BASIS_DCT, 2.0f);
        cv::Mat lr_f;
        lr_ref.convertTo(lr_f, CV_32F, 1.0 / 255.0);
        cv::Mat hr_warm;
        cv::resize(lr_f, hr_warm, cv::Size(80, 80), 0, 0, cv::INTER_LINEAR);
        if (!hr_warm.isContinuous()) hr_warm = hr_warm.clone();
        std::vector<float> hrD(80 * 80, 0.0f);
        cs_sr_direct_luma(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
            8, 2, 0, 2.0f, 2, 0.25f, hrD.data(), (const float*)hr_warm.data);
        t_dD = ((double)cv::getTickCount() - t4) / cv::getTickFrequency();
        cv::Mat mD(80, 80, CV_32F, hrD.data()), mD8;
        mD.convertTo(mD8, CV_8U, 255.0);
        p_dD = cs_quality::psnr(truth_gray, mD8);
    }
    std::printf("       sr-direct: 4x-budget %.2f dB / %.3fs | cheapLR-float %.2f dB / %.3fs\n",
        p_dC, t_dC, p_dD, t_dD);
    check(std::isfinite(p_dC) && std::isfinite(p_dD), "sr-direct: extended configs finite");
    // (e) bias sensitivity: identical 4x-budget solve on uniformly biased
    // samples (+0.031, like the BGR-packed driver input). A convex solver
    // cannot diverge from a DC shift; if this collapses, the instability is
    // b-dependent and the Lipschitz bound (not the driver) is at fault.
    {
        std::vector<float> bb = b;
        for (float& v : bb) v += 0.031f;
        std::vector<float> hrE(80 * 80, 0.0f);
        cs_sr_direct_luma(bb.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
            8, 2, 128, 2.0f, 0, 0.0f, hrE.data());
        cv::Mat mE(80, 80, CV_32F, hrE.data()), mE8;
        mE.convertTo(mE8, CV_8U, 255.0);
        const double p_dE = cs_quality::psnr(truth_gray, mE8);
        cv::Scalar mnE = cv::mean(mE8);
        std::printf("       sr-direct: biased-4x %.2f dB (mean %.1f)\n", p_dE, mnE[0]);
        check(std::isfinite(p_dE), "sr-direct: biased config finite");
    }
    // (f) cascade init + STANDARD budget: the premium-killer test. If this
    // matches zeros+4x above, the blocker was cold-start theory (init
    // quality), not iteration count — and production could drop the 4x tax.
    {
        std::vector<float> hrF(80 * 80, 0.0f), hrF2(80 * 80, 0.0f);
        cs_sr_cascade_init(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0, 2.0f, hrF.data());
        cs_sr_cascade_init(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0, 2.0f, hrF2.data());
        check(hrF == hrF2, "sr-direct: cascade deterministic");
        bool finite = true;
        for (float v : hrF) if (!std::isfinite(v)) finite = false;
        check(finite, "sr-direct: cascade output finite");
        std::vector<float> hrG(80 * 80, 0.0f);
        const double t5 = (double)cv::getTickCount();
        cs_sr_direct_luma(b.data(), rx.data(), ry.data(), m, 40, 40, 0.018f, 0.0f,
            8, 2, 0, 2.0f, 2, 0.25f, hrG.data(), hrF.data());
        const double t_dF = ((double)cv::getTickCount() - t5) / cv::getTickFrequency();
        cv::Mat mF(80, 80, CV_32F, hrG.data()), mF8;
        mF.convertTo(mF8, CV_8U, 255.0);
        const double p_dF = cs_quality::psnr(truth_gray, mF8);
        std::printf("       sr-direct: cascade+1x %.2f dB / %.3fs\n", p_dF, t_dF);
        check(std::isfinite(p_dF) && p_dF > 8.0, "sr-direct: cascade config sane");
    }
}

void test_coupled_dict() {
    // OMP single-atom recovery: y = 3*e2 + tiny fixed noise over a dict
    // whose first 4 atoms are unit axes. Greedy correlation must pick 2.
    {
        const int dim = 8, atoms = 8;
        std::vector<float> D((size_t)atoms * dim, 0.0f);
        for (int a = 0; a < 4; ++a) D[(size_t)a * dim + a] = 1.0f;
        std::mt19937 rng(7);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (int a = 4; a < atoms; ++a) {
            double nrm = 0.0;
            for (int p = 0; p < dim; ++p) {
                D[(size_t)a * dim + p] = gauss(rng);
                nrm += (double)D[(size_t)a * dim + p] * D[(size_t)a * dim + p];
            }
            nrm = std::sqrt(nrm);
            for (int p = 0; p < dim; ++p) D[(size_t)a * dim + p] = (float)(D[(size_t)a * dim + p] / nrm);
        }
        float y[8] = { 0.01f, -0.01f, 3.0f, 0.01f, -0.01f, 0.01f, -0.01f, 0.01f };
        float code[8] = {};
        cs_omp_encode_vec(y, D.data(), dim, atoms, 1, code);
        check(code[2] != 0.0f, "dict omp: dominant atom selected");
        int nnz = 0;
        for (int a = 0; a < atoms; ++a) if (code[a] != 0.0f) ++nnz;
        check(nnz == 1 && std::fabs(code[2] - 3.0f) < 0.1f, "dict omp: exact support and amplitude");
    }
    // CSD2 save/load roundtrip preserves bits
    {
        cs_coupled_dict d;
        d.atoms = 4; d.plr = 8; d.phr = 16; d.feat = 256;
        d.Dh.assign(4 * 256, 0.5f);
        d.Dl.assign(4 * 256, 0.25f);
        const std::string path = ".cs_test_dict.csd2";
        check(cs_save_coupled_dictionary(path, d), "dict csd2: save");
        cs_coupled_dict back;
        check(cs_load_coupled_dictionary(path, back), "dict csd2: load");
        check(back.atoms == 4 && back.plr == 8 && back.phr == 16 && back.feat == 256 &&
            back.Dh == d.Dh && back.Dl == d.Dl, "dict csd2: roundtrip identical");
        cs_coupled_dict bad;
        check(!cs_load_coupled_dictionary(".cs_test_nope.csd2", bad), "dict csd2: missing file rejected");
        check(cs_sr_dict_cached(".cs_test_nope.csd2") == nullptr, "dict cache: missing file -> null");
        std::remove(path.c_str());
    }
    // micro joint training: 2 tiny images, 16 atoms, 2 iters (seconds)
    cs_coupled_dict micro;
    {
        cv::Mat a = make_test_image(128, 128);
        cv::Mat b;
        cv::flip(a, b, 1);
        check(cs_train_coupled_dictionary({ a, b }, 16, 2, 400, 8, micro), "dict train: micro joint training succeeds");
        check(micro.atoms == 16 && (int)micro.Dh.size() == 16 * 256 && (int)micro.Dl.size() == 16 * 256,
            "dict train: shapes");
        double dl_norm0 = 0.0;
        for (int q = 0; q < 256; ++q) dl_norm0 += (double)micro.Dl[q] * micro.Dl[q];
        check(std::fabs(std::sqrt(dl_norm0) - 1.0) < 1e-4, "dict train: Dl atoms unit norm");
        bool finite = true;
        for (float v : micro.Dh) if (!std::isfinite(v)) finite = false;
        check(finite, "dict train: Dh finite");
    }
    // dict SR on a biased tile: geometry, determinism, AVIR fallback
    {
        const cv::Mat original = make_test_image(80, 80);
        cv::Mat measured, solved;
        cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
        cv::add(measured, cv::Scalar(8, 8, 8), solved);
        cv::Mat baseline, out1, out2, fb;
        cs_upscale_2x_avir(solved, baseline);
        cs_sr_upscale_dict_2x(solved, &micro, out1);
        cs_sr_upscale_dict_2x(solved, &micro, out2);
        check(out1.size() == original.size() && out1.type() == CV_8UC3,
            "dict sr: 2x geometry and color type");
        check(cv::norm(out1, out2, cv::NORM_INF) == 0, "dict sr: deterministic");
        cs_sr_upscale_dict_2x(solved, nullptr, fb);
        check(cv::norm(fb, baseline, cv::NORM_INF) == 0, "dict sr: null dict falls back to AVIR");
    }
}

// Model asset for the DnCNN tests (mirrors the opt-in photo pattern):
// $CS_TEST_DNCNN, dncnn_color.onnx next to the test binary, or the
// vendored models/dncnn/dncnn_color.onnx.
static std::string cs_test_dncnn_model() {
    if (const char* env = std::getenv("CS_TEST_DNCNN")) return env;
    const char* cands[] = { "dncnn_color.onnx", "models/dncnn/dncnn_color.onnx" };
    for (auto c : cands) {
        std::ifstream f(c, std::ios::binary);
        if (f.good()) return c;
    }
    return "";
}

static std::string cs_test_scunet_model() {
    if (const char* env = std::getenv("CS_TEST_SCUNET")) return env;
    const char* cands[] = { "scunet_color_real_psnr.onnx",
        "models/scunet/scunet_color_real_psnr.onnx" };
    for (auto c : cands) {
        std::ifstream f(c, std::ios::binary);
        if (f.good()) return c;
    }
    return "";
}

void test_dncnn_red() {
    check(!cs_dncnn_available(".cs_test_nope.onnx"), "dncnn: missing model rejected");
    cv::Mat probe, out;
    check(!cs_dncnn_denoise_bgr(probe, out, ".cs_test_nope.onnx"), "dncnn: missing model run fails");
    check(!cs_dncnn_denoise_bgr(cv::Mat(), out, "whatever"), "dncnn: empty input rejected");
    const std::string model = cs_test_dncnn_model();
    if (model.empty()) {
        std::printf("[SKIP] dncnn: no model found (set CS_TEST_DNCNN=<path>)\n");
        return;
    }
    check(cs_dncnn_available(model), "dncnn: model loads");
    // residual convention probe: a clean image must come back nearly
    // unchanged (small residual), or the [0,1]/residual assumption is wrong
    cv::Mat clean = make_test_image(128, 128);
    cv::Mat dn;
    check(cs_dncnn_denoise_bgr(clean, dn, model), "dncnn: denoise runs");
    if (dn.empty()) {
        check(false, "dncnn: output readable");
        return;
    }
    check(dn.size() == clean.size() && dn.type() == CV_8UC3, "dncnn: geometry preserved");
    const double stay = cs_quality::psnr(clean, dn);
    std::printf("       dncnn clean-in PSNR: %.2f dB\n", stay);
    check(stay > 28.0, "dncnn: near-identity on clean input");
    // noisy image must move toward clean (validates the denoiser does work
    // in this harness, not just pass through)
    cv::Mat cleanf, nse, noisyf, noisy, dn_noisy;
    clean.convertTo(cleanf, CV_32F);
    nse.create(clean.size(), CV_32FC3);
    cv::randn(nse, 0.0, 25.0);
    noisyf = cleanf + nse;
    noisyf.convertTo(noisy, CV_8U);
    const double p_noisy = cs_quality::psnr(clean, noisy);
    check(cs_dncnn_denoise_bgr(noisy, dn_noisy, model), "dncnn: noisy denoise runs");
    const double p_dn = cs_quality::psnr(clean, dn_noisy);
    std::printf("       dncnn noisy %.2f dB -> denoised %.2f dB\n", p_noisy, p_dn);
    check(p_dn > p_noisy + 2.0, "dncnn: noisy input improves >= 2 dB");
    cv::Mat dn2;
    cs_dncnn_denoise_bgr(clean, dn2, model);
    check(cv::norm(dn, dn2, cv::NORM_INF) == 0, "dncnn: deterministic");
    // RED integration: dncnn branch of the cs-sr tile solve completes sanely
    {
        cv::Mat original = make_test_image(80, 80);
        cv::Mat measured, solved;
        cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
        cv::add(measured, cv::Scalar(8, 8, 8), solved);
        const int m = measured.rows * measured.cols;
        cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<int> rx, ry;
        for (int row = 0; row < measured.rows; ++row)
            for (int col = 0; col < measured.cols; ++col) {
                rx.push_back(row); ry.push_back(col);
                packed.at<cv::Vec3b>(CS_HEADER_PIXELS + row * measured.cols + col) =
                    measured.at<cv::Vec3b>(row, col);
            }
        cv::Mat refined;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, refined, 0.5f, "dncnn", model);
        check(refined.size() == original.size() && refined.type() == CV_8UC3,
            "dncnn red: 2x geometry and color type");
        if (!refined.empty()) {
            cv::Mat rl;
            cv::resize(refined, rl, measured.size(), 0, 0, cv::INTER_AREA);
            check(std::isfinite(cv::norm(rl, measured, cv::NORM_L2)), "dncnn red: residual finite");
        }
    }
}

void test_onnx_denoisers() {
    // kind mapping + model resolution are pure and unconditional
    check(cs_onnx_kind_for("dncnn") == CS_ONNX_DNCNN &&
        cs_onnx_kind_for("SCUNet") == CS_ONNX_SCUNET &&
        cs_onnx_kind_for("restormer") == CS_ONNX_RESTORMER &&
        cs_onnx_kind_for("nlmeans") < 0 && cs_onnx_kind_for("bogus") < 0,
        "onnx: kind mapping");
    check(cs_red_model_for("nlmeans", "d.onnx", "") == "" &&
        cs_red_model_for("dncnn", "d.onnx", "") == "d.onnx" &&
        cs_red_model_for("dncnn", "", "") == CS_DNCNN_DEFAULT_MODEL &&
        cs_red_model_for("scunet", "d.onnx", "") == CS_SCUNET_DEFAULT_MODEL &&
        cs_red_model_for("restormer", "d.onnx", "") == CS_RESTORMER_DEFAULT_MODEL &&
        cs_red_model_for("scunet", "d.onnx", "r.onnx") == "r.onnx",
        "onnx: model path resolution");
    // missing models fail gracefully per kind (restormer has no weights here)
    check(!cs_onnx_available(".cs_test_nope.onnx", CS_ONNX_SCUNET),
        "onnx: missing scunet model rejected");
    check(!cs_onnx_available(".cs_test_nope.onnx", CS_ONNX_RESTORMER),
        "onnx: missing restormer model rejected");
    cv::Mat probe, out;
    check(!cs_onnx_denoise_bgr(probe, out, ".cs_test_nope.onnx", CS_ONNX_SCUNET),
        "onnx: missing model run fails");
    check(!cs_onnx_denoise_bgr(cv::Mat(), out, "whatever", CS_ONNX_RESTORMER),
        "onnx: empty input rejected");
    // SCUNet-real weights (opt-in asset, gitignored): clean pass-through,
    // noisy improvement, determinism — same contract as the dncnn test.
    const std::string model = cs_test_scunet_model();
    if (model.empty()) {
        std::printf("[SKIP] scunet: no model found (set CS_TEST_SCUNET=<path>)\n");
        return;
    }
    check(cs_onnx_available(model, CS_ONNX_SCUNET), "scunet: model loads");
    cv::Mat clean = make_test_image(128, 128);
    cv::Mat dn;
    check(cs_onnx_denoise_bgr(clean, dn, model, CS_ONNX_SCUNET), "scunet: denoise runs");
    if (dn.empty()) {
        check(false, "scunet: output readable");
        return;
    }
    check(dn.size() == clean.size() && dn.type() == CV_8UC3, "scunet: geometry preserved");
    const double stay = cs_quality::psnr(clean, dn);
    std::printf("       scunet clean-in PSNR: %.2f dB\n", stay);
    check(stay > 28.0, "scunet: near-identity on clean input");
    cv::Mat cleanf, nse, noisyf, noisy, dn_noisy;
    clean.convertTo(cleanf, CV_32F);
    nse.create(clean.size(), CV_32FC3);
    cv::randn(nse, 0.0, 25.0);
    noisyf = cleanf + nse;
    noisyf.convertTo(noisy, CV_8U);
    const double p_noisy = cs_quality::psnr(clean, noisy);
    check(cs_onnx_denoise_bgr(noisy, dn_noisy, model, CS_ONNX_SCUNET), "scunet: noisy denoise runs");
    const double p_dn = cs_quality::psnr(clean, dn_noisy);
    std::printf("       scunet noisy %.2f dB -> denoised %.2f dB\n", p_noisy, p_dn);
    check(p_dn > p_noisy + 2.0, "scunet: noisy input improves >= 2 dB");
    cv::Mat dn2;
    cs_onnx_denoise_bgr(clean, dn2, model, CS_ONNX_SCUNET);
    {
        // absdiff-based (cv::norm INF flaked on identical mats in this
        // static OpenCV build; the transparent reduction never does).
        cv::Mat d;
        cv::absdiff(dn, dn2, d);
        double mn, mx;
        cv::minMaxLoc(d.reshape(1), &mn, &mx);
        check(mx == 0.0, "scunet: deterministic");
    }
    // RED integration: scunet branch of the cs-sr tile solve completes
    {
        cv::Mat original = make_test_image(80, 80);
        cv::Mat measured, solved;
        cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
        cv::add(measured, cv::Scalar(8, 8, 8), solved);
        const int m = measured.rows * measured.cols;
        cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<int> rx, ry;
        for (int row = 0; row < measured.rows; ++row)
            for (int col = 0; col < measured.cols; ++col) {
                rx.push_back(row); ry.push_back(col);
                packed.at<cv::Vec3b>(CS_HEADER_PIXELS + row * measured.cols + col) =
                    measured.at<cv::Vec3b>(row, col);
            }
        cv::Mat refined;
        cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
            8, 2, 0, CS_BASIS_DCT, 2.0f, refined, 0.5f, "scunet", "", false, model);
        check(refined.size() == original.size() && refined.type() == CV_8UC3,
            "scunet red: 2x geometry and color type");
    }
}

// ---------------------------------------------------------------------------
// HR-detail payload (--hr-detail): native-luma edge anchors as a trailing
// CTR-encrypted body section (modes 4/9). Unit pins for selection, codec
// and apply, plus container roundtrips through the real encrypt/decrypt.
// ---------------------------------------------------------------------------
void test_hr_detail() {
    check(cs_is_hrdetail(CS_MODE_BGR_HRDETAIL) &&
        cs_is_hrdetail(CS_MODE_YCC420_HRDETAIL),
        "hr-detail: backend selection");
    check(!cs_is_hrdetail(CS_MODE_RANDOM) && !cs_is_hrdetail(CS_MODE_YCC420),
        "hr-detail: non-hr modes rejected");
    check(cs_is_ycc(CS_MODE_YCC420_HRDETAIL) && !cs_is_ycc(CS_MODE_BGR_HRDETAIL),
        "hr-detail: ycc membership");

    // Selection: structured image yields in-range deterministic anchors
    // whose values match the reference gray plane.
    const cv::Mat edge = make_test_image(128, 96);
    cv::Mat edge_gray;
    cv::cvtColor(edge, edge_gray, cv::COLOR_BGR2GRAY);
    const std::vector<CsHrSample> a1 = cs_select_hr_detail(edge, 128, 96, 64);
    const std::vector<CsHrSample> a2 = cs_select_hr_detail(edge, 128, 96, 64);
    check(a1.size() == 64, "hr-detail: selection count");
    bool same = (a1.size() == a2.size());
    bool in_range = true, values_ok = true;
    for (size_t k = 0; k < a1.size() && (same || in_range || values_ok); ++k) {
        if (a1[k].idx != a2[k].idx || a1[k].y != a2[k].y) same = false;
        if (a1[k].idx >= (uint32_t)(128 * 96)) { in_range = false; continue; }
        if (a1[k].y != edge_gray.at<uint8_t>(a1[k].idx / 128, a1[k].idx % 128))
            values_ok = false;
    }
    check(same, "hr-detail: selection deterministic");
    check(in_range, "hr-detail: selection in range");
    check(values_ok, "hr-detail: selection values match gray");
    check(cs_select_hr_detail(edge, 128, 96, 0).empty(),
        "hr-detail: zero count selects nothing");
    check(cs_select_hr_detail(cv::Mat(), 128, 96, 64).empty(),
        "hr-detail: empty image selects nothing");
    check(cs_select_hr_detail(cv::Mat::zeros(64, 64, CV_8UC3), 64, 64, 64).empty(),
        "hr-detail: flat image selects nothing");

    // Codec roundtrip under a fixed key/IV, plus corrupt-length rejection.
    uint8_t key[32], iv[16];
    for (int i = 0; i < 32; ++i) key[i] = (uint8_t)(i * 3 + 1);
    for (int i = 0; i < 16; ++i) iv[i] = (uint8_t)(i * 5 + 2);
    std::vector<uint8_t> wire(a1.size() * CS_HRDETAIL_SAMPLE_BYTES);
    cs_write_hr_payload(wire.data(), a1, key, iv);
    std::vector<CsHrSample> back;
    check(cs_read_hr_payload(wire.data(), wire.size(), key, iv, back),
        "hr-detail: codec roundtrip parses");
    bool codec_same = (back.size() == a1.size());
    for (size_t k = 0; k < back.size() && codec_same; ++k)
        codec_same = (back[k].idx == a1[k].idx && back[k].y == a1[k].y);
    check(codec_same, "hr-detail: codec roundtrip identical");
    std::vector<CsHrSample> junk;
    check(!cs_read_hr_payload(wire.data(), 0, key, iv, junk),
        "hr-detail: empty payload rejected");
    check(!cs_read_hr_payload(wire.data(), 7, key, iv, junk),
        "hr-detail: misaligned payload rejected");
    check(!cs_read_hr_payload(wire.data(),
        (size_t)(CS_HRDETAIL_MAX_SAMPLES + 1) * CS_HRDETAIL_SAMPLE_BYTES,
        key, iv, junk),
        "hr-detail: oversized payload rejected");

    // Apply: exact luma replacement at anchor positions, OOB skipped.
    // Neutral-gray canvas admits every Y (no gamut clipping).
    cv::Mat canvas(24, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    const std::vector<CsHrSample> put = { {0, 200}, {100, 50}, {(uint32_t)(24 * 32 - 1), 128},
        {(uint32_t)(24 * 32 + 5), 200} };
    const size_t applied = cs_apply_hr_detail(canvas, put);
    check(applied == 3, "hr-detail: apply skips out-of-range");
    cv::Mat canvas_ycc;
    cv::cvtColor(canvas, canvas_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> canvas_chs;
    cv::split(canvas_ycc, canvas_chs);
    check(canvas_chs[0].at<uint8_t>(0, 0) == 200 &&
        canvas_chs[0].at<uint8_t>(100 / 32, 100 % 32) == 50 &&
        canvas_chs[0].at<uint8_t>(23, 31) == 128,
        "hr-detail: apply sets exact luma");

    // Container roundtrips (BGR mode 4 + YCC420 mode 9) through the real
    // encrypt/decrypt: mode header, parsed sample count, and exact anchor
    // values on the final output.
    const std::string input_path = ".cs_test_hr_input.png";
    const std::string output_path = ".cs_test_hr_output.png";
    const std::string password = "hr-detail-roundtrip-password";
    const cv::Mat original = make_test_image(160, 120);
    check(cv::imwrite(input_path, original), "hr-detail: write source image");
    cv::Mat original_gray;
    cv::cvtColor(original, original_gray, cv::COLOR_BGR2GRAY);
    for (int splitmode : {0, 1}) {
        const bool ycc = (splitmode == 1);
        const char* tag = ycc ? "hr-detail ycc420" : "hr-detail bgr";
        const int want_mode = ycc ? CS_MODE_YCC420_HRDETAIL : CS_MODE_BGR_HRDETAIL;
        cv::Mat encrypted;
        const int encrypt_rc = encrypt_image::encrypt_image_tiled(
            input_path, "", password, 0.5f, 0, &encrypted,
            false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f,
            ycc, false, 8, 0, 0, false, false, "", 48);
        check(encrypt_rc == 0, (std::string(tag) + ": encrypt").c_str());
        if (encrypt_rc != 0) continue;
        decrypt_image container(encrypted, password);
        check(container.get_sampling_mode() == want_mode,
            (std::string(tag) + ": mode header").c_str());
        if (ycc) {
            cv::Mat y, ym, cr, crm, cb, cbm;
            container.get_sampled_ycc420(y, ym, cr, crm, cb, cbm);
            check(container.get_hr_samples().size() == 48,
                (std::string(tag) + ": payload parsed").c_str());
        } else {
            cv::Mat sm, mm;
            container.get_sampled_mat(sm, mm);
            check(container.get_hr_samples().size() == 48,
                (std::string(tag) + ": payload parsed").c_str());
        }
        CSencryption::params = AUTO_PARAM;
        size_t admitted = 0;
        CsPhotoUpscalerOptions no_up;
        const int decrypt_rc = decrypt_image::decrypt_image_tiled(
            encrypted, output_path, password, 1, 8, 3, 2, 0.01f, false, 0.0f,
            false, CS_SOLVER_FISTA, 0, 2, CS_BASIS_DCT, 2.0f, no_up,
            false, false, 0.0f, false, &admitted);
        check(decrypt_rc == 0, (std::string(tag) + ": decrypt").c_str());
        const cv::Mat decoded = cv::imread(output_path, cv::IMREAD_COLOR);
        check(!decoded.empty(), (std::string(tag) + ": output readable").c_str());
        if (!decoded.empty() && decoded.size() == original.size()) {
            cv::Mat dec_ycc;
            cv::cvtColor(decoded, dec_ycc, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> dec_chs;
            cv::split(dec_ycc, dec_chs);
            // Admitted anchors land within conversion noise; gamut-clipped
            // ones are reverted by construction, so the drift-free count
            // must match the decrypt-reported admitted count exactly.
            int clean = 0, worst = 0;
            for (const CsHrSample& s : container.get_hr_samples()) {
                if (s.idx >= (uint32_t)(decoded.cols * decoded.rows)) continue;
                const int v = dec_chs[0].at<uint8_t>(s.idx / decoded.cols, s.idx % decoded.cols);
                const int d = std::abs(v - (int)s.y);
                worst = (std::max)(worst, d);
                if (d <= 1) ++clean;
            }
            std::printf("       %s anchors admitted: %d/%d, worst drift: %d\n",
                tag, clean, (int)container.get_hr_samples().size(), worst);
            check(admitted > 0 && clean == (int)admitted,
                (std::string(tag) + ": anchors exact on output").c_str());
        }
        std::remove(output_path.c_str());
    }
    // Prototype scope guard: non-uniform sampling refuses the payload.
    {
        cv::Mat enc_bad;
        check(encrypt_image::encrypt_image_tiled(
            input_path, "", password, 0.5f, 64, &enc_bad,
            true, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f,
            false, false, 8, 0, 0, false, false, "", 48) != 0,
            "hr-detail: adaptive combination refused");
    }
    std::remove(input_path.c_str());
}

// ---------------------------------------------------------------------------
// HR anchors inside the CS-SR solve (--hr-weight): exact-HR data terms pull
// anchor pixels AND their neighborhoods, beyond post-hoc replacement.
// ---------------------------------------------------------------------------
static double hr_anchor_mean_err(const cv::Mat& hr_bgr, const cv::Mat& truth_gray,
    const std::vector<CsHrSample>& anchors, int stride, bool centers_only) {
    cv::Mat ycc;
    cv::cvtColor(hr_bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    double sum = 0.0;
    int cnt = 0;
    const int W = chs[0].cols, H = chs[0].rows;
    for (const CsHrSample& s : anchors) {
        const int gx = (int)(s.idx % (uint32_t)stride);
        const int gy = (int)(s.idx / (uint32_t)stride);
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (centers_only && (dx != 0 || dy != 0)) continue;
                if (!centers_only && dx == 0 && dy == 0) continue;
                const int x = gx + dx, y = gy + dy;
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                if (x >= truth_gray.cols || y >= truth_gray.rows) continue;
                sum += std::abs((int)chs[0].at<uint8_t>(y, x) -
                    (int)truth_gray.at<uint8_t>(y, x));
                ++cnt;
            }
        }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

void test_hr_sr_anchors() {
    // Same synthetic rig as test_cs_super_resolution: true 80x80 HR,
    // 40x40 measured LR, biased solve, full-grid packed container.
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
    // True-HR anchors at spread positions, plus one out-of-tile anchor
    // (exercises the skip path).
    const int pts[8][2] = {{10,10},{30,12},{55,20},{70,33},{15,45},{40,60},{62,70},{25,25}};
    std::vector<CsHrSample> anchors;
    for (int k = 0; k < 8; ++k) {
        CsHrSample s;
        s.idx = (uint32_t)(pts[k][1] * 80 + pts[k][0]);
        s.y = orig_gray.at<uint8_t>(pts[k][1], pts[k][0]);
        anchors.push_back(s);
    }
    anchors.push_back({(uint32_t)(200 * 80 + 200), 128});
    CsHrSrCtx ctx;
    ctx.anchors = &anchors;
    ctx.stride = 80;
    ctx.ox = 0; ctx.oy = 0;
    ctx.weight = 1.0f;

    cv::Mat plain, anchored;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, plain);
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, anchored, 0.0f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, ctx);
    check(anchored.size() == original.size() && anchored.type() == CV_8UC3,
        "hr-sr: anchored solve keeps 2x geometry and color type");
    // Empty anchors: bit-identical to the unanchored call.
    CsHrSrCtx empty;
    cv::Mat nohr;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, nohr, 0.0f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, empty);
    check(cv::norm(nohr, plain, cv::NORM_INF) == 0,
        "hr-sr: empty anchors are bit-identical to plain");
    // Weight 0: likewise identical (post-hoc replacement only).
    CsHrSrCtx zero = ctx;
    zero.weight = 0.0f;
    cv::Mat zerow;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, zerow, 0.0f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, zero);
    check(cv::norm(zerow, plain, cv::NORM_INF) == 0,
        "hr-sr: zero weight is bit-identical to plain");
    // Same tile-local mapping framed in a larger canvas (stride 160,
    // origin 40,40): bit-identical output proves origin/stride plumbing.
    std::vector<CsHrSample> reframed;
    for (const CsHrSample& s : anchors) {
        const int gx = (int)(s.idx % 80u), gy = (int)(s.idx / 80u);
        CsHrSample r;
        r.idx = (uint32_t)((gy + 40) * 160 + (gx + 40));
        r.y = s.y;
        reframed.push_back(r);
    }
    CsHrSrCtx ctx2;
    ctx2.anchors = &reframed;
    ctx2.stride = 160;
    ctx2.ox = 40; ctx2.oy = 40;
    ctx2.weight = 1.0f;
    cv::Mat refr;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f, 0.0f,
        8, 2, 0, CS_BASIS_DCT, 2.0f, refr, 0.0f, "nlmeans",
        CS_DNCNN_DEFAULT_MODEL, false, "", 1.0f, ctx2);
    check(cv::norm(refr, anchored, cv::NORM_INF) == 0,
        "hr-sr: reframed anchors are bit-identical");
    // The actual claim: anchors AND their neighborhoods improve.
    if (!anchored.empty() && !plain.empty()) {
        const double plain_c = hr_anchor_mean_err(plain, orig_gray, anchors, 80, true);
        const double anchored_c = hr_anchor_mean_err(anchored, orig_gray, anchors, 80, true);
        const double plain_n = hr_anchor_mean_err(plain, orig_gray, anchors, 80, false);
        const double anchored_n = hr_anchor_mean_err(anchored, orig_gray, anchors, 80, false);
        std::printf("       hr-sr center err: %.3f -> %.3f, neighbor err: %.3f -> %.3f\n",
            plain_c, anchored_c, plain_n, anchored_n);
        check(anchored_c < plain_c, "hr-sr: anchor pixels improve in-solve");
        check(anchored_n < plain_n, "hr-sr: anchor neighborhoods improve in-solve");
    }
}

// ---------------------------------------------------------------------------
// --enhance input planes (enh_net.cpp): the decoder-side maps the restoration
// CNN was trained with (x64/GpuBuild/enh_inputs.py), on a hand-made case.
// ---------------------------------------------------------------------------
void test_enh_planes() {
    const int H = 6, W = 8;
    const cv::Mat bgr(H, W, CV_8UC3, cv::Scalar(100, 100, 100));   // luma 100 everywhere
    cv::Mat M = cv::Mat::zeros(H / 2, W / 2, CV_8U), S = cv::Mat::zeros(H / 2, W / 2, CV_8U);
    M.at<uint8_t>(0, 0) = 1; S.at<uint8_t>(0, 0) = 108;        // residual (108 - 100) / 8 = 1
    M.at<uint8_t>(1, 2) = 1; S.at<uint8_t>(1, 2) = 96;         // residual -0.5
    std::vector<CsHrSample> anchors(3);
    anchors[0].idx = 1 * W + 1; anchors[0].y = 164;            // residual (164 - 100) / 64 = 1
    anchors[1].idx = 1 * W + 2; anchors[1].y = 255;            // 155 / 64 clipped to 1
    anchors[2].idx = H * W + 5; anchors[2].y = 7;              // out of range: skipped
    CsEnhInputs in; in.y_mask = &M; in.y_samp = &S; in.anchors = &anchors;
    std::vector<cv::Mat> P;
    cs_enh_planes(bgr, in, 5, P);
    bool shapes = P.size() == 5;
    for (const cv::Mat& p : P) shapes = shapes && p.type() == CV_32F && p.rows == H && p.cols == W;
    check(shapes, "enh-planes: 5 float planes at the image size");
    if (!shapes) return;
    const auto at = [&](int k, int r, int c) { return P[(size_t)k].at<float>(r, c); };
    check(cv::sum(P[1])[0] == 2.0 && at(1, 1, 1) == 1.0f && at(1, 1, 2) == 1.0f, "enh-planes: anchor indicator, out-of-range index skipped");
    check(std::fabs(at(2, 1, 1) - 1.0f) < 1e-5f && at(2, 1, 2) == 1.0f && at(2, 0, 0) == 0.0f, "enh-planes: anchor residual / 64, clipped");
    bool blocks = true;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            blocks = blocks && at(3, r, c) == 1.0f && std::fabs(at(4, r, c) - 1.0f) < 1e-5f &&
                     at(3, 2 + r, 4 + c) == 1.0f && std::fabs(at(4, 2 + r, 4 + c) + 0.5f) < 1e-5f;
    check(blocks && cv::sum(P[3])[0] == 8.0 && at(4, 4, 6) == 0.0f, "enh-planes: LR sample mask/residual on 2x2 blocks");
    double dmin, dmax;
    cv::minMaxLoc(P[0], &dmin, &dmax);
    check(dmin >= -0.5 && dmax <= 0.5 && dmax > dmin, "enh-planes: density centred in [-0.5, 0.5]");
    std::vector<cv::Mat> P3;
    cs_enh_planes(bgr, in, 3, P3);
    check(P3.size() == 3 && cv::norm(P3[2], P[2], cv::NORM_INF) == 0.0, "enh-planes: 3-plane variant is the first 3");
    cv::Mat img = bgr.clone();
    check(!cs_enhance_net(img, "no_such_model.onnx", in) && cv::norm(img, bgr, cv::NORM_INF) == 0.0,
          "enhance: missing model leaves the image untouched");
}
