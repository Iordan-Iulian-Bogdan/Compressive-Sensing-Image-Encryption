// Backend tests: FISTA/joint solvers, DCT and wavelet bases, auto-reg, GPU agreement.
#include "test_common.hpp"

// ---------------------------------------------------------------------------
// 4b. proximal solvers (reweighted-L1 FISTA + SOMP joint)
// ---------------------------------------------------------------------------
void test_fista_solvers() {
    int s = -1;
    check(cs_solver_from_name("fista", s) == 0 && s == CS_SOLVER_FISTA, "fista: fista parses to 1");
    check(cs_solver_from_name("joint", s) == 0 && s == CS_SOLVER_FISTA_JOINT, "fista: joint parses to 2");
    check(cs_solver_from_name("somp", s) == 0 && s == CS_SOLVER_FISTA_JOINT, "fista: somp aliases joint");
    check(cs_solver_from_name("owlqn", s) != 0, "fista: removed owlqn name rejected");
    check(cs_solver_from_name("admm", s) != 0, "fista: removed admm name rejected");
    check(cs_solver_from_name("bogus", s) != 0, "fista: unknown solver name rejected");

    const std::string password = "fista-test-password";
    cv::Mat original = make_test_image(320, 240);
    const std::string tmp_in = ".cs_test_fista_in.png";
    const std::string tmp_out = ".cs_test_fista_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 1.0f, 0, &encrypted) != 0) {
        check(false, "fista: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }

    const int solvers[2] = { CS_SOLVER_FISTA, CS_SOLVER_FISTA_JOINT };
    const char* names[2] = { "fista", "joint" };
    for (int k = 0; k < 2; ++k) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "fista: %s decrypt succeeds", names[k]);
        const int rc = decrypt_image::decrypt_image_tiled(
            encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false, solvers[k]);
        check(rc == 0, tag);
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(tag, sizeof(tag), "fista: %s output readable", names[k]);
        check(!dec.empty(), tag);
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       fista %s PSNR: %.2f dB\n", names[k], p);
            std::snprintf(tag, sizeof(tag), "fista: %s PSNR finite and sane", names[k]);
            check(std::isfinite(p) && p > 12.0, tag);
        }
    }

    // native FISTA controls: few steps + single pass must still solve sanely
    {
        const int rc = decrypt_image::decrypt_image_tiled(
            encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false,
            CS_SOLVER_FISTA, 8, 1);
        check(rc == 0, "fista: fista-iters/reweights override decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "fista: override output readable");
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       fista override (8 iters, 1 pass) PSNR: %.2f dB\n", p);
            check(std::isfinite(p) && p > 10.0, "fista: override PSNR finite and sane");
        }
    }
    // under compression (ratio 0.5): FISTA vs joint on the same container --
    // the regime where the data Hessian only fills half the grid
    {
        cv::Mat enc_half;
        if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc_half) != 0) {
            check(false, "fista: ratio-0.5 encrypt");
        } else {
            const int half[2] = { CS_SOLVER_FISTA, CS_SOLVER_FISTA_JOINT };
            const char* hnames[2] = { "fista", "joint" };
            for (int k = 0; k < 2; ++k) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "fista: %s ratio-0.5 decrypt succeeds", hnames[k]);
                const int rc = decrypt_image::decrypt_image_tiled(
                    enc_half, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false, half[k]);
                check(rc == 0, tag);
                cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
                std::snprintf(tag, sizeof(tag), "fista: %s ratio-0.5 output readable", hnames[k]);
                check(!dec.empty(), tag);
                if (!dec.empty()) {
                    cv::Mat sized;
                    cv::resize(dec, sized, original.size());
                    const double p = cs_quality::psnr(original, sized);
                    std::printf("       %s ratio 0.5 PSNR: %.2f dB\n", hnames[k], p);
                    std::snprintf(tag, sizeof(tag), "fista: %s ratio-0.5 PSNR sane", hnames[k]);
                    check(std::isfinite(p) && p > 12.0, tag);
                }
            }
        }
    }

    // invalid values rejected before the solve
    check(decrypt_image::decrypt_image_tiled(
        encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false,
        CS_SOLVER_FISTA, 501, 2) == -1, "fista: fista-iters > 500 rejected");
    check(decrypt_image::decrypt_image_tiled(
        encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false,
        99, 0, 2) == -1, "fista: unknown solver id rejected");

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 4c. CDF 9/7 wavelet basis (lifting DWT roundtrip + FISTA solves)
// ---------------------------------------------------------------------------
void test_dct_adjoint() {
    // The FISTA gradient assumes S^T = forward DCT for S = inverse DCT.
    // If cv::dct is not orthonormal, the solve fixed point drifts off the
    // data even at full sampling. Check S^T S = I and <S a, b> = <a, S^T b>.
    // Same operations as cs_synth (cv::dct inverse) and cs_synth_adj (cv::dct forward).
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    cv::Mat a(16, 24, CV_32F), bvec(16, 24, CV_32F);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) { a.at<float>(i, j) = gauss(rng); bvec.at<float>(i, j) = gauss(rng); }
    cv::Mat sa, sadj_b, rtv;
    cv::dct(a, sa, cv::DCT_INVERSE);
    cv::dct(bvec, sadj_b, 0);
    double lhs = 0.0, rhs = 0.0, rt = 0.0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) {
            lhs += (double)sa.at<float>(i, j) * bvec.at<float>(i, j);
            rhs += (double)a.at<float>(i, j) * sadj_b.at<float>(i, j);
        }
    cv::Mat s1, s2;
    cv::dct(a, s1, cv::DCT_INVERSE);
    cv::dct(s1, s2, 0);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) rt = std::max(rt, (double)std::fabs(s2.at<float>(i, j) - a.at<float>(i, j)));
    std::printf("       dct adjoint: <Sa,b>=%.6f <a,S^Tb>=%.6f roundtrip max err=%.2e\n", lhs, rhs, rt);
    check(std::fabs(lhs - rhs) < 1e-3 * (1.0 + std::fabs(lhs)), "dct: forward DCT is adjoint of inverse");
    check(rt < 1e-3, "dct: adjoint of synthesis inverts synthesis (S^T S = I)");
}

void test_block_dct() {
    // Block-DCT (--dct-block) must stay orthogonal per block, including
    // edge remainder blocks when dims are not multiples of B. Odd sizes
    // (70x66) exercise 6-wide and 2-wide remainders at B=16.
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int B : {8, 16, 32}) {
        cv::Mat a(70, 66, CV_32F), bvec(70, 66, CV_32F);
        for (int i = 0; i < 70; ++i)
            for (int j = 0; j < 66; ++j) { a.at<float>(i, j) = gauss(rng); bvec.at<float>(i, j) = gauss(rng); }
        cv::Mat sa = a.clone(), sb = bvec.clone();
        cs_dct_inverse(sa, B);
        cs_dct_forward(sb, B);
        double lhs = 0.0, rhs = 0.0;
        for (int i = 0; i < 70; ++i)
            for (int j = 0; j < 66; ++j) {
                lhs += (double)sa.at<float>(i, j) * bvec.at<float>(i, j);
                rhs += (double)a.at<float>(i, j) * sb.at<float>(i, j);
            }
        cv::Mat rtv = a.clone();
        cs_dct_inverse(rtv, B);
        cs_dct_forward(rtv, B);
        double rt = 0.0;
        for (int i = 0; i < 70; ++i)
            for (int j = 0; j < 66; ++j) rt = std::max(rt, (double)std::fabs(rtv.at<float>(i, j) - a.at<float>(i, j)));
        std::printf("       block-dct B=%d: adjoint err=%.2e roundtrip max err=%.2e\n",
            B, std::fabs(lhs - rhs) / (1.0 + std::fabs(lhs)), rt);
        check(std::fabs(lhs - rhs) < 1e-3 * (1.0 + std::fabs(lhs)), "block-dct: adjoint identity holds per block");
        check(rt < 1e-3, "block-dct: roundtrip inverts (incl. remainder blocks)");
    }
    // basis plumbing: block travels via cs_make_basis, wavelet unaffected
    cs_fista_basis bx = cs_make_basis(64, 64, CS_BASIS_DCT, 2.0f, 8);
    check(bx.dct_block == 8, "block-dct: explicit size stored in basis");
    cs_set_default_dct_block(16);
    check(cs_default_dct_block() == 16, "block-dct: global default settable");
    cs_fista_basis bx2 = cs_make_basis(64, 64, CS_BASIS_DCT, 2.0f);
    check(bx2.dct_block == 16, "block-dct: default picked up by make_basis");
    cs_fista_basis bxw = cs_make_basis(64, 64, CS_BASIS_CDF97, 2.0f);
    check(bxw.dct_block == 0, "block-dct: wavelet basis ignores block");
    cs_set_default_dct_block(0);
    check(cs_default_dct_block() == 0, "block-dct: default restored to whole-plane");
    // frequency ramp: DC x1, peak AC x wscale, uniform at wscale <= 1
    {
        cs_fista_basis bu = cs_make_basis(64, 48, CS_BASIS_DCT, 1.0f);
        bool ones = true;
        for (float v : bu.wscale) ones = ones && (v == 1.0f);
        check(ones, "dct-ramp: wscale=1 keeps uniform weights");
        cs_fista_basis bw = cs_make_basis(64, 48, CS_BASIS_DCT, 2.0f);
        check(bw.wscale[0] == 1.0f, "dct-ramp: DC protected (x1)");
        const float peak = bw.wscale[(size_t)63 * 48 + 47];
        check(std::fabs(peak - 2.0f) < 1e-5f, "dct-ramp: peak AC carries x wscale");
        bool mono = true;
        for (int i = 1; i < 64; ++i) mono = mono && (bw.wscale[(size_t)i * 48] >= bw.wscale[(size_t)(i - 1) * 48]);
        check(mono, "dct-ramp: weights rise monotonically with frequency");
    }
}

void test_wavelet_basis() {
    test_dct_adjoint();
    test_block_dct();
    int b = -1;
    check(cs_basis_from_name("dct", b) == 0 && b == CS_BASIS_DCT, "wavelet: dct parses to 0");
    check(cs_basis_from_name("wavelet", b) == 0 && b == CS_BASIS_CDF97, "wavelet: wavelet parses to 1");
    check(cs_basis_from_name("cdf97", b) == 0 && b == CS_BASIS_CDF97, "wavelet: cdf97 aliases wavelet");
    check(cs_basis_from_name("bogus", b) != 0, "wavelet: unknown basis name rejected");

    // perfect-reconstruction unit check on several sizes (even, odd, tiny,
    // non-square): forward + inverse must recover the input
    {
        cv::RNG rng(777);
        const int sizes[][2] = { {8,8}, {16,16}, {63,63}, {64,64}, {65,65}, {100,77}, {37,53}, {240,180} };
        bool ok = true;
        for (const auto& wh : sizes) {
            const int rows = wh[1], cols = wh[0];
            const int n = rows * cols;
            std::vector<float> v((size_t)n), ref((size_t)n);
            for (int i = 0; i < n; ++i) {
                v[(size_t)i] = ref[(size_t)i] = (float)rng.uniform(-1.0, 1.0);
            }
            const int lv = cs_dwt_levels(rows, cols);
            cs_dwt_forward(v.data(), rows, cols, lv);
            // transform must actually do something (energy spreads, DC in LL)
            cs_dwt_inverse(v.data(), rows, cols, lv);
            double worst = 0.0;
            for (int i = 0; i < n; ++i) {
                worst = (std::max)(worst, std::fabs((double)v[(size_t)i] - ref[(size_t)i]));
            }
            if (!(worst < 1e-3)) {
                std::printf("       wavelet roundtrip %dx%d worst err %.6f\n", cols, rows, worst);
                ok = false;
            }
        }
        check(ok, "wavelet: fwd+inv roundtrip < 1e-3 on all sizes");
        check(cs_dwt_levels(64, 64) == 3, "wavelet: 64px -> 3 levels");
        check(cs_dwt_levels(480, 360) == 4, "wavelet: 480px -> 4 levels (capped)");
    }

    // adjoint identity <S a, b> == <a, S^T b>: the FISTA gradient
    // differentiates through S^T, so this must hold or the solve freezes
    {
        cv::RNG rng(4242);
        const int sizes[][2] = { {64,64}, {100,77}, {37,53} };
        bool ok = true;
        for (const auto& wh : sizes) {
            const int rows = wh[1], cols = wh[0];
            const int n = rows * cols;
            const int lv = cs_dwt_levels(rows, cols);
            std::vector<float> a((size_t)n), b((size_t)n);
            for (int i = 0; i < n; ++i) {
                a[(size_t)i] = (float)rng.uniform(-1.0, 1.0);
                b[(size_t)i] = (float)rng.uniform(-1.0, 1.0);
            }
            std::vector<float> sa = a, stb = b;
            cs_dwt_inverse(sa.data(), rows, cols, lv);
            cs_dwt_synth_adjoint(stb.data(), rows, cols, lv);
            double lhs = 0.0, rhs = 0.0, nrm = 0.0;
            for (int i = 0; i < n; ++i) {
                lhs += (double)sa[(size_t)i] * b[(size_t)i];
                rhs += (double)a[(size_t)i] * stb[(size_t)i];
                nrm += (double)sa[(size_t)i] * sa[(size_t)i];
            }
            const double rel = std::fabs(lhs - rhs) / (std::sqrt(nrm * n) + 1e-9);
            if (!(rel < 1e-3)) {
                std::printf("       wavelet adjoint %dx%d rel err %.6f\n", cols, rows, rel);
                ok = false;
            }
        }
        check(ok, "wavelet: <Sa,b> == <a,S^Tb> within 1e-3");
    }

    // per-scale threshold ramp (cs_make_basis wscale vector): LL and the
    // coarsest detail band stay x1, the finest band gets x wscale; DCT
    // maps the knob to a DC(x1)-to-peak(x wscale) frequency ramp
    // (uniform only at wscale <= 1, the legacy behavior)
    {
        bool ok = true;
        // 64x64 -> 3 levels, LL is 8x8
        cs_fista_basis bx = cs_make_basis(64, 64, CS_BASIS_CDF97);
        ok = ok && (int)bx.wscale.size() == 64 * 64;
        ok = ok && std::fabs(bx.wscale[7 * 64 + 7] - 1.0f) < 1e-6f;  // LL corner
        ok = ok && std::fabs(bx.wscale[8 * 64] - 1.0f) < 1e-6f;      // coarsest detail band
        ok = ok && std::fabs(bx.wscale[63 * 64] - 2.0f) < 1e-5f;     // finest band (default 2.0)
        float mn = 1e30f, mx = -1e30f;
        for (float v : bx.wscale) { mn = (std::min)(mn, v); mx = (std::max)(mx, v); }
        ok = ok && mn >= 1.0f - 1e-6f && mx <= 2.0f + 1e-5f;
        // custom finest multiplier respected
        cs_fista_basis bx3 = cs_make_basis(64, 64, CS_BASIS_CDF97, 3.0f);
        ok = ok && std::fabs(bx3.wscale[63 * 64] - 3.0f) < 1e-4f;
        ok = ok && std::fabs(bx3.wscale[0] - 1.0f) < 1e-6f;
        // wscale 1.0 disables the ramp
        cs_fista_basis bx1 = cs_make_basis(64, 64, CS_BASIS_CDF97, 1.0f);
        for (float v : bx1.wscale) ok = ok && (std::fabs(v - 1.0f) < 1e-6f);
        // DCT maps the knob to a frequency ramp (DC x1, peak x wscale)
        cs_fista_basis bd = cs_make_basis(64, 64, CS_BASIS_DCT, 3.0f);
        ok = ok && (std::fabs(bd.wscale[0] - 1.0f) < 1e-6f);
        ok = ok && (std::fabs(bd.wscale[(size_t)63 * 64 + 63] - 3.0f) < 1e-4f);
        cs_fista_basis bd1 = cs_make_basis(64, 64, CS_BASIS_DCT, 1.0f);
        for (float v : bd1.wscale) ok = ok && (std::fabs(v - 1.0f) < 1e-6f);
        check(ok, "wavelet: per-scale ramp LL=1, finest=wscale, DCT frequency ramp");
    }

    // end-to-end: wavelet-basis FISTA solves decrypt sanely
    const std::string password = "wavelet-test-password";
    cv::Mat original = make_test_image(320, 240);
    const std::string tmp_in = ".cs_test_wavelet_in.png";
    const std::string tmp_out = ".cs_test_wavelet_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 1.0f, 0, &encrypted) != 0) {
        check(false, "wavelet: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }

    const int solvers[2] = { CS_SOLVER_FISTA, CS_SOLVER_FISTA_JOINT };
    const char* names[2] = { "fista", "joint" };
    for (int k = 0; k < 2; ++k) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "wavelet: %s+wavelet decrypt succeeds", names[k]);
        const int rc = decrypt_image::decrypt_image_tiled(
            encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, 0.0f, false,
            solvers[k], 0, 2, CS_BASIS_CDF97);
        check(rc == 0, tag);
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(tag, sizeof(tag), "wavelet: %s+wavelet output readable", names[k]);
        check(!dec.empty(), tag);
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       wavelet %s PSNR: %.2f dB\n", names[k], p);
            std::snprintf(tag, sizeof(tag), "wavelet: %s+wavelet PSNR finite and sane", names[k]);
            check(std::isfinite(p) && p > 12.0, tag);
        }
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

void test_auto_reg_params() {
    // Params-test fitted laws (single pin each; full curves in results.csv):
    // seg 0.045*r^-0.75, uniform 0.01875/r, tile factor (n/64)^0.15 in
    // [0.5, 1.2] below ratio 0.5, tv 0.01 for seg below 0.5 else 0.
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_SEGMASK, 0.3f, 64) - 0.1112f) < 0.002f,
        "auto-reg: seg 0.3 t64 ~= 0.111");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_SEGMASK, 0.15f, 1) - 0.103f) < 0.004f,
        "auto-reg: seg 0.15 t1 ~= 0.103 (tile factor)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_RANDOM, 0.5f, 64) - 0.0375f) < 1e-6f,
        "auto-reg: uniform 0.5 t64 == 0.0375 (legacy formula)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_SEGMASK, 0.5f, 1) - 0.0418f) < 0.002f,
        "auto-reg: seg 0.5 t1 tile-factored (no-factor would 2x overshoot)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_RANDOM, 0.5f, 1) - 0.0375f) < 1e-6f,
        "auto-reg: uniform ignores tile factor at 0.5");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_RANDOM, 0.3f, 1) - 0.0335f) < 0.002f,
        "auto-reg: uniform tile-factored below 0.5");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_RANDOM, 0.15f, 64) - 0.125f) < 0.002f,
        "auto-reg: uniform 0.15 t64 == 0.125 (extended below 0.5)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC420, 0.3f, 64) - 0.075f) < 0.002f,
        "auto-reg: YCC420 covered (0.0225/r)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC420, 0.75f, 64) - 0.03f) < 0.002f,
        "auto-reg: YCC420 covered above 0.5");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC422, 0.25f, 64) - 0.09f) < 0.002f,
        "auto-reg: YCC422 covered (0.0225/r)");
    check(cs_auto_reg_coefficient(CS_MODE_ADAPTIVE, 0.3f, 64) < 0.0f,
        "auto-reg: adaptive uncovered");
    check(cs_auto_reg_tv(CS_MODE_SEGMASK, 0.3f) == 0.01f, "auto-reg: seg tv 0.01 below 0.5");
    check(cs_auto_reg_tv(CS_MODE_SEGMASK, 0.5f) == 0.0f, "auto-reg: seg tv 0 at 0.5");
    check(cs_auto_reg_tv(CS_MODE_RANDOM, 0.15f) == 0.0f, "auto-reg: uniform tv 0");
    check(cs_auto_reg_tv(CS_MODE_YCC420, 0.15f) < 0.0f, "auto-reg: YCC tv uncovered");
}

// ---------------------------------------------------------------------------
// GPU/CPU FISTA agreement (--device gpu): without a HIP device,
// set_enabled(true) stays off and both runs take the CPU path (trivially
// identical); on GPU machines this measures real device-vs-host agreement.
// Always restores CPU-only afterwards so later tests (none, this runs
// last) and profiling stay deterministic.
// ---------------------------------------------------------------------------
void test_gpu_fista_agreement() {
    const std::string password = "gpu-fista-test-password";
    const std::string tmp_in = ".cs_test_gpu_in.png";
    const std::string tmp_cpu = ".cs_test_gpu_cpu.png";
    const std::string tmp_gpu = ".cs_test_gpu_gpu.png";
    cv::Mat original = make_test_image(96, 72);
    cv::imwrite(tmp_in, original);
    cv::Mat enc;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc,
        false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8) == 0, "gpu-fista: encrypt");
    const int solvers[2] = { CS_SOLVER_FISTA, CS_SOLVER_FISTA_JOINT };
    const char* names[2] = { "fista", "joint" };
    for (int s = 0; s < 2; ++s) {
        cs_gpu::set_enabled(false);
        check(decrypt_image::decrypt_image_tiled(enc, tmp_cpu, password,
            1, 0, 5, 2, 0.01f, false, 0.0f, false,
            solvers[s], 0, 1, CS_BASIS_DCT, 2.0f,
            CsPhotoUpscalerOptions(), false, false) == 0, "gpu-fista: cpu decrypt");
        cs_gpu::set_enabled(true);
        check(decrypt_image::decrypt_image_tiled(enc, tmp_gpu, password,
            1, 0, 5, 2, 0.01f, false, 0.0f, false,
            solvers[s], 0, 1, CS_BASIS_DCT, 2.0f,
            CsPhotoUpscalerOptions(), false, false) == 0, "gpu-fista: gpu decrypt");
        cs_gpu::set_enabled(false);
        cv::Mat dec_cpu = cv::imread(tmp_cpu, cv::IMREAD_COLOR);
        cv::Mat dec_gpu = cv::imread(tmp_gpu, cv::IMREAD_COLOR);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "gpu-fista: %s cpu/gpu agreement", names[s]);
        if (!dec_cpu.empty() && !dec_gpu.empty() && dec_cpu.size() == dec_gpu.size()) {
            const double p = cs_quality::psnr(dec_cpu, dec_gpu);
            std::printf("       gpu-fista %s agreement: %.2f dB\n", names[s], p);
            check(p > 50.0, tag);
        } else {
            check(false, tag);
        }
        std::remove(tmp_cpu.c_str());
        std::remove(tmp_gpu.c_str());
    }
    std::remove(tmp_in.c_str());
}

