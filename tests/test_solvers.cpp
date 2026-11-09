// Backend tests: DCT transforms, auto-reg, GPU agreement.
#include "test_common.hpp"

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
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC420, 0.3f, 64) - 0.0375f) < 0.002f,
        "auto-reg: YCC420 covered (0.01125/r)");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC420, 0.75f, 64) - 0.015f) < 0.002f,
        "auto-reg: YCC420 covered above 0.5");
    check(std::fabs(cs_auto_reg_coefficient(CS_MODE_YCC422, 0.25f, 64) - 0.045f) < 0.002f,
        "auto-reg: YCC422 covered (0.01125/r)");
    check(cs_auto_reg_coefficient(CS_MODE_ADAPTIVE, 0.3f, 64) < 0.0f,
        "auto-reg: adaptive uncovered");
    check(cs_auto_reg_tv(CS_MODE_SEGMASK, 0.3f) == 0.01f, "auto-reg: seg tv 0.01 below 0.5");
    check(cs_auto_reg_tv(CS_MODE_SEGMASK, 0.5f) == 0.0f, "auto-reg: seg tv 0 at 0.5");
    check(cs_auto_reg_tv(CS_MODE_RANDOM, 0.15f) == 0.0f, "auto-reg: uniform tv 0");
    check(cs_auto_reg_tv(CS_MODE_YCC420, 0.15f) < 0.0f, "auto-reg: YCC tv uncovered");
}

// ---------------------------------------------------------------------------
// GPU/CPU agreement of the production decode (--device gpu): without a HIP
// device, set_enabled(true) stays off and both runs take the CPU path
// (trivially identical); on GPU machines this measures real device-vs-host
// agreement. Always restores CPU-only afterwards so profiling stays
// deterministic.
// ---------------------------------------------------------------------------
void test_gpu_fista_agreement() {
    const std::string password = "gpu-fista-test-password";
    const std::string tmp_in = ".cs_test_gpu_in.png";
    const std::string tmp_enc = ".cs_test_gpu_in.enc.png";
    const std::string tmp_cpu = ".cs_test_gpu_cpu.png";
    const std::string tmp_gpu = ".cs_test_gpu_gpu.png";
    cv::Mat original = make_test_image(96, 72);
    cv::imwrite(tmp_in, original);
    check(prod_encrypt(tmp_in, tmp_enc, password, 0.5f) == 0, "gpu-fista: encrypt");
    const cv::Mat enc = read_container(tmp_enc);
    cs_gpu::set_enabled(false);
    check(prod_decrypt(enc, tmp_cpu, password) == 0, "gpu-fista: cpu decrypt");
    cs_gpu::set_enabled(true);
    check(prod_decrypt(enc, tmp_gpu, password) == 0, "gpu-fista: gpu decrypt");
    cs_gpu::set_enabled(false);
    cv::Mat dec_cpu = cv::imread(tmp_cpu, cv::IMREAD_COLOR);
    cv::Mat dec_gpu = cv::imread(tmp_gpu, cv::IMREAD_COLOR);
    if (!dec_cpu.empty() && !dec_gpu.empty() && dec_cpu.size() == dec_gpu.size()) {
        const double p = cs_quality::psnr(dec_cpu, dec_gpu);
        std::printf("       gpu-fista agreement: %.2f dB\n", p);
        check(p > 50.0, "gpu-fista: cpu/gpu agreement");
    } else {
        check(false, "gpu-fista: cpu/gpu agreement");
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_cpu.c_str());
    std::remove(tmp_gpu.c_str());
}
