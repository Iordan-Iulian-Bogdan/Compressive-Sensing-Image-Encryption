// Test suite for the CS image encryption backend.
// Covers: crypto primitives (KDF determinism, seal/verify, tamper detection),
// index shuffle determinism, tiling helpers, and a full encrypt->decrypt
// roundtrip scored by PSNR + SSIM.
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

namespace {

int g_failures = 0;

void check(bool ok, const char* name) {
    if (ok) {
        std::printf("[PASS] %s\n", name);
    }
    else {
        std::printf("[FAIL] %s\n", name);
        g_failures++;
    }
}

// procedural test image: smooth gradient + shapes (structure, no noise)
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

// ---------------------------------------------------------------------------
// 1. crypto primitives
// ---------------------------------------------------------------------------
void test_crypto() {
    uint8_t salt[CS_SALT_BYTES];
    check(cs_random_bytes(salt, CS_SALT_BYTES), "crypto: rng fills salt");

    uint8_t k1[32], k2[32], k3[32];
    const std::string pw = "test-passphrase-2026";
    check(cs_derive_key(pw, salt, k1), "crypto: PBKDF2 derive");
    check(cs_derive_key(pw, salt, k2), "crypto: PBKDF2 derive (repeat)");
    check(std::memcmp(k1, k2, 32) == 0, "crypto: same password+salt -> same key (seed determinism)");

    uint8_t salt2[CS_SALT_BYTES];
    std::memcpy(salt2, salt, CS_SALT_BYTES);
    salt2[0] ^= 1;
    check(cs_derive_key(pw, salt2, k3) && std::memcmp(k1, k3, 32) != 0, "crypto: different salt -> different key");

    const std::string meta = cs_build_metadata(1234567, 2016, 1512, 3024, 4032);
    check(meta.size() == CS_METADATA_BYTES, "crypto: metadata is 32 bytes");

    std::vector<uint8_t> buf(CS_HEADER_BYTES + 900);
    check(cs_write_header(buf.data(), buf.size(), meta, k1, salt), "crypto: v2 header write");
    check(buf[CS_OFF_VERSION] == 2, "crypto: v2 version byte");
    check(std::memcmp(buf.data() + CS_OFF_META, meta.data(), CS_METADATA_BYTES) != 0,
        "crypto: metadata ciphertext differs from plaintext");
    check(cs_build_metadata(1234567, 2016, 1512, 3024, 4032) ==
        cs_build_metadata(1234567, 2016, 1512, 3024, 4032), "crypto: metadata build deterministic");

    for (size_t i = CS_HEADER_BYTES; i < buf.size(); i++) buf[i] = (uint8_t)(i * 31);
    int tag_set = 0;
    for (int i = 0; i < CS_TAG_BYTES; i++) tag_set |= buf[CS_OFF_TAG + i];
    check(tag_set == 0, "crypto: tag zero before seal");
    check(cs_seal_header(buf.data(), buf.size(), k1), "crypto: seal");
    tag_set = 0;
    for (int i = 0; i < CS_TAG_BYTES; i++) tag_set |= buf[CS_OFF_TAG + i];
    check(tag_set != 0, "crypto: tag set after seal");

    CsHeaderInfo info;
    check(cs_parse_header(buf.data(), buf.size(), pw, info) &&
        info.version == 2 && info.key_valid && !info.auth_failed &&
        info.m == 1234567 && info.rows == 2016 && info.cols == 1512 &&
        info.org_h == 3024 && info.org_w == 4032, "crypto: v2 parse roundtrip with password");

    CsHeaderInfo bad;
    check(!cs_parse_header(buf.data(), buf.size(), "wrong password 12", bad) && bad.auth_failed,
        "crypto: wrong password rejected");
    buf[buf.size() - 1] ^= 1;
    check(!cs_parse_header(buf.data(), buf.size(), pw, bad) && bad.auth_failed,
        "crypto: tampered body rejected");
    buf[buf.size() - 1] ^= 1;
    buf[CS_OFF_SALT] ^= 1;
    check(!cs_parse_header(buf.data(), buf.size(), pw, bad) && bad.auth_failed,
        "crypto: tampered header rejected");
    buf[CS_OFF_SALT] ^= 1;
    check(!cs_parse_header(buf.data(), buf.size(), "", bad), "crypto: v2 without password rejected");

    std::vector<uint8_t> tile_buf(CS_HEADER_BYTES);
    cs_write_header_plain(tile_buf.data(), tile_buf.size(), meta);
    CsHeaderInfo tile_info;
    check(cs_parse_header(tile_buf.data(), tile_buf.size(), "", tile_info) &&
        tile_info.version == 1 && tile_info.m == 1234567 && !tile_info.key_valid,
        "crypto: v1 tile header parses without password");

    std::vector<uint8_t> legacy(96, (uint8_t)'-');
    check(!cs_parse_header(legacy.data(), legacy.size(), "x", bad) && bad.legacy,
        "crypto: legacy container detected");

    bool threw = false;
    try { cs_build_metadata(999999999, 99999, 99999, 999999, 999999); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw, "crypto: oversized metadata throws");

    cs_wipe(k1, sizeof(k1)); cs_wipe(k2, sizeof(k2)); cs_wipe(k3, sizeof(k3));
}

// ---------------------------------------------------------------------------
// 2. shuffle determinism / reversal
// ---------------------------------------------------------------------------
void test_shuffle() {
    std::vector<int> orig(5000);
    std::iota(orig.begin(), orig.end(), 0);

    std::vector<int> a = orig, b = orig;
    shuffle(a, 12345u);
    shuffle(b, 12345u);
    check(a == b, "shuffle: same seed -> same permutation");
    check(a != orig, "shuffle: permutation actually applied");

    reverseShuffle(a, 12345u);
    check(a == orig, "shuffle: reverseShuffle restores original order");
}

// ---------------------------------------------------------------------------
// 3. tile grid helpers
// ---------------------------------------------------------------------------
void test_tile_helpers() {
    const std::vector<std::string> order = spiralOrder(4);
    check(order.size() == 16, "spiralOrder: 16 entries for 4x4 grid");
    std::set<std::string> uniq(order.begin(), order.end());
    check(uniq.size() == 16, "spiralOrder: all entries unique");

    // reconstructImage placement
    const int ts = 8, gap = 0;
    std::vector<std::vector<cv::Mat>> tiles(2, std::vector<cv::Mat>(2));
    std::vector<std::vector<TileCoord>> coords(2, std::vector<TileCoord>(2));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            tiles[i][j] = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(uchar(20 + 40 * j), uchar(30 + 40 * i), uchar(100 + i * j)));
            coords[i][j] = { j * (ts + gap), i * (ts + gap) };
        }
    }
    cv::Mat re = reconstructImage(tiles, coords);
    check(!re.empty() && re.cols == 2 * ts && re.rows == 2 * ts, "reconstructImage: output size");
    bool placed = true;
    for (int i = 0; i < 2 && placed; i++) {
        for (int j = 0; j < 2 && placed; j++) {
            const cv::Vec3b expect = tiles[i][j].at<cv::Vec3b>(0, 0);
            const cv::Vec3b got = re.at<cv::Vec3b>(coords[i][j].y, coords[i][j].x);
            placed = got == expect;
        }
    }
    check(placed, "reconstructImage: tiles placed at coordinates");

    // blend: alpha=1 -> tile wins
    cv::Mat target = cv::Mat(2 * ts, 2 * ts, CV_8UC3, cv::Scalar(10, 10, 10));
    cv::Mat blended = blendTilesWithImage(tiles, coords, target, 1.0f);
    check(blended.at<cv::Vec3b>(0, 0) == tiles[0][0].at<cv::Vec3b>(0, 0), "blend: alpha=1 takes tile value");

    // blend: two identical solid tiles stacked at same coordinate -> half of each channel
    std::vector<std::vector<cv::Mat>> t2(1, std::vector<cv::Mat>(1));
    std::vector<std::vector<TileCoord>> c2(1, std::vector<TileCoord>(1));
    t2[0][0] = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(100, 100, 100));
    c2[0][0] = { 0, 0 };
    cv::Mat base = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::Mat out = blendTilesWithImage(t2, c2, base, 0.5f);
    const cv::Vec3b px = out.at<cv::Vec3b>(0, 0);
    check(std::abs((int)px[0] - 60) <= 1, "blend: alpha=0.5 averages overlapping tiles");
}

// ---------------------------------------------------------------------------
// 4. full roundtrip + PSNR + auth rejection
// ---------------------------------------------------------------------------
void test_roundtrip() {
    const std::string password = "roundtrip-test-password";
    const int W = 960, H = 720;
    cv::Mat original = make_test_image(W, H);

cv::Mat encrypted;
    const int enc_rc = encrypt_image::encrypt_image_tiled("", "", password, 1.0f, 0, &encrypted);
    check(enc_rc != 0, "roundtrip: missing input file rejected");

    const std::string tmp_in = ".cs_test_input.png";
    const std::string tmp_out = ".cs_test_output.png";
    cv::Mat encrypted2;
    cv::imwrite(tmp_in, original);
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 1.0f, 0, &encrypted2) == 0,
        "roundtrip: encrypt to memory");

    // wrong password must fail authentication before the solve
    const int wrong_rc = decrypt_image::decrypt_image_tiled(encrypted2, tmp_out, "wrong-password-99", 24, 24, 5, 4, 0.01f, false);
    check(wrong_rc == -2, "roundtrip: wrong password rejected with -2");

    // tamper: flip one byte in the container body
    cv::Mat tampered = encrypted2.clone();
    tampered.data[tampered.total() * 3 - 1] ^= 0x80;
    const int tamper_rc = decrypt_image::decrypt_image_tiled(tampered, tmp_out, password, 24, 24, 5, 4, 0.01f, false);
    check(tamper_rc == -2, "roundtrip: tampered container rejected with -2");

    // correct password: full solve
    check(decrypt_image::decrypt_image_tiled(encrypted2, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
        "roundtrip: decrypt succeeds");

    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "roundtrip: decrypted file readable");

    // output is 2x the original size (solve -> org_size -> 2x upscale)
    cv::Mat dec_sized;
    cv::resize(decrypted, dec_sized, original.size());
    const double p = cs_quality::psnr(original, dec_sized);
    const double s = cs_quality::ssim(original, dec_sized);
    std::printf("       roundtrip PSNR: %.2f dB\n", p);
    std::printf("       roundtrip SSIM: %.4f\n", s);
    // Regression guard, not a quality benchmark: with wavefront neighbor
    // warm-starts the synthetic roundtrip measures ~26.8 dB today (19.2 dB
    // with the old independent-solve order). A broken index pipeline
    // (wrong sampling, misaligned offsets) collapses to single-digit dB,
    // so 24 dB leaves ~2.7 dB of headroom for noise/rounding variance.
    check(p > 24.0, "roundtrip: PSNR > 24 dB");
    // SSIM is the perceptual companion to PSNR: structural breaks (tile seams,
    // channel swaps) show up here even when MSE stays moderate. Baseline on
    // the synthetic image with wavefront warm-starts is well above 0.9;
    // 0.85 keeps headroom while still failing hard on structural damage.
    if (s >= 0.0) check(s > 0.85, "roundtrip: SSIM > 0.85");
    else check(false, "roundtrip: SSIM unavailable");

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

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
void test_wavelet_basis() {
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
    // coarsest detail band stay x1, the finest band gets x wscale, DCT
    // stays all-ones (bit-identical legacy behavior)
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
        // DCT is always uniform regardless of the knob
        cs_fista_basis bd = cs_make_basis(64, 64, CS_BASIS_DCT, 3.0f);
        for (float v : bd.wscale) ok = ok && (std::fabs(v - 1.0f) < 1e-6f);
        check(ok, "wavelet: per-scale ramp LL=1, finest=wscale, DCT uniform");
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

// ---------------------------------------------------------------------------
// 5. adaptive sample-count budget (pure function) + full adaptive roundtrip
// ---------------------------------------------------------------------------
void test_adaptive_counts() {
    // deterministic: same inputs -> same counts, sum respects capacity
    const int rows = 64, cols = 64, ts = 32;
    const int tiles_cols = cols / ts, tiles_rows = rows / ts;
    std::vector<uint8_t> lod((size_t)tiles_rows * tiles_cols);
    for (size_t i = 0; i < lod.size(); i++) lod[i] = (uint8_t)(i * 50); // spread

    const int m = 64 * 64 / 2; // 50% budget
    const auto a = cs_compute_tile_sample_counts(lod, m, rows, cols, ts);
    const auto b = cs_compute_tile_sample_counts(lod, m, rows, cols, ts);
    check(a == b, "adaptive counts: pure function is deterministic");
    check((int)a.size() == tiles_rows * tiles_cols, "adaptive counts: one entry per tile");

    long long sum = 0, cap_sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i];
        const int tr = (int)i / tiles_cols, tc = (int)i % tiles_cols;
        const int tw = (ts < cols - tc * ts) ? ts : cols - tc * ts;
        const int th = (ts < rows - tr * ts) ? ts : rows - tr * ts;
        cap_sum += tw * th;
        if (a[i] < 0 || a[i] > tw * th) {
            check(false, "adaptive counts: each count in [0, tile capacity]");
            break;
        }
    }
    check(sum == m, "adaptive counts: total equals requested m");

    // higher LOD must not receive fewer samples than a lower-LOD twin tile
    // when capacities match (weights are weight_base + lod, default base 256)
    std::vector<uint8_t> lod2 = { 10, 250, 10, 250 };
    const auto c = cs_compute_tile_sample_counts(lod2, 1000, 64, 64, 32);
    check(c[1] > c[0] && c[3] > c[2], "adaptive counts: high LOD gets more samples");

    // strength mapping: 0 -> uniform, 0.5 -> default 256, 1 -> max bias
    check(cs_adaptive_base_from_strength(0.0) == 65535, "strength 0 -> base 65535 (uniform)");
    check(cs_adaptive_base_from_strength(0.5) == 256, "strength 0.5 -> base 256 (default)");
    check(cs_adaptive_base_from_strength(1.0) == 1, "strength 1 -> base 1 (max bias)");

    // low strength must not widen the LOD gap as much as high strength
    const auto mild = cs_compute_tile_sample_counts(lod2, 1000, 64, 64, 32, 65535);
    const auto strong = cs_compute_tile_sample_counts(lod2, 1000, 64, 64, 32, 1);
    const long long gap_mild = (long long)mild[1] - mild[0];
    const long long gap_strong = (long long)strong[1] - strong[0];
    check(gap_strong > gap_mild, "adaptive counts: strength 1 widens LOD gap vs strength 0");

    // zero m -> all zeros
    const auto z = cs_compute_tile_sample_counts(lod2, 0, 64, 64, 32);
    check(!z.empty() && std::all_of(z.begin(), z.end(), [](int v) { return v == 0; }),
        "adaptive counts: m=0 yields all zeros");

    // lod length mismatch throws
    bool threw = false;
    try { cs_compute_tile_sample_counts(std::vector<uint8_t>{1, 2}, m, rows, cols, ts); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw, "adaptive counts: wrong lod length throws");

    // regression: skewed LOD at full budget must still sum to m (the old
    // clamp loop dropped ~12k samples when detail tiles hit capacity first)
    const int rows2 = 480, cols2 = 360, ts2 = 64;
    const int tc2 = (cols2 + ts2 - 1) / ts2, tr2 = (rows2 + ts2 - 1) / ts2;
    std::vector<uint8_t> lod_skew((size_t)tr2 * tc2);
    for (size_t i = 0; i < lod_skew.size(); i++)
        lod_skew[i] = (uint8_t)(32 + ((int)i * 7 + (int)i * 13) % 224);
    const int m_full = rows2 * cols2;
    const auto sk = cs_compute_tile_sample_counts(lod_skew, m_full, rows2, cols2, ts2);
    long long sk_sum = 0;
    for (int v : sk) sk_sum += v;
    check(sk_sum == m_full, "adaptive counts: skewed lod at full budget sums to m");

    // --lod-full: threshold 0 is bit-identical to the legacy path
    const auto legacy = cs_compute_tile_sample_counts(lod2, 1000, 64, 64, 32);
    const auto t0 = cs_compute_tile_sample_counts(lod2, 1000, 64, 64, 32, 256, 0);
    check(legacy == t0, "lod-full: threshold 0 matches legacy");

    // qualifying tiles take full capacity; the pool splits the remainder
    std::vector<uint8_t> lod3 = { 10, 200, 30, 250 };
    const auto f = cs_compute_tile_sample_counts(lod3, 2500, 64, 64, 32, 256, 128);
    check(f[1] == 1024 && f[3] == 1024, "lod-full: qualifying tiles fully sampled");
    long long fsum = 0;
    for (int v : f) fsum += v;
    check(fsum == 2500, "lod-full: total still equals m");

    // fit-what-fits: tile 3 (lod 250) fits in 1500, tile 1 (lod 200) does not
    const auto g = cs_compute_tile_sample_counts(lod3, 1500, 64, 64, 32, 256, 128);
    check(g[3] == 1024 && g[1] < 1024, "lod-full: over-budget qualifier stays pooled");
    long long gsum = 0;
    for (int v : g) gsum += v;
    check(gsum == 1500, "lod-full: over-budget total still equals m");

    // budget below one tile capacity: nothing forced, everything pooled
    const auto h = cs_compute_tile_sample_counts(lod3, 500, 64, 64, 32, 256, 128);
    check(h[3] < 1024 && h[1] < 1024, "lod-full: tiny budget forces nothing");
    long long hsum = 0;
    for (int v : h) hsum += v;
    check(hsum == 500, "lod-full: tiny budget total still equals m");
}

void test_adaptive_roundtrip() {
    const std::string password = "adaptive-test-password";
    const int W = 960, H = 720;
    cv::Mat original = make_test_image(W, H);

    const std::string tmp_in = ".cs_test_adaptive_in.png";
    const std::string tmp_out = ".cs_test_adaptive_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &encrypted, /*adaptive*/ true, 32) != 0) {
        check(false, "adaptive roundtrip: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    check(!encrypted.empty(), "adaptive roundtrip: encrypt to memory");

    // mode byte must be CS_MODE_ADAPTIVE; default strength lands base=256
    const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
    check(flat.data[CS_OFF_PAD] == CS_MODE_ADAPTIVE, "adaptive roundtrip: header mode=2");
    const int base_written =
        flat.data[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
        (flat.data[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
    check(base_written == 256, "adaptive roundtrip: default strength writes base=256");

    // wrong password rejected before solve
    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, "wrong-password-99", 24, 24, 5, 4, 0.01f, false) == -2,
        "adaptive roundtrip: wrong password rejected with -2");

    // correct password: full solve
    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
        "adaptive roundtrip: decrypt succeeds");

    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "adaptive roundtrip: decrypted file readable");

    cv::Mat dec_sized;
    cv::resize(decrypted, dec_sized, original.size());
    const double p = cs_quality::psnr(original, dec_sized);
    const double s = cs_quality::ssim(original, dec_sized);
    std::printf("       adaptive roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
    std::printf("       adaptive roundtrip SSIM: %.4f\n", s);
    // AUTO_PARAM overrides the suite params (1 tile, 8 iters, coef 0.0375):
    // at 50% adaptive lands ~17 dB / ~0.33 SSIM on this synthetic. Floors
    // sit well above a broken index pipeline (single-digit dB, near-zero
    // SSIM) to catch misaligned lod offsets or wrong measurement origins
    // without over-fitting to solver noise.
    check(p > 14.0, "adaptive roundtrip: PSNR > 14 dB at ratio 0.5");
    if (s >= 0.0) check(s > 0.25, "adaptive roundtrip: SSIM > 0.25");
    else check(false, "adaptive roundtrip: SSIM unavailable");

    // second encrypt with same inputs must produce a working decrypt too
    // (salt is random, so ciphertexts differ � only quality path is checked)
    cv::Mat enc2;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 1.0f, 64, &enc2, true, 32) == 0,
        "adaptive roundtrip: encrypt at ratio 1.0");
    check(decrypt_image::decrypt_image_tiled(enc2, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
        "adaptive roundtrip: decrypt at ratio 1.0");
    cv::Mat dec2 = cv::imread(tmp_out, cv::IMREAD_COLOR);
    if (!dec2.empty()) {
        cv::resize(dec2, dec_sized, original.size());
        const double p1 = cs_quality::psnr(original, dec_sized);
        std::printf("       adaptive roundtrip PSNR (ratio 1.0): %.2f dB\n", p1);
        check(p1 > 24.0, "adaptive roundtrip: PSNR > 24 dB at ratio 1.0");
    }

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 4b. --lod-full detail guarantee roundtrip (mode 2 + threshold byte)
// ---------------------------------------------------------------------------
void test_lod_full_roundtrip() {
    const std::string password = "lod-full-test-password";
    const int W = 480, H = 360;
    cv::Mat original = make_test_image(W, H);

    const std::string tmp_in = ".cs_test_lodfull_in.png";
    const std::string tmp_out = ".cs_test_lodfull_out.png";
    cv::imwrite(tmp_in, original);

    // adaptive + threshold 200 (full_res keeps the solve small and fast)
    cv::Mat enc;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &enc,
        true, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8, 0, 200) == 0, "lod-full: encrypt");
    {
        decrypt_image probe(enc, password);
        check(probe.get_lod_full_threshold() == 200, "lod-full: header reads back 200");
    }
    {
        // threshold byte travels in the authenticated pad region
        const cv::Mat flat = enc.reshape(0, (int)enc.total());
        check(flat.data[CS_OFF_PAD + CS_OFF_LOD_FULL] == 200, "lod-full: pad byte stamped");
    }
    check(decrypt_image::decrypt_image_tiled(enc, tmp_out, password, 4, 16, 5, 4, 0.01f, false) == 0,
        "lod-full: decrypt");
    cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!dec.empty(), "lod-full: output readable");
    if (!dec.empty()) {
        cv::Mat sized;
        cv::resize(dec, sized, original.size());
        const double p = cs_quality::psnr(original, sized);
        std::printf("       lod-full roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
        check(p > 12.0, "lod-full: PSNR sane");
    }
    {
        // tampered threshold byte fails authentication (pad under MAC)
        cv::Mat tampered = enc.clone();
        tampered.data[CS_OFF_PAD + CS_OFF_LOD_FULL] ^= 1;
        bool rejected = false;
        try { decrypt_image probe(tampered, password); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "lod-full: tampered threshold rejected");
    }

    // out-of-range threshold rejected, not silently clamped
    cv::Mat enc_bad;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &enc_bad,
        true, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8, 0, 300) == -1, "lod-full: out-of-range threshold rejected");

    // hf-focus (own budgeting) rejects the adaptive-family guarantee
    cv::Mat enc_hf;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 32, &enc_hf,
        false, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, true, 8, 0, 100) == -1, "lod-full: hf-focus rejects guarantee");

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 5a2. YCC 4:2:0 split-sampling roundtrip (mode 3)
// ---------------------------------------------------------------------------
// Luma sampled full-res, each chroma plane sampled on its half-res grid;
// body is byte-packed [Y][Cr][Cb] so the container is ~half the BGR size at
// the same luma ratio. Decrypt auto-selects the split pipeline from the
// header mode byte.
void test_ycc420_roundtrip() {
    const std::string password = "ycc420-test-password";
    const int W = 480, H = 360; // mult of 8 and of 2 (half-res pipeline safe)
    cv::Mat original = make_test_image(W, H);

    const std::string tmp_in = ".cs_test_ycc420_in.png";
    const std::string tmp_out = ".cs_test_ycc420_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &encrypted,
            /*adaptive*/ false, 32, 0.5f, /*show_mask*/ false, /*full_res*/ false,
            /*two_pass*/ false, 0.05f, "", 0.5f, 0.0f, /*ycc420*/ true) != 0) {
        check(false, "ycc420 roundtrip: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    check(!encrypted.empty(), "ycc420 roundtrip: encrypt to memory");

    const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
    check(flat.data[CS_OFF_PAD] == CS_MODE_YCC420, "ycc420 roundtrip: header mode=3");

    // same-ratio BGR container must be substantially larger (3 B/px vs ~1.5)
    cv::Mat enc_bgr;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc_bgr) == 0,
        "ycc420 roundtrip: reference BGR encrypt");
    std::printf("       ycc420 container px: %d vs BGR: %d\n",
        (int)encrypted.total(), (int)enc_bgr.total());
    check(encrypted.total() < enc_bgr.total() * 2 / 3, "ycc420 roundtrip: container < 2/3 of BGR");

    // wrong password rejected before solve
    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, "wrong-password-99", 4, 24, 5, 4, 0.01f, false) == -2,
        "ycc420 roundtrip: wrong password rejected with -2");

    // correct password: full solve through the split pipeline
    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
        "ycc420 roundtrip: decrypt succeeds");

    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "ycc420 roundtrip: decrypted file readable");

    cv::Mat dec_sized;
    cv::resize(decrypted, dec_sized, original.size());
    const double p = cs_quality::psnr(original, dec_sized);
    const double s = cs_quality::ssim(original, dec_sized);
    std::printf("       ycc420 roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
    std::printf("       ycc420 roundtrip SSIM: %.4f\n", s);
    check(p > 14.0, "ycc420 roundtrip: PSNR > 14 dB at ratio 0.5");
    if (s >= 0.0) check(s > 0.4, "ycc420 roundtrip: SSIM > 0.4");
    else check(false, "ycc420 roundtrip: SSIM unavailable");

    // joint SOMP through the same split pipeline (degrades to per-channel
    // FISTA on the ycc420 grid)
    const std::string tmp_out_joint = ".cs_test_ycc420_joint.png";
    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out_joint, password, 4, 24, 5, 4, 0.01f, false,
        0.0f, false, CS_SOLVER_FISTA_JOINT) == 0, "ycc420 roundtrip: joint decrypt succeeds");
    cv::Mat dec_joint = cv::imread(tmp_out_joint, cv::IMREAD_COLOR);
    check(!dec_joint.empty(), "ycc420 roundtrip: joint output readable");
    if (!dec_joint.empty()) {
        cv::Mat joint_sized;
        cv::resize(dec_joint, joint_sized, original.size());
        const double pa = cs_quality::psnr(original, joint_sized);
        std::printf("       ycc420 joint PSNR (ratio 0.5): %.2f dB\n", pa);
        check(std::isfinite(pa) && pa > 14.0, "ycc420 roundtrip: joint PSNR > 14 dB");
    }
    std::remove(tmp_out_joint.c_str());

    // manual mode with overlap 0: tiles solve independently and composite
    // without feathering (previously rejected by the overlap validation)
    const std::string tmp_out0 = ".cs_test_ycc420_ov0.png";
    CSencryption::params = MANUAL_PARAM;
    const int rc0 = decrypt_image::decrypt_image_tiled(encrypted, tmp_out0, password, 4, 0, 5, 4, 0.01f, false);
    CSencryption::params = AUTO_PARAM;
    check(rc0 == 0, "ycc420 overlap 0: manual decrypt succeeds");
    cv::Mat dec0 = cv::imread(tmp_out0, cv::IMREAD_COLOR);
    check(!dec0.empty(), "ycc420 overlap 0: decrypted file readable");
    if (!dec0.empty()) {
        cv::resize(dec0, dec0, original.size());
        const double p0 = cs_quality::psnr(original, dec0);
        std::printf("       ycc420 overlap 0 PSNR (manual 5 iters): %.2f dB\n", p0);
        check(p0 > 8.0, "ycc420 overlap 0: PSNR sane without overlap");
    }
    std::remove(tmp_out0.c_str());

    // LOD-driven luma (regions): same mode 3, luma budget follows the region
    // scores while chroma stays uniform; lod bytes ship in the container
    cv::Mat enc_lod;
    const std::string regions = "{'regions': [{'box': [250,250,750,750], 'detail': 1.0}]}";
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &enc_lod,
            /*adaptive*/ false, 8, 0.66f, false, false, false, 0.05f,
            regions, 0.5f, 1.0f, /*ycc420*/ true) != 0) {
        check(false, "ycc420 regions: encrypt");
    } else {
        const cv::Mat flat2 = enc_lod.reshape(0, (int)enc_lod.total());
        check(flat2.data[CS_OFF_PAD] == CS_MODE_YCC420, "ycc420 regions: header mode=3");
        const int tile_written = flat2.data[CS_OFF_PAD + 1] | (flat2.data[CS_OFF_PAD + 2] << 8);
        check(tile_written == 64, "ycc420 regions: header carries tile_size 64");
        check(decrypt_image::decrypt_image_tiled(enc_lod, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "ycc420 regions: decrypt succeeds");
        cv::Mat dec_lod = cv::imread(tmp_out, cv::IMREAD_COLOR);
        if (!dec_lod.empty()) {
            cv::resize(dec_lod, dec_lod, original.size());
            const double pl = cs_quality::psnr(original, dec_lod);
            std::printf("       ycc420 regions PSNR (ratio 0.5): %.2f dB\n", pl);
            check(pl > 14.0, "ycc420 regions: PSNR > 14 dB at ratio 0.5");
        } else {
            check(false, "ycc420 regions: decrypted file readable");
        }
    }

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}
// ---------------------------------------------------------------------------
// Same container contract as Laplacian adaptive (mode=2 + lod bytes), so
// decrypt needs no new flags; only the scoring path differs (pilot uniform
// sample + cheap recon, residual-driven LOD).
void test_twopass_roundtrip() {
    const std::string password = "twopass-test-password";
    cv::Mat original = make_test_image(480, 360);
    const std::string tmp_in = ".cs_test_twopass_in.png";
    const std::string tmp_out = ".cs_test_twopass_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &encrypted,
        /*adaptive*/ true, 32, 0.5f, /*show_mask*/ false, /*full_res*/ false,
        /*two_pass*/ true, 0.05f) != 0) {
        check(false, "twopass roundtrip: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    check(!encrypted.empty(), "twopass roundtrip: encrypt to memory");

    const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
    check(flat.data[CS_OFF_PAD] == CS_MODE_ADAPTIVE, "twopass roundtrip: header mode=2");

    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
        "twopass roundtrip: decrypt succeeds");
    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "twopass roundtrip: decrypted file readable");
    if (!decrypted.empty()) {
        cv::Mat dec_sized;
        cv::resize(decrypted, dec_sized, original.size());
        const double p = cs_quality::psnr(original, dec_sized);
        const double s = cs_quality::ssim(original, dec_sized);
        std::printf("       twopass roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
        std::printf("       twopass roundtrip SSIM: %.4f\n", s);
        // Regression guard (same weak-params band as the adaptive sibling
        // test): broken scoring collapses to single-digit dB, so these
        // floors catch miswired pilot/residual paths without over-fitting
        // to solver noise. Head-to-head quality is covered by benches.
        check(p > 14.0, "twopass roundtrip: PSNR > 14 dB at ratio 0.5");
        if (s >= 0.0) check(s > 0.2, "twopass roundtrip: SSIM > 0.2");
        else check(false, "twopass roundtrip: SSIM unavailable");
    }

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 5c. LLM region-guided sampling roundtrip (canned parrot-style JSON)
// ---------------------------------------------------------------------------
// Inline --regions JSON blends box-weighted tile scores over the signal
// base; container stays mode=2 so decrypt is unchanged. Malformed JSON must
// fail the encrypt (return -1), never crash.
void test_regions_roundtrip() {
    const std::string password = "regions-test-password";
    cv::Mat original = make_test_image(480, 360);
    const std::string tmp_in = ".cs_test_regions_in.png";
    const std::string tmp_out = ".cs_test_regions_out.png";
    cv::imwrite(tmp_in, original);

    // parrot-style spec: head box (highest) + wing box, 0-1000 coords
    const std::string regions =
        "{\"regions\": ["
        "{\"box\": [325,165,605,445], \"detail\": 1.0, \"label\": \"head\"}, "
        "{\"box\": [367,512,825,996], \"detail\": 0.75, \"label\": \"wing\"}, "
        "{\"box\": [290,330,405,985], \"detail\": 0.5}"
        "]}";

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &encrypted,
        /*adaptive*/ true, 32, 0.5f, /*show_mask*/ false, /*full_res*/ false,
        /*two_pass*/ false, 0.05f, regions, 0.5f) != 0) {
        check(false, "regions roundtrip: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    check(!encrypted.empty(), "regions roundtrip: encrypt to memory");

    const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
    check(flat.data[CS_OFF_PAD] == CS_MODE_ADAPTIVE, "regions roundtrip: header mode=2");

    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
        "regions roundtrip: decrypt succeeds");
    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "regions roundtrip: decrypted file readable");
    if (!decrypted.empty()) {
        cv::Mat dec_sized;
        cv::resize(decrypted, dec_sized, original.size());
        const double p = cs_quality::psnr(original, dec_sized);
        const double s = cs_quality::ssim(original, dec_sized);
        std::printf("       regions roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
        std::printf("       regions roundtrip SSIM: %.4f\n", s);
        check(p > 14.0, "regions roundtrip: PSNR > 14 dB at ratio 0.5");
        if (s >= 0.0) check(s > 0.2, "regions roundtrip: SSIM > 0.2");
        else check(false, "regions roundtrip: SSIM unavailable");
    }

    // malformed JSON rejected (no crash, no container)
    cv::Mat bad;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &bad,
        true, 32, 0.5f, false, false, false, 0.05f, "{not json", 0.5f) != 0,
        "regions roundtrip: malformed JSON rejected");
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &bad,
        true, 32, 0.5f, false, false, false, 0.05f, "{\"regions\": [{\"label\": \"nobox\"}]}", 0.5f) != 0,
        "regions roundtrip: region without box rejected");

    // empty array degrades to the signal base gracefully
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &bad,
        true, 32, 0.5f, false, false, false, 0.05f, "{\"regions\": []}", 0.5f) == 0,
        "regions roundtrip: empty array falls back to base");

    // single-quoted variant (PowerShell-safe spelling) parses identically
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &bad,
        true, 32, 0.5f, false, false, false, 0.05f, "{'regions': [{'box': [0,0,1000,1000], 'detail': 1}]}", 0.5f) == 0,
        "regions roundtrip: single quotes accepted");

    // smoothed density field decrypts through the same mode-2 container
    cv::Mat smoothed;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &smoothed,
        true, 32, 0.5f, false, false, false, 0.05f, regions, 0.5f, 1.5f) == 0,
        "regions roundtrip: smoothed encrypt succeeds");
    if (!smoothed.empty()) {
        check(decrypt_image::decrypt_image_tiled(smoothed, tmp_out, password, 24, 24, 5, 4, 0.01f, false) == 0,
            "regions roundtrip: smoothed decrypt succeeds");
        cv::Mat sdec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        if (!sdec.empty()) {
            cv::Mat sized;
            cv::resize(sdec, sized, original.size());
            check(cs_quality::psnr(original, sized) > 14.0, "regions roundtrip: smoothed PSNR sane");
        }
    }

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

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
        // No nunif here: a bogus command must degrade to AVIR, not fail.
        CsPhotoUpscalerOptions opt;
        opt.backend = "waifu2x";
        opt.waifu2x_cmd = "cs-nonexistent-waifu2x-cmd";
        auto grid = make_grid();
        cs_upscale_tiles_2x(grid, opt);
        check(all_2x(grid), "photo-upscaler: waifu2x failure falls back to AVIR");
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
        check(cs_waifu2x_ncnn_effective_args(opt) == "-n -1 -s 2",
            "photo-upscaler: ncnn default args are pure 2x upscale");
        opt.waifu2x_method = "noise_scale";
        opt.waifu2x_noise = 2;
        check(cs_waifu2x_ncnn_effective_args(opt) == "-n 2 -s 2",
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
        CsPhotoUpscalerOptions opt;
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

void test_hf_focus_roundtrip() {
    const std::string input_path = ".cs_test_hf_input.png";
    const std::string output_path = ".cs_test_hf_output.png";
    const std::string password = "hf-focus-roundtrip-password";
    cv::Mat original = make_test_image(320, 240);
    check(cv::imwrite(input_path, original), "hf-focus: write source image");

    for (bool ycc420 : {false, true}) {
        cv::Mat encrypted;
        const int encrypt_rc = encrypt_image::encrypt_image_tiled(
            input_path, "", password, 0.25f, 32, &encrypted, false, 32, 0.5f,
            false, false, false, 0.05f, "", 0.5f, 0.0f, ycc420, true);
        check(encrypt_rc == 0, ycc420 ? "hf-focus ycc: encrypt" : "hf-focus: encrypt");
        if (encrypt_rc != 0) continue;

        decrypt_image container(encrypted, password);
        const int expected_mode = ycc420 ? CS_MODE_YCC420_HF : CS_MODE_HF_FOCUS;
        check(container.get_sampling_mode() == expected_mode,
              ycc420 ? "hf-focus ycc: mode 6 header" : "hf-focus: mode 5 header");
        CSencryption::params = AUTO_PARAM;
        const int decrypt_rc = decrypt_image::decrypt_image_tiled(
            encrypted, output_path, password, 1, 24, 5, 4, 0.01f, false);
        check(decrypt_rc == 0,
              ycc420 ? "hf-focus ycc: decrypt" : "hf-focus: decrypt");
        const cv::Mat decoded = cv::imread(output_path, cv::IMREAD_COLOR);
        check(!decoded.empty(),
              ycc420 ? "hf-focus ycc: output readable" : "hf-focus: output readable");
        std::remove(output_path.c_str());
    }

    std::remove(input_path.c_str());
}

// ---------------------------------------------------------------------------
// Sample bit-depth (--sample-bits): quantize/pack/unpack helpers and a
// packed roundtrip. Plumbing breakage (wrong offsets, misaligned sections)
// collapses to garbage, so loose relative PSNR bounds suffice.
// ---------------------------------------------------------------------------
void test_sample_bits() {
    // helper unit checks: identity at 8, bounded error below
    check(cs_quantize_sample(0, 4) == 0 && cs_quantize_sample(255, 4) == 15,
        "sample-bits: quantize endpoints at 4 bits");
    check(cs_dequantize_sample(0, 4) == 0 && cs_dequantize_sample(15, 4) == 255,
        "sample-bits: dequantize endpoints at 4 bits");
    {
        bool ok = true;
        for (int v = 0; v < 256; ++v) {
            const uint8_t q = cs_quantize_sample((uint8_t)v, 5);
            const uint8_t r = cs_dequantize_sample(q, 5);
            if (std::abs((int)r - v) > 5) { ok = false; break; } // 5-bit step ~8.2
        }
        check(ok, "sample-bits: 5-bit roundtrip within half a step");
    }
    check(cs_packed_bytes(3, 8) == 3 && cs_packed_bytes(3, 4) == 2 &&
        cs_packed_bytes(8, 1) == 1 && cs_packed_bytes(0, 4) == 0,
        "sample-bits: packed size math");
    check(cs_packed_bgr_bytes(4, 6, 4) == 7,
        "sample-bits: BGR split-depth size math");
    {
        // pack/unpack roundtrip: pack quantizes raw 8-bit values, unpack
        // inverts to the quantize->dequantize fixed points
        const uint8_t vals[8] = { 0, 17, 34, 128, 200, 255, 7, 99 };
        uint8_t packed[8], back[8];
        cs_pack_samples(vals, 8, 4, packed);
        cs_unpack_samples(packed, 8, 4, back);
        bool ok = true;
        for (int i = 0; i < 8; ++i) ok = ok && (back[i] == cs_dequantize_sample(cs_quantize_sample(vals[i], 4), 4));
        check(ok, "sample-bits: 4-bit pack/unpack roundtrip exact");
    }
    {
        const uint8_t vals[12] = { 2, 70, 245, 22, 110, 230, 42, 150, 210, 62, 190, 180 };
        uint8_t packed[16] = {}, back[12] = {};
        cs_pack_samples_bgr(vals, 4, 6, 4, packed);
        cs_unpack_samples_bgr(packed, 4, 6, 4, back);
        bool ok = true;
        for (int i = 0; i < 12; ++i) {
            const int bits = (i % 3 == 1) ? 6 : 4;
            ok = ok && back[i] == cs_dequantize_sample(cs_quantize_sample(vals[i], bits), bits);
        }
        check(ok, "sample-bits: interleaved BGR split-depth roundtrip");
    }

    const std::string password = "sample-bits-test-password";
    const std::string tmp_in = ".cs_test_sb_in.png";
    const std::string tmp_out = ".cs_test_sb_out.png";
    cv::Mat original = make_test_image(480, 360);
    cv::imwrite(tmp_in, original);

    // 8-bit packed path must match the legacy unpacked layout exactly
    // (full-res: native geometry, no half-res downscale, for a strong baseline)
    cv::Mat enc8;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc8,
        false, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8) == 0, "sample-bits: encrypt 8-bit");
    {
        decrypt_image probe(enc8, password);
        check(probe.get_sample_bits() == 8, "sample-bits: header reads back 8");
    }
    check(decrypt_image::decrypt_image_tiled(enc8, tmp_out, password, 4, 16, 5, 4, 0.01f, false) == 0,
        "sample-bits: decrypt 8-bit");
    cv::Mat dec8 = cv::imread(tmp_out, cv::IMREAD_COLOR);
    // default pipeline decrypts at 2x the encrypted (half-res) geometry:
    // resize back before scoring, same as test_roundtrip
    cv::Mat dec8_sized;
    cv::resize(dec8, dec8_sized, original.size());
    const double p8 = cs_quality::psnr(original, dec8_sized);

    // 4-bit container: smaller, header reads 4, wrong password still -2
    cv::Mat enc4;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc4,
        false, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 4) == 0, "sample-bits: encrypt 4-bit");
    check(enc4.total() < enc8.total(), "sample-bits: 4-bit container is smaller");
    {
        decrypt_image probe(enc4, password);
        check(probe.get_sample_bits() == 4 && probe.get_sample_bits_chroma() == 4,
            "sample-bits: header reads back uniform 4");
    }
    check(decrypt_image::decrypt_image_tiled(enc4, tmp_out, "wrong-password-99", 4, 16, 5, 4, 0.01f, false) == -2,
        "sample-bits: packed container rejects wrong password");
    check(decrypt_image::decrypt_image_tiled(enc4, tmp_out, password, 4, 16, 5, 4, 0.01f, false) == 0,
        "sample-bits: decrypt 4-bit");
    cv::Mat dec4 = cv::imread(tmp_out, cv::IMREAD_COLOR);
    cv::Mat dec4_sized;
    cv::resize(dec4, dec4_sized, original.size());
    const double p4 = cs_quality::psnr(original, dec4_sized);
    std::printf("       sample-bits PSNR 8-bit: %.2f dB, 4-bit: %.2f dB\n", p8, p4);
    // Regression guard, not a quality benchmark: the 5-iteration manual
    // solve at ratio 0.5 measures ~17.6 dB today. Plumbing breakage
    // (wrong offsets, misaligned sections) collapses to single-digit dB,
    // so 15 dB keeps margin while failing hard on real damage.
    check(std::isfinite(p8) && p8 > 15.0, "sample-bits: 8-bit PSNR sane");
    check(std::isfinite(p4) && p4 > 13.0 && p4 > p8 - 10.0, "sample-bits: 4-bit PSNR within its quantization budget");

    // Split depth: Y uses 6 bits, Cr/Cb use 4 bits.
    cv::Mat enc64;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc64,
        false, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        true, false, 6, 4) == 0, "sample-bits: encrypt split-depth YCC");
    {
        decrypt_image probe(enc64, password);
        check(probe.get_sample_bits() == 6 && probe.get_sample_bits_chroma() == 4,
            "sample-bits: header reads split depths");
    }
    check(decrypt_image::decrypt_image_tiled(enc64, tmp_out, password, 4, 16, 5, 4, 0.01f, false) == 0,
        "sample-bits: decrypt split-depth YCC");
    cv::Mat dec64 = cv::imread(tmp_out, cv::IMREAD_COLOR);
    cv::Mat dec64_sized;
    cv::resize(dec64, dec64_sized, original.size());
    check(cs_quality::psnr(original, dec64_sized) > 15.0,
        "sample-bits: split-depth YCC PSNR sane");
    {
        cv::Mat tampered = enc64.clone();
        uint8_t* raw = tampered.data;
        raw[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] ^= 1;
        bool rejected = false;
        try { decrypt_image probe(tampered, password); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "sample-bits: tampered split-depth header rejected");
    }

    // Out-of-range depths are rejected, not silently ignored.
    cv::Mat enc_bad;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc_bad,
        false, 32, 0.5f, false, true, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 9) == -1, "sample-bits: out-of-range depth rejected");

    std::remove(tmp_in.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// Per-tile coef (--per-tile-coef): scaling math + an adaptive end-to-end
// where tile counts differ (uniform containers would scale by exactly 1).
// ---------------------------------------------------------------------------
void test_per_tile_coef() {
    // exact math: sqrt scaling with [0.25x, 4x] clamps and degenerate guards
    check(cs_per_tile_coef(0.01f, 100, 400.0) == 0.02f, "per-tile-coef: sqrt upscale");
    check(cs_per_tile_coef(0.01f, 400, 100.0) == 0.005f, "per-tile-coef: sqrt downscale");
    check(cs_per_tile_coef(0.01f, 1, 400.0) == 0.04f, "per-tile-coef: clamps at 4x");
    check(cs_per_tile_coef(0.01f, 100, 1.0) == 0.0025f, "per-tile-coef: clamps at 0.25x");
    check(cs_per_tile_coef(0.01f, 0, 400.0) == 0.01f, "per-tile-coef: empty tile unchanged");
    check(cs_per_tile_coef(0.01f, 100, 0.0) == 0.01f, "per-tile-coef: zero mean unchanged");
    check(cs_per_tile_coef(0.01f, 250, 250.0) == 0.01f, "per-tile-coef: mean tile unchanged");

    // adaptive container (unequal tile counts) + tiles=2: flag must engage
    // (output moves) and stay sane
    const std::string password = "per-tile-coef-test-password";
    const std::string tmp_in = ".cs_test_ptc_in.png";
    const std::string tmp_off = ".cs_test_ptc_off.png";
    const std::string tmp_on = ".cs_test_ptc_on.png";
    cv::Mat original = make_test_image(192, 144);
    cv::imwrite(tmp_in, original);
    cv::Mat enc;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 32, &enc,
        true, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8, 0, 200) == 0, "per-tile-coef: encrypt adaptive");
    // manual mode: honor the tiles=3 grid below (AUTO would collapse to 1
    // tile, where every factor is exactly 1 and the flag is a no-op)
    const int saved_params = CSencryption::params;
    CSencryption::params = MANUAL_PARAM;
    check(decrypt_image::decrypt_image_tiled(enc, tmp_off, password,
        3, 0, 5, 2, 0.01f, false, 0.0f, false,
        CS_SOLVER_FISTA, 0, 1, CS_BASIS_DCT, 2.0f,
        CsPhotoUpscalerOptions(), false) == 0, "per-tile-coef: decrypt flag off");
    check(decrypt_image::decrypt_image_tiled(enc, tmp_on, password,
        3, 0, 5, 2, 0.01f, false, 0.0f, false,
        CS_SOLVER_FISTA, 0, 1, CS_BASIS_DCT, 2.0f,
        CsPhotoUpscalerOptions(), true) == 0, "per-tile-coef: decrypt flag on");
    cv::Mat dec_off = cv::imread(tmp_off, cv::IMREAD_COLOR);
    cv::Mat dec_on = cv::imread(tmp_on, cv::IMREAD_COLOR);
    check(!dec_off.empty() && !dec_on.empty() && dec_off.size() == dec_on.size(),
        "per-tile-coef: outputs readable at expected geometry");
    if (!dec_off.empty() && !dec_on.empty() && dec_off.size() == dec_on.size()) {
        cv::Mat diff;
        cv::absdiff(dec_off, dec_on, diff);
        const double change = cv::mean(diff)[0] + cv::mean(diff)[1] + cv::mean(diff)[2];
        check(change > 1e-6, "per-tile-coef: flag moves pixels vs off");
        cv::Mat sized;
        cv::resize(dec_on, sized, original.size());
        const double p = cs_quality::psnr(original, sized);
        std::printf("       per-tile-coef PSNR: %.2f dB\n", p);
        check(p > 10.0, "per-tile-coef: output sane");
    }
    // tv flag needs a nonzero base tv to scale (scaling 0 stays 0); use a
    // strong base so engagement is visible above 8U rounding, and compare
    // against the same base tv without the flag to isolate the flag
    const std::string tmp_tv = ".cs_test_ptc_tv.png";
    const std::string tmp_tvbase = ".cs_test_ptc_tvbase.png";
    check(decrypt_image::decrypt_image_tiled(enc, tmp_tvbase, password,
        3, 0, 5, 2, 0.01f, false, 0.2f, false,
        CS_SOLVER_FISTA, 0, 1, CS_BASIS_DCT, 2.0f,
        CsPhotoUpscalerOptions(), false, false) == 0, "per-tile-tv: decrypt base tv");
    check(decrypt_image::decrypt_image_tiled(enc, tmp_tv, password,
        3, 0, 5, 2, 0.01f, false, 0.2f, false,
        CS_SOLVER_FISTA, 0, 1, CS_BASIS_DCT, 2.0f,
        CsPhotoUpscalerOptions(), false, true) == 0, "per-tile-tv: decrypt flag on");
    cv::Mat dec_tv = cv::imread(tmp_tv, cv::IMREAD_COLOR);
    cv::Mat dec_tvbase = cv::imread(tmp_tvbase, cv::IMREAD_COLOR);
    check(!dec_tv.empty() && !dec_tvbase.empty() && dec_tv.size() == dec_tvbase.size(),
        "per-tile-tv: outputs readable at expected geometry");
    if (!dec_tv.empty() && !dec_tvbase.empty() && dec_tv.size() == dec_tvbase.size()) {
        // no moves-pixels gate here by design: tv acts smoothly (unlike the
        // discontinuous L1 threshold), so per-tile tv lands below 8U output
        // precision on this content — bit-identical is the honest result
        cv::Mat sized;
        cv::resize(dec_tv, sized, original.size());
        check(cs_quality::psnr(original, sized) > 10.0, "per-tile-tv: output sane");
    }
    std::remove(tmp_tv.c_str());
    std::remove(tmp_tvbase.c_str());
    std::remove(tmp_in.c_str());
    std::remove(tmp_off.c_str());
    std::remove(tmp_on.c_str());
    // unfused branch (waifu2x backend degrades to the post-join AVIR batch):
    // same op as fused per-tile upscale, so near-identical output.
    // NOTE: stays under MANUAL_PARAM like the runs above — an AUTO run
    // would derive different solve params and is not comparable.
    {
        CsPhotoUpscalerOptions wx;
        wx.backend = "waifu2x";
        wx.waifu2x_cmd = "cs-nonexistent-waifu2x-cmd";
        const std::string tmp_wx = ".cs_test_ptc_wx.png";
        check(decrypt_image::decrypt_image_tiled(enc, tmp_wx, password,
            3, 0, 5, 2, 0.01f, false, 0.0f, false,
            CS_SOLVER_FISTA, 0, 1, CS_BASIS_DCT, 2.0f,
            wx, false, false) == 0, "per-tile-coef: unfused fallback decrypt");
        cv::Mat dec_wx = cv::imread(tmp_wx, cv::IMREAD_COLOR);
        if (!dec_wx.empty() && !dec_off.empty() && dec_wx.size() == dec_off.size()) {
            cv::Mat wxdiff;
            cv::absdiff(dec_wx, dec_off, wxdiff);
            const double wxchange = cv::mean(wxdiff)[0] + cv::mean(wxdiff)[1] + cv::mean(wxdiff)[2];
            double wxmax = 0.0;
            for (int yy = 0; yy < wxdiff.rows; ++yy) {
                const cv::Vec3b* row = wxdiff.ptr<cv::Vec3b>(yy);
                for (int xx = 0; xx < wxdiff.cols; ++xx)
                    for (int c = 0; c < 3; ++c)
                        wxmax = (std::max)(wxmax, (double)row[xx][c]);
            }
            cv::Mat so, sw;
            cv::resize(dec_off, so, original.size());
            cv::resize(dec_wx, sw, original.size());
            std::printf("       fused-vs-batch mean=%.4f max=%.0f (off %.2f dB, wx %.2f dB)\n",
                wxchange, wxmax, cs_quality::psnr(original, so), cs_quality::psnr(original, sw));
            check(wxchange < 0.01, "per-tile-coef: fused matches batch upscale");
        } else {
            check(false, "per-tile-coef: fused matches batch upscale");
        }
        std::remove(tmp_wx.c_str());
    }
    CSencryption::params = saved_params;
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

} // namespace

int main() {
    // Large photos trip CL_MEM_OBJECT_ALLOCATION_FAILURE on this host's
    // OpenCL runtime during heavy GEMM; CPU path is deterministic for tests.
    cv::ocl::setUseOpenCL(false);

    test_crypto();
    test_shuffle();
    test_tile_helpers();
    test_roundtrip();
    test_fista_solvers();
    test_wavelet_basis();
    test_adaptive_counts();
    test_adaptive_roundtrip();
    test_lod_full_roundtrip();
    test_ycc420_roundtrip();
    test_twopass_roundtrip();
    test_regions_roundtrip();
    test_photo_roundtrip();
    test_photo_upscaler();
    test_hf_focus_roundtrip();
    test_sample_bits();
    test_per_tile_coef();
    test_gpu_fista_agreement();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
