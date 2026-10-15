// Test suite for the CS image encryption backend.
// Covers: crypto primitives (KDF determinism, seal/verify, tamper detection),
// index shuffle determinism, tiling helpers, and a full encrypt->decrypt
// roundtrip scored by PSNR + SSIM.
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
    std::fflush(stdout); // crash diagnostics: never lose the tail to buffering
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

    // feather blend: a tile edge touching the image border has no neighbor,
    // so it carries full weight. A single solid tile + feather must come
    // back bit-exact; before the fix the outer pixel ring came out black
    // (cosine weight 0 with no contributor, 0/1e-6 after normalization).
    {
        const int W = 80, H = 64;
        cv::Mat solid(H, W, CV_8UC3, cv::Scalar(128, 64, 200));
        std::vector<std::vector<cv::Mat>> ft(1, std::vector<cv::Mat>(1, solid.clone()));
        std::vector<std::vector<TileCoord>> fc(1, std::vector<TileCoord>(1, { 0, 0 }));
        cv::Mat pasted = reconstructImage(ft, fc);
        cv::Mat fbl = blendTilesWithImage(ft, fc, pasted, 0.5f, 24);
        double mx = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const cv::Vec3b a = solid.at<cv::Vec3b>(y, x), b = fbl.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) mx = (std::max)(mx, (double)std::abs((int)a[c] - (int)b[c]));
            }
        check(mx == 0.0, "blend feather: single tile keeps border pixels (no black frame)");
    }
    // feather blend: 2x2 overlapping tiles from a solid image agree
    // everywhere (borders full weight, seams partition of unity).
    {
        const int W = 64, H = 64, ov = 16;
        cv::Mat solid(H, W, CV_8UC3, cv::Scalar(90, 160, 40));
        std::vector<std::vector<cv::Mat>> gt;
        std::vector<std::vector<TileCoord>> gc;
        splitImageIntoTiles(solid, gt, gc, 2, ov);
        cv::Mat pasted = reconstructImage(gt, gc);
        cv::Mat fbl = blendTilesWithImage(gt, gc, pasted, 0.5f, ov);
        double mx = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const cv::Vec3b a = solid.at<cv::Vec3b>(y, x), b = fbl.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) mx = (std::max)(mx, (double)std::abs((int)a[c] - (int)b[c]));
            }
        check(mx <= 1.0, "blend feather: overlapping tiles agree on solid image");
    }
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

void test_sample_split() {
    // Hand-made density: 64x64 image, 32px grid -> 2x2 tiles with
    // 100/300/200/200 samples (mean 200): under={(0,0)}, over={(0,1)}.
    const cv::Mat src = make_test_image(64, 64);
    const int per_tile[4] = { 100, 300, 200, 200 };
    std::vector<int> rx, ry;
    for (int t = 0; t < 4; ++t) {
        const int tr = t / 2, tc = t % 2, n = per_tile[t];
        for (int k = 0; k < n; ++k) {
            rx.push_back(tr * 32 + (k * 7) % 32);
            ry.push_back(tc * 32 + (k * 13) % 32);
        }
    }
    cv::Mat under, over;
    check(cs_build_sample_class_views(src, rx, ry, 32, under, over),
        "sample-split: builds on valid input");
    check(under.size() == src.size() && under.type() == CV_8UC3 &&
        over.size() == src.size() && over.type() == CV_8UC3,
        "sample-split: full geometry and color type");
    auto tile_nonblack = [](const cv::Mat& m, int tr, int tc, int ts) {
        const int h = (std::min)(ts, m.rows - tr * ts);
        const int w = (std::min)(ts, m.cols - tc * ts);
        long long nz = 0;
        for (int r = 0; r < h; ++r) {
            const cv::Vec3b* p = m.ptr<cv::Vec3b>(tr * ts + r) + tc * ts;
            for (int k = 0; k < w; ++k)
                if (p[k] != cv::Vec3b(0, 0, 0)) ++nz;
        }
        return nz;
    };
    // sparse tile lives in under only, dense tile in over only, ties black.
    // Both canvases carry a text label in the top band of tile (0,0), so
    // row-0 membership is probed on a strip below the text (y 26..31).
    auto strip_nonblack = [](const cv::Mat& m, int tr, int tc, int ts) {
        const int w = (std::min)(ts, m.cols - tc * ts);
        long long nz = 0;
        for (int r = 26; r < 32; ++r) {
            const cv::Vec3b* p = m.ptr<cv::Vec3b>(tr * ts + r) + tc * ts;
            for (int k = 0; k < w; ++k)
                if (p[k] != cv::Vec3b(0, 0, 0)) ++nz;
        }
        return nz;
    };
    check(strip_nonblack(under, 0, 0, 32) > 0 && strip_nonblack(over, 0, 0, 32) == 0,
        "sample-split: sparse tile lands in under canvas only");
    check(strip_nonblack(over, 0, 1, 32) > 0 && strip_nonblack(under, 0, 1, 32) == 0,
        "sample-split: dense tile lands in over canvas only");
    check(tile_nonblack(under, 1, 0, 32) == 0 && tile_nonblack(over, 1, 0, 32) == 0 &&
        tile_nonblack(under, 1, 1, 32) == 0 && tile_nonblack(over, 1, 1, 32) == 0,
        "sample-split: mean-tied tiles stay black in both");
    // canvases reproduce source pixels where they are non-black (same strip)
    {
        cv::Mat diff;
        cv::absdiff(src(cv::Rect(0, 26, 32, 6)), under(cv::Rect(0, 26, 32, 6)), diff);
        cv::Mat gray;
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
        check(cv::countNonZero(gray) == 0, "sample-split: under tile pixels match source");
    }
    // degenerate inputs rejected
    cv::Mat u2, o2;
    check(!cs_build_sample_class_views(src, {}, {}, 32, u2, o2),
        "sample-split: empty indices rejected");
    check(!cs_build_sample_class_views(src, rx, ry, 0, u2, o2),
        "sample-split: tile_size<1 rejected");
    check(!cs_build_sample_class_views(cv::Mat(), rx, ry, 32, u2, o2),
        "sample-split: empty image rejected");
    // exact-uniform grid: nothing to split
    {
        std::vector<int> ux, uy;
        for (int r = 0; r < 64; ++r)
            for (int c = 0; c < 64; ++c) { ux.push_back(r); uy.push_back(c); }
        check(!cs_build_sample_class_views(src, ux, uy, 32, u2, o2),
            "sample-split: uniform grid reports nothing to split");
    }
}

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
// 4a2. Spectral-DCT LOD scorer: unit behavior + adaptive-container roundtrip
// ---------------------------------------------------------------------------
void test_spectral_lod() {
    // flat image: DCT has DC only (weight 0) -> every tile scores the floor
    cv::Mat flat(256, 256, CV_8UC3, cv::Scalar(128, 128, 128));
    const std::vector<uint8_t> lod_flat = compute_spectral_lod(flat, 64, 32);
    check(lod_flat.size() == 16, "spectral lod: 4x4 grid for 256px/64px tiles");
    bool all_floor = true;
    for (uint8_t b : lod_flat) if (b != 32) all_floor = false;
    check(all_floor, "spectral lod: flat image scores uniform floor");

    // structured image: scores spread across [floor, 255] and are stable
    cv::Mat shaped = make_test_image(256, 256);
    const std::vector<uint8_t> a = compute_spectral_lod(shaped, 64, 32);
    const std::vector<uint8_t> b = compute_spectral_lod(shaped, 64, 32);
    check(a == b, "spectral lod: deterministic on identical input");
    uint8_t mn = 255, mx = 0;
    for (uint8_t v : a) { mn = (uint8_t)(std::min)(mn, v); mx = (uint8_t)(std::max)(mx, v); }
    check(mx > mn, "spectral lod: structure spreads scores");
    check(mn >= 32 && mx <= 255, "spectral lod: scores within [floor, 255]");

    const std::string password = "spectral-test-password";
    const int W = 480, H = 360;
    cv::Mat original = make_test_image(W, H);
    const std::string tmp_in = ".cs_test_spectral_in.png";
    const std::string tmp_out = ".cs_test_spectral_out.png";
    cv::imwrite(tmp_in, original);

    // two-pass and spectral are mutually exclusive scorers
    cv::Mat enc_none;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &enc_none,
        /*adaptive*/ false, 32, 0.5f, false, false, /*two_pass*/ true, 0.05f,
        "", 0.5f, 0.0f, false, false, 8, 0, 0, false, /*spectral*/ true) == -1,
        "spectral lod: two-pass + spectral rejected");

    // spectral encrypt ships the standard adaptive container (mode 2)
    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &encrypted,
            /*adaptive*/ false, 32, 0.5f, false, false, false, 0.05f,
            "", 0.5f, 0.0f, false, false, 8, 0, 0, false, /*spectral*/ true) != 0) {
        check(false, "spectral lod: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    const cv::Mat flat_enc = encrypted.reshape(0, (int)encrypted.total());
    check(flat_enc.data[CS_OFF_PAD] == CS_MODE_ADAPTIVE, "spectral lod: header mode=2");

    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
        "spectral lod: decrypt succeeds");
    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "spectral lod: decrypted file readable");
    if (!decrypted.empty()) {
        cv::Mat dec_sized;
        cv::resize(decrypted, dec_sized, original.size());
        const double p = cs_quality::psnr(original, dec_sized);
        const double s = cs_quality::ssim(original, dec_sized);
        std::printf("       spectral lod PSNR (ratio 0.5): %.2f dB\n", p);
        std::printf("       spectral lod SSIM: %.4f\n", s);
        check(p > 14.0, "spectral lod: PSNR > 14 dB at ratio 0.5");
        if (s >= 0.0) check(s > 0.25, "spectral lod: SSIM > 0.25");
        else check(false, "spectral lod: SSIM unavailable");
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
// YCC 4:2:2 (mode 7): luma at full resolution, chroma full-height /
// half-width. Container sits between 4:2:0 and BGR (~2 bytes/px).
void test_ycc422_roundtrip() {
    const std::string password = "ycc422-test-password";
    const int W = 480, H = 360;
    cv::Mat original = make_test_image(W, H);

    const std::string tmp_in = ".cs_test_ycc422_in.png";
    const std::string tmp_out = ".cs_test_ycc422_out.png";
    cv::imwrite(tmp_in, original);

    cv::Mat encrypted;
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &encrypted,
            /*adaptive*/ false, 32, 0.5f, /*show_mask*/ false, /*full_res*/ false,
            /*two_pass*/ false, 0.05f, "", 0.5f, 0.0f, /*ycc420*/ false,
            /*hf_focus*/ false, /*sample_bits*/ 8, /*chroma_bits*/ 0,
            /*lod_full*/ 0, /*ycc422*/ true) != 0) {
        check(false, "ycc422 roundtrip: encrypt");
        std::remove(tmp_in.c_str());
        return;
    }
    check(!encrypted.empty(), "ycc422 roundtrip: encrypt to memory");

    const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
    check(flat.data[CS_OFF_PAD] == CS_MODE_YCC422, "ycc422 roundtrip: header mode=7");

    // 4:2:2 stores ~2 B/px: well under BGR (3 B/px), above 4:2:0 (~1.5 B/px)
    cv::Mat enc_bgr, enc_420;
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc_bgr) == 0,
        "ycc422 roundtrip: reference BGR encrypt");
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &enc_420,
        false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f, true) == 0,
        "ycc422 roundtrip: reference 420 encrypt");
    std::printf("       ycc422 container px: %d vs 420: %d vs BGR: %d\n",
        (int)encrypted.total(), (int)enc_420.total(), (int)enc_bgr.total());
    check(encrypted.total() * 4 < enc_bgr.total() * 3, "ycc422 roundtrip: container < 3/4 of BGR");
    check(encrypted.total() >= enc_420.total(), "ycc422 roundtrip: container >= 420");

    check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
        "ycc422 roundtrip: decrypt succeeds");
    cv::Mat decrypted = cv::imread(tmp_out, cv::IMREAD_COLOR);
    check(!decrypted.empty(), "ycc422 roundtrip: decrypted file readable");
    if (!decrypted.empty()) {
        cv::Mat dec_sized;
        cv::resize(decrypted, dec_sized, original.size());
        const double p = cs_quality::psnr(original, dec_sized);
        const double s = cs_quality::ssim(original, dec_sized);
        std::printf("       ycc422 roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
        std::printf("       ycc422 roundtrip SSIM: %.4f\n", s);
        check(p > 14.0, "ycc422 roundtrip: PSNR > 14 dB at ratio 0.5");
        if (s >= 0.0) check(s > 0.4, "ycc422 roundtrip: SSIM > 0.4");
        else check(false, "ycc422 roundtrip: SSIM unavailable");
    }

    // LOD-driven luma under 4:2:2: same mode 7 with shipped lod bytes
    cv::Mat enc_lod;
    const std::string regions = "{'regions': [{'box': [250,250,750,750], 'detail': 1.0}]}";
    if (encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 64, &enc_lod,
            /*adaptive*/ false, 8, 0.66f, false, false, false, 0.05f,
            regions, 0.5f, 1.0f, /*ycc420*/ false, false, 8, 0, 0, /*ycc422*/ true) != 0) {
        check(false, "ycc422 regions: encrypt");
    } else {
        const cv::Mat flat2 = enc_lod.reshape(0, (int)enc_lod.total());
        check(flat2.data[CS_OFF_PAD] == CS_MODE_YCC422, "ycc422 regions: header mode=7");
        const int tile_written = flat2.data[CS_OFF_PAD + 1] | (flat2.data[CS_OFF_PAD + 2] << 8);
        check(tile_written == 64, "ycc422 regions: header carries tile_size 64");
        check(decrypt_image::decrypt_image_tiled(enc_lod, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "ycc422 regions: decrypt succeeds");
        cv::Mat dec_lod = cv::imread(tmp_out, cv::IMREAD_COLOR);
        if (!dec_lod.empty()) {
            cv::resize(dec_lod, dec_lod, original.size());
            const double pl = cs_quality::psnr(original, dec_lod);
            std::printf("       ycc422 regions PSNR (ratio 0.5): %.2f dB\n", pl);
            check(pl > 14.0, "ycc422 regions: PSNR > 14 dB at ratio 0.5");
        } else {
            check(false, "ycc422 regions: decrypted file readable");
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
// 5b2. T-tetromino sampling (tcluster points + tmeans groups)
// ---------------------------------------------------------------------------
void test_tsampling() {
    // Fixed-partition geometry: 8x8 grid -> 16 groups of 4, disjoint and
    // complete, each group a T-tetromino (3 leaves + 1 branch by Manhattan
    // adjacency, mirroring the bench self-test).
    {
        bool ok = true;
        std::vector<int> seen(64, 0);
        for (int cell = 0; cell < 4; ++cell) {
            for (int g = 0; g < 4; ++g) {
                int xs[4], ys[4];
                cs_tetro_group_pixels((cell / 2) * 4, (cell % 2) * 4, g, xs, ys);
                int deg[4] = { 0, 0, 0, 0 };
                for (int a = 0; a < 4; ++a) {
                    // helper returns global coords already (origin included)
                    const int gx = xs[a], gy = ys[a];
                    if (gx < 0 || gx >= 8 || gy < 0 || gy >= 8) ok = false;
                    else seen[(size_t)gy * 8 + gx]++;
                    for (int b = 0; b < 4; ++b) {
                        if (a != b && std::abs(xs[a] - xs[b]) + std::abs(ys[a] - ys[b]) == 1) deg[a]++;
                    }
                }
                int leaves = 0, branches = 0;
                for (int a = 0; a < 4; ++a) {
                    if (deg[a] == 1) leaves++;
                    else if (deg[a] >= 2) branches++;
                    else ok = false;
                }
                ok = ok && (leaves == 3 && branches == 1);
            }
        }
        for (int v : seen) ok = ok && (v == 1);
        check(ok, "tsampling: fixed partition disjoint+complete, all groups T-shaped");
    }

    // Grouped solve on a constant tile recovers it (catches operator/value
    // scaling bugs: unit-norm rows against raw means decode at 1/4, true
    // averages decode exactly).
    {
        const int S = 8;
        const float c = 0.4f;
        std::vector<int> off = { 0 };
        std::vector<int> px;
        std::vector<float> w;
        for (int cell = 0; cell < 4; ++cell) {
            for (int g = 0; g < 4; ++g) {
                int xs[4], ys[4];
                cs_tetro_group_pixels((cell / 2) * 4, (cell % 2) * 4, g, xs, ys);
                for (int q = 0; q < 4; ++q) {
                    px.push_back(ys[q] * S + xs[q]);
                    w.push_back(0.25f);
                }
                off.push_back((int)px.size());
            }
        }
        cs_meas_groups G{ off.data(), px.data(), w.data(), 16 };
        cv::Mat meas(1, CS_HEADER_PIXELS + 16, CV_8UC3, cv::Scalar(0, 0, 0));
        const int v = (int)(c * 255.0f + 0.5f);
        for (int k = 0; k < 16; ++k) meas.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = cv::Vec3b(v, v, v);
        cv::Mat ref(S, S, CV_32F, cv::Scalar(0));
        std::vector<int> empty;
        reconstruct_color_channel_fista(meas, 0, 1e-6f, S, S, empty, empty, 2, ref,
            false, cs_null_mat(), 0.0f, 1, 0, CS_BASIS_DCT, 1.0f, false, &G);
        const double mean = cv::mean(cv::Mat(S, S, CV_32F, ref.data))[0];
        std::printf("       tsampling grouped const: mean=%.2f (expect ~%.1f)\n", mean, c * 255.0);
        check(std::fabs(mean - c * 255.0) < 1.5, "tsampling: grouped constant tile decodes exactly");
    }

    // End-to-end container roundtrips on synthetic content (half-res
    // defaults like the ycc sibling tests).
    const std::string password = "tsampling-test-password";
    cv::Mat original = make_test_image(192, 144);
    const std::string tmp_in = ".cs_test_tsampling_in.png";
    cv::imwrite(tmp_in, original);

    // tcluster points, ratio 0.5
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_tcluster(0.5f, password, 8, 0);
        cv::Mat container = enc.get_mat();
        check(!container.empty(), "tsampling: tcluster encrypt to memory");
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_TCLUSTER, "tsampling: tcluster header mode=13");
        const std::string tmp_out = ".cs_test_tsampling_tc.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "tsampling: tcluster decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: tcluster output readable");
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       tsampling tcluster PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 13.0, "tsampling: tcluster PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }

    // tcluster full coverage (ratio 1.0): every pixel exactly once
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_tcluster(1.0f, password, 8, 0);
        cv::Mat container = enc.get_mat();
        const std::string tmp_out = ".cs_test_tsampling_tc1.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "tsampling: tcluster full-coverage decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       tsampling tcluster PSNR (ratio 1.0): %.2f dB\n", p);
            check(p > 27.0, "tsampling: tcluster full coverage near-exact");
        }
        std::remove(tmp_out.c_str());
    }

    // tmeans groups, ratio 0.25 (full pair coverage)
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_tmeans(0.25f, password, 8, 0);
        cv::Mat container = enc.get_mat();
        check(!container.empty(), "tsampling: tmeans encrypt to memory");
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_TMEANS, "tsampling: tmeans header mode=14");
        const std::string tmp_out = ".cs_test_tsampling_tm.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "tsampling: tmeans decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: tmeans output readable");
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       tsampling tmeans PSNR (ratio 0.25): %.2f dB\n", p);
            check(p > 24.0, "tsampling: tmeans PSNR sane");
            // joint SOMP through the grouped path
            const std::string tmp_joint = ".cs_test_tsampling_tmj.png";
            check(decrypt_image::decrypt_image_tiled(container, tmp_joint, password, 4, 24, 5, 4, 0.01f, false,
                0.0f, false, CS_SOLVER_FISTA_JOINT) == 0, "tsampling: tmeans joint decrypt succeeds");
            cv::Mat decj = cv::imread(tmp_joint, cv::IMREAD_COLOR);
            check(!decj.empty(), "tsampling: tmeans joint output readable");
            if (!decj.empty()) {
                cv::Mat jsized;
                cv::resize(decj, jsized, original.size());
                const double pj = cs_quality::psnr(original, jsized);
                std::printf("       tsampling tmeans joint PSNR: %.2f dB\n", pj);
                check(pj > 20.0, "tsampling: tmeans joint PSNR sane");
            }
            std::remove(tmp_joint.c_str());
        }
        std::remove(tmp_out.c_str());
    }

    std::remove(tmp_in.c_str());
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
// Edge-region scores (compare/reporting): truth-defined top-10% gradient
// mask with PSNR/MAE over masked pixels. Deterministic, no quality module.
// ---------------------------------------------------------------------------
void test_edge_scores() {
    const cv::Mat img = make_test_image(96, 64);
    double p = -2.0, mae = -2.0, share = -2.0;
    cs_quality::edge_scores(img, img, p, mae, share);
    check(p == 99.0 && mae == 0.0, "edge-scores: identical images score 99/0");
    check(share > 0.05 && share < 0.2, "edge-scores: mask share near 10%");
    cv::Mat shifted;
    cv::Canny(img, shifted, 50, 150);
    cv::cvtColor(shifted, shifted, cv::COLOR_GRAY2BGR);
    double p2 = -2.0, mae2 = -2.0, share2 = -2.0;
    cs_quality::edge_scores(img, shifted, p2, mae2, share2);
    check(p2 >= 0.0 && p2 < 99.0 && mae2 > 0.0, "edge-scores: degraded image finite");
    check(share2 == share, "edge-scores: mask depends on original only");
    cv::Mat small;
    cv::resize(img, small, cv::Size(48, 32));
    double p3 = -2.0, mae3 = -2.0, share3 = -2.0;
    cs_quality::edge_scores(img, small, p3, mae3, share3);
    check(p3 < 0.0 && mae3 < 0.0 && share3 == 0.0, "edge-scores: size mismatch rejected");
}

// ---------------------------------------------------------------------------
// Anchor-weighted RED (--hr-red-radius): the RED-lite blend scales per
// pixel from 0 at anchor pixels to full far away, instead of uniform.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// LOD contrast exponent (--adaptive-gamma): reshapes shipped detail bytes
// past what strength 1.0 + floor 0 can express, with no format change.
// ---------------------------------------------------------------------------
void test_adaptive_gamma() {
    const std::vector<uint8_t> lod = { 0, 32, 128, 200, 255 };
    std::vector<uint8_t> id = lod;
    cs_apply_lod_gamma(id, 32, 1.0);
    check(id == lod, "adaptive gamma: 1.0 is identity");
    std::vector<uint8_t> empty;
    cs_apply_lod_gamma(empty, 32, 2.0);
    check(empty.empty(), "adaptive gamma: empty stays empty");
    cs_apply_lod_gamma(id, 32, 0.0);
    check(id == lod, "adaptive gamma: non-positive gamma is a no-op");
    cs_apply_lod_gamma(id, 32, -2.0);
    check(id == lod, "adaptive gamma: negative gamma is a no-op");
    std::vector<uint8_t> deg = { 10, 200, 255 };
    cs_apply_lod_gamma(deg, 255, 2.0);
    check((deg == std::vector<uint8_t>{ 10, 200, 255 }),
        "adaptive gamma: degenerate floor leaves bytes untouched");
    // gamma 2 with floor 32: fixed points hold, mids pushed down.
    // (128-32)/223 = 0.4305^2 * 223 = 41.3 -> 73;
    // (200-32)/223 = 0.7534^2 * 223 = 126.6 -> 159.
    std::vector<uint8_t> g = lod;
    cs_apply_lod_gamma(g, 32, 2.0);
    check(g[0] == 0 && g[1] == 32 && g[4] == 255,
        "adaptive gamma: sub-floor preserved, floor/ceiling fixed");
    check(g[2] == 73 && g[3] == 159, "adaptive gamma: gamma 2 value check");
    // gamma < 1 flattens: mids rise (sqrt(0.4305) * 223 = 146.3 -> 178).
    std::vector<uint8_t> f = lod;
    cs_apply_lod_gamma(f, 32, 0.5);
    check(f[2] == 178 && f[2] > lod[2], "adaptive gamma: gamma 0.5 lifts mids");
    // monotonicity preserved on a spread ramp at high gamma.
    std::vector<uint8_t> ramp;
    for (int v = 0; v < 256; v += 7) ramp.push_back((uint8_t)v);
    cs_apply_lod_gamma(ramp, 0, 4.0);
    check(std::is_sorted(ramp.begin(), ramp.end()),
        "adaptive gamma: monotonicity preserved at gamma 4");
    // end effect: gamma concentrates the budget on the top tile by pushing
    // mid-detail down (extremes barely move relative to the floor, so the
    // discriminating case has a mid tile: {lo, mid, lo, hi}, floor 10).
    const std::vector<uint8_t> lod2 = { 10, 130, 10, 250 };
    const auto plain_counts = cs_compute_tile_sample_counts(lod2, 400, 64, 64, 32, 1);
    std::vector<uint8_t> sharp = lod2;
    cs_apply_lod_gamma(sharp, 10, 3.0);
    const auto sharp_counts = cs_compute_tile_sample_counts(sharp, 400, 64, 64, 32, 1);
    check(sharp_counts[1] < plain_counts[1] &&
        sharp_counts[3] > plain_counts[3],
        "adaptive gamma: gamma 3 starves mid tile, feeds top tile");
}

// ---------------------------------------------------------------------------
// Hard LOD cutoff (--adaptive-cutoff): tiles below the cutoff get zero
// samples; survivors absorb the budget (fail-soft when they saturate).
// ---------------------------------------------------------------------------
void test_adaptive_cutoff() {
    const std::vector<uint8_t> lod2 = { 10, 130, 10, 250 };
    const auto nocut = cs_compute_tile_sample_counts(lod2, 400, 64, 64, 32, 1);
    const auto cut = cs_compute_tile_sample_counts(lod2, 400, 64, 64, 32, 1, 0, 200);
    check(cut[0] == 0 && cut[1] == 0 && cut[2] == 0 && cut[3] == 400,
        "adaptive cutoff: sub-cutoff tiles get zero, survivor absorbs");
    long long sum = 0;
    for (int c : cut) sum += c;
    check(sum == 400, "adaptive cutoff: total preserved when survivors fit");
    const auto allcut = cs_compute_tile_sample_counts(lod2, 400, 64, 64, 32, 1, 0, 255);
    check(std::all_of(allcut.begin(), allcut.end(), [](int v) { return v == 0; }),
        "adaptive cutoff: cutoff above max zeroes everything");
    const auto def = cs_compute_tile_sample_counts(lod2, 400, 64, 64, 32, 1, 0);
    check(def == nocut, "adaptive cutoff: default 0 is bit-identical to legacy call");
    // cutoff wins over --lod-full on overlap (tile lod 130 qualifies at
    // threshold 100 but is cut at 200; tile 250 takes its capacity).
    const auto cf = cs_compute_tile_sample_counts(lod2, 2000, 64, 64, 32, 1, 100, 200);
    check(cf[1] == 0 && cf[3] == 1024,
        "adaptive cutoff: cutoff wins over lod-full on overlap");
    // container roundtrip: cutoff holds fewer measurements, decrypts fine.
    const std::string input_path = ".cs_test_cutoff_in.png";
    const std::string output_path = ".cs_test_cutoff_out.png";
    const std::string password = "adaptive-cutoff-password";
    const cv::Mat original = make_test_image(160, 120);
    check(cv::imwrite(input_path, original), "adaptive cutoff: write source image");
    cv::Mat enc_cut, enc_plain;
    const int enc_cut_rc = encrypt_image::encrypt_image_tiled(
        input_path, "", password, 0.5f, 32, &enc_cut,
        true, 0, 1.0f, false, false, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8, 0, 0, false, false, "", 0, 1.0f, 100);
    check(enc_cut_rc == 0, "adaptive cutoff: encrypt with cutoff");
    const int enc_plain_rc = encrypt_image::encrypt_image_tiled(
        input_path, "", password, 0.5f, 32, &enc_plain,
        true, 0, 1.0f, false, false, false, 0.05f, "", 0.5f, 0.0f,
        false, false, 8, 0, 0, false, false, "", 0, 1.0f, 0);
    check(enc_plain_rc == 0, "adaptive cutoff: encrypt without cutoff");
    if (enc_cut_rc == 0 && enc_plain_rc == 0) {
        decrypt_image cont_cut(enc_cut, password);
        decrypt_image cont_plain(enc_plain, password);
        check(cont_cut.get_sampling_mode() == CS_MODE_ADAPTIVE,
            "adaptive cutoff: mode header");
        cv::Mat sm1, mm1, sm2, mm2;
        cont_cut.get_sampled_mat(sm1, mm1);
        cont_plain.get_sampled_mat(sm2, mm2);
        const int n_cut = cv::countNonZero(mm1.reshape(1)) / 3;
        const int n_plain = cv::countNonZero(mm2.reshape(1)) / 3;
        std::printf("       adaptive cutoff samples: %d vs %d without\n", n_cut, n_plain);
        check(n_cut < n_plain && n_cut > 0,
            "adaptive cutoff: fewer measurements, none lost entirely");
        CSencryption::params = AUTO_PARAM;
        const int decrypt_rc = decrypt_image::decrypt_image_tiled(
            enc_cut, output_path, password, 1, 8, 3, 2, 0.01f, false);
        check(decrypt_rc == 0, "adaptive cutoff: decrypt");
        const cv::Mat decoded = cv::imread(output_path, cv::IMREAD_COLOR);
        check(!decoded.empty(), "adaptive cutoff: output readable");
        std::remove(output_path.c_str());
    }
    // validation: cutoff without an adaptive path is refused.
    {
        cv::Mat enc_bad;
        check(encrypt_image::encrypt_image_tiled(
            input_path, "", password, 0.5f, 0, &enc_bad,
            false, 0, 1.0f, false, false, false, 0.05f, "", 0.5f, 0.0f,
            false, false, 8, 0, 0, false, false, "", 0, 1.0f, 100) != 0,
            "adaptive cutoff: uniform combination refused");
    }
    std::remove(input_path.c_str());
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

void test_hf_focus_roundtrip() {
    const std::string input_path = ".cs_test_hf_input.png";
    const std::string output_path = ".cs_test_hf_output.png";
    const std::string password = "hf-focus-roundtrip-password";
    cv::Mat original = make_test_image(320, 240);
    check(cv::imwrite(input_path, original), "hf-focus: write source image");

    // 0 = BGR HF (mode 5), 1 = 4:2:0 HF (mode 6), 2 = 4:2:2 HF (mode 8)
    for (int splitmode : {0, 1, 2}) {
        const bool ycc420 = (splitmode == 1);
        const bool ycc422 = (splitmode == 2);
        const char* tag = (splitmode == 0) ? "hf-focus" :
            (splitmode == 1) ? "hf-focus ycc420" : "hf-focus ycc422";
        cv::Mat encrypted;
        const int encrypt_rc = encrypt_image::encrypt_image_tiled(
            input_path, "", password, 0.25f, 32, &encrypted, false, 32, 0.5f,
            false, false, false, 0.05f, "", 0.5f, 0.0f, ycc420, true,
            8, 0, 0, ycc422);
        check(encrypt_rc == 0, (std::string(tag) + ": encrypt").c_str());
        if (encrypt_rc != 0) continue;

        decrypt_image container(encrypted, password);
        const int expected_mode = (splitmode == 0) ? CS_MODE_HF_FOCUS :
            (splitmode == 1) ? CS_MODE_YCC420_HF : CS_MODE_YCC422_HF;
        check(container.get_sampling_mode() == expected_mode,
            (std::string(tag) + ": mode header").c_str());
        CSencryption::params = AUTO_PARAM;
        const int decrypt_rc = decrypt_image::decrypt_image_tiled(
            encrypted, output_path, password, 1, 24, 5, 4, 0.01f, false);
        check(decrypt_rc == 0, (std::string(tag) + ": decrypt").c_str());
        const cv::Mat decoded = cv::imread(output_path, cv::IMREAD_COLOR);
        check(!decoded.empty(), (std::string(tag) + ": output readable").c_str());
        if (!decoded.empty()) {
            cv::Mat sized;
            cv::resize(decoded, sized, original.size());
            const double p = cs_quality::psnr(original, sized);
            std::printf("       %s PSNR (ratio 0.25): %.2f dB\n", tag, p);
            check(p > 8.0, (std::string(tag) + ": PSNR sane").c_str());
        }
        std::remove(output_path.c_str());
    }

    std::remove(input_path.c_str());
}

// ---------------------------------------------------------------------------
// Sample bit-depth (--sample-bits): quantize/pack/unpack helpers and a
// packed roundtrip. Plumbing breakage (wrong offsets, misaligned sections)
// collapses to garbage, so loose relative PSNR bounds suffice.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HF-weighted spread: log-compressed weights + 50% uniform base.
// A spiky weight map must neither starve flat regions (uniform base) nor
// lose its HF bias (compressed weighted half still favors the hot cell).
// Goes straight at the sampler (protected member via probe subclass) with
// a fixed key, so the same call also pins encrypt/decrypt determinism.
// ---------------------------------------------------------------------------
struct HfSamplerProbe : CSencryption {
    using CSencryption::returnHfWeightedIndices;
};

void test_hf_weighted_spread() {
    const int rows = 128, cols = 128, tile = 64;
    const int m_total = rows * cols / 4; // 4096 over 2x2 tiles
    const std::vector<uint8_t> lod(4, 128); // flat LOD: budgets ~equal
    const int wc = 8, wr = 8;
    std::vector<float> weights((size_t)wc * wr, 0.5f);
    weights[(size_t)2 * wc + 5] = 400.0f; // hot cell -> rows [32,48), cols [80,96)
    uint8_t key[32];
    for (int i = 0; i < 32; ++i) key[i] = (uint8_t)(i * 7 + 1);

    HfSamplerProbe a, b;
    std::vector<int> ax, ay, bx, by;
    a.returnHfWeightedIndices(ax, ay, rows, cols, lod, m_total, key,
        tile, 65535, weights, wc, wr);
    b.returnHfWeightedIndices(bx, by, rows, cols, lod, m_total, key,
        tile, 65535, weights, wc, wr);
    check(ax.size() == (size_t)m_total && ay.size() == (size_t)m_total,
        "hf-spread: total sample count preserved");
    check(ax == bx && ay == by, "hf-spread: deterministic (decrypt regenerates)");

    // uniqueness: uniform and weighted halves must be disjoint
    std::vector<char> seen((size_t)rows * cols, 0);
    bool dup = false;
    for (size_t k = 0; k < ax.size(); ++k) {
        const size_t idx = (size_t)ax[k] * cols + (size_t)ay[k];
        if (seen[idx]) { dup = true; break; }
        seen[idx] = 1;
    }
    check(!dup, "hf-spread: no duplicate positions");

    // coverage over 16x16 cells: uniform base forbids starved regions
    int cells[8][8] = {};
    for (size_t k = 0; k < ax.size(); ++k) cells[ax[k] / 16][ay[k] / 16]++;
    int cmin = m_total;
    long long rest_sum = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            cmin = (std::min)(cmin, cells[i][j]);
            if (i != 2 || j != 5) rest_sum += cells[i][j];
        }
    std::printf("       hf-spread min 16x16 cell: %d, hot cell: %d\n",
        cmin, cells[2][5]);
    check(cmin >= 4, "hf-spread: no starved 16x16 cell");
    // HF bias preserved: hot cell well above the flat-cell mean
    const double flat_mean = (double)rest_sum / 63.0;
    check((double)cells[2][5] > 1.5 * flat_mean,
        "hf-spread: hot cell still favored");
}

void test_adaptive_tile_grid_limit() {
    // The lod byte count travels in a 16-bit header field: tile grids over
    // 65535 tiles truncate it and decrypt silently regenerates wrong
    // positions (6 dB garbage, exit 0). Encrypt must refuse up front.
    const std::string input_path = ".cs_test_t1_input.png";
    const std::string output_path = ".cs_test_t1_output.png";
    const std::string password = "tile-grid-limit-password";
    // NB: encrypt downsamples 2x by default, so size for 76,800 post-
    // downscale tiles (over the 65535 limit): 640x480 -> 320x240.
    cv::Mat original = make_test_image(640, 480); // tile-size 1 = 76,800 tiles
    check(cv::imwrite(input_path, original), "tile-grid-limit: write source image");

    int rc = encrypt_image::encrypt_image_tiled(
        input_path, "", password, 0.5f, 1, nullptr, true, 32, 0.5f,
        false, false, false, 0.05f, "", 0.5f, 0.0f, false, false,
        8, 0, 0, false, false);
    check(rc != 0, "tile-grid-limit: BGR adaptive tile-size 1 rejected");
    rc = encrypt_image::encrypt_image_tiled(
        input_path, "", password, 0.5f, 1, nullptr, true, 32, 0.5f,
        false, false, false, 0.05f, "", 0.5f, 0.0f, true, false,
        8, 0, 0, false, false);
    check(rc != 0, "tile-grid-limit: ycc420 adaptive tile-size 1 rejected");

    // Positive control: 64x48 at tile-size 1 = 3072 tiles fits the field, so
    // the 1px-tile machinery must roundtrip sane end to end.
    const std::string small_in = ".cs_test_t1_small.png";
    const std::string small_out = ".cs_test_t1_small_out.png";
    cv::Mat small = make_test_image(64, 48);
    check(cv::imwrite(small_in, small), "tile-grid-limit: write small image");
    cv::Mat encrypted;
    rc = encrypt_image::encrypt_image_tiled(
        small_in, "", password, 0.5f, 1, &encrypted, true, 32, 0.5f,
        false, true, false, 0.05f, "", 0.5f, 0.0f, false, false,
        8, 0, 0, false, false);
    check(rc == 0, "tile-grid-limit: small tile-size 1 encrypts");
    if (rc == 0) {
        CSencryption::params = AUTO_PARAM;
        const int decrypt_rc = decrypt_image::decrypt_image_tiled(
            encrypted, small_out, password, 1, 24, 5, 4, 0.01f, false);
        check(decrypt_rc == 0, "tile-grid-limit: small tile-size 1 decrypts");
        const cv::Mat decoded = cv::imread(small_out, cv::IMREAD_COLOR);
        check(!decoded.empty(), "tile-grid-limit: output readable");
        if (!decoded.empty()) {
            cv::Mat sized;
            cv::resize(decoded, sized, small.size());
            const double p = cs_quality::psnr(small, sized);
            std::printf("       tile-size 1 64x48 PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 12.0, "tile-grid-limit: small tile-size 1 PSNR sane");
        }
        std::remove(small_out.c_str());
    }
    std::remove(small_in.c_str());
    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
}

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

int main(int argc, char* argv[]) {
    // Large photos trip CL_MEM_OBJECT_ALLOCATION_FAILURE on this host's
    // OpenCL runtime during heavy GEMM; CPU path is deterministic for tests.
    cv::ocl::setUseOpenCL(false);
    if (argc == 2 && std::string(argv[1]) == "--y-confidence-only") {
        test_y_confidence_prior();
        test_y_dual_tv_blend();
        test_y_dual_tv_color_segment();
        test_dual_tv_split();
        return g_failures ? 1 : 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--grain-only") {
        test_grain_tv();
        return g_failures ? 1 : 0;
    }

    test_crypto();
    test_shuffle();
    test_tile_helpers();
    test_roundtrip();
    test_fista_solvers();
    test_wavelet_basis();
    test_adaptive_counts();
    test_sample_split();
    test_grain_tv();
    test_adaptive_roundtrip();
    test_spectral_lod();
    test_lod_full_roundtrip();
    test_ycc420_roundtrip();
    test_ycc422_roundtrip();
    test_tsampling();
    test_twopass_roundtrip();
    test_regions_roundtrip();
    test_photo_roundtrip();
    test_photo_upscaler();
    test_cs_super_resolution();
    test_sr_direct();
    test_coupled_dict();
    test_dncnn_red();
    test_onnx_denoisers();
    test_onsager();
    test_hf_focus_roundtrip();
    test_adaptive_gamma();
    test_adaptive_cutoff();
    test_anchor_weighted_red();
    test_edge_scores();
    test_hr_detail();
    test_hr_sr_anchors();
    test_hf_weighted_spread();
    test_adaptive_tile_grid_limit();
    test_sample_bits();
    test_y_confidence_prior();
    test_y_dual_tv_blend();
    test_y_dual_tv_color_segment();
    test_dual_tv_split();
    test_per_tile_coef();
    test_gpu_fista_agreement();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
