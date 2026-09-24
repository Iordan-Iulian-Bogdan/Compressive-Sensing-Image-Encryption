// Test suite for the CS image encryption backend.
// Covers: crypto primitives (KDF determinism, seal/verify, tamper detection),
// index shuffle determinism, tiling helpers, and a full encrypt->decrypt
// roundtrip scored by PSNR + SSIM.
#include "crypto_utils.hpp"
#include "image_encryption.hpp"
#include "image_decryption.hpp"
#include "quality_utils.hpp"

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
// 4b. FISTA solvers (reweighted-L1 single + SOMP-structured joint)
// ---------------------------------------------------------------------------
void test_fista_solvers() {
    int s = -1;
    check(cs_solver_from_name("owlqn", s) == 0 && s == CS_SOLVER_OWLQN, "fista: owlqn parses to 0");
    check(cs_solver_from_name("fista", s) == 0 && s == CS_SOLVER_FISTA, "fista: fista parses to 1");
    check(cs_solver_from_name("joint", s) == 0 && s == CS_SOLVER_FISTA_JOINT, "fista: joint parses to 2");
    check(cs_solver_from_name("somp", s) == 0 && s == CS_SOLVER_FISTA_JOINT, "fista: somp aliases joint");
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
            encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, false, 0.0f, "", false, -1, solvers[k]);
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
            encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, false, 0.0f, "", false, -1,
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
    // invalid values rejected before the solve
    check(decrypt_image::decrypt_image_tiled(
        encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, false, 0.0f, "", false, -1,
        CS_SOLVER_FISTA, 501, 2) == -1, "fista: fista-iters > 500 rejected");
    check(decrypt_image::decrypt_image_tiled(
        encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false, false, 0.0f, "", false, -1,
        99, 0, 2) == -1, "fista: unknown solver id rejected");

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
    // At 50% with the weak suite solve params (5 iters, coef 0.01) even
    // uniform random lands ~16 dB / ~0.52 SSIM on this synthetic; adaptive
    // tracks that band. Floors sit well above a broken index pipeline
    // (single-digit dB, near-zero SSIM) to catch misaligned lod offsets or
    // wrong measurement origins without over-fitting to solver noise.
    check(p > 14.0, "adaptive roundtrip: PSNR > 14 dB at ratio 0.5");
    if (s >= 0.0) check(s > 0.45, "adaptive roundtrip: SSIM > 0.45");
    else check(false, "adaptive roundtrip: SSIM unavailable");

    // second encrypt with same inputs must produce a working decrypt too
    // (salt is random, so ciphertexts differ — only quality path is checked)
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
// 5b. two-pass coarse-to-fine sampling roundtrip
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
    test_adaptive_counts();
    test_adaptive_roundtrip();
    test_twopass_roundtrip();
    test_regions_roundtrip();
    test_photo_roundtrip();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}




