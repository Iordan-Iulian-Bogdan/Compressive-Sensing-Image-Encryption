// Backend tests: crypto, shuffle, tiling, production roundtrips, sample bits.
#include "test_common.hpp"

// Regression floors for the production roundtrips (measured 2026-11-09 on the GPU build: 960x720 ratio 0.5
// 32.3 dB / SSIM 0.974 single- and multi-scale; 480x360 sample-bits 27.1..29.4 dB).
static const double ROUNDTRIP_MIN_PSNR = 30.0, ROUNDTRIP_MIN_SSIM = 0.95, SAMPLE_BITS_MIN_PSNR = 25.0;

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
    check(cs_write_header(buf.data(), buf.size(), meta, k1, salt), "crypto: v3 header write");
    check(buf[CS_OFF_VERSION] == 3, "crypto: v3 version byte");
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
        info.version == 3 && info.key_valid && !info.auth_failed &&
        info.m == 1234567 && info.rows == 2016 && info.cols == 1512 &&
        info.org_h == 3024 && info.org_w == 4032, "crypto: v3 parse roundtrip with password");

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

    // odd-size parity: optional 6th field, absent for even images (text unchanged)
    check(cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 0) == meta, "crypto: parity 0 keeps the 5-field metadata");
    check(tile_info.odd_parity == 0, "crypto: 5-field metadata parses as parity 0");
    for (int p = 1; p <= 3; ++p) {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header_plain(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, p));
        CsHeaderInfo pinfo;
        check(cs_parse_header(pbuf.data(), pbuf.size(), "", pinfo) && pinfo.odd_parity == p &&
            pinfo.org_h == 3024 && pinfo.org_w == 4032, "crypto: parity field roundtrips");
    }
    {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header_plain(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 4));
        CsHeaderInfo pinfo;
        check(!cs_parse_header(pbuf.data(), pbuf.size(), "", pinfo), "crypto: parity > 3 rejected");
    }

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
// 4. full production roundtrip (single- and multi-scale) + PSNR + auth rejection
// ---------------------------------------------------------------------------
void test_roundtrip() {
    const std::string password = "roundtrip-test-password";
    const cv::Mat original = make_test_image(960, 720);
    const std::string tmp_in = ".cs_test_input.png";
    const std::string tmp_enc = ".cs_test_input.enc.png";
    const std::string tmp_out = ".cs_test_output.png";

    check(prod_encrypt(".cs_test_missing.png", tmp_enc, password, 0.5f) != 0, "roundtrip: missing input file rejected");
    cv::imwrite(tmp_in, original);
    for (const bool ms : { false, true }) {
        const char* tag = ms ? "roundtrip multi-scale" : "roundtrip single-scale";
        char name[96];
        std::snprintf(name, sizeof(name), "%s: encrypt", tag);
        check(prod_encrypt(tmp_in, tmp_enc, password, 0.5f, ms) == 0, name);
        const cv::Mat enc = read_container(tmp_enc);
        const int want_mode = ms ? CS_MODE_MULTISCALE : CS_MODE_YCC420_SEG_HRDETAIL;
        std::snprintf(name, sizeof(name), "%s: header mode %d", tag, want_mode);
        check(!enc.empty() && enc.data[CS_OFF_PAD] == want_mode, name);

        // wrong password and a tampered body must fail authentication before the solve
        // (-2 from the single-scale path, -1 from the multi-scale wrapper)
        const int want_rc = ms ? -1 : -2;
        std::snprintf(name, sizeof(name), "%s: wrong password rejected with %d", tag, want_rc);
        check(prod_decrypt(enc, tmp_out, "wrong-password-99") == want_rc, name);
        cv::Mat tampered = enc.clone();
        tampered.data[tampered.total() * 3 - 1] ^= 0x80;
        std::snprintf(name, sizeof(name), "%s: tampered container rejected with %d", tag, want_rc);
        check(prod_decrypt(tampered, tmp_out, password) == want_rc, name);

        std::snprintf(name, sizeof(name), "%s: decrypt succeeds", tag);
        check(prod_decrypt(enc, tmp_out, password) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(name, sizeof(name), "%s: decodes at the native size", tag);
        check(dec.size() == original.size(), name);
        if (dec.size() != original.size()) continue;
        const double p = cs_quality::psnr(original, dec);
        const double s = cs_quality::ssim(original, dec);
        std::printf("       %s PSNR: %.2f dB, SSIM: %.4f\n", tag, p, s);
        // Regression guards, not quality benchmarks: a broken index pipeline (wrong sampling,
        // misaligned offsets) collapses to single-digit dB and SSIM well below 0.5.
        std::snprintf(name, sizeof(name), "%s: PSNR > %.0f dB", tag, ROUNDTRIP_MIN_PSNR);
        check(p > ROUNDTRIP_MIN_PSNR, name);
        std::snprintf(name, sizeof(name), "%s: SSIM > %.2f", tag, ROUNDTRIP_MIN_SSIM);
        if (s >= 0.0) check(s > ROUNDTRIP_MIN_SSIM, name);
        else check(false, "roundtrip: SSIM unavailable");
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 4a. odd-size roundtrip: the 2x pipeline keeps the native size and pixel grid
// ---------------------------------------------------------------------------
void test_odd_size_roundtrip() {
    const std::string password = "odd-size-test-password";
    const std::string tmp_in = ".cs_test_odd_in.png";
    const std::string tmp_enc = ".cs_test_odd_in.enc.png";
    const std::string tmp_out = ".cs_test_odd_out.png";
    const cv::Mat full = make_test_image(321, 241);
    double p_even = 0.0;
    const int sizes[4][2] = { { 240, 320 }, { 241, 320 }, { 240, 321 }, { 241, 321 } };   // rows, cols
    for (const auto& hw : sizes) {
        const cv::Mat original = full(cv::Rect(0, 0, hw[1], hw[0])).clone();
        cv::imwrite(tmp_in, original);
        char tag[96];
        bool ok = prod_encrypt(tmp_in, tmp_enc, password, 0.5f) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d encrypt", hw[0], hw[1]);
        check(ok, tag);
        if (!ok) continue;
        ok = prod_decrypt(read_container(tmp_enc), tmp_out, password) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d decrypt", hw[0], hw[1]);
        check(ok, tag);
        if (!ok) continue;
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d decodes at the native size", hw[0], hw[1]);
        check(dec.size() == original.size(), tag);
        if (dec.size() != original.size()) continue;
        // pixel-aligned with the original (no resample): about as good as the even crop of the same image (the
        // production decode measures up to 1.3 dB lower on these small crops; a shifted grid costs several dB)
        const double p = cs_quality::psnr(original, dec);
        std::printf("       odd size %dx%d PSNR: %.2f dB\n", hw[0], hw[1], p);
        if (hw[0] % 2 == 0 && hw[1] % 2 == 0) p_even = p;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d PSNR within 2 dB of the even crop", hw[0], hw[1]);
        check(p > p_even - 2.0, tag);
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
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
    const std::string tmp_enc = ".cs_test_sb_in.enc.png";
    const std::string tmp_out = ".cs_test_sb_out.png";
    const cv::Mat original = make_test_image(480, 360);
    cv::imwrite(tmp_in, original);

    // encode at the given depths (0 = the default 6-bit luma / 8-bit chroma), check the header, decode, score
    const auto run = [&](int bits, int chroma_bits, const char* tag, int want_y, int want_c, cv::Mat& enc) {
        char name[96];
        std::snprintf(name, sizeof(name), "sample-bits %s: encrypt", tag);
        const bool ok = prod_encrypt(tmp_in, tmp_enc, password, 0.5f, false, 2000, bits, chroma_bits) == 0;
        check(ok, name);
        if (!ok) return -1.0;
        enc = read_container(tmp_enc);
        {
            decrypt_image probe(enc, password);
            std::snprintf(name, sizeof(name), "sample-bits %s: header reads back %d/%d", tag, want_y, want_c);
            check(probe.get_sample_bits() == want_y && probe.get_sample_bits_chroma() == want_c, name);
        }
        std::snprintf(name, sizeof(name), "sample-bits %s: decrypt", tag);
        check(prod_decrypt(enc, tmp_out, password) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        return dec.size() == original.size() ? cs_quality::psnr(original, dec) : -1.0;
    };
    cv::Mat enc8, enc4, enc_def, enc64;
    const double p8 = run(8, 0, "8-bit", 8, 8, enc8);
    const double p4 = run(4, 0, "4-bit", 4, 4, enc4);
    const double pd = run(0, 0, "default", 6, 8, enc_def);
    const double p64 = run(6, 4, "split 6/4", 6, 4, enc64);
    std::printf("       sample-bits PSNR 8-bit: %.2f dB, 4-bit: %.2f dB, default 6/8: %.2f dB, 6/4: %.2f dB\n",
        p8, p4, pd, p64);
    check(enc4.total() < enc8.total(), "sample-bits: 4-bit container is smaller");
    check(prod_decrypt(enc4, tmp_out, "wrong-password-99") == -2, "sample-bits: packed container rejects wrong password");
    // Regression guards, not quality benchmarks: plumbing breakage (wrong offsets, misaligned sections)
    // collapses to single-digit dB.
    check(std::isfinite(p8) && p8 > SAMPLE_BITS_MIN_PSNR, "sample-bits: 8-bit PSNR sane");
    check(std::isfinite(p4) && p4 > p8 - 10.0, "sample-bits: 4-bit PSNR within its quantization budget");
    check(std::isfinite(pd) && pd > SAMPLE_BITS_MIN_PSNR, "sample-bits: default 6-bit PSNR sane");
    check(std::isfinite(p64) && p64 > SAMPLE_BITS_MIN_PSNR, "sample-bits: split-depth PSNR sane");
    {
        cv::Mat tampered = enc64.clone();
        tampered.data[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] ^= 1;
        bool rejected = false;
        try { decrypt_image probe(tampered, password); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "sample-bits: tampered split-depth header rejected");
    }

    // Out-of-range depths are rejected, not silently ignored.
    check(prod_encrypt(tmp_in, tmp_enc, password, 0.5f, false, 2000, 9) == -1, "sample-bits: out-of-range depth rejected");

    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
}

