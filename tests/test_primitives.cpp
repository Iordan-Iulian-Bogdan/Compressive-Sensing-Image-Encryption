// Backend tests: crypto, shuffle, tiling, base roundtrip, sample split/bits, T-sampling.
#include "test_common.hpp"

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

    // YCC420 with T-tetromino luma (full-res manual like the CLI runs).
    // Sparse YCC needs iterations to converge (same as uniform YCC at
    // matched ratios); tmeans uses 40, tcluster defaults.
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_ycc420_t(0.5f, password, 8, 0, CS_MODE_YCC420, CS_MODE_YCC420_TCLUSTER);
        cv::Mat container = enc.get_mat();
        check(!container.empty(), "tsampling: ycc420 tcluster encrypt to memory");
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_YCC420_TCLUSTER, "tsampling: ycc420 tcluster header mode=15");
        const std::string tmp_out = ".cs_test_tsampling_ytc.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 1, 24, 5, 4, 0.01f, false,
            0.0f, true) == 0, "tsampling: ycc420 tcluster decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: ycc420 tcluster output readable");
        if (!dec.empty()) {
            const double p = cs_quality::psnr(original, dec);
            std::printf("       tsampling ycc420 tcluster PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 13.0, "tsampling: ycc420 tcluster PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_ycc420_t(0.25f, password, 8, 0, CS_MODE_YCC420, CS_MODE_YCC420_TMEANS);
        cv::Mat container = enc.get_mat();
        check(!container.empty(), "tsampling: ycc420 tmeans encrypt to memory");
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_YCC420_TMEANS, "tsampling: ycc420 tmeans header mode=16");
        const std::string tmp_out = ".cs_test_tsampling_ytm.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 1, 24, 40, 4, 0.01f, false,
            0.0f, true) == 0, "tsampling: ycc420 tmeans decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: ycc420 tmeans output readable");
        if (!dec.empty()) {
            const double p = cs_quality::psnr(original, dec);
            std::printf("       tsampling ycc420 tmeans PSNR (ratio 0.25, it40): %.2f dB\n", p);
            // Sparse YCC needs iterations to converge (same as uniform YCC
            // at matched ratios); the bar pins the wired-up grouped path
            // (~9 dB garbage when miswired), not content quality.
            check(p > 16.0, "tsampling: ycc420 tmeans PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_ycc420_t(0.5f, password, 8, 0, CS_MODE_YCC422, CS_MODE_YCC422_TCLUSTER);
        cv::Mat container = enc.get_mat();
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_YCC422_TCLUSTER, "tsampling: ycc422 tcluster header mode=17");
        const std::string tmp_out = ".cs_test_tsampling_ytc422.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 1, 24, 5, 4, 0.01f, false,
            0.0f, true) == 0, "tsampling: ycc422 tcluster decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: ycc422 tcluster output readable");
        if (!dec.empty()) {
            const double p = cs_quality::psnr(original, dec);
            std::printf("       tsampling ycc422 tcluster PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 13.0, "tsampling: ycc422 tcluster PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }
    {
        encrypt_image enc(tmp_in);
        enc.encrypt_ycc420_t(0.25f, password, 8, 0, CS_MODE_YCC422, CS_MODE_YCC422_TMEANS);
        cv::Mat container = enc.get_mat();
        const cv::Mat flat = container.reshape(0, (int)container.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_YCC422_TMEANS, "tsampling: ycc422 tmeans header mode=18");
        const std::string tmp_out = ".cs_test_tsampling_ytm422.png";
        check(decrypt_image::decrypt_image_tiled(container, tmp_out, password, 1, 24, 40, 4, 0.01f, false,
            0.0f, true) == 0, "tsampling: ycc422 tmeans decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "tsampling: ycc422 tmeans output readable");
        if (!dec.empty()) {
            const double p = cs_quality::psnr(original, dec);
            std::printf("       tsampling ycc422 tmeans PSNR (ratio 0.25, it40): %.2f dB\n", p);
            // Same iteration physics as the 420 sibling (see above).
            check(p > 16.0, "tsampling: ycc422 tmeans PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }

    std::remove(tmp_in.c_str());
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

