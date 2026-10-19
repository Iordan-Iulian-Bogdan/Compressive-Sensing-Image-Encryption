// Backend tests: adaptive LOD, spectral/lod-full, gamma/cutoff, HF focus, two-pass, per-tile coef.
#include "test_common.hpp"

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

