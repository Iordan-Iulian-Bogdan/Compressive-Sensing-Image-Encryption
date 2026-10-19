// Backend tests: YCC 4:2:0 / 4:2:2, region-guided and segmentation sampling.
#include "test_common.hpp"

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

void test_seg_auto() {
    // Built-in segmentation: deterministic ids, sane classes, sane rates.
    cv::Mat img = make_test_image(192, 144);
    {
        cv::Mat a, b;
        cs_seg_auto_ids(img, 8, 144, 192, a);
        cs_seg_auto_ids(img, 8, 144, 192, b);
        check(a.size() == img.size() && a.type() == CV_8U, "seg-auto: ids geometry/type");
        bool same = true;
        for (int y = 0; y < a.rows && same; ++y) {
            const uint8_t* ra = a.ptr<uint8_t>(y);
            const uint8_t* rb = b.ptr<uint8_t>(y);
            for (int x = 0; x < a.cols; ++x) {
                if (ra[x] != rb[x] || ra[x] >= 8) { same = false; break; }
            }
        }
        check(same, "seg-auto: deterministic, classes in range");
    }
    // Degenerate inputs rejected, tiny images accepted.
    {
        cv::Mat ids;
        bool threw = false;
        try { cs_seg_auto_ids(img, 1, 144, 192, ids); }
        catch (...) { threw = true; }
        check(threw, "seg-auto: 1 class rejected");
        threw = false;
        try { cs_seg_auto_ids(cv::Mat(), 4, 0, 0, ids); }
        catch (...) { threw = true; }
        check(threw, "seg-auto: empty image rejected");
        cv::Mat tiny(24, 32, CV_8UC3, cv::Scalar(90, 90, 90));
        cs_seg_auto_ids(tiny, 2, 24, 32, ids);
        check(ids.rows == 24 && ids.cols == 32, "seg-auto: tiny image works");
    }
    // Auto rates: range, flat-is-uniform, detail ordering.
    {
        cv::Mat ids;
        cs_seg_auto_ids(img, 4, 144, 192, ids);
        const std::vector<float> r = cs_seg_auto_rates(img, ids, 4);
        bool range = r.size() == 4;
        for (float v : r) range = range && v >= 0.0f && v <= 1.0f;
        check(range, "seg-auto rates: 4 values in [0,1]");
        cv::Mat flat(64, 64, CV_8UC3, cv::Scalar(120, 120, 120));
        cv::Mat fids;
        cs_seg_auto_ids(flat, 2, 64, 64, fids);
        const std::vector<float> fr = cs_seg_auto_rates(flat, fids, 2);
        check(fr.size() == 2 && fr[0] == 1.0f && fr[1] == 1.0f, "seg-auto rates: flat image uniform 1.0");
        // left half flat, right half noisy: noisy class must win
        cv::Mat half(64, 64, CV_8UC3, cv::Scalar(120, 120, 120));
        cv::randu(half(cv::Rect(32, 0, 32, 64)), cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::Mat hids(64, 64, CV_8U, cv::Scalar(0));
        hids(cv::Rect(32, 0, 32, 64)).setTo(1);
        const std::vector<float> hr = cs_seg_auto_rates(half, hids, 2);
        check(hr.size() == 2 && hr[1] > hr[0], "seg-auto rates: detail class outranks flat");
    }
    // End-to-end seg-auto roundtrip (mode 10, auto rates).
    const std::string password = "seg-auto-test-password";
    const std::string tmp_in = ".cs_test_segauto_in.png";
    cv::imwrite(tmp_in, img);
    {
        cv::Mat encrypted;
        check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &encrypted,
            false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f, false, false,
            8, 0, 0, false, false, "", 0, 1.0f, 0, "", "", 0, "points", 8) == 0,
            "seg-auto roundtrip: encrypt");
        check(!encrypted.empty(), "seg-auto roundtrip: encrypt to memory");
        const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_SEGMASK, "seg-auto roundtrip: header mode=10");
        const std::string tmp_out = ".cs_test_segauto_out.png";
        check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "seg-auto roundtrip: decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "seg-auto roundtrip: output readable");
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, img.size());
            const double p = cs_quality::psnr(img, sized);
            std::printf("       seg-auto roundtrip PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 9.0, "seg-auto roundtrip: PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }
    // YCC seg-auto smoke (mode 11, auto rates).
    {
        cv::Mat encrypted;
        check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &encrypted,
            false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f, true, false,
            8, 0, 0, false, false, "", 0, 1.0f, 0, "", "", 0, "points", 6) == 0,
            "seg-auto ycc: encrypt");
        const cv::Mat flat = encrypted.reshape(0, (int)encrypted.total());
        check(flat.data[CS_OFF_PAD] == CS_MODE_YCC420_SEG, "seg-auto ycc: header mode=11");
        const std::string tmp_out = ".cs_test_segauto_ycc.png";
        check(decrypt_image::decrypt_image_tiled(encrypted, tmp_out, password, 4, 24, 5, 4, 0.01f, false) == 0,
            "seg-auto ycc: decrypt succeeds");
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        check(!dec.empty(), "seg-auto ycc: output readable");
        if (!dec.empty()) {
            cv::Mat sized;
            cv::resize(dec, sized, img.size());
            const double p = cs_quality::psnr(img, sized);
            std::printf("       seg-auto ycc PSNR (ratio 0.5): %.2f dB\n", p);
            check(p > 9.0, "seg-auto ycc: PSNR sane");
        }
        std::remove(tmp_out.c_str());
    }
    // Validation: bad class count and mask+auto conflict fail before solving.
    check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, nullptr,
        false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f, false, false,
        8, 0, 0, false, false, "", 0, 1.0f, 0, "", "", 0, "points", 1) != 0,
        "seg-auto: 1 class rejected");
    {
        // rates override with auto: accepted (trailing classes score 0)
        cv::Mat encrypted;
        check(encrypt_image::encrypt_image_tiled(tmp_in, "", password, 0.5f, 0, &encrypted,
            false, 32, 0.5f, false, false, false, 0.05f, "", 0.5f, 0.0f, false, false,
            8, 0, 0, false, false, "", 0, 1.0f, 0, "", "0.9,0.1", 0, "points", 4) == 0,
            "seg-auto: --seg-rates override accepted");
    }
    std::remove(tmp_in.c_str());
}

