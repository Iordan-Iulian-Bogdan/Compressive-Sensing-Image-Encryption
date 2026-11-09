// Backend tests: built-in segmentation and encoder geometry.
#include "test_common.hpp"

// Measured 24.3..24.8 dB at ratio 1.0 (2026-11-09); the alignment check below is the real test.
static const double GEOMETRY_MIN_PSNR = 22.0;

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
}

// Encoder geometry: the container's samples must come from the image at the positions the
// decoder puts them. Sizes not divisible by 8 used to be stretched to a multiple of 8 for the
// sampling pixels while the sample positions stayed on the unstretched grid, shifting content
// by up to 4 px (5-9 dB lower decode PSNR on real photos).
void test_encoder_geometry() {
    // Sharp-edged content makes any shift cost PSNR.
    const std::string password = "geometry-test-password";
    // 100x76 and 98x74 are not multiples of 8; 96x72 is the control
    for (const cv::Size sz : { cv::Size(100, 76), cv::Size(98, 74), cv::Size(96, 72) }) {
        cv::Mat img(sz, CV_8UC3, cv::Scalar(90, 100, 110));
        cv::rectangle(img, cv::Rect(sz.width / 5, sz.height / 5, sz.width / 2, sz.height / 2), cv::Scalar(30, 40, 230), cv::FILLED);
        cv::rectangle(img, cv::Rect(sz.width * 3 / 5, sz.height * 3 / 5, sz.width / 4, sz.height / 4), cv::Scalar(230, 220, 40), cv::FILLED);
        cv::circle(img, cv::Point(sz.width * 3 / 4, sz.height / 4), sz.height / 8, cv::Scalar(20, 230, 20), cv::FILLED);
        const std::string tmp_in = ".cs_test_geom_in.png", tmp_out = ".cs_test_geom_out.png";
        cv::imwrite(tmp_in, img);
        const std::string tmp_enc = ".cs_test_geom_in.enc.png";
        const int rc = prod_encrypt(tmp_in, tmp_enc, password, 1.0f);
        char name[96];
        std::snprintf(name, sizeof(name), "geometry %dx%d: encrypt", sz.width, sz.height);
        check(rc == 0, name);
        const int rd = rc == 0 ? prod_decrypt(read_container(tmp_enc), tmp_out, password) : -1;
        std::snprintf(name, sizeof(name), "geometry %dx%d: decrypt", sz.width, sz.height);
        check(rd == 0, name);
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(name, sizeof(name), "geometry %dx%d: output has the input size", sz.width, sz.height);
        check(dec.size() == img.size(), name);
        if (dec.size() == img.size()) {
            // ratio 1.0: the decode is the input up to the half-res solve and the 2x cs-sr stage
            const double p = cs_quality::psnr(img, dec);
            // the decode must not be closer to the input shifted by 1..4 px than to the input itself
            double best_shift = 1e9, aligned = 1e9;
            for (int dy = -4; dy <= 4; ++dy)
                for (int dx = -4; dx <= 4; ++dx) {
                    const cv::Rect a(std::max(0, dx), std::max(0, dy), sz.width - std::abs(dx), sz.height - std::abs(dy));
                    const cv::Rect b(std::max(0, -dx), std::max(0, -dy), a.width, a.height);
                    const double e = cv::norm(img(a), dec(b), cv::NORM_L1) / ((double)a.area() * 3);
                    if (dx == 0 && dy == 0) aligned = e;
                    else best_shift = std::min(best_shift, e);
                }
            std::printf("       geometry %dx%d: PSNR %.2f dB, error aligned %.3f vs best shifted %.3f\n",
                sz.width, sz.height, p, aligned, best_shift);
            std::snprintf(name, sizeof(name), "geometry %dx%d: decode aligned with the input (PSNR > %.0f)", sz.width, sz.height, GEOMETRY_MIN_PSNR);
            check(p > GEOMETRY_MIN_PSNR, name);
            std::snprintf(name, sizeof(name), "geometry %dx%d: no shifted copy matches better", sz.width, sz.height);
            check(aligned <= best_shift, name);
        }
        std::remove(tmp_in.c_str());
        std::remove(tmp_enc.c_str());
        std::remove(tmp_out.c_str());
    }
}
