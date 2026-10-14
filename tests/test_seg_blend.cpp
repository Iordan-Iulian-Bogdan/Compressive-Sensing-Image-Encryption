#include "CS_encryption.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>
#include <set>

class Sampling : public CSencryption {
public:
    using CSencryption::returnSegMaskIndices;
};

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        cv::Mat labels(256, 512, CV_8U, cv::Scalar(0));
        labels.colRange(256, 512).setTo(1);
        const std::vector<uint8_t> rates{51, 204};
        const cv::Mat p = cs_segmask_density(labels, rates, 128);
        require(p.at<uint16_t>(0, 0) == 51 * 257, "low interior altered");
        require(p.at<uint16_t>(0, 511) == 204 * 257, "high interior altered");
        for (int x = 1; x < 512; ++x) {
            require(p.at<uint16_t>(0, x) >= p.at<uint16_t>(0, x - 1), "non-monotonic ramp");
            require(p.at<uint16_t>(0, x) - p.at<uint16_t>(0, x - 1) <= 306, "density cliff remains");
        }
        require(std::abs((int)p.at<uint16_t>(0, 255) + p.at<uint16_t>(0, 256) - 65535) <= 1,
            "frontier rates do not meet at midpoint");
        cv::Mat constant(64, 64, CV_8U, cv::Scalar(1));
        const cv::Mat flat = cs_segmask_density(constant, rates, 128);
        require(cv::countNonZero(flat != 204 * 257) == 0, "constant segment changed");
        Sampling s;
        uint8_t key[32] = {7};
        std::vector<int> x, y, xx, yy;
        s.returnSegMaskIndices(x, y, 256, 512, labels, rates, key, 128, true);
        s.returnSegMaskIndices(xx, yy, 256, 512, labels, rates, key, 128, true);
        require(x == xx && y == yy, "draw regeneration differs");
        std::set<int> unique;
        for (size_t i = 0; i < x.size(); ++i) unique.insert(x[i] * 512 + y[i]);
        require(unique.size() == x.size(), "duplicate samples");
        s.returnSegMaskIndices(x, y, 256, 512, labels, rates, key, 0, true);
        s.returnSegMaskIndices(xx, yy, 256, 512, labels, rates, key);
        require(x == xx && y == yy, "width zero changed legacy draw");
        for (const std::vector<uint8_t> prefix : {
            std::vector<uint8_t>{2, 51, 204, 1, 0, 0, 0, 42},
            std::vector<uint8_t>{2, 51, 204, 1, 0, 0, 128, 42},
            std::vector<uint8_t>{0, 1, 128, 2, 51, 204, 1, 0, 0, 0, 42}}) {
            CsSegPrefix parsed;
            require(cs_segmask_parse_prefix(prefix.data(), prefix.size(), parsed), "prefix rejected");
            require(parsed.total == prefix.size() && parsed.seglen == 1, "wrong prefix offsets");
            require(parsed.smooth_blend == (prefix[0] == 0), "wrong algorithm version");
        }
        std::cout << "PASS: compact support, monotonic profile, midpoint continuity, deterministic unique draw, legacy mode, all prefix versions\n";
        std::cout << "Density profile (x offset from frontier):\n";
        for (int offset : {-129, -96, -64, -32, -1, 0, 32, 64, 96, 128})
            std::cout << offset << ": " << p.at<uint16_t>(0, 256 + offset) / 65535.0 << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
