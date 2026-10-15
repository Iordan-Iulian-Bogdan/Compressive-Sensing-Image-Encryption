// Dump the deterministic --seg-auto label map for an input image so it can
// feed --seg-solve. Replicates the encrypt pipeline geometry (half-res
// downscale unless --full-res) then cs_seg_auto_ids.
#include <cstdio>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "image_encryption.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: segmask_dump <input.png> <k> <out.png> [--full-res]\n");
        return 2;
    }
    const bool full_res = (argc > 4 && std::string(argv[4]) == "--full-res");
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { std::printf("bad input\n"); return 1; }
    if (!full_res) cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    const int k = std::stoi(argv[2]);
    cv::Mat ids;
    cs_seg_auto_ids(img, k, img.rows, img.cols, ids);
    std::vector<long long> sz((size_t)k, 0);
    for (int y = 0; y < ids.rows; ++y) {
        const uint8_t* r = ids.ptr<uint8_t>(y);
        for (int x = 0; x < ids.cols; ++x) if (r[x] < k) sz[(size_t)r[x]]++;
    }
    for (int c = 0; c < k; ++c) std::printf("class %d px=%lld\n", c, sz[(size_t)c]);
    cv::imwrite(argv[3], ids);
    std::printf("wrote %s (%dx%d)\n", argv[3], ids.cols, ids.rows);
    return 0;
}
