// Usage: bench_seg_boundary reference.png palette_mask.png output_prefix image.png ...
// Measures reconstruction error on the low-sampled side of the bird boundary.
// Crops are saved beside output_prefix for direct inspection at native resolution.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 5) throw std::runtime_error("expected reference mask output_prefix reconstruction...");
        cv::Mat ref = cv::imread(argv[1]), labels = cv::imread(argv[2]);
        if (ref.empty() || labels.empty() || ref.size() != labels.size())
            throw std::runtime_error("invalid reference/mask geometry");
        cv::Mat bird;
        cv::inRange(labels, cv::Scalar(70, 200, 70), cv::Scalar(70, 200, 70), bird);
        if (!cv::countNonZero(bird)) throw std::runtime_error("mask needs the scene bird palette color");
        cv::Mat outer, band, blue;
        cv::dilate(bird, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, {49, 49}));
        cv::inRange(labels, cv::Scalar(230, 120, 40), cv::Scalar(230, 120, 40), blue);
        cv::bitwise_and(outer, ~bird, band);
        cv::bitwise_and(band, blue, band);
        cv::Mat rf, rg, rl;
        ref.convertTo(rf, CV_32FC3);
        cv::cvtColor(rf, rg, cv::COLOR_BGR2GRAY);
        cv::Laplacian(rg, rl, CV_32F);
        const cv::Rect roi = cv::boundingRect(bird);
        const cv::Rect crop((std::max)(0, roi.x - 80), (std::max)(0, roi.y - 80),
            (std::min)(1200, ref.cols - (std::max)(0, roi.x - 80)),
            (std::min)(900, ref.rows - (std::max)(0, roi.y - 80)));
        cv::imwrite(std::string(argv[3]) + ".reference.crop.png", ref(crop));
        for (int a = 4; a < argc; ++a) {
            cv::Mat img = cv::imread(argv[a]);
            if (img.empty() || img.size() != ref.size()) throw std::runtime_error("reconstruction size mismatch");
            cv::Mat f, g, lap, diff;
            img.convertTo(f, CV_32FC3);
            cv::absdiff(f, rf, diff);
            const cv::Scalar mae = cv::mean(diff, band);
            cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            cv::Laplacian(g, lap, CV_32F);
            cv::absdiff(lap, rl, diff);
            const double hf = cv::mean(diff, band)[0];
            // Compare gradients across the actual label frontier to the
            // reference, rather than penalizing the real bird/background edge.
            double seam = 0; size_t edges = 0;
            for (int y = 0; y < ref.rows - 1; ++y)
                for (int x = 0; x < ref.cols - 1; ++x)
                    for (const cv::Point d : {cv::Point(1, 0), cv::Point(0, 1)}) {
                        const cv::Point p(x, y), q = p + d;
                        if (bird.at<uchar>(p) == bird.at<uchar>(q)) continue;
                        if (!(blue.at<uchar>(p) || blue.at<uchar>(q))) continue;
                        const auto delta = (f.at<cv::Vec3f>(p) - f.at<cv::Vec3f>(q))
                            - (rf.at<cv::Vec3f>(p) - rf.at<cv::Vec3f>(q));
                        seam += (std::abs(delta[0]) + std::abs(delta[1]) + std::abs(delta[2])) / 3;
                        ++edges;
                    }
            std::cout << argv[a] << " low-side band pixels=" << cv::countNonZero(band)
                << " MAE=" << (mae[0] + mae[1] + mae[2]) / 3
                << " HF-error=" << hf << " frontier-gradient-error="
                << (edges ? seam / edges : 0) << '\n';
            cv::imwrite(std::string(argv[3]) + "." + std::to_string(a - 4) + ".crop.png", img(crop));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
