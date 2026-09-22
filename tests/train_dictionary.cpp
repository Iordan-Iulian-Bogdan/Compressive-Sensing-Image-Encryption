// K-SVD patch dictionary trainer.
//
//   train_dictionary <output.dict> [atoms=256] [ksvd_iters=15] <image1.png> [image2.png ...]
//
// Extracts 8x8 patches (stride 4, all channels) from the given images and
// learns an overcomplete dictionary with OMP sparse coding + SVD atom
// updates. The result is saved in CSD1 format for the solver's --dict mode.
#include "helper_functions.hpp"

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cout << "Usage: train_dictionary <output.dict> <image1.png> [image2.png ...] [atoms=256] [iters=15] [max_patches=60000]\n";
        return 64;
    }

    const std::string out_path = argv[1];
    int atoms = 256, iters = 15, max_patches = 60000;
    std::vector<cv::Mat> images;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        try {
            if (a.rfind("atoms=", 0) == 0) { atoms = std::stoi(a.substr(6)); continue; }
            if (a.rfind("iters=", 0) == 0) { iters = std::stoi(a.substr(6)); continue; }
            if (a.rfind("max_patches=", 0) == 0) { max_patches = std::stoi(a.substr(12)); continue; }
        }
        catch (...) {
            std::cerr << "Bad option: " << a << std::endl;
            return 64;
        }
        cv::Mat img = cv::imread(a, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to load: " << a << std::endl;
            return 64;
        }
        // cap very large images so the trainer stays bounded
        if (img.cols > 2400) {
            cv::resize(img, img, cv::Size(2400, img.rows * 2400 / img.cols));
        }
        std::cout << "using " << a << " (" << img.cols << "x" << img.rows << ")" << std::endl;
        images.push_back(img);
    }

    std::cout << "training: atoms=" << atoms << " iters=" << iters
        << " max_patches=" << max_patches << std::endl;

    cs_dictionary dict;
    if (!cs_train_dictionary(images, atoms, iters, max_patches, dict)) {
        std::cerr << "Training failed (not enough patch data?)" << std::endl;
        return 1;
    }

    if (!cs_save_dictionary(out_path, dict)) {
        std::cerr << "Failed to write: " << out_path << std::endl;
        return 1;
    }
    std::cout << "saved " << dict.atoms << " atoms (patch " << dict.patch << ") to "
        << out_path << " (" << (dict.D.size() * 4 / 1024) << " KB)" << std::endl;
    return 0;
}
