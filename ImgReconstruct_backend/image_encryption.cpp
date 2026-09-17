#include "image_encryption.hpp"
#include <cstring>
#include <cmath>

namespace {
// smallest perfect square >= x (the encrypted container is square)
int next_perfect_square(int x) {
    int side = (int)std::ceil(std::sqrt((double)x));
    while ((long long)side * side < x) side++;
    return side * side;
}
}

encrypt_image::encrypt_image(std::string input_path) {
    input_img = cv::imread(input_path, cv::IMREAD_COLOR);
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
}

encrypt_image::encrypt_image(const cv::Mat& input, bool global_image) {
    input.copyTo(input_img);
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
    if (global_image) {
        cv::resize(input_img, input_img, cv::Size(nextClosestDivisible(input_img.cols, 8), nextClosestDivisible(input_img.rows, 8)));
    }
}

void encrypt_image::get_mat(cv::Mat& dest) {
    encrypted_img.copyTo(dest);
}

cv::Mat encrypt_image::get_mat() {
    return encrypted_img;// .clone();
}

/** @brief top-level encryption: v2 authenticated header.
 * The passphrase is stretched with PBKDF2-HMAC-SHA256 using a random 16-byte
 * salt stored in the header; the resulting 256-bit key seeds the index RNG
 * and encrypts/authenticates the metadata header (AES-256-CTR + HMAC-SHA256
 * over header and measurement body). Key material is wiped after sealing.
 */
void encrypt_image::encrypt(const float& pixel_p, const std::string& password) {
    bm = pixel_p;
    m = rows * cols * bm;
    int n = rows * cols;
    ri_x.resize(m);
    ri_y.resize(m);

    uint8_t salt[CS_SALT_BYTES];
    if (!cs_random_bytes(salt, CS_SALT_BYTES)) {
        throw std::runtime_error("cryptographic RNG unavailable");
    }
    if (!cs_derive_key(password, salt, cs_key)) {
        throw std::runtime_error("key derivation (PBKDF2) failed");
    }
    cs_key_valid = true;

    returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3);

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    int i = CS_HEADER_PIXELS;
    int k = 0;

    for (; (i < total) && k < m - 1; i++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
        k++;
    }

    // seal after the body exists: the tag covers header + all measurements
    if (!cs_seal_header(buf, total * 3, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    // key material is no longer needed once the container is sealed
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)total));
}

/** @brief tile re-encryption used in decrypt_image_tiled: v1 plaintext header.
 * These containers are ephemeral (in-memory only) and carry no password
 * context; measurements start at CS_HEADER_PIXELS, same as v2 containers.
 */
void encrypt_image::encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g) {
    m = ri_x_g.size();
    int n = rows * cols;
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3);

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    cs_write_header_plain(encrypted_img.data, total * 3, text);

    int i = CS_HEADER_PIXELS;
    int k = 0;

    for (; (i < total) && k < m - 1; i++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = input_img.at<cv::Vec3b>(ri_x_g[k], ri_y_g[k]);
        k++;
    }

    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)total));
}

cv::Size encrypt_image::get_size() {
    cv::Size tile_size;
    tile_size.width = cols;
    tile_size.height = rows;

    return tile_size;
}

void encrypt_image::writeEncryptedImageToDisk(const std::string& output_path) {
    cv::imwrite(output_path, encrypted_img);
}

int encrypt_image::encrypt_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& password,
    float compression_ratio,
    cv::Mat* encrypted_out
){
    try {

        if (CSencryption::params == AUTO_PARAM)
        {
            compression_ratio = 1.0f;
        }

        if (compression_ratio < 0.25f || compression_ratio > 1.0f) {
            throw std::runtime_error("Compression ratio is outside the acceptable range of (0.25, 1.0)");
        }

        if (password.size() < 10) {
            throw std::runtime_error("Password should be at least 10 characters");
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    try {
        cv::Mat input_img;
        input_img = cv::imread(input_path);

        if (input_img.empty()) {
            throw std::runtime_error("Failed to load input image: " + input_path);
        }

        cv::resize(input_img, input_img, cv::Size(input_img.cols / 2, input_img.rows / 2));
        encrypt_image encrypt_img(input_img, true);
        encrypt_img.encrypt(compression_ratio, password);

        if (encrypted_out) {
            encrypt_img.get_mat(*encrypted_out);
        }

        if (!output_path.empty()) {
            cv::imwrite(output_path, encrypt_img.get_mat());
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
