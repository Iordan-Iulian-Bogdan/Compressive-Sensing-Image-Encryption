#include "image_encryption.hpp"
#include <opencv2/highgui.hpp>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {
// smallest perfect square >= x (the encrypted container is square)
int next_perfect_square(int x) {
    int side = (int)std::ceil(std::sqrt((double)x));
    while ((long long)side * side < x) side++;
    return side * side;
}

// Per-tile level-of-detail bytes in [lod_min, 255]: mean |Laplacian| per
// tile, min-max normalized across the grid, floored so flat regions keep a
// minimum budget. Index = tr * tiles_cols + tc (row-major), matching
// cs_compute_tile_sample_counts / returnAdaptiveIndices.
std::vector<uint8_t> compute_adaptive_lod(const cv::Mat& img, int tile_size, int lod_min) {
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }
    lod_min = (std::max)(0, (std::min)(255, lod_min));

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img;
    }
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Mat abs_lap;
    cv::absdiff(lap, cv::Scalar(0), abs_lap);

    const int rows = img.rows;
    const int cols = img.cols;
    const int tiles_cols = (cols + tile_size - 1) / tile_size;
    const int tiles_rows = (rows + tile_size - 1) / tile_size;
    std::vector<uint8_t> lod((size_t)tiles_rows * tiles_cols, (uint8_t)lod_min);

    std::vector<float> energy(lod.size(), 0.0f);
    float e_min = 0.0f, e_max = 0.0f;
    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            const int row0 = tr * tile_size;
            const int col0 = tc * tile_size;
            const int tile_w = (tile_size < cols - col0) ? tile_size : cols - col0;
            const int tile_h = (tile_size < rows - row0) ? tile_size : rows - row0;
            const cv::Rect r(col0, row0, tile_w, tile_h);
            const float e = (float)cv::mean(abs_lap(r))[0];
            const size_t i = (size_t)tr * tiles_cols + tc;
            energy[i] = e;
            if (tr == 0 && tc == 0) {
                e_min = e_max = e;
            }
            else {
                e_min = (std::min)(e_min, e);
                e_max = (std::max)(e_max, e);
            }
        }
    }

    const float range = e_max - e_min;
    for (size_t i = 0; i < energy.size(); ++i) {
        float n = 0.0f;
        if (range > 1e-6f) {
            n = (energy[i] - e_min) / range;
        }
        int v = lod_min + (int)std::lround((double)n * (255 - lod_min));
        lod[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return lod;
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

bool encrypt_image::build_sampling_mask_view(cv::Mat& out) const {
    if (input_img.empty() || ri_x.empty() || ri_x.size() != ri_y.size()) {
        return false;
    }
    const int r = input_img.rows, c = input_img.cols;
    cv::Mat mask(r, c, CV_8UC1, cv::Scalar(0));
    for (size_t k = 0; k < ri_x.size(); ++k) {
        if (ri_x[k] >= 0 && ri_x[k] < r && ri_y[k] >= 0 && ri_y[k] < c) {
            mask.at<uchar>(ri_x[k], ri_y[k]) = 255;
        }
    }

    cv::Mat gray;
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input_img;
    }
    cv::Mat gray3;
    cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);

    cv::Mat mask3;
    cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);

    cv::Mat overlay = gray3.clone();
    overlay.setTo(cv::Scalar(60, 60, 60));
    cv::Mat tint(r, c, CV_8UC3, cv::Scalar(0, 0, 0));
    tint.setTo(cv::Scalar(255, 180, 0), mask);
    cv::addWeighted(overlay, 0.55, gray3, 0.45, 0, overlay);
    (cv::max)(overlay, tint, overlay);

    auto label = [](cv::Mat& panel, const std::string& t) {
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    char info[96];
    std::snprintf(info, sizeof(info), "mask m=%zu ratio~%.2f", ri_x.size(),
        (double)ri_x.size() / (double)((long long)r * c));
    label(gray3, "source");
    label(mask3, info);
    label(overlay, "cyan=sampled");

    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{gray3, mask3, overlay}, row);
    out = row;
    return true;
}

void encrypt_image::present_sampling_mask(const std::string& output_path) const {
    cv::Mat view;
    if (!build_sampling_mask_view(view)) {
        std::cerr << "Warning: --show-mask: no sampling indices available\n";
        return;
    }

    std::string mask_path;
    if (!output_path.empty()) {
        mask_path = output_path;
        const size_t dot = mask_path.find_last_of('.');
        if (dot != std::string::npos && dot > mask_path.find_last_of("/\\")) {
            mask_path.insert(dot, ".mask");
        } else {
            mask_path += ".mask";
        }
        if (mask_path.size() < 4 || mask_path.substr(mask_path.size() - 4) != ".png") {
            mask_path += ".png";
        }
        if (!cv::imwrite(mask_path, view)) {
            std::cerr << "Warning: --show-mask: failed to write " << mask_path << "\n";
            mask_path.clear();
        } else {
            std::cout << "Sampling mask written to " << mask_path << std::endl;
        }
    }

    try {
        const char* win = "sampling mask";
        cv::namedWindow(win, cv::WINDOW_NORMAL);
        // fit roughly on screen without depending on desktop metrics
        const int max_w = 1600, max_h = 900;
        double s = 1.0;
        if (view.cols > max_w) s = (double)max_w / view.cols;
        if (view.rows * s > max_h) s = (double)max_h / view.rows;
        cv::Mat show;
        if (s < 1.0) {
            cv::resize(view, show, cv::Size(), s, s, cv::INTER_AREA);
        } else {
            show = view;
        }
        cv::imshow(win, show);
        std::cout << "Press any key in the mask window to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyWindow(win);
    } catch (const cv::Exception&) {
        // headless / no GUI backend: file output above is enough
        if (mask_path.empty()) {
            std::cerr << "Warning: --show-mask: GUI unavailable and no mask path set\n";
        }
    }
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
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    for (int k = 0; k < m; k++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
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

/** @brief Periodic tile-based encryption: generates random indices within one
 * canonical tile, then tiles that pattern across the entire image. This is
 * useful for creating a structured sampling pattern that can be more efficient
 * for certain reconstruction algorithms.
 * @param pixel_p : sampling ratio (0.001 - 1.0)
 * @param password : encryption password
 * @param tile_size : size of the canonical tile (must divide image dimensions)
 */
void encrypt_image::encrypt_periodic(const float& pixel_p, const std::string& password, int tile_size) {
    bm = pixel_p;

    uint8_t salt[CS_SALT_BYTES];
    if (!cs_random_bytes(salt, CS_SALT_BYTES)) {
        throw std::runtime_error("cryptographic RNG unavailable");
    }
    if (!cs_derive_key(password, salt, cs_key)) {
        throw std::runtime_error("key derivation (PBKDF2) failed");
    }
    cs_key_valid = true;

    // samples per canonical tile from the requested ratio; the total count m
    // follows from the generated pattern (edge tiles hold fewer samples).
    // At extreme low ratios (e.g. 0.001 with a small tile) the product can
    // round to 0 — keep one sample per tile so the pattern stays valid.
    int samples_per_tile = (int)std::lround((double)tile_size * tile_size * pixel_p);
    if (samples_per_tile < 1) samples_per_tile = 1;
    returnPeriodicIndices(ri_x, ri_y, rows, cols, samples_per_tile, cs_key, tile_size);
    m = (int)ri_x.size();

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    // record the sampling mode in the authenticated pad region:
    // [pad+0] = 1 (periodic), [pad+1..3) tile_size LE, [pad+3..5) samples_per_tile LE
    buf[CS_OFF_PAD] = 1;
    buf[CS_OFF_PAD + 1] = (uint8_t)(tile_size & 0xFF);
    buf[CS_OFF_PAD + 2] = (uint8_t)((tile_size >> 8) & 0xFF);
    buf[CS_OFF_PAD + 3] = (uint8_t)(samples_per_tile & 0xFF);
    buf[CS_OFF_PAD + 4] = (uint8_t)((samples_per_tile >> 8) & 0xFF);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    for (int k = 0; k < m; k++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
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

/** @brief Adaptive periodic encryption: score every tile with an 8-bit LOD
 * byte (mean |Laplacian|, min-max + floor), split the measurement budget m
 * across tiles proportional to those bytes, draw a distinct in-tile pattern
 * per tile, and ship the lod byte array in the body (mode=2). Decrypt
 * regenerates identical indices from (key, lod, m, geometry).
 * @param pixel_p : sampling ratio (0.001 - 1.0)
 * @param password : encryption password
 * @param tile_size : adaptive grid tile size (must be > 0)
 * @param lod_min : floor byte so flat tiles keep a minimum sample share
 */
void encrypt_image::encrypt_adaptive(const float& pixel_p, const std::string& password, int tile_size,
    int lod_min, int weight_base) {
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }
    if (weight_base < 1) weight_base = 1;
    if (weight_base > 65535) weight_base = 65535;
    bm = pixel_p;
    m = rows * cols * bm;

    uint8_t salt[CS_SALT_BYTES];
    if (!cs_random_bytes(salt, CS_SALT_BYTES)) {
        throw std::runtime_error("cryptographic RNG unavailable");
    }
    if (!cs_derive_key(password, salt, cs_key)) {
        throw std::runtime_error("key derivation (PBKDF2) failed");
    }
    cs_key_valid = true;

    const std::vector<uint8_t> lod = compute_adaptive_lod(input_img, tile_size, lod_min);
    const int lod_bytes = (int)lod.size();
    const int lod_pixels = (lod_bytes + 2) / 3;

    // m and weight_base are fixed BEFORE index generation so decrypt
    // recomputes the same per-tile counts from the header — not from ri_x.size()
    returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size, weight_base);
    const int m_written = (int)ri_x.size();

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + lod_pixels + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    // metadata carries the budget m (count-input), not m_written: capacity
    // clamping can drop a few samples but both sides recompute counts from m
    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    // mode=2, tile_size LE, lod_bytes LE (sanity), weight_base LE in the
    // authenticated pad — decrypt restores the exact aggression setting
    buf[CS_OFF_PAD] = CS_MODE_ADAPTIVE;
    buf[CS_OFF_PAD + 1] = (uint8_t)(tile_size & 0xFF);
    buf[CS_OFF_PAD + 2] = (uint8_t)((tile_size >> 8) & 0xFF);
    buf[CS_OFF_PAD + 3] = (uint8_t)(lod_bytes & 0xFF);
    buf[CS_OFF_PAD + 4] = (uint8_t)((lod_bytes >> 8) & 0xFF);
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] = (uint8_t)(weight_base & 0xFF);
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] = (uint8_t)((weight_base >> 8) & 0xFF);

    // lod region immediately after the header (raw bytes, zero-padded to a
    // whole pixel so measurements stay pixel-aligned at CS_HEADER_PIXELS+lod_pixels)
    std::memcpy(buf + CS_HEADER_BYTES, lod.data(), (size_t)lod_bytes);

    // write measurements at regenerated indices
    const int meas0 = CS_HEADER_PIXELS + lod_pixels;
    for (int k = 0; k < m_written; k++) {
        encrypted_img.at<cv::Vec3b>(k + meas0) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
    }

    if (!cs_seal_header(buf, total * 3, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

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
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    cs_write_header_plain(encrypted_img.data, total * 3, text);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    for (int k = 0; k < m; k++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = input_img.at<cv::Vec3b>(ri_x_g[k], ri_y_g[k]);
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
    int tile_size,
    cv::Mat* encrypted_out,
    bool adaptive,
    int lod_min,
    float adaptive_strength,
    bool show_mask,
    bool full_res
){
    try {
        // the caller-requested ratio is honored in every mode: decryption
        // derives its own coef/iterations from the container's actual ratio,
        // so auto mode has no reason to force full sampling

        if (compression_ratio < 0.001f || compression_ratio > 1.0f) {
            throw std::runtime_error("Compression ratio is outside the acceptable range of (0.001, 1.0]");
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

        // Default pipeline runs at half resolution (encrypt downsamples, decrypt
        // 2x-upscales the solve). --full-res skips both so geometry stays native.
        if (!full_res) {
            cv::resize(input_img, input_img, cv::Size(input_img.cols / 2, input_img.rows / 2));
        }
        encrypt_image encrypt_img(input_img, true);

        if (adaptive) {
            // LOD-based per-tile sampling; default grid when tile_size unset
            const int ts = tile_size > 0 ? tile_size : 64;
            const int wbase = cs_adaptive_base_from_strength(adaptive_strength);
            encrypt_img.encrypt_adaptive(compression_ratio, password, ts, lod_min, wbase);
        }
        else if (tile_size > 0) {
            // periodic tile-based sampling: one random per-tile pattern,
            // replicated across the image; the mode travels in the header
            encrypt_img.encrypt_periodic(compression_ratio, password, tile_size);
        } else {
            // standard random global sampling
            encrypt_img.encrypt(compression_ratio, password);
        }

        if (encrypted_out) {
            encrypt_img.get_mat(*encrypted_out);
        }

        if (!output_path.empty()) {
            cv::imwrite(output_path, encrypt_img.get_mat());
        }

        // compression stage just finished: render the mask that was used
        if (show_mask) {
            encrypt_img.present_sampling_mask(output_path);
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
