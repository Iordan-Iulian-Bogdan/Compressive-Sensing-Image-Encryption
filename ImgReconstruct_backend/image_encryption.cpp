#include "image_encryption.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iostream>

// Global radial spectral decay of a gray image: thumbnail (max 128px),
// orthonormal DCT, mean |F|^2 per radial bin, log-log least-squares slope.
// Returns alpha in E(f) ~ f^-alpha (natural images ~2). Any degenerate fit
// (flat image, too few bins, NaN) returns 2.0. Pure function of pixels.
static double cs_fit_spectral_decay(const cv::Mat& gray) {
    const int maxd = (std::max)(gray.rows, gray.cols);
    if (maxd <= 0) return 2.0;
    cv::Mat thumb = gray;
    if (maxd > 128) {
        cv::resize(gray, thumb, cv::Size(), 128.0 / maxd, 128.0 / maxd, cv::INTER_AREA);
    }
    cv::Mat f;
    thumb.convertTo(f, CV_32F, 1.0 / 255.0);
    cv::dct(f, f);
    const int R = f.rows, C = f.cols;
    if (R < 4 || C < 4) return 2.0;
    const int fmax = (int)std::sqrt((double)(R - 1) * (R - 1) + (double)(C - 1) * (C - 1));
    std::vector<double> sum((size_t)fmax + 1, 0.0), cnt((size_t)fmax + 1, 0.0);
    for (int u = 0; u < R; ++u) {
        const float* row = f.ptr<float>(u);
        for (int v = 0; v < C; ++v) {
            if (u == 0 && v == 0) continue; // DC carries brightness, not detail
            int b = (int)std::lround(std::sqrt((double)u * u + (double)v * v));
            if (b > fmax) b = fmax;
            const double e = (double)row[v] * row[v];
            sum[(size_t)b] += e;
            cnt[(size_t)b] += 1.0;
        }
    }
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int n = 0;
    for (int b = 1; b <= fmax; ++b) {
        if (cnt[(size_t)b] < 4.0 || sum[(size_t)b] <= 0.0) continue;
        const double x = std::log((double)b);
        const double y = std::log(sum[(size_t)b] / cnt[(size_t)b]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
        ++n;
    }
    if (n < 3) return 2.0;
    const double denom = n * sxx - sx * sx;
    if (!(denom > 1e-9)) return 2.0;
    const double alpha = -((n * sxy - sx * sy) / denom);
    if (!(alpha >= 0.5)) return 2.0; // NaN or rising spectrum: natural-image prior
    return (std::min)(alpha, 4.0);
}

// Per-tile spectral level-of-detail bytes in [lod_min, 255]: each tile's
// gray DCT energy whitened by the image's global radial decay, i.e. tiles
// pay for high-frequency energy the natural-image prior does NOT predict.
// Smooth gradients (steep local spectra, DCT-sparse) score low even when
// bright; texture/noise (flat local spectra) score high. Same grid,
// min-max + floor normalization, and row-major layout as the Laplacian
// path, so counts/container/decrypt are unchanged. Pure function of pixels.
std::vector<uint8_t> compute_spectral_lod(const cv::Mat& img, int tile_size, int lod_min) {
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

    // Expected-energy weights for a tile_size x tile_size DCT: (f+1)^alpha
    // so each coefficient contributes surprise, not raw energy. DC weight 0
    // (brightness must not buy budget).
    const double alpha = cs_fit_spectral_decay(gray);
    const int T = tile_size;
    std::vector<float> w((size_t)T * T, 0.0f);
    for (int u = 0; u < T; ++u) {
        for (int v = 0; v < T; ++v) {
            if (u == 0 && v == 0) continue;
            const double f = std::sqrt((double)u * u + (double)v * v);
            w[(size_t)u * T + v] = (float)std::pow(f + 1.0, alpha);
        }
    }

    const int rows = img.rows;
    const int cols = img.cols;
    const int tiles_cols = (cols + tile_size - 1) / tile_size;
    const int tiles_rows = (rows + tile_size - 1) / tile_size;
    std::vector<uint8_t> lod((size_t)tiles_rows * tiles_cols, (uint8_t)lod_min);

    std::vector<double> energy(lod.size(), 0.0);
    double e_min = 0.0, e_max = 0.0;
    bool first = true;
    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            const int row0 = tr * tile_size;
            const int col0 = tc * tile_size;
            const int tile_w = (tile_size < cols - col0) ? tile_size : cols - col0;
            const int tile_h = (tile_size < rows - row0) ? tile_size : rows - row0;
            // Replicate-pad edge tiles to the common DCT grid so frequency
            // bins (and the whitening) are comparable across all tiles.
            cv::Mat buf(T, T, CV_32F);
            for (int i = 0; i < T; ++i) {
                const int sy = row0 + ((i < tile_h) ? i : tile_h - 1);
                const uint8_t* srow = gray.ptr<uint8_t>(sy);
                float* drow = buf.ptr<float>(i);
                for (int j = 0; j < T; ++j) {
                    const int sx = col0 + ((j < tile_w) ? j : tile_w - 1);
                    drow[j] = srow[sx] * (1.0f / 255.0f);
                }
            }
            cv::dct(buf, buf);
            if (!buf.isContinuous()) buf = buf.clone();
            const float* d = buf.ptr<float>(0);
            double e = 0.0;
            for (int i = 0; i < T * T; ++i) {
                const double c = d[i];
                e += (double)w[i] * c * c;
            }
            e /= (double)((long long)T * T);
            const size_t k = (size_t)tr * tiles_cols + tc;
            energy[k] = e;
            if (first) { e_min = e_max = e; first = false; }
            else {
                e_min = (std::min)(e_min, e);
                e_max = (std::max)(e_max, e);
            }
        }
    }

    const double range = e_max - e_min;
    for (size_t i = 0; i < energy.size(); ++i) {
        double n = 0.0;
        if (range > 1e-9) {
            n = (energy[i] - e_min) / range;
        }
        int v = lod_min + (int)std::lround(n * (255 - lod_min));
        lod[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return lod;
}
namespace {
// LLM region spec: box in 0-1000 normalized coords (origin top-left),
// detail in [0,1]. Passed inline via --regions as
// {"regions": [{"box": [x0,y0,x1,y1], "detail": 1.0, "label": "..."}]}.
struct LlmRegion {
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    double detail = 1.0;
};

void regions_skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && std::isspace((unsigned char)s[p])) ++p;
}

void regions_expect(const std::string& s, size_t& p, char c, const char* ctx) {
    regions_skip_ws(s, p);
    if (p >= s.size() || s[p] != c) {
        throw std::runtime_error(std::string("regions: expected '") + c + "' " + ctx);
    }
    ++p;
}

double regions_number(const std::string& s, size_t& p, const char* ctx) {
    regions_skip_ws(s, p);
    const char* b = s.c_str() + p;
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b) {
        throw std::runtime_error(std::string("regions: bad number ") + ctx);
    }
    p = (size_t)(e - s.c_str());
    return v;
}

std::string regions_string(const std::string& s, size_t& p) {
    regions_skip_ws(s, p);
    // double or single quotes (single-quoted JSON survives Windows
    // PowerShell 5.1 native-arg passing, which strips embedded "...")
    if (p >= s.size() || (s[p] != '"' && s[p] != '\'')) {
        throw std::runtime_error("regions: expected string key");
    }
    const char q = s[p];
    ++p;
    std::string out;
    while (p < s.size() && s[p] != q) {
        if (s[p] == '\\' && p + 1 < s.size()) {
            ++p;
            out.push_back(s[p++]);
        } else {
            out.push_back(s[p++]);
        }
    }
    if (p >= s.size()) {
        throw std::runtime_error("regions: unterminated string");
    }
    ++p;
    return out;
}

// skip one JSON value (string/number/array/object/literal) for unknown keys
void regions_skip_value(const std::string& s, size_t& p) {
    regions_skip_ws(s, p);
    if (p >= s.size()) {
        throw std::runtime_error("regions: unexpected end");
    }
    if (s[p] == '"') {
        regions_string(s, p);
    } else if (s[p] == '[') {
        ++p;
        regions_skip_ws(s, p);
        if (p < s.size() && s[p] == ']') { ++p; return; }
        while (true) {
            regions_skip_value(s, p);
            regions_skip_ws(s, p);
            if (p < s.size() && s[p] == ',') { ++p; continue; }
            break;
        }
        regions_expect(s, p, ']', "closing array");
    } else if (s[p] == '{') {
        ++p;
        regions_skip_ws(s, p);
        if (p < s.size() && s[p] == '}') { ++p; return; }
        while (true) {
            regions_string(s, p);
            regions_expect(s, p, ':', "after key");
            regions_skip_value(s, p);
            regions_skip_ws(s, p);
            if (p < s.size() && s[p] == ',') { ++p; continue; }
            break;
        }
        regions_expect(s, p, '}', "closing object");
    } else {
        while (p < s.size() && s[p] != ',' && s[p] != ']' && s[p] != '}') ++p;
    }
}

std::vector<LlmRegion> parse_regions_json(const std::string& s) {
    size_t p = 0;
    regions_expect(s, p, '{', "at top level");
    bool have_regions = false;
    std::vector<LlmRegion> out;
    regions_skip_ws(s, p);
    if (p < s.size() && s[p] == '}') return out; // {} -> no regions key
    while (true) {
        const std::string key = regions_string(s, p);
        regions_expect(s, p, ':', "after key");
        if (key == "regions") {
            have_regions = true;
            regions_expect(s, p, '[', "after \"regions\"");
            regions_skip_ws(s, p);
            if (p < s.size() && s[p] == ']') { ++p; }
            else {
                while (true) {
                    regions_expect(s, p, '{', "opening region");
                    LlmRegion r;
                    bool have_box = false;
                    regions_skip_ws(s, p);
                    if (p >= s.size() || s[p] == '}') {
                        throw std::runtime_error("regions: empty region object");
                    }
                    while (true) {
                        const std::string k = regions_string(s, p);
                        regions_expect(s, p, ':', "after region key");
                        if (k == "box") {
                            regions_expect(s, p, '[', "after \"box\"");
                            r.x0 = regions_number(s, p, "in box");
                            regions_expect(s, p, ',', "in box");
                            r.y0 = regions_number(s, p, "in box");
                            regions_expect(s, p, ',', "in box");
                            r.x1 = regions_number(s, p, "in box");
                            regions_expect(s, p, ',', "in box");
                            r.y1 = regions_number(s, p, "in box");
                            regions_expect(s, p, ']', "closing box");
                            have_box = true;
                        } else if (k == "detail") {
                            r.detail = regions_number(s, p, "for detail");
                        } else {
                            regions_skip_value(s, p); // label and friends
                        }
                        regions_skip_ws(s, p);
                        if (p < s.size() && s[p] == ',') { ++p; continue; }
                        break;
                    }
                    regions_expect(s, p, '}', "closing region");
                    if (!have_box) {
                        throw std::runtime_error("regions: region without \"box\"");
                    }
                    // sanitize: clamp to [0,1000], fix inverted, drop empty
                    r.x0 = (std::max)(0.0, (std::min)(1000.0, r.x0));
                    r.x1 = (std::max)(0.0, (std::min)(1000.0, r.x1));
                    r.y0 = (std::max)(0.0, (std::min)(1000.0, r.y0));
                    r.y1 = (std::max)(0.0, (std::min)(1000.0, r.y1));
                    if (r.x0 > r.x1) std::swap(r.x0, r.x1);
                    if (r.y0 > r.y1) std::swap(r.y0, r.y1);
                    r.detail = (std::max)(0.0, (std::min)(1.0, r.detail));
                    if (r.x1 > r.x0 && r.y1 > r.y0) {
                        out.push_back(r);
                    }
                    regions_skip_ws(s, p);
                    if (p < s.size() && s[p] == ',') { ++p; continue; }
                    break;
                }
                regions_expect(s, p, ']', "closing regions");
            }
        } else {
            regions_skip_value(s, p);
        }
        regions_skip_ws(s, p);
        if (p < s.size() && s[p] == ',') { ++p; continue; }
        break;
    }
    regions_expect(s, p, '}', "at top level");
    regions_skip_ws(s, p);
    if (p != s.size()) {
        throw std::runtime_error("regions: trailing characters after object");
    }
    if (!have_regions) {
        throw std::runtime_error("regions: missing \"regions\" array");
    }
    return out;
}

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


void build_hf_thumb_png(const cv::Mat& input, std::vector<uint8_t>& encoded) {
    if (input.empty()) throw std::runtime_error("hf thumbnail: empty input image");
    const int max_dimension = (std::max)(input.cols, input.rows);
    const double scale = max_dimension > 128 ? 128.0 / max_dimension : 1.0;
    cv::Mat thumbnail;
    cv::resize(input, thumbnail, cv::Size(), scale, scale, cv::INTER_AREA);
    const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 6};
    if (!cv::imencode(".png", thumbnail, encoded, params) || encoded.empty() ||
        encoded.size() > (static_cast<size_t>(1) << 22)) {
        throw std::runtime_error("hf thumbnail PNG encode failed");
    }
}

void seal_hf_thumb(uint8_t* data, size_t length, const uint8_t* iv,
                   const uint8_t key[32]) {
    uint8_t counter[CS_IV_BYTES];
    std::memcpy(counter, iv, CS_IV_BYTES);
    counter[0] ^= 0x54;
    if (!cs_aes256_ctr_xor(data, length, key, counter)) {
        cs_wipe(counter, sizeof(counter));
        throw std::runtime_error("hf thumbnail encryption failed");
    }
    cs_wipe(counter, sizeof(counter));
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
void encrypt_image::encrypt(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits) {
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
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: bit-pack the flat measurement section (3 values per
    // sample at byte offset CS_HEADER_BYTES); skipped at 8 bits.
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        encrypted_img = cs_pack_container_body(encrypted_img,
            { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
        buf = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }

    // seal after the body exists: the tag covers header + all measurements
    if (!cs_seal_header(buf, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    // key material is no longer needed once the container is sealed
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

/** @brief Periodic tile-based encryption: generates random indices within one
 * canonical tile, then tiles that pattern across the entire image. This is
 * useful for creating a structured sampling pattern that can be more efficient
 * for certain reconstruction algorithms.
 * @param pixel_p : sampling ratio (0.001 - 1.0)
 * @param password : encryption password
 * @param tile_size : size of the canonical tile (must divide image dimensions)
 */
void encrypt_image::encrypt_periodic(const float& pixel_p, const std::string& password, int tile_size, int sample_bits, int chroma_bits) {
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
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: bit-pack the flat measurement section (3 values per
    // sample at byte offset CS_HEADER_BYTES); skipped at 8 bits.
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        encrypted_img = cs_pack_container_body(encrypted_img,
            { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
        buf = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }

    // seal after the body exists: the tag covers header + all measurements
    if (!cs_seal_header(buf, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    // key material is no longer needed once the container is sealed
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

/** @brief YCC 4:2:0 split encryption (mode 3): BGR -> YCrCb, chroma planes
 * downsampled 2x (INTER_AREA), then luma sampled at full resolution and each
 * chroma plane sampled on its own coarse grid with a domain-separated key.
 * The body is byte-packed [Y x m][Cr x m_chroma][Cb x m_chroma] so the
 * container holds ~1.5 bytes per luma sample instead of 3 (BGR modes).
 * Metadata carries the luma budget m; chroma counts re-derive from it.
 */
void encrypt_image::encrypt_ycc420(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits, int mode) {
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

    returnYcc420Indices(rows, cols, m, cs_key, mode);
    const int mC = m_chroma;

    cv::Mat ycc;
    cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> planes;
    cv::split(ycc, planes); // 0:Y 1:Cr 2:Cb to match COLOR_YCrCb2BGR order
    int crows, ccols;
    cs_chroma_dims(mode, rows, cols, crows, ccols);
    cv::Mat cr_small, cb_small;
    cv::resize(planes[1], cr_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
    cv::resize(planes[2], cb_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);

    const size_t body_bytes = (size_t)m + (size_t)2 * mC;
    const int body_pixels = (int)((body_bytes + 2) / 3);
    const int total = next_perfect_square(body_pixels + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    // mode (3 or 7) in the authenticated pad region; decrypt restores it
    // from here
    buf[CS_OFF_PAD] = (uint8_t)mode;

    uint8_t* body = buf + CS_HEADER_BYTES;
    for (int k = 0; k < m; k++) {
        body[k] = planes[0].at<uint8_t>(ri_x[k], ri_y[k]);
    }
    for (int k = 0; k < mC; k++) {
        body[m + k] = cr_small.at<uint8_t>(ri_cx1[k], ri_cy1[k]);
        body[m + mC + k] = cb_small.at<uint8_t>(ri_cx2[k], ri_cy2[k]);
    }
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: bit-pack the three planes independently (luma, Cr,
    // Cb); skipped at 8 bits.
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        const size_t y0 = (size_t)CS_HEADER_BYTES;
        encrypted_img = cs_pack_container_body(encrypted_img, {
            { y0, (size_t)m, sample_bits, 0, false },
            { y0 + (size_t)m, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)m + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
        buf = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }

    // seal after the body exists: the tag covers header + all measurements
    if (!cs_seal_header(buf, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
    sampling_mode = mode;
}

// Forward declarations (defined alongside encrypt_adaptive below): LOD
// region blending + density smoothing shared with encrypt_adaptive.
std::vector<uint8_t> blend_llm_lod(const std::vector<uint8_t>& base,
    const std::vector<LlmRegion>& regs, int rows_l, int cols_l, int tile_size,
    int lod_min, float blend);
std::vector<uint8_t> smooth_lod_grid(const std::vector<uint8_t>& lod,
    int rows_l, int cols_l, int tile_size, int lod_min, float sigma);

/** @brief YCC 4:2:0 encryption with LOD-driven luma (mode 3 + lod region):
 * the luma budget splits across tiles by the same 8-bit LOD bytes as mode 2
 * (Laplacian, two-pass residual, or LLM-region blend + smoothing — identical
 * helpers, identical shipped layout), while each chroma plane keeps its
 * uniform coarse-grid draw. Body: [lod bytes][Y x mY_written][Cr][Cb].
 * Metadata carries the luma budget m; decrypt regenerates both the per-tile
 * luma counts and the chroma draws from (key, lod, m, geometry).
 */
void encrypt_image::encrypt_ycc420_adaptive(const float& pixel_p, const std::string& password,
    int tile_size, int lod_min, int weight_base, bool two_pass, float pilot_ratio,
    const std::string& regions_json, float region_blend, float lod_smooth, int sample_bits, int chroma_bits, int lod_full, int mode, bool spectral_lod) {
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }
    // 16-bit lod-count header field: grids over 65535 tiles truncate it and
    // decrypt silently regenerates wrong positions. Fail fast (see
    // encrypt_adaptive for the full rationale).
    {
        const int ntiles = cs_lod_tile_count(input_img.rows, input_img.cols, tile_size);
        if (ntiles > 65535) {
            const int min_ts = (std::max)(1, (int)std::ceil(std::sqrt(
                (double)input_img.rows * (double)input_img.cols / 65535.0)));
            throw std::runtime_error("adaptive sampling: --tile-size " +
                std::to_string(tile_size) + " gives " + std::to_string(ntiles) +
                " tiles, but the container stores at most 65535 lod bytes; use --tile-size >= " +
                std::to_string(min_ts));
        }
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

    // LOD source priority mirrors encrypt_adaptive: inline LLM regions >
    // two-pass residual / spectral DCT > Laplacian; smoothing applies to all.
    if (!regions_json.empty() && (region_blend < 0.0f || region_blend > 1.0f)) {
        throw std::runtime_error("regions: blend must be in [0, 1]");
    }
    const std::vector<uint8_t> base_lod = two_pass
        ? compute_twopass_lod(tile_size, lod_min, pilot_ratio)
        : spectral_lod ? compute_spectral_lod(input_img, tile_size, lod_min)
        : compute_adaptive_lod(input_img, tile_size, lod_min);
    std::vector<uint8_t> lod = regions_json.empty()
        ? base_lod
        : blend_llm_lod(base_lod, parse_regions_json(regions_json),
            rows, cols, tile_size, lod_min, region_blend);
    lod = smooth_lod_grid(lod, rows, cols, tile_size, lod_min, lod_smooth);
    const int lod_bytes = (int)lod.size();
    const int lod_pixels = (lod_bytes + 2) / 3;

    // luma follows the LOD budget (capacity clamp may drop a few samples);
    // chroma stays uniform on the coarse grids
    returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size, weight_base, lod_full);
    const int mY_written = (int)ri_x.size();
    returnYcc420ChromaIndices(rows, cols, m, cs_key, mode);
    const int mC = m_chroma;

    cv::Mat ycc;
    cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> planes;
    cv::split(ycc, planes); // 0:Y 1:Cr 2:Cb to match COLOR_YCrCb2BGR order
    int crows, ccols;
    cs_chroma_dims(mode, rows, cols, crows, ccols);
    cv::Mat cr_small, cb_small;
    cv::resize(planes[1], cr_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
    cv::resize(planes[2], cb_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);

    const size_t meas_bytes = (size_t)mY_written + (size_t)2 * mC;
    const int meas_pixels = (int)((meas_bytes + 2) / 3);
    const int total = next_perfect_square(meas_pixels + CS_HEADER_PIXELS + lod_pixels + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }

    // mode (3 or 7) with the adaptive pad layout (tile_size, lod count, base)
    // so decrypt restores the exact budgeting
    buf[CS_OFF_PAD] = (uint8_t)mode;
    buf[CS_OFF_PAD + 1] = (uint8_t)(tile_size & 0xFF);
    buf[CS_OFF_PAD + 2] = (uint8_t)((tile_size >> 8) & 0xFF);
    buf[CS_OFF_PAD + 3] = (uint8_t)(lod_bytes & 0xFF);
    buf[CS_OFF_PAD + 4] = (uint8_t)((lod_bytes >> 8) & 0xFF);
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] = (uint8_t)(weight_base & 0xFF);
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] = (uint8_t)((weight_base >> 8) & 0xFF);
    buf[CS_OFF_PAD + CS_OFF_LOD_FULL] = (uint8_t)lod_full;

    std::memcpy(buf + CS_HEADER_BYTES, lod.data(), (size_t)lod_bytes);

    uint8_t* body = buf + CS_HEADER_BYTES + (size_t)lod_pixels * 3;
    for (int k = 0; k < mY_written; k++) {
        body[k] = planes[0].at<uint8_t>(ri_x[k], ri_y[k]);
    }
    for (int k = 0; k < mC; k++) {
        body[mY_written + k] = cr_small.at<uint8_t>(ri_cx1[k], ri_cy1[k]);
        body[mY_written + mC + k] = cb_small.at<uint8_t>(ri_cx2[k], ri_cy2[k]);
    }
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: lod stays raw; the three planes pack independently.
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        const size_t y0 = (size_t)CS_HEADER_BYTES + (size_t)lod_pixels * 3;
        encrypted_img = cs_pack_container_body(encrypted_img, {
            { y0, (size_t)mY_written, sample_bits, 0, false },
            { y0 + (size_t)mY_written, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
        buf = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }

    if (!cs_seal_header(buf, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
    sampling_mode = mode;
}

std::vector<uint8_t> encrypt_image::compute_twopass_lod(int tile_size, int lod_min, float pilot_ratio) {    if (tile_size <= 0) {
        throw std::runtime_error("two-pass sampling: tile size must be positive");
    }
    lod_min = (std::max)(0, (std::min)(255, lod_min));
    if (!cs_key_valid) {
        throw std::runtime_error("two-pass sampling: key not derived");
    }
    if (pilot_ratio < 0.01f) pilot_ratio = 0.01f;
    if (pilot_ratio > 0.25f) pilot_ratio = 0.25f;

    const int rows_l = input_img.rows;
    const int cols_l = input_img.cols;
    const int tiles_cols = (cols_l + tile_size - 1) / tile_size;
    const int tiles_rows = (rows_l + tile_size - 1) / tile_size;
    const int nt = tiles_rows * tiles_cols;

    // pilot RNG keyed deterministically (decrypt never sees these positions,
    // but both encrypt runs with the same password+salt must agree)
    uint32_t key_seed = 0;
    std::memcpy(&key_seed, cs_key, sizeof(key_seed));

    std::vector<float> energy((size_t)nt, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < nt; ++t) {
        const int tr = t / tiles_cols;
        const int tc = t % tiles_cols;
        const int row0 = tr * tile_size;
        const int col0 = tc * tile_size;
        const int tile_w = (tile_size < cols_l - col0) ? tile_size : cols_l - col0;
        const int tile_h = (tile_size < rows_l - row0) ? tile_size : rows_l - row0;
        const int tp = tile_w * tile_h;

        // uniform pilot: fixed fraction of tile pixels, min 16 for scoring
        // stability on small/edge tiles
        int pilot_c = (int)std::lround((double)tp * pilot_ratio);
        if (pilot_c < 16) pilot_c = 16;
        if (pilot_c > tp) pilot_c = tp;

        // partial Fisher-Yates over the tile (same draw style as the
        // keyed index routines, seeded per tile)
        std::vector<int> pool((size_t)tp);
        for (int i = 0; i < tp; ++i) pool[(size_t)i] = i;
        std::mt19937 rng(key_seed ^ (uint32_t)(t * 0x9E3779B1u + 0x85EBCA6Bu));
        std::vector<int> lx;
        std::vector<int> ly;
        lx.reserve((size_t)pilot_c);
        ly.reserve((size_t)pilot_c);
        for (int k = 0; k < pilot_c; ++k) {
            std::uniform_int_distribution<int> pick(k, tp - 1);
            const int j = pick(rng);
            std::swap(pool[(size_t)k], pool[(size_t)j]);
            lx.push_back(pool[(size_t)k] / tile_w);
            ly.push_back(pool[(size_t)k] % tile_w);
        }

        // cheap pilot recon: v1 ephemeral tile container + short per-channel
        // FISTA solves (16 steps, single unweighted pass, B->G->R chained)
        // from a DC warm start at the pilot mean — measures actual solver
        // difficulty, not a proxy, with the stronger solver
        cv::Mat tile = input_img(cv::Rect(col0, row0, tile_w, tile_h)).clone();
        encrypt_image tile_enc(tile, false);
        tile_enc.encrypt(lx, ly);
        const cv::Mat cont = tile_enc.get_mat();

        cv::Mat refs[3];
        for (int ch = 0; ch < 3; ++ch) {
            double mean = 0.0;
            for (int k = 0; k < pilot_c; ++k) {
                mean += tile.at<cv::Vec3b>(lx[k], ly[k])[ch] / 255.0;
            }
            mean /= pilot_c > 0 ? pilot_c : 1;
            refs[ch] = cv::Mat(tile_h, tile_w, CV_32F, cv::Scalar(0));
            // DCT/10 convention: DCT(const c) has DC = c*sqrt(n)
            refs[ch].at<float>(0, 0) = (float)(mean * std::sqrt((double)tp) / 10.0);
        }
        constexpr float pilot_coef = 0.045f; // low-ratio regime, matches auto rule
        constexpr int pilot_steps = 16;    // FISTA steps per channel (native
        constexpr int pilot_passes = 1;    // units; single pass keeps it cheap)
        reconstruct_color_channel_fista(cont, 0, pilot_coef, tile_h, tile_w, lx, ly, 0, refs[0], true, refs[1], 0.0f, pilot_passes, pilot_steps);
        reconstruct_color_channel_fista(cont, 1, pilot_coef, tile_h, tile_w, lx, ly, 0, refs[1], true, refs[2], 0.0f, pilot_passes, pilot_steps);
        reconstruct_color_channel_fista(cont, 2, pilot_coef, tile_h, tile_w, lx, ly, 0, refs[2], false, refs[2], 0.0f, pilot_passes, pilot_steps);

        // per-tile MSE vs source in [0,1] units (refs hold 0..255 planes)
        double se = 0.0;
        for (int ch = 0; ch < 3; ++ch) {
            const float* rp = (const float*)refs[ch].data;
            for (int y = 0; y < tile_h; ++y) {
                const cv::Vec3b* sp = tile.ptr<cv::Vec3b>(y);
                for (int x = 0; x < tile_w; ++x) {
                    const double d = (rp[(size_t)y * tile_w + x] - sp[x][ch]) / 255.0;
                    se += d * d;
                }
            }
        }
        float e = (float)(se / ((double)tp * 3.0));
        if (!std::isfinite(e)) e = 1e6f; // solver blowup => needs samples
        energy[(size_t)t] = e;
    }

    // same min-max + floor normalization as the Laplacian path
    float e_min = energy[0], e_max = energy[0];
    for (size_t i = 1; i < energy.size(); ++i) {
        e_min = (std::min)(e_min, energy[i]);
        e_max = (std::max)(e_max, energy[i]);
    }
    std::vector<uint8_t> lod((size_t)nt, (uint8_t)lod_min);
    const float range = e_max - e_min;
    for (size_t i = 0; i < energy.size(); ++i) {
        float n = 0.0f;
        if (range > 1e-9f) {
            n = (energy[i] - e_min) / range;
        }
        int v = lod_min + (int)std::lround((double)n * (255 - lod_min));
        lod[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return lod;
}

// Blend LLM region scores with a signal-metric base (Laplacian or
// two-pass bytes): per-tile score = detail * box/tile overlap fraction,
// unit-normalized, mixed with the unit-normalized base, then min-max mapped
// to [lod_min, 255]. A zero-range LLM map (no regions / all identical)
// falls back to the base untouched.
std::vector<uint8_t> blend_llm_lod(const std::vector<uint8_t>& base,
    const std::vector<LlmRegion>& regs, int rows_l, int cols_l, int tile_size,
    int lod_min, float blend) {
    const int tiles_cols = (cols_l + tile_size - 1) / tile_size;
    const int tiles_rows = (rows_l + tile_size - 1) / tile_size;
    const size_t nt = (size_t)tiles_rows * tiles_cols;
    if (base.size() != nt) {
        throw std::runtime_error("regions: base LOD grid mismatch");
    }

    std::vector<double> raw(nt, 0.0);
    for (const auto& r : regs) {
        const double rx0 = r.x0 / 1000.0 * cols_l;
        const double rx1 = r.x1 / 1000.0 * cols_l;
        const double ry0 = r.y0 / 1000.0 * rows_l;
        const double ry1 = r.y1 / 1000.0 * rows_l;
        int tc0 = (std::max)(0, (int)(rx0 / tile_size));
        int tc1 = (std::min)(tiles_cols - 1, (int)((rx1 - 1e-9) / tile_size));
        int tr0 = (std::max)(0, (int)(ry0 / tile_size));
        int tr1 = (std::min)(tiles_rows - 1, (int)((ry1 - 1e-9) / tile_size));
        for (int tr = tr0; tr <= tr1; ++tr) {
            for (int tc = tc0; tc <= tc1; ++tc) {
                const double col0 = (double)tc * tile_size;
                const double row0 = (double)tr * tile_size;
                const double tw = (std::min)((double)tile_size, (double)cols_l - col0);
                const double th = (std::min)((double)tile_size, (double)rows_l - row0);
                const double ix = (std::max)(0.0, (std::min)(rx1, col0 + tw) - (std::max)(rx0, col0));
                const double iy = (std::max)(0.0, (std::min)(ry1, row0 + th) - (std::max)(ry0, row0));
                if (ix > 0.0 && iy > 0.0) {
                    raw[(size_t)tr * tiles_cols + tc] += r.detail * (ix * iy) / (tw * th);
                }
            }
        }
    }

    double rmin = raw[0], rmax = raw[0];
    for (size_t i = 1; i < raw.size(); ++i) {
        rmin = (std::min)(rmin, raw[i]);
        rmax = (std::max)(rmax, raw[i]);
    }
    if (rmax - rmin <= 1e-12) {
        return base; // nothing to blend in
    }

    const double bden = (double)(255 - lod_min);
    std::vector<double> mixed(nt);
    for (size_t i = 0; i < nt; ++i) {
        const double lunit = (raw[i] - rmin) / (rmax - rmin);
        const double bunit = bden > 0.0 ? ((double)base[i] - lod_min) / bden : 0.0;
        mixed[i] = (double)blend * lunit + (1.0 - (double)blend) * bunit;
    }
    double mmin = mixed[0], mmax = mixed[0];
    for (size_t i = 1; i < mixed.size(); ++i) {
        mmin = (std::min)(mmin, mixed[i]);
        mmax = (std::max)(mmax, mixed[i]);
    }
    std::vector<uint8_t> lod(nt, (uint8_t)lod_min);
    const double mrange = mmax - mmin;
    for (size_t i = 0; i < mixed.size(); ++i) {
        double n = 0.0;
        if (mrange > 1e-12) {
            n = (mixed[i] - mmin) / mrange;
        }
        const int v = lod_min + (int)std::lround(n * (255 - lod_min));
        lod[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return lod;
}

// Gaussian smoothing of the LOD tile grid (sigma in tile units): removes
// the abrupt per-tile density rectangles by spreading budget across tile
// borders. The BLURRED bytes are shipped, so decrypt recomputes identical
// counts with no format change; the largest-remainder split keeps the total
// exactly m and the capacity clamp still holds. sigma <= 0 returns input.
std::vector<uint8_t> smooth_lod_grid(const std::vector<uint8_t>& lod,
    int rows_l, int cols_l, int tile_size, int lod_min, float sigma) {
    if (sigma <= 0.0f) return lod;
    if (sigma > 8.0f) sigma = 8.0f;
    const int tiles_cols = (cols_l + tile_size - 1) / tile_size;
    const int tiles_rows = (rows_l + tile_size - 1) / tile_size;
    const size_t nt = (size_t)tiles_rows * tiles_cols;
    if (lod.size() != nt || nt == 0) return lod;

    const int radius = (std::min)((std::max)(tiles_cols, tiles_rows) - 1,
        (int)std::ceil(3.0 * sigma));
    std::vector<double> kernel((size_t)2 * radius + 1);
    double ksum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double v = std::exp(-0.5 * (i / sigma) * (i / sigma));
        kernel[(size_t)(i + radius)] = v;
        ksum += v;
    }
    for (double& v : kernel) v /= ksum;

    std::vector<double> f(nt);
    for (size_t i = 0; i < nt; ++i) f[i] = lod[i];
    std::vector<double> tmp(nt);
    // separable passes with clamped edges
    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int cc = (std::max)(0, (std::min)(tiles_cols - 1, tc + i));
                acc += kernel[(size_t)(i + radius)] * f[(size_t)tr * tiles_cols + cc];
            }
            tmp[(size_t)tr * tiles_cols + tc] = acc;
        }
    }
    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int rr = (std::max)(0, (std::min)(tiles_rows - 1, tr + i));
                acc += kernel[(size_t)(i + radius)] * tmp[(size_t)rr * tiles_cols + tc];
            }
            f[(size_t)tr * tiles_cols + tc] = acc;
        }
    }

    // restore full contrast (blur compresses range), then quantize
    double fmin = f[0], fmax = f[0];
    for (size_t i = 1; i < f.size(); ++i) {
        fmin = (std::min)(fmin, f[i]);
        fmax = (std::max)(fmax, f[i]);
    }
    std::vector<uint8_t> out(nt, (uint8_t)lod_min);
    const double frange = fmax - fmin;
    for (size_t i = 0; i < f.size(); ++i) {
        double n = 0.0;
        if (frange > 1e-9) n = (f[i] - fmin) / frange;
        const int v = lod_min + (int)std::lround(n * (255 - lod_min));
        out[i] = (uint8_t)(std::max)(0, (std::min)(255, v));
    }
    return out;
}

/** @brief Adaptive periodic encryption: score every tile with an 8-bit LOD
 * byte (mean |Laplacian|, min-max + floor), split the measurement budget m
 * across tiles proportional to those bytes, draw a distinct in-tile pattern
 * per tile, and ship the lod byte array in the body (mode=2). Decrypt
 * regenerates identical indices from (key, lod, m, geometry).
 * With two_pass=true the LOD bytes come from a pilot uniform sample +
 * cheap per-tile recon (residual scoring) instead of the Laplacian; the
 * container format is identical so decrypt is unchanged.
 * @param pixel_p : sampling ratio (0.001 - 1.0)
 * @param password : encryption password
 * @param tile_size : adaptive grid tile size (must be > 0)
 * @param lod_min : floor byte so flat tiles keep a minimum sample share
 */
void encrypt_image::encrypt_hf_focus(const float& pixel_p,
    const std::string& password, int tile_size, int lod_min, int sample_bits, int chroma_bits) {
    if (tile_size <= 0) throw std::runtime_error("hf-focus tile size must be positive");
    // 16-bit lod-count header field: grids over 65535 tiles truncate it and
    // decrypt silently regenerates wrong positions. Fail fast (see
    // encrypt_adaptive for the full rationale).
    {
        const int ntiles = cs_lod_tile_count(input_img.rows, input_img.cols, tile_size);
        if (ntiles > 65535) {
            const int min_ts = (std::max)(1, (int)std::ceil(std::sqrt(
                (double)input_img.rows * (double)input_img.cols / 65535.0)));
            throw std::runtime_error("hf-focus: --tile-size " +
                std::to_string(tile_size) + " gives " + std::to_string(ntiles) +
                " tiles, but the container stores at most 65535 lod bytes; use --tile-size >= " +
                std::to_string(min_ts));
        }
    }
    bm = pixel_p;
    m = rows * cols * bm;
    uint8_t salt[CS_SALT_BYTES];
    if (!cs_random_bytes(salt, CS_SALT_BYTES) || !cs_derive_key(password, salt, cs_key))
        throw std::runtime_error("hf-focus key setup failed");
    cs_key_valid = true;
    constexpr int kWeightBase = 65535;
    const std::vector<uint8_t> lod = compute_adaptive_lod(input_img, tile_size, lod_min);
    const int lod_bytes = static_cast<int>(lod.size());
    const int lod_pixels = (lod_bytes + 2) / 3;
    std::vector<uint8_t> thumbnail_png;
    build_hf_thumb_png(input_img, thumbnail_png);
    const cv::Mat thumbnail = cv::imdecode(thumbnail_png, cv::IMREAD_COLOR);
    if (thumbnail.empty()) throw std::runtime_error("hf-focus thumbnail decode failed");
    std::vector<float> weights;
    int weight_cols = 0, weight_rows = 0;
    cs_hf_thumb_weights(thumbnail, weights, weight_cols, weight_rows);
    returnHfWeightedIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size,
        kWeightBase, weights, weight_cols, weight_rows);

    const uint32_t thumb_len = static_cast<uint32_t>(thumbnail_png.size());
    const size_t thumb_padded = (static_cast<size_t>(thumb_len) + 2) / 3 * 3;
    const int thumb_pixels = static_cast<int>(thumb_padded / 3);
    const int written = static_cast<int>(ri_x.size());
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + lod_pixels + thumb_pixels + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::string metadata = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buffer = encrypted_img.data;
    if (!cs_write_header(buffer, static_cast<size_t>(total) * 3, metadata, cs_key, salt))
        throw std::runtime_error("failed to write hf-focus header");
    buffer[CS_OFF_PAD] = CS_MODE_HF_FOCUS;
    buffer[CS_OFF_PAD + 1] = static_cast<uint8_t>(tile_size & 0xff);
    buffer[CS_OFF_PAD + 2] = static_cast<uint8_t>((tile_size >> 8) & 0xff);
    buffer[CS_OFF_PAD + 3] = static_cast<uint8_t>(lod_bytes & 0xff);
    buffer[CS_OFF_PAD + 4] = static_cast<uint8_t>((lod_bytes >> 8) & 0xff);
    buffer[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] = 0xff;
    buffer[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] = 0xff;
    for (int i = 0; i < 4; ++i)
        buffer[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + i] = static_cast<uint8_t>((thumb_len >> (i * 8)) & 0xff);
    std::memcpy(buffer + CS_HEADER_BYTES, lod.data(), lod.size());
    uint8_t* thumbnail_data = buffer + CS_HEADER_BYTES + static_cast<size_t>(lod_pixels) * 3;
    std::memcpy(thumbnail_data, thumbnail_png.data(), thumb_len);
    if (thumb_padded > thumb_len) std::memset(thumbnail_data + thumb_len, 0, thumb_padded - thumb_len);
    seal_hf_thumb(thumbnail_data, thumb_len, buffer + CS_OFF_IV, cs_key);
    const int measurement_offset = CS_HEADER_PIXELS + lod_pixels + thumb_pixels;
    for (int i = 0; i < written; ++i)
        encrypted_img.at<cv::Vec3b>(i + measurement_offset) = input_img.at<cv::Vec3b>(ri_x[i], ri_y[i]);
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: lod + encrypted thumbnail stay raw; flat measurements pack.
    size_t sealed_bytes = static_cast<size_t>(total) * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        encrypted_img = cs_pack_container_body(encrypted_img,
            { { (size_t)measurement_offset * 3, (size_t)3 * (size_t)written, sample_bits, chroma_bits, true } });
        buffer = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }
    if (!cs_seal_header(buffer, sealed_bytes, cs_key))
        throw std::runtime_error("failed to authenticate hf-focus header");
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    sampling_mode = CS_MODE_HF_FOCUS;
    periodic_tile = tile_size;
    periodic_samples = lod_bytes;
    adaptive_base = kWeightBase;
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

void encrypt_image::encrypt_ycc420_hf(const float& pixel_p,
    const std::string& password, int tile_size, int sample_bits, int chroma_bits, int mode) {
    if (tile_size <= 0) throw std::runtime_error("hf-focus tile size must be positive");
    bm = pixel_p;
    m = rows * cols * bm;
    uint8_t salt[CS_SALT_BYTES];
    if (!cs_random_bytes(salt, CS_SALT_BYTES) || !cs_derive_key(password, salt, cs_key))
        throw std::runtime_error("hf-focus key setup failed");
    cs_key_valid = true;
    std::vector<uint8_t> thumbnail_png;
    build_hf_thumb_png(input_img, thumbnail_png);
    const cv::Mat thumbnail = cv::imdecode(thumbnail_png, cv::IMREAD_COLOR);
    if (thumbnail.empty()) throw std::runtime_error("hf-focus thumbnail decode failed");
    std::vector<float> weights;
    int weight_cols = 0, weight_rows = 0;
    cs_hf_thumb_weights(thumbnail, weights, weight_cols, weight_rows);
    constexpr int kWeightBase = 65535;
    const int tile_cols = (cols + tile_size - 1) / tile_size;
    const int tile_rows = (rows + tile_size - 1) / tile_size;
    const std::vector<uint8_t> flat_lod(static_cast<size_t>(tile_cols) * tile_rows, 0);
    returnHfWeightedIndices(ri_x, ri_y, rows, cols, flat_lod, m, cs_key,
        tile_size, kWeightBase, weights, weight_cols, weight_rows);
    const int luma_count = static_cast<int>(ri_x.size());
    returnYcc420ChromaIndices(rows, cols, m, cs_key, mode);

    cv::Mat ycc;
    cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> planes;
    cv::split(ycc, planes);
    int chroma_rows = 0, chroma_cols = 0;
    cs_chroma_dims(mode, rows, cols, chroma_rows, chroma_cols);
    cv::Mat cr, cb;
    cv::resize(planes[1], cr, cv::Size(chroma_cols, chroma_rows), 0, 0, cv::INTER_AREA);
    cv::resize(planes[2], cb, cv::Size(chroma_cols, chroma_rows), 0, 0, cv::INTER_AREA);
    const uint32_t thumb_len = static_cast<uint32_t>(thumbnail_png.size());
    const size_t thumb_padded = (static_cast<size_t>(thumb_len) + 2) / 3 * 3;
    const int thumb_pixels = static_cast<int>(thumb_padded / 3);
    const size_t measure_bytes = static_cast<size_t>(luma_count) + static_cast<size_t>(2) * m_chroma;
    const int measure_pixels = static_cast<int>((measure_bytes + 2) / 3);
    const int total = next_perfect_square(CS_HEADER_PIXELS + thumb_pixels + measure_pixels + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::string metadata = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buffer = encrypted_img.data;
    if (!cs_write_header(buffer, static_cast<size_t>(total) * 3, metadata, cs_key, salt))
        throw std::runtime_error("failed to write hf-focus header");
    buffer[CS_OFF_PAD] = (uint8_t)mode;
    buffer[CS_OFF_PAD + 1] = static_cast<uint8_t>(tile_size & 0xff);
    buffer[CS_OFF_PAD + 2] = static_cast<uint8_t>((tile_size >> 8) & 0xff);
    buffer[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] = 0xff;
    buffer[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] = 0xff;
    for (int i = 0; i < 4; ++i)
        buffer[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + i] = static_cast<uint8_t>((thumb_len >> (i * 8)) & 0xff);
    uint8_t* thumbnail_data = buffer + CS_HEADER_BYTES;
    std::memcpy(thumbnail_data, thumbnail_png.data(), thumb_len);
    if (thumb_padded > thumb_len) std::memset(thumbnail_data + thumb_len, 0, thumb_padded - thumb_len);
    seal_hf_thumb(thumbnail_data, thumb_len, buffer + CS_OFF_IV, cs_key);
    uint8_t* body = thumbnail_data + thumb_padded;
    for (int i = 0; i < luma_count; ++i) body[i] = planes[0].at<uint8_t>(ri_x[i], ri_y[i]);
    for (int i = 0; i < m_chroma; ++i) {
        body[luma_count + i] = cr.at<uint8_t>(ri_cx1[i], ri_cy1[i]);
        body[luma_count + m_chroma + i] = cb.at<uint8_t>(ri_cx2[i], ri_cy2[i]);
    }
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buffer[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: encrypted thumbnail stays raw; Y/Cr/Cb pack independently.
    size_t sealed_bytes = static_cast<size_t>(total) * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        const size_t y0 = (size_t)CS_HEADER_BYTES + thumb_padded;
        encrypted_img = cs_pack_container_body(encrypted_img, {
            { y0, (size_t)luma_count, sample_bits, 0, false },
            { y0 + (size_t)luma_count, (size_t)m_chroma, chroma_bits, 0, false },
            { y0 + (size_t)luma_count + (size_t)m_chroma, (size_t)m_chroma, chroma_bits, 0, false } });
        buffer = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }
    if (!cs_seal_header(buffer, sealed_bytes, cs_key))
        throw std::runtime_error("failed to authenticate hf-focus header");
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    sampling_mode = mode;
    periodic_tile = tile_size;
    periodic_samples = 0;
    adaptive_base = kWeightBase;
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

void encrypt_image::encrypt_adaptive(const float& pixel_p, const std::string& password, int tile_size,
    int lod_min, int weight_base, bool two_pass, float pilot_ratio,
    const std::string& regions_json, float region_blend, float lod_smooth, int sample_bits, int chroma_bits, int lod_full, bool spectral_lod) {
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }
    // The lod byte count travels in a 16-bit header field: grids over 65535
    // tiles would truncate it and decrypt would silently regenerate wrong
    // positions (6 dB garbage, exit 0). Fail fast with the minimum viable
    // tile size instead.
    {
        const int ntiles = cs_lod_tile_count(input_img.rows, input_img.cols, tile_size);
        if (ntiles > 65535) {
            const int min_ts = (std::max)(1, (int)std::ceil(std::sqrt(
                (double)input_img.rows * (double)input_img.cols / 65535.0)));
            throw std::runtime_error("adaptive sampling: --tile-size " +
                std::to_string(tile_size) + " gives " + std::to_string(ntiles) +
                " tiles, but the container stores at most 65535 lod bytes; use --tile-size >= " +
                std::to_string(min_ts));
        }
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

    // LOD source priority: inline LLM regions (blended over the base) >
    // two-pass residual / spectral DCT > Laplacian. All reuse the identical
    // lod container layout, so decrypt is unchanged either way.
    if (!regions_json.empty() && (region_blend < 0.0f || region_blend > 1.0f)) {
        throw std::runtime_error("regions: blend must be in [0, 1]");
    }
    const std::vector<uint8_t> base_lod = two_pass
        ? compute_twopass_lod(tile_size, lod_min, pilot_ratio)
        : spectral_lod ? compute_spectral_lod(input_img, tile_size, lod_min)
        : compute_adaptive_lod(input_img, tile_size, lod_min);
    std::vector<uint8_t> lod = regions_json.empty()
        ? base_lod
        : blend_llm_lod(base_lod, parse_regions_json(regions_json),
            rows, cols, tile_size, lod_min, region_blend);
    // optional density smoothing: blurred bytes are shipped, so decrypt
    // agrees exactly with no format change
    lod = smooth_lod_grid(lod, rows, cols, tile_size, lod_min, lod_smooth);
    const int lod_bytes = (int)lod.size();
    const int lod_pixels = (lod_bytes + 2) / 3;

    // m and weight_base are fixed BEFORE index generation so decrypt
    // recomputes the same per-tile counts from the header — not from ri_x.size()
    returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size, weight_base, lod_full);
    const int m_written = (int)ri_x.size();
    if (lod_full > 0) {
        // --lod-full guarantee audit: qualifying tiles that cannot all fit
        // keep top-detail priority inside the shared counts function; say so
        long long want = 0;
        int nqual = 0;
        const int tx = (cols + tile_size - 1) / tile_size;
        const int ty = (rows + tile_size - 1) / tile_size;
        for (int tr = 0; tr < ty; ++tr) {
            for (int tc = 0; tc < tx; ++tc) {
                const int i = tr * tx + tc;
                if (lod[i] >= lod_full) {
                    ++nqual;
                    const int tw = (tile_size < cols - tc * tile_size) ? tile_size : cols - tc * tile_size;
                    const int th = (tile_size < rows - tr * tile_size) ? tile_size : rows - tr * tile_size;
                    want += (long long)tw * th;
                }
            }
        }
        if (nqual > 0 && want > m) {
            std::cerr << "Warning: --lod-full: " << nqual
                << " tiles qualify but the budget fits fewer; top-detail tiles win" << std::endl;
        }
    }

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
    buf[CS_OFF_PAD + CS_OFF_LOD_FULL] = (uint8_t)lod_full;

    // lod region immediately after the header (raw bytes, zero-padded to a
    // whole pixel so measurements stay pixel-aligned at CS_HEADER_PIXELS+lod_pixels)
    std::memcpy(buf + CS_HEADER_BYTES, lod.data(), (size_t)lod_bytes);

    // write measurements at regenerated indices
    const int meas0 = CS_HEADER_PIXELS + lod_pixels;
    for (int k = 0; k < m_written; k++) {
        encrypted_img.at<cv::Vec3b>(k + meas0) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
    }
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;

    // --sample-bits: lod stays raw; the flat measurement section packs.
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        encrypted_img = cs_pack_container_body(encrypted_img,
            { { (size_t)meas0 * 3, (size_t)3 * (size_t)m_written, sample_bits, chroma_bits, true } });
        buf = encrypted_img.data;
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }

    if (!cs_seal_header(buf, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }

    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
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

    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
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
    bool full_res,
    bool two_pass,
    float pilot_ratio,
    const std::string& regions_json,
    float region_blend,
    float lod_smooth,
    bool ycc420,
    bool hf_focus,
    int sample_bits,
    int chroma_bits,
    int lod_full,
    bool ycc422,
    bool spectral
){
    try {
        // the caller-requested ratio is honored in every mode: decryption
        // derives its own coef/iterations from the container's actual ratio,
        // so auto mode has no reason to force full sampling

        if (ycc420 && ycc422) {
            throw std::runtime_error("--ycc420 and --ycc422 are mutually exclusive");
        }

        if (compression_ratio < 0.001f || compression_ratio > 1.0f) {
            throw std::runtime_error("Compression ratio is outside the acceptable range of (0.001, 1.0]");
        }

        if (password.size() < 10) {
            throw std::runtime_error("Password should be at least 10 characters");
        }
        if (!cs_sample_bits_valid(sample_bits) ||
            (chroma_bits != 0 && !cs_sample_bits_valid(chroma_bits))) {
            throw std::runtime_error("--sample-bits luma and chroma values must each be in [1, 8]");
        }
        if (lod_full < 0 || lod_full > 255) {
            throw std::runtime_error("--lod-full must be in [0, 255] (0 = off)");
        }
        if (hf_focus && (adaptive || two_pass || !regions_json.empty() ||
            lod_smooth > 0.0f || adaptive_strength != 0.5f || lod_full > 0 || spectral)) {
            throw std::runtime_error("--hf-focus cannot be combined with adaptive sampling controls");
        }
        if (two_pass && spectral) {
            throw std::runtime_error("--two-pass and --spectral are mutually exclusive LOD scorers");
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

        if (hf_focus) {
            const int hf_tile_size = tile_size > 0 ? tile_size : 32;
            if (ycc420) {
                encrypt_img.encrypt_ycc420_hf(compression_ratio, password, hf_tile_size, sample_bits, chroma_bits);
            } else if (ycc422) {
                encrypt_img.encrypt_ycc420_hf(compression_ratio, password, hf_tile_size, sample_bits, chroma_bits, CS_MODE_YCC422_HF);
            } else {
                encrypt_img.encrypt_hf_focus(compression_ratio, password, hf_tile_size, lod_min, sample_bits, chroma_bits);
            }
        }
        else if (ycc420 || ycc422) {
            // luma/chroma-split sampling (mode travels in the header so
            // decrypt needs no new flags). Adaptive LOD flags
            // (--adaptive/--two-pass/--regions/--lod-smooth/--lod-full) are
            // absorbed: they drive the luma budget while chroma stays uniform.
            const int split_mode = ycc422 ? CS_MODE_YCC422 : CS_MODE_YCC420;
            const bool ycc_adaptive = adaptive || two_pass || !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0 || spectral;
            if (ycc_adaptive) {
                const int ts = tile_size > 0 ? tile_size : 64;
                const int wbase = cs_adaptive_base_from_strength(adaptive_strength);
                encrypt_img.encrypt_ycc420_adaptive(compression_ratio, password, ts, lod_min, wbase,
                    two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, sample_bits, chroma_bits, lod_full, split_mode, spectral);
            } else {
                encrypt_img.encrypt_ycc420(compression_ratio, password, sample_bits, chroma_bits, split_mode);
            }
        }
        else if (adaptive || lod_full > 0 || spectral) {
            // LOD-based per-tile sampling; default grid when tile_size unset.
            // two_pass / spectral swap the Laplacian scores for pilot-residual
            // / spectral-DCT scores; container format is identical so decrypt
            // needs no new flags.
            // lod_full>0 alone selects this path (detail guarantee needs LOD).
            const int ts = tile_size > 0 ? tile_size : 64;
            const int wbase = cs_adaptive_base_from_strength(adaptive_strength);
            encrypt_img.encrypt_adaptive(compression_ratio, password, ts, lod_min, wbase, two_pass, pilot_ratio,
                regions_json, region_blend, lod_smooth, sample_bits, chroma_bits, lod_full, spectral);
        }
        else if (tile_size > 0) {
            // periodic tile-based sampling: one random per-tile pattern,
            // replicated across the image; the mode travels in the header
            encrypt_img.encrypt_periodic(compression_ratio, password, tile_size, sample_bits, chroma_bits);
        } else {
            // standard random global sampling
            encrypt_img.encrypt(compression_ratio, password, sample_bits, chroma_bits);
        }

        if (encrypted_out) {
            encrypt_img.get_mat(*encrypted_out);
        }

        if (!output_path.empty()) {
            try {
                if (!cv::imwrite(output_path, encrypt_img.get_mat()))
                    throw std::runtime_error("failed to write container image: " + output_path);
            } catch (const cv::Exception& e) {
                throw std::runtime_error(std::string("failed to write container image: ") + e.what());
            }
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
