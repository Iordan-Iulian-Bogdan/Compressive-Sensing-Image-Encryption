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

// Measurement stamping helpers. cv::Mat::at<T>(r,c) recomputes
// data + step*r + c*elemSize() on every element and the compiler cannot hoist
// step out of the loop because the source and destination Mats alias through
// their members. Hoisting the row stride and copying fixed-size elements with
// memcpy lets the compiler emit a bare load/store per sample instead of an
// integer multiply plus a member reload. Output is bit-identical.
static inline void cs_gather_bgr(const uint8_t* base, size_t step,
                                 const int* rx, const int* ry, int m, uint8_t* dst) {
    for (int k = 0; k < m; ++k) {
        std::memcpy(dst + (size_t)k * 3,
                    base + (size_t)rx[k] * step + (size_t)ry[k] * 3, 3);
    }
}

static inline void cs_gather_u8(const uint8_t* base, size_t step,
                                const int* rx, const int* ry, int m, uint8_t* dst) {
    for (int k = 0; k < m; ++k) {
        dst[k] = base[(size_t)rx[k] * step + (size_t)ry[k]];
    }
}

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
    // No mult-8 stretch of global images: every sampler draws indices on the
    // rows x cols grid and gathers pixels from input_img, so a stretched
    // input_img shifted content by up to 4 output px on sizes not divisible
    // by 8 (cropping, not undoing, the stretch). Tilers handle partial tiles.
    (void)global_image;
}

// ---- container build steps shared by every encrypt_* mode ----

// Fresh random salt + PBKDF2-derived key for a new container.
void encrypt_image::derive_container_key(const std::string& password, uint8_t salt[CS_SALT_BYTES]) {
    if (!cs_random_bytes(salt, CS_SALT_BYTES)) {
        throw std::runtime_error("cryptographic RNG unavailable");
    }
    if (!cs_derive_key(password, salt, cs_key)) {
        throw std::runtime_error("key derivation (PBKDF2) failed");
    }
    cs_key_valid = true;
}

// Allocates the 1 x total container and writes its authenticated header.
// Metadata carries the budget m; returns the container byte buffer.
uint8_t* encrypt_image::alloc_container(int total, const uint8_t salt[CS_SALT_BYTES]) {
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    uint8_t* buf = encrypted_img.data;
    if (!cs_write_header(buf, total * 3, text, cs_key, salt)) {
        throw std::runtime_error("failed to write authenticated header");
    }
    return buf;
}

// Records the sample bit depths in the authenticated pad (chroma 0 = same as luma).
void encrypt_image::stamp_sample_bits(uint8_t* buf, int sample_bits, int& chroma_bits) {
    if (chroma_bits == 0) chroma_bits = sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;
}

// Bit-packs the given body sections (skipped at 8 bits), seals the header
// after the body exists (the tag covers header + all measurements), then wipes
// the key material and reshapes the container to a square.
void encrypt_image::finish_container(int total, int sample_bits, int chroma_bits,
    const std::vector<cs_body_section>& sections) {
    size_t sealed_bytes = (size_t)total * 3;
    if (sample_bits != 8 || chroma_bits != 8) {
        encrypted_img = cs_pack_container_body(encrypted_img, sections);
        sealed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    }
    if (!cs_seal_header(encrypted_img.data, sealed_bytes, cs_key)) {
        throw std::runtime_error("failed to seal header (HMAC)");
    }
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
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

bool cs_build_sample_class_views(const cv::Mat& input,
    const std::vector<int>& rix, const std::vector<int>& riy,
    int tile_size, cv::Mat& under, cv::Mat& over)
{
    under.release();
    over.release();
    if (input.empty() || input.type() != CV_8UC3) return false;
    if (rix.size() != riy.size() || rix.empty() || tile_size < 1) return false;
    const int rows = input.rows, cols = input.cols;
    const int ntr = (rows + tile_size - 1) / tile_size;
    const int ntc = (cols + tile_size - 1) / tile_size;
    if (ntr < 1 || ntc < 1) return false;
    // per-tile sample counts (ri_x = row, ri_y = col, same convention as
    // the container writers); out-of-range samples are ignored.
    std::vector<long long> counts((size_t)ntr * ntc, 0);
    for (size_t k = 0; k < rix.size(); ++k) {
        const int r = rix[k], c = riy[k];
        if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
        counts[(size_t)(r / tile_size) * ntc + (c / tile_size)] += 1;
    }
    long long total = 0;
    for (long long v : counts) total += v;
    if (total <= 0) return false;
    const double mean = (double)total / (double)counts.size();
    under = cv::Mat::zeros(rows, cols, CV_8UC3);
    over = cv::Mat::zeros(rows, cols, CV_8UC3);
    long long n_under = 0, n_over = 0;
    for (int tr = 0; tr < ntr; ++tr) {
        for (int tc = 0; tc < ntc; ++tc) {
            const long long cnt = counts[(size_t)tr * ntc + tc];
            cv::Mat* dst = nullptr;
            if ((double)cnt < mean) { dst = &under; ++n_under; }
            else if ((double)cnt > mean) { dst = &over; ++n_over; }
            else continue;  // ties go in neither canvas
            const int y0 = tr * tile_size, x0 = tc * tile_size;
            const int h = (std::min)(tile_size, rows - y0);
            const int w = (std::min)(tile_size, cols - x0);
            if (h <= 0 || w <= 0) continue;
            input(cv::Rect(x0, y0, w, h)).copyTo((*dst)(cv::Rect(x0, y0, w, h)));
        }
    }
    if (n_under == 0 && n_over == 0) {
        // exact-uniform grid (e.g. periodic): nothing to split
        under.release();
        over.release();
        return false;
    }
    auto label = [](cv::Mat& panel, const char* t) {
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    char info[96];
    std::snprintf(info, sizeof(info), "under-sampled tiles: %lld", n_under);
    label(under, info);
    std::snprintf(info, sizeof(info), "over-sampled tiles: %lld", n_over);
    label(over, info);
    return true;
}

void encrypt_image::present_sample_classes(const std::string& prefix, int tile_size) const {
    cv::Mat under, over;
    if (!cs_build_sample_class_views(input_img, ri_x, ri_y, tile_size, under, over)) {
        std::cerr << "Warning: --sample-split: no samplable tile grid "
                     "(uniform counts or missing indices); nothing written\n";
        return;
    }
    const std::string under_path = prefix + "_under.png";
    const std::string over_path = prefix + "_over.png";
    bool ok = true;
    if (!cv::imwrite(under_path, under)) {
        std::cerr << "Warning: --sample-split: failed to write " << under_path << "\n";
        ok = false;
    }
    if (!cv::imwrite(over_path, over)) {
        std::cerr << "Warning: --sample-split: failed to write " << over_path << "\n";
        ok = false;
    }
    if (ok) {
        std::cout << "Sample-class tiles written to " << under_path
                  << " and " << over_path << std::endl;
    }
}

/** @brief top-level encryption: v2 authenticated header.
 * The passphrase is stretched with PBKDF2-HMAC-SHA256 using a random 16-byte
 * salt stored in the header; the resulting 256-bit key seeds the index RNG
 * and encrypts/authenticates the metadata header (AES-256-CTR + HMAC-SHA256
 * over header and measurement body). Key material is wiped after sealing.
 */
void encrypt_image::encrypt(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits, const std::vector<CsHrSample>& hr_detail) {
    bm = pixel_p;
    m = rows * cols * bm;
    int n = rows * cols;
    ri_x.resize(m);
    ri_y.resize(m);

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);

    const int hr_pixels = (int)hr_detail.size() * (CS_HRDETAIL_SAMPLE_BYTES / 3);
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8 + hr_pixels);
    uint8_t* buf = alloc_container(total, salt);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), m,
                  encrypted_img.data + (size_t)CS_HEADER_PIXELS * 3);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // HR-detail anchors ride as a trailing CTR-encrypted body section
    // (mode 4); bit-packing would corrupt the ciphertext, so the prototype
    // requires 8-bit sample depths here.
    if (!hr_detail.empty()) {
        if (sample_bits != 8 || chroma_bits != 8) {
            throw std::runtime_error("--hr-detail requires 8-bit samples (no --sample-bits packing)");
        }
        buf[CS_OFF_PAD] = CS_MODE_BGR_HRDETAIL;
        const uint32_t hr_n = (uint32_t)hr_detail.size();
        for (int b = 0; b < 4; b++) {
            buf[CS_OFF_PAD + CS_OFF_HR_COUNT + b] = (uint8_t)((hr_n >> (b * 8)) & 0xff);
        }
        cs_write_hr_payload(buf + CS_HEADER_BYTES + (size_t)3 * (size_t)m,
            hr_detail, cs_key, buf + CS_OFF_IV);
        sampling_mode = CS_MODE_BGR_HRDETAIL;
    }

    // --sample-bits: bit-pack the flat measurement section (3 values per
    // sample at byte offset CS_HEADER_BYTES); skipped at 8 bits.
    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
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
    derive_container_key(password, salt);

    // samples per canonical tile from the requested ratio; the total count m
    // follows from the generated pattern (edge tiles hold fewer samples).
    // At extreme low ratios (e.g. 0.001 with a small tile) the product can
    // round to 0 — keep one sample per tile so the pattern stays valid.
    int samples_per_tile = (int)std::lround((double)tile_size * tile_size * pixel_p);
    if (samples_per_tile < 1) samples_per_tile = 1;
    returnPeriodicIndices(ri_x, ri_y, rows, cols, samples_per_tile, cs_key, tile_size);
    m = (int)ri_x.size();

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    uint8_t* buf = alloc_container(total, salt);

    // record the sampling mode in the authenticated pad region:
    // [pad+0] = 1 (periodic), [pad+1..3) tile_size LE, [pad+3..5) samples_per_tile LE
    buf[CS_OFF_PAD] = 1;
    buf[CS_OFF_PAD + 1] = (uint8_t)(tile_size & 0xFF);
    buf[CS_OFF_PAD + 2] = (uint8_t)((tile_size >> 8) & 0xFF);
    buf[CS_OFF_PAD + 3] = (uint8_t)(samples_per_tile & 0xFF);
    buf[CS_OFF_PAD + 4] = (uint8_t)((samples_per_tile >> 8) & 0xFF);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), m,
                  encrypted_img.data + (size_t)CS_HEADER_PIXELS * 3);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // --sample-bits: bit-pack the flat measurement section (3 values per
    // sample at byte offset CS_HEADER_BYTES); skipped at 8 bits.
    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
}

/** @brief T-cluster point sampling (mode 13): random subset of (4x4 cell,
 * T-group) pairs from the fixed bench partition; all 4 pixels of each
 * selected pair are stored as individual point measurements (gaps between
 * clusters). Same container layout as uniform random; only the index draw
 * differs, so decrypt needs no new flags beyond the mode byte.
 */
void encrypt_image::encrypt_tcluster(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits) {
    bm = pixel_p;
    const int want = (int)((long long)rows * cols * bm);
    // pairs of 4 point measurements; ratio 1.0 selects every pair (full
    // coverage, every pixel exactly once). Keep at least one pair so the
    // pattern stays valid at extreme low ratios.
    int pairs = (want + 3) / 4;
    if (pairs < 1) pairs = 1;
    const int max_pairs = (rows / 4) * (cols / 4) * 4;
    if (pairs > max_pairs) pairs = max_pairs;

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    std::vector<int> pair_ids;
    returnTetrominoPairIds(pair_ids, rows, cols, pairs, cs_key);
    ri_x.clear();
    ri_y.clear();
    ri_x.reserve((size_t)pairs * 4);
    ri_y.reserve((size_t)pairs * 4);
    const int cell_cols = cols / 4;
    int xs[4], ys[4];
    for (int p : pair_ids) {
        const int cell = p / 4, g = p % 4;
        cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
        for (int q = 0; q < 4; ++q) {
            ri_x.push_back(ys[q]);
            ri_y.push_back(xs[q]);
        }
    }
    m = (int)ri_x.size();

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = CS_MODE_TCLUSTER;
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), m,
                  encrypted_img.data + (size_t)CS_HEADER_PIXELS * 3);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
    sampling_mode = CS_MODE_TCLUSTER;
}

/** @brief T-means grouped sampling (mode 14): random subset of (4x4 cell,
 * T-group) pairs; each selected pair contributes ONE measurement per
 * channel = the rounded mean of its 4 pixels. Max useful ratio is 0.25
 * (every pair selected = the bench fixed-geometry case); higher requests
 * clamp with a note. Solver needs the averaging operator (weight-aware
 * scatter); point-sample code paths must not interpret these values.
 */
void encrypt_image::encrypt_tmeans(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits) {
    bm = pixel_p;
    const int max_pairs = (rows / 4) * (cols / 4) * 4;
    int pairs = (int)((long long)rows * cols * bm);
    if (pairs < 1) pairs = 1;
    if (pairs > max_pairs) {
        std::cout << "Note: --sampling tmeans clamps to full pair coverage ("
                  << max_pairs << " measurements)" << std::endl;
        pairs = max_pairs;
    }

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    std::vector<int> pair_ids;
    returnTetrominoPairIds(pair_ids, rows, cols, pairs, cs_key);
    // member ri_* keep every group pixel (mask display + sample-split
    // classification); the solve uses the group structure, not these.
    ri_x.clear();
    ri_y.clear();
    ri_x.reserve((size_t)pairs * 4);
    ri_y.reserve((size_t)pairs * 4);
    const int cell_cols = cols / 4;
    int xs[4], ys[4];
    std::vector<cv::Vec3b> means;
    means.reserve((size_t)pairs);
    for (int p : pair_ids) {
        const int cell = p / 4, g = p % 4;
        cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
        int acc[3] = { 0, 0, 0 };
        for (int q = 0; q < 4; ++q) {
            ri_x.push_back(ys[q]);
            ri_y.push_back(xs[q]);
            const cv::Vec3b v = input_img.at<cv::Vec3b>(ys[q], xs[q]);
            acc[0] += v[0];
            acc[1] += v[1];
            acc[2] += v[2];
        }
        // rounded to nearest, half ties up (bench convention)
        means.push_back(cv::Vec3b((uint8_t)((acc[0] + 2) / 4), (uint8_t)((acc[1] + 2) / 4), (uint8_t)((acc[2] + 2) / 4)));
    }
    m = (int)means.size();

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = CS_MODE_TMEANS;
    for (int k = 0; k < m; k++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = means[(size_t)k];
    }
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)CS_HEADER_BYTES, (size_t)3 * (size_t)m, sample_bits, chroma_bits, true } });
    sampling_mode = CS_MODE_TMEANS;
}

/** @brief YCC 4:2:0 split encryption (mode 3): BGR -> YCrCb, chroma planes
 * downsampled 2x (INTER_AREA), then luma sampled at full resolution and each
 * chroma plane sampled on its own coarse grid with a domain-separated key.
 * The body is byte-packed [Y x m][Cr x m_chroma][Cb x m_chroma] so the
 * container holds ~1.5 bytes per luma sample instead of 3 (BGR modes).
 * Metadata carries the luma budget m; chroma counts re-derive from it.
 */
void encrypt_image::encrypt_ycc420(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits, int mode, const std::vector<CsHrSample>& hr_detail) {
    bm = pixel_p;
    m = rows * cols * bm;

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

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
    const int hr_pixels = (int)hr_detail.size() * (CS_HRDETAIL_SAMPLE_BYTES / 3);
    const int total = next_perfect_square(body_pixels + CS_HEADER_PIXELS + 8 + hr_pixels);
    uint8_t* buf = alloc_container(total, salt);

    // mode (3 or 7) in the authenticated pad region; decrypt restores it
    // from here
    buf[CS_OFF_PAD] = (uint8_t)mode;

    uint8_t* body = buf + CS_HEADER_BYTES;
    cs_gather_u8(planes[0].data, planes[0].step, ri_x.data(), ri_y.data(), m, body);
    cs_gather_u8(cr_small.data, cr_small.step, ri_cx1.data(), ri_cy1.data(), mC, body + m);
    cs_gather_u8(cb_small.data, cb_small.step, ri_cx2.data(), ri_cy2.data(), mC, body + m + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // HR-detail anchors ride as a trailing CTR-encrypted body section;
    // bit-packing would corrupt the ciphertext, so 8-bit depths only.
    if (!hr_detail.empty()) {
        if (mode != CS_MODE_YCC420_HRDETAIL) {
            throw std::runtime_error("--hr-detail ycc payload needs mode 9 (internal error)");
        }
        if (sample_bits != 8 || chroma_bits != 8) {
            throw std::runtime_error("--hr-detail requires 8-bit samples (no --sample-bits packing)");
        }
        const uint32_t hr_n = (uint32_t)hr_detail.size();
        for (int b = 0; b < 4; b++) {
            buf[CS_OFF_PAD + CS_OFF_HR_COUNT + b] = (uint8_t)((hr_n >> (b * 8)) & 0xff);
        }
        cs_write_hr_payload(body + body_bytes, hr_detail, cs_key, buf + CS_OFF_IV);
    }

    // --sample-bits: bit-pack the three planes independently (luma, Cr,
    // Cb); skipped at 8 bits.
    const size_t y0 = (size_t)CS_HEADER_BYTES;
    finish_container(total, sample_bits, chroma_bits,
        {
            { y0, (size_t)m, sample_bits, 0, false },
            { y0 + (size_t)m, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)m + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
    sampling_mode = mode;
}

/** @brief YCC split with T-tetromino luma (modes 15-18): luma sampled from
 * the fixed bench T partition on the full-res Y plane — T-cluster stores Y
 * at all 4 pixels of each selected pair, T-means stores one rounded Y mean
 * per pair (max useful ratio 0.25, higher clamps with a note). Chroma keeps
 * the uniform coarse-grid draw on the base split geometry (420/422 via
 * split_mode). Body layout identical to encrypt_ycc420: [Y x m][Cr][Cb].
 * Metadata carries the luma budget m; decrypt regenerates the pair stream
 * and the chroma draws from (key, m, geometry).
 */
void encrypt_image::encrypt_ycc420_t(const float& pixel_p, const std::string& password, int sample_bits, int chroma_bits,
    int split_mode, int tmode) {
    if ((split_mode != CS_MODE_YCC420 && split_mode != CS_MODE_YCC422) ||
        (tmode != CS_MODE_YCC420_TCLUSTER && tmode != CS_MODE_YCC420_TMEANS &&
         tmode != CS_MODE_YCC422_TCLUSTER && tmode != CS_MODE_YCC422_TMEANS)) {
        throw std::runtime_error("ycc tetromino: bad split/t mode (internal error)");
    }
    if ((split_mode == CS_MODE_YCC420) !=
        (tmode == CS_MODE_YCC420_TCLUSTER || tmode == CS_MODE_YCC420_TMEANS)) {
        throw std::runtime_error("ycc tetromino: split/mode mismatch (internal error)");
    }
    bm = pixel_p;
    const bool means = (tmode == CS_MODE_YCC420_TMEANS || tmode == CS_MODE_YCC422_TMEANS);
    const int max_pairs = (rows / 4) * (cols / 4) * 4;
    int pairs;
    if (means) {
        pairs = (int)((long long)rows * cols * bm);
        if (pairs < 1) pairs = 1;
        if (pairs > max_pairs) {
            std::cout << "Note: --sampling tmeans clamps to full pair coverage ("
                      << max_pairs << " luma measurements)" << std::endl;
            pairs = max_pairs;
        }
    } else {
        const int want = (int)((long long)rows * cols * bm);
        pairs = (want + 3) / 4;
        if (pairs < 1) pairs = 1;
        if (pairs > max_pairs) pairs = max_pairs;
    }

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    std::vector<int> pair_ids;
    returnTetrominoPairIds(pair_ids, rows, cols, pairs, cs_key);
    const int cell_cols = cols / 4;
    int xs[4], ys[4];
    // member ri_* keep every luma group pixel (mask display + sample-split
    // classification); the luma solve uses the pair stream, not these.
    ri_x.clear();
    ri_y.clear();
    ri_x.reserve(pair_ids.size() * 4);
    ri_y.reserve(pair_ids.size() * 4);
    cv::Mat ycc;
    cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> planes;
    cv::split(ycc, planes);
    std::vector<uint8_t> luma;
    luma.reserve(pair_ids.size() * (means ? 1 : 4));
    for (int p : pair_ids) {
        const int cell = p / 4, g = p % 4;
        cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
        if (means) {
            int acc = 0;
            for (int q = 0; q < 4; ++q) {
                ri_x.push_back(ys[q]);
                ri_y.push_back(xs[q]);
                acc += planes[0].at<uint8_t>(ys[q], xs[q]);
            }
            luma.push_back((uint8_t)((acc + 2) / 4));
        } else {
            for (int q = 0; q < 4; ++q) {
                ri_x.push_back(ys[q]);
                ri_y.push_back(xs[q]);
                luma.push_back(planes[0].at<uint8_t>(ys[q], xs[q]));
            }
        }
    }
    m = (int)luma.size();

    // chroma stays uniform on the base split geometry
    returnYcc420ChromaIndices(rows, cols, m, cs_key, split_mode);
    const int mC = m_chroma;
    int crows, ccols;
    cs_chroma_dims(split_mode, rows, cols, crows, ccols);
    cv::Mat cr_small, cb_small;
    cv::resize(planes[1], cr_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
    cv::resize(planes[2], cb_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);

    const size_t body_bytes = (size_t)m + (size_t)2 * mC;
    const int body_pixels = (int)((body_bytes + 2) / 3);
    const int total = next_perfect_square(body_pixels + CS_HEADER_PIXELS + 8);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = (uint8_t)tmode;

    uint8_t* body = buf + CS_HEADER_BYTES;
    for (int k = 0; k < m; k++) {
        body[k] = luma[(size_t)k];
    }
    cs_gather_u8(cr_small.data, cr_small.step, ri_cx1.data(), ri_cy1.data(), mC, body + m);
    cs_gather_u8(cb_small.data, cb_small.step, ri_cx2.data(), ri_cy2.data(), mC, body + m + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    const size_t y0 = (size_t)CS_HEADER_BYTES;
    finish_container(total, sample_bits, chroma_bits,
        {
            { y0, (size_t)m, sample_bits, 0, false },
            { y0 + (size_t)m, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)m + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
    sampling_mode = tmode;
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
    const std::string& regions_json, float region_blend, float lod_smooth, int sample_bits, int chroma_bits, int lod_full, int mode, bool spectral_lod, float lod_gamma, int lod_cutoff) {
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
    derive_container_key(password, salt);

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
    // LOD contrast exponent (--adaptive-gamma): reshapes the shipped bytes
    // (decrypt recomputes identical counts, no format change).
    cs_apply_lod_gamma(lod, lod_min, lod_gamma);
    const int lod_bytes = (int)lod.size();
    const int lod_pixels = (lod_bytes + 2) / 3;

    // luma follows the LOD budget (capacity clamp may drop a few samples);
    // chroma stays uniform on the coarse grids
    returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size, weight_base, lod_full, lod_cutoff);
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
    uint8_t* buf = alloc_container(total, salt);

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
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_CUTOFF] = (uint8_t)lod_cutoff;

    std::memcpy(buf + CS_HEADER_BYTES, lod.data(), (size_t)lod_bytes);
    // v3: the LOD map is a per-tile detail surrogate -> seal it (CS_LOD_TAG).
    cs_blob_seal_xor(buf + CS_HEADER_BYTES, (size_t)lod_bytes, cs_key, buf + CS_OFF_IV, CS_LOD_TAG);

    uint8_t* body = buf + CS_HEADER_BYTES + (size_t)lod_pixels * 3;
    cs_gather_u8(planes[0].data, planes[0].step, ri_x.data(), ri_y.data(), mY_written, body);
    cs_gather_u8(cr_small.data, cr_small.step, ri_cx1.data(), ri_cy1.data(), mC, body + mY_written);
    cs_gather_u8(cb_small.data, cb_small.step, ri_cx2.data(), ri_cy2.data(), mC, body + mY_written + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // --sample-bits: lod stays raw; the three planes pack independently.
    const size_t y0 = (size_t)CS_HEADER_BYTES + (size_t)lod_pixels * 3;
    finish_container(total, sample_bits, chroma_bits,
        {
            { y0, (size_t)mY_written, sample_bits, 0, false },
            { y0 + (size_t)mY_written, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
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

// --seg-mask/--seg-rates (mode 10): per-segment sampling -------------------

std::vector<float> parse_seg_rates(const std::string& s) {
    std::vector<float> out;
    size_t p = 0;
    while (true) {
        const size_t q = s.find(',', p);
        std::string tok = (q == std::string::npos) ? s.substr(p) : s.substr(p, q - p);
        size_t b = 0;
        while (b < tok.size() && std::isspace((unsigned char)tok[b])) ++b;
        size_t e = tok.size();
        while (e > b && std::isspace((unsigned char)tok[e - 1])) --e;
        tok = tok.substr(b, e - b);
        if (tok.empty()) {
            throw std::runtime_error("seg-rates: empty entry (rates are comma-separated, e.g. 0.02,0.6)");
        }
        try {
            size_t pos = 0;
            const float v = std::stof(tok, &pos);
            if (pos != tok.size()) throw std::runtime_error("trailing characters");
            out.push_back(v);
        }
        catch (...) {
            throw std::runtime_error("seg-rates: bad number '" + tok + "' (each rate in [0, 1])");
        }
        if (q == std::string::npos) break;
        p = q + 1;
    }
    if (out.empty() || out.size() > 255) {
        throw std::runtime_error("seg-rates: need 1..255 per-class rates");
    }
    for (float r : out) {
        if (!(r >= 0.0f) || !(r <= 1.0f)) {
            throw std::runtime_error("seg-rates: each rate must be in [0, 1]");
        }
    }
    return out;
}

// Scene palette (BGR) matching scene.segmentation.json: color masks resolve
// each pixel to its nearest palette entry (squared Euclidean distance).
static const cv::Vec3b cs_segmask_palette[8] = {
    cv::Vec3b(110,110,110), cv::Vec3b(70,200,70), cv::Vec3b(230,120,40), cv::Vec3b(100,225,245),
    cv::Vec3b(50,110,180), cv::Vec3b(210,70,200), cv::Vec3b(235,220,80), cv::Vec3b(50,170,245)
};

cv::Mat cs_segmask_resolve_ids(const cv::Mat& mask_raw, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("seg-mask: bad pipeline geometry");
    }
    if (mask_raw.empty()) {
        throw std::runtime_error("seg-mask: empty mask image");
    }
    if (mask_raw.type() == CV_8UC1) {
        cv::Mat ids;
        if (mask_raw.rows != rows || mask_raw.cols != cols) {
            cv::resize(mask_raw, ids, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
        }
        else {
            ids = mask_raw.clone();
        }
        return ids;
    }
    if (mask_raw.type() == CV_8UC3 || mask_raw.type() == CV_8UC4) {
        cv::Mat bgr;
        if (mask_raw.type() == CV_8UC4) cv::cvtColor(mask_raw, bgr, cv::COLOR_BGRA2BGR);
        else bgr = mask_raw;
        if (bgr.rows != rows || bgr.cols != cols) {
            cv::resize(bgr, bgr, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
        }
        cv::Mat ids(bgr.size(), CV_8U);
        for (int y = 0; y < bgr.rows; ++y) {
            const cv::Vec3b* srow = bgr.ptr<cv::Vec3b>(y);
            uint8_t* drow = ids.ptr<uint8_t>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                int best = 0, bd = -1;
                for (int c = 0; c < 8; ++c) {
                    const int db = (int)srow[x][0] - (int)cs_segmask_palette[c][0];
                    const int dg = (int)srow[x][1] - (int)cs_segmask_palette[c][1];
                    const int dr = (int)srow[x][2] - (int)cs_segmask_palette[c][2];
                    const int d = db * db + dg * dg + dr * dr;
                    if (bd < 0 || d < bd) { bd = d; best = c; }
                }
                drow[x] = (uint8_t)best;
            }
        }
        return ids;
    }
    throw std::runtime_error("seg-mask: need an 8-bit grayscale label map or a BGR color mask");
}

void cs_seg_auto_ids(const cv::Mat& bgr, int classes, int rows, int cols, cv::Mat& ids) {
    if (classes < 2 || classes > 255) {
        throw std::runtime_error("--seg-auto needs 2..255 classes");
    }
    if (bgr.empty() || bgr.channels() != 3) {
        throw std::runtime_error("--seg-auto needs a color image");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("--seg-auto: bad pipeline geometry");
    }
    // Working copy capped at 480px long side (INTER_AREA down, k-means is
    // O(pixels * classes)); labels return NEAREST to pipeline geometry.
    const int lw = bgr.cols, lh = bgr.rows;
    const double s = (std::max)(lw, lh) > 480 ? 480.0 / (std::max)(lw, lh) : 1.0;
    cv::Mat small;
    if (s < 1.0) cv::resize(bgr, small, cv::Size(), s, s, cv::INTER_AREA);
    else small = bgr;
    cv::Mat lab;
    cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);
    cv::Mat data;
    lab.convertTo(data, CV_32F);
    data = data.reshape(1, small.rows * small.cols);

    // Deterministic: fixed OpenCV RNG seed, global state restored after.
    const uint64_t saved_rng = cv::theRNG().state;
    cv::setRNGSeed(0x5EEDC10C);
    cv::Mat labels, centers;
    cv::kmeans(data, classes, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS, centers);
    cv::theRNG().state = saved_rng;

    cv::Mat flat = labels.reshape(1, small.rows);
    // Remap by size desc (class 0 = largest): stable, content-ordered ids.
    std::vector<int> sizes((size_t)classes, 0);
    for (int y = 0; y < flat.rows; ++y) {
        const int32_t* row = flat.ptr<int32_t>(y);
        for (int x = 0; x < flat.cols; ++x) {
            if (row[x] >= 0 && row[x] < classes) sizes[(size_t)row[x]]++;
        }
    }
    std::vector<int> order((size_t)classes);
    for (int c = 0; c < classes; ++c) order[(size_t)c] = c;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return sizes[(size_t)a] > sizes[(size_t)b]; });
    int lut[256];
    for (int c = 0; c < 256; ++c) lut[c] = 0;
    int next_id = 0;
    for (int c : order) {
        if (sizes[(size_t)c] == 0) continue; // k-means dropped this cluster
        lut[c] = next_id++;
    }
    cv::Mat remapped(flat.size(), CV_8U);
    for (int y = 0; y < flat.rows; ++y) {
        const int32_t* srow = flat.ptr<int32_t>(y);
        uint8_t* drow = remapped.ptr<uint8_t>(y);
        for (int x = 0; x < flat.cols; ++x) drow[x] = (uint8_t)lut[srow[x]];
    }

    // Reassign speckle components (< max(48, 0.05% of seg pixels)) to the
    // 5x5 neighborhood mode from the pre-cleanup map (one pass). Skipped
    // when the image is too fragmented (> 25% speckle: keep raw k-means).
    const int total = remapped.rows * remapped.cols;
    const int min_area = (std::max)(48, total * 5 / 10000);
    cv::Mat tiny = cv::Mat::zeros(remapped.size(), CV_8U);
    long long tiny_count = 0;
    for (int c = 0; c < next_id; ++c) {
        cv::Mat is_c = (remapped == c);
        cv::Mat comp, stats, cent;
        const int ncomp = cv::connectedComponentsWithStats(is_c, comp, stats, cent, 8, CV_32S);
        for (int k = 1; k < ncomp; ++k) {
            if (stats.at<int>(k, cv::CC_STAT_AREA) < min_area) {
                cv::Mat sel = (comp == k);
                sel.copyTo(tiny, sel);
                tiny_count += stats.at<int>(k, cv::CC_STAT_AREA);
            }
        }
    }
    cv::Mat clean = remapped.clone();
    if (tiny_count > 0 && tiny_count <= total / 4) {
        for (int y = 0; y < clean.rows; ++y) {
            uint8_t* drow = clean.ptr<uint8_t>(y);
            for (int x = 0; x < clean.cols; ++x) {
                if (!tiny.at<uint8_t>(y, x)) continue;
                int hist[256] = { 0 };
                for (int dy = -2; dy <= 2; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= clean.rows) continue;
                    const uint8_t* srow = remapped.ptr<uint8_t>(yy);
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int xx = x + dx;
                        if (xx < 0 || xx >= clean.cols) continue;
                        hist[srow[xx]]++;
                    }
                }
                int best = drow[x], bv = -1;
                for (int c = 0; c < next_id; ++c) {
                    if (hist[c] > bv) { bv = hist[c]; best = c; }
                }
                drow[x] = (uint8_t)best;
            }
        }
    }

    if (clean.rows == rows && clean.cols == cols) {
        ids = clean;
        return;
    }
    // Nearest-up of the 480px map leaves an ~8x blocky, edge-UNAWARE contour.
    // Instead guided-filter each class indicator with the full-resolution
    // image so every boundary snaps to a real image edge. Guided filter is
    // built from box means, so no contrib dependency.
    {
        cv::Mat guide;
        {
            cv::Mat g = bgr;
            if (g.size() != cv::Size(cols, rows))
                cv::resize(bgr, g, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
            cv::cvtColor(g, guide, cv::COLOR_BGR2GRAY);
            guide.convertTo(guide, CV_32F, 1.0 / 255.0);
        }
        const double factor = (double)(std::max)(rows, cols) / (double)(std::max)(clean.rows, clean.cols);
        const int radius = (std::max)(4, (std::min)(16, (int)std::lround(factor)));
        const float eps = 1e-3f;
        const cv::Size k(2 * radius + 1, 2 * radius + 1);
        cv::Mat I2, mI, mII;
        cv::multiply(guide, guide, I2);
        cv::boxFilter(guide, mI, CV_32F, k);
        cv::boxFilter(I2, mII, CV_32F, k);
        cv::Mat best(rows, cols, CV_32F, cv::Scalar(-1.0f));
        cv::Mat out(rows, cols, CV_8U, cv::Scalar(0));
        for (int c = 0; c < next_id; ++c) {
            cv::Mat ind = (clean == c);
            ind.convertTo(ind, CV_32F);
            cv::Mat up;
            cv::resize(ind, up, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
            cv::Mat Ip, mp, mIp;
            cv::multiply(guide, up, Ip);
            cv::boxFilter(up, mp, CV_32F, k);
            cv::boxFilter(Ip, mIp, CV_32F, k);
            cv::Mat varI = mII - mI.mul(mI);
            cv::Mat covIp = mIp - mI.mul(mp);
            cv::Mat a = covIp / (varI + eps);
            cv::Mat b = mp - a.mul(mI);
            cv::Mat ma, mb, q;
            cv::boxFilter(a, ma, CV_32F, k);
            cv::boxFilter(b, mb, CV_32F, k);
            q = ma.mul(guide) + mb;
            for (int y = 0; y < rows; ++y) {
                const float* qr = q.ptr<float>(y);
                float* br = best.ptr<float>(y);
                uint8_t* orow = out.ptr<uint8_t>(y);
                for (int x = 0; x < cols; ++x) {
                    if (qr[x] > br[x]) { br[x] = qr[x]; orow[x] = (uint8_t)c; }
                }
            }
        }
        cv::medianBlur(out, ids, 3);
    }
}

std::vector<float> cs_seg_auto_rates(const cv::Mat& bgr, const cv::Mat& ids, int classes) {
    std::vector<float> rates((size_t)(classes > 0 ? classes : 1), 1.0f);
    if (bgr.empty() || ids.empty() || bgr.size() != ids.size() || classes <= 0) {
        return rates;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    std::vector<double> sum((size_t)classes, 0.0);
    std::vector<long long> cnt((size_t)classes, 0);
    for (int y = 0; y < ids.rows; ++y) {
        const uint8_t* irow = ids.ptr<uint8_t>(y);
        const float* mrow = mag.ptr<float>(y);
        for (int x = 0; x < ids.cols; ++x) {
            const int c = irow[x];
            if (c < 0 || c >= classes) continue;
            sum[(size_t)c] += mrow[x];
            cnt[(size_t)c]++;
        }
    }
    double dmax = 0.0;
    std::vector<double> det((size_t)classes, 0.0);
    for (int c = 0; c < classes; ++c) {
        if (cnt[(size_t)c] > 0) det[(size_t)c] = sum[(size_t)c] / (double)cnt[(size_t)c];
        if (det[(size_t)c] > dmax) dmax = det[(size_t)c];
    }
    if (!(dmax > 0.0)) return rates; // flat image: uniform 1.0
    // Bottom-anchored density law: flat classes bottom out at 0.08, the
    // busiest at 1.0.
    for (int c = 0; c < classes; ++c) {
        const double f = det[(size_t)c] / dmax;
        rates[(size_t)c] = (float)(0.08 + 0.92 * std::pow(f, 0.75));
    }
    return rates;
}

/** @brief Per-segment encryption: each class id of the mask draws its own
 * budgeted share (count_c = round(pix_c * q_c / 255) with q the quantized
 * rate AFTER the global pixel_p scale), then one global shuffle fixes the
 * measurement order. The body prefix [nclasses][rates][seglen][label PNG]
 * ships everything decrypt needs (mode=10); geometry downstream is the
 * standard BGR pipeline, so tiling/decrypt are unchanged.
 */
void encrypt_image::encrypt_segmask(const float& pixel_p, const std::string& password,
    const std::string& mask_path, const std::vector<float>& rates,
    int sample_bits, int chroma_bits, int seg_blend) {
    cv::Mat mask_raw = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
    if (mask_raw.empty()) {
        throw std::runtime_error("seg-mask: cannot load '" + mask_path + "'");
    }
    cv::Mat ids = cs_segmask_resolve_ids(mask_raw, rows, cols);
    encrypt_segmask_ids(pixel_p, password, ids, rates, sample_bits, chroma_bits, seg_blend);
}

void encrypt_image::encrypt_segmask_ids(const float& pixel_p, const std::string& password,
    const cv::Mat& ids, const std::vector<float>& rates,
    int sample_bits, int chroma_bits, int seg_blend) {
    if (rates.empty() || rates.size() > 255) {
        throw std::runtime_error("seg-mask: need 1..255 per-class rates (--seg-rates)");
    }
    if (seg_blend < 0 || seg_blend > CS_SEGBLEND_MAX) {
        throw std::runtime_error("--seg-blend must be in [0, 128] pixels");
    }
    if (ids.empty() || ids.type() != CV_8U || ids.rows != rows || ids.cols != cols) {
        throw std::runtime_error("seg-mask: label map must be 8-bit at pipeline geometry");
    }

    // Quantize AFTER the global scale so decrypt (which only sees the
    // shipped bytes) reproduces the counts exactly. Ids past the rate list
    // score 0.
    std::vector<uint8_t> q;
    q.reserve(rates.size());
    for (float r : rates) {
        float e = r * pixel_p;
        if (e < 0.0f) e = 0.0f;
        if (e > 1.0f) e = 1.0f;
        q.push_back(cs_segmask_quantize_rate(e));
    }

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    returnSegMaskIndices(ri_x, ri_y, rows, cols, ids, q, cs_key, seg_blend, true);
    const int m_written = (int)ri_x.size();
    if (m_written == 0) {
        throw std::runtime_error("seg-mask: rates select zero samples (raise a rate)");
    }
    m = m_written;
    bm = (float)m / (float)((long long)rows * cols);
    if (seg_blend > 0) {
        // Bitrate cost of the boosted boundary bands vs the unblended draw.
        std::vector<int> bx, by;
        returnSegMaskIndices(bx, by, rows, cols, ids, q, cs_key, 0);
        const double over = bx.empty() ? 0.0 : 100.0 * ((double)m_written - (double)bx.size()) / (double)bx.size();
        std::cout << "seg-mask: smooth blend " << seg_blend << "px sample change "
            << over << "% samples (" << bx.size() << " -> " << m_written << ")" << std::endl;
    }

    // Label-map payload at pipeline geometry (decrypt decodes + redraws).
    std::vector<uint8_t> png;
    const std::vector<int> png_params = { cv::IMWRITE_PNG_COMPRESSION, 6 };
    if (!cv::imencode(".png", ids, png, png_params) || png.empty() ||
        png.size() > (static_cast<size_t>(1) << 24)) {
        throw std::runtime_error("seg-mask: label PNG encode failed");
    }
    std::vector<uint8_t> prefix;
    prefix.reserve(1 + q.size() + 4);
    if (seg_blend > 0) prefix.insert(prefix.end(), {0, 1, (uint8_t)seg_blend});
    prefix.push_back((uint8_t)q.size());
    prefix.insert(prefix.end(), q.begin(), q.end());
    const uint32_t sl = (uint32_t)png.size();
    for (int b = 0; b < 4; ++b) prefix.push_back((uint8_t)((sl >> (b * 8)) & 0xff));
    const int pre_pixels = (int)((prefix.size() + png.size() + 2) / 3);

    const int total = next_perfect_square(m + CS_HEADER_PIXELS + pre_pixels + 8);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = CS_MODE_SEGMASK;
    std::memcpy(buf + CS_HEADER_BYTES, prefix.data(), prefix.size());
    std::memcpy(buf + CS_HEADER_BYTES + prefix.size(), png.data(), png.size());
    // v3: the label map is a content surrogate -> seal the whole prefix with a
    // domain-separated CTR stream; its byte length rides in the MAC-covered pad
    // so decrypt can find it without reading the (now opaque) prefix.
    const size_t pre_bytes = prefix.size() + png.size();
    for (int b = 0; b < 4; ++b)
        buf[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + b] = (uint8_t)((pre_bytes >> (b * 8)) & 0xff);
    if (!cs_blob_seal_xor(buf + CS_HEADER_BYTES, pre_bytes, cs_key, buf + CS_OFF_IV, CS_SEG_TAG))
        throw std::runtime_error("seg-mask: label prefix encryption failed");

    const int meas0 = CS_HEADER_PIXELS + pre_pixels;
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), m_written,
                  encrypted_img.data + (size_t)meas0 * 3);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // --sample-bits: prefix + label PNG stay raw; the flat BGR measurement
    // section packs.
    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)meas0 * 3, (size_t)3 * (size_t)m_written, sample_bits, chroma_bits, true } });
    sampling_mode = CS_MODE_SEGMASK;
}

/** @brief Per-segment luma + uniform chroma (modes 11/12): the luma budget
 * follows the seg-mask class rates exactly like encrypt_segmask while each
 * chroma plane keeps its uniform coarse-grid draw (domain-separated keys).
 * Body: [seg prefix + label PNG][Y x mY_written][Cr x mC][Cb x mC].
 * Metadata carries the luma budget m; decrypt regenerates the luma draw
 * from the shipped prefix and re-derives the chroma draws from (key, m).
 */
void encrypt_image::encrypt_ycc420_segmask(const float& pixel_p, const std::string& password,
    const std::string& mask_path, const std::vector<float>& rates,
    int sample_bits, int chroma_bits, int mode, int seg_blend) {
    cv::Mat mask_raw = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
    if (mask_raw.empty()) {
        throw std::runtime_error("seg-mask: cannot load '" + mask_path + "'");
    }
    cv::Mat ids = cs_segmask_resolve_ids(mask_raw, rows, cols);
    encrypt_ycc420_segmask_ids(pixel_p, password, ids, rates, sample_bits, chroma_bits, mode, seg_blend);
}

void encrypt_image::encrypt_ycc420_segmask_ids(const float& pixel_p, const std::string& password,
    const cv::Mat& ids, const std::vector<float>& rates,
    int sample_bits, int chroma_bits, int mode, int seg_blend) {
    if (mode != CS_MODE_YCC420_SEG && mode != CS_MODE_YCC422_SEG) {
        throw std::runtime_error("seg-mask ycc: bad split mode (internal error)");
    }
    if (rates.empty() || rates.size() > 255) {
        throw std::runtime_error("seg-mask: need 1..255 per-class rates (--seg-rates)");
    }
    if (seg_blend < 0 || seg_blend > CS_SEGBLEND_MAX) {
        throw std::runtime_error("--seg-blend must be in [0, 128] pixels");
    }
    if (ids.empty() || ids.type() != CV_8U || ids.rows != rows || ids.cols != cols) {
        throw std::runtime_error("seg-mask: label map must be 8-bit at pipeline geometry");
    }

    std::vector<uint8_t> q;
    q.reserve(rates.size());
    for (float r : rates) {
        float e = r * pixel_p;
        if (e < 0.0f) e = 0.0f;
        if (e > 1.0f) e = 1.0f;
        q.push_back(cs_segmask_quantize_rate(e));
    }

    uint8_t salt[CS_SALT_BYTES];
    derive_container_key(password, salt);

    // luma follows the segment budget; chroma stays uniform on the mode's
    // coarse grids (same helper as the uniform/LOD-luma split paths)
    returnSegMaskIndices(ri_x, ri_y, rows, cols, ids, q, cs_key, seg_blend, true);
    const int mY_written = (int)ri_x.size();
    if (mY_written == 0) {
        throw std::runtime_error("seg-mask: rates select zero samples (raise a rate)");
    }
    m = mY_written;
    bm = (float)m / (float)((long long)rows * cols);
    // chroma dims/counts key off the 4:2:0 vs 4:2:2 geometry (modes 3/7),
    // not the seg-luma variants, so map to the base split mode first
    const int base = (mode == CS_MODE_YCC422_SEG) ? CS_MODE_YCC422 : CS_MODE_YCC420;
    returnYcc420ChromaIndices(rows, cols, m, cs_key, base);
    const int mC = m_chroma;

    cv::Mat ycc;
    cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> planes;
    cv::split(ycc, planes); // 0:Y 1:Cr 2:Cb to match COLOR_YCrCb2BGR order
    int crows, ccols;
    cs_chroma_dims(base, rows, cols, crows, ccols);
    cv::Mat cr_small, cb_small;
    cv::resize(planes[1], cr_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
    cv::resize(planes[2], cb_small, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);

    std::vector<uint8_t> png;
    const std::vector<int> png_params = { cv::IMWRITE_PNG_COMPRESSION, 6 };
    if (!cv::imencode(".png", ids, png, png_params) || png.empty() ||
        png.size() > (static_cast<size_t>(1) << 24)) {
        throw std::runtime_error("seg-mask: label PNG encode failed");
    }
    std::vector<uint8_t> prefix;
    prefix.reserve(1 + q.size() + 4);
    if (seg_blend > 0) prefix.insert(prefix.end(), {0, 1, (uint8_t)seg_blend});
    prefix.push_back((uint8_t)q.size());
    prefix.insert(prefix.end(), q.begin(), q.end());
    const uint32_t sl = (uint32_t)png.size();
    for (int b = 0; b < 4; ++b) prefix.push_back((uint8_t)((sl >> (b * 8)) & 0xff));
    if (seg_blend > 0) {
        std::vector<int> bx, by;
        returnSegMaskIndices(bx, by, rows, cols, ids, q, cs_key, 0);
        const double over = bx.empty() ? 0.0 : 100.0 * ((double)mY_written - (double)bx.size()) / (double)bx.size();
        std::cout << "seg-mask: smooth blend " << seg_blend << "px luma sample change "
            << over << "% luma samples (" << bx.size() << " -> " << mY_written << ")" << std::endl;
    }
    const size_t pre_bytes = prefix.size() + png.size();

    const size_t meas_bytes = (size_t)mY_written + (size_t)2 * mC;
    const int meas_pixels = (int)((meas_bytes + 2) / 3);
    const int pre_pixels = (int)((pre_bytes + 2) / 3);
    const int total = next_perfect_square(meas_pixels + CS_HEADER_PIXELS + pre_pixels + 8);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = (uint8_t)mode;
    std::memcpy(buf + CS_HEADER_BYTES, prefix.data(), prefix.size());
    std::memcpy(buf + CS_HEADER_BYTES + prefix.size(), png.data(), png.size());
    // v3: seal the seg prefix + label map (see encrypt_segmask_ids).
    for (int b = 0; b < 4; ++b)
        buf[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + b] = (uint8_t)((pre_bytes >> (b * 8)) & 0xff);
    if (!cs_blob_seal_xor(buf + CS_HEADER_BYTES, pre_bytes, cs_key, buf + CS_OFF_IV, CS_SEG_TAG))
        throw std::runtime_error("seg-mask: label prefix encryption failed");

    uint8_t* body = buf + CS_HEADER_BYTES + pre_bytes;
    cs_gather_u8(planes[0].data, planes[0].step, ri_x.data(), ri_y.data(), mY_written, body);
    cs_gather_u8(cr_small.data, cr_small.step, ri_cx1.data(), ri_cy1.data(), mC, body + mY_written);
    cs_gather_u8(cb_small.data, cb_small.step, ri_cx2.data(), ri_cy2.data(), mC, body + mY_written + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // --sample-bits: prefix + label PNG stay raw; the three planes pack
    // independently.
    const size_t y0 = (size_t)CS_HEADER_BYTES + pre_bytes;
    finish_container(total, sample_bits, chroma_bits,
        {
            { y0, (size_t)mY_written, sample_bits, 0, false },
            { y0 + (size_t)mY_written, (size_t)mC, chroma_bits, 0, false },
            { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, chroma_bits, 0, false } });
    sampling_mode = mode;
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
    cs_blob_seal_xor(buffer + CS_HEADER_BYTES, lod.size(), cs_key, buffer + CS_OFF_IV, CS_LOD_TAG);
    uint8_t* thumbnail_data = buffer + CS_HEADER_BYTES + static_cast<size_t>(lod_pixels) * 3;
    std::memcpy(thumbnail_data, thumbnail_png.data(), thumb_len);
    if (thumb_padded > thumb_len) std::memset(thumbnail_data + thumb_len, 0, thumb_padded - thumb_len);
    seal_hf_thumb(thumbnail_data, thumb_len, buffer + CS_OFF_IV, cs_key);
    const int measurement_offset = CS_HEADER_PIXELS + lod_pixels + thumb_pixels;
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), written,
                  encrypted_img.data + (size_t)measurement_offset * 3);
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
    cs_gather_u8(planes[0].data, planes[0].step, ri_x.data(), ri_y.data(), luma_count, body);
    cs_gather_u8(cr.data, cr.step, ri_cx1.data(), ri_cy1.data(), m_chroma, body + luma_count);
    cs_gather_u8(cb.data, cb.step, ri_cx2.data(), ri_cy2.data(), m_chroma, body + luma_count + m_chroma);
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
    const std::string& regions_json, float region_blend, float lod_smooth, int sample_bits, int chroma_bits, int lod_full, bool spectral_lod, float lod_gamma, int lod_cutoff) {
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
    derive_container_key(password, salt);

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
    // LOD contrast exponent (--adaptive-gamma): reshapes the shipped bytes
    // (decrypt recomputes identical counts, no format change).
    cs_apply_lod_gamma(lod, lod_min, lod_gamma);
    const int lod_bytes = (int)lod.size();
    const int lod_pixels = (lod_bytes + 2) / 3;

    // m and weight_base are fixed BEFORE index generation so decrypt
    // recomputes the same per-tile counts from the header — not from ri_x.size()
    returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, tile_size, weight_base, lod_full, lod_cutoff);
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
    // metadata carries the budget m (count-input), not m_written: capacity
    // clamping can drop a few samples but both sides recompute counts from m
    uint8_t* buf = alloc_container(total, salt);

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
    buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_CUTOFF] = (uint8_t)lod_cutoff;

    // lod region immediately after the header, sealed (CS_LOD_TAG); zero-padded
    // to a whole pixel so measurements stay pixel-aligned at
    // CS_HEADER_PIXELS+lod_pixels
    std::memcpy(buf + CS_HEADER_BYTES, lod.data(), (size_t)lod_bytes);
    cs_blob_seal_xor(buf + CS_HEADER_BYTES, (size_t)lod_bytes, cs_key, buf + CS_OFF_IV, CS_LOD_TAG);

    // write measurements at regenerated indices
    const int meas0 = CS_HEADER_PIXELS + lod_pixels;
    cs_gather_bgr(input_img.data, input_img.step, ri_x.data(), ri_y.data(), m_written,
                  encrypted_img.data + (size_t)meas0 * 3);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // --sample-bits: lod stays raw; the flat measurement section packs.
    finish_container(total, sample_bits, chroma_bits,
        { { (size_t)meas0 * 3, (size_t)3 * (size_t)m_written, sample_bits, chroma_bits, true } });
}

/** @brief tile re-encryption used in decrypt_image_tiled: v1 plaintext header.
 * These containers are ephemeral (in-memory only) and carry no password
 * context; measurements start at CS_HEADER_PIXELS, same as v2 containers.
 */
void encrypt_image::encrypt_values(const std::vector<cv::Vec3b>& vals) {
    m = (int)vals.size();
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    cs_write_header_plain(encrypted_img.data, total * 3, text);

    for (int k = 0; k < m; k++) {
        encrypted_img.at<cv::Vec3b>(k + CS_HEADER_PIXELS) = vals[(size_t)k];
    }

    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

void encrypt_image::encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g) {
    m = ri_x_g.size();
    int n = rows * cols;
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width);
    cs_write_header_plain(encrypted_img.data, total * 3, text);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    cs_gather_bgr(input_img.data, input_img.step, ri_x_g.data(), ri_y_g.data(), m,
                  encrypted_img.data + (size_t)CS_HEADER_PIXELS * 3);

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
    bool spectral,
    const std::string& sample_split,
    int hr_detail,
    float lod_gamma,
    int lod_cutoff,
    const std::string& seg_mask,
    const std::string& seg_rates,
    int seg_blend,
    const std::string& sampling,
    int seg_auto
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
        if (lod_gamma < 0.1f || lod_gamma > 4.0f) {
            throw std::runtime_error("--adaptive-gamma must be in [0.1, 4] (1 = off)");
        }
        if (lod_gamma != 1.0f && !(adaptive || two_pass || spectral ||
            !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0)) {
            throw std::runtime_error("--adaptive-gamma needs an adaptive sampling path (--adaptive/--two-pass/--spectral/--regions/--lod-smooth/--lod-full)");
        }
        if (lod_cutoff < 0 || lod_cutoff > 255) {
            throw std::runtime_error("--adaptive-cutoff must be in [0, 255] (0 = off)");
        }
        if (lod_cutoff > 0 && !(adaptive || two_pass || spectral ||
            !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0)) {
            throw std::runtime_error("--adaptive-cutoff needs an adaptive sampling path (--adaptive/--two-pass/--spectral/--regions/--lod-smooth/--lod-full)");
        }
        if (hf_focus && (adaptive || two_pass || !regions_json.empty() ||
            lod_smooth > 0.0f || adaptive_strength != 0.5f || lod_full > 0 || spectral)) {
            throw std::runtime_error("--hf-focus cannot be combined with adaptive sampling controls");
        }
        if (two_pass && spectral) {
            throw std::runtime_error("--two-pass and --spectral are mutually exclusive LOD scorers");
        }
        if (hr_detail < 0 || hr_detail > CS_HRDETAIL_MAX_SAMPLES) {
            throw std::runtime_error("--hr-detail must be in [0, 65536] (0 = off)");
        }
        // T-shaped sampling (modes 13/14) is BGR-only and exclusive with
        // every other sampling path; decrypt auto-detects from the mode.
        const bool tmodes = (sampling == "tcluster" || sampling == "tmeans");
        if (sampling != "points" && !tmodes) {
            throw std::runtime_error("--sampling must be points|tcluster|tmeans");
        }
        if (tmodes && (hf_focus || adaptive || two_pass || spectral || !regions_json.empty() ||
            lod_smooth > 0.0f || lod_full > 0 || tile_size > 0 || hr_detail > 0 ||
            lod_cutoff > 0 || lod_gamma != 1.0f || !seg_mask.empty() || !seg_rates.empty() ||
            seg_blend != 0)) {
            throw std::runtime_error("--sampling tcluster|tmeans is exclusive with other sampling flags (--adaptive/--periodic/--tile-size/--hf-focus/--hr-detail/--seg-mask/--seg-rates/--seg-blend/--lod-*)");
        }
        // Per-segment sampling (mode 10) is exclusive with every other
        // sampling path; the unused adaptive knobs self-reject below
        // (--adaptive-gamma/--adaptive-cutoff need an adaptive path) except
        // strength/floor, which are rejected here.
        if (seg_auto != 0 && (seg_auto < 2 || seg_auto > 255)) {
            throw std::runtime_error("--seg-auto needs 2..255 classes (0 = off)");
        }
        if (seg_auto > 0 && !seg_mask.empty()) {
            throw std::runtime_error("--seg-auto and --seg-mask are mutually exclusive (two label sources)");
        }
        const bool seg_mode = !seg_mask.empty() || !seg_rates.empty() || seg_auto > 0;
        if (seg_mode && !seg_auto && (seg_mask.empty() || seg_rates.empty())) {
            throw std::runtime_error("--seg-mask and --seg-rates must be given together");
        }
        if (seg_blend != 0 && !seg_mode) {
            throw std::runtime_error("--seg-blend needs --seg-mask/--seg-rates");
        }
        if (seg_blend < 0 || seg_blend > CS_SEGBLEND_MAX) {
            throw std::runtime_error("--seg-blend must be in [0, 128] pixels");
        }
        if (seg_mode && (hf_focus || adaptive || two_pass || spectral || !regions_json.empty() ||
            lod_smooth > 0.0f || lod_full > 0 || tile_size > 0 || hr_detail > 0 ||
            lod_cutoff > 0 || adaptive_strength != 0.5f || lod_min != 32)) {
            throw std::runtime_error("--seg-mask/--seg-auto is exclusive with other sampling flags (--adaptive/--two-pass/--spectral/--regions/--lod-smooth/--lod-full/--adaptive-cutoff/--adaptive-strength/--adaptive-floor/--periodic/--tile-size/--hf-focus/--hr-detail)");
        }
        if (hr_detail > 0) {
            // Prototype scope: uniform sampling only (BGR mode 0, YCC420
            // mode 3), 8-bit depths, no HF/adaptive/periodic/spectral
            // controls. Decrypt auto-detects the payload from the mode byte.
            if (hf_focus || adaptive || two_pass || spectral || !regions_json.empty() ||
                lod_smooth > 0.0f || lod_full > 0 || tile_size > 0) {
                throw std::runtime_error("--hr-detail needs uniform sampling (no --adaptive/--two-pass/--spectral/--regions/--lod-smooth/--lod-full/--periodic/--hf-focus/--tile-size)");
            }
            if (ycc422) {
                throw std::runtime_error("--hr-detail supports --ycc420 only (no --ycc422 in this prototype)");
            }
            const int cb_eff = (chroma_bits == 0) ? sample_bits : chroma_bits;
            if (sample_bits != 8 || cb_eff != 8) {
                throw std::runtime_error("--hr-detail requires 8-bit samples (no --sample-bits packing)");
            }
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

        // HR-detail anchors are ground-truth native luma, so select them
        // before the default 2x downscale. Reference geometry = decrypt
        // output geometry (native, or native-resized on odd dimensions).
        std::vector<CsHrSample> hr_samples;
        if (hr_detail > 0) {
            const int out_w = full_res ? input_img.cols : input_img.cols / 2 * 2;
            const int out_h = full_res ? input_img.rows : input_img.rows / 2 * 2;
            hr_samples = cs_select_hr_detail(input_img, out_w, out_h, hr_detail);
            if (hr_samples.empty()) {
                throw std::runtime_error("--hr-detail found no edge anchors (flat image?)");
            }
            std::cout << "HR-detail: " << hr_samples.size() << " anchors ("
                      << hr_samples.size() * CS_HRDETAIL_SAMPLE_BYTES << " payload bytes)" << std::endl;
        }

        // Default pipeline runs at half resolution (encrypt downsamples, decrypt
        // 2x-upscales the solve). --full-res skips both so geometry stays native.
        if (!full_res) {
            cv::resize(input_img, input_img, cv::Size(input_img.cols / 2, input_img.rows / 2));
        }
        encrypt_image encrypt_img(input_img, true);

        if (!seg_mask.empty()) {
            // per-segment sampling from the label/color mask: BGR (mode 10),
            // or seg-luma + uniform chroma under --ycc420/--ycc422 (modes
            // 11/12). Rates travel in the body prefix in both cases, so
            // decrypt needs no new flags.
            const std::vector<float> seg_rate_list = parse_seg_rates(seg_rates);
            if (ycc420 || ycc422) {
                const int seg_mode = ycc422 ? CS_MODE_YCC422_SEG : CS_MODE_YCC420_SEG;
                encrypt_img.encrypt_ycc420_segmask(compression_ratio, password, seg_mask, seg_rate_list, sample_bits, chroma_bits, seg_mode, seg_blend);
            }
            else {
                encrypt_img.encrypt_segmask(compression_ratio, password, seg_mask, seg_rate_list, sample_bits, chroma_bits, seg_blend);
            }
        }
        else if (seg_auto > 0) {
            // built-in segmentation: k-means ids at pipeline geometry plus
            // detail-driven rates (or the --seg-rates override, whose short
            // lists leave trailing classes at 0). Same containers/modes as
            // --seg-mask, so decrypt needs no new flags.
            // Segment at the sampling geometry (rows x cols members, which
            // the ctor leaves unstretched; the mult-8 stretch applies to
            // input_img pixels only and sampling never addresses it).
            cv::Mat auto_ids;
            cs_seg_auto_ids(input_img, seg_auto, input_img.rows, input_img.cols, auto_ids);
            std::vector<float> auto_rates = seg_rates.empty()
                ? cs_seg_auto_rates(input_img, auto_ids, seg_auto)
                : parse_seg_rates(seg_rates);
            {
                std::vector<long long> cls_sz((size_t)seg_auto, 0);
                for (int y = 0; y < auto_ids.rows; ++y) {
                    const uint8_t* row = auto_ids.ptr<uint8_t>(y);
                    for (int x = 0; x < auto_ids.cols; ++x) {
                        if (row[x] < seg_auto) cls_sz[(size_t)row[x]]++;
                    }
                }
                std::cout << "seg-auto: " << seg_auto << " classes";
                for (int c = 0; c < seg_auto; ++c) {
                    char rb[32];
                    std::snprintf(rb, sizeof(rb), "%.2f",
                        c < (int)auto_rates.size() ? auto_rates[(size_t)c] : 0.0f);
                    std::cout << " [" << c << "]px=" << cls_sz[(size_t)c] << " r=" << rb;
                }
                std::cout << (seg_rates.empty() ? " (auto rates)" : " (--seg-rates override)") << std::endl;
            }
            if (ycc420 || ycc422) {
                const int seg_mode = ycc422 ? CS_MODE_YCC422_SEG : CS_MODE_YCC420_SEG;
                encrypt_img.encrypt_ycc420_segmask_ids(compression_ratio, password, auto_ids, auto_rates, sample_bits, chroma_bits, seg_mode, seg_blend);
            }
            else {
                encrypt_img.encrypt_segmask_ids(compression_ratio, password, auto_ids, auto_rates, sample_bits, chroma_bits, seg_blend);
            }
        }
        else if (hf_focus) {
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
            // decrypt needs no new flags). T-tetromino luma rides the same
            // split geometry with uniform chroma (modes 15-18).
            const int split_mode = ycc422 ? CS_MODE_YCC422 : CS_MODE_YCC420;
            if (sampling == "tcluster" || sampling == "tmeans") {
                const bool is_means = (sampling == "tmeans");
                const int tmode = ycc422
                    ? (is_means ? CS_MODE_YCC422_TMEANS : CS_MODE_YCC422_TCLUSTER)
                    : (is_means ? CS_MODE_YCC420_TMEANS : CS_MODE_YCC420_TCLUSTER);
                encrypt_img.encrypt_ycc420_t(compression_ratio, password, sample_bits, chroma_bits,
                    split_mode, tmode);
            }
            else {
                // Adaptive LOD flags
                // (--adaptive/--two-pass/--regions/--lod-smooth/--lod-full) are
                // absorbed: they drive the luma budget while chroma stays uniform.
                const bool ycc_adaptive = adaptive || two_pass || !regions_json.empty() || lod_smooth > 0.0f || lod_full > 0 || spectral;
                if (ycc_adaptive) {
                    const int ts = tile_size > 0 ? tile_size : 64;
                    const int wbase = cs_adaptive_base_from_strength(adaptive_strength);
                    encrypt_img.encrypt_ycc420_adaptive(compression_ratio, password, ts, lod_min, wbase,
                        two_pass, pilot_ratio, regions_json, region_blend, lod_smooth, sample_bits, chroma_bits, lod_full, split_mode, spectral, lod_gamma, lod_cutoff);
                } else {
                    const int hr_mode = hr_samples.empty() ? split_mode : CS_MODE_YCC420_HRDETAIL;
                    encrypt_img.encrypt_ycc420(compression_ratio, password, sample_bits, chroma_bits, hr_mode, hr_samples);
                }
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
                regions_json, region_blend, lod_smooth, sample_bits, chroma_bits, lod_full, spectral, lod_gamma, lod_cutoff);
        }
        else if (sampling == "tcluster") {
            encrypt_img.encrypt_tcluster(compression_ratio, password, sample_bits, chroma_bits);
        }
        else if (sampling == "tmeans") {
            encrypt_img.encrypt_tmeans(compression_ratio, password, sample_bits, chroma_bits);
        }
        else if (tile_size > 0) {
            // periodic tile-based sampling: one random per-tile pattern,
            // replicated across the image; the mode travels in the header
            encrypt_img.encrypt_periodic(compression_ratio, password, tile_size, sample_bits, chroma_bits);
        } else {
            // standard random global sampling
            encrypt_img.encrypt(compression_ratio, password, sample_bits, chroma_bits, hr_samples);
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

        // under/over-sampled tile dump, on the same LOD grid the sampling
        // used (hf defaults to 32, adaptive paths to 64, periodic passes
        // its tile size through; exact-uniform grids warn and skip).
        if (!sample_split.empty()) {
            int grid = tile_size > 0 ? tile_size : 64;
            if (hf_focus && tile_size <= 0) grid = 32;
            encrypt_img.present_sample_classes(sample_split, grid);
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
