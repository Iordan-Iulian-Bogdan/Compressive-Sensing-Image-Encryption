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
namespace {

// smallest perfect square >= x (the encrypted container is square)
int next_perfect_square(int x) {
    int side = (int)std::ceil(std::sqrt((double)x));
    while ((long long)side * side < x) side++;
    return side * side;
}

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
    if (keep_key && cs_key_valid) {
        std::memcpy(salt, kept_salt, CS_SALT_BYTES);
        return;
    }
    if (!cs_random_bytes(salt, CS_SALT_BYTES)) {
        throw std::runtime_error("cryptographic RNG unavailable");
    }
    if (!cs_derive_key(password, salt, cs_key)) {
        throw std::runtime_error("key derivation (PBKDF2) failed");
    }
    std::memcpy(kept_salt, salt, CS_SALT_BYTES);
    cs_key_valid = true;
}

// Allocates the 1 x total container and writes its authenticated header.
// Metadata carries the budget m; returns the container byte buffer.
uint8_t* encrypt_image::alloc_container(int total, const uint8_t salt[CS_SALT_BYTES]) {
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width, odd_parity);
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
    if (!keep_key) release_key();   // reuse_key() holds it for the next container
    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

void encrypt_image::use_key(const uint8_t salt[CS_SALT_BYTES], const uint8_t key[32]) {
    std::memcpy(kept_salt, salt, CS_SALT_BYTES);
    std::memcpy(cs_key, key, 32);
    cs_key_valid = true;
    keep_key = true;
}

void encrypt_image::release_key() {
    keep_key = false;
    cs_key_valid = false;
    cs_wipe(cs_key, sizeof(cs_key));
    cs_wipe(kept_salt, sizeof(kept_salt));
}

void encrypt_image::get_mat(cv::Mat& dest) {
    encrypted_img.copyTo(dest);
}

cv::Mat encrypt_image::get_mat() {
    return encrypted_img;// .clone();
}

static bool cs_sampling_mask_view(const cv::Mat& input_img, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& out);
static void cs_present_mask_view(const cv::Mat& view, const std::string& output_path);

bool encrypt_image::build_sampling_mask_view(cv::Mat& out) const {
    return cs_sampling_mask_view(input_img, ri_x, ri_y, out);
}

void encrypt_image::sample_positions(std::vector<int>& rows, std::vector<int>& cols) const {
    rows = ri_x; cols = ri_y;
}

// source | binary mask | cyan overlay of the sample positions (rx = row, ry = col) on img.
static bool cs_sampling_mask_view(const cv::Mat& input_img, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& out) {
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
    cs_present_mask_view(view, output_path);
}

// --show-mask output: <output>.mask.png, then a window until a key (CS_MASK_NO_WINDOW: file only).
static void cs_present_mask_view(const cv::Mat& view, const std::string& output_path) {

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
    if (std::getenv("CS_MASK_NO_WINDOW")) return;   // scripted runs: file only, no blocking window

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

// --seg-auto: built-in scene segmentation and per-class rates -------------

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
        // one pass marks every tiny component (a full-map compare per component was O(comps x pixels))
        std::vector<uint8_t> is_tiny((size_t)ncomp, 0);
        for (int k = 1; k < ncomp; ++k) {
            if (stats.at<int>(k, cv::CC_STAT_AREA) < min_area) {
                is_tiny[(size_t)k] = 255;
                tiny_count += stats.at<int>(k, cv::CC_STAT_AREA);
            }
        }
        for (int y = 0; y < comp.rows; ++y) {
            const int32_t* cr = comp.ptr<int32_t>(y);
            uint8_t* tr = tiny.ptr<uint8_t>(y);
            for (int x = 0; x < comp.cols; ++x)
                if (is_tiny[(size_t)cr[x]]) tr[x] = 255;
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
        const cv::Mat den = (mII - mI.mul(mI)) + eps;   // varI + eps: the same for every class
        // Classes filter independently (in parallel); the argmax then runs in class order
        // so ties resolve exactly as a sequential loop would.
        std::vector<cv::Mat> qs((size_t)next_id);
        cv::parallel_for_(cv::Range(0, next_id), [&](const cv::Range& range) {
            for (int c = range.start; c < range.end; ++c) {
                cv::Mat ind = (clean == c);
                ind.convertTo(ind, CV_32F);
                cv::Mat up;
                cv::resize(ind, up, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
                cv::Mat Ip, mp, mIp;
                cv::multiply(guide, up, Ip);
                cv::boxFilter(up, mp, CV_32F, k);
                cv::boxFilter(Ip, mIp, CV_32F, k);
                cv::Mat covIp = mIp - mI.mul(mp);
                cv::Mat a = covIp / den;
                cv::Mat b = mp - a.mul(mI);
                cv::Mat ma, mb;
                cv::boxFilter(a, ma, CV_32F, k);
                cv::boxFilter(b, mb, CV_32F, k);
                qs[(size_t)c] = ma.mul(guide) + mb;
            }
        });
        cv::Mat best(rows, cols, CV_32F, cv::Scalar(-1.0f));
        cv::Mat out(rows, cols, CV_8U, cv::Scalar(0));
        for (int c = 0; c < next_id; ++c) {
            const cv::Mat& q = qs[(size_t)c];
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

void cs_seg_tile_refine(const cv::Mat& bgr, cv::Mat& ids, std::vector<float>& rates, int classes,
    float pixel_p) {
    constexpr int TILE = 64, LEVELS = 8;
    if (classes < 1 || classes * LEVELS > 256) {
        throw std::runtime_error("--seg-tile needs --seg-auto <= 32 classes");
    }
    if (bgr.empty() || ids.empty() || bgr.size() != ids.size() || (int)rates.size() < classes) {
        throw std::runtime_error("--seg-tile: geometry/rates mismatch (internal error)");
    }
    const int H = ids.rows, W = ids.cols;
    // |4-neighbour Laplacian| of the luma, clipped to 127 like the 8-bit
    // filter of the prototype, then the mean per (edge-clipped) tile.
    cv::Mat gray, lap;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    const cv::Mat k = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::filter2D(gray, lap, CV_32F, k, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
    const int th = (H + TILE - 1) / TILE, tw = (W + TILE - 1) / TILE;
    std::vector<float> tl((size_t)th * tw, 0.0f);
    for (int i = 0; i < th; ++i)
        for (int j = 0; j < tw; ++j) {
            const int y1 = (std::min)((i + 1) * TILE, H), x1 = (std::min)((j + 1) * TILE, W);
            double s = 0.0;
            for (int y = i * TILE; y < y1; ++y) {
                const float* l = lap.ptr<float>(y);
                for (int x = j * TILE; x < x1; ++x) s += (std::min)(std::abs(l[x]), 127.0f);
            }
            tl[(size_t)i * tw + j] = (float)(s / ((double)(y1 - i * TILE) * (x1 - j * TILE)));
        }
    const auto mm = std::minmax_element(tl.begin(), tl.end());
    const float lo = *mm.first, span = (std::max)(*mm.second - lo, 1e-6f);
    std::vector<int> lvl_t(tl.size());
    for (size_t t = 0; t < tl.size(); ++t) {
        const float lod = 32.0f + 223.0f * (tl[t] - lo) / span;
        lvl_t[t] = (std::min)((int)((lod - 32.0f) / 223.0f * LEVELS), LEVELS - 1);
    }
    double w_lvl[LEVELS];
    for (int l = 0; l < LEVELS; ++l) w_lvl[l] = 256.0 + 32.0 + (l + 0.5) * 223.0 / LEVELS;
    // Relabel, and per class the pixel-weighted mean level weight.
    std::vector<double> wsum((size_t)classes, 0.0);
    std::vector<long long> n((size_t)classes * LEVELS, 0);
    for (int y = 0; y < H; ++y) {
        uint8_t* id = ids.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const int c = id[x];
            if (c >= classes) continue;
            const int l = lvl_t[(size_t)(y / TILE) * tw + x / TILE];
            wsum[(size_t)c] += w_lvl[l];
            n[(size_t)c * LEVELS + l]++;
            id[x] = (uint8_t)(c * LEVELS + l);
        }
    }
    std::vector<float> r2((size_t)classes * LEVELS, 0.0f);
    for (int c = 0; c < classes; ++c) {
        long long nc = 0;
        for (int l = 0; l < LEVELS; ++l) nc += n[(size_t)c * LEVELS + l];
        if (nc == 0) continue;
        const double mean_w = wsum[(size_t)c] / (double)nc;
        for (int l = 0; l < LEVELS; ++l)
            r2[(size_t)c * LEVELS + l] = (float)(rates[(size_t)c] * w_lvl[l] / mean_w);
    }
    // Solve the scale s for the luma budget pixel_p x H x W (monotone in s).
    const double target = (double)pixel_p * H * W;
    auto taken = [&](double s) {
        double t = 0.0;
        for (size_t i = 0; i < r2.size(); ++i) t += (double)n[i] * (std::min)(1.0, s * r2[i]);
        return t;
    };
    double s_lo = 0.0, s_hi = 1.0;
    while (taken(s_hi) < target && s_hi < 1e6) s_hi *= 2.0;
    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (s_lo + s_hi);
        (taken(mid) < target ? s_lo : s_hi) = mid;
    }
    rates.assign(r2.size(), 0.0f);
    for (size_t i = 0; i < r2.size(); ++i) rates[i] = (float)(std::min)(1.0, s_hi * r2[i]);
    std::cout << "seg-tile: " << classes * LEVELS << " labels, scale " << s_hi << std::endl;
}

void encrypt_image::encrypt_ycc420_segmask_ids(const float& pixel_p, const std::string& password,
    const cv::Mat& ids, const std::vector<float>& rates,
    int sample_bits, int chroma_bits, const CsAnchorPlan* anchors, bool chroma_masked) {
    const int out_mode = anchors ? CS_MODE_YCC420_SEG_HRDETAIL : CS_MODE_YCC420_SEG;
    if (rates.empty() || rates.size() > 255) {
        throw std::runtime_error("seg-mask: need 1..255 per-class rates");
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

    // luma follows the segment budget; chroma is drawn on the 4:2:0 grid
    returnSegMaskIndices(ri_x, ri_y, rows, cols, ids, q, cs_key);
    const int mY_written = (int)ri_x.size();
    if (mY_written == 0) {
        throw std::runtime_error("seg-mask: rates select zero samples (raise a rate)");
    }
    m = mY_written;
    bm = (float)m / (float)((long long)rows * cols);
    // chroma dims/counts key off the 4:2:0 geometry (mode 3), not the seg-luma variants
    const int base = CS_MODE_YCC420;
    chroma_support = chroma_masked ? cs_seg_chroma_support(ids, q, base) : cv::Mat();
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
    if (chroma_masked) prefix.insert(prefix.end(), {0, 2, 0});   // seg prefix version 2: chroma inside the labels
    prefix.push_back((uint8_t)q.size());
    prefix.insert(prefix.end(), q.begin(), q.end());
    const uint32_t sl = (uint32_t)png.size();
    for (int b = 0; b < 4; ++b) prefix.push_back((uint8_t)((sl >> (b * 8)) & 0xff));
    const size_t pre_bytes = prefix.size() + png.size();

    const size_t meas_bytes = (size_t)mY_written + (size_t)2 * mC;
    const int meas_pixels = (int)((meas_bytes + 2) / 3);
    const int pre_pixels = (int)((pre_bytes + 2) / 3);
    // Compact anchor payload (positions drawn from the key, so it must be
    // built after key derivation; it is CTR-sealed once the IV exists).
    std::vector<uint8_t> anchor_payload;
    if (anchors) {
        anchor_payload = cs_anchor_payload(*anchors, cs_key);
        const uint32_t k = (uint32_t)anchor_payload[0] | ((uint32_t)anchor_payload[1] << 8) |
            ((uint32_t)anchor_payload[2] << 16) | ((uint32_t)anchor_payload[3] << 24);
        std::cout << "HR anchors: " << k << " (" << anchor_payload.size() << " payload bytes)" << std::endl;
    }
    const int hr_pixels = (int)((anchor_payload.size() + 2) / 3);
    const int total = next_perfect_square(meas_pixels + CS_HEADER_PIXELS + pre_pixels + 8 + hr_pixels);
    uint8_t* buf = alloc_container(total, salt);

    buf[CS_OFF_PAD] = (uint8_t)out_mode;
    std::memcpy(buf + CS_HEADER_BYTES, prefix.data(), prefix.size());
    std::memcpy(buf + CS_HEADER_BYTES + prefix.size(), png.data(), png.size());
    // v3: the label map is a content surrogate -> seal the whole prefix with a
    // domain-separated CTR stream; its byte length rides in the MAC-covered pad
    // so decrypt can find it without reading the (now opaque) prefix.
    for (int b = 0; b < 4; ++b)
        buf[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + b] = (uint8_t)((pre_bytes >> (b * 8)) & 0xff);
    if (!cs_blob_seal_xor(buf + CS_HEADER_BYTES, pre_bytes, cs_key, buf + CS_OFF_IV, CS_SEG_TAG))
        throw std::runtime_error("seg-mask: label prefix encryption failed");

    uint8_t* body = buf + CS_HEADER_BYTES + pre_bytes;
    cs_gather_u8(planes[0].data, planes[0].step, ri_x.data(), ri_y.data(), mY_written, body);
    cs_gather_u8(cr_small.data, cr_small.step, ri_cx1.data(), ri_cy1.data(), mC, body + mY_written);
    cs_gather_u8(cb_small.data, cb_small.step, ri_cx2.data(), ri_cy2.data(), mC, body + mY_written + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // Compact anchors ride as a trailing CTR-encrypted section after Cb; the
    // payload byte length sits in the pad after the mode byte (the seg
    // prefix length already holds CS_OFF_HR_COUNT's bytes).
    const size_t anchor_len = anchors ? anchor_payload.size() : 0;
    if (anchors) {
        const uint32_t len = (uint32_t)anchor_payload.size();
        for (int b = 0; b < 4; b++) {
            buf[CS_OFF_PAD + 1 + b] = (uint8_t)((len >> (b * 8)) & 0xff);
        }
        uint8_t* dst = body + mY_written + 2 * mC;
        std::memcpy(dst, anchor_payload.data(), anchor_payload.size());
        cs_wipe(anchor_payload.data(), anchor_payload.size());
        if (!cs_anchor_payload_xor(dst, len, cs_key, buf + CS_OFF_IV)) {
            throw std::runtime_error("anchor payload encryption failed");
        }
    }

    // --sample-bits: prefix + label PNG stay raw; the three planes pack
    // independently; the anchor payload follows them as an 8-bit section.
    const size_t y0 = (size_t)CS_HEADER_BYTES + pre_bytes;
    std::vector<cs_body_section> sections = {
        { y0, (size_t)mY_written, sample_bits, 0, false },
        { y0 + (size_t)mY_written, (size_t)mC, chroma_bits, 0, false },
        { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, chroma_bits, 0, false } };
    if (anchors) {
        sections.push_back({ y0 + (size_t)mY_written + (size_t)2 * mC, anchor_len, 8, 0, false });
    }
    finish_container(total, sample_bits, chroma_bits, sections);
    sampling_mode = out_mode;
}

void encrypt_image::encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g) {
    m = ri_x_g.size();
    int n = rows * cols;
    const int total = next_perfect_square(m + CS_HEADER_PIXELS + 8);
    encrypted_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width, odd_parity);
    cs_write_header_plain(encrypted_img.data, total * 3, text);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    cs_gather_bgr(input_img.data, input_img.step, ri_x_g.data(), ri_y_g.data(), m,
                  encrypted_img.data + (size_t)CS_HEADER_PIXELS * 3);

    encrypted_img = encrypted_img.reshape(0, (int)std::sqrt((double)encrypted_img.total()));
}

int encrypt_image::encrypt_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& password,
    float compression_ratio,
    int seg_auto,
    int hr_detail,
    int sample_bits,
    int chroma_bits,
    bool show_mask,
    const std::function<cv::Mat(const cv::Mat&)>& anchor_decode
){
    // sample_bits 0 = the production default: 6-bit luma / 8-bit chroma, sample ratio raised to keep the 8-bit byte
    // budget of --ratio
    const bool luma6 = (sample_bits == 0);
    if (luma6) { sample_bits = 6; chroma_bits = 8; }
    try {
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
        if (seg_auto < 2 || seg_auto > 32) {
            throw std::runtime_error("--seg-auto needs 2..32 classes (with --seg-tile)");
        }
        if (hr_detail < 1 || hr_detail > CS_HRDETAIL_MAX_SAMPLES) {
            throw std::runtime_error("--hr-detail must be in [1, " + std::to_string(CS_HRDETAIL_MAX_SAMPLES) + "]");
        }
        if (!anchor_decode) {
            throw std::runtime_error("the anchor pass needs a decoder (internal error)");
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

        // The 2x pipeline needs even sides: an odd side is padded by one
        // replicated row / column, recorded in the container, and cropped back
        // by decrypt, so the output keeps the native size and pixel grid.
        const int odd_parity = (input_img.rows % 2) | ((input_img.cols % 2) << 1);
        if (odd_parity != 0) {
            cv::copyMakeBorder(input_img, input_img, 0, input_img.rows % 2, 0, input_img.cols % 2,
                cv::BORDER_REPLICATE);
        }

        // The anchors are picked in a closed loop after sampling (compact mode-19 payload); keep the native image for
        // their exact luma. Reference geometry = decrypt output geometry (native, padded to even sides).
        const int out_w = input_img.cols / 2 * 2;
        const int out_h = input_img.rows / 2 * 2;
        const cv::Mat anchor_native = input_img;   // the 2x downscale below allocates a new buffer

        // The pipeline runs at half resolution (encrypt downsamples, decrypt 2x-upscales the solve).
        cv::resize(input_img, input_img, cv::Size(input_img.cols / 2, input_img.rows / 2));
        encrypt_image encrypt_img(input_img, true);
        encrypt_img.set_odd_parity(odd_parity);

        // Built-in segmentation: k-means ids at the sampling geometry plus detail-driven rates.
        if (luma6) {
            // Default 6-bit luma: same body bytes as 8-bit at --ratio, m6 (6/8 + 2k) = m8 (1 + 2k)
            // with k the chroma/luma count ratio, so ~1.2x the samples; capped at full sampling.
            int crows, ccols;
            cs_chroma_dims(CS_MODE_YCC420, input_img.rows, input_img.cols, crows, ccols);
            const double k = (double)crows * ccols / ((double)input_img.rows * input_img.cols);
            const float r8 = compression_ratio;
            compression_ratio = (float)std::min(1.0, r8 * (1.0 + 2.0 * k) / (0.75 + 2.0 * k));
            std::printf("6-bit luma: --ratio %.3f budget -> sample ratio %.3f\n", r8, compression_ratio);
        }
        cv::Mat auto_ids;
        cs_seg_auto_ids(input_img, seg_auto, input_img.rows, input_img.cols, auto_ids);
        std::vector<float> auto_rates = cs_seg_auto_rates(input_img, auto_ids, seg_auto);
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
            std::cout << " (auto rates)" << std::endl;
        }
        // Per-tile detail levels inside each class (--seg-tile); the rates come
        // back final (budget = --ratio x pixels), so encrypt at 1.0.
        auto_rates.resize((size_t)seg_auto, 0.0f);
        cs_seg_tile_refine(input_img, auto_ids, auto_rates, seg_auto, compression_ratio);
        encrypt_img.reuse_key();   // the anchored re-encrypt keeps this salt + key
        encrypt_img.encrypt_ycc420_segmask_ids(1.0f, password, auto_ids, auto_rates, sample_bits, chroma_bits);
        // Closed loop: decode this container, flag the worst 5% of its
        // 16x16 blocks, and re-encrypt with the same key (same samples)
        // plus anchors drawn there.
        cv::Mat dec = anchor_decode(encrypt_img.get_mat());
        if (dec.empty() || dec.type() != CV_8UC3 || dec.cols != out_w || dec.rows != out_h) {
            throw std::runtime_error("anchor pass: decode failed or has the wrong geometry");
        }
        cv::Mat ref = anchor_native;
        if (ref.size() != dec.size()) cv::resize(anchor_native, ref, dec.size(), 0, 0, cv::INTER_LINEAR);
        CsAnchorPlan plan;
        plan.flags = cs_error_anchor_flags(ref, dec, hr_detail);
        if (plan.flags.empty()) throw std::runtime_error("anchor pass: image smaller than one 16x16 block");
        cv::Mat ycc;
        cv::cvtColor(ref, ycc, cv::COLOR_BGR2YCrCb);
        cv::extractChannel(ycc, plan.luma, 0);
        plan.count = hr_detail;
        std::cout << "anchors: " << cv::countNonZero(plan.flags) << " of " << plan.flags.total()
                  << " blocks flagged by the closed-loop decode" << std::endl;
        encrypt_img.encrypt_ycc420_segmask_ids(1.0f, password, auto_ids, auto_rates, sample_bits, chroma_bits, &plan);
        encrypt_img.release_key();

        try {
            if (!cv::imwrite(output_path, encrypt_img.get_mat()))
                throw std::runtime_error("failed to write container image: " + output_path);
        } catch (const cv::Exception& e) {
            throw std::runtime_error(std::string("failed to write container image: ") + e.what());
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

// Multi-scale planner features, per 120 px tile and scale s = 2..6 (the layer as cs_encrypt_multiscale builds it:
// the photo area-shrunk by s/2, its LR = bilinear half): D = RGB MSE of the layer cubic-upscaled back to full size
// (the scale's floor; 0 at 2x), G = luma MSE between the layer and its LR bilinear-upscaled (detail cs-sr must
// recover), V = mean squared Laplacian of the LR luma (how hard the CS solve is). Tiles by pixel centre.
struct MsTileFeat { float D, G, V; };
static std::vector<MsTileFeat> ms_tile_features(const cv::Mat& src, int s, int T) {
    const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T;
    const double f = s / 2.0;
    const int lw = (int)(W / f) / 2 * 2, lh = (int)(H / f) / 2 * 2;
    cv::Mat layer, lr;
    if (s == 2) layer = src(cv::Rect(0, 0, lw, lh));
    else cv::resize(src, layer, cv::Size(lw, lh), 0, 0, cv::INTER_AREA);
    cv::resize(layer, lr, cv::Size(lw / 2, lh / 2));
    std::vector<double> sd((size_t)th * tw, 0.0), sg((size_t)th * tw, 0.0), sv((size_t)th * tw, 0.0);
    std::vector<double> nd((size_t)th * tw, 0.0), ng((size_t)th * tw, 0.0), nv((size_t)th * tw, 0.0);
    // Per-pixel squared error, summed per tile; rows grouped by tile row so threads never share a tile.
    const auto per_tile = [&](const cv::Mat& e, std::vector<double>& sum, std::vector<double>& cnt) {
        const int eh = e.rows, ew = e.cols;
        std::vector<int> tx((size_t)ew);
        for (int x = 0; x < ew; ++x) tx[(size_t)x] = (std::min)(tw - 1, (int)((x + 0.5) * W / ew) / T);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int r = 0; r < th; ++r) {
            for (int y = 0; y < eh; ++y) {
                if ((std::min)(th - 1, (int)((y + 0.5) * H / eh) / T) != r) continue;
                const float* p = e.ptr<float>(y);
                for (int x = 0; x < ew; ++x) { sum[(size_t)r * tw + tx[(size_t)x]] += p[x]; cnt[(size_t)r * tw + tx[(size_t)x]] += 1.0; }
            }
        }
    };
    if (s > 2) {
        cv::Mat up, d;
        cv::resize(layer, up, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);
        cv::absdiff(up, src, d);
        d.convertTo(d, CV_32FC3);
        d = d.mul(d);
        cv::Mat ch[3];
        cv::split(d, ch);
        cv::Mat e = (ch[0] + ch[1] + ch[2]) / 3.0f;
        per_tile(e, sd, nd);
    }
    cv::Mat ly, ry;
    cv::cvtColor(layer, ly, cv::COLOR_BGR2GRAY);
    cv::cvtColor(lr, ry, cv::COLOR_BGR2GRAY);
    ly.convertTo(ly, CV_32F);
    ry.convertTo(ry, CV_32F);
    cv::Mat rup, eg, lap;
    cv::resize(ry, rup, ly.size(), 0, 0, cv::INTER_LINEAR);
    eg = ly - rup;
    eg = eg.mul(eg);
    per_tile(eg, sg, ng);
    cv::Laplacian(ry, lap, CV_32F, 1, 1.0, 0.0, cv::BORDER_REPLICATE);
    lap = lap.mul(lap);
    per_tile(lap, sv, nv);
    std::vector<MsTileFeat> out((size_t)th * tw);
    for (size_t t = 0; t < out.size(); ++t)
        out[t] = { (float)(nd[t] > 0 ? sd[t] / nd[t] : 0.0), (float)(ng[t] > 0 ? sg[t] / ng[t] : 0.0), (float)(nv[t] > 0 ? sv[t] / nv[t] : 0.0) };
    return out;
}

// A multi-scale layer's luma rate per LR pixel (lrw x lrh), x 255: its tile's where the tile is scale s, else the rate
// of a scale-s tile within kMsMargin px (full res; 8 neighbours), else 0. The margin lets the decoder's 17 px feather
// blend two sampled layers (IMG_0620 r0.3: -0.86 -> -0.20 dB vs production at equal bytes; 16 px fixes it fully but
// costs the other photos 0.15-0.4 dB).
constexpr int kMsMargin = 8;
static cv::Mat ms_layer_rates(const cv::Mat& map, const cv::Mat& qmap, int s, int W, int H, int T, int lrw, int lrh) {
    const int th = map.rows, tw = map.cols;
    cv::Mat q8(lrh, lrw, CV_8U);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < lrh; ++y) {
        const int ty = (std::min)(th - 1, (int)((y + 0.5) * H / lrh) / T);
        uint8_t* qr = q8.ptr<uint8_t>(y);
        for (int x = 0; x < lrw; ++x) {
            const int tx = (std::min)(tw - 1, (int)((x + 0.5) * W / lrw) / T);
            int q = map.at<uint8_t>(ty, tx) == s ? qmap.at<uint8_t>(ty, tx) : 0;
            if (q == 0) {
                const double X = (x + 0.5) * W / lrw, Y = (y + 0.5) * H / lrh;
                for (int k = 0; k < 8 && q == 0; ++k) {
                    static const int ox[8] = { -1, 1, 0, 0, -1, 1, -1, 1 }, oy[8] = { 0, 0, -1, 1, -1, -1, 1, 1 };
                    const int nx = (int)(X + ox[k] * kMsMargin), ny = (int)(Y + oy[k] * kMsMargin);
                    if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                    if (map.at<uint8_t>(ny / T, nx / T) == s) q = qmap.at<uint8_t>(ny / T, nx / T);
                }
            }
            qr[x] = (uint8_t)q;
        }
    }
    return q8;
}

// Per-tile error model of the planner, per scale 2x..6x: MSE = D + exp(a . [1, lD, lG, lV, lG^2, lV^2, lG lV]
// + lq min(b . [1, lD, lG, lV], -0.05)), lX = log(1 + X) of ms_tile_features clamped to the training maximum, lq =
// log(rate): the scale's floor D plus a reconstruction error that falls with the rate. Fitted on 16 dataset photos
// (none of the evaluation set), each scale as a whole-photo layer at random rates held over 6x6-tile blocks (as a
// plan's regions; per-tile random rates let starved tiles borrow from their neighbours), production decode; held-out
// log error RMSE 0.56 (2x) / 0.40-0.43 (3x-6x).
struct MsErrModel { double a[7], b[4], lmax[3]; };
static const MsErrModel kMsModel[5] = {
    { { -1.04656, 0, 0.732277, 0.505001, -0.0323643, -0.0571419, 0.0693871 }, { 0.252447, 0, 0.316446, -0.277803 }, { 0, 7.05747, 10.1007 } },
    { { -0.685989, 0.156419, 0.989368, 0.11643, 0.110375, 0.0369118, -0.142217 }, { 0.427264, 0.00145417, 0.494178, -0.417393 }, { 6.18446, 7.0095, 9.75632 } },
    { { -0.565965, 0.173049, 0.975387, 0.0586176, 0.0448857, 0.0146763, -0.0620851 }, { 0.502667, -0.0198731, 0.560642, -0.471172 }, { 6.75201, 7.30921, 9.88583 } },
    { { -0.915025, 0.360086, 0.466349, 0.337929, 0.0130645, -0.0166716, 0.00610654 }, { 0.331764, 0.102952, 0.354456, -0.37899 }, { 7.21561, 6.98675, 9.93173 } },
    { { -1.10041, 0.325245, 0.567029, 0.312329, 0.0504057, -0.000978317, -0.0446536 }, { 0.159896, 0.113276, 0.305696, -0.337293 }, { 7.46319, 6.9594, 10.1227 } }
};

// `--multiscale auto`: per tile the (scale, rate) of least predicted error for the budget of a production container
// at --ratio (6-bit luma: 1.2 x ratio of the LR pixels, capped at 1, plus the anchors), by a Lagrangian over all
// tiles. Costs in luma samples: one carries 0.75 byte + its chroma share 0.5, an anchor ~1 byte, so an anchor = 0.8.
// The tiles' own samples leave out the margins; the plan is redone with the budget scaled by planned / counted
// samples (twice), which lands the container within a few % of production's bytes.
static void ms_auto_plan(const cv::Mat& src, int T, float ratio, int hr_detail, cv::Mat& map, cv::Mat& qmap) {
    const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T, nt = th * tw;
    constexpr int NQ = 20;
    double qs[NQ];
    for (int i = 0; i < NQ; ++i) qs[i] = std::exp(std::log(0.02) * (1.0 - i / (NQ - 1.0)));
    constexpr double kAnchor = 0.8;
    const double anchor_cost = hr_detail > 0 ? kAnchor * hr_detail / nt : 0.0;   // per 2x tile
    std::vector<double> E((size_t)nt * 5 * NQ), C((size_t)nt * 5 * NQ);   // error, cost
    for (int si = 0; si < 5; ++si) {
        const int s = si + 2;
        const std::vector<MsTileFeat> ft = ms_tile_features(src, s, T);
        const double f = s / 2.0;
        const int lrw = (int)(W / f) / 2 * 2 / 2, lrh = (int)(H / f) / 2 * 2 / 2;
        std::vector<double> rows_in((size_t)th, 0.0), cols_in((size_t)tw, 0.0);   // LR pixels per tile row / column
        for (int y = 0; y < lrh; ++y) rows_in[(size_t)(std::min)(th - 1, (int)((y + 0.5) * H / lrh) / T)] += 1.0;
        for (int x = 0; x < lrw; ++x) cols_in[(size_t)(std::min)(tw - 1, (int)((x + 0.5) * W / lrw) / T)] += 1.0;
        for (int t = 0; t < nt; ++t) {
            const int ty = t / tw, tx = t % tw;
            const double npx = rows_in[(size_t)ty] * cols_in[(size_t)tx];
            const double full = (double)(std::min)(T, H - ty * T) * (std::min)(T, W - tx * T);
            const MsErrModel& m = kMsModel[si];
            const double lD = (std::min)(m.lmax[0], std::log1p((double)ft[(size_t)t].D)), lG = (std::min)(m.lmax[1], std::log1p((double)ft[(size_t)t].G)),
                         lV = (std::min)(m.lmax[2], std::log1p((double)ft[(size_t)t].V));
            const double base = m.a[0] + m.a[1] * lD + m.a[2] * lG + m.a[3] * lV + m.a[4] * lG * lG + m.a[5] * lV * lV + m.a[6] * lG * lV;
            const double slope = (std::min)(-0.05, m.b[0] + m.b[1] * lD + m.b[2] * lG + m.b[3] * lV);
            for (int i = 0; i < NQ; ++i) {
                const size_t o = ((size_t)t * 5 + si) * NQ + i;
                E[o] = full * (ft[(size_t)t].D + std::exp(base + slope * std::log(qs[i])));
                C[o] = qs[i] * npx + (s == 2 ? anchor_cost : 0.0);
            }
        }
    }
    const double budget = (std::min)(1.0, 1.2 * ratio) * (double)(W / 2) * (H / 2) + kAnchor * hr_detail;
    std::vector<int> pick((size_t)nt);
    const auto choose = [&](double lam) {
        double cost = 0.0;
        for (int t = 0; t < nt; ++t) {
            double best = 1e300; int bo = 0;
            for (int o = 0; o < 5 * NQ; ++o) {
                const size_t k = (size_t)t * 5 * NQ + o;
                const double v = E[k] + lam * C[k];
                if (v < best) { best = v; bo = o; }
            }
            pick[(size_t)t] = bo;
            cost += C[(size_t)t * 5 * NQ + bo];
        }
        return cost;
    };
    const auto fit_budget = [&](double target) {   // lambda for the budget
        double lo = 1e-6, hi = 1e6;   // cost falls as lambda grows
        for (int it = 0; it < 60; ++it) {
            const double mid = std::sqrt(lo * hi);
            if (choose(mid) > target) lo = mid; else hi = mid;
        }
        return choose(hi);
    };
    map.create(th, tw, CV_8U);
    qmap.create(th, tw, CV_8U);
    int count[7] = {};
    double target = budget, cost = 0.0, counted = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        cost = fit_budget(target);
        std::fill(count, count + 7, 0);
        for (int t = 0; t < nt; ++t) {
            const int si = pick[(size_t)t] / NQ, i = pick[(size_t)t] % NQ;
            map.at<uint8_t>(t / tw, t % tw) = (uint8_t)(si + 2);
            qmap.at<uint8_t>(t / tw, t % tw) = (uint8_t)(std::min)(254.0, (std::max)(1.0, std::round(qs[i] * 255.0)));
            ++count[si + 2];
        }
        counted = kAnchor * hr_detail * count[2] / nt;   // the samples the layers will draw, margins included
        for (int s = 2; s <= 6; ++s) {
            if (!count[s]) continue;
            const double f = s / 2.0;
            const cv::Mat q8 = ms_layer_rates(map, qmap, s, W, H, T, (int)(W / f) / 2 * 2 / 2, (int)(H / f) / 2 * 2 / 2);
            counted += cv::sum(q8)[0] / 255.0;
        }
        if (std::abs(counted / budget - 1.0) < 0.01) break;
        target *= budget / counted;
    }
    std::printf("multi-scale plan: budget %.0f luma samples, planned %.0f, with margins %.0f; tiles 2x %d, 3x %d, 4x %d, 5x %d, 6x %d\n",
        budget, cost, counted, count[2], count[3], count[4], count[5], count[6]);
}

int cs_encrypt_multiscale(const std::string& input_path, const std::string& output_path, const std::string& password,
    int hr_detail, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode, float ratio, bool show_mask) {
    constexpr int T = 120;
    try {
        const cv::Mat src = cv::imread(input_path, cv::IMREAD_COLOR);
        if (src.empty()) throw std::runtime_error("cannot read '" + input_path + "'");
        const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T;
        if (W > 65535 || H > 65535) throw std::runtime_error("multi-scale: image wider or taller than 65535");
        if (const char* fd = std::getenv("CS_MS_DUMP_FEATURES")) {   // planner data: tile,scale,D,G,V
            FILE* f = std::fopen(fd, "w");
            if (!f) throw std::runtime_error(std::string("cannot write '") + fd + "'");
            for (int sc = 2; sc <= 6; ++sc) {
                const std::vector<MsTileFeat> ft = ms_tile_features(src, sc, T);
                for (size_t t = 0; t < ft.size(); ++t) std::fprintf(f, "%zu,%d,%.5g,%.5g,%.5g\n", t, sc, ft[t].D, ft[t].G, ft[t].V);
            }
            std::fclose(f);
        }
        cv::Mat map(th, tw, CV_8U), qmap(th, tw, CV_8U);
        ms_auto_plan(src, T, ratio, hr_detail, map, qmap);
        if (const char* dp = std::getenv("CS_MS_DUMP_PLAN")) {   // the plan as a PNG (B = scale, G = rate)
            cv::Mat ch[3] = { map, qmap, cv::Mat::zeros(th, tw, CV_8U) }, png;
            cv::merge(ch, 3, png);
            cv::imwrite(dp, png);
        }
        // One seg-mask layer per scale in use: the photo pre-shrunk by s/2 (area), then the pipeline's 2x
        // (bilinear, as encrypt_image_tiled); labels = the tile's quantized rate where the tile is this
        // scale's, rate 0 elsewhere; chroma drawn inside that support (seg prefix version 2).
        // One PBKDF2 for the container: every layer is sealed with its salt + key (each header has its own IV).
        uint8_t salt[CS_SALT_BYTES], key[32];
        if (!cs_random_bytes(salt, CS_SALT_BYTES) || !cs_derive_key(password, salt, key))
            throw std::runtime_error("multi-scale: key derivation failed");
        std::vector<uint8_t> scales;
        std::vector<cv::Mat> layers;
        const int gw = W / 2 * 2 / 2, gh = H / 2 * 2 / 2;   // --show-mask: the 2x layer's sampling grid
        std::vector<int> mask_r, mask_c;
        for (int s = 2; s <= 6; ++s) {
            if (cv::countNonZero(map == s) == 0) continue;
            const double f = s / 2.0;
            const int lw = (int)(W / f) / 2 * 2, lh = (int)(H / f) / 2 * 2;
            cv::Mat layer;
            if (s == 2) layer = src(cv::Rect(0, 0, lw, lh));
            else cv::resize(src, layer, cv::Size(lw, lh), 0, 0, cv::INTER_AREA);
            cv::Mat lr;
            cv::resize(layer, lr, cv::Size(lw / 2, lh / 2));
            cv::Mat ids(lr.rows, lr.cols, CV_8U);
            int label_of[256];
            std::fill(label_of, label_of + 256, -1);
            std::vector<float> rates = { 0.0f };
            label_of[0] = 0;
            const cv::Mat q8 = ms_layer_rates(map, qmap, s, W, H, T, lr.cols, lr.rows);
            for (int y = 0; y < lr.rows; ++y) {
                const uint8_t* qr = q8.ptr<uint8_t>(y);
                uint8_t* id = ids.ptr<uint8_t>(y);
                for (int x = 0; x < lr.cols; ++x) {
                    const int q = qr[x];
                    if (label_of[q] < 0) { label_of[q] = (int)rates.size(); rates.push_back(q / 255.0f); }
                    id[x] = (uint8_t)label_of[q];
                }
            }
            encrypt_image e(lr, true);
            int chroma_bits = 8;
            // The 2x layer: production's closed loop (decode, flag the worst 16x16 blocks, re-encrypt with the same
            // key plus anchors there), the flags limited to its own tiles (the others decode flat).
            const int n2 = s == 2 ? cv::countNonZero(map == 2) : 0;
            const bool anchored = n2 > 0 && hr_detail > 0 && anchor_decode;
            e.use_key(salt, key);
            e.encrypt_ycc420_segmask_ids(1.0f, password, ids, rates, 6, chroma_bits, nullptr, true);
            if (anchored) {
                const cv::Mat dec = anchor_decode(e.get_mat());
                if (dec.empty() || dec.type() != CV_8UC3 || dec.cols != lw || dec.rows != lh)
                    throw std::runtime_error("multi-scale: anchor pass decode failed or has the wrong geometry");
                cv::Mat ref = layer.clone();
                for (int y = 0; y < lh; ++y)
                    for (int x = 0; x < lw; ++x)
                        if (map.at<uint8_t>(y / T, x / T) != 2) ref.at<cv::Vec3b>(y, x) = dec.at<cv::Vec3b>(y, x);
                CsAnchorPlan ap;
                ap.count = (int)std::lround((double)hr_detail * n2 / ((double)th * tw));
                ap.flags = cs_error_anchor_flags(ref, dec, ap.count);
                cv::Mat ycc;
                cv::cvtColor(layer, ycc, cv::COLOR_BGR2YCrCb);
                cv::extractChannel(ycc, ap.luma, 0);
                std::cout << "multi-scale: 2x layer anchors " << ap.count << ", " << cv::countNonZero(ap.flags)
                          << " blocks flagged" << std::endl;
                e.encrypt_ycc420_segmask_ids(1.0f, password, ids, rates, 6, chroma_bits, &ap, true);
            }
            e.release_key();
            if (show_mask) {   // this layer's LR pixel (r, c) covers s x s photo pixels: its centre on the grid
                std::vector<int> rr, cc;
                e.sample_positions(rr, cc);
                for (size_t k = 0; k < rr.size(); ++k) {
                    mask_r.push_back((std::min)(gh - 1, (int)((rr[k] + 0.5) * s / 2.0)));
                    mask_c.push_back((std::min)(gw - 1, (int)((cc[k] + 0.5) * s / 2.0)));
                }
            }
            cv::Mat c = e.get_mat();
            if (!c.isContinuous()) c = c.clone();
            std::cout << "multi-scale: " << s << "x layer " << lw << "x" << lh << ", " << cv::countNonZero(map == s)
                      << " tiles, " << rates.size() << " labels, " << c.total() * 3 << " bytes" << std::endl;
            scales.push_back((uint8_t)s);
            layers.push_back(c);
        }
        // Descriptor + layers in one outer container.
        std::vector<uint8_t> png;
        cv::imencode(".png", map, png, { cv::IMWRITE_PNG_COMPRESSION, 9 });
        std::vector<uint8_t> desc;
        const auto put = [&](uint32_t v, int n) { for (int b = 0; b < n; ++b) desc.push_back((uint8_t)(v >> (8 * b))); };
        put(1, 1); put(W, 2); put(H, 2); put(T, 2); put((uint32_t)layers.size(), 1);
        for (size_t i = 0; i < layers.size(); ++i) { put(scales[i], 1); put((uint32_t)(layers[i].total() * 3), 4); }
        put((uint32_t)png.size(), 4);
        desc.insert(desc.end(), png.begin(), png.end());
        size_t body = desc.size();
        for (const cv::Mat& c : layers) body += c.total() * 3;
        const int total = next_perfect_square(CS_HEADER_PIXELS + (int)((body + 2) / 3) + 8);
        cv::Mat out(1, total, CV_8UC3, cv::Scalar(0, 0, 0));
        uint8_t* buf = out.data;
        if (!cs_write_header(buf, (size_t)total * 3, cs_build_metadata((int)layers.size(), H, W, H, W), key, salt))
            throw std::runtime_error("multi-scale: header write failed");
        buf[CS_OFF_PAD] = CS_MODE_MULTISCALE;
        const size_t desc_len = desc.size();
        for (int b = 0; b < 4; ++b) buf[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + b] = (uint8_t)(desc_len >> (8 * b));
        std::memcpy(buf + CS_HEADER_BYTES, desc.data(), desc_len);
        cs_wipe(desc.data(), desc_len);
        if (!cs_blob_seal_xor(buf + CS_HEADER_BYTES, desc_len, key, buf + CS_OFF_IV, CS_MS_TAG))
            throw std::runtime_error("multi-scale: descriptor encryption failed");
        size_t off = CS_HEADER_BYTES + desc_len;
        for (const cv::Mat& c : layers) { std::memcpy(buf + off, c.data, c.total() * 3); off += c.total() * 3; }
        const bool sealed = cs_seal_header(buf, (size_t)total * 3, key);
        cs_wipe(key, sizeof(key));
        if (!sealed) throw std::runtime_error("multi-scale: sealing failed");
        const int side = (int)std::lround(std::sqrt((double)total));
        out = out.reshape(3, side);
        if (!cv::imwrite(output_path, out)) throw std::runtime_error("failed to write '" + output_path + "'");
        std::cout << "multi-scale: " << layers.size() << " layers, container " << (size_t)total * 3 << " bytes" << std::endl;
        if (show_mask) {   // on the photo shrunk as the 2x layer is (crop to even, then the pipeline's bilinear 2x)
            cv::Mat grid, view;
            cv::resize(src(cv::Rect(0, 0, gw * 2, gh * 2)), grid, cv::Size(gw, gh));
            if (cs_sampling_mask_view(grid, mask_r, mask_c, view)) cs_present_mask_view(view, output_path);
            else std::cerr << "Warning: --show-mask: no sampling indices available\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
