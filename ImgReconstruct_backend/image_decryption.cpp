#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include "image_preview.hpp"
#include "cs_gpu.h"
#include "cs_dict.hpp"
#include "avir.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <utility>
#include <algorithm>
#include <thread>
#include <vector>
#include <omp.h>

namespace {
// Opt-in phase profiler for decrypt_image_tiled: set CS_PROFILE=1 to print
// one "profile[...]" line per call on stderr with phase=milliseconds since
// the previous mark. When off, overhead is a handful of clock reads.
bool cs_profile_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CS_PROFILE");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

struct PhaseLog {
    bool on;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last;
    std::vector<std::pair<const char*, double>> phases;

    PhaseLog() : on(cs_profile_enabled()),
        start(std::chrono::steady_clock::now()), last(start) {}

    void mark(const char* name) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        phases.emplace_back(name, std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
    }

    void dump(const char* tag) const {
        if (!on) return;
        const double total = std::chrono::duration<double, std::milli>(last - start).count();
        std::fprintf(stderr, "profile[%s]:", tag);
        for (const auto& p : phases) std::fprintf(stderr, " %s=%.1f", p.first, p.second);
        std::fprintf(stderr, " total=%.1f\n", total);
    }
};
} // namespace

decrypt_image::decrypt_image(const std::string input_path, const std::string& password) {
    parse_container_header(cv::imread(input_path, cv::IMREAD_COLOR), password);
}

decrypt_image::decrypt_image(cv::Mat input, const std::string& password) {
    parse_container_header(input, password);
}

void decrypt_image::parse_container_header(const cv::Mat& input, const std::string& password) {
    if (input.empty()) {
        throw std::runtime_error("failed to load encrypted image");
    }

    input.copyTo(encrypted_img);
    encrypted_img = encrypted_img.reshape(0, encrypted_img.total());

    CsHeaderInfo info;
    if (!cs_parse_header(encrypted_img.data, encrypted_img.total() * 3, password, info)) {
        if (info.legacy) {
            throw std::runtime_error("legacy container format: this image was encrypted with an older version, please re-encrypt");
        }
        if (info.auth_failed) {
            throw std::runtime_error("authentication failed: wrong password or corrupted/tampered image");
        }
        throw std::runtime_error("failed to parse encrypted image header");
    }

    m = info.m;
    rows = info.rows;
    cols = info.cols;
    org_size.height = info.org_h;
    org_size.width = info.org_w;

    // sampling mode travels in the authenticated header: periodic and
    // adaptive modes restore automatically, no decrypt-side flags needed
    sampling_mode = info.sampling_mode;
    periodic_tile = info.periodic_tile;
    periodic_samples = info.periodic_samples;
    adaptive_base = info.adaptive_base;
    lod_full_threshold = info.full_threshold;
    sample_bits = info.sample_bits;
    sample_bits_chroma = info.sample_bits_chroma;

    if (info.key_valid) {
        std::memcpy(cs_key, info.key, sizeof(cs_key));
        cs_key_valid = true;
        cs_wipe(info.key, sizeof(info.key));
    }
}

// Restores the raw measurement layout from a bit-packed (--sample-bits)
// container: each section unpacks independently into a fresh raw-size Mat
// that replaces the packed container, so every downstream reader runs
// unchanged. Sections are (raw byte offset, value count) in ascending
// order. No-op at 8 bits.
void decrypt_image::unpack_container_measurements(
    const std::vector<cs_body_section>& sections) {
    if ((sample_bits == 8 && sample_bits_chroma == 8) || sections.empty()) {
        return;
    }
    const size_t packed_bytes = encrypted_img.total() * encrypted_img.elemSize();
    const size_t raw_len = cs_unpacked_body_bytes(sections);
    const size_t raw_pixels = (raw_len + 2) / 3;
    cv::Mat raw(1, (int)raw_pixels, CV_8UC3, cv::Scalar(0, 0, 0));
    cs_unpack_container_body(encrypted_img.data, packed_bytes, sections, raw.data);
    encrypted_img = raw.reshape(0, (int)raw_pixels);
}

void decrypt_image::decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, float tv, int solver, int fista_iters, int reweights, int basis, float wscale) {

    // FISTA solvers (BGR only): reweighted-L1 per channel, or
    // SOMP-structured joint group-L2,1 over all three channels. Same
    // warm-start refs and the same IDCT+255 tail convention throughout,
    // so the merge/convert below applies unchanged.
    int num_iterations_offset = (num_iterations / 2);

    if (solver == CS_SOLVER_FISTA) {
        // The three channels are independent solves: each warm-starts from
        // the neighbor field built before decrypt(). Submit all three first,
        // then finish them -- the GPU worker sees 3x the jobs per wave and
        // its pipelined kernels overlap instead of draining between one-
        // channel passes (the old copy_next_ref chain serialized them and
        // clobbered the neighbor warm starts of ch1/ch2).
        fista_channel_task ct[3];
        for (int ch = 0; ch < 3; ++ch)
            fista_channel_begin(ct[ch], encrypted_img, ch, coef, rows, cols,
                ri_x_g, ri_y_g,
                ch == 0 ? num_iterations : num_iterations - num_iterations_offset,
                ref[ch], tv, reweights, fista_iters, basis, wscale);
        for (int ch = 0; ch < 3; ++ch)
            fista_channel_end(ct[ch], ref[ch]);
        cv::merge(ref, 3, out);
        out.convertTo(out, CV_8UC3);
        return;
    }
    if (solver == CS_SOLVER_FISTA_JOINT) {
        reconstruct_image_fista_joint(encrypted_img, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref, tv, reweights, fista_iters, basis, wscale);
        cv::merge(ref, 3, out);
        out.convertTo(out, CV_8UC3);
        return;
    }

    throw std::runtime_error("unknown solver id (only fista and joint remain)");
}

void decrypt_image::get_mat(cv::Mat& dest) {
    decrypted_img.copyTo(dest);
}

cv::Mat decrypt_image::get_mat() {
    return decrypted_img.clone();
}

void decrypt_image::get_sampled_mat(cv::Mat& sampled_mat, cv::Mat& masked_mat) {
    if (!cs_key_valid) {
        throw std::runtime_error("derived key not available: header must be parsed with the password first");
    }

    sampled_mat = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    masked_mat = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    // measurement pixel origin inside the container: adaptive mode inserts a
    // lod byte region (padded to whole pixels) between header and body
    int meas0 = CS_HEADER_PIXELS;
    std::vector<uint8_t> lod;

    if (sampling_mode == CS_MODE_ADAPTIVE) {
        if (periodic_tile <= 0) {
            throw std::runtime_error("adaptive container: missing tile size in header");
        }
        const int lod_bytes = cs_lod_bytes(rows, cols, periodic_tile);
        const int lod_pixels = cs_lod_pixels(rows, cols, periodic_tile);
        meas0 = CS_HEADER_PIXELS + lod_pixels;
        const size_t need = (size_t)CS_HEADER_BYTES + (size_t)lod_bytes;
        if (encrypted_img.total() * 3 < need) {
            throw std::runtime_error("adaptive container: body too short for lod region");
        }
        lod.assign(encrypted_img.data + CS_HEADER_BYTES, encrypted_img.data + CS_HEADER_BYTES + lod_bytes);
        if ((int)lod.size() != periodic_samples) {
            // header pad sanity field disagrees with geometry — reject
            throw std::runtime_error("adaptive container: lod length mismatch");
        }
        returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, periodic_tile, adaptive_base, lod_full_threshold);
    }
    else if (sampling_mode == CS_MODE_PERIODIC && periodic_tile > 0 && periodic_samples > 0) {
        // the container was encrypted with periodic tile sampling: regenerate
        // the identical per-tile pattern from the key + header geometry
        returnPeriodicIndices(ri_x, ri_y, rows, cols, periodic_samples, cs_key, periodic_tile);
    }
    else if (sampling_mode == CS_MODE_HF_FOCUS) {
        if (periodic_tile <= 0 || periodic_samples != cs_lod_bytes(rows, cols, periodic_tile))
            throw std::runtime_error("hf-focus container has invalid tile/LOD geometry");
        const int lod_pixels = cs_lod_pixels(rows, cols, periodic_tile);
        const size_t thumbnail_offset = CS_HEADER_BYTES + static_cast<size_t>(lod_pixels) * 3;
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        const uint32_t thumbnail_length =
            static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN]) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1]) << 8) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2]) << 16) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3]) << 24);
        if (thumbnail_length == 0 || thumbnail_length > (1u << 22) ||
            thumbnail_offset + thumbnail_length > total_bytes)
            throw std::runtime_error("hf-focus container has invalid thumbnail length");
        lod.assign(encrypted_img.data + CS_HEADER_BYTES,
                   encrypted_img.data + CS_HEADER_BYTES + periodic_samples);
        std::vector<uint8_t> png(encrypted_img.data + thumbnail_offset,
                                 encrypted_img.data + thumbnail_offset + thumbnail_length);
        uint8_t counter[CS_IV_BYTES];
        std::memcpy(counter, encrypted_img.data + CS_OFF_IV, CS_IV_BYTES);
        counter[0] ^= 0x54;
        if (!cs_aes256_ctr_xor(png.data(), png.size(), cs_key, counter)) {
            cs_wipe(counter, sizeof(counter));
            throw std::runtime_error("hf-focus thumbnail decryption failed");
        }
        cs_wipe(counter, sizeof(counter));
        const cv::Mat thumbnail = cv::imdecode(png, cv::IMREAD_COLOR);
        cs_wipe(png.data(), png.size());
        if (thumbnail.empty()) throw std::runtime_error("hf-focus thumbnail decode failed");
        std::vector<float> weights;
        int weight_cols = 0, weight_rows = 0;
        cs_hf_thumb_weights(thumbnail, weights, weight_cols, weight_rows);
        returnHfWeightedIndices(ri_x, ri_y, rows, cols, lod, m, cs_key,
            periodic_tile, adaptive_base, weights, weight_cols, weight_rows);
        thumb_seed = cv::Mat();
        cv::resize(thumbnail, thumb_seed, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
        meas0 = CS_HEADER_PIXELS + lod_pixels +
                static_cast<int>((static_cast<size_t>(thumbnail_length) + 2) / 3);
    }
    else {
        // returnRandomIndices writes into [0, m) without resizing
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);
    }

    // scatter measurements back to their sampled pixel positions; chunked by
    // 32 for cache locality, last chunk may be partial when m % 32 != 0
    // Bit-packed (--sample-bits) bodies unpack first: one flat section of
    // 3 values per measurement at the pixel offset meas0 (lod/thumbnail raw
    // prefixes already accounted in meas0 above).
    const int m_count = (int)ri_x.size();
    if (sample_bits != 8 || sample_bits_chroma != 8) {
        unpack_container_measurements({ { (size_t)meas0 * 3,
            (size_t)3 * (size_t)m_count, sample_bits, sample_bits_chroma, true } });
    }
    const int enc_limit = (int)encrypted_img.total();
    for (int base = 0; base < m_count; base += 32) {
        const int chunk = (std::min)(32, m_count - base);
        const int enc = meas0 + base;
        if (enc + chunk > enc_limit) break;
        for (int j = 0; j < chunk; ++j) {
            const int k = base + j;
            sampled_mat.at<cv::Vec3b>(ri_x[k], ri_y[k]) = encrypted_img.at<cv::Vec3b>(enc + j);
            masked_mat.at<cv::Vec3b>(ri_x[k], ri_y[k]) = cv::Vec3b(1, 1, 1);
        }
    }
}

void decrypt_image::get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
    cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask) {
    if (!cs_key_valid) {
        throw std::runtime_error("derived key not available: header must be parsed with the password first");
    }
    if (!cs_is_ycc(sampling_mode)) {
        throw std::runtime_error("ycc unpack requested on a non-YCC container");
    }

    // LOD-luma variant: lod bytes sit between header and measurements (same
    // padded layout as mode 2); the luma draw regenerates from them while
    // chroma stays uniform. Pure-uniform mode 3 has no lod region.
    size_t body_off = 0;
    int mY_written = m;
    if (cs_is_ycc_hf(sampling_mode)) {
        if (periodic_tile <= 0) throw std::runtime_error("ycc-hf container missing HF tile size");
        const uint32_t thumb_len =
            static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN]) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1]) << 8) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2]) << 16) |
            (static_cast<uint32_t>(encrypted_img.data[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3]) << 24);
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        if (thumb_len == 0 || thumb_len > (1u << 22) ||
            static_cast<size_t>(CS_HEADER_BYTES) + thumb_len > total_bytes)
            throw std::runtime_error("ycc-hf container has invalid thumbnail length");
        std::vector<uint8_t> png(encrypted_img.data + CS_HEADER_BYTES,
                                 encrypted_img.data + CS_HEADER_BYTES + thumb_len);
        uint8_t counter[CS_IV_BYTES];
        std::memcpy(counter, encrypted_img.data + CS_OFF_IV, CS_IV_BYTES);
        counter[0] ^= 0x54;
        if (!cs_aes256_ctr_xor(png.data(), png.size(), cs_key, counter)) {
            cs_wipe(counter, sizeof(counter));
            throw std::runtime_error("ycc-hf thumbnail decryption failed");
        }
        cs_wipe(counter, sizeof(counter));
        const cv::Mat thumbnail = cv::imdecode(png, cv::IMREAD_COLOR);
        cs_wipe(png.data(), png.size());
        if (thumbnail.empty()) throw std::runtime_error("ycc-hf thumbnail decode failed");
        std::vector<float> weights;
        int weight_cols = 0, weight_rows = 0;
        cs_hf_thumb_weights(thumbnail, weights, weight_cols, weight_rows);
        const int tile_cols = (cols + periodic_tile - 1) / periodic_tile;
        const int tile_rows = (rows + periodic_tile - 1) / periodic_tile;
        const std::vector<uint8_t> flat_lod(static_cast<size_t>(tile_cols) * tile_rows, 0);
        returnHfWeightedIndices(ri_x, ri_y, rows, cols, flat_lod, m, cs_key,
            periodic_tile, adaptive_base, weights, weight_cols, weight_rows);
        mY_written = static_cast<int>(ri_x.size());
        body_off = (static_cast<size_t>(thumb_len) + 2) / 3 * 3;
        cv::resize(thumbnail, thumb_seed, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
    }
    else if (periodic_tile > 0) {
        const int lod_bytes = cs_lod_bytes(rows, cols, periodic_tile);
        const int lod_pixels = cs_lod_pixels(rows, cols, periodic_tile);
        const size_t need_lod = (size_t)CS_HEADER_BYTES + (size_t)lod_bytes;
        if (encrypted_img.total() * 3 < need_lod) {
            throw std::runtime_error("ycc container: body too short for lod region");
        }
        std::vector<uint8_t> lod(encrypted_img.data + CS_HEADER_BYTES,
            encrypted_img.data + CS_HEADER_BYTES + lod_bytes);
        if ((int)lod.size() != periodic_samples) {
            throw std::runtime_error("ycc container: lod length mismatch");
        }
        returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, periodic_tile, adaptive_base, lod_full_threshold);
        mY_written = (int)ri_x.size();
        body_off = (size_t)lod_pixels * 3;
    } else {
        // uniform luma draw from the base key
        ri_x.resize((size_t)m);
        ri_y.resize((size_t)m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);
    }
    // chroma draws are uniform in every variant (tagged keys); dims and
    // counts follow the container mode (4:2:0 halves both axes, 4:2:2
    // halves the width only)
    returnYcc420ChromaIndices(rows, cols, m, cs_key, sampling_mode);
    const int mC = m_chroma;

    int crows, ccols;
    cs_chroma_dims(sampling_mode, rows, cols, crows, ccols);

    // Bit-packed (--sample-bits) bodies unpack first: three independent
    // sections (luma, Cr, Cb) after the raw thumbnail/lod prefix. The
    // length check below then runs against the restored raw geometry.
    if (sample_bits != 8 || sample_bits_chroma != 8) {
        const size_t y0 = (size_t)CS_HEADER_BYTES + body_off;
        unpack_container_measurements({
            { y0, (size_t)mY_written, sample_bits, 0, false },
            { y0 + (size_t)mY_written, (size_t)mC, sample_bits_chroma, 0, false },
            { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, sample_bits_chroma, 0, false } });
    }

    const size_t need = (size_t)CS_HEADER_BYTES + body_off + (size_t)mY_written + (size_t)2 * mC;
    if (encrypted_img.total() * 3 < need) {
        throw std::runtime_error("ycc container: body too short for luma/chroma payload");
    }
    const uint8_t* body = encrypted_img.data + CS_HEADER_BYTES + body_off;

    y = cv::Mat::zeros(rows, cols, CV_8U);
    y_mask = cv::Mat::zeros(rows, cols, CV_8U);
    cr = cv::Mat::zeros(crows, ccols, CV_8U);
    cr_mask = cv::Mat::zeros(crows, ccols, CV_8U);
    cb = cv::Mat::zeros(crows, ccols, CV_8U);
    cb_mask = cv::Mat::zeros(crows, ccols, CV_8U);

    for (int k = 0; k < mY_written; k++) {
        y.at<uint8_t>(ri_x[k], ri_y[k]) = body[k];
        y_mask.at<uint8_t>(ri_x[k], ri_y[k]) = 1;
    }
    for (int k = 0; k < mC; k++) {
        cr.at<uint8_t>(ri_cx1[k], ri_cy1[k]) = body[mY_written + k];
        cr_mask.at<uint8_t>(ri_cx1[k], ri_cy1[k]) = 1;
        cb.at<uint8_t>(ri_cx2[k], ri_cy2[k]) = body[mY_written + mC + k];
        cb_mask.at<uint8_t>(ri_cx2[k], ri_cy2[k]) = 1;
    }
}

float decrypt_image::get_compression_ratio() {
    return float(m) / (rows * cols);
}
cv::Size decrypt_image::get_org_size() {
    return org_size;
}

// Builds the initial solver solution for one tile from its already-solved
// west/north neighbors: their overlap strips contain actual reconstructed
// image content for this tile's border region, a much stronger prior than
// the generic reference. Pixel-domain copy first, then the solver-domain
// transform (DCT or multilevel CDF 9/7, chosen by basis) with the /10
// convention createRefSolutions uses, so the solvers accept it unchanged.
// ycrcb/chroma_sub serve the ycc420 pipeline only (BGR passes false):
// strips and fallback refs are produced in the YCrCb domain and chroma
// warm-start planes are downsampled to half resolution to match the
// half-res chroma solves.
static void build_neighbor_warm_start(cv::Mat refs[3],
    const std::vector<cv::Mat>& generic_refs,
    const std::vector<std::vector<cv::Mat>>& solved,
    const std::vector<std::vector<TileCoord>>& coordinates,
    int i, int j, bool ycrcb, bool chroma_sub, int basis,
    const cv::Mat& thumbnail_seed = cv::Mat(), bool chroma_sub_v = true)
{
    // pixel plane -> solver-domain coefficients (/10 convention)
    auto to_basis = [&](cv::Mat& plane) {
        if (basis == CS_BASIS_CDF97) {
            cs_dwt_forward((float*)plane.data, plane.rows, plane.cols,
                cs_dwt_levels(plane.rows, plane.cols));
        } else {
            cv::dct(plane, plane, 0);
        }
        plane /= 10.0f;
    };
    // generic DCT/10 ref -> solver domain (linearity preserves the /10 scale:
    // IDCT yields pix/10, whose forward transform is exactly coeff(pix)/10)
    auto generic_to_basis = [&](cv::Mat m) {
        if (basis == CS_BASIS_CDF97) {
            cv::dct(m, m, cv::DCT_INVERSE);
            cs_dwt_forward((float*)m.data, m.rows, m.cols,
                cs_dwt_levels(m.rows, m.cols));
        }
        return m;
    };
    const cv::Rect t_rect(coordinates[i][j].x, coordinates[i][j].y, solved[i][j].cols, solved[i][j].rows);
    cv::Mat pix(t_rect.height, t_rect.width, CV_32FC3, cv::Scalar(0, 0, 0)); // pixel domain, [0,1]
    bool any_neighbor = false;

    const int west[2] = { i, j - 1 };
    const int north[2] = { i - 1, j };
    for (int n = 0; n < 2; ++n) {
        const int ni = west[0] + (north[0] - west[0]) * n; // (i, j-1) then (i-1, j)
        const int nj = west[1] + (north[1] - west[1]) * n;
        if (ni < 0 || nj < 0 || solved[ni][nj].empty()) continue;
        const cv::Rect n_rect(coordinates[ni][nj].x, coordinates[ni][nj].y, solved[ni][nj].cols, solved[ni][nj].rows);
        const cv::Rect inter = n_rect & t_rect;
        if (inter.width <= 0 || inter.height <= 0) continue;

        const cv::Rect dst_local(inter.x - t_rect.x, inter.y - t_rect.y, inter.width, inter.height);
        const cv::Rect src_local(inter.x - n_rect.x, inter.y - n_rect.y, inter.width, inter.height);
        cv::Mat strip = solved[ni][nj](src_local).clone();
        if (ycrcb) {
            // the solved tiles hold BGR pixels; re-encode to YCrCb so the
            // warm start lives in the same domain as the solve
            cv::cvtColor(strip, strip, cv::COLOR_BGR2YCrCb);
        }
        cv::Mat strip32;
        strip.convertTo(strip32, CV_32FC3, 1.0 / 255.0);
        strip32.copyTo(pix(dst_local));
        any_neighbor = true;
    }

    if (!any_neighbor) {
        if (ycrcb && !thumbnail_seed.empty() && thumbnail_seed.type() == CV_8UC3 &&
            t_rect.x >= 0 && t_rect.y >= 0 &&
            t_rect.x + t_rect.width <= thumbnail_seed.cols &&
            t_rect.y + t_rect.height <= thumbnail_seed.rows) {
            cv::Mat thumbnail_ycc;
            cv::cvtColor(thumbnail_seed, thumbnail_ycc, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> thumbnail_planes;
            cv::split(thumbnail_ycc, thumbnail_planes);
            cv::Mat y_crop;
            thumbnail_planes[0](t_rect).convertTo(y_crop, CV_32F, 1.0 / 255.0);
            to_basis(y_crop);
            refs[0] = y_crop;
            // 4:2:2 halves the width only (chroma_sub_v == false keeps full
            // height); 4:2:0 halves both axes.
            const cv::Rect chroma_rect(t_rect.x / 2, chroma_sub_v ? t_rect.y / 2 : t_rect.y,
                (t_rect.x + t_rect.width + 1) / 2 - t_rect.x / 2,
                chroma_sub_v ? (t_rect.y + t_rect.height + 1) / 2 - t_rect.y / 2 : t_rect.height);
            for (int ch = 1; ch < 3; ++ch) {
                cv::Mat coarse, seed;
                cv::resize(thumbnail_planes[ch], coarse,
                    cv::Size((thumbnail_seed.cols + 1) / 2,
                        chroma_sub_v ? (thumbnail_seed.rows + 1) / 2 : thumbnail_seed.rows),
                    0, 0, cv::INTER_AREA);
                coarse(chroma_rect).convertTo(seed, CV_32F, 1.0 / 255.0);
                to_basis(seed);
                refs[ch] = seed;
            }
            return;
        }
        if (!thumbnail_seed.empty() && thumbnail_seed.type() == CV_8UC3 &&
            t_rect.x >= 0 && t_rect.y >= 0 &&
            t_rect.x + t_rect.width <= thumbnail_seed.cols &&
            t_rect.y + t_rect.height <= thumbnail_seed.rows) {
            cv::Mat crop;
            thumbnail_seed(t_rect).convertTo(crop, CV_32FC3, 1.0 / 255.0);
            std::vector<cv::Mat> planes;
            cv::split(crop, planes);
            for (int ch = 0; ch < 3; ++ch) {
                to_basis(planes[ch]);
                refs[ch] = planes[ch];
            }
            return;
        }
        // first wave (or failed neighbors): generic reference, resized to
        // this tile's grid so merge/FISTA always see the split tile's shape
        // (tile_size is only the last processed tile and can disagree with
        // edge/clamped tiles — that size mismatch left the top tile row
        // short and produced black gaps in the composite)
        cv::Size want(t_rect.width, t_rect.height);
        auto fit = [&](cv::Mat m) {
            if (m.empty() || m.size() != want) {
                cv::Mat r;
                if (m.empty()) {
                    r = cv::Mat(want, CV_32F, cv::Scalar(0));
                    cv::dct(r, r, 0);
                    r /= 10.0f;
                } else {
                    cv::resize(m, r, want, 0, 0, cv::INTER_AREA);
                }
                return r;
            }
            return m.clone();
        };
        if (ycrcb) {
            // Y gets the generic image-like ref; chroma planes start from a
            // neutral flat-chroma solution (their statistics are very different)
            refs[0] = generic_to_basis(fit(generic_refs[0]));
            // 4:2:0 halves both axes; 4:2:2 halves the width only
            // (chroma_sub_v == false keeps full height).
            cv::Size chroma_size = chroma_sub
                ? cv::Size((want.width + 1) / 2, chroma_sub_v ? (want.height + 1) / 2 : want.height)
                : want;
            refs[1] = cv::Mat(chroma_size, CV_32F, cv::Scalar(0.5f));
            to_basis(refs[1]);
            refs[2] = refs[1].clone();
        }
        else {
            for (int ch = 0; ch < 3; ++ch) {
                refs[ch] = generic_to_basis(fit(generic_refs[ch]));
            }
        }
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(pix, planes);
    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat plane = planes[ch];
        if (chroma_sub && ch > 0) {
            // match the coarse chroma solves' unknown space (4:2:0 halves
            // both axes, 4:2:2 halves the width only)
            cv::resize(plane, plane, cv::Size((plane.cols + 1) / 2,
                chroma_sub_v ? (plane.rows + 1) / 2 : plane.rows), 0, 0, cv::INTER_AREA);
        }
        to_basis(plane);
        refs[ch] = plane;
    }
}


// decrypts tiles in wavefront (anti-diagonal) order
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, float tv, int solver, int fista_iters, int reweights, int basis, float wscale, const cv::Mat& thumbnail_seed, bool per_tile_coef, bool per_tile_tv, std::vector<std::vector<cv::Mat>>* hr_out, bool fuse_upscale, CsUpscalePipeline* pipe,     std::mutex* hr_grid_mutex, bool superres_2x, float sr_red, const cs_coupled_dict* sr_dict, const std::string& sr_red_denoiser, const std::string& sr_dncnn_model, const std::string& sr_fsrcnn_model) {

    // solve-stage profiler: one reset here, one dump line at exit (env-gated)
    cs_solveprof_reset();
    const bool sp = g_solveprof.on;
    auto sp_wall0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    // GPU mode: solver threads block on their channel solve, so threads =
    // in-flight problems = batch size. A wave holds num_tiles tiles; each
    // tile has one outstanding job at a time, so num_tiles threads already
    // fill the batch (pend caps there) -- extra threads just contend with
    // the CPU-side blend/PNG stages on this machine.
    if (cs_gpu::enabled()) {
        int want = num_tiles;
        if (want > 96) want = 96;
        if (want < num_threads) want = num_threads;
        if (want > num_threads) {
            std::cerr << "[gpu] raising solve threads " << num_threads << " -> "
                      << want << " (batching)" << std::endl;
            num_threads = want;
        }
    }

    // per-tile coef (--per-tile-coef): mean sample count over non-empty
    // tiles, fixed before the wavefront (indices are all known upfront)
    double coef_mean_m = 0.0;
    if (per_tile_coef) {
        long long sum = 0;
        int cnt = 0;
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                const int mm = (int)indices[i][j].ri_x_g.size();
                if (mm > 0) { sum += mm; ++cnt; }
            }
        if (cnt > 0) coef_mean_m = (double)sum / cnt;
    }

    // we use a reference image as the initial solution for tiles without
    // neighbors (first wave); this helps speed up convergence
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);

    // wavefront over the tile grid: wave w = i + j. Tiles on the same
    // anti-diagonal share no west or north neighbors, so every tile of a
    // wave runs in parallel while warm-starting from tiles of earlier
    // waves only -- no ordering races, and roughly 2N-1 waves instead of
    // num_tiles^2 independent solves.
    for (int w = 0; w <= 2 * (num_tiles - 1); ++w) {
        const int i_lo = (std::max)(0, w - (num_tiles - 1));
        const int i_hi = (std::min)(num_tiles - 1, w);
        const int wave_count = i_hi - i_lo + 1;
        auto sp_w0 = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int t = 0; t < wave_count; ++t) {
            try {
                const int i = i_lo + t;
                const int j = w - i;

                auto sp_a = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                cv::Mat x0[3];
                build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j, false, false, basis, thumbnail_seed);
                auto sp_b = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                if (sp) g_solveprof.warm_ns += cs_solveprof_ns_since(sp_a);

                decrypt_image dimgs = decrypt_image(mats_in[i][j]);
                auto sp_c = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                if (sp) g_solveprof.ctor_ns += cs_solveprof_ns_since(sp_b);

                // starved tiles solve under a stronger prior, rich tiles relaxed
                // (tv follows the same law under its own flag; base tv 0 stays 0)
                const float tile_coef = per_tile_coef
                    ? cs_per_tile_coef(coef, (int)indices[i][j].ri_x_g.size(), coef_mean_m)
                    : coef;
                const float tile_tv = per_tile_tv
                    ? cs_per_tile_coef(tv, (int)indices[i][j].ri_x_g.size(), coef_mean_m)
                    : tv;
                dimgs.decrypt(x0, indices[i][j].ri_x_g, indices[i][j].ri_y_g, iterations, tile_coef, mats_out[i][j], tile_tv, solver, fista_iters, reweights, basis, wscale);
                if (sp) { g_solveprof.call_ns += cs_solveprof_ns_since(sp_c); g_solveprof.tiles += 1; }
                // streaming pipeline (subprocess backends): hand the solved
                // tile to the background upscaler now; it writes the 2x
                // result into the separate hr grid while solving continues.
                // (Checked first: pipe and fuse_upscale are mutually
                // exclusive by construction at the call sites.)
                if (pipe && !mats_out[i][j].empty()) {
                    pipe->enqueue(i, j);
                }
                // fused 2x upscale: this tile is final the moment its solve
                // finishes (same op as the post-join batch, so bit-identical
                // for AVIR). With the cs-sr backend the tile instead gets a
                // CS super-resolution refinement: the HR DCT coefficients are
                // re-solved against the tile's original LR samples
                // (warm-started from AVIR); an empty SR result falls back to
                // AVIR so no tile is ever lost.
                // A separate grid keeps the live preview's LR reads safe.
                else if (fuse_upscale && hr_out && !mats_out[i][j].empty()) {
                    cv::Mat hr;
                    if (sr_dict) {
                        // Coupled-dict SR refines the solved tile (no
                        // measurements needed: patch-OMP synthesis +
                        // back-projection operate post-solve).
                        cs_sr_upscale_dict_2x(mats_out[i][j], sr_dict, hr);
                    }
                    else if (!sr_fsrcnn_model.empty()) {
                        // FSRCNN neural upscale (in-process CPU session per
                        // worker inside the helper); misses fall through to
                        // AVIR below so no tile is ever lost.
                        cs_fsrcnn_upscale_2x(mats_out[i][j], hr, sr_fsrcnn_model);
                    }
                    else if (superres_2x) {
                        cs_sr_upscale_tile_2x(mats_out[i][j], mats_in[i][j],
                            indices[i][j].ri_x_g, indices[i][j].ri_y_g,
                            tile_coef, tile_tv, iterations, reweights, fista_iters,
                            basis, wscale, hr, sr_red, sr_red_denoiser, sr_dncnn_model);
                    }
                    if (hr.empty())
                        cs_upscale_2x_avir(mats_out[i][j], hr);
                    // Cell header replacement races the preview's
                    // reconstructImage: serialize it under the grid mutex.
                    if (!hr.empty() && hr_grid_mutex) {
                        std::lock_guard<std::mutex> lk(*hr_grid_mutex);
                        hr.copyTo((*hr_out)[i][j]);
                    }
                    else if (!hr.empty()) {
                        hr.copyTo((*hr_out)[i][j]);
                    }
                }
            }
            catch (const std::exception& e) {
                // an exception escaping an OpenMP region terminates the process;
                // log it and degrade this one tile gracefully instead
                #pragma omp critical
                {
                    std::cerr << "Tile decryption failed: " << e.what() << std::endl;
                }
            }
        }
        if (sp) {
            // wall time of this wave and the thread capacity it could use;
            // sum(work)/sum(capacity) over the solve = wavefront efficiency
            const long long ww = cs_solveprof_ns_since(sp_w0);
            g_solveprof.wave_wall_ns += ww;
            g_solveprof.wave_cap_ns += ww * (long long)((std::min)(wave_count, num_threads));
        }
    }
    if (sp) cs_solveprof_dump(num_tiles, num_threads, cs_solveprof_ns_since(sp_wall0));
}

// YCC 4:2:0 tile solver: wavefront (anti-diagonal) order mirroring
// decrypt_tiles, but each tile solves luma at full resolution and each chroma
// plane natively on its coarse grid with per-channel FISTA. Warm starts
// come from the already-solved BGR neighbors via build_neighbor_warm_start
// (ycrcb + chroma_sub), whose ref layout (full-res Y, coarse chroma) matches
// this solve exactly. Solved planes merge YCrCb -> BGR per tile.
static void decrypt_tiles_ycc420(int num_threads,
    const std::vector<std::vector<cv::Mat>>& y_meas,
    const std::vector<std::vector<indices>>& y_idx,
    const std::vector<std::vector<cv::Mat>>& cr_meas,
    const std::vector<std::vector<indices>>& cr_idx,
    const std::vector<std::vector<cv::Mat>>& cb_meas,
    const std::vector<std::vector<indices>>& cb_idx,
    std::vector<std::vector<cv::Mat>>& mats_out,
    const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, float tv,
    int solver, int fista_iters, int reweights, int basis, float wscale,
    const cv::Mat& thumbnail_seed, bool per_tile_coef = false, bool per_tile_tv = false,
    std::vector<std::vector<cv::Mat>>* hr_out = nullptr, bool fuse_upscale = false,
    CsUpscalePipeline* pipe = nullptr, std::mutex* hr_grid_mutex = nullptr, bool chroma_sub_v = true, const cs_coupled_dict* sr_dict = nullptr,
    const std::string& sr_fsrcnn_model = "") {

    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);
    const int iters_off = iterations / 2;
    const int iters_c = iterations - iters_off;
    const float coef_c = coef * 0.5f;
    // joint group-sparsity needs one shared RGB grid; the per-plane calls
    // below are per-channel FISTA either way, so no dispatch is needed.

    // GPU mode: same batching requirement as decrypt_tiles -- solver threads
    // block on each plane solve, so threads = in-flight problems = batch
    // size. Without this the default 8 threads cap every flush at 8 jobs
    // (measured: 1731 micro-flushes of P<=8 on the 64x64 grid) and the
    // per-flush pack/H2D/launch tax outweighs the GPU solve. A wave holds
    // num_tiles tiles, so num_tiles threads fill the batch.
    if (cs_gpu::enabled()) {
        int want = num_tiles;
        if (want > 96) want = 96;
        if (want < num_threads) want = num_threads;
        if (want > num_threads) {
            std::cerr << "[gpu] raising solve threads " << num_threads << " -> "
                      << want << " (batching)" << std::endl;
            num_threads = want;
        }
    }

    // per-tile coef (--per-tile-coef): mean luma sample count over non-empty
    // tiles; only the luma draw varies per tile (chroma stays uniform)
    double coef_mean_my = 0.0;
    if (per_tile_coef) {
        long long sum = 0;
        int cnt = 0;
        for (int ii = 0; ii < num_tiles; ++ii)
            for (int jj = 0; jj < num_tiles; ++jj) {
                const int mm = (int)y_idx[ii][jj].ri_x_g.size();
                if (mm > 0) { sum += mm; ++cnt; }
            }
        if (cnt > 0) coef_mean_my = (double)sum / cnt;
    }

    for (int w = 0; w <= 2 * (num_tiles - 1); ++w) {
        const int i_lo = (std::max)(0, w - (num_tiles - 1));
        const int i_hi = (std::min)(num_tiles - 1, w);
        const int wave_count = i_hi - i_lo + 1;

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int t = 0; t < wave_count; ++t) {
            try {
                const int i = i_lo + t;
                const int j = w - i;
                if (y_meas[i][j].empty()) continue;

                cv::Mat x0[3];
                build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j, true, true, basis, thumbnail_seed, chroma_sub_v);

                const int Th = mats_out[i][j].rows;
                const int Tw = mats_out[i][j].cols;
                // coarse tile from the same halving the prep used: exact cover
                // (4:2:2 keeps full height, halves the width only)
                const int xo = coordinates[i][j].x, yo = coordinates[i][j].y;
                const int cTw = (xo + Tw + 1) / 2 - xo / 2;
                const int cTh = chroma_sub_v ? (yo + Th + 1) / 2 - yo / 2 : Th;
                const cv::Size coarse_want(cTw, cTh);
                // neighbor-derived chroma refs are halved t_rect; parity on
                // edge tiles can leave them 1px off the solved grid — refit
                // (warm start only, the solve re-converges regardless)
                for (int c = 1; c < 3; ++c) {
                    if (x0[c].empty() || x0[c].size() != coarse_want) {
                        if (x0[c].empty()) {
                            x0[c] = cv::Mat(coarse_want, CV_32F, cv::Scalar(0.5f));
                            if (basis == CS_BASIS_CDF97) {
                                cs_dwt_forward((float*)x0[c].data, cTh, cTw,
                                    cs_dwt_levels(cTh, cTw));
                            } else {
                                cv::dct(x0[c], x0[c], 0);
                            }
                            x0[c] /= 10.0f;
                        } else {
                            cv::resize(x0[c], x0[c], coarse_want, 0, 0, cv::INTER_LINEAR);
                        }
                    }
                }

                cv::Mat dummy;
                // luma-only per-tile scaling (chroma draws stay uniform, so
                // coef_c is untouched; tv follows under its own flag)
                const float tile_coef_y = per_tile_coef
                    ? cs_per_tile_coef(coef, (int)y_idx[i][j].ri_x_g.size(), coef_mean_my)
                    : coef;
                const float tile_tv_y = per_tile_tv
                    ? cs_per_tile_coef(tv, (int)y_idx[i][j].ri_x_g.size(), coef_mean_my)
                    : tv;
                // FISTA-only (joint degrades to per-channel FISTA at the
                // dispatcher): luma at full tile resolution, chroma natively
                // on the half-resolution grid, where proximal updates are
                // exact at the near-flat warm starts.
                auto solve_plane = [&](const cv::Mat& meas, float c, int R, int C,
                    const std::vector<int>& rx, const std::vector<int>& ry,
                    int it, cv::Mat& r, bool nxt, cv::Mat& nr, float t) {
                    reconstruct_color_channel_fista(meas, 0, c, R, C, rx, ry, it, r, nxt, nr, t, reweights, fista_iters, basis, wscale);
                };
                solve_plane(y_meas[i][j], tile_coef_y, Th, Tw,
                    y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g, iterations, x0[0],
                    false, dummy, tile_tv_y);
                solve_plane(cr_meas[i][j], coef_c, cTh, cTw,
                    cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, iters_c, x0[1],
                    true, x0[2], 0.0f);
                solve_plane(cb_meas[i][j], coef_c, cTh, cTw,
                    cb_idx[i][j].ri_x_g, cb_idx[i][j].ri_y_g, iters_c, x0[2],
                    false, dummy, 0.0f);

                cv::Mat cr_full, cb_full;
                cv::resize(x0[1], cr_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                cv::resize(x0[2], cb_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                cv::Mat chs[3] = { x0[0], cr_full, cb_full }; // (Y, Cr, Cb)
                cv::Mat ycc_tile;
                cv::merge(chs, 3, ycc_tile);
                ycc_tile.convertTo(ycc_tile, CV_8UC3);
                cv::cvtColor(ycc_tile, mats_out[i][j], cv::COLOR_YCrCb2BGR);
                // streaming pipeline: same handoff as decrypt_tiles above
                if (pipe && !mats_out[i][j].empty()) {
                    pipe->enqueue(i, j);
                }
                // fused AVIR upscale: same op as the post-join batch, so
                // bit-identical; separate grid keeps preview reads safe.
                // A coupled dict refines the merged BGR tile (post-solve, so
                // split-grid sampling needs no special operator here).
                else if (fuse_upscale && hr_out && !mats_out[i][j].empty()) {
                    cv::Mat hr;
                    if (sr_dict) {
                        cs_sr_upscale_dict_2x(mats_out[i][j], sr_dict, hr);
                    }
                    else if (!sr_fsrcnn_model.empty()) {
                        cs_fsrcnn_upscale_2x(mats_out[i][j], hr, sr_fsrcnn_model);
                    }
                    if (hr.empty())
                        cs_upscale_2x_avir(mats_out[i][j], hr);
                    // Serialize the cell write under the grid mutex (see
                    // the BGR path above).
                    if (!hr.empty() && hr_grid_mutex) {
                        std::lock_guard<std::mutex> lk(*hr_grid_mutex);
                        hr.copyTo((*hr_out)[i][j]);
                    }
                    else if (!hr.empty()) {
                        hr.copyTo((*hr_out)[i][j]);
                    }
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Tile decryption failed: " << e.what() << std::endl;
                }
            }
        }
    }
}

int decrypt_image::decrypt_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& password,
    int num_tiles,
    int overlap,
    int iterations,
    int nun_threads,
    float coef,
    bool show_preview,
    float tv,
    bool full_res,
    int solver,
    int fista_iters,
    int reweights,
    int basis,
    float wscale,
    const CsPhotoUpscalerOptions& photo_up,
    bool per_tile_coef,
    bool per_tile_tv
) {
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    return decrypt_image_tiled(encrypted_img_g, output_path, password, num_tiles, overlap, iterations, nun_threads, coef, show_preview, tv, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv);
}

int decrypt_image::decrypt_image_tiled(
    const cv::Mat& encrypted_img_in,
    const std::string& output_path,
    const std::string& password,
    int num_tiles,
    int overlap,
    int iterations,
    int nun_threads,
    float coef,
    bool show_preview,
    float tv,
    bool full_res,
    int solver,
    int fista_iters,
    int reweights,
    int basis,
    float wscale,
    const CsPhotoUpscalerOptions& photo_up,
    bool per_tile_coef,
    bool per_tile_tv
) {
    PhaseLog prof;
    if (solver != CS_SOLVER_FISTA && solver != CS_SOLVER_FISTA_JOINT) {
        std::cerr << "Error: unknown solver id " << solver << " (1=fista, 2=joint)" << std::endl;
        return -1;
    }
    if (basis != CS_BASIS_DCT && basis != CS_BASIS_CDF97) {
        std::cerr << "Error: unknown basis id " << basis << " (0=dct, 1=wavelet)" << std::endl;
        return -1;
    }
    if (fista_iters < 0 || fista_iters > 500) {
        std::cerr << "Error: --fista-iters must be in [0, 500] (0 = auto-map from --iterations)" << std::endl;
        return -1;
    }
    if (reweights < 1 || reweights > 5) {
        std::cerr << "Error: --reweights must be in [1, 5]" << std::endl;
        return -1;
    }
        if (wscale <= 0.0f || wscale > 16.0f) {
            std::cerr << "Error: --wscale must be in (0, 16]" << std::endl;
            return -1;
        }
        const cv::Mat& encrypted_img_g = encrypted_img_in;
    try {
        if (CSencryption::params == MANUAL_PARAM) {
            if (overlap < 0 || overlap > 96) {
                throw std::runtime_error("Overlap is outside the acceptable range of [0, 96]");
            }

            if (num_tiles < 1) {
                throw std::runtime_error("Number of tiles must be at least 1");
            }

            if (coef <= 0.0f) {
                throw std::runtime_error("Coef must be positive");
            }

            if (nun_threads < 1) {
                throw std::runtime_error("Number of threads is less than 1");
            }

            if (CSencryption::params != AUTO_PARAM && CSencryption::params != MANUAL_PARAM) {
                throw std::runtime_error("Parameters type is incorrect");
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }    

    // header parse + PBKDF2 key derivation + MAC verification happen here;
    // a wrong password or a tampered container is rejected before any solve
    decrypt_image dimgs;
    try {
        dimgs = decrypt_image(encrypted_img_g, password);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
    cv::Size org_size = dimgs.get_org_size();
    prof.mark("kdf");

    if (CSencryption::params == AUTO_PARAM) {
        // iterations: the solver saturates by ~5 steps across compression
        // ratios 0.25..1.0 (tests/bench_iterations.cpp: the PSNR plateau
        // starts at 5 everywhere, time scales linearly beyond it), so the
        // old 10*(1/ratio) formula (capped 30) overshot by 2-6x. Keep a
        // knee-proportional allocation with a cap of 8.
        iterations = (int)std::ceil(5.0f / dimgs.get_compression_ratio());
        if (iterations > 8) {
            iterations = 8;
        }

        if (dimgs.get_compression_ratio() < 0.5f) {
            coef = 0.045f;
        }
        else
        {
            coef = (1.0f / dimgs.get_compression_ratio()) * 0.01875f;
        }
        num_tiles = 1;
        overlap = 24;
        nun_threads = omp_get_max_threads();
    }

    // mode-3 (YCC 4:2:0) containers take a dedicated tile pipeline: luma at
    // full resolution, chroma natively on the coarse grid. The mode travels
    // in the header so no decrypt-side flags are needed.
    if (dimgs.get_sampling_mode() == CS_MODE_YCC420 ||
        dimgs.get_sampling_mode() == CS_MODE_YCC422 ||
        dimgs.get_sampling_mode() == CS_MODE_YCC420_HF ||
        dimgs.get_sampling_mode() == CS_MODE_YCC422_HF) {
        return decrypt_image_tiled_ycc420(dimgs, output_path, num_tiles, overlap,
            iterations, nun_threads, coef, show_preview, tv,
            full_res, solver, fista_iters, reweights, basis, wscale,
            photo_up, per_tile_coef, per_tile_tv);
    }

    //int N_reconfigured = tiles;
    std::string windowName = output_path.empty() ? "decrypted tiles" : output_path;

    try {
        std::vector<std::vector<cv::Mat>> encrypted_image_tiles;
        std::vector<std::vector<cv::Mat>> decrypted_image_tiles;
        decrypted_image_tiles.resize(num_tiles, std::vector<cv::Mat>(num_tiles));
        std::vector<std::vector<indices>> indices_reconfigured(num_tiles, std::vector<indices>(num_tiles));
        std::vector<std::vector<TileCoord>> coordinates;
        cv::Mat sampled_mat;
        cv::Mat masked_mat;

        // extracting all the sampled pixels (sampled_mat) and thier coordinates (masked_mat)
        dimgs.get_sampled_mat(sampled_mat, masked_mat);

        // we split the sampled image into tiles
        // every tile overlaps with other neighboring tiles  
        // this is done because otherwise the titles wont quite match with eachother along the borders
        // this becomes more obvious as the numer of samples goes down, aka more compression 
        splitImageIntoTiles(sampled_mat, encrypted_image_tiles, coordinates, num_tiles, overlap);

        cv::Size tile_size;
        int estimated_number_of_samples = 0;

        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {

                std::vector<int> ri_x_g, ri_y_g;

                // reserving space to avoid realocations
                ri_x_g.reserve(estimated_number_of_samples);
                ri_y_g.reserve(estimated_number_of_samples);

                // tile origin as produced by splitImageIntoTiles: the last
                // row/column tiles are anchored to the image edge there, so
                // the origin cannot be re-derived arithmetically here
                const int base_row = coordinates[i][j].y;
                const int base_col = coordinates[i][j].x;

                for (int q = 0; q < encrypted_image_tiles[i][j].rows; q++) {
                    for (int k = 0; k < encrypted_image_tiles[i][j].cols; k++) {
                        if (masked_mat.at<cv::Vec3b>(base_row + q, base_col + k) == cv::Vec3b(1, 1, 1)) {
                            ri_x_g.push_back(q);
                            ri_y_g.push_back(k);
                        }
                    }
                }

                // we use the the number of previous samples as estimations, 
                // all tiles are going to have roughly the same number of samples
                estimated_number_of_samples = ri_x_g.size();

                decrypted_image_tiles[i][j] = cv::Mat::zeros(encrypted_image_tiles[i][j].rows, encrypted_image_tiles[i][j].cols, CV_8UC3);
                encrypt_image img(encrypted_image_tiles[i][j], false);

                indices_reconfigured[i][j] = { ri_x_g, ri_y_g };
                img.encrypt(ri_x_g, ri_y_g);
                encrypted_image_tiles[i][j] = img.get_mat();
                tile_size = img.get_size();
            }
        }
        prof.mark("prep");

        cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);

    std::thread decrypt_tiles_thread;
    // Fused AVIR upscale (default backend, non-full-res): each worker
    // upscales its tile into hr_tiles the moment its solve finishes, so
    // tiles are final without waiting for the wavefront. Subprocess
    // backends (waifu2x/ncnn/realcugan) stream instead: solved tiles are
    // handed to one background upscaler thread tile-by-tile while the
    // wavefront keeps solving, so upscale latency hides behind the solve
    // instead of running as one post-join batch.
    std::vector<std::vector<cv::Mat>> hr_tiles(
        num_tiles, std::vector<cv::Mat>(num_tiles));
    const bool is_sr = cs_is_superres_backend(photo_up.backend);
    // Coupled-dict SR (takes precedence over the FISTA cs-sr solve; works
    // in every container mode since it refines solved tiles): warmed here
    // so the fused workers only read the cache. Missing file = hard error
    // (explicit user flag, fail fast rather than silent AVIR).
    const cs_coupled_dict* sr_d = nullptr;
    if (!photo_up.sr_dict.empty() && !full_res) {
        if (photo_up.sr_red > 0.0f) {
            std::cerr << "Note: --red is ignored with --sr-dict (dict path has no FISTA passes)" << std::endl;
        }
        sr_d = cs_sr_dict_cached(photo_up.sr_dict);
        if (!sr_d) {
            std::cerr << "Error: --sr-dict model failed to load" << std::endl;
            return -3;
        }
    }
    const bool use_dict = (sr_d != nullptr);
    // FSRCNN neural upscale runs fused like AVIR (same post-join grid);
    // a missing model degrades per tile inside the helper (warn-once).
    const std::string fsrcnn_model =
        (!full_res && photo_up.backend == "fsrcnn") ? photo_up.fsrcnn_model : "";
    // DnCNN RED denoiser: warmed here (fail fast on an explicit flag) so
    // the fused workers only read the session cache. NLM needs no warm-up.
    const bool use_dncnn_red = is_sr && !full_res && !use_dict &&
        photo_up.sr_red > 0.0f && photo_up.sr_red_denoiser == "dncnn";
    if (use_dncnn_red && !cs_dncnn_available(photo_up.sr_dncnn_model)) {
        std::cerr << "Error: DnCNN model failed to load" << std::endl;
        return -3;
    }
    const bool fuse_upscale =
        !full_res && (photo_up.backend == "avir" || is_sr || use_dict ||
            !fsrcnn_model.empty());
    const bool pipe_upscale =
        !full_res && !fuse_upscale &&
        (photo_up.backend == "waifu2x" || photo_up.backend == "waifu2x-ncnn" ||
         photo_up.backend == "realcugan");

    // Live preview reads the HR grid when tiles upscale during the solve:
    // a tile appears only once its upscale has landed (pending cells stay
    // black). Otherwise it reads the LR grid as before. hr_grid_mutex
    // serializes cell writes against the preview's reconstructImage.
    ImagePreview preview;
    std::mutex hr_grid_mutex;
    std::vector<std::vector<TileCoord>> hr_coordinates;
    if ((fuse_upscale || pipe_upscale) && show_preview) {
        hr_coordinates = coordinates;
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                hr_coordinates[i][j].y = coordinates[i][j].y * 2;
            }
        preview.Start(windowName, reconstructed, hr_coordinates, hr_tiles,
                      &hr_grid_mutex);
    }
    else if (show_preview) {
        preview.Start(windowName, reconstructed, coordinates, decrypted_image_tiles);
    }

    CsUpscalePipeline pipe;
    if (pipe_upscale)
        pipe.start(photo_up, decrypted_image_tiles, hr_tiles, &hr_grid_mutex);
        decrypt_tiles_thread = std::thread([&]() {
            decrypt_tiles(nun_threads, encrypted_image_tiles, indices_reconfigured,
                decrypted_image_tiles, coordinates, num_tiles, overlap, iterations,
                tile_size, coef, tv, solver,
                fista_iters, reweights, basis, wscale, dimgs.thumb_seed, per_tile_coef, per_tile_tv,
                &hr_tiles, fuse_upscale, pipe_upscale ? &pipe : nullptr,
                (fuse_upscale || pipe_upscale) ? &hr_grid_mutex : nullptr,
                is_sr && !full_res, photo_up.sr_red, sr_d,
                photo_up.sr_red_denoiser, photo_up.sr_dncnn_model, fsrcnn_model);
        });
    decrypt_tiles_thread.join();
    if (pipe_upscale) pipe.finish();

    if (show_preview) {
        preview.Stop();
    }
    prof.mark("solve");

        // 2x upscale via the selected photo backend (AVIR default, waifu2x
        // opt-in): every tile ends up 2x, so all origins scale. full_res
        // solves at native geometry, so its tiles are already final.
        // Fused/pipelined tiles are adopted (with an AVIR fallback for any
        // tile the worker missed); otherwise the batch upscale runs as before.
        int blend_feather = overlap;
        if (fuse_upscale || pipe_upscale) {
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    if (hr_tiles[i][j].empty() && !decrypted_image_tiles[i][j].empty()) {
                        // FSRCNN misses retry neural first (same warn-once),
                        // then AVIR; other backends go straight to AVIR.
                        if (!fsrcnn_model.empty() &&
                            cs_fsrcnn_upscale_2x(decrypted_image_tiles[i][j],
                                hr_tiles[i][j], fsrcnn_model))
                            continue;
                        cs_upscale_2x_avir(decrypted_image_tiles[i][j], hr_tiles[i][j]);
                    }
                }
            decrypted_image_tiles = std::move(hr_tiles);
        }
        if (!full_res && !fuse_upscale && !pipe_upscale) {
            cs_upscale_tiles_2x(decrypted_image_tiles, photo_up);
        }
        if (!full_res) {
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    coordinates[i][j].x *= 2;
                    coordinates[i][j].y *= 2;
                }
            blend_feather = overlap * 2;
        }

        reconstructed = reconstructImage(decrypted_image_tiles, coordinates);

        // cosine-feathered compositing across the overlapping tile borders:
        // ramp width == scaled overlap means non-overlapped interiors keep
        // full weight and the overlap zones sum to a smooth transition
        // (fewer visible seams than the legacy 0.5 alpha blend)
        reconstructed = blendTilesWithImage(decrypted_image_tiles, coordinates, reconstructed, 0.5f, blend_feather);

        // Solve geometry may differ from org_size by container padding;
        // the standard output is 2x org_size (org_size itself under
        // --full-res, which skips the 2x stages on both sides).
        {
            const cv::Size final_size(full_res ? org_size.width : org_size.width * 2,
                full_res ? org_size.height : org_size.height * 2);
            if (final_size.width > 0 && final_size.height > 0 && reconstructed.size() != final_size) {
                cv::resize(reconstructed, reconstructed, final_size);
            }
        }
        prof.mark("composite");

        cv::imwrite(output_path, reconstructed);
        prof.mark("write");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: decryption pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    char prof_tag[40];
    std::snprintf(prof_tag, sizeof(prof_tag), "decrypt tiles=%d", num_tiles);
    prof.dump(prof_tag);
    return 0;
}

int decrypt_image::decrypt_image_tiled_ycc420(decrypt_image& dimgs,
    const std::string& output_path, int num_tiles, int overlap, int iterations,
    int nun_threads, float coef, bool show_preview, float tv,
    bool full_res, int solver,
    int fista_iters, int reweights, int basis, float wscale,
    const CsPhotoUpscalerOptions& photo_up, bool per_tile_coef, bool per_tile_tv) {
    if (solver == CS_SOLVER_FISTA_JOINT) {
        std::cerr << "Warning: joint solver needs one shared RGB grid; ycc split sampling degrades to per-channel FISTA" << std::endl;
    }

    try {
        const cv::Size org_size = dimgs.get_org_size();
        const std::string windowName = output_path.empty() ? "decrypted tiles" : output_path;

        // per-plane sampled images + 0/1 masks (luma full-res, chroma coarse)
        cv::Mat Y, Ym, Cr, Crm, Cb, Cbm;
        dimgs.get_sampled_ycc420(Y, Ym, Cr, Crm, Cb, Cbm);
        const int rows = Y.rows, cols = Y.cols;

        // luma tiling reuses the standard splitter; chroma tiles cover the
        // matching coarse rects (4:2:0 halves both axes, 4:2:2 halves the
        // width only — see the solver for the same formula, the two must
        // agree)
        std::vector<std::vector<cv::Mat>> y_tiles, y_mask_tiles;
        std::vector<std::vector<TileCoord>> coordinates, mask_coords;
        splitImageIntoTiles(Y, y_tiles, coordinates, num_tiles, overlap);
        splitImageIntoTiles(Ym, y_mask_tiles, mask_coords, num_tiles, overlap);
        const bool chroma_sub_v = !cs_is_ycc422(dimgs.get_sampling_mode());

        std::vector<std::vector<cv::Mat>> cr_tiles(num_tiles, std::vector<cv::Mat>(num_tiles)),
            cr_mask_tiles(num_tiles, std::vector<cv::Mat>(num_tiles)),
            cb_tiles(num_tiles, std::vector<cv::Mat>(num_tiles)),
            cb_mask_tiles(num_tiles, std::vector<cv::Mat>(num_tiles));
        std::vector<std::vector<indices>> y_idx(num_tiles, std::vector<indices>(num_tiles)),
            cr_idx(num_tiles, std::vector<indices>(num_tiles)),
            cb_idx(num_tiles, std::vector<indices>(num_tiles));
        std::vector<std::vector<cv::Mat>> y_eph(num_tiles, std::vector<cv::Mat>(num_tiles)),
            cr_eph(num_tiles, std::vector<cv::Mat>(num_tiles)),
            cb_eph(num_tiles, std::vector<cv::Mat>(num_tiles));
        std::vector<std::vector<cv::Mat>> decrypted(num_tiles, std::vector<cv::Mat>(num_tiles));

        cv::Size tile_size(0, 0);
        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                if (y_tiles[i][j].empty() || y_mask_tiles[i][j].empty()) continue;
                const int x = coordinates[i][j].x, y0 = coordinates[i][j].y;
                const int w = y_tiles[i][j].cols, h = y_tiles[i][j].rows;
                const int cx = x / 2, cy = chroma_sub_v ? y0 / 2 : y0;
                const int cw = (x + w + 1) / 2 - x / 2;
                const int ch = chroma_sub_v ? (y0 + h + 1) / 2 - y0 / 2 : h;
                const cv::Rect crect(cx, cy, cw, ch);
                cr_tiles[i][j] = Cr(crect).clone();
                cr_mask_tiles[i][j] = Crm(crect).clone();
                cb_tiles[i][j] = Cb(crect).clone();
                cb_mask_tiles[i][j] = Cbm(crect).clone();

                // local indices by scanning the tile masks (1 == sampled)
                std::vector<int> ylx, yly, crlx, crly, cblx, cbly;
                ylx.reserve(1024); yly.reserve(1024);
                for (int q = 0; q < y_mask_tiles[i][j].rows; q++) {
                    const uint8_t* mp = y_mask_tiles[i][j].ptr<uint8_t>(q);
                    for (int k = 0; k < y_mask_tiles[i][j].cols; k++) {
                        if (mp[k]) { ylx.push_back(q); yly.push_back(k); }
                    }
                }
                crlx.reserve(256); crly.reserve(256); cblx.reserve(256); cbly.reserve(256);
                for (int q = 0; q < cr_mask_tiles[i][j].rows; q++) {
                    const uint8_t* mp = cr_mask_tiles[i][j].ptr<uint8_t>(q);
                    for (int k = 0; k < cr_mask_tiles[i][j].cols; k++) {
                        if (mp[k]) { crlx.push_back(q); crly.push_back(k); }
                    }
                }
                for (int q = 0; q < cb_mask_tiles[i][j].rows; q++) {
                    const uint8_t* mp = cb_mask_tiles[i][j].ptr<uint8_t>(q);
                    for (int k = 0; k < cb_mask_tiles[i][j].cols; k++) {
                        if (mp[k]) { cblx.push_back(q); cbly.push_back(k); }
                    }
                }

                // ephemeral v1 containers: gray planes as BGR, channel 0
                // carries the samples (same convention reconstruct reads)
                cv::Mat y_bgr, cr_bgr, cb_bgr;
                cv::cvtColor(y_tiles[i][j], y_bgr, cv::COLOR_GRAY2BGR);
                cv::cvtColor(cr_tiles[i][j], cr_bgr, cv::COLOR_GRAY2BGR);
                cv::cvtColor(cb_tiles[i][j], cb_bgr, cv::COLOR_GRAY2BGR);
                encrypt_image y_img(y_bgr, false), cr_img(cr_bgr, false), cb_img(cb_bgr, false);
                y_img.encrypt(ylx, yly);
                cr_img.encrypt(crlx, crly);
                cb_img.encrypt(cblx, cbly);
                y_eph[i][j] = y_img.get_mat();
                cr_eph[i][j] = cr_img.get_mat();
                cb_eph[i][j] = cb_img.get_mat();
                y_idx[i][j] = { ylx, yly };
                cr_idx[i][j] = { crlx, crly };
                cb_idx[i][j] = { cblx, cbly };
                tile_size = y_img.get_size();

                decrypted[i][j] = cv::Mat::zeros(h, w, CV_8UC3);
            }
        }

        cv::Mat reconstructed = cv::Mat::zeros(rows, cols, CV_8UC3);

        // Fused AVIR upscale like the BGR pipeline; subprocess backends
        // stream through the background pipeline while solving continues.
        std::vector<std::vector<cv::Mat>> ycc_hr_tiles(
            num_tiles, std::vector<cv::Mat>(num_tiles));
        const bool ycc_is_sr = cs_is_superres_backend(photo_up.backend);
        // CS-SR refines BGR tiles against full-grid random samples; ycc split
        // sampling (luma/chroma on separate grids) keeps the AVIR fused path
        // here (one note, no failure).
        if (ycc_is_sr && !full_res)
            std::cerr << "Note: cs-sr falls back to AVIR for ycc containers"
                " (luma/chroma split sampling)" << std::endl;
        // Coupled-dict SR needs no split-grid operator (post-solve tiles),
        // so it stays available here; warmed before the workers start.
        const cs_coupled_dict* ycc_sr_d = nullptr;
        if (!photo_up.sr_dict.empty() && !full_res) {
            if (photo_up.sr_red > 0.0f) {
                std::cerr << "Note: --red is ignored with --sr-dict (dict path has no FISTA passes)" << std::endl;
            }
            ycc_sr_d = cs_sr_dict_cached(photo_up.sr_dict);
            if (!ycc_sr_d) {
                std::cerr << "Error: --sr-dict model failed to load" << std::endl;
                return -3;
            }
        }
        const bool ycc_fuse = !full_res && (photo_up.backend == "avir" || ycc_is_sr || ycc_sr_d ||
            photo_up.backend == "fsrcnn");
        // (the fsrcnn model path is threaded to the ycc tile solver below)
        const bool ycc_pipe =
            !full_res && !ycc_fuse &&
            (photo_up.backend == "waifu2x" ||
             photo_up.backend == "waifu2x-ncnn" ||
             photo_up.backend == "realcugan");

        // Preview follows the HR grid when tiles upscale during the solve
        // (a tile appears only once upscaled), the LR grid otherwise.
        ImagePreview preview;
        std::mutex ycc_hr_grid_mutex;
        std::vector<std::vector<TileCoord>> ycc_hr_coordinates;
        if ((ycc_fuse || ycc_pipe) && show_preview) {
            ycc_hr_coordinates = coordinates;
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    ycc_hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                    ycc_hr_coordinates[i][j].y = coordinates[i][j].y * 2;
                }
            preview.Start(windowName, reconstructed, ycc_hr_coordinates,
                          ycc_hr_tiles, &ycc_hr_grid_mutex);
        }
        else if (show_preview) {
            preview.Start(windowName, reconstructed, coordinates, decrypted);
        }

        CsUpscalePipeline ycc_upipe;
        if (ycc_pipe)
            ycc_upipe.start(photo_up, decrypted, ycc_hr_tiles,
                            &ycc_hr_grid_mutex);
        decrypt_tiles_ycc420(nun_threads, y_eph, y_idx, cr_eph, cr_idx, cb_eph, cb_idx,
            decrypted, coordinates, num_tiles, overlap, iterations, tile_size, coef, tv,
            solver, fista_iters, reweights, basis, wscale,
            std::cref(dimgs.thumb_seed), per_tile_coef, per_tile_tv,
            &ycc_hr_tiles, ycc_fuse, ycc_pipe ? &ycc_upipe : nullptr,
            (ycc_fuse || ycc_pipe) ? &ycc_hr_grid_mutex : nullptr, chroma_sub_v, ycc_sr_d,
            (!full_res && photo_up.backend == "fsrcnn") ? photo_up.fsrcnn_model : "");
        if (ycc_pipe) ycc_upipe.finish();

        if (show_preview) {
            preview.Stop();
        }

        // Per-tile 2x upscale via the selected photo backend (same as the
        // BGR pipeline); blending happens in 2x solve geometry. full_res
        // solves at native geometry, so its tiles are already final.
        // Fused/pipelined tiles are adopted (AVIR fallback for misses).
        int ycc_blend_feather = overlap;
        if (ycc_fuse || ycc_pipe) {
            const bool ycc_fsr = !full_res && photo_up.backend == "fsrcnn";
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    if (ycc_hr_tiles[i][j].empty() && !decrypted[i][j].empty()) {
                        if (!(ycc_fsr && cs_fsrcnn_upscale_2x(decrypted[i][j],
                                ycc_hr_tiles[i][j], photo_up.fsrcnn_model)))
                            cs_upscale_2x_avir(decrypted[i][j], ycc_hr_tiles[i][j]);
                    }
                }
            decrypted = std::move(ycc_hr_tiles);
        }
        if (!full_res && !ycc_fuse && !ycc_pipe) {
            cs_upscale_tiles_2x(decrypted, photo_up);
        }
        if (!full_res) {
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    coordinates[i][j].x *= 2;
                    coordinates[i][j].y *= 2;
                }
            ycc_blend_feather = overlap * 2;
        }

        reconstructed = reconstructImage(decrypted, coordinates);

        // shared tail with the BGR pipeline: feathered compositing in 2x
        // geometry, resize to 2x the recorded size
        reconstructed = blendTilesWithImage(decrypted, coordinates, reconstructed, 0.5f, ycc_blend_feather);

        {
            const cv::Size final_size(full_res ? org_size.width : org_size.width * 2,
                full_res ? org_size.height : org_size.height * 2);
            if (final_size.width > 0 && final_size.height > 0 && reconstructed.size() != final_size) {
                cv::resize(reconstructed, reconstructed, final_size);
            }
        }

        cv::imwrite(output_path, reconstructed);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: ycc decryption pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    return 0;
}


