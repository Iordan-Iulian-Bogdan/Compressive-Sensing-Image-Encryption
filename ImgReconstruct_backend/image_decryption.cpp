#include "image_decryption.hpp"
#include "helper_functions.hpp"
#include "image_encryption.hpp"
#include "image_preview.hpp"
#include "cs_gpu.h"
#include "cs_dict.hpp"
#include "cs_wavelet.hpp"
#include "avir.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <map>
#include <utility>
#include <algorithm>
#include <thread>
#include <vector>
#include <omp.h>

namespace {
// Per-tile solver settings for the single-stage segment solve (majority
// vote per tile). Defined up here so both wavefront solvers can take an
// optional per-tile grid; vote resolution lives further below.
struct CsSegClassParams {
    int iterations = 5;
    float coef = 0.01f;
    float tv = 0.0f;
    int solver = 0;
    int fista_iters = 0;
    int reweights = 2;
    int basis = 0;
    float wscale = 2.0f;
};
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
    adaptive_cutoff = info.adaptive_cutoff;
    sample_bits = info.sample_bits;
    sample_bits_chroma = info.sample_bits_chroma;
    hr_detail = info.hr_detail;
    hr_count = info.hr_count;
    hr_samples.clear();

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

void decrypt_image::decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, float tv, int solver, int fista_iters, int reweights, int basis, float wscale, bool use_onsager, const cs_meas_groups* groups) {

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
                ref[ch], tv, reweights, fista_iters, basis, wscale, use_onsager, groups);
        for (int ch = 0; ch < 3; ++ch)
            fista_channel_end(ct[ch], ref[ch]);
        cv::merge(ref, 3, out);
        out.convertTo(out, CV_8UC3);
        return;
    }
    if (solver == CS_SOLVER_FISTA_JOINT) {
        reconstruct_image_fista_joint(encrypted_img, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref, tv, reweights, fista_iters, basis, wscale, groups);
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
    if (hr_detail && (sample_bits != 8 || sample_bits_chroma != 8)) {
        throw std::runtime_error("hr-detail container has packed samples (corrupt body)");
    }

    sampled_mat = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    masked_mat = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    // measurement pixel origin inside the container: adaptive mode inserts a
    // lod byte region (padded to whole pixels) between header and body
    int meas0 = CS_HEADER_PIXELS;
    std::vector<uint8_t> lod;

    if (sampling_mode == CS_MODE_SEGMASK) {
        // Per-segment container: the body prefix [nclasses][rates][seglen]
        // [label PNG] carries everything the sampler needs; measurements
        // follow at the first whole pixel after the prefix.
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        CsSegPrefix pre;
        if (total_bytes <= CS_HEADER_BYTES ||
            !cs_segmask_parse_prefix(encrypted_img.data + CS_HEADER_BYTES,
                total_bytes - CS_HEADER_BYTES, pre)) {
            throw std::runtime_error("seg-mask container: corrupt prefix");
        }
        meas0 = CS_HEADER_PIXELS + (int)((pre.total + 2) / 3);
        const uint8_t* png0 = encrypted_img.data + CS_HEADER_BYTES + (pre.total - pre.seglen);
        std::vector<uint8_t> png(png0, png0 + pre.seglen);
        cv::Mat labels = cv::imdecode(png, cv::IMREAD_GRAYSCALE);
        if (labels.empty()) {
            throw std::runtime_error("seg-mask container: label decode failed");
        }
        if (labels.rows != rows || labels.cols != cols) {
            cv::resize(labels, labels, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
        }
        seg_rates_q = pre.rates_q;
        returnSegMaskIndices(ri_x, ri_y, rows, cols, labels, pre.rates_q, cs_key, pre.blend, pre.smooth_blend);
        if ((int)ri_x.size() != m) {
            throw std::runtime_error("seg-mask container: count mismatch");
        }
    }
    else if (sampling_mode == CS_MODE_ADAPTIVE) {
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
        returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, periodic_tile, adaptive_base, lod_full_threshold, adaptive_cutoff);
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
    else if (sampling_mode == CS_MODE_TCLUSTER) {
        // T-cluster points: regenerate the identical pair stream, expand to
        // pixel coords in the same order encrypt wrote them (pair-contiguous,
        // no second shuffle on either side).
        std::vector<int> pairs;
        returnTetrominoPairIds(pairs, rows, cols, (m + 3) / 4, cs_key);
        const int cell_cols = cols / 4;
        ri_x.clear();
        ri_y.clear();
        ri_x.reserve(pairs.size() * 4);
        ri_y.reserve(pairs.size() * 4);
        int xs[4], ys[4];
        for (int p : pairs) {
            const int cell = p / 4, g = p % 4;
            cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
            for (int q = 0; q < 4; ++q) {
                ri_x.push_back(ys[q]);
                ri_y.push_back(xs[q]);
            }
        }
        if ((int)ri_x.size() != m) {
            throw std::runtime_error("tcluster container: count mismatch");
        }
    }
    else if (sampling_mode == CS_MODE_TMEANS) {
        // T-means groups: same pair stream as encrypt (body order = pair
        // order); group k owns body[k]. Builds the global group list for
        // per-tile filtering; per-tile code skips the mask scan for this
        // mode. sampled/masked mats get the mean splatted on the 4 pixels
        // (display/coverage only; the solve uses tgrp_*).
        std::vector<int> pairs;
        returnTetrominoPairIds(pairs, rows, cols, m, cs_key);
        if ((int)pairs.size() != m) {
            throw std::runtime_error("tmeans container: count mismatch");
        }
        const int cell_cols = cols / 4;
        tgrp_off.assign(m + 1, 0);
        tgrp_px.assign((size_t)m * 4, 0);
        // True-average rows (1/4 per pixel): predictions equal the stored
        // means exactly. (Unit-norm 0.5 rows would need sqrt-scaled payloads
        // that overflow u8; the conservative fixed FISTA step stays stable
        // since the true Lipschitz only shrinks.)
        tgrp_w.assign((size_t)m * 4, 0.25f);
        ri_x.clear();
        ri_y.clear();
        ri_x.reserve((size_t)m * 4);
        ri_y.reserve((size_t)m * 4);
        int xs[4], ys[4];
        for (int k = 0; k < m; ++k) {
            const int cell = pairs[(size_t)k] / 4, g = pairs[(size_t)k] % 4;
            cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
            tgrp_off[(size_t)k + 1] = (k + 1) * 4;
            for (int q = 0; q < 4; ++q) {
                tgrp_px[(size_t)k * 4 + q] = ys[q] * cols + xs[q];
                ri_x.push_back(ys[q]);
                ri_y.push_back(xs[q]);
                // values land below (post-unpack); mask marks coverage now
                masked_mat.at<cv::Vec3b>(ys[q], xs[q]) = cv::Vec3b(1, 1, 1);
            }
        }
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
    // T-means carries m group values (not 4m pixel values); the shared
    // 1:1 scatter below does not apply (group values land per-group after).
    const int m_count = (sampling_mode == CS_MODE_TMEANS) ? m : (int)ri_x.size();
    if (sample_bits != 8 || sample_bits_chroma != 8) {
        unpack_container_measurements({ { (size_t)meas0 * 3,
            (size_t)3 * (size_t)m_count, sample_bits, sample_bits_chroma, true } });
    }
    if (sampling_mode == CS_MODE_TMEANS) {
        tgrp_val.assign((size_t)m, cv::Vec3b(0, 0, 0));
        for (int k = 0; k < m; ++k) {
            const cv::Vec3b v = encrypted_img.at<cv::Vec3b>(meas0 + k);
            tgrp_val[(size_t)k] = v;
            for (int q = tgrp_off[(size_t)k]; q < tgrp_off[(size_t)k + 1]; ++q) {
                const int px = tgrp_px[(size_t)q];
                sampled_mat.at<cv::Vec3b>(px / cols, px % cols) = v;
            }
        }
    }
    else {
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

    // Trailing HR-detail payload (mode 4): CTR-decrypt the exact stored
    // bytes after the measurement section into native-luma anchors.
    if (hr_detail) {
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        const size_t pay_off = (size_t)CS_HEADER_BYTES + (size_t)3 * (size_t)m_count;
        const size_t pay_len = (size_t)hr_count * CS_HRDETAIL_SAMPLE_BYTES;
        if (hr_count == 0 || pay_off + pay_len > total_bytes) {
            throw std::runtime_error("hr-detail container: corrupt payload");
        }
        if (!cs_read_hr_payload(encrypted_img.data + pay_off, pay_len,
                cs_key, encrypted_img.data + CS_OFF_IV, hr_samples)) {
            throw std::runtime_error("hr-detail container: corrupt payload");
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
    if (hr_detail && (sample_bits != 8 || sample_bits_chroma != 8)) {
        throw std::runtime_error("hr-detail container has packed samples (corrupt body)");
    }

    // LOD-luma variant: lod bytes sit between header and measurements (same
    // padded layout as mode 2); the luma draw regenerates from them while
    // chroma stays uniform. Pure-uniform mode 3 has no lod region.
    size_t body_off = 0;
    int mY_written = m;
    int chroma_mode = sampling_mode;
    // T-luma modes (15-18): chroma machinery runs on the base split mode.
    if (cs_is_tcluster(sampling_mode) || cs_is_tmeans(sampling_mode)) {
        chroma_mode = cs_is_ycc422(sampling_mode) ? CS_MODE_YCC422 : CS_MODE_YCC420;
    }
    if (sampling_mode == CS_MODE_YCC420_SEG || sampling_mode == CS_MODE_YCC422_SEG) {
        // Seg-luma variant: the body prefix [nclasses][rates][seglen][label
        // PNG] regenerates the luma draw; chroma stays uniform on the base
        // split geometry (4:2:0 vs 4:2:2).
        chroma_mode = (sampling_mode == CS_MODE_YCC422_SEG) ? CS_MODE_YCC422 : CS_MODE_YCC420;
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        CsSegPrefix pre;
        if (total_bytes <= CS_HEADER_BYTES ||
            !cs_segmask_parse_prefix(encrypted_img.data + CS_HEADER_BYTES,
                total_bytes - CS_HEADER_BYTES, pre)) {
            throw std::runtime_error("seg-mask container: corrupt prefix");
        }
        body_off = pre.total;
        const uint8_t* png0 = encrypted_img.data + CS_HEADER_BYTES + (pre.total - pre.seglen);
        std::vector<uint8_t> png(png0, png0 + pre.seglen);
        cv::Mat labels = cv::imdecode(png, cv::IMREAD_GRAYSCALE);
        if (labels.empty()) {
            throw std::runtime_error("seg-mask container: label decode failed");
        }
        if (labels.rows != rows || labels.cols != cols) {
            cv::resize(labels, labels, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
        }
        seg_rates_q = pre.rates_q;
        returnSegMaskIndices(ri_x, ri_y, rows, cols, labels, pre.rates_q, cs_key, pre.blend, pre.smooth_blend);
        if ((int)ri_x.size() != m) {
            throw std::runtime_error("seg-mask container: count mismatch");
        }
        mY_written = m;
    }
    else if (cs_is_tcluster(sampling_mode) || cs_is_tmeans(sampling_mode)) {
        // T-luma: same pair stream as BGR modes on the full-res luma grid;
        // chroma stays uniform (handled below on chroma_mode). m is points
        // (tcluster) or groups (tmeans). Group geometry lands in tgrp_*
        // for per-tile filtering; values land post-unpack below.
        std::vector<int> pairs;
        const int want = cs_is_tmeans(sampling_mode) ? m : (m + 3) / 4;
        returnTetrominoPairIds(pairs, rows, cols, want, cs_key);
        const int cell_cols = cols / 4;
        ri_x.clear();
        ri_y.clear();
        int xs[4], ys[4];
        if (cs_is_tmeans(sampling_mode)) {
            if ((int)pairs.size() != m) {
                throw std::runtime_error("tmeans container: count mismatch");
            }
            tgrp_off.assign(m + 1, 0);
            tgrp_px.assign((size_t)m * 4, 0);
            tgrp_w.assign((size_t)m * 4, 0.25f);
            tgrp_val.assign((size_t)m, cv::Vec3b(0, 0, 0));
            ri_x.reserve((size_t)m * 4);
            ri_y.reserve((size_t)m * 4);
            for (int k = 0; k < m; ++k) {
                const int cell = pairs[(size_t)k] / 4, g = pairs[(size_t)k] % 4;
                cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
                tgrp_off[(size_t)k + 1] = (k + 1) * 4;
                for (int q = 0; q < 4; ++q) {
                    tgrp_px[(size_t)k * 4 + q] = ys[q] * cols + xs[q];
                    ri_x.push_back(ys[q]);
                    ri_y.push_back(xs[q]);
                }
            }
            mY_written = m;
        }
        else {
            ri_x.reserve(pairs.size() * 4);
            ri_y.reserve(pairs.size() * 4);
            for (int p : pairs) {
                const int cell = p / 4, g = p % 4;
                cs_tetro_group_pixels((cell / cell_cols) * 4, (cell % cell_cols) * 4, g, xs, ys);
                for (int q = 0; q < 4; ++q) {
                    ri_x.push_back(ys[q]);
                    ri_y.push_back(xs[q]);
                }
            }
            if ((int)ri_x.size() != m) {
                throw std::runtime_error("tcluster container: count mismatch");
            }
            mY_written = m;
        }
    }
    else if (cs_is_ycc_hf(sampling_mode)) {
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
        returnAdaptiveIndices(ri_x, ri_y, rows, cols, lod, m, cs_key, periodic_tile, adaptive_base, lod_full_threshold, adaptive_cutoff);
        mY_written = (int)ri_x.size();
        body_off = (size_t)lod_pixels * 3;
    } else {
        // uniform luma draw from the base key
        ri_x.resize((size_t)m);
        ri_y.resize((size_t)m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);
    }
    // chroma draws are uniform in every variant (tagged keys); dims and
    // counts follow the base split mode (4:2:0 halves both axes, 4:2:2
    // halves the width only)
    returnYcc420ChromaIndices(rows, cols, m, cs_key, chroma_mode);
    const int mC = m_chroma;

    int crows, ccols;
    cs_chroma_dims(chroma_mode, rows, cols, crows, ccols);

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

    // Trailing HR-detail payload (mode 9): CTR-decrypt the exact stored
    // bytes after the luma/chroma sections into native-luma anchors.
    if (hr_detail) {
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        const size_t pay_off = (size_t)CS_HEADER_BYTES + body_off +
            (size_t)mY_written + (size_t)2 * (size_t)mC;
        const size_t pay_len = (size_t)hr_count * CS_HRDETAIL_SAMPLE_BYTES;
        if (hr_count == 0 || pay_off + pay_len > total_bytes) {
            throw std::runtime_error("hr-detail container: corrupt payload");
        }
        if (!cs_read_hr_payload(encrypted_img.data + pay_off, pay_len,
                cs_key, encrypted_img.data + CS_OFF_IV, hr_samples)) {
            throw std::runtime_error("hr-detail container: corrupt payload");
        }
    }

    y = cv::Mat::zeros(rows, cols, CV_8U);
    y_mask = cv::Mat::zeros(rows, cols, CV_8U);
    cr = cv::Mat::zeros(crows, ccols, CV_8U);
    cr_mask = cv::Mat::zeros(crows, ccols, CV_8U);
    cb = cv::Mat::zeros(crows, ccols, CV_8U);
    cb_mask = cv::Mat::zeros(crows, ccols, CV_8U);

    if (cs_is_tmeans(sampling_mode)) {
        // Group values (post-unpack body): splat each mean on its 4 luma
        // pixels (display/coverage); the solve uses tgrp_* + tgrp_val.
        for (int k = 0; k < m; ++k) {
            tgrp_val[(size_t)k] = cv::Vec3b(body[k], body[k], body[k]);
            for (int q = tgrp_off[(size_t)k]; q < tgrp_off[(size_t)k + 1]; ++q) {
                const int px = tgrp_px[(size_t)q];
                y.at<uint8_t>(px / cols, px % cols) = body[k];
                y_mask.at<uint8_t>(px / cols, px % cols) = 1;
            }
        }
    }
    else for (int k = 0; k < mY_written; k++) {
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
            cs_dct_forward(plane, cs_default_dct_block());
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
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, float tv, int solver, int fista_iters, int reweights, int basis, float wscale, const cv::Mat& thumbnail_seed, bool per_tile_coef, bool per_tile_tv, std::vector<std::vector<cv::Mat>>* hr_out, bool fuse_upscale, CsUpscalePipeline* pipe,     std::mutex* hr_grid_mutex, bool superres_2x, float sr_red, const cs_coupled_dict* sr_dict, const std::string& sr_red_denoiser, const std::string& sr_dncnn_model,     const std::string& sr_fsrcnn_model, bool sr_direct_2x, bool sr_cascade, float grain_tv, bool use_onsager,
    const std::string& red_model, float red_decay, bool sr_dip_2x, const CsHrSrCtx& hr_sr,
    const std::vector<std::vector<CsSegClassParams>>* seg_tile_params = nullptr) {

    // solve-stage profiler: one reset here, one dump line at exit (env-gated)
    cs_solveprof_reset();
    // Averaged (T-means) tiles have no in-solve SR/DIP operator yet: those
    // backends read ri vectors, which stay empty on grouped tiles.
    if (superres_2x || sr_direct_2x || sr_dip_2x) {
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j)
                if (!indices[i][j].rg_off.empty()) {
                    throw std::runtime_error("--sampling tmeans needs the standard/AVIR path (no sr-direct/cs-sr/dip operator yet)");
                }
    }
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
                const int mm = cs_tile_meas_count(indices[i][j]);
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

                if (mats_in[i][j].empty()) continue;
                // Single-stage segment solve: this tile's class params win;
                // without them the global solve settings apply (bit-identical).
                const CsSegClassParams* STP = seg_tile_params ? &(*seg_tile_params)[i][j] : nullptr;
                const int t_iterations = STP ? STP->iterations : iterations;
                const float t_coef_base = STP ? STP->coef : coef;
                const float t_tv_base = STP ? STP->tv : tv;
                const int t_solver = STP ? STP->solver : solver;
                const int t_fista_iters = STP ? STP->fista_iters : fista_iters;
                const int t_reweights = STP ? STP->reweights : reweights;
                const int t_basis = STP ? STP->basis : basis;
                const float t_wscale = STP ? STP->wscale : wscale;
                auto sp_a = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                cv::Mat x0[3];
                build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j, false, false, t_basis, thumbnail_seed);
                auto sp_b = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                if (sp) g_solveprof.warm_ns += cs_solveprof_ns_since(sp_a);

                decrypt_image dimgs = decrypt_image(mats_in[i][j]);
                auto sp_c = sp ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                if (sp) g_solveprof.ctor_ns += cs_solveprof_ns_since(sp_b);

                // starved tiles solve under a stronger prior, rich tiles relaxed
                // (tv follows the same law under its own flag; base tv 0 stays 0)
                // Grouped tiles count groups (ri lists stay empty there).
                const struct indices& t_idx = indices[i][j];
                const int t_count = cs_tile_meas_count(t_idx);
                const float tile_coef = per_tile_coef
                    ? cs_per_tile_coef(t_coef_base, t_count, coef_mean_m)
                    : t_coef_base;
                const float tile_tv = per_tile_tv
                    ? cs_per_tile_coef(t_tv_base, t_count, coef_mean_m)
                    : t_tv_base;
                cs_meas_groups grps;
                const cs_meas_groups* pgrps = nullptr;
                if (!t_idx.rg_off.empty() && (int)t_idx.rg_off.size() > 1) {
                    grps.off = t_idx.rg_off.data();
                    grps.px = t_idx.rg_px.data();
                    grps.w = t_idx.rg_w.data();
                    grps.ng = (int)t_idx.rg_off.size() - 1;
                    pgrps = &grps;
                }
                dimgs.decrypt(x0, t_idx.ri_x_g, t_idx.ri_y_g, t_iterations, tile_coef, mats_out[i][j], tile_tv, t_solver, t_fista_iters, t_reweights, t_basis, t_wscale, use_onsager, pgrps);
                if (sp) { g_solveprof.call_ns += cs_solveprof_ns_since(sp_c); g_solveprof.tiles += 1; }
                // grain-adaptive per-tile TV (--grain-tv): tiles whose
                // measured noise exceeds the reference get a luma ROF
                // smooth proportional to the excess, before any upscale
                // handoff; clean tiles pass through untouched.
                if (grain_tv > 0.0f && !mats_out[i][j].empty()) {
                    const float lam = cs_grain_tv_lambda(
                        cs_estimate_noise_sigma(mats_out[i][j]), grain_tv);
                    if (lam > 0.0f)
                        cs_tv_denoise_tile(mats_out[i][j], lam);
                }
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
                    else if (sr_direct_2x) {
                        // Single-stage HR solve directly from the LR samples
                        // (no LR-solve warm start in the coefficients).
                        cs_sr_direct_upscale_tile_2x(mats_out[i][j], mats_in[i][j],
                            indices[i][j].ri_x_g, indices[i][j].ri_y_g,
                            tile_coef, tile_tv, t_iterations, t_reweights, t_fista_iters,
                            t_basis, t_wscale, hr, sr_cascade);
                    }
                    else if (superres_2x) {
                        // Tile-mapped HR anchors join the solve (neighborhood
                        // improvement); post-hoc replacement still pins the
                        // anchor pixels exactly afterwards.
                        CsHrSrCtx tile_hr = hr_sr;
                        tile_hr.ox = coordinates[i][j].x * 2;
                        tile_hr.oy = coordinates[i][j].y * 2;
                        cs_sr_upscale_tile_2x(mats_out[i][j], mats_in[i][j],
                            indices[i][j].ri_x_g, indices[i][j].ri_y_g,
                            tile_coef, tile_tv, t_iterations, t_reweights, t_fista_iters,
                            t_basis, t_wscale, hr, sr_red, sr_red_denoiser, sr_dncnn_model,
                            sr_cascade, red_model, red_decay, tile_hr);
                    }
                    else if (sr_dip_2x) {
                        // Untrained-prior refinement: tiny generator fit to
                        // the tile's own LR samples (no weights, no basis).
                        cs_dip_upscale_tile_2x(mats_out[i][j], mats_in[i][j],
                            indices[i][j].ri_x_g, indices[i][j].ri_y_g, hr);
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
                else if (hr_out && hr_grid_mutex && !pipe && !fuse_upscale) {
                    // Publish an immutable completed LR tile for preview;
                    // never let the UI read a tile being solved in place.
                    std::lock_guard<std::mutex> lk(*hr_grid_mutex);
                    (*hr_out)[i][j] = mats_out[i][j].clone();
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
    const std::string& sr_fsrcnn_model = "", float y_confidence = 0.0f,
    float y_dual_tv = 0.0f,
    std::vector<std::vector<cv::Mat>>* low_tv_tiles = nullptr, float grain_tv = 0.0f,
    bool use_onsager = false, bool superres_2x = false, float sr_red = 0.0f,
    const std::string& sr_red_denoiser = "nlmeans",
    const std::string& sr_dncnn_model = CS_DNCNN_DEFAULT_MODEL,
    bool sr_direct_2x = false, bool sr_cascade = false,
    const std::string& red_model = "", float red_decay = 1.0f,
    bool sr_dip_2x = false, const CsHrSrCtx& hr_sr = CsHrSrCtx(),
    int chroma_iterations = -1, float chroma_coef = -1.0f,
    const std::vector<std::vector<CsSegClassParams>>* seg_tile_params = nullptr) {

    if (y_confidence > 0.0f && basis != CS_BASIS_DCT)
        std::cerr << "Note: --y-confidence requires --basis dct; ignoring for wavelet" << std::endl;

    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);
    const int iters_off = iterations / 2;
    // Optional chroma settings; segment solves use their own class settings.
    const int chroma_it = chroma_iterations >= 0 ? chroma_iterations : iterations;
    const int iters_c = chroma_it - chroma_it / 2;
    const float coef_c = (chroma_coef >= 0.0f ? chroma_coef : coef) * 0.5f;
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
                const int mm = cs_tile_meas_count(y_idx[ii][jj]);
                if (mm > 0) { sum += mm; ++cnt; }
            }
        if (cnt > 0) coef_mean_my = (double)sum / cnt;
    }
    // Averaged (T-means) luma has no in-solve SR/DIP operator yet.
    if (superres_2x || sr_direct_2x || sr_dip_2x) {
        for (int ii = 0; ii < num_tiles; ++ii)
            for (int jj = 0; jj < num_tiles; ++jj)
                if (!y_idx[ii][jj].rg_off.empty()) {
                    throw std::runtime_error("--sampling tmeans needs the standard/AVIR path (no sr-direct/cs-sr/dip operator yet)");
                }
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

                const CsSegClassParams* STP = seg_tile_params ? &(*seg_tile_params)[i][j] : nullptr;
                const int t_iterations = STP ? STP->iterations : iterations;
                const float t_coef_base = STP ? STP->coef : coef;
                const float t_tv_base = STP ? STP->tv : tv;
                const int t_solver = STP ? STP->solver : solver;
                const int t_fista_iters = STP ? STP->fista_iters : fista_iters;
                const int t_reweights = STP ? STP->reweights : reweights;
                const int t_basis = STP ? STP->basis : basis;
                const float t_wscale = STP ? STP->wscale : wscale;
                // Chroma follows the tile's class params (same convention as
                // the per-class passes); otherwise the pinned/global settings.
                const int t_chroma_it = STP ? STP->iterations : chroma_it;
                const int t_iters_c = t_chroma_it - t_chroma_it / 2;
                const float t_coef_c = (STP ? STP->coef : (chroma_coef >= 0.0f ? chroma_coef : coef)) * 0.5f;

                cv::Mat x0[3];
                build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j, true, true, t_basis, thumbnail_seed, chroma_sub_v);

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
                            if (t_basis == CS_BASIS_CDF97) {
                                cs_dwt_forward((float*)x0[c].data, cTh, cTw,
                                    cs_dwt_levels(cTh, cTw));
                            } else {
                                cs_dct_forward(x0[c], cs_default_dct_block());
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
                const struct indices& y_tidx = y_idx[i][j];
                const float tile_coef_y = per_tile_coef
                    ? cs_per_tile_coef(t_coef_base, cs_tile_meas_count(y_tidx), coef_mean_my)
                    : t_coef_base;
                const float tile_tv_y = per_tile_tv
                    ? cs_per_tile_coef(t_tv_base, cs_tile_meas_count(y_tidx), coef_mean_my)
                    : t_tv_base;
                // Grouped luma (T-means) rides the weight-aware core;
                // chroma draws are always uniform points.
                cs_meas_groups y_grps;
                const cs_meas_groups* y_pgrps = nullptr;
                if (!y_tidx.rg_off.empty() && (int)y_tidx.rg_off.size() > 1) {
                    y_grps.off = y_tidx.rg_off.data();
                    y_grps.px = y_tidx.rg_px.data();
                    y_grps.w = y_tidx.rg_w.data();
                    y_grps.ng = (int)y_tidx.rg_off.size() - 1;
                    y_pgrps = &y_grps;
                }
                // FISTA-only (joint degrades to per-channel FISTA at the
                // dispatcher): luma at full tile resolution, chroma natively
                // on the half-resolution grid, where proximal updates are
                // exact at the near-flat warm starts.
                auto solve_plane = [&](const cv::Mat& meas, float c, int R, int C,
                    const std::vector<int>& rx, const std::vector<int>& ry,
                    int it, cv::Mat& r, bool nxt, cv::Mat& nr, float t,
                    const cs_meas_groups* pg = nullptr) {
                    reconstruct_color_channel_fista(meas, 0, c, R, C, rx, ry, it, r, nxt, nr, t, t_reweights, t_fista_iters, t_basis, t_wscale, use_onsager, pg);
                };
                // Solve high-TV first as the base, then low-TV from the same
                // neighbour warm start (independent, not a continuation).
                const bool mixed = low_tv_tiles && y_dual_tv > 0.0f &&
                    cs_y_mixed_density(y_tidx.ri_x_g, y_tidx.ri_y_g, Th, Tw);
                cv::Mat y_low = mixed ? x0[0].clone() : cv::Mat();
                if (!y_low.empty())
                    solve_plane(y_meas[i][j], tile_coef_y, Th, Tw,
                        y_tidx.ri_x_g, y_tidx.ri_y_g, t_iterations,
                        x0[0], false, dummy, y_dual_tv, y_pgrps);
                cv::Mat& low = y_low.empty() ? x0[0] : y_low;
                solve_plane(y_meas[i][j], tile_coef_y, Th, Tw,
                    y_tidx.ri_x_g, y_tidx.ri_y_g, t_iterations, low,
                    false, dummy, tile_tv_y, y_pgrps);
                if (y_confidence > 0.0f)
                    cs_refine_y_confidence(y_meas[i][j], y_tidx.ri_x_g,
                        y_tidx.ri_y_g, low, y_confidence, tile_coef_y,
                        tile_tv_y, t_iterations, t_fista_iters, t_basis, t_wscale);
                solve_plane(cr_meas[i][j], t_coef_c, cTh, cTw,
                    cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, t_iters_c, x0[1],
                    true, x0[2], 0.0f);
                solve_plane(cb_meas[i][j], t_coef_c, cTh, cTw,
                    cb_idx[i][j].ri_x_g, cb_idx[i][j].ri_y_g, t_iters_c, x0[2],
                    false, dummy, 0.0f);

                cv::Mat cr_full, cb_full;
                cv::resize(x0[1], cr_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                cv::resize(x0[2], cb_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                cv::Mat chs[3] = { x0[0], cr_full, cb_full }; // (Y, Cr, Cb)
                cv::Mat ycc_tile;
                cv::merge(chs, 3, ycc_tile);
                ycc_tile.convertTo(ycc_tile, CV_8UC3);
                cv::cvtColor(ycc_tile, mats_out[i][j], cv::COLOR_YCrCb2BGR);
                // grain-adaptive per-tile TV (--grain-tv): same post-solve
                // ROF smooth as the BGR path, on the merged tile.
                if (grain_tv > 0.0f && !mats_out[i][j].empty()) {
                    const float lam = cs_grain_tv_lambda(
                        cs_estimate_noise_sigma(mats_out[i][j]), grain_tv);
                    if (lam > 0.0f)
                        cs_tv_denoise_tile(mats_out[i][j], lam);
                }
                if (low_tv_tiles) {
                    if (mixed) {
                        cv::Mat low_ycc;
                        cv::Mat low_chs[3] = { y_low, cr_full, cb_full };
                        cv::merge(low_chs, 3, low_ycc);
                        low_ycc.convertTo(low_ycc, CV_8UC3);
                        cv::cvtColor(low_ycc, (*low_tv_tiles)[i][j], cv::COLOR_YCrCb2BGR);
                    } else {
                        (*low_tv_tiles)[i][j] = mats_out[i][j];
                    }
                }
                // streaming pipeline: same handoff as decrypt_tiles above
                if (pipe && !mats_out[i][j].empty()) {
                    pipe->enqueue(i, j);
                }
                // fused upscale: same op as the post-join batch, so
                // bit-identical; separate grid keeps preview reads safe.
                // A coupled dict refines the merged BGR tile (post-solve, so
                // split-grid sampling needs no special operator here). The
                // CS SR backends refine luma-only against the tile's own LR
                // luma samples: y_meas channel 0 carries luma, and the
                // helpers' BGR2Y weights collapse to 1.0 on the gray-as-BGR
                // container, so no separate chroma operator is needed
                // (chroma comes from the AVIR upscale inside the helper).
                else if (fuse_upscale && hr_out && !mats_out[i][j].empty()) {
                    cv::Mat hr;
                    if (sr_dict) {
                        cs_sr_upscale_dict_2x(mats_out[i][j], sr_dict, hr);
                    }
                    else if (!sr_fsrcnn_model.empty()) {
                        cs_fsrcnn_upscale_2x(mats_out[i][j], hr, sr_fsrcnn_model);
                    }
                    else if (sr_direct_2x) {
                        cs_sr_direct_upscale_tile_2x(mats_out[i][j], y_meas[i][j],
                            y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g,
                            tile_coef_y, tile_tv_y, t_iterations, t_reweights, t_fista_iters,
                            t_basis, t_wscale, hr, sr_cascade);
                    }
                    else if (superres_2x) {
                        CsHrSrCtx tile_hr = hr_sr;
                        tile_hr.ox = coordinates[i][j].x * 2;
                        tile_hr.oy = coordinates[i][j].y * 2;
                        cs_sr_upscale_tile_2x(mats_out[i][j], y_meas[i][j],
                            y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g,
                            tile_coef_y, tile_tv_y, t_iterations, t_reweights, t_fista_iters,
                            t_basis, t_wscale, hr, sr_red, sr_red_denoiser, sr_dncnn_model,
                            sr_cascade, red_model, red_decay, tile_hr);
                    }
                    else if (sr_dip_2x) {
                        cs_dip_upscale_tile_2x(mats_out[i][j], y_meas[i][j],
                            y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g, hr);
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
                else if (hr_out && hr_grid_mutex && !pipe && !fuse_upscale) {
                    std::lock_guard<std::mutex> lk(*hr_grid_mutex);
                    (*hr_out)[i][j] = mats_out[i][j].clone();
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
    bool per_tile_tv,
    float grain_tv,
    bool use_onsager,
    const std::string& seg_solve_mask,
    const std::string& seg_params
) {
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    return decrypt_image_tiled(encrypted_img_g, output_path, password, num_tiles, overlap, iterations, nun_threads, coef, show_preview, tv, full_res, solver, fista_iters, reweights, basis, wscale, photo_up, per_tile_coef, per_tile_tv, grain_tv, use_onsager, nullptr, seg_solve_mask, seg_params);
}

// --seg-params per-class solver overrides ("cls:knob=value,...;...") ------
// Knobs: iterations|coef|tv|solver|fista-iters|reweights|basis|wscale.
// Everything is validated up front (fail fast before any solve); classes
// without an entry inherit the global decrypt settings.
struct CsSegOverride {
    bool has_iterations = false;
    bool has_coef = false;
    bool has_tv = false;
    bool has_solver = false;
    bool has_fista_iters = false;
    bool has_reweights = false;
    bool has_basis = false;
    bool has_wscale = false;
    int iterations = 0;
    float coef = 0.0f;
    float tv = 0.0f;
    int solver = 0;
    int fista_iters = 0;
    int reweights = 0;
    int basis = 0;
    float wscale = 0.0f;
};

static std::string cs_seg_trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static int cs_seg_parse_int(const std::string& tok, const char* what) {
    try {
        size_t pos = 0;
        const int v = std::stoi(tok, &pos);
        if (pos != tok.size()) throw std::runtime_error("x");
        return v;
    }
    catch (...) {
        throw std::runtime_error(std::string("seg-params: bad integer for ") + what + " in '" + tok + "'");
    }
}

static float cs_seg_parse_float(const std::string& tok, const char* what) {
    try {
        size_t pos = 0;
        const float v = std::stof(tok, &pos);
        if (pos != tok.size()) throw std::runtime_error("x");
        return v;
    }
    catch (...) {
        throw std::runtime_error(std::string("seg-params: bad number for ") + what + " in '" + tok + "'");
    }
}

static std::map<int, CsSegOverride> cs_seg_parse_spec(const std::string& spec) {
    std::map<int, CsSegOverride> out;
    size_t p = 0;
    while (true) {
        const size_t q = spec.find(';', p);
        const std::string entry = cs_seg_trim((q == std::string::npos) ? spec.substr(p) : spec.substr(p, q - p));
        if (!entry.empty()) {
            const size_t c = entry.find(':');
            if (c == std::string::npos) {
                throw std::runtime_error("seg-params: entry without 'cls:' in '" + entry + "'");
            }
            const int cls = cs_seg_parse_int(cs_seg_trim(entry.substr(0, c)), "class id");
            if (cls < 0 || cls > 255) {
                throw std::runtime_error("seg-params: class id must be in [0, 255]");
            }
            CsSegOverride& o = out[cls];
            size_t k = c + 1;
            while (true) {
                const size_t m = entry.find(',', k);
                const std::string kv = cs_seg_trim((m == std::string::npos) ? entry.substr(k) : entry.substr(k, m - k));
                if (kv.empty()) {
                    throw std::runtime_error("seg-params: empty knob in '" + entry + "'");
                }
                const size_t e = kv.find('=');
                if (e == std::string::npos) {
                    throw std::runtime_error("seg-params: knob without '=' in '" + kv + "'");
                }
                const std::string knob = cs_seg_trim(kv.substr(0, e));
                const std::string val = cs_seg_trim(kv.substr(e + 1));
                if (knob == "iterations") {
                    o.iterations = cs_seg_parse_int(val, "iterations");
                    if (o.iterations < 1 || o.iterations > 50) throw std::runtime_error("seg-params: iterations must be in [1, 50]");
                    o.has_iterations = true;
                }
                else if (knob == "coef") {
                    o.coef = cs_seg_parse_float(val, "coef");
                    if (!(o.coef > 0.0f)) throw std::runtime_error("seg-params: coef must be positive");
                    o.has_coef = true;
                }
                else if (knob == "tv") {
                    o.tv = cs_seg_parse_float(val, "tv");
                    if (!(o.tv >= 0.0f)) throw std::runtime_error("seg-params: tv must be >= 0");
                    o.has_tv = true;
                }
                else if (knob == "solver") {
                    if (cs_solver_from_name(val, o.solver) != 0) throw std::runtime_error("seg-params: solver must be fista or joint");
                    o.has_solver = true;
                }
                else if (knob == "fista-iters") {
                    o.fista_iters = cs_seg_parse_int(val, "fista-iters");
                    if (o.fista_iters < 0 || o.fista_iters > 500) throw std::runtime_error("seg-params: fista-iters must be in [0, 500]");
                    o.has_fista_iters = true;
                }
                else if (knob == "reweights") {
                    o.reweights = cs_seg_parse_int(val, "reweights");
                    if (o.reweights < 1 || o.reweights > 5) throw std::runtime_error("seg-params: reweights must be in [1, 5]");
                    o.has_reweights = true;
                }
                else if (knob == "basis") {
                    if (cs_basis_from_name(val, o.basis) != 0) throw std::runtime_error("seg-params: basis must be dct or wavelet");
                    o.has_basis = true;
                }
                else if (knob == "wscale") {
                    o.wscale = cs_seg_parse_float(val, "wscale");
                    if (!(o.wscale > 0.0f) || !(o.wscale <= 16.0f)) throw std::runtime_error("seg-params: wscale must be in (0, 16]");
                    o.has_wscale = true;
                }
                else {
                    throw std::runtime_error("seg-params: unknown knob '" + knob + "' (iterations|coef|tv|solver|fista-iters|reweights|basis|wscale)");
                }
                if (m == std::string::npos) break;
                k = m + 1;
            }
        }
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

// Effective per-solve settings: globals with one class's overrides applied.
// Retain every tile intersecting the segment, including mixed-label tiles.
// The integral mask makes scheduling independent of measurement density.
struct CsSegmentTiles {
    cv::Mat segment, integral, support;
    int kept = 0, boundary = 0, skipped = 0;
    explicit CsSegmentTiles(const cv::Mat& mask) : segment(mask) {
        if (!mask.empty()) {
            cv::Mat binary;
            mask.convertTo(binary, CV_8U, 1.0 / 255.0);
            cv::integral(binary, integral, CV_64F);
            support = cv::Mat::zeros(mask.size(), CV_8U);
        }
    }
    bool retain(const cv::Rect& r) {
        if (segment.empty()) return true;
        const double count = integral.at<double>(r.y+r.height, r.x+r.width)
            - integral.at<double>(r.y, r.x+r.width)
            - integral.at<double>(r.y+r.height, r.x) + integral.at<double>(r.y,r.x);
        if (count == 0) { ++skipped; return false; }
        ++kept;
        if (count < r.area()) ++boundary;
        support(r).setTo(255);
        return true;
    }
    cv::Mat weights(cv::Size size, int overlap) const {
        if (segment.empty()) return {};
        cv::Mat w, covered;
        segment.convertTo(w, CV_32F, 1.0/255.0);
        // Only a narrow boundary band mixes class candidates. Full-tile
        // context is retained, but unsupported predictions never contribute.
        if (overlap > 0)
            cv::GaussianBlur(w, w, {}, (std::min)(8.0, overlap / 3.0), 0, cv::BORDER_REPLICATE);
        if (w.size() != size) cv::resize(w, w, size, 0, 0, cv::INTER_LINEAR);
        cv::resize(support, covered, size, 0, 0, cv::INTER_NEAREST);
        w.setTo(0, covered == 0);
        std::cout << "seg-tiles: kept=" << kept << " boundary=" << boundary
                  << " skipped=" << skipped << std::endl;
        return w;
    }
};

struct CsSegmentComposite {
    cv::Mat sum, weight, detail;
    void add(const cv::Mat& canvas, const cv::Mat& w,
        const cv::Mat& owner_mask, int overlap, double density) {
        if (sum.empty()) {
            sum = cv::Mat::zeros(canvas.size(), CV_32FC3);
            weight = cv::Mat::zeros(canvas.size(), CV_32F);
            detail = cv::Mat::zeros(canvas.size(), CV_32FC3);
        }
        if (canvas.size() != sum.size() || w.size() != sum.size())
            throw std::runtime_error("segment composite: inconsistent canvas geometry");
        cv::Mat f, w3, owner;
        canvas.convertTo(f, CV_32FC3);
        cv::resize(owner_mask, owner, canvas.size(), 0, 0, cv::INTER_NEAREST);
        owner.convertTo(owner, CV_32F, 1.0 / 255.0);
        if (overlap > 0 && density < 0.4) {
            // Range weights follow actual image edges, not the approximate
            // segmentation contour. This damps ringing without turning a
            // slightly misregistered label into a new color discontinuity.
            cv::Mat owned3, smooth, distance, gate;
            cv::bilateralFilter(f, smooth, 13, 18.0, 4.0, cv::BORDER_REPLICATE);
            cv::Mat binary;
            owner.convertTo(binary, CV_8U, 255.0);
            cv::distanceTransform(binary, distance, cv::DIST_L2, 3);
            gate = 1.0f - distance / (float)overlap;
            cv::max(gate, 0, gate);
            cv::min(gate, 1, gate);
            cv::multiply(gate, owner, gate);
            cv::merge(std::vector<cv::Mat>{gate, gate, gate}, owned3);
            cv::Mat correction;
            cv::multiply(smooth - f, owned3, correction);
            f += correction;
        }
        // Two-band reconciliation: only low frequencies are feathered.
        // All high-frequency content (including the segment's denoising)
        // stays with its original label. Normalize the blur by valid support
        // so sparse class canvases cannot pull black pixels into the seam.
        cv::Mat low;
        if (overlap > 0) {
            cv::Mat valid = (w > 0), valid_f, valid3, numerator, denominator;
            valid.convertTo(valid_f, CV_32F, 1.0 / 255.0);
            cv::merge(std::vector<cv::Mat>{valid_f, valid_f, valid_f}, valid3);
            cv::multiply(f, valid3, numerator);
            const double sigma = (std::min)(8.0, overlap / 3.0);
            cv::GaussianBlur(numerator, numerator, {}, sigma, 0, cv::BORDER_REPLICATE);
            cv::GaussianBlur(valid_f, denominator, {}, sigma, 0, cv::BORDER_REPLICATE);
            cv::max(denominator, 1e-6, denominator);
            cv::merge(std::vector<cv::Mat>{denominator, denominator, denominator}, valid3);
            cv::divide(numerator, valid3, low);
        } else low = f;
        cv::Mat owner3, residual;
        cv::merge(std::vector<cv::Mat>{owner, owner, owner}, owner3);
        cv::multiply(f - low, owner3, residual);
        detail += residual;
        cv::merge(std::vector<cv::Mat>{w,w,w}, w3);
        cv::Mat weighted_low;
        cv::multiply(low, w3, weighted_low);
        sum += weighted_low;
        weight += w;
    }
    cv::Mat finish(bool partial = false) const {
        if (sum.empty() || (!partial && cv::countNonZero(weight <= 0) != 0))
            throw std::runtime_error("segment composite: uncovered pixels");
        std::vector<cv::Mat> channels;
        cv::split(sum, channels);
        cv::Mat divisor;
        cv::max(weight, 1e-12, divisor);
        for (auto& c : channels) cv::divide(c, divisor, c);
        cv::Mat out;
        cv::merge(channels, out);
        out += detail;
        out.convertTo(out, CV_8UC3);
        return out;
    }
};

// Cheap single-image boundary calm for the single-stage solve: edge-aware
// bilateral correction near low-density segment frontiers, no re-solves.
// Range weights follow actual image edges rather than the approximate label
// contour, so ringing is damped without baking a misregistered contour into
// a new discontinuity. Strength fades to zero one overlap width inside the
// class; dense segments pass through untouched.
static void cs_seg_calm_boundaries(cv::Mat& img, const cv::Mat& seg_ids,
    const cv::Mat& sample_mask, const std::vector<int>& classes, int overlap) {
    if (overlap <= 0 || img.empty() || seg_ids.empty() || sample_mask.empty()) return;
    cv::Mat ids = seg_ids, mask = sample_mask;
    if (ids.size() != img.size())
        cv::resize(ids, ids, img.size(), 0, 0, cv::INTER_NEAREST);
    if (mask.size() != img.size())
        cv::resize(mask, mask, img.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat chan;
    if (mask.channels() == 1) chan = mask;
    else cv::extractChannel(mask, chan, 0);
    // Restrict the expensive bilateral to the union of low-density boundary
    // bands (dilated by the filter radius): bands are a small fraction of
    // the frame, so this is ~50x cheaper than full-frame filtering.
    cv::Mat band_union = cv::Mat::zeros(ids.size(), CV_8U);
    std::vector<cv::Mat> low_masks;
    for (int c : classes) {
        cv::Mat in_c = (ids == c);
        const int area = cv::countNonZero(in_c);
        if (area == 0) continue;
        cv::Mat m;
        cv::bitwise_and(chan, in_c, m);
        if ((double)cv::countNonZero(m) / (double)area >= 0.4) continue;
        low_masks.push_back(in_c);
        cv::Mat dist;
        cv::distanceTransform(in_c, dist, cv::DIST_L2, 3);
        cv::bitwise_or(band_union, dist <= (float)overlap, band_union);
    }
    if (low_masks.empty()) return;
    cv::Mat work_zone;
    cv::dilate(band_union, work_zone, cv::getStructuringElement(cv::MORPH_ELLIPSE, {33, 33}));
    cv::Rect crop = cv::boundingRect(work_zone);
    crop &= cv::Rect(0, 0, img.cols, img.rows);
    if (crop.empty()) return;
    cv::Mat f;
    img.convertTo(f, CV_32FC3);
    cv::Mat smooth_crop;
    cv::bilateralFilter(f(crop), smooth_crop, 13, 18.0, 4.0, cv::BORDER_REPLICATE);
    int touched = 0;
    for (const cv::Mat& in_full : low_masks) {
        cv::Mat in_c = in_full(crop);
        cv::Mat owner, distance, gate, gate3, correction;
        in_c.convertTo(owner, CV_32F, 1.0 / 255.0);
        cv::distanceTransform(in_c, distance, cv::DIST_L2, 3);
        gate = 1.0f - distance / (float)overlap;
        cv::max(gate, 0, gate);
        cv::min(gate, 1, gate);
        cv::multiply(gate, owner, gate);
        cv::merge(std::vector<cv::Mat>{gate, gate, gate}, gate3);
        cv::Mat f_crop = f(crop);
        cv::multiply(smooth_crop - f_crop, gate3, correction);
        f_crop += correction;
        touched++;
    }
    if (touched > 0) {
        f.convertTo(img, CV_8UC3);
        std::cout << "seg-solve: boundary calm on " << touched << " low-density classes" << std::endl;
    }
}

static void cs_seg_apply_override(CsSegClassParams& p, const CsSegOverride& o) {
    if (o.has_iterations) p.iterations = o.iterations;
    if (o.has_coef) p.coef = o.coef;
    if (o.has_tv) p.tv = o.tv;
    if (o.has_solver) p.solver = o.solver;
    if (o.has_fista_iters) p.fista_iters = o.fista_iters;
    if (o.has_reweights) p.reweights = o.reweights;
    if (o.has_basis) p.basis = o.basis;
    if (o.has_wscale) p.wscale = o.wscale;
}

// Single-stage segment solve: every tile is solved exactly once — pure tiles
// with their own class's parameters, mixed tiles with the highest-rate
// intersecting class's. No per-class solver passes, no cross-canvas blending;
// neighboring tiles meet through the standard cosine-feathered tile blend,
// exactly like global per-tile coef scaling.
struct CsSegSinglePass {
    const cv::Mat* ids = nullptr; // solve-geometry label map (ids-geometry == tile geometry)
    CsSegClassParams base;
    const std::map<int, CsSegOverride>* spec = nullptr;
    const std::vector<uint8_t>* rates_q = nullptr; // tie-break only (higher rate wins)
};

// Resolve one CsSegClassParams per tile. Pure tiles use their own class.
// Mixed tiles use the highest-rate intersecting class: the dense side keeps
// its detail, and over-serving the sparse side is harmless (verified: full
// background at bird iterations loses nothing). Majority vote only breaks
// ties without rate information (lower id wins). One scan per tile —
// negligible next to a FISTA solve.
static void cs_seg_resolve_tile_params(const CsSegSinglePass& sp,
    const std::vector<std::vector<TileCoord>>& coords,
    const std::vector<std::vector<cv::Mat>>& tile_mats, int num_tiles,
    std::vector<std::vector<CsSegClassParams>>& out, std::vector<int>& winners) {
    out.assign(num_tiles, std::vector<CsSegClassParams>(num_tiles, sp.base));
    winners.assign(num_tiles * num_tiles, -1);
    if (!sp.ids || sp.ids->empty()) return;
    const cv::Mat& ids = *sp.ids;
    std::vector<int> counts(256, 0);
    const auto rate_of = [&](int c) {
        return (sp.rates_q && c >= 0 && c < (int)sp.rates_q->size()) ? (*sp.rates_q)[c] : 0;
    };
    for (int i = 0; i < num_tiles; ++i) {
        for (int j = 0; j < num_tiles; ++j) {
            if (tile_mats[i][j].empty()) continue;
            const int x0 = coords[i][j].x, y0 = coords[i][j].y;
            const int xe = (std::min)(x0 + tile_mats[i][j].cols, ids.cols);
            const int ye = (std::min)(y0 + tile_mats[i][j].rows, ids.rows);
            std::fill(counts.begin(), counts.end(), 0);
            for (int y = y0; y < ye; ++y) {
                const uint8_t* row = ids.ptr<uint8_t>(y);
                for (int x = x0; x < xe; ++x) counts[row[x]]++;
            }
            int distinct = 0, sole = 0;
            for (int c = 0; c < 256; ++c) {
                if (counts[c] > 0) { distinct++; sole = c; }
            }
            int best = sole, best_n = counts[sole];
            if (distinct > 1) {
                // Mixed tile: highest sampled rate wins (detail preserved
                // where it matters); ties fall back to majority, then id.
                best = -1;
                best_n = -1;
                int best_q = -1;
                for (int c = 0; c < 256; ++c) {
                    if (counts[c] == 0) continue;
                    const int qc = rate_of(c);
                    if (qc > best_q || (qc == best_q && (best < 0 || counts[c] > best_n ||
                        (counts[c] == best_n && c < best)))) {
                        best = c;
                        best_n = counts[c];
                        best_q = qc;
                    }
                }
            }
            winners[i * num_tiles + j] = best;
            CsSegClassParams cp = sp.base;
            if (sp.spec) {
                const auto oit = sp.spec->find(best);
                if (oit != sp.spec->end()) cs_seg_apply_override(cp, oit->second);
            }
            out[i][j] = cp;
        }
    }
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
    bool per_tile_tv,
    float grain_tv,
    bool use_onsager,
    size_t* hr_applied_out,
    const std::string& seg_solve_mask,
    const std::string& seg_params
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

    // YCC containers take a dedicated tile pipeline: luma at
    // full resolution, chroma natively on the coarse grid. The mode travels
    // in the header so no decrypt-side flags are needed.
    if (cs_is_ycc(dimgs.get_sampling_mode())) {
        return decrypt_image_tiled_ycc420(dimgs, output_path, num_tiles, overlap,
            iterations, nun_threads, coef, show_preview, tv,
            full_res, solver, fista_iters, reweights, basis, wscale,
            photo_up, per_tile_coef, per_tile_tv, grain_tv, use_onsager,
            hr_applied_out, seg_solve_mask, seg_params);
    }
    if (photo_up.y_dual_tv > 0.0f || photo_up.y_confidence > 0.0f)
        std::cerr << "Note: --y-dual-tv/--y-confidence require a YCC container; ignoring for BGR" << std::endl;
    if (!photo_up.dual_tv_split.empty())
        std::cerr << "Note: --dual-tv-split needs a YCC container (--ycc420/--ycc422); nothing will be written" << std::endl;
    if (!photo_up.dual_tv_cutouts.empty())
        std::cerr << "Note: --dual-tv-cutouts needs a YCC container (--ycc420/--ycc422); nothing will be written" << std::endl;

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

        // Tile-split + per-tile solve + composite as one reusable unit: the
        // standard path runs it once on all samples; --seg-solve runs it once
        // with per-tile class parameters (majority vote, no per-class passes).
        // (Body kept at its original indentation; sm/mm are the inputs.)
        auto solve_grid = [&](const cv::Mat& sm, const cv::Mat& mm, bool use_preview,
                const CsSegClassParams& P, cv::Mat& grid_out,
                const cv::Mat& active_segment, cv::Mat& blend_weight,
                const CsSegSinglePass* segpass = nullptr) -> int {
        CsSegmentTiles active(active_segment);
        for (auto& row : decrypted_image_tiles) for (auto& tile : row) tile.release();
        for (auto& row : indices_reconfigured) for (auto& idx : row) {
            idx.ri_x_g.clear(); idx.ri_y_g.clear();
        }
        // we split the sampled image into tiles
        // every tile overlaps with other neighboring tiles  
        // this is done because otherwise the titles wont quite match with eachother along the borders
        // this becomes more obvious as the numer of samples goes down, aka more compression 
        splitImageIntoTiles(sm, encrypted_image_tiles, coordinates, num_tiles, overlap);
        // Single-stage segment solve: one resolved param set per tile by
        // majority vote; every tile is solved exactly once with its own
        // class's settings (no per-class passes, no cross-canvas blending).
        std::vector<std::vector<CsSegClassParams>> seg_tiles;
        std::vector<int> seg_winners;
        if (segpass) {
            cs_seg_resolve_tile_params(*segpass, coordinates, encrypted_image_tiles,
                num_tiles, seg_tiles, seg_winners);
            std::vector<int> per_class(256, 0);
            for (int v : seg_winners) if (v >= 0) per_class[v]++;
            std::cout << "seg-solve: single stage, per-class tile counts:";
            for (int c = 0; c < 256; ++c)
                if (per_class[c] > 0) std::cout << " " << c << "=" << per_class[c];
            std::cout << std::endl;
        }
        const auto* seg_tile_params = seg_tiles.empty() ? nullptr : &seg_tiles;

        cv::Size tile_size;
        int estimated_number_of_samples = 0;

        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {

                const cv::Mat& tile = encrypted_image_tiles[i][j];
                if (tile.empty()) continue;
                const cv::Rect rect(coordinates[i][j].x, coordinates[i][j].y, tile.cols, tile.rows);
                if (!active.retain(rect)) {
                    encrypted_image_tiles[i][j].release();
                    continue;
                }
                std::vector<int> ri_x_g, ri_y_g;
                indices tile_idx;

                // reserving space to avoid realocations
                ri_x_g.reserve(estimated_number_of_samples);
                ri_y_g.reserve(estimated_number_of_samples);

                // tile origin as produced by splitImageIntoTiles: the last
                // row/column tiles are anchored to the image edge there, so
                // the origin cannot be re-derived arithmetically here
                const int base_row = coordinates[i][j].y;
                const int base_col = coordinates[i][j].x;
                const int tile_w = encrypted_image_tiles[i][j].cols;
                const int tile_h = encrypted_image_tiles[i][j].rows;

                if (dimgs.get_sampling_mode() == CS_MODE_TMEANS) {
                    // Grouped measurements: keep groups with any pixel in
                    // this tile's rect (Centroid-free overlap rule, same as
                    // point double-counting; feathering renormalizes).
                    // ri_x_g/ri_y_g stay empty; counts derive from ng.
                    const std::vector<int>& goff = dimgs.get_tgrp_off();
                    const std::vector<int>& gpx = dimgs.get_tgrp_px();
                    const std::vector<float>& ggw = dimgs.get_tgrp_w();
                    const std::vector<cv::Vec3b>& gval = dimgs.get_tgrp_val();
                    const int ng = (int)goff.size() - 1;
                    const int gw_all = sm.cols;
                    tile_idx.rg_off.reserve((size_t)ng + 1);
                    tile_idx.rg_px.reserve((size_t)ng * 4);
                    tile_idx.rg_w.reserve((size_t)ng * 4);
                    tile_idx.rg_off.push_back(0);
                    std::vector<cv::Vec3b> gvals;
                    gvals.reserve((size_t)ng);
                    for (int g = 0; g < ng; ++g) {
                        bool hit = false;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= base_col && gx < base_col + tile_w &&
                                gy >= base_row && gy < base_row + tile_h) {
                                hit = true;
                                break;
                            }
                        }
                        if (!hit) continue;
                        // Clip straddling groups to this tile's rect and
                        // renormalize the kept weights (x4/kept) so each
                        // tile's rows still predict the stored mean.
                        int kept = 0;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= base_col && gx < base_col + tile_w &&
                                gy >= base_row && gy < base_row + tile_h) {
                                ++kept;
                            }
                        }
                        if (kept == 0) continue;
                        const float renorm = 4.0f / (float)kept;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= base_col && gx < base_col + tile_w &&
                                gy >= base_row && gy < base_row + tile_h) {
                                tile_idx.rg_px.push_back((gy - base_row) * tile_w + (gx - base_col));
                                tile_idx.rg_w.push_back(ggw[(size_t)e] * renorm);
                            }
                        }
                        tile_idx.rg_off.push_back((int)tile_idx.rg_px.size());
                        gvals.push_back(gval[(size_t)g]);
                    }
                    estimated_number_of_samples = (int)tile_idx.rg_off.size() - 1;

                    decrypted_image_tiles[i][j] = cv::Mat::zeros(tile_h, tile_w, CV_8UC3);
                    encrypt_image img(encrypted_image_tiles[i][j], false);
                    indices_reconfigured[i][j] = tile_idx;
                    img.encrypt_values(gvals);
                    encrypted_image_tiles[i][j] = img.get_mat();
                    tile_size = img.get_size();
                }
                else {
                    for (int q = 0; q < encrypted_image_tiles[i][j].rows; q++) {
                        for (int k = 0; k < encrypted_image_tiles[i][j].cols; k++) {
                            if (mm.at<cv::Vec3b>(base_row + q, base_col + k) == cv::Vec3b(1, 1, 1)) {
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
        }
        prof.mark("prep");

        cv::Mat reconstructed = cv::Mat::zeros(sm.rows, sm.cols, CV_8UC3);

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
    const bool is_direct = cs_is_srdirect_backend(photo_up.backend);
    const bool is_dip = cs_is_dip_backend(photo_up.backend);
    const bool sr_cascade = (photo_up.sr_init == "cascade");
    // Coupled-dict SR (takes precedence over the FISTA cs-sr solve; works
    // in every container mode since it refines solved tiles): warmed here
    // so the fused workers only read the cache. Missing file = hard error
    // (explicit user flag, fail fast rather than silent AVIR).
    const cs_coupled_dict* sr_d = nullptr;
    if (!photo_up.sr_dict.empty() && !full_res) {
        if (photo_up.sr_red > 0.0f) {
            std::cerr << "Note: --red is ignored with --sr-dict (dict path has no FISTA passes)" << std::endl;
        }
        if (is_direct) {
            std::cerr << "Note: --sr-dict takes precedence over sr-direct (dict refines solved tiles)" << std::endl;
        }
        sr_d = cs_sr_dict_cached(photo_up.sr_dict);
        if (!sr_d) {
            std::cerr << "Error: --sr-dict model failed to load" << std::endl;
            return -3;
        }
    }
    const bool use_dict = (sr_d != nullptr);
    if (is_direct && !full_res && !use_dict && photo_up.sr_red > 0.0f) {
        std::cerr << "Note: --red is ignored with sr-direct (single-stage solve has no RED passes)" << std::endl;
    }
    if (is_dip && !full_res && !use_dict && photo_up.sr_red > 0.0f) {
        std::cerr << "Note: --red is ignored with dip (untrained-prior solve has no RED passes)" << std::endl;
    }
    if (sr_cascade && full_res) {
        std::cerr << "Warning: --sr-init is meaningless with --full-res; ignoring" << std::endl;
    }
    else if (sr_cascade && !is_sr && !is_direct && !use_dict) {
        std::cerr << "Warning: --sr-init cascade needs a CS SR backend (--photo-upscaler cs-sr/sr-direct) or --sr-dict; ignoring" << std::endl;
    }
    // FSRCNN neural upscale runs fused like AVIR (same post-join grid);
    // a missing model degrades per tile inside the helper (warn-once).
    const std::string fsrcnn_model =
        (!full_res && photo_up.backend == "fsrcnn") ? photo_up.fsrcnn_model : "";
    // ONNX RED denoiser (dncnn/scunet/restormer): model resolved once and
    // warmed here (fail fast on an explicit flag) so the fused workers only
    // read the session cache. NLM needs no warm-up.
    const int red_kind = cs_onnx_kind_for(photo_up.sr_red_denoiser);
    const bool use_onnx_red = is_sr && !full_res && !use_dict &&
        photo_up.sr_red > 0.0f && red_kind >= 0;
    const std::string red_model = use_onnx_red ?
        cs_red_model_for(photo_up.sr_red_denoiser, photo_up.sr_dncnn_model,
            photo_up.sr_red_model) : std::string();
    if (use_onnx_red && !cs_onnx_available(red_model, red_kind)) {
        std::cerr << "Error: RED model failed to load ('" << red_model << "')" << std::endl;
        return -3;
    }
    const bool fuse_upscale =
        !full_res && (photo_up.backend == "avir" || is_sr || is_direct || is_dip || use_dict ||
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
    if ((fuse_upscale || pipe_upscale) && use_preview) {
        hr_coordinates = coordinates;
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                hr_coordinates[i][j].y = coordinates[i][j].y * 2;
            }
        preview.Start(windowName, reconstructed, hr_coordinates, hr_tiles,
                      &hr_grid_mutex);
    }
    else if (use_preview) {
        preview.Start(windowName, reconstructed, coordinates, hr_tiles,
                      &hr_grid_mutex);
    }

    CsUpscalePipeline pipe;
    if (pipe_upscale)
        pipe.start(photo_up, decrypted_image_tiles, hr_tiles, &hr_grid_mutex);
    // HR anchors for the fused cs-sr solve: global output-geometry list +
    // output stride; each worker maps its own tile origin (see decrypt_tiles).
    CsHrSrCtx hr_sr_ctx;
    if (is_sr && !full_res && !use_dict && !dimgs.hr_samples.empty() &&
        (photo_up.sr_hr_weight > 0.0f ||
            (photo_up.sr_hr_red_radius > 0.0f && photo_up.sr_red > 0.0f))) {
        hr_sr_ctx.anchors = &dimgs.hr_samples;
            hr_sr_ctx.stride = sm.cols * 2;
        hr_sr_ctx.weight = photo_up.sr_hr_weight;
        hr_sr_ctx.red_radius = photo_up.sr_hr_red_radius;
    }
    if (photo_up.sr_hr_red_radius > 0.0f && photo_up.sr_red > 0.0f &&
        !(is_sr && !full_res && !use_dict && !dimgs.hr_samples.empty())) {
        std::cerr << "Note: --hr-red-radius needs cs-sr with --hr-detail anchors; using uniform RED" << std::endl;
    }
        decrypt_tiles_thread = std::thread([&]() {
            decrypt_tiles(nun_threads, encrypted_image_tiles, indices_reconfigured,
                decrypted_image_tiles, coordinates, num_tiles, overlap, P.iterations,
                tile_size, P.coef, P.tv, P.solver,
                P.fista_iters, P.reweights, P.basis, P.wscale, dimgs.thumb_seed, per_tile_coef, per_tile_tv,
                &hr_tiles, fuse_upscale, pipe_upscale ? &pipe : nullptr,
                (fuse_upscale || pipe_upscale || use_preview) ? &hr_grid_mutex : nullptr,
                is_sr && !full_res && !use_dict, photo_up.sr_red, sr_d,
                photo_up.sr_red_denoiser, photo_up.sr_dncnn_model, fsrcnn_model,
                is_direct && !full_res && !use_dict, sr_cascade, grain_tv, use_onsager,
                red_model, photo_up.sr_red_decay, is_dip && !full_res && !use_dict,
                hr_sr_ctx, seg_tile_params);
        });
    decrypt_tiles_thread.join();
    if (pipe_upscale) pipe.finish();

    if (use_preview) {
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

        // Blend canvas: the plain tile paste for the standard path (the
        // zero-overlap alpha blend composites tiles against it, so zeros
        // would halve every pixel). Segment solves keep the zeros canvas:
        // released (inactive-class) tiles must contribute nothing.
        if (segpass) {
            reconstructed = cv::Mat::zeros(sm.rows * (full_res ? 1 : 2),
                sm.cols * (full_res ? 1 : 2), CV_8UC3);
        } else {
            reconstructed = reconstructImage(decrypted_image_tiles, coordinates);
        }

        // cosine-feathered compositing across the overlapping tile borders:
        // ramp width == scaled overlap means non-overlapped interiors keep
        // full weight and the overlap zones sum to a smooth transition
        // (fewer visible seams than the legacy 0.5 alpha blend)
        reconstructed = blendTilesWithImage(decrypted_image_tiles, coordinates, reconstructed, 0.5f, blend_feather);
            grid_out = reconstructed;
            blend_weight = active.weights(grid_out.size(), overlap);
            return 0;
        }; // solve_grid

        cv::Mat reconstructed;
        // Base per-solve settings (globals, possibly auto-derived above);
        // --seg-params overrides them per class. The spec parses once here
        // so malformed entries fail before any solve runs.
        CsSegClassParams base_p;
        base_p.iterations = iterations;
        base_p.coef = coef;
        base_p.tv = tv;
        base_p.solver = solver;
        base_p.fista_iters = fista_iters;
        base_p.reweights = reweights;
        base_p.basis = basis;
        base_p.wscale = wscale;
        const std::map<int, CsSegOverride> seg_spec = seg_params.empty()
            ? std::map<int, CsSegOverride>() : cs_seg_parse_spec(seg_params);
        if (!seg_params.empty() && seg_solve_mask.empty()) {
            throw std::runtime_error("--seg-params needs --seg-solve");
        }
        if (!seg_solve_mask.empty()) {
            // Keep mixed tiles in each intersecting class's solve: a global
            // boundary solve would discard the class's regularization.
            if (cs_is_ycc(dimgs.get_sampling_mode())) {
                throw std::runtime_error("--seg-solve needs a BGR container; YCC-seg containers solve luma/chroma jointly");
            }
            cv::Mat mask_raw = cv::imread(seg_solve_mask, cv::IMREAD_UNCHANGED);
            if (mask_raw.empty()) {
                throw std::runtime_error("--seg-solve: cannot load '" + seg_solve_mask + "'");
            }
            cv::Mat seg_ids = cs_segmask_resolve_ids(mask_raw, sampled_mat.rows, sampled_mat.cols);
            bool seen[256] = { false };
            for (int y = 0; y < seg_ids.rows; ++y) {
                const uint8_t* row = seg_ids.ptr<uint8_t>(y);
                for (int x = 0; x < seg_ids.cols; ++x) seen[row[x]] = true;
            }
            std::vector<int> classes;
            for (int c = 0; c < 256; ++c) if (seen[c]) classes.push_back(c);
            if (classes.empty()) {
                throw std::runtime_error("--seg-solve: mask holds no classes");
            }
            // Single solver stage: every tile is solved exactly once, with
            // the resolved parameters of its majority class (--seg-params
            // overrides apply per tile). Neighboring tiles meet through the
            // standard cosine-feathered tile blend, so there is no second
            // solver stage and no cross-canvas seam reconciliation.
            std::cout << "seg-solve: " << classes.size() << " classes, single stage" << std::endl;
            for (const auto& kv : seg_spec) {
                CsSegClassParams cp = base_p;
                cs_seg_apply_override(cp, kv.second);
                std::cout << "seg-solve: class " << kv.first
                          << " [iters=" << cp.iterations << " coef=" << cp.coef
                          << " tv=" << cp.tv << " solver=" << cp.solver
                          << " fista-iters=" << cp.fista_iters << " reweights=" << cp.reweights
                          << " basis=" << cp.basis << " wscale=" << cp.wscale << "]" << std::endl;
            }
            CsSegSinglePass segpass;
            segpass.ids = &seg_ids;
            segpass.base = base_p;
            segpass.spec = &seg_spec;
            segpass.rates_q = &dimgs.get_seg_rates_q();
            cv::Mat unused_weight;
            const int grid_rc = solve_grid(sampled_mat, masked_mat, show_preview, base_p,
                reconstructed, {}, unused_weight, &segpass);
            if (grid_rc != 0) return grid_rc;
            cs_seg_calm_boundaries(reconstructed, seg_ids, masked_mat, classes, overlap);
        }
        else {
            cv::Mat unused_weight;
            const int grid_rc = solve_grid(sampled_mat, masked_mat, show_preview, base_p, reconstructed, {}, unused_weight);
            if (grid_rc != 0) return grid_rc;
        }

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

        // Final polish (--final-polish): neural 2x upscale of the finished
        // composite + downsample back (misses keep the image untouched).
        if (!photo_up.final_polish.empty() && !reconstructed.empty() &&
            cs_final_polish_updown(reconstructed, photo_up)) {
            prof.mark("polish");
        }
        // HR-detail anchors (modes 4/9): exact native-luma replacements on
        // the final composite, applied last so smoothing passes cannot move
        // them. Out-of-range anchors (geometry mismatch) are skipped.
        if (!dimgs.hr_samples.empty() && !reconstructed.empty()) {
            const size_t applied = cs_apply_hr_detail(reconstructed, dimgs.hr_samples);
            dimgs.hr_applied = applied;
            if (hr_applied_out) *hr_applied_out = applied;
            std::cout << "HR-detail: applied " << applied << "/"
                      << dimgs.hr_samples.size() << " anchors" << std::endl;
        }
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
    const CsPhotoUpscalerOptions& photo_up, bool per_tile_coef, bool per_tile_tv, float grain_tv, bool use_onsager, size_t* hr_applied_out,
    const std::string& seg_solve_mask, const std::string& seg_params) {
    if (solver == CS_SOLVER_FISTA_JOINT) {
        std::cerr << "Warning: joint solver needs one shared RGB grid; ycc split sampling degrades to per-channel FISTA" << std::endl;
    }

    try {
        const cv::Size org_size = dimgs.get_org_size();
        const std::string windowName = output_path.empty() ? "decrypted tiles" : output_path;

        // per-plane sampled images + 0/1 masks (luma full-res, chroma coarse)
        cv::Mat Y, Ym, Cr, Crm, Cb, Cbm;
        dimgs.get_sampled_ycc420(Y, Ym, Cr, Crm, Cb, Cbm);

        // Luma/chroma split + per-plane solve + merge as one reusable unit:
        // the standard path runs it once; --seg-solve runs it once with
        // per-tile class parameters (majority vote, no per-class passes).
        auto solve_ycc_grid = [&](const cv::Mat& Y0, const cv::Mat& Ym0,
                const cv::Mat& Cr0, const cv::Mat& Crm0,
                const cv::Mat& Cb0, const cv::Mat& Cbm0,
                const CsSegClassParams& P, const CsSegClassParams& Pc, bool use_preview,
                cv::Mat& grid_out, const cv::Mat& active_segment,
                cv::Mat& blend_weight, const CsSegSinglePass* segpass = nullptr) -> int {
        CsSegmentTiles active(active_segment);
        const int rows = Y0.rows, cols = Y0.cols;

        // luma tiling reuses the standard splitter; chroma tiles cover the
        // matching coarse rects (4:2:0 halves both axes, 4:2:2 halves the
        // width only — see the solver for the same formula, the two must
        // agree)
        std::vector<std::vector<cv::Mat>> y_tiles, y_mask_tiles;
        std::vector<std::vector<TileCoord>> coordinates, mask_coords;
        splitImageIntoTiles(Y0, y_tiles, coordinates, num_tiles, overlap);
        splitImageIntoTiles(Ym0, y_mask_tiles, mask_coords, num_tiles, overlap);
        std::vector<std::vector<CsSegClassParams>> seg_tiles;
        std::vector<int> seg_winners;
        if (segpass) {
            cs_seg_resolve_tile_params(*segpass, coordinates, y_tiles,
                num_tiles, seg_tiles, seg_winners);
            std::vector<int> per_class(256, 0);
            for (int v : seg_winners) if (v >= 0) per_class[v]++;
            std::cout << "seg-solve: single stage, per-class tile counts:";
            for (int c = 0; c < 256; ++c)
                if (per_class[c] > 0) std::cout << " " << c << "=" << per_class[c];
            std::cout << std::endl;
        }
        const auto* seg_tile_params = seg_tiles.empty() ? nullptr : &seg_tiles;
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
        std::vector<std::vector<cv::Mat>> low_tv_tiles;
        const bool dual_tv = photo_up.y_dual_tv > 0.0f;
        if (dual_tv) low_tv_tiles.resize(num_tiles, std::vector<cv::Mat>(num_tiles));

        cv::Size tile_size(0, 0);
        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                if (y_tiles[i][j].empty() || y_mask_tiles[i][j].empty()) continue;
                const int x = coordinates[i][j].x, y0 = coordinates[i][j].y;
                const int w = y_tiles[i][j].cols, h = y_tiles[i][j].rows;
                if (!active.retain(cv::Rect(x,y0,w,h))) continue;
                const int cx = x / 2, cy = chroma_sub_v ? y0 / 2 : y0;
                const int cw = (x + w + 1) / 2 - x / 2;
                const int ch = chroma_sub_v ? (y0 + h + 1) / 2 - y0 / 2 : h;
                const cv::Rect crect(cx, cy, cw, ch);
                cr_tiles[i][j] = Cr0(crect).clone();
                cr_mask_tiles[i][j] = Crm0(crect).clone();
                cb_tiles[i][j] = Cb0(crect).clone();
                cb_mask_tiles[i][j] = Cbm0(crect).clone();

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
                if (cs_is_tmeans(dimgs.get_sampling_mode())) {
                    // Grouped luma: filter the global T groups by this tile
                    // rect (clip + renormalize, BGR-prep convention); ri
                    // lists stay empty, counts derive from ng.
                    const std::vector<int>& goff = dimgs.get_tgrp_off();
                    const std::vector<int>& gpx = dimgs.get_tgrp_px();
                    const std::vector<float>& ggw = dimgs.get_tgrp_w();
                    const std::vector<cv::Vec3b>& gval = dimgs.get_tgrp_val();
                    const int ng = (int)goff.size() - 1;
                    const int gw_all = Y0.cols;
                    indices ygroups;
                    ygroups.rg_off.reserve((size_t)ng + 1);
                    ygroups.rg_px.reserve((size_t)ng * 4);
                    ygroups.rg_w.reserve((size_t)ng * 4);
                    ygroups.rg_off.push_back(0);
                    std::vector<cv::Vec3b> yvals;
                    yvals.reserve((size_t)ng);
                    for (int g = 0; g < ng; ++g) {
                        bool hit = false;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= x && gx < x + w && gy >= y0 && gy < y0 + h) {
                                hit = true;
                                break;
                            }
                        }
                        if (!hit) continue;
                        int kept = 0;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= x && gx < x + w && gy >= y0 && gy < y0 + h) ++kept;
                        }
                        if (kept == 0) continue;
                        const float renorm = 4.0f / (float)kept;
                        for (int e = goff[(size_t)g]; e < goff[(size_t)g + 1]; ++e) {
                            const int gx = gpx[(size_t)e] % gw_all;
                            const int gy = gpx[(size_t)e] / gw_all;
                            if (gx >= x && gx < x + w && gy >= y0 && gy < y0 + h) {
                                ygroups.rg_px.push_back((gy - y0) * w + (gx - x));
                                ygroups.rg_w.push_back(ggw[(size_t)e] * renorm);
                            }
                        }
                        ygroups.rg_off.push_back((int)ygroups.rg_px.size());
                        const uint8_t v = gval[(size_t)g][0];
                        yvals.push_back(cv::Vec3b(v, v, v));
                    }
                    y_idx[i][j] = ygroups;
                    y_img.encrypt_values(yvals);
                }
                else {
                    y_img.encrypt(ylx, yly);
                    y_idx[i][j] = { ylx, yly };
                }
                cr_img.encrypt(crlx, crly);
                cb_img.encrypt(cblx, cbly);
                y_eph[i][j] = y_img.get_mat();
                cr_eph[i][j] = cr_img.get_mat();
                cb_eph[i][j] = cb_img.get_mat();
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
        const bool ycc_is_direct = cs_is_srdirect_backend(photo_up.backend);
        const bool ycc_is_dip = cs_is_dip_backend(photo_up.backend);
        const bool ycc_sr_cascade = (photo_up.sr_init == "cascade");
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
        const bool ycc_use_dict = (ycc_sr_d != nullptr);
        // ONNX RED denoiser: same fail-fast warm-up as the BGR pipeline so
        // the fused workers only read the session cache (NLM needs none).
        const int ycc_red_kind = cs_onnx_kind_for(photo_up.sr_red_denoiser);
        const bool ycc_use_onnx_red = ycc_is_sr && !full_res && !ycc_use_dict &&
            photo_up.sr_red > 0.0f && ycc_red_kind >= 0;
        const std::string ycc_red_model = ycc_use_onnx_red ?
            cs_red_model_for(photo_up.sr_red_denoiser, photo_up.sr_dncnn_model,
                photo_up.sr_red_model) : std::string();
        if (ycc_use_onnx_red && !cs_onnx_available(ycc_red_model, ycc_red_kind)) {
            std::cerr << "Error: RED model failed to load ('" << ycc_red_model << "')" << std::endl;
            return -3;
        }
        if (ycc_is_direct && !full_res && !ycc_use_dict && photo_up.sr_red > 0.0f) {
            std::cerr << "Note: --red is ignored with sr-direct (single-stage solve has no RED passes)" << std::endl;
        }
        if (ycc_is_dip && !full_res && !ycc_use_dict && photo_up.sr_red > 0.0f) {
            std::cerr << "Note: --red is ignored with dip (untrained-prior solve has no RED passes)" << std::endl;
        }
        if (ycc_sr_cascade && full_res) {
            std::cerr << "Warning: --sr-init is meaningless with --full-res; ignoring" << std::endl;
        }
        const bool ycc_fuse = !dual_tv && !full_res && (photo_up.backend == "avir" || ycc_is_sr || ycc_is_direct || ycc_is_dip || ycc_sr_d ||
            photo_up.backend == "fsrcnn");
        // (the fsrcnn model path is threaded to the ycc tile solver below)
        const bool ycc_pipe =
            !dual_tv && !full_res && !ycc_fuse &&
            (photo_up.backend == "waifu2x" ||
             photo_up.backend == "waifu2x-ncnn" ||
             photo_up.backend == "realcugan");

        // Preview follows the HR grid when tiles upscale during the solve
        // (a tile appears only once upscaled), the LR grid otherwise.
        ImagePreview preview;
        std::mutex ycc_hr_grid_mutex;
        std::vector<std::vector<TileCoord>> ycc_hr_coordinates;
        if ((ycc_fuse || ycc_pipe) && use_preview) {
            ycc_hr_coordinates = coordinates;
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    ycc_hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                    ycc_hr_coordinates[i][j].y = coordinates[i][j].y * 2;
                }
            preview.Start(windowName, reconstructed, ycc_hr_coordinates,
                          ycc_hr_tiles, &ycc_hr_grid_mutex);
        }
        else if (use_preview) {
            preview.Start(windowName, reconstructed, coordinates, ycc_hr_tiles,
                          &ycc_hr_grid_mutex);
        }

        CsUpscalePipeline ycc_upipe;
        if (ycc_pipe)
            ycc_upipe.start(photo_up, decrypted, ycc_hr_tiles,
                            &ycc_hr_grid_mutex);
        // HR anchors for the fused cs-sr solve (luma tile origin maps the
        // same way as the BGR path; stride is the luma output width).
        CsHrSrCtx ycc_hr_sr_ctx;
        if (ycc_is_sr && !full_res && !ycc_use_dict && !dimgs.hr_samples.empty() &&
            (photo_up.sr_hr_weight > 0.0f ||
                (photo_up.sr_hr_red_radius > 0.0f && photo_up.sr_red > 0.0f))) {
            ycc_hr_sr_ctx.anchors = &dimgs.hr_samples;
            ycc_hr_sr_ctx.stride = cols * 2;
            ycc_hr_sr_ctx.weight = photo_up.sr_hr_weight;
            ycc_hr_sr_ctx.red_radius = photo_up.sr_hr_red_radius;
        }
        if (photo_up.sr_hr_red_radius > 0.0f && photo_up.sr_red > 0.0f &&
            !(ycc_is_sr && !full_res && !ycc_use_dict && !dimgs.hr_samples.empty())) {
            std::cerr << "Note: --hr-red-radius needs cs-sr with --hr-detail anchors; using uniform RED" << std::endl;
        }
        if (dual_tv && photo_up.backend != "avir")
            std::cerr << "Note: --y-dual-tv uses AVIR for both candidate grids" << std::endl;
        decrypt_tiles_ycc420(nun_threads, y_eph, y_idx, cr_eph, cr_idx, cb_eph, cb_idx,
            decrypted, coordinates, num_tiles, overlap, P.iterations, tile_size, P.coef, P.tv,
            P.solver, P.fista_iters, P.reweights, P.basis, P.wscale,
            std::cref(dimgs.thumb_seed), per_tile_coef, per_tile_tv,
            &ycc_hr_tiles, ycc_fuse, ycc_pipe ? &ycc_upipe : nullptr,
            (ycc_fuse || ycc_pipe || use_preview) ? &ycc_hr_grid_mutex : nullptr, chroma_sub_v, ycc_sr_d,
            (!full_res && photo_up.backend == "fsrcnn") ? photo_up.fsrcnn_model : "",
            photo_up.y_confidence, photo_up.y_dual_tv,
            dual_tv ? &low_tv_tiles : nullptr, grain_tv, use_onsager,
            ycc_is_sr && !full_res && !ycc_use_dict, photo_up.sr_red,
            photo_up.sr_red_denoiser, photo_up.sr_dncnn_model,
            ycc_is_direct && !full_res && !ycc_use_dict, ycc_sr_cascade,
            ycc_red_model, photo_up.sr_red_decay,
            ycc_is_dip && !full_res && !ycc_use_dict, ycc_hr_sr_ctx,
            Pc.iterations, Pc.coef, seg_tile_params);
        if (ycc_pipe) ycc_upipe.finish();

        if (use_preview) {
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
            if (dual_tv) {
                CsPhotoUpscalerOptions avir_up;
                cs_upscale_tiles_2x(decrypted, avir_up);
                cs_upscale_tiles_2x(low_tv_tiles, avir_up);
            } else cs_upscale_tiles_2x(decrypted, photo_up);
        }
        if (!full_res) {
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    coordinates[i][j].x *= 2;
                    coordinates[i][j].y *= 2;
                }
            ycc_blend_feather = overlap * 2;
        }

        // Plain tile paste as the blend canvas (zeros would halve every
        // pixel in the zero-overlap alpha blend; see the BGR solve_grid).
        reconstructed = reconstructImage(decrypted, coordinates);

        // shared tail with the BGR pipeline: feathered compositing in 2x
        // geometry, resize to 2x the recorded size
        reconstructed = blendTilesWithImage(decrypted, coordinates, reconstructed, 0.5f, ycc_blend_feather);

        const cv::Size final_size(full_res ? org_size.width : org_size.width * 2,
            full_res ? org_size.height : org_size.height * 2);
        if (final_size.width > 0 && final_size.height > 0 && reconstructed.size() != final_size)
            cv::resize(reconstructed, reconstructed, final_size);

        if (dual_tv) {
            cv::Mat low_composite = reconstructImage(low_tv_tiles, coordinates);
            low_composite = blendTilesWithImage(low_tv_tiles, coordinates,
                low_composite, 0.5f, ycc_blend_feather);
            if (low_composite.size() != reconstructed.size())
                cv::resize(low_composite, low_composite, reconstructed.size());
            cv::Mat edge_blended, dual_weight;
            const bool need_weight = !photo_up.y_dual_tv_mask.empty() ||
                !photo_up.dual_tv_split.empty() || !photo_up.dual_tv_cutouts.empty();
            cs_blend_y_dual_tv(low_composite, reconstructed, Ym, edge_blended,
                need_weight ? &dual_weight : nullptr);
            // Pixel-exact cutouts of the two candidates (before the move
            // below consumes edge_blended).
            if (!edge_blended.empty() && !dual_weight.empty() &&
                !photo_up.dual_tv_cutouts.empty()) {
                cv::Mat lowcut, highcut;
                if (cs_build_dual_tv_cutouts(edge_blended, low_composite,
                        dual_weight, lowcut, highcut)) {
                    const std::string low_path = photo_up.dual_tv_cutouts + "_low.png";
                    const std::string high_path = photo_up.dual_tv_cutouts + "_high.png";
                    bool ok = true;
                    if (!cv::imwrite(low_path, lowcut)) {
                        std::cerr << "Warning: --dual-tv-cutouts: failed to write "
                                  << low_path << "\n";
                        ok = false;
                    }
                    if (!cv::imwrite(high_path, highcut)) {
                        std::cerr << "Warning: --dual-tv-cutouts: failed to write "
                                  << high_path << "\n";
                        ok = false;
                    }
                    if (ok) {
                        std::cout << "Dual-TV cutouts written to " << low_path
                                  << " and " << high_path << std::endl;
                    }
                }
                else {
                    std::cerr << "Warning: --dual-tv-cutouts: degenerate blend; "
                                 "nothing written\n";
                }
            }
            if (!edge_blended.empty()) reconstructed = std::move(edge_blended);
            if (!dual_weight.empty() && !photo_up.y_dual_tv_mask.empty()) {
                cv::Mat mask_view;
                dual_weight.convertTo(mask_view, CV_8U, 255.0);
                if (cv::imwrite(photo_up.y_dual_tv_mask, mask_view))
                    std::cout << "Dual-TV blend mask written to "
                              << photo_up.y_dual_tv_mask << std::endl;
                else
                    std::cerr << "Warning: --y-dual-tv-mask: failed to write "
                              << photo_up.y_dual_tv_mask << "\n";
            }
            // Solved-tile class canvases: tiles taken mostly from the
            // low/high-TV solve, at original positions (see helper).
            if (!dual_weight.empty() && !photo_up.dual_tv_split.empty()) {
                cv::Mat lowtv, hightv;
                if (cs_build_dual_tv_class_views(reconstructed, dual_weight,
                        decrypted, coordinates, lowtv, hightv)) {
                    const std::string low_path = photo_up.dual_tv_split + "_lowtv.png";
                    const std::string high_path = photo_up.dual_tv_split + "_hightv.png";
                    bool ok = true;
                    if (!cv::imwrite(low_path, lowtv)) {
                        std::cerr << "Warning: --dual-tv-split: failed to write "
                                  << low_path << "\n";
                        ok = false;
                    }
                    if (!cv::imwrite(high_path, hightv)) {
                        std::cerr << "Warning: --dual-tv-split: failed to write "
                                  << high_path << "\n";
                        ok = false;
                    }
                    if (ok) {
                        std::cout << "Dual-TV tile classes written to " << low_path
                                  << " and " << high_path << std::endl;
                    }
                }
                else {
                    std::cerr << "Warning: --dual-tv-split: no tile split "
                                 "(degenerate blend); nothing written\n";
                }
            }
        }
            grid_out = reconstructed;
            blend_weight = active.weights(grid_out.size(), overlap);
            return 0;
        }; // solve_ycc_grid

        cv::Mat reconstructed;
        CsSegClassParams base_p;
        base_p.iterations = iterations;
        base_p.coef = coef;
        base_p.tv = tv;
        base_p.solver = solver;
        base_p.fista_iters = fista_iters;
        base_p.reweights = reweights;
        base_p.basis = basis;
        base_p.wscale = wscale;
        const std::map<int, CsSegOverride> seg_spec = seg_params.empty()
            ? std::map<int, CsSegOverride>() : cs_seg_parse_spec(seg_params);
        if (!seg_params.empty() && seg_solve_mask.empty()) {
            throw std::runtime_error("--seg-params needs --seg-solve");
        }
        if (!seg_solve_mask.empty()) {
            // Preserve each class's solver settings even on mixed tiles.
            if (photo_up.y_dual_tv > 0.0f) {
                throw std::runtime_error("--seg-solve cannot be combined with --y-dual-tv");
            }
            cv::Mat mask_raw = cv::imread(seg_solve_mask, cv::IMREAD_UNCHANGED);
            if (mask_raw.empty()) {
                throw std::runtime_error("--seg-solve: cannot load '" + seg_solve_mask + "'");
            }
            cv::Mat seg_ids = cs_segmask_resolve_ids(mask_raw, Y.rows, Y.cols);
            bool seen[256] = { false };
            for (int y = 0; y < seg_ids.rows; ++y) {
                const uint8_t* row = seg_ids.ptr<uint8_t>(y);
                for (int x = 0; x < seg_ids.cols; ++x) seen[row[x]] = true;
            }
            std::vector<int> classes;
            for (int c = 0; c < 256; ++c) if (seen[c]) classes.push_back(c);
            if (classes.empty()) {
                throw std::runtime_error("--seg-solve: mask holds no classes");
            }
            std::cout << "seg-solve: " << classes.size() << " classes, single stage" << std::endl;
            for (const auto& kv : seg_spec) {
                CsSegClassParams cp = base_p;
                cs_seg_apply_override(cp, kv.second);
                std::cout << "seg-solve: class " << kv.first
                          << " [iters=" << cp.iterations << " coef=" << cp.coef
                          << " tv=" << cp.tv << " solver=" << cp.solver
                          << " fista-iters=" << cp.fista_iters << " reweights=" << cp.reweights
                          << " basis=" << cp.basis << " wscale=" << cp.wscale << "]" << std::endl;
            }
            CsSegSinglePass segpass;
            segpass.ids = &seg_ids;
            segpass.base = base_p;
            segpass.spec = &seg_spec;
            segpass.rates_q = &dimgs.get_seg_rates_q();
            cv::Mat unused_weight;
            const int grid_rc = solve_ycc_grid(Y, Ym, Cr, Crm, Cb, Cbm, base_p, base_p,
                show_preview, reconstructed, {}, unused_weight, &segpass);
            if (grid_rc != 0) return grid_rc;
            cs_seg_calm_boundaries(reconstructed, seg_ids, Ym, classes, overlap);
        }
        else {
            cv::Mat unused_weight;
            const int grid_rc = solve_ycc_grid(Y, Ym, Cr, Crm, Cb, Cbm, base_p, base_p,
                show_preview, reconstructed, {}, unused_weight);
            if (grid_rc != 0) return grid_rc;
        }

        // Final polish (--final-polish): same whole-image up-then-down pass
        // as the BGR pipeline (misses keep the image untouched).
        if (!photo_up.final_polish.empty() && !reconstructed.empty())
            cs_final_polish_updown(reconstructed, photo_up);

        // HR-detail anchors (modes 4/9): exact native-luma replacements on
        // the final composite (see the BGR pipeline above).
        if (!dimgs.hr_samples.empty() && !reconstructed.empty()) {
            const size_t applied = cs_apply_hr_detail(reconstructed, dimgs.hr_samples);
            dimgs.hr_applied = applied;
            if (hr_applied_out) *hr_applied_out = applied;
            std::cout << "HR-detail: applied " << applied << "/"
                      << dimgs.hr_samples.size() << " anchors" << std::endl;
        }

        cv::imwrite(output_path, reconstructed);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: ycc decryption pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    return 0;
}


