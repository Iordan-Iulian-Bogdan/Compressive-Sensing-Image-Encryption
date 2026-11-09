#include "image_decryption.hpp"
#include "helper_functions.hpp"
#include "image_encryption.hpp"
#include "image_preview.hpp"
#include "cs_gpu.h"
#include "enh_net.hpp"
#include "gap_tv.hpp"
#ifdef CS_GPU_TILES
#include "gpu_decrypt.hpp"
#include "gpu_admm_broker.hpp"
#include "gpu_composite.h"
#include "gpu_calm.h"
#include "gpu_tiles.h"
#endif
#ifdef CS_FAST_PNG
#include "png_fast.hpp"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <tuple>
#include <set>
#include <utility>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>

// Multi-scale (mode 20) layers decode on threads of their own: this thread's lock, when set, is held from a layer's
// grid solve through its enhance nets, so the GPU stages run one layer at a time while the others unpack and prep.
static thread_local std::mutex* t_ms_gpu_lock = nullptr;
// Its holder: taken by the LR solve right before its device work (cs_ms_gpu_acquire; the host prep and layouts before it
// run unlocked), released after the enhance nets (cs_ms_gpu_release, also by the guard on every exit of the decode).
static thread_local std::unique_lock<std::mutex> t_ms_gpu_hold;
static void cs_ms_gpu_acquire() {
    if (t_ms_gpu_lock && !t_ms_gpu_hold.owns_lock()) t_ms_gpu_hold = std::unique_lock<std::mutex>(*t_ms_gpu_lock);
}
static void cs_ms_gpu_release() {
    if (t_ms_gpu_hold.owns_lock()) t_ms_gpu_hold.unlock();
}
// The layer this thread decodes (scale s of the W x H photo, T px scale map): the enhance nets run only on the 8x8
// blocks of the layer that the composite reads.
struct CsMsNeed { const cv::Mat* map; int T, W, H, s; };
static thread_local const CsMsNeed* t_ms_need = nullptr;

// --enhance-low: the first-stage net for a container with `bits` luma sample bits per output pixel (all layers of a
// multi-scale container together). Default multi-scale encodes give 0.19-0.29 at r0.1, 0.37-0.46 at r0.2 and
// 0.55-0.64 at r0.3 (32 photos); the low-ratio net + --enhance2 beat the standard pair below 0.43 (+0.26 dB under
// 0.3, +0.07..0.10 dB at 0.3-0.43) and lost above it (2026-11-07, docs/enhance-nets.md).
static constexpr double CS_ENHANCE_LOW_BITS = 0.43;
static const std::string& cs_enhance_pick(const CsPhotoUpscalerOptions& pu, double bits) {
    const bool low = !pu.enhance_low_model.empty() && bits < CS_ENHANCE_LOW_BITS;
    std::printf("enhance: %.3f luma sample bits per pixel, %s net\n", bits, low ? "low-ratio" : "standard");
    return low ? pu.enhance_low_model : pu.enhance_model;
}

// Flags (ceil(h / 8) x ceil(w / 8)) of the w x h layer's 8x8 blocks the composite reads: within the 17x17 feather box
// (8 px) plus the cubic resample's taps of a tile of the layer's scale.
static cv::Mat cs_ms_needed_blocks(const CsMsNeed& n, int w, int h) {
    const int th = n.map->rows, tw = n.map->cols, ah = (h + 7) / 8, aw = (w + 7) / 8;
    const double fx = (double)n.W / w, fy = (double)n.H / h;
    const int d = 8 + 2 * (int)std::ceil(std::max(fx, fy)) + 1;
    cv::Mat a(ah, aw, CV_8U);
    for (int by = 0; by < ah; ++by) {
        const int ty0 = std::max(0, (int)std::floor(8 * by * fy - d) / n.T), ty1 = std::min(th - 1, (int)std::ceil(8 * (by + 1) * fy + d) / n.T);
        for (int bx = 0; bx < aw; ++bx) {
            const int tx0 = std::max(0, (int)std::floor(8 * bx * fx - d) / n.T), tx1 = std::min(tw - 1, (int)std::ceil(8 * (bx + 1) * fx + d) / n.T);
            uint8_t v = 0;
            for (int ty = ty0; !v && ty <= ty1; ++ty)
                for (int tx = tx0; !v && tx <= tx1; ++tx) v = n.map->at<uint8_t>(ty, tx) == n.s;
            a.at<uint8_t>(by, bx) = v;
        }
    }
    return a;
}

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

// A side thread joined when its scope ends (also on exceptions).
struct CsJoinedThread {
    std::thread t;
    ~CsJoinedThread() { if (t.joinable()) t.join(); }
};
} // namespace

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
    odd_parity = info.odd_parity;

    // the sampling mode travels in the authenticated header (no decrypt-side
    // flags needed)
    sampling_mode = info.sampling_mode;
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

// Opens the sealed seg-mask body prefix [nclasses][rates][seglen][label PNG],
// caches the rates/labels and regenerates the sample indices into ri_x/ri_y.
CsSegPrefix decrypt_image::open_segmask_prefix() {
    const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
    const uint32_t pre_bytes =
        (uint32_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN] |
        ((uint32_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 1] << 8) |
        ((uint32_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 2] << 16) |
        ((uint32_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 3] << 24);
    if (pre_bytes == 0 || CS_HEADER_BYTES + (size_t)pre_bytes > total_bytes) {
        throw std::runtime_error("seg-mask container: corrupt prefix length");
    }
    std::vector<uint8_t> prefix(encrypted_img.data + CS_HEADER_BYTES,
                                encrypted_img.data + CS_HEADER_BYTES + pre_bytes);
    if (!cs_blob_seal_xor(prefix.data(), prefix.size(), cs_key,
                          encrypted_img.data + CS_OFF_IV, CS_SEG_TAG)) {
        throw std::runtime_error("seg-mask container: prefix decryption failed");
    }
    CsSegPrefix pre;
    if (!cs_segmask_parse_prefix(prefix.data(), prefix.size(), pre) ||
        pre.total != pre_bytes) {
        cs_wipe(prefix.data(), prefix.size());
        throw std::runtime_error("seg-mask container: corrupt prefix");
    }
    std::vector<uint8_t> png(prefix.data() + (pre.total - pre.seglen),
                             prefix.data() + pre.total);
    cv::Mat labels = cv::imdecode(png, cv::IMREAD_GRAYSCALE);
    cs_wipe(prefix.data(), prefix.size());
    cs_wipe(png.data(), png.size());
    if (labels.empty()) {
        throw std::runtime_error("seg-mask container: label decode failed");
    }
    if (labels.rows != rows || labels.cols != cols) {
        cv::resize(labels, labels, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
    }
    if (pre.blend != 0) {
        throw std::runtime_error("seg-mask container with --seg-blend bands: not a production format");
    }
    seg_rates_q = pre.rates_q;
    seg_ids = labels;
    returnSegMaskIndices(ri_x, ri_y, rows, cols, labels, pre.rates_q, cs_key);
    if ((int)ri_x.size() != m) {
        throw std::runtime_error("seg-mask container: count mismatch");
    }
    return pre;
}

void decrypt_image::get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
    cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask) {
    if (!cs_key_valid) {
        throw std::runtime_error("derived key not available: header must be parsed with the password first");
    }
    if (sampling_mode != CS_MODE_YCC420_SEG && sampling_mode != CS_MODE_YCC420_SEG_HRDETAIL) {
        throw std::runtime_error("not a production container (seg-luma 4:2:0, mode 11 or 19)");
    }

    // Chroma draws are uniform (tagged keys) on the 4:2:0 grid. They do not depend on the luma draw, so they run
    // alongside it. Seg prefix version 2 (multi-scale layers) draws chroma inside the label support, so that draw
    // waits for the label map (opened below); its first two (CTR-sealed) prefix bytes tell.
    const int chroma_mode = CS_MODE_YCC420;
    bool chroma_masked = false;
    if (encrypted_img.total() * encrypted_img.elemSize() >= (size_t)CS_HEADER_BYTES + 2) {
        uint8_t hd[2] = { encrypted_img.data[CS_HEADER_BYTES], encrypted_img.data[CS_HEADER_BYTES + 1] };
        if (cs_blob_seal_xor(hd, 2, cs_key, encrypted_img.data + CS_OFF_IV, CS_SEG_TAG))
            chroma_masked = hd[0] == 0 && hd[1] == 2;
        cs_wipe(hd, sizeof(hd));
    }
    ms_layer = chroma_masked;
    std::exception_ptr chroma_error;
    CsJoinedThread chroma_draw;
    if (!chroma_masked) {
        chroma_support = cv::Mat();
        chroma_draw.t = std::thread([&] {
            try { returnYcc420ChromaIndices(rows, cols, m, cs_key, chroma_mode); }
            catch (...) { chroma_error = std::current_exception(); }
        });
    }
    // Trailing mode-19 anchor payload (compact flag map + luma, hr_count = byte length): CTR-decrypt the bytes after
    // the luma/chroma sections into native-luma anchors.
    auto read_hr = [&](size_t boff, int mY, int mCh) {
        const size_t total_bytes = encrypted_img.total() * encrypted_img.elemSize();
        const size_t pay_off = (size_t)CS_HEADER_BYTES + boff + (size_t)mY + (size_t)2 * (size_t)mCh;
        const size_t pay_len = (size_t)hr_count;
        if (hr_count == 0 || pay_off + pay_len > total_bytes ||
            !cs_read_anchor_payload(encrypted_img.data + pay_off, pay_len, cs_key, encrypted_img.data + CS_OFF_IV, hr_samples)) {
            throw std::runtime_error("hr-detail container: corrupt payload");
        }
    };
    // Its offset is known before the luma draw (the prefix length sits in the header pad and the luma count is m), so
    // it is read on a side thread meanwhile.
    std::exception_ptr hr_error;
    CsJoinedThread hr_read;
    const bool packed = sample_bits != 8 || sample_bits_chroma != 8;   // unpack replaces encrypted_img below
    if (hr_detail && !packed) {
        const size_t pre_bytes =
            (size_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN] |
            ((size_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 1] << 8) |
            ((size_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 2] << 16) |
            ((size_t)encrypted_img.data[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + 3] << 24);
        const int mCh = cs_chroma_count(chroma_mode, m, rows, cols);
        hr_read.t = std::thread([&, pre_bytes, mCh] {
            try { read_hr(pre_bytes, m, mCh); }
            catch (...) { hr_error = std::current_exception(); }
        });
    }
    // The sealed body prefix [nclasses][rates][seglen][label PNG] (v3, CS_SEG_TAG) regenerates the luma draw.
    const CsSegPrefix pre = open_segmask_prefix();
    const size_t body_off = pre.total;
    const int mY_written = m;
    if (pre.chroma_masked != chroma_masked) throw std::runtime_error("seg-mask container: corrupt prefix");
    if (chroma_masked) {
        chroma_support = cs_seg_chroma_support(seg_ids, seg_rates_q, chroma_mode);
        returnYcc420ChromaIndices(rows, cols, m, cs_key, chroma_mode);
    }
    if (chroma_draw.t.joinable()) chroma_draw.t.join();
    if (chroma_error) std::rethrow_exception(chroma_error);
    const int mC = m_chroma;

    int crows, ccols;
    cs_chroma_dims(chroma_mode, rows, cols, crows, ccols);

    // Bit-packed (--sample-bits) bodies unpack first: three independent
    // sections (luma, Cr, Cb) after the raw seg prefix, then the raw anchor
    // payload. The length check below then runs against the restored raw geometry.
    if (packed) {
        const size_t y0 = (size_t)CS_HEADER_BYTES + body_off;
        std::vector<cs_body_section> sections = {
            { y0, (size_t)mY_written, sample_bits, 0, false },
            { y0 + (size_t)mY_written, (size_t)mC, sample_bits_chroma, 0, false },
            { y0 + (size_t)mY_written + (size_t)mC, (size_t)mC, sample_bits_chroma, 0, false } };
        if (hr_detail) {   // compact anchors ride raw after Cb
            sections.push_back({ y0 + (size_t)mY_written + (size_t)2 * mC, (size_t)hr_count, 8, 0, false });
        }
        unpack_container_measurements(sections);
    }

    const size_t need = (size_t)CS_HEADER_BYTES + body_off + (size_t)mY_written + (size_t)2 * mC;
    if (encrypted_img.total() * 3 < need) {
        throw std::runtime_error("ycc container: body too short for luma/chroma payload");
    }
    const uint8_t* body = encrypted_img.data + CS_HEADER_BYTES + body_off;

    if (hr_read.t.joinable()) {
        hr_read.t.join();
        if (hr_error) std::rethrow_exception(hr_error);
    }
    else if (hr_detail) {
        read_hr(body_off, mY_written, mC);
    }

    y = cv::Mat::zeros(rows, cols, CV_8U);
    y_mask = cv::Mat::zeros(rows, cols, CV_8U);
    cr = cv::Mat::zeros(crows, ccols, CV_8U);
    cr_mask = cv::Mat::zeros(crows, ccols, CV_8U);
    cb = cv::Mat::zeros(crows, ccols, CV_8U);
    cb_mask = cv::Mat::zeros(crows, ccols, CV_8U);

    // The seg-luma draws and the chroma draws pick distinct positions (partial Fisher-Yates over disjoint pools), so
    // those scatters run in parallel.
    #pragma omp parallel for
    for (int k = 0; k < mY_written; k++) {
        y.at<uint8_t>(ri_x[k], ri_y[k]) = body[k];
        y_mask.at<uint8_t>(ri_x[k], ri_y[k]) = 1;
    }
    #pragma omp parallel for
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
// the generic reference. Pixel-domain copy first (re-encoded to YCrCb), then
// the DCT with the /10 convention createRefSolutions uses, so the solvers
// accept it unchanged. Chroma warm-start planes are downsampled to the
// 4:2:0 grid of the chroma solves.
static void build_neighbor_warm_start(cv::Mat refs[3],
    const std::vector<cv::Mat>& generic_refs,
    const std::vector<std::vector<cv::Mat>>& solved,
    const std::vector<std::vector<TileCoord>>& coordinates,
    int i, int j)
{
    // pixel plane -> solver-domain coefficients (/10 convention)
    auto to_basis = [&](cv::Mat& plane) {
        cs_dct_forward(plane, cs_default_dct_block());
        plane /= 10.0f;
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
        // the solved tiles hold BGR pixels; re-encode to YCrCb so the
        // warm start lives in the same domain as the solve
        cv::cvtColor(strip, strip, cv::COLOR_BGR2YCrCb);
        cv::Mat strip32;
        strip.convertTo(strip32, CV_32FC3, 1.0 / 255.0);
        strip32.copyTo(pix(dst_local));
        any_neighbor = true;
    }

    if (!any_neighbor) {
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
        // Y gets the generic image-like ref; chroma planes start from a
        // neutral flat-chroma solution (their statistics are very different)
        refs[0] = fit(generic_refs[0]);
        const cv::Size chroma_size((want.width + 1) / 2, (want.height + 1) / 2);
        refs[1] = cv::Mat(chroma_size, CV_32F, cv::Scalar(0.5f));
        to_basis(refs[1]);
        refs[2] = refs[1].clone();
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(pix, planes);
    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat plane = planes[ch];
        if (ch > 0) {
            // match the 4:2:0 chroma solves' unknown space
            cv::resize(plane, plane, cv::Size((plane.cols + 1) / 2, (plane.rows + 1) / 2), 0, 0, cv::INTER_AREA);
        }
        to_basis(plane);
        refs[ch] = plane;
    }
}

// YCC 4:2:0 tile solver: wavefront (anti-diagonal) order, each tile solves
// luma at full resolution and each chroma plane natively on its coarse grid
// with per-channel FISTA (DCT, 2 reweights, wscale 2). Warm starts come from
// the already-solved BGR neighbors via build_neighbor_warm_start, whose ref
// layout (full-res Y, coarse chroma) matches this solve exactly. Solved
// planes merge YCrCb -> BGR per tile, and every tile goes through the cs-sr
// 2x stage (cs_sr_upscale_tile_2x, or its batched GPU form) into hr_out.
// tile_coef (per-segment decode) replaces the luma coef per tile.
static void decrypt_tiles_ycc420(int num_threads,
    const std::vector<std::vector<cv::Mat>>& y_meas,
    const std::vector<std::vector<indices>>& y_idx,
    const std::vector<std::vector<cv::Mat>>& cr_meas,
    const std::vector<std::vector<indices>>& cr_idx,
    const std::vector<std::vector<cv::Mat>>& cb_meas,
    const std::vector<std::vector<indices>>& cb_idx,
    std::vector<std::vector<cv::Mat>>& mats_out,
    const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int iterations, cv::Size tile_size, float coef, float tv, int fista_iters,
    std::vector<std::vector<cv::Mat>>& hr_out, std::mutex& hr_grid_mutex,
    const CsHrSrCtx& hr_sr, int admm_iters,
    const std::vector<std::vector<float>>* tile_coef,
    std::shared_ptr<void>* free_later,   // receives host state the caller can free off the critical path
    bool ms_layer) {   // a multi-scale layer: tiles without luma samples skip cs-sr (the composite drops them)

    constexpr int reweights = 2;
    constexpr float wscale = 2.0f;
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);
    const int iters_c = iterations - iterations / 2;
    const float coef_c = coef * 0.5f;

    // GPU mode: solver threads block on each plane solve, so threads =
    // in-flight problems = batch size. Without this the default 8 threads cap
    // every flush at 8 jobs (measured: 1731 micro-flushes of P<=8 on the
    // 64x64 grid) and the per-flush pack/H2D/launch tax outweighs the GPU
    // solve. A wave holds num_tiles tiles, so num_tiles threads fill the batch.
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
#ifdef CS_GPU_TILES
    num_threads = cs_gpu_admm_threads(num_tiles, num_threads);
#endif

    // Luma coef of a tile: its class's (per-segment decode), else the global one.
    auto coef_y_of = [&](int i, int j) { return tile_coef ? (*tile_coef)[i][j] : coef; };
    // coarse (chroma) tile from the same halving the prep used: exact cover
    auto coarse_size = [&](int i, int j) {
        const int Th = mats_out[i][j].rows, Tw = mats_out[i][j].cols;
        const int xo = coordinates[i][j].x, yo = coordinates[i][j].y;
        return cv::Size((xo + Tw + 1) / 2 - xo / 2, (yo + Th + 1) / 2 - yo / 2);
    };

    // True when every non-empty tile was solved by the GPU pre-pass: the wavefront then
    // collapses into one flat parallel loop.
    bool flat_tiles = false;
    std::vector<std::vector<char>> gpu_done((size_t)num_tiles, std::vector<char>((size_t)num_tiles, 0));
    std::vector<std::vector<cv::Mat>> sr_hr((size_t)num_tiles, std::vector<cv::Mat>((size_t)num_tiles));
    std::vector<std::vector<char>> sr_done((size_t)num_tiles, std::vector<char>((size_t)num_tiles, 0));
    // Their 2x tiles (flat as well, the bilinear 2x of the fill) share one buffer per size, read-only: the composite
    // uploads it once (gpu_composite::blend_tiles) instead of every tile.
    std::map<std::pair<int, int>, cv::Mat> ms_flat_hr;
    std::vector<std::vector<char>> ms_flat((size_t)num_tiles, std::vector<char>((size_t)num_tiles, 0));
    // A multi-scale layer's tiles without any sample (luma or chroma) lie outside its scale's tiles and their margin,
    // where the composite gives the layer no weight: a flat fill stands in for their solve (counted as solved).
    if (ms_layer)
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j)
                if (!y_meas[i][j].empty() && !mats_out[i][j].empty() && y_idx[i][j].ri_x_g.empty() &&
                    cr_idx[i][j].ri_x_g.empty() && cb_idx[i][j].ri_x_g.empty()) {
                    mats_out[i][j] = cv::Mat(mats_out[i][j].rows, mats_out[i][j].cols, CV_8UC3, cv::Scalar::all(128));
                    gpu_done[(size_t)i][(size_t)j] = 1;
                    cv::Mat& f = ms_flat_hr[{ mats_out[i][j].cols, mats_out[i][j].rows }];
                    if (f.empty()) f = cv::Mat(2 * mats_out[i][j].rows, 2 * mats_out[i][j].cols, CV_8UC3, cv::Scalar::all(128));
                    ms_flat[(size_t)i][(size_t)j] = 1;
                }
#ifdef CS_GPU_TILES
    // The batched cs-sr stage below (taken when every tile ends up GPU-solved): its tile list, and the host prep that
    // does not read the LR pixels, which runs on a side thread while the LR batches are on the GPU.
    std::vector<CsSrBatchTile> bt;
    std::vector<std::pair<int, int>> bt_where;
    for (int i = 0; i < num_tiles; ++i)
        for (int j = 0; j < num_tiles; ++j) {
            if (y_meas[i][j].empty() || mats_out[i][j].empty()) continue;
            if (ms_layer && y_idx[i][j].ri_x_g.empty()) continue;   // bilinear in the tile loop below
            CsSrBatchTile T{};
            T.lr_tile = &mats_out[i][j];
            T.meas = &y_meas[i][j];
            T.ri_x = &y_idx[i][j].ri_x_g;
            T.ri_y = &y_idx[i][j].ri_y_g;
            T.coef = coef_y_of(i, j);
            T.tv = tv;
            T.iterations = iterations;
            T.fista_iters = fista_iters;
            T.wscale = wscale;
            T.hr_sr = hr_sr;
            T.hr_sr.ox = coordinates[i][j].x * 2;
            T.hr_sr.oy = coordinates[i][j].y * 2;
            T.hr_out = &sr_hr[(size_t)i][(size_t)j];
            bt.push_back(T);
            bt_where.push_back({ i, j });
        }
    std::shared_ptr<CsSrBatchPrep> sr_prep;
    CsJoinedThread sr_prep_thread;
    // GPU pre-pass (env CS_GPU_TILES=1): Y, Cr and Cb of every eligible tile in one batch, each
    // from a smooth fill of its own samples (no neighbour warm start).
    if (std::getenv("CS_GPU_TILES")) {
        std::vector<GpuYccTileJob> jobs;
        std::vector<std::pair<int, int>> where;
        int skipped = 0;
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                if (y_meas[i][j].empty() || gpu_done[(size_t)i][(size_t)j]) continue;   // gpu_done: flat-filled above
                if (cs_default_dct_block() > 0 || tv < 0.0f) { ++skipped; continue; }
                const cv::Size cs = coarse_size(i, j);
                GpuYccTileJob J{};
                J.plane[0] = { &y_meas[i][j], mats_out[i][j].rows, mats_out[i][j].cols, &y_idx[i][j].ri_x_g, &y_idx[i][j].ri_y_g,
                    iterations, coef_y_of(i, j), tv };
                J.plane[1] = { &cr_meas[i][j], cs.height, cs.width, &cr_idx[i][j].ri_x_g, &cr_idx[i][j].ri_y_g, iters_c, coef_c, 0.0f };
                J.plane[2] = { &cb_meas[i][j], cs.height, cs.width, &cb_idx[i][j].ri_x_g, &cb_idx[i][j].ri_y_g, iters_c, coef_c, 0.0f };
                J.fista_iters = fista_iters; J.reweights = reweights; J.wscale = wscale;
                J.out = &mats_out[i][j];
                jobs.push_back(J);
                where.push_back({ i, j });
            }
        std::vector<char> solved;
        const auto g0 = std::chrono::steady_clock::now();
        cs_gpu_solve_tiles_ycc(jobs, solved, [&] {
            cs_ms_gpu_acquire();   // a multi-scale layer: the GPU stages from here on run one layer at a time
            if (!bt.empty())
                sr_prep_thread.t = std::thread([&] {
                    omp_set_num_threads(8);   // a third of the cores: the LR batches' own host work keeps the rest
                    sr_prep = cs_sr_gpu_batch_prepare(bt, admm_iters);
                });
        });
        int n_ok = 0;
        for (size_t k = 0; k < solved.size(); ++k)
            if (solved[k]) { gpu_done[(size_t)where[k].first][(size_t)where[k].second] = 1; ++n_ok; }
        std::cout << "[gpu-tiles] ycc: solved " << n_ok << " of " << jobs.size() << " eligible tiles on the GPU in "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count()
                  << " ms (no neighbour warm start); skipped " << skipped << std::endl;
        flat_tiles = std::getenv("CS_GPU_WAVES") == nullptr;   // CS_GPU_WAVES=1 keeps the wavefront (A/B)
        for (int i = 0; i < num_tiles && flat_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j)
                if (!y_meas[i][j].empty() && !gpu_done[(size_t)i][(size_t)j]) { flat_tiles = false; break; }
    }

    // Normally taken already by the LR solve's on_gpu; without one (no LR jobs, CS_GPU_TILES unset) the layer takes the
    // GPU lock here, before any cs-sr work.
    cs_ms_gpu_acquire();

    // Batched GPU cs-sr stage: the luma SR of every tile in one GPU pass. The per-tile call below
    // takes the merged BGR tile with the luma container (gray-as-BGR, so the BGR2Y weights
    // collapse to 1), which is exactly what a CsSrBatchTile carries.
    if (sr_prep_thread.t.joinable()) sr_prep_thread.t.join();
    if (flat_tiles) {
        std::vector<char> done;
        const auto s0 = std::chrono::steady_clock::now();
        cs_sr_gpu_batch(bt, admm_iters, done, sr_prep.get());
        int n_ok = 0;
        for (size_t k = 0; k < done.size(); ++k)
            if (done[k]) { sr_done[(size_t)bt_where[k].first][(size_t)bt_where[k].second] = 1; ++n_ok; }
        if (!bt.empty())
            std::cout << "[gpu-sr] ycc: " << n_ok << " of " << bt.size() << " tiles upscaled in one GPU pass in "
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count()
                      << " ms" << std::endl;
    }
    if (free_later) *free_later = std::move(sr_prep);   // ~4k tiles' sample and anchor lists
#else
    (void)free_later;
#endif

    const int n_waves = flat_tiles ? 1 : 2 * (num_tiles - 1) + 1;
    for (int w = 0; w < n_waves; ++w) {
        const int i_lo = flat_tiles ? 0 : (std::max)(0, w - (num_tiles - 1));
        const int i_hi = flat_tiles ? num_tiles - 1 : (std::min)(num_tiles - 1, w);
        const int wave_count = flat_tiles ? num_tiles * num_tiles : i_hi - i_lo + 1;

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int t = 0; t < wave_count; ++t) {
            try {
                const int i = flat_tiles ? t / num_tiles : i_lo + t;
                const int j = flat_tiles ? t % num_tiles : w - i;
                if (y_meas[i][j].empty()) continue;

                const float tile_coef_y = coef_y_of(i, j);
                const struct indices& y_tidx = y_idx[i][j];
                if (!gpu_done[(size_t)i][(size_t)j]) {
                    cv::Mat x0[3];
                    build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j);

                    const int Th = mats_out[i][j].rows;
                    const int Tw = mats_out[i][j].cols;
                    const cv::Size coarse_want = coarse_size(i, j);
                    const int cTw = coarse_want.width, cTh = coarse_want.height;
                    // neighbor-derived chroma refs are halved t_rect; parity on
                    // edge tiles can leave them 1px off the solved grid — refit
                    // (warm start only, the solve re-converges regardless)
                    for (int c = 1; c < 3; ++c) {
                        if (x0[c].empty() || x0[c].size() != coarse_want) {
                            if (x0[c].empty()) {
                                x0[c] = cv::Mat(coarse_want, CV_32F, cv::Scalar(0.5f));
                                cs_dct_forward(x0[c], cs_default_dct_block());
                                x0[c] /= 10.0f;
                            } else {
                                cv::resize(x0[c], x0[c], coarse_want, 0, 0, cv::INTER_LINEAR);
                            }
                        }
                    }

                    cv::Mat dummy;
                    // Luma at full tile resolution, chroma natively on the half-resolution grid, where proximal
                    // updates are exact at the near-flat warm starts.
                    auto solve_plane = [&](const cv::Mat& meas, float c, int R, int C,
                        const std::vector<int>& rx, const std::vector<int>& ry,
                        int it, cv::Mat& r, bool nxt, cv::Mat& nr, float t) {
                        reconstruct_color_channel_fista(meas, 0, c, R, C, rx, ry, it, r, nxt, nr, t, reweights, fista_iters, CS_BASIS_DCT, wscale);
                    };
                    // Planes start from a smooth fill of their own samples, as the GPU
                    // pre-pass does (gpu_decrypt.cpp). The neighbour start is pixels/10 in a solver
                    // whose tail is IDCT x 255, a 10x-dark start that small coefs do not converge
                    // from in the auto iteration count (IMG_3690 at coef x0.2: Cr 23.8 vs 50.8 dB).
                    auto fill_start = [&](const cv::Mat& meas, int R, int C,
                        const std::vector<int>& rx, const std::vector<int>& ry, cv::Mat& r) {
                        if (rx.empty()) return;
                        std::vector<float> b, f;
                        cs_point_sample_fill(meas, 0, R, C, rx, ry, b, f);
                        cv::Mat X(R, C, CV_32F, f.data());
                        cs_dct_forward(X, cs_default_dct_block());
                        r = X.clone();
                    };
                    fill_start(y_meas[i][j], Th, Tw, y_tidx.ri_x_g, y_tidx.ri_y_g, x0[0]);
                    fill_start(cr_meas[i][j], cTh, cTw, cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, x0[1]);
                    fill_start(cb_meas[i][j], cTh, cTw, cb_idx[i][j].ri_x_g, cb_idx[i][j].ri_y_g, x0[2]);
                    solve_plane(y_meas[i][j], tile_coef_y, Th, Tw,
                        y_tidx.ri_x_g, y_tidx.ri_y_g, iterations, x0[0],
                        false, dummy, tv);
                    solve_plane(cr_meas[i][j], coef_c, cTh, cTw,
                        cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, iters_c, x0[1],
                        cb_idx[i][j].ri_x_g.empty(), x0[2], 0.0f);   // Cb keeps its own fill when it has samples
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
                }   // CPU solve (tiles the GPU pre-pass did not take)
                if (mats_out[i][j].empty()) continue;
                // Fused 2x: the cs-sr stage refines luma-only against the tile's own LR luma
                // samples: y_meas channel 0 carries luma, and the BGR2Y weights collapse to 1.0
                // on the gray-as-BGR container (chroma comes from the AVIR upscale inside).
                cv::Mat hr;
                if (sr_done[(size_t)i][(size_t)j]) {
                    hr = sr_hr[(size_t)i][(size_t)j];   // batched GPU pass above
                }
                else if (ms_layer && y_idx[i][j].ri_x_g.empty()) {
                    // A multi-scale layer's tile outside its scale's tiles and their margin: no samples, and the
                    // composite gives it no weight, so a bilinear 2x stands in for the cs-sr solve.
                    if (ms_flat[(size_t)i][(size_t)j]) hr = ms_flat_hr.at({ mats_out[i][j].cols, mats_out[i][j].rows });
                    else cv::resize(mats_out[i][j], hr, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
                }
                else {
                    CsHrSrCtx tile_hr = hr_sr;
                    tile_hr.ox = coordinates[i][j].x * 2;
                    tile_hr.oy = coordinates[i][j].y * 2;
                    cs_sr_upscale_tile_2x(mats_out[i][j], y_meas[i][j],
                        y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g,
                        tile_coef_y, tv, iterations, fista_iters, wscale, admm_iters, hr, tile_hr);
                }
                if (hr.empty())
                    cs_upscale_2x_avir(mats_out[i][j], hr);
                // Serialize the cell write under the grid mutex (the preview reads the grid). hr is
                // this tile's own buffer (fresh, or the batched pass's), so the cell takes it without a copy.
                if (!hr.empty()) {
                    std::lock_guard<std::mutex> lk(hr_grid_mutex);
                    hr_out[i][j] = hr;
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
    bool show_preview,
    float tv,
    int fista_iters,
    const CsPhotoUpscalerOptions& photo_up
) {
#ifdef CS_GPU_TILES
    // HIP runtime start-up overlaps the container read, key derivation and unpacking; so does loading the
    // --enhance weights (their upload waits for the runtime).
    CsJoinedThread gpu_warm, enh_prefetch;
    if (std::getenv("CS_GPU_TILES")) gpu_warm.t = std::thread(gpu_tiles::warmup);
    if (!photo_up.enhance_model.empty())
        enh_prefetch.t = std::thread([&photo_up] {
            cs_enhance_prefetch(photo_up.enhance_model);
            if (!photo_up.enhance2_model.empty()) cs_enhance_prefetch(photo_up.enhance2_model);
            if (!photo_up.enhance_low_model.empty()) cs_enhance_prefetch(photo_up.enhance_low_model);
        });
#endif
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    return decrypt_image_tiled(encrypted_img_g, output_path, password, show_preview, tv, fista_iters, photo_up);
}

// Cheap single-image boundary calm for the single-stage solve: edge-aware
// bilateral correction near low-density segment frontiers, no re-solves.
// Range weights follow actual image edges rather than the approximate label
// contour, so ringing is damped without baking a misregistered contour into
// a new discontinuity. Strength fades to zero one overlap width inside the
// class; dense segments pass through untouched.
static void cs_seg_calm_boundaries(cv::Mat& img, const cv::Mat& seg_ids,
    const cv::Mat& sample_mask, const std::vector<int>& classes, int overlap) {
    if (overlap <= 0 || img.empty() || seg_ids.empty() || sample_mask.empty()) return;
    if (std::getenv("CS_NO_SEG_CALM")) return;   // diagnostic: skip the calm pass
    bool low_known = false, lowf[256] = { false };   // low-density classes, from the label-grid pass below
    // Low-density test of every class in one pass (area and sampled count per class) before any
    // full-frame work: most decodes have no low-density class and return here. On the label grid
    // when ids and mask share it and the image is 1x or 2x of it (nearest 2x replicates each label
    // 2x2, so both counts scale by 4 and every ratio is unchanged).
    if (seg_ids.type() == CV_8U && sample_mask.depth() == CV_8U) {
        const bool native = seg_ids.size() == sample_mask.size() &&
            (img.size() == seg_ids.size() || img.size() == cv::Size(2 * seg_ids.cols, 2 * seg_ids.rows));
        cv::Mat ids = seg_ids, chan = sample_mask;
        if (!native) {
            if (ids.size() != img.size()) cv::resize(ids, ids, img.size(), 0, 0, cv::INTER_NEAREST);
            if (chan.size() != img.size()) cv::resize(chan, chan, img.size(), 0, 0, cv::INTER_NEAREST);
        }
        if (chan.channels() > 1) cv::extractChannel(chan, chan, 0);
        long long area[256] = { 0 }, sampled[256] = { 0 };
        #pragma omp parallel
        {
            long long a[256] = { 0 }, s[256] = { 0 };
            #pragma omp for nowait
            for (int y = 0; y < ids.rows; ++y) {
                const uint8_t* ir = ids.ptr<uint8_t>(y);
                const uint8_t* mr = chan.ptr<uint8_t>(y);
                for (int x = 0; x < ids.cols; ++x) { ++a[ir[x]]; s[ir[x]] += mr[x] != 0; }
            }
            #pragma omp critical
            for (int c = 0; c < 256; ++c) { area[c] += a[c]; sampled[c] += s[c]; }
        }
        bool any_low = false;
        for (int c : classes)
            if (c >= 0 && c < 256 && area[c] > 0 && (double)sampled[c] / (double)area[c] < 0.4) any_low = lowf[c] = true;
        if (!any_low) return;
        low_known = true;
#ifdef CS_GPU_TILES
        // GPU pass (gpu_calm.hip): only the gated pixels, on the label grid's frontiers; CS_GPU_CALM=0 keeps the CPU pass.
        const char* gc = std::getenv("CS_GPU_CALM");
        if (!(gc && gc[0] == '0') && img.type() == CV_8UC3 && img.isContinuous() && seg_ids.isContinuous()) {
            double ms = 0.0;
            if (gpu_calm::run(img.data, img.rows, img.cols, seg_ids.data, seg_ids.rows, seg_ids.cols, lowf, overlap, &ms)) {
                int touched = 0;
                for (int c : classes) touched += (c >= 0 && c < 256 && lowf[c]) ? 1 : 0;
                std::cout << "seg-solve: boundary calm on " << touched << " low-density classes (GPU)" << std::endl;
                if (cs_profile_enabled()) std::fprintf(stderr, "profile[calm]: gpu=%.1f\n", ms);
                return;
            }
        }
#endif
    }
    PhaseLog cprof;   // CS_PROFILE=1: one profile[calm] line
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
        if (low_known && (c < 0 || c >= 256 || !lowf[c])) continue;   // dense: the label-grid pass already decided
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
    cprof.mark("bands");
    cv::Mat work_zone;
    cv::dilate(band_union, work_zone, cv::getStructuringElement(cv::MORPH_ELLIPSE, {33, 33}));
    cv::Rect crop = cv::boundingRect(work_zone);
    crop &= cv::Rect(0, 0, img.cols, img.rows);
    if (crop.empty()) return;
    cprof.mark("zone");
    if (cprof.on)
        std::fprintf(stderr, "calm: %zu low classes, crop %dx%d = %.0f%% of %dx%d, work zone %.1f%% of the frame\n",
            low_masks.size(), crop.width, crop.height, 100.0 * crop.area() / ((double)img.cols * img.rows), img.cols, img.rows,
            100.0 * cv::countNonZero(work_zone) / ((double)img.cols * img.rows));
    cv::Mat f;
    img.convertTo(f, CV_32FC3);
    cv::Mat smooth_crop;
    cv::bilateralFilter(f(crop), smooth_crop, 13, 18.0, 4.0, cv::BORDER_REPLICATE);
    cprof.mark("bilateral");
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
    cprof.mark("apply");
    if (touched > 0) {
        f.convertTo(img, CV_8UC3);
        std::cout << "seg-solve: boundary calm on " << touched << " low-density classes" << std::endl;
    }
    cprof.mark("store");
    cprof.dump("calm");
}

// Tile paste + cosine-feathered blend of the solved tiles over the plain tile paste. The GPU
// build (CS_GPU_TILES) runs the feathered blend on the device (gpu_composite, bit-identical);
// CS_GPU_COMPOSITE_VERIFY=1 also runs the CPU blend and reports any difference. Anything else
// uses blendTilesWithImage.
static cv::Mat cs_composite_tiles(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coords, int feather)
{
#ifdef CS_GPU_TILES
    if (feather > 0 && std::getenv("CS_GPU_TILES")) {
        cv::Size sz(0, 0);
        bool ok = true;
        for (size_t i = 0; i < tiles.size() && ok; ++i)
            for (size_t j = 0; j < tiles[i].size(); ++j) {
                const cv::Mat& t = tiles[i][j];
                if (t.empty()) continue;
                if (t.type() != CV_8UC3) { ok = false; break; }
                sz = cv::Size((std::max)(sz.width, coords[i][j].x + t.cols), (std::max)(sz.height, coords[i][j].y + t.rows));
            }
        if (ok && sz.width > 0 && sz.height > 0) {
            // Weight maps from the CPU's own cs_feather_weights, one per distinct placement geometry.
            std::map<std::tuple<int, int, int>, cv::Mat> wmaps;
            std::vector<gpu_composite::TileRef> refs;
            std::vector<cv::Mat> keep;
            keep.reserve(tiles.size() * (tiles.empty() ? 0 : tiles[0].size()));
            for (size_t i = 0; i < tiles.size() && ok; ++i)
                for (size_t j = 0; j < tiles[i].size(); ++j) {
                    const cv::Mat& t = tiles[i][j];
                    if (t.empty()) continue;
                    const int x = coords[i][j].x, y = coords[i][j].y;
                    if (x < 0 || y < 0 || x + t.cols > sz.width || y + t.rows > sz.height) { ok = false; break; }
                    const int touch = (x <= 0 ? 1 : 0) | (y <= 0 ? 2 : 0) | (x + t.cols >= sz.width ? 4 : 0) | (y + t.rows >= sz.height ? 8 : 0);
                    cv::Mat& wm = wmaps[std::make_tuple(t.cols, t.rows, touch)];
                    if (wm.empty())
                        wm = cs_feather_weights(t.cols, t.rows, 0, 0, t.cols, t.rows, touch & 1, touch & 2, touch & 4, touch & 8, feather);
                    keep.push_back(t.isContinuous() ? t : t.clone());
                    refs.push_back({ (int)i, (int)j, x, y, t.cols, t.rows, keep.back().data, wm.ptr<float>(0) });
                }
            cv::Mat out(sz, CV_8UC3);
            if (ok && !refs.empty() && gpu_composite::blend_tiles(refs, sz.width, sz.height, out.data)) {
                if (std::getenv("CS_GPU_COMPOSITE_VERIFY")) {
                    cv::Mat base = reconstructImage(tiles, coords);
                    cv::Mat d;
                    cv::absdiff(blendTilesWithImage(tiles, coords, base, 0.5f, feather), out, d);
                    double mx = 0;
                    cv::minMaxLoc(d.reshape(1), nullptr, &mx);
                    std::cout << "[gpu-composite verify] " << cv::countNonZero(d.reshape(1)) << " of " << d.total() * 3
                              << " values differ from blendTilesWithImage (max " << mx << ")" << std::endl;
                }
                return out;
            }
        }
    }
#endif
    cv::Mat base = reconstructImage(tiles, coords);
    return blendTilesWithImage(tiles, coords, base, 0.5f, feather);
}

// Sorted class ids present in a label map.
static std::vector<int> cs_seg_present_classes(const cv::Mat& ids) {
    bool seen[256] = { false };
    for (int y = 0; y < ids.rows; ++y) {
        const uint8_t* row = ids.ptr<uint8_t>(y);
        for (int x = 0; x < ids.cols; ++x) seen[row[x]] = true;
    }
    std::vector<int> classes;
    for (int c = 0; c < 256; ++c) if (seen[c]) classes.push_back(c);
    return classes;
}

// Per-tile luma coef of the automatic per-segment decode (a YCC-seg container carries its label map). Pure tiles
// take their own class. Mixed tiles take the highest-rate intersecting class: the dense side keeps its detail, and
// over-serving the sparse side is harmless (verified: full background at bird iterations loses nothing); ties fall
// back to majority, then the lower id. The coef is the global one scaled by the class's own sampled rate vs the
// container's achieved ratio (sqrt density rationale of a per-tile law, evaluated per segment), clamped to
// [0.5x, 2x]: sparser classes (fewer measurements) get a stronger prior, denser ones weaker. Verified: +0.2..1.8 dB /
// +0.004..0.028 SSIM and lower CIEDE2000-p95 across 4 images x 3 ratios. winners receives each tile's class (-1 for
// empty tiles). One scan per tile — negligible next to a FISTA solve.
static void cs_seg_resolve_tile_coef(const cv::Mat& ids, const std::vector<uint8_t>& rates_q, float coef,
    float base_ratio, const std::vector<std::vector<TileCoord>>& coords,
    const std::vector<std::vector<cv::Mat>>& tile_mats, int num_tiles,
    std::vector<std::vector<float>>& out, std::vector<int>& winners) {
    out.assign(num_tiles, std::vector<float>(num_tiles, coef));
    winners.assign(num_tiles * num_tiles, -1);
    if (ids.empty()) return;
    std::vector<int> counts(256, 0);
    const auto rate_of = [&](int c) {
        return (c >= 0 && c < (int)rates_q.size()) ? rates_q[c] : 0;
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
            float c = coef;
            if (best >= 0 && c > 0.0f && base_ratio > 0.0f) {
                const double rc = (double)rate_of(best) / 255.0;
                if (rc > 0.0) {
                    double f = (double)base_ratio / rc;
                    if (f < 0.5) f = 0.5;
                    if (f > 2.0) f = 2.0;
                    c = (float)((double)c * f);
                }
            }
            out[i][j] = c;
        }
    }
}

int decrypt_image::decrypt_image_tiled(
    const cv::Mat& encrypted_img_in,
    const std::string& output_path,
    const std::string& password,
    bool show_preview,
    float tv,
    int fista_iters,
    const CsPhotoUpscalerOptions& photo_up,
    cv::Mat* decoded_out
) {
    // Multi-scale container (mode 20, crypto_utils.hpp): verify, open the descriptor, decode every layer with these
    // same options, scale each to full size and blend them by the tile scale map (17 px box feather).
    if (encrypted_img_in.type() == CV_8UC3 && encrypted_img_in.isContinuous() &&
        encrypted_img_in.total() * 3 >= (size_t)CS_HEADER_BYTES && encrypted_img_in.data[CS_OFF_VERSION] == CS_VERSION &&
        encrypted_img_in.data[CS_OFF_PAD] == CS_MODE_MULTISCALE) {
        try {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const uint8_t* buf = encrypted_img_in.data;
            const size_t bytes = encrypted_img_in.total() * 3;
            uint8_t key[32];
            if (!cs_derive_key(password, buf + CS_OFF_SALT, key)) throw std::runtime_error("key derivation failed");
            const bool ok = cs_verify_header(buf, bytes, key);
            size_t dlen = 0;
            for (int b = 0; b < 4; ++b) dlen |= (size_t)buf[CS_OFF_PAD + CS_OFF_SEGPRE_LEN + b] << (8 * b);
            std::vector<uint8_t> d;
            if (ok && dlen >= 12 && CS_HEADER_BYTES + dlen <= bytes) {
                d.assign(buf + CS_HEADER_BYTES, buf + CS_HEADER_BYTES + dlen);
                if (!cs_blob_seal_xor(d.data(), d.size(), key, buf + CS_OFF_IV, CS_MS_TAG)) d.clear();
            }
            cs_wipe(key, sizeof(key));
            if (!ok) throw std::runtime_error("authentication failed (wrong password or tampered container)");
            size_t p = 0;
            const auto get = [&](int n) {
                if (p + n > d.size()) throw std::runtime_error("multi-scale: corrupt descriptor");
                uint32_t v = 0;
                for (int b = 0; b < n; ++b) v |= (uint32_t)d[p++] << (8 * b);
                return v;
            };
            if (get(1) != 1) throw std::runtime_error("multi-scale: unknown descriptor version");
            const int W = (int)get(2), H = (int)get(2), T = (int)get(2), n = (int)get(1);
            if (W < 2 || H < 2 || T < 8 || n < 1 || n > 5) throw std::runtime_error("multi-scale: corrupt descriptor");
            std::vector<int> scale(n);
            std::vector<size_t> len(n);
            for (int i = 0; i < n; ++i) { scale[i] = (int)get(1); len[i] = get(4); }
            const size_t mlen = get(4);
            if (p + mlen > d.size()) throw std::runtime_error("multi-scale: corrupt descriptor");
            const cv::Mat map = cv::imdecode(std::vector<uint8_t>(d.begin() + (ptrdiff_t)p, d.begin() + (ptrdiff_t)(p + mlen)), cv::IMREAD_GRAYSCALE);
            const int th = (H + T - 1) / T, tw = (W + T - 1) / T;
            if (map.rows != th || map.cols != tw) throw std::runtime_error("multi-scale: corrupt scale map");
            cs_wipe(d.data(), d.size());
            // Layers decoded first (all with the container's key: PBKDF2 runs once, see cs_derive_key), then the
            // composite tile by tile: a tile whose 3x3 neighbourhood holds one scale is that layer resampled; the
            // others blend their neighbours' layers with the weights of a 17x17 box over the tile-constant scale map
            // (separable on the tile grid: per row / column at most two tiles, by their share of the window).
            const auto ms = [](auto a, auto b) { return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
            std::vector<cv::Mat> decs(n);
            int layer_of[7] = { -1, -1, -1, -1, -1, -1, -1 };
            std::vector<cv::Mat> layers(n);
            size_t off = CS_HEADER_BYTES + dlen;
            for (int i = 0; i < n; ++i) {
                const int side = (int)std::lround(std::sqrt((double)len[i] / 3.0));
                if ((size_t)side * side * 3 != len[i] || off + len[i] > bytes) throw std::runtime_error("multi-scale: corrupt layer table");
                if (scale[i] < 2 || scale[i] > 6 || layer_of[scale[i]] >= 0) throw std::runtime_error("multi-scale: corrupt layer table");
                layer_of[scale[i]] = i;
                layers[i] = cv::Mat(side, side, CV_8UC3, const_cast<uint8_t*>(buf + off)).clone();
                off += len[i];
            }
            // One thread per layer: each unpacks and preps on its own, and its grid solve through the enhance nets
            // holds gpu_lock (t_ms_gpu_lock), so the GPU stages stay one layer at a time while the CPU stages overlap.
            std::mutex gpu_lock;
            std::vector<int> rc(n, -1);
            // --enhance-low is chosen once for the whole container, from the luma sample bits of all its layers.
            CsPhotoUpscalerOptions pu = photo_up;
            if (!pu.enhance_low_model.empty() && !pu.enhance_model.empty()) {
                double bits = 0.0;
                for (int i = 0; i < n; ++i) {
                    CsHeaderInfo hi;
                    if (cs_parse_header(layers[i].data, layers[i].total() * 3, password, hi)) bits += (double)hi.m * hi.sample_bits;
                    cs_wipe(hi.key, sizeof(hi.key));
                }
                pu.enhance_model = cs_enhance_pick(pu, bits / ((double)W * H));
            }
            pu.enhance_low_model.clear();
            CsJoinedThread enh_reserve;   // arena room for the biggest layer's net, while the small layers hold the GPU
            if (!pu.enhance_model.empty() && std::getenv("CS_GPU_TILES") && layer_of[2] >= 0)
                enh_reserve.t = std::thread([&] { cs_enhance_reserve(pu.enhance_model, H, W); });
            {
                std::vector<std::thread> th;
                for (int i = 0; i < n; ++i)
                    th.emplace_back([&, i] {
                        t_ms_gpu_lock = &gpu_lock;
                        const CsMsNeed need{ &map, T, W, H, scale[i] };
                        t_ms_need = std::getenv("CS_MS_ENH_ALL") ? nullptr : &need;
                        const auto tl = std::chrono::high_resolution_clock::now();
                        try {
                            rc[i] = decrypt_image_tiled(layers[i], "", password, false, tv, fista_iters, pu, &decs[i]);
                        } catch (...) { rc[i] = -1; }
                        t_ms_gpu_lock = nullptr;
                        t_ms_need = nullptr;
                        std::printf("multi-scale: %dx layer done %lld ms after the start\n", scale[i], ms(tl, std::chrono::high_resolution_clock::now()));
                    });
                for (std::thread& t : th) t.join();
            }
            for (int i = 0; i < n; ++i)
                if (rc[i] != 0 || decs[i].empty() || decs[i].type() != CV_8UC3)
                    throw std::runtime_error("multi-scale: layer " + std::to_string(scale[i]) + "x failed to decode");
            for (int y = 0; y < th; ++y)
                for (int x = 0; x < tw; ++x)
                    if (map.at<uint8_t>(y, x) > 6 || layer_of[map.at<uint8_t>(y, x)] < 0) throw std::runtime_error("multi-scale: corrupt scale map");
            const auto tc = std::chrono::high_resolution_clock::now();
            constexpr int R = 8;   // the 17x17 box (0, 4 and 16 measured worse with the encoder's 8 px margin)
            struct Share { int t0, t1; float f0; };   // window share of tile t0 (f0) and t1 (1 - f0)
            const auto shares = [&](int len_px) {
                std::vector<Share> v((size_t)len_px);
                for (int p = 0; p < len_px; ++p) {
                    int t0 = -1, t1 = -1, c0 = 0;
                    for (int k = -R; k <= R; ++k) {
                        const int t = (std::min)(len_px - 1, (std::max)(0, p + k)) / T;   // BORDER_REPLICATE
                        if (t0 < 0 || t == t0) { t0 = t; ++c0; } else t1 = t;
                    }
                    v[(size_t)p] = { t0, t1 < 0 ? t0 : t1, c0 / (2.0f * R + 1.0f) };
                }
                return v;
            };
            const std::vector<Share> sy = shares(H), sx = shares(W);
            cv::Mat out(H, W, CV_8UC3);
            #pragma omp parallel for schedule(dynamic, 4)
            for (int t = 0; t < th * tw; ++t) {
                const int ty = t / tw, tx = t % tw;
                const cv::Rect r(tx * T, ty * T, (std::min)(T, W - tx * T), (std::min)(T, H - ty * T));
                bool used[7] = {};
                int nused = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = ty + dy, xx = tx + dx;
                        if (yy < 0 || yy >= th || xx < 0 || xx >= tw) continue;
                        const int s = map.at<uint8_t>(yy, xx);
                        nused += !used[s];
                        used[s] = true;
                    }
                // Each layer in use here resampled over the tile (cubic, the pixel-centre mapping of cv::resize).
                cv::Mat res[7];
                for (int s = 2; s <= 6; ++s) {
                    if (!used[s]) continue;
                    const cv::Mat& d = decs[(size_t)layer_of[s]];
                    if (d.size() == cv::Size(W, H)) { res[s] = d(r); continue; }
                    const float fx = (float)d.cols / W, fy = (float)d.rows / H;
                    cv::Mat mx(r.height, r.width, CV_32F), my(r.height, r.width, CV_32F);
                    for (int y = 0; y < r.height; ++y)
                        for (int x = 0; x < r.width; ++x) {
                            mx.at<float>(y, x) = (r.x + x + 0.5f) * fx - 0.5f;
                            my.at<float>(y, x) = (r.y + y + 0.5f) * fy - 0.5f;
                        }
                    cv::remap(d, res[s], mx, my, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
                }
                if (nused == 1) { res[map.at<uint8_t>(ty, tx)].copyTo(out(r)); continue; }
                for (int y = 0; y < r.height; ++y) {
                    const Share& a = sy[(size_t)(r.y + y)];
                    uint8_t* o = out.ptr<uint8_t>(r.y + y) + 3 * r.x;
                    for (int x = 0; x < r.width; ++x) {
                        const Share& b = sx[(size_t)(r.x + x)];
                        const int st[4] = { map.at<uint8_t>(a.t0, b.t0), map.at<uint8_t>(a.t0, b.t1),
                                            map.at<uint8_t>(a.t1, b.t0), map.at<uint8_t>(a.t1, b.t1) };
                        const float wt[4] = { a.f0 * b.f0, a.f0 * (1 - b.f0), (1 - a.f0) * b.f0, (1 - a.f0) * (1 - b.f0) };
                        float acc[3] = { 0, 0, 0 };
                        for (int k = 0; k < 4; ++k) {
                            if (wt[k] == 0.0f) continue;
                            const uint8_t* p = res[st[k]].ptr<uint8_t>(y) + 3 * x;
                            for (int c = 0; c < 3; ++c) acc[c] += wt[k] * p[c];
                        }
                        for (int c = 0; c < 3; ++c) o[3 * x + c] = cv::saturate_cast<uint8_t>(acc[c]);
                    }
                }
            }
            std::cout << "multi-scale: composite " << ms(tc, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
#ifdef CS_FAST_PNG
            if (!output_path.empty() && !cs_write_image(output_path, out))   // parallel PNG encode, as the other modes
#else
            if (!output_path.empty() && !cv::imwrite(output_path, out))
#endif
                throw std::runtime_error("failed to write '" + output_path + "'");
            if (decoded_out) *decoded_out = out;
            std::cout << "multi-scale: " << n << " layers composited, " << ms(t0, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
            return 0;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
    }
    PhaseLog prof;
    if (fista_iters < 0 || fista_iters > 500) {
        std::cerr << "Error: fista_iters must be in [0, 500] (0 = auto-map from the iterations)" << std::endl;
        return -1;
    }

    // header parse + PBKDF2 key derivation + MAC verification happen here;
    // a wrong password or a tampered container is rejected before any solve
    decrypt_image dimgs;
    try {
        dimgs = decrypt_image(encrypted_img_in, password);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
    const int sm = dimgs.get_sampling_mode();
    if (sm != CS_MODE_YCC420_SEG && sm != CS_MODE_YCC420_SEG_HRDETAIL) {
        std::cerr << "Error: container mode " << sm << " is not a production format (this build decodes modes 19 and 20)"
                  << std::endl;
        return -2;
    }
    cv::Size org_size = dimgs.get_org_size();
    prof.mark("kdf");

    // iterations: the solver saturates by ~5 steps across compression
    // ratios 0.25..1.0 (the PSNR plateau starts at 5 everywhere, time scales
    // linearly beyond it). Keep a knee-proportional allocation with a cap of 8.
    int iterations = (int)std::ceil(5.0f / dimgs.get_compression_ratio());
    if (iterations > 8) {
        iterations = 8;
    }

    // Tile grid of ~32-px tiles (about 64x64 on a 2000 px image), the
    // regime cs_auto_reg_coefficient was fitted for. One whole-image tile
    // measured up to ~4 dB worse on some images (flat on others) and was
    // several times slower. org_size is already the solve-grid size here
    // (the output is 2x org_size), so no halving.
    int num_tiles;
    {
        const int edge = (std::max)(org_size.width, org_size.height);
        num_tiles = (std::min)(96, (std::max)(1, (edge + 16) / 32));
    }
    const int overlap = photo_up.auto_overlap;
    const int reg_tiles = num_tiles;   // grid size the regularization law sees (the square count)
    // Per-axis counts: at least the square count above, and enough that no tile
    // exceeds the batched SR stage's fast WMMA FISTA kernel (gpu_gemm.hpp k_fista_w:
    // 96 x 128 output px, i.e. 48 x 64 solve px, rows x cols), so every image shape
    // and size keeps that path. Images that already fit keep the square grid as is.
    {
        constexpr int kMaxRows = 48, kMaxCols = 64;
        const auto need = [&](int len, int max_len) {
            const int step = max_len - overlap;   // stride between tile origins
            return step > 0 ? (len - overlap + step - 1) / step : num_tiles;
        };
        dimgs.grid_rows = (std::max)(num_tiles, need(org_size.height, kMaxRows));
        dimgs.grid_cols = (std::max)(num_tiles, need(org_size.width, kMaxCols));
        if (dimgs.grid_rows != num_tiles || dimgs.grid_cols != num_tiles)
            std::cout << "tile grid: " << dimgs.grid_rows << " x " << dimgs.grid_cols
                      << " (tiles kept within the SR kernel's 96 x 128 px)" << std::endl;
        num_tiles = (std::max)(dimgs.grid_rows, dimgs.grid_cols);
    }
    const int nun_threads = omp_get_max_threads();

    // Automatic regularization (Params-test sweep): coef from the fitted law of the container's mode, tv from it
    // only when not given (tv == -1, the CLI default).
    float coef;
    {
        const float ar = dimgs.get_compression_ratio();
        const int nt = reg_tiles > 0 ? reg_tiles : 1;
        const float dc = cs_auto_reg_coefficient(sm, ar, nt);
        if (dc >= 0.0f) {
            coef = dc;
        }
        else if (ar < 0.5f) {
            coef = 0.045f;
        }
        else {
            coef = (1.0f / ar) * 0.01875f;
        }
        // Diagnostic (opt-in): CS_COEF_MULT=<f> scales the derived coef (luma, chroma and SR
        // correction follow it). Used by the lambda sweeps.
        // NLM-off decoder (cs-sr ADMM with the identity prox, CS_ADMM_PROX_OFF=1): the fitted laws over-regularize it;
        // x0.2 is the 77-photo sweep's optimum after the post-net at every ratio (x0.15-0.2 plateau, +0.20 dB).
        // An explicit CS_COEF_MULT replaces this factor.
        const char* prox_off = std::getenv("CS_ADMM_PROX_OFF");
        const bool nlm_off = prox_off && prox_off[0] == '1';
        if (const char* cm = std::getenv("CS_COEF_MULT"); cm && *cm) coef *= (float)std::atof(cm);
        else if (nlm_off) coef *= 0.2f;
        if (tv == -1.0f) {
            const float dtv = cs_auto_reg_tv(sm, ar);
            tv = (dtv >= 0.0f) ? dtv : 0.0f;
        }
    }

    // Diagnostic (opt-in): CS_HR_SIDECAR=<file> loads HR anchors from a side file ([u32 n] then n x {u32 idx, u8 y};
    // idx in output geometry) when the container carries none. Used to test anchors that enter the cs-sr solve.
    if (const char* sc = std::getenv("CS_HR_SIDECAR"); sc && *sc && dimgs.hr_samples.empty()) {
        if (FILE* f = std::fopen(sc, "rb")) {
            uint32_t n = 0;
            if (std::fread(&n, 4, 1, f) == 1 && n < (1u << 26)) {
                dimgs.hr_samples.resize(n);
                for (uint32_t k = 0; k < n; ++k) {
                    uint32_t idx = 0; uint8_t y = 0;
                    if (std::fread(&idx, 4, 1, f) != 1 || std::fread(&y, 1, 1, f) != 1) { dimgs.hr_samples.resize(k); break; }
                    dimgs.hr_samples[k].idx = idx; dimgs.hr_samples[k].y = y;
                }
            }
            std::fclose(f);
            std::cout << "HR sidecar: " << dimgs.hr_samples.size() << " anchors" << std::endl;
        }
    }

    // Luma at full resolution, chroma natively on the 4:2:0 grid.
    return decrypt_image_tiled_ycc420(dimgs, output_path, num_tiles, overlap,
        iterations, nun_threads, coef, show_preview, tv, fista_iters, photo_up, decoded_out);
}

int decrypt_image::decrypt_image_tiled_ycc420(decrypt_image& dimgs,
    const std::string& output_path, int num_tiles, int overlap, int iterations,
    int nun_threads, float coef, bool show_preview, float tv, int fista_iters,
    const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out) {
    PhaseLog prof;   // CS_PROFILE=1: one profile[ycc] line
    try {
        const cv::Size org_size = dimgs.get_org_size();
        const std::string windowName = output_path.empty() ? "decrypted tiles" : output_path;

        // per-plane sampled images + 0/1 masks (luma full-res, chroma coarse)
        cv::Mat Y, Ym, Cr, Crm, Cb, Cbm;
        dimgs.get_sampled_ycc420(Y, Ym, Cr, Crm, Cb, Cbm);
        prof.mark("unpack");
        // Diagnostic dumps (opt-in, for the enhance net's training data). A value of "*" writes next to the output
        // (<output path>.<suffix>), so a batch decode keeps one set per image.
        const auto dump_path = [&](const char* var, const char* suffix) -> std::string {
            const char* v = std::getenv(var);
            if (!v || !*v) return {};
            return std::strcmp(v, "*") == 0 ? (output_path.empty() ? std::string() : output_path + "." + suffix) : std::string(v);
        };
        const auto write_mask = [](const std::string& path, const cv::Mat& M) {   // 0 / 255
            cv::Mat m = M, out;
            if (m.channels() > 1) cv::extractChannel(m, m, 0);
            double mx = 0.0;
            cv::minMaxLoc(m, nullptr, &mx);
            m.convertTo(out, CV_8U, mx > 0.0 ? 255.0 / mx : 1.0);
            cv::imwrite(path, out);
        };
        // CS_DUMP_YMASK=<png> writes the luma sample mask (LR grid, 0 / 255).
        if (const std::string dump = dump_path("CS_DUMP_YMASK", "ymask.png"); !dump.empty() && !Ym.empty()) write_mask(dump, Ym);
        // CS_DUMP_YSAMP=<png> writes the luma sample values (LR grid, 8 bit, 0 where unsampled).
        if (const std::string dump = dump_path("CS_DUMP_YSAMP", "ysamp.png"); !dump.empty() && !Y.empty())
            cv::imwrite(dump, Y);
        // CS_DUMP_CHROMA=<prefix> writes <prefix>crmask.png, crsamp.png, cbmask.png, cbsamp.png (chroma grid, as above).
        if (const std::string dump = dump_path("CS_DUMP_CHROMA", ""); !dump.empty() && !Crm.empty() && !Cbm.empty()) {
            write_mask(dump + "crmask.png", Crm); cv::imwrite(dump + "crsamp.png", Cr);
            write_mask(dump + "cbmask.png", Cbm); cv::imwrite(dump + "cbsamp.png", Cb);
        }
        // CS_DUMP_HR=<file> writes the container's HR anchors in the CS_HR_SIDECAR format.
        if (const std::string dump = dump_path("CS_DUMP_HR", "anchors.bin"); !dump.empty() && !dimgs.hr_samples.empty()) {
            if (FILE* f = std::fopen(dump.c_str(), "wb")) {
                const uint32_t n = (uint32_t)dimgs.hr_samples.size();
                std::fwrite(&n, 4, 1, f);
                for (const auto& s : dimgs.hr_samples) {
                    const uint32_t idx = (uint32_t)s.idx; const uint8_t y = s.y;
                    std::fwrite(&idx, 4, 1, f); std::fwrite(&y, 1, 1, f);
                }
                std::fclose(f);
            }
        }

        // Experimental (opt-in): anchors as LR information. CS_ANCHOR_LR=1 promotes each unsampled 2x2 block whose four
        // HR pixels are all anchors to an LR luma sample (their rounded mean; an LR sample is the 2x2 box mean of the
        // luma), so the LR FISTA stage sees it. CS_ANCHOR_DERIVE=1 adds the fourth pixel of a sampled block holding three
        // anchors as an extra anchor (4 * sample - their sum, clamped). The dumps above and the restoration net keep the
        // container's own samples and anchors, which the net was trained on.
        cv::Mat Y_net = Y, Ym_net = Ym;
        std::vector<CsHrSample> hr_net;
        {
            const char* elr = std::getenv("CS_ANCHOR_LR");
            const char* eder = std::getenv("CS_ANCHOR_DERIVE");
            const bool alr = elr && elr[0] == '1', ader = eder && eder[0] == '1';
            if ((alr || ader) && !dimgs.hr_samples.empty() && Y.type() == CV_8U && Ym.type() == CV_8U) {
                Y_net = Y.clone(); Ym_net = Ym.clone();
                if (ader) hr_net = dimgs.hr_samples;
                const int lrows = Y.rows, lcols = Y.cols, W = lcols * 2;
                const size_t nb = (size_t)lrows * lcols;
                std::vector<uint8_t> cnt(nb, 0), bits(nb, 0);
                std::vector<uint16_t> sum(nb, 0);
                for (const auto& s : dimgs.hr_samples) {
                    const int r = (int)(s.idx / (uint32_t)W), c = (int)(s.idx % (uint32_t)W);
                    if (r / 2 >= lrows) continue;
                    const size_t b = (size_t)(r / 2) * lcols + c / 2;
                    ++cnt[b]; sum[b] += s.y; bits[b] |= (uint8_t)(1 << ((r & 1) * 2 + (c & 1)));
                }
                size_t promoted = 0, derived = 0;
                for (size_t b = 0; b < nb; ++b) {
                    const int br = (int)(b / lcols), bc = (int)(b % lcols);
                    uint8_t& m = Ym.at<uint8_t>(br, bc);
                    if (alr && cnt[b] == 4 && !m) {
                        Y.at<uint8_t>(br, bc) = (uint8_t)((sum[b] + 2) / 4); m = 1; ++promoted;
                    }
                    else if (ader && cnt[b] == 3 && m) {
                        int slot = 0;
                        while (bits[b] & (1 << slot)) ++slot;
                        const int v = 4 * (int)Y.at<uint8_t>(br, bc) - (int)sum[b];
                        CsHrSample s;
                        s.idx = (uint32_t)(br * 2 + (slot >> 1)) * (uint32_t)W + (uint32_t)(bc * 2 + (slot & 1));
                        s.y = (uint8_t)std::clamp(v, 0, 255);
                        dimgs.hr_samples.push_back(s); ++derived;
                    }
                }
                if (derived)
                    std::sort(dimgs.hr_samples.begin(), dimgs.hr_samples.end(),
                        [](const CsHrSample& a, const CsHrSample& b) { return a.idx < b.idx; });
                std::cout << "anchor-lr: " << promoted << " LR samples promoted, " << derived << " anchors derived" << std::endl;
            }
        }

        // The grid solve's per-tile containers (tens of thousands of small buffers) are freed on
        // this thread while the calm pass and the write run; joined when the decode returns.
        CsJoinedThread tile_frees;
        struct MsGpuGuard { ~MsGpuGuard() { cs_ms_gpu_release(); } } ms_gpu_guard;   // t_ms_gpu_lock, from the grid solve through the enhance nets
        // Automatic per-segment decode: the container carries its label map, so each tile takes a luma
        // coef scaled by its own class's sampled rate (chroma keeps the global coef at 0.5x).
        // CS_NO_SEG_PERCLASS=1 forces the global coef.
        const bool autoseg = !dimgs.get_seg_ids().empty() && !std::getenv("CS_NO_SEG_PERCLASS");
        if (autoseg)
            std::cout << "seg-solve: per-class auto params ("
                      << (int)dimgs.get_seg_rates_q().size() << " classes)" << std::endl;

        cv::Mat reconstructed;
        {
        const int rows = Y.rows, cols = Y.cols;

        // luma tiling reuses the standard splitter; chroma tiles cover the
        // matching 4:2:0 rects (the solver uses the same formula, the two must agree)
        std::vector<std::vector<cv::Mat>> y_tiles, y_mask_tiles;
        std::vector<std::vector<TileCoord>> coordinates, mask_coords;
        const int grid_r = dimgs.grid_rows > 0 ? dimgs.grid_rows : num_tiles;
        const int grid_c = dimgs.grid_cols > 0 ? dimgs.grid_cols : num_tiles;
        splitImageIntoTiles(Y, y_tiles, coordinates, grid_r, grid_c, overlap);
        splitImageIntoTiles(Ym, y_mask_tiles, mask_coords, grid_r, grid_c, overlap);
        std::vector<std::vector<float>> seg_coef;
        if (autoseg) {
            std::vector<int> seg_winners;
            cs_seg_resolve_tile_coef(dimgs.get_seg_ids(), dimgs.get_seg_rates_q(), coef, dimgs.get_compression_ratio(),
                coordinates, y_tiles, num_tiles, seg_coef, seg_winners);
            std::vector<int> per_class(256, 0);
            for (int v : seg_winners) if (v >= 0) per_class[v]++;
            std::cout << "seg-solve: single stage, per-class tile counts:";
            for (int c = 0; c < 256; ++c)
                if (per_class[c] > 0) std::cout << " " << c << "=" << per_class[c];
            std::cout << std::endl;
        }

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
        // Kept tiles first, then the per-tile prep in parallel; tile_size is the last kept
        // tile's, as in the serial order.
        std::vector<char> kept_tile((size_t)num_tiles * num_tiles, 0);
        for (int i = 0; i < num_tiles; i++)
            for (int j = 0; j < num_tiles; j++) {
                if (y_tiles[i][j].empty() || y_mask_tiles[i][j].empty()) continue;
                kept_tile[(size_t)i * num_tiles + j] = 1;
                tile_size = y_tiles[i][j].size();
            }
        std::exception_ptr prep_error;
        #pragma omp parallel for schedule(dynamic, 8)
        for (int t = 0; t < num_tiles * num_tiles; t++) {
            if (!kept_tile[(size_t)t]) continue;
            const int i = t / num_tiles, j = t % num_tiles;
            try {
                const int x = coordinates[i][j].x, y0 = coordinates[i][j].y;
                const int w = y_tiles[i][j].cols, h = y_tiles[i][j].rows;
                const int cx = x / 2, cy = y0 / 2;
                const int cw = (x + w + 1) / 2 - x / 2;
                const int ch = (y0 + h + 1) / 2 - y0 / 2;
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
                y_idx[i][j] = { ylx, yly };
                cr_img.encrypt(crlx, crly);
                cb_img.encrypt(cblx, cbly);
                y_eph[i][j] = y_img.get_mat();
                cr_eph[i][j] = cr_img.get_mat();
                cb_eph[i][j] = cb_img.get_mat();
                cr_idx[i][j] = { crlx, crly };
                cb_idx[i][j] = { cblx, cbly };

                decrypted[i][j] = cv::Mat::zeros(h, w, CV_8UC3);
            }
            catch (...) {
                #pragma omp critical
                if (!prep_error) prep_error = std::current_exception();
            }
        }
        if (prep_error) std::rethrow_exception(prep_error);
        prof.mark("prep");

        reconstructed = cv::Mat::zeros(rows, cols, CV_8UC3);

        // Every tile is upscaled 2x (cs-sr) during the solve; the preview follows the HR grid
        // (a tile appears only once upscaled).
        std::vector<std::vector<cv::Mat>> ycc_hr_tiles(
            num_tiles, std::vector<cv::Mat>(num_tiles));
        ImagePreview preview;
        std::mutex ycc_hr_grid_mutex;
        std::vector<std::vector<TileCoord>> ycc_hr_coordinates;
        if (show_preview) {
            ycc_hr_coordinates = coordinates;
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    ycc_hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                    ycc_hr_coordinates[i][j].y = coordinates[i][j].y * 2;
                }
            preview.Start(windowName, reconstructed, ycc_hr_coordinates,
                          ycc_hr_tiles, &ycc_hr_grid_mutex);
        }

        // HR anchors for the fused cs-sr solve (stride is the luma output width).
        CsHrSrCtx ycc_hr_sr_ctx;
        if (!dimgs.hr_samples.empty() && photo_up.sr_hr_weight > 0.0f) {
            ycc_hr_sr_ctx.anchors = &dimgs.hr_samples;
            ycc_hr_sr_ctx.stride = cols * 2;
            ycc_hr_sr_ctx.weight = photo_up.sr_hr_weight;
        }
        std::shared_ptr<void> sr_free_later;   // freed on tile_frees with the tile containers
        decrypt_tiles_ycc420(nun_threads, y_eph, y_idx, cr_eph, cr_idx, cb_eph, cb_idx,
            decrypted, coordinates, num_tiles, iterations, tile_size, coef, tv, fista_iters,
            ycc_hr_tiles, ycc_hr_grid_mutex, ycc_hr_sr_ctx, photo_up.sr_pnp_iters,
            autoseg ? &seg_coef : nullptr, &sr_free_later, dimgs.ms_layer);
        prof.mark("solve");

        if (show_preview) {
            preview.Stop();
        }

        // Fused tiles are adopted (AVIR fallback for misses); blending happens in 2x solve geometry.
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                if (ycc_hr_tiles[i][j].empty() && !decrypted[i][j].empty())
                    cs_upscale_2x_avir(decrypted[i][j], ycc_hr_tiles[i][j]);
            }
        decrypted = std::move(ycc_hr_tiles);
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                coordinates[i][j].x *= 2;
                coordinates[i][j].y *= 2;
            }

        // Feathered compositing in 2x geometry over a plain tile paste (zeros would halve every
        // pixel in the zero-overlap alpha blend), on the GPU when available; then resize to 2x
        // the recorded size
        reconstructed = cs_composite_tiles(decrypted, coordinates, overlap * 2);
        prof.mark("composite");

        const cv::Size final_size(org_size.width * 2, org_size.height * 2);
        if (final_size.width > 0 && final_size.height > 0 && reconstructed.size() != final_size)
            cv::resize(reconstructed, reconstructed, final_size);

        if (tile_frees.t.joinable()) tile_frees.t.join();
        tile_frees.t = std::thread([a = std::move(y_tiles), b = std::move(y_mask_tiles), c = std::move(cr_tiles),
            d = std::move(cr_mask_tiles), e = std::move(cb_tiles), f = std::move(cb_mask_tiles),
            g = std::move(y_idx), h = std::move(cr_idx), i = std::move(cb_idx), j = std::move(y_eph),
            k = std::move(cr_eph), l = std::move(cb_eph), m = std::move(decrypted),
            o = std::move(sr_free_later)] {});
        }

        // Frontier calm (luma mask Ym): the per-class decode otherwise leaves the low-density
        // boundary bands uncalmed (halo where the class coef steps).
        if (autoseg) {
            const cv::Mat& sid = dimgs.get_seg_ids();
            cs_seg_calm_boundaries(reconstructed, sid, Ym, cs_seg_present_classes(sid), overlap);
        }

        prof.mark("calm");
        prof.mark("polish");
        // Restoration CNN (--enhance): on the decode it was trained on (no
        // post-hoc anchor replacement). The old whole-image write-back below
        // measured worse after it (IMG_3690: -0.12 dB, dE-p95 +0.18: its 8-bit
        // YCrCb round trip moves every pixel); cs_enh_project changes only the
        // measured components. A load/run failure leaves the image as is.
        bool enhanced = false;
        if (!photo_up.enhance_model.empty() && !reconstructed.empty()) {
            CsEnhInputs ein;
            ein.y_mask = &Ym_net; ein.y_samp = &Y_net; ein.anchors = hr_net.empty() ? &dimgs.hr_samples : &hr_net;
            ein.cr_mask = &Crm; ein.cr_samp = &Cr; ein.cb_mask = &Cbm; ein.cb_samp = &Cb;
            ein.y_bits = dimgs.get_sample_bits(); ein.c_bits = dimgs.get_sample_bits_chroma();
            // Multi-scale layer: the nets run on the blocks the composite reads; the first net on those dilated by the
            // second's receptive field (its input there must be exact too).
            cv::Mat need, need1;
            if (t_ms_need) {
                need = cs_ms_needed_blocks(*t_ms_need, reconstructed.cols, reconstructed.rows);
                const int rf2 = photo_up.enhance2_model.empty() ? 0 : cs_enhance_receptive_field(photo_up.enhance2_model);
                if (rf2 > 0) {
                    const int k = 2 * ((rf2 + 7) / 8) + 1;
                    cv::dilate(need, need1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
                } else if (photo_up.enhance2_model.empty()) need1 = need;
                ein.active = need1.empty() ? nullptr : &need1;
            }
            const std::string& net1 = photo_up.enhance_low_model.empty() ? photo_up.enhance_model
                : cs_enhance_pick(photo_up, (double)dimgs.get_m() * dimgs.get_sample_bits() / reconstructed.total());
            enhanced = cs_enhance_net(reconstructed, net1, ein);
            ein.active = need.empty() ? nullptr : &need;
            // Data consistency: the net's output back onto the container's samples and anchors (all 32 test decodes
            // better, +0.04 dB at r0.3 to +0.10 dB at r0.9, dE-p95 -0.02 to -0.05).
            if (enhanced) cs_enh_project(reconstructed, ein);
            // Second unrolled stage (--enhance2): its planes are the residuals of the same measurements against the
            // projected first-stage output (the trainer's enh_dc.replanes), then the projection again.
            if (enhanced && !photo_up.enhance2_model.empty() && cs_enhance_net(reconstructed, photo_up.enhance2_model, ein))
                cs_enh_project(reconstructed, ein);
        }

        prof.mark("enhance");
        cs_ms_gpu_release();
        // HR-detail anchors (modes 4/9): exact native-luma replacements on
        // the final composite (see the BGR pipeline above); skipped after the net.
        if (!dimgs.hr_samples.empty() && !reconstructed.empty() && !enhanced && !std::getenv("CS_HR_NO_POSTHOC")) {
            const size_t applied = cs_apply_hr_detail(reconstructed, dimgs.hr_samples);
            dimgs.hr_applied = applied;
            std::cout << "HR-detail: applied " << applied << "/"
                      << dimgs.hr_samples.size() << " anchors" << std::endl;
        }

        if (decoded_out) *decoded_out = reconstructed;   // in-memory caller (encoder anchor pass): padded geometry
        else {
        dimgs.crop_odd_parity(reconstructed);
#ifdef CS_FAST_PNG
        cs_write_image(output_path, reconstructed);   // parallel PNG encode
#else
        cv::imwrite(output_path, reconstructed);
#endif
        }
        prof.mark("anchors+write");
        prof.dump("ycc");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: ycc decryption pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    return 0;
}

