#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"
#include "photo_upscaler.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

struct cs_coupled_dict; // cs_dict.hpp (pointer only: model handle)

/** @brief streaming upscale pipeline: solver workers enqueue (i,j) as tiles
solve; one background thread upscales tiles one at a time (one serialized
subprocess per tile via cs_upscale_one_tile_2x) into hr while solving
continues, so upscale latency hides behind the solve instead of running as
a post-join batch. The LR grid is never mutated (warm-start reads stay
safe); HR cells are written once by the consumer and adopted post-join
exactly like the fused-AVIR path. finish() joins the worker; the
destructor also joins, so an early throw cannot std::terminate.
*/
struct CsUpscalePipeline {
    const CsPhotoUpscalerOptions* opt = nullptr;
    const std::vector<std::vector<cv::Mat>>* lr = nullptr;
    std::vector<std::vector<cv::Mat>>* hr = nullptr;
    // Serializes HR cell writes (consumer) against HR grid reads (preview).
    // Either owned here or bound to the caller's hr_grid_mutex via start();
    // never nested with m below (taken separately, in either order).
    std::mutex owned_grid_m;
    std::mutex* grid_m = nullptr;
    std::mutex m;
    std::condition_variable cv;
    std::vector<std::pair<int, int>> q;
    bool done = false;
    bool running = false;
    std::thread worker;
    long upscaled = 0;
    long fallback = 0;

    ~CsUpscalePipeline() { finish(); }

    void start(const CsPhotoUpscalerOptions& o,
               const std::vector<std::vector<cv::Mat>>& lr_tiles,
               std::vector<std::vector<cv::Mat>>& hr_tiles,
               std::mutex* grid_mutex = nullptr) {
        opt = &o;
        lr = &lr_tiles;
        hr = &hr_tiles;
        grid_m = grid_mutex ? grid_mutex : &owned_grid_m;
        running = true;
        worker = std::thread([this] { loop(); });
    }

    void enqueue(int i, int j) {
        std::lock_guard<std::mutex> lk(m);
        if (!running) return;
        q.emplace_back(i, j);
        cv.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            if (!running) return;
            done = true;
            cv.notify_all();
        }
        if (worker.joinable()) worker.join();
        running = false;
    }

    void loop() {
        for (;;) {
            int i = -1, j = -1;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return done || !q.empty(); });
                if (!q.empty()) {
                    const auto p = q.back();
                    q.pop_back();
                    i = p.first;
                    j = p.second;
                }
                else if (done) {
                    break;
                }
                else {
                    continue;
                }
            }
            try {
                cv::Mat out;
                const bool ok =
                    cs_upscale_one_tile_2x((*lr)[i][j], out, *opt);
                if (!out.empty()) {
                    {
                        std::lock_guard<std::mutex> gl(*grid_m);
                        (*hr)[i][j] = std::move(out);
                    }
                    std::lock_guard<std::mutex> lk(m);
                    if (ok) ++upscaled;
                    else ++fallback;
                }
            }
            catch (...) {
                // leave the cell empty; the post-join adoption pass
                // AVIR-fills any tile the pipeline missed
            }
        }
    }
};

class decrypt_image : CSencryption {
private:
    /** @brief shared header parsing + MAC verification for both constructors.
    Throws std::runtime_error on legacy containers, undecryptable headers or
    failed authentication (v2). On success fills m/rows/cols/org_size and,
    for v2 containers, caches the derived key for returnRandomIndices.
    */
    void parse_container_header(const cv::Mat& input, const std::string& password);

    // --sample-bits depths restored from the header (8/8 = legacy unpacked)
    int sample_bits = 8;
    int sample_bits_chroma = 8;

    // Restores the raw measurement layout from a bit-packed (--sample-bits)
    // container: each section unpacks independently into a fresh raw-size Mat
    // that replaces the packed container, so every downstream reader runs
    // unchanged. Sections are (raw byte offset, value count) in ascending
    // order. No-op at 8 bits.
    void unpack_container_measurements(const std::vector<cs_body_section>& sections);

    // Decrypts + parses the seg-mask body prefix, caches seg_rates_q/seg_ids and
    // fills ri_x/ri_y; shared by the BGR and YCC-seg sampling paths.
    CsSegPrefix open_segmask_prefix();

    // Shipped per-class quantized rates of a seg container (empty for other
    // modes); cached during sampling so the single-stage solver can break
    // mixed-tile votes toward the higher sampled class.
    std::vector<uint8_t> seg_rates_q;

    // Solve-geometry label map of a seg container (empty for other modes);
    // cached during sampling so the default BGR decode can derive per-class
    // solver coef/tv from each class's own sampled rate without --seg-solve.
    cv::Mat seg_ids;

    // Mode 14 (CS_MODE_TMEANS) global group list in container-body order:
    // tgrp_off[g]..tgrp_off[g+1] index tgrp_px (flat row-major pixel
    // indices), tgrp_w holds the unit-norm row weights (0.5f). Empty for
    // point-sample modes, whose per-tile lists stay in ri_x_g/ri_y_g.
    std::vector<int> tgrp_off, tgrp_px;
    std::vector<float> tgrp_w;
    // Per-group BGR values in body order (mode 14), for per-tile
    // mini-containers in the prep loop.
    std::vector<cv::Vec3b> tgrp_val;

public:
    decrypt_image(const std::string input_path, const std::string& password = "");

    decrypt_image() {}

    decrypt_image(cv::Mat input, const std::string& password = "");

    void decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, float tv = 0.0f, int solver = CS_SOLVER_FISTA, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, bool use_onsager = false, const cs_meas_groups* groups = nullptr);

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    /** @brief extracts the sampled pixels and their coordinates. Requires the
    derived key cached by the constructor (v2 containers authenticated with
    the password); index regeneration uses the cached 256-bit key.
    */
    void get_sampled_mat(cv::Mat& sampled_mat, cv::Mat& masked_mat);

    float get_compression_ratio();

    cv::Size get_org_size();

    int get_sampling_mode() const { return sampling_mode; }
    const std::vector<uint8_t>& get_seg_rates_q() const { return seg_rates_q; }
    const cv::Mat& get_seg_ids() const { return seg_ids; }
    // Mode 14 group accessors (empty unless get_sampled_mat ran on a
    // T-means container); per-tile code filters these by tile rect.
    const std::vector<int>& get_tgrp_off() const { return tgrp_off; }
    const std::vector<int>& get_tgrp_px() const { return tgrp_px; }
    const std::vector<float>& get_tgrp_w() const { return tgrp_w; }
    const std::vector<cv::Vec3b>& get_tgrp_val() const { return tgrp_val; }
    bool has_hr_detail() const { return hr_detail; }
    const std::vector<CsHrSample>& get_hr_samples() const { return hr_samples; }
    size_t get_hr_applied() const { return hr_applied; }
    int get_sample_bits() const { return sample_bits; }
    int get_sample_bits_chroma() const { return sample_bits_chroma; }
    int get_lod_full_threshold() const { return lod_full_threshold; }
    bool get_thumb_seed(cv::Mat& out) const {
        if (thumb_seed.empty()) return false;
        thumb_seed.copyTo(out);
        return true;
    }

    /** @brief unpacks a split-color container into per-plane
    sampled images + binary masks: luma at full resolution (CV_8U), each
    chroma plane on its coarse grid (CV_8U; 4:2:0 halves both axes, 4:2:2
    halves the width only). Masks hold 0/1.
    Regenerates the identical index sets from (key, m, geometry). Supports
    uniform and HF-focus YCC420 plus uniform/adaptive-luma YCC422 containers.
    */
    void get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
        cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask);

    /** @brief decrypts a given image and writes the result to disk
    @param input_path:  path to input image
    @param output_path:  path to output image
    @param password:  password to be used for encryption and decryption
    @param parameters_type: use AUTO_PARAM for automatically choosing parameters or MANUAL_PARAM for fine tuning
    @param num_tiles : number of tiles to be processed
    @param overlap : how many pixels should the tiles overlap, bigger number may result in better quality
    @param iterations : number of iterations, higher value may result in better quality
    @param nun_threads : number of CPU threads
    @param coef : coeficient used for the solver, higher value should be used for a more compressed image
    */
    /** @brief decrypts tiles from an already-loaded encrypted image (in-memory, no PNG roundtrip).
    Shared implementation for both overloads. Note: the Mat is const but the impl needs a copy.
    */
    static int decrypt_image_tiled(const cv::Mat& encrypted_img_in, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_FISTA, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false, float grain_tv = 0.0f, bool use_onsager = false, size_t* hr_applied_out = nullptr, const std::string& seg_solve_mask = "", const std::string& seg_params = "");

    static int decrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_FISTA, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false, float grain_tv = 0.0f, bool use_onsager = false, const std::string& seg_solve_mask = "", const std::string& seg_params = "");

    /** @brief YCC 4:2:0 tile pipeline for mode-3 containers (auto-selected by
    decrypt_image_tiled from the header; never called directly with other
    modes). dimgs is the already-parsed container (key cached); remaining
    args mirror decrypt_image_tiled. Solves luma at full tile resolution and
    each chroma plane natively on its half-resolution grid, then merges
    YCrCb -> BGR per tile. FISTA_JOINT degrades to per-channel FISTA.
    */
    static int decrypt_image_tiled_ycc420(decrypt_image& dimgs, const std::string& output_path, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_FISTA, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false, float grain_tv = 0.0f, bool use_onsager = false, size_t* hr_applied_out = nullptr, const std::string& seg_solve_mask = "", const std::string& seg_params = "", bool coef_auto = false);

};

/** @brief solves all tiles in wavefront (anti-diagonal) order: wave w = i + j,
all tiles of a wave run in parallel and every tile is warm-started from the
already-solved west/north neighbors' overlap strips (falling back to the
generic reference solution when no neighbor exists, e.g. the first wave).
*/
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, float tv, int solver, int fista_iters, int reweights, int basis, float wscale, const cv::Mat& thumbnail_seed, bool per_tile_coef, bool per_tile_tv, std::vector<std::vector<cv::Mat>>* hr_out, bool fuse_upscale, CsUpscalePipeline* pipe,     std::mutex* hr_grid_mutex, bool superres_2x = false, float sr_red = 0.0f, const cs_coupled_dict* sr_dict = nullptr,
    const std::string& sr_red_denoiser = "nlmeans", const std::string& sr_dncnn_model = "models/dncnn/dncnn_color.onnx",
    const std::string& sr_fsrcnn_model = "", bool sr_direct_2x = false, bool sr_cascade = false, float grain_tv = 0.0f, bool use_onsager = false,
    const std::string& red_model = "", float red_decay = 1.0f, bool sr_dip_2x = false,
    const CsHrSrCtx& hr_sr = CsHrSrCtx(),
    const CsSrPrior& sr_prior = CsSrPrior());
#endif
