#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"
#include "photo_upscaler.hpp"

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

public:
    decrypt_image(const std::string input_path, const std::string& password = "");

    decrypt_image() {}

    decrypt_image(cv::Mat input, const std::string& password = "");

    void decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, bool ycrcb = false, bool chroma_sub = false, float tv = 0.0f, int solver = CS_SOLVER_OWLQN, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f);

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
    chroma plane on its half-resolution grid (CV_8U). Masks hold 0/1.
    Regenerates the identical index sets from (key, m, geometry). Supports
    uniform YCC420 and HF-focus YCC420 containers.
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
    static int decrypt_image_tiled(const cv::Mat& encrypted_img_in, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_OWLQN, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false);

    static int decrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_OWLQN, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false);

    /** @brief YCC 4:2:0 tile pipeline for mode-3 containers (auto-selected by
    decrypt_image_tiled from the header; never called directly with other
    modes). dimgs is the already-parsed container (key cached); remaining
    args mirror decrypt_image_tiled. Solves luma at full tile resolution and
    each chroma plane natively on its half-resolution grid, then merges
    YCrCb -> BGR per tile. FISTA_JOINT degrades to per-channel FISTA.
    */
    static int decrypt_image_tiled_ycc420(decrypt_image& dimgs, const std::string& output_path, int num_tiles = 24, int overlap = 48, int iterations = 5, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, float tv = 0.0f, bool full_res = false, int solver = CS_SOLVER_OWLQN, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const CsPhotoUpscalerOptions& photo_up = CsPhotoUpscalerOptions(), bool per_tile_coef = false, bool per_tile_tv = false);

};

/** @brief solves all tiles in wavefront (anti-diagonal) order: wave w = i + j,
all tiles of a wave run in parallel and every tile is warm-started from the
already-solved west/north neighbors' overlap strips (falling back to the
generic reference solution when no neighbor exists, e.g. the first wave).
*/
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, bool ycrcb = false, bool chroma_sub = false, float tv = 0.0f, int solver = CS_SOLVER_OWLQN, int fista_iters = 0, int reweights = 2, int basis = CS_BASIS_DCT, float wscale = 2.0f, const cv::Mat& thumbnail_seed = cv::Mat(), bool per_tile_coef = false, bool per_tile_tv = false, std::vector<std::vector<cv::Mat>>* hr_out = nullptr, bool fuse_upscale = false);
#endif
