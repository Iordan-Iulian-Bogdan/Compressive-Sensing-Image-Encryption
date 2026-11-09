#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"
#include "photo_upscaler.hpp"

#include <vector>

class decrypt_image : CSencryption {
private:
    /** @brief header parsing + MAC verification.
    Throws std::runtime_error on legacy containers, undecryptable headers or
    failed authentication (v2). On success fills m/rows/cols/org_size and,
    for v2 containers, caches the derived key for the index regeneration.
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
    // fills ri_x/ri_y.
    CsSegPrefix open_segmask_prefix();

    // Shipped per-class quantized rates of the seg container; cached during
    // sampling so the per-segment decode can scale each tile's coef by its
    // class's sampled rate and break mixed-tile votes toward the higher one.
    std::vector<uint8_t> seg_rates_q;

    // Solve-geometry label map of the seg container; cached during sampling.
    cv::Mat seg_ids;

public:
    // Set by get_sampled_ycc420: a multi-scale (mode 20) layer, i.e. seg prefix version 2 (chroma inside the labels).
    bool ms_layer = false;

    decrypt_image() {}

    decrypt_image(cv::Mat input, const std::string& password = "");

    float get_compression_ratio();

    cv::Size get_org_size();

    // Crops the 2x output back to the native size when the encoder padded an odd side
    // (cs_build_metadata parity). No-op for even images.
    void crop_odd_parity(cv::Mat& img) const {
        if (odd_parity == 0 || img.rows < 2 || img.cols < 2) return;
        img = img(cv::Rect(0, 0, img.cols - ((odd_parity >> 1) & 1), img.rows - (odd_parity & 1))).clone();
    }

    int get_sampling_mode() const { return sampling_mode; }
    const std::vector<uint8_t>& get_seg_rates_q() const { return seg_rates_q; }
    const cv::Mat& get_seg_ids() const { return seg_ids; }
    // Rectangular tile grid chosen in auto mode (rows x cols, 0 = the square num_tiles grid).
    int grid_rows = 0, grid_cols = 0;
    int get_sample_bits() const { return sample_bits; }
    int get_m() const { return m; }
    int get_sample_bits_chroma() const { return sample_bits_chroma; }

    /** @brief unpacks a production container (seg luma, 4:2:0 chroma; modes 11 and 19) into
    per-plane sampled images + binary masks: luma at full resolution (CV_8U), each
    chroma plane on its half-resolution grid (CV_8U). Masks hold 0/1.
    Regenerates the identical index sets from (key, m, geometry, label map);
    mode 19 also reads its HR anchors into hr_samples.
    */
    void get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
        cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask);

    /** @brief decrypts a production container (mode 19, or mode 20 = its layers composited) with
    automatic parameters (iterations, tile grid, overlap, threads, coef) and writes the result to
    output_path, or to decoded_out when given (in memory, padded geometry).
    @param show_preview : live tile preview window
    @param tv : TV weight of the tile solve; -1 = auto (cs_auto_reg_tv)
    @param fista_iters : FISTA steps per pass, 0 = auto-map from the iterations (the encoder's anchor pass: 12)
    @param photo_up : the cs-sr stage's options and the restoration nets
    */
    static int decrypt_image_tiled(const cv::Mat& encrypted_img_in, const std::string& output_path, const std::string& password, bool show_preview, float tv, int fista_iters, const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out = nullptr);

    static int decrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, bool show_preview, float tv, int fista_iters, const CsPhotoUpscalerOptions& photo_up);

    /** @brief the tile pipeline of decrypt_image_tiled. dimgs is the already-parsed container
    (key cached); remaining args are the resolved parameters. Solves luma at full tile resolution
    and each chroma plane natively on its half-resolution grid, merges YCrCb -> BGR per tile,
    upscales every tile 2x with cs-sr, composites, then runs the restoration nets.
    */
    static int decrypt_image_tiled_ycc420(decrypt_image& dimgs, const std::string& output_path, int num_tiles, int overlap, int iterations, int nun_threads, float coef, bool show_preview, float tv, int fista_iters, const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out = nullptr);

};

#endif
