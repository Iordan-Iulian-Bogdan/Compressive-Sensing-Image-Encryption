#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"

class encrypt_image : CSencryption
{
private:
    float bm = 0.0f;

public:
    encrypt_image(std::string input_path);

    encrypt_image(const cv::Mat& input, bool global_image);

    encrypt_image() {}

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void encrypt(const float& pixel_p, const std::string& password);
    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    // Periodic tile-based encryption: generate random pattern within one tile,
    // then repeat across the image. tile_size should divide the image dimensions.
    void encrypt_periodic(const float& pixel_p, const std::string& password, int tile_size);

    // Adaptive periodic encryption: per-tile LOD bytes drive sample counts
    // (high detail -> more measurements). lod_min floors the 8-bit score so
    // flat tiles are never starved. adaptive_strength in [0,1] scales LOD bias
    // (0=uniform, 0.5=default, 1=max). Writes mode=2 + lod + weight_base.
    void encrypt_adaptive(const float& pixel_p, const std::string& password, int tile_size,
        int lod_min, int weight_base = 256, bool two_pass = false, float pilot_ratio = 0.05f,
        const std::string& regions_json = "", float region_blend = 0.5f, float lod_smooth = 0.0f);

    void writeEncryptedImageToDisk(const std::string& output_path);

    cv::Size get_size();

    // Build source | binary mask | cyan overlay from the indices used by the
    // last encrypt_* call. Returns false when no indices are available.
    bool build_sampling_mask_view(cv::Mat& out) const;

    // --show-mask: write <output>.mask.png and open a window until a key is
    // pressed (skipped when headless/no window support). output_path may be
    // empty (in-memory encrypt); then only the window is used.
    void present_sampling_mask(const std::string& output_path) const;

    /** @brief encrypts a given image and writes the result to disk
    @param input_path : path to input image
    @param output_path : path to output image
    @param password : password to be used for encryption and decryption
    @param compression_ratio : compression ratio given as a value from 0.001 to 1.0, lower value means higher compression
    @param tile_size : > 0 selects periodic tile sampling with this tile size; 0 = random global sampling
    @param encrypted_out : if non-null, receives the encrypted image in memory (no PNG roundtrip)
    @param adaptive : if true, LOD-based per-tile sampling (implies tiling; tile_size used as grid)
    @param lod_min : floor byte for LOD scores in adaptive mode (default 32)
    @param adaptive_strength : LOD bias aggression in [0,1] (0=uniform, 0.5=default, 1=max)
    @param show_mask : if true, after encrypt render and show the sampling mask
    @param full_res : if true, keep native input resolution (skip default 2x downscale)
    */
    static int encrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, float compression_ratio = 1.0f, int tile_size = 0, cv::Mat* encrypted_out = nullptr, bool adaptive = false, int lod_min = 32, float adaptive_strength = 0.5f, bool show_mask = false, bool full_res = false, bool two_pass = false, float pilot_ratio = 0.05f, const std::string& regions_json = "", float region_blend = 0.5f, float lod_smooth = 0.0f);

private:
    // Two-pass coarse-to-fine LOD: uniform pilot sampling of the source at
    // pilot_ratio of each tile's pixels, cheap per-tile FISTA recon, then
    // per-tile MSE vs the source becomes the LOD byte (same grid layout and
    // min-max + floor normalization as the Laplacian path). Requires the
    // derived cs_key (pilot draw is keyed deterministically). Pilot positions
    // are scoring-only and discarded; the final budget draw reuses the
    // standard adaptive path, so container format and decrypt are unchanged.
    std::vector<uint8_t> compute_twopass_lod(int tile_size, int lod_min, float pilot_ratio);
};

/** @brief encrypts a given image and writes the result to disk
@param input_path : path to input image
@param output_path : path to output image
@param password : password to be used for encryption and decryption
@param compression_ratio : compression ratio given as a value from 0 to 1, lower value means higher compression
*/

#endif