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

    void encrypt(const float& pixel_p, const std::string& password, int sample_bits = 8, int chroma_bits = 0, const std::vector<CsHrSample>& hr_detail = {});
    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    // YCC 4:2:0 split encryption (mode 3; mode 7 = 4:2:2): BGR -> YCrCb,
    // chroma downsampled to the mode's coarse grid, luma sampled at full
    // resolution and each chroma plane sampled on its coarse grid. Body
    // layout: [Y x m][Cr x m_chroma][Cb x m_chroma]. pixel_p is the LUMA
    // sampling ratio; total stored ~= 1.5x luma budget (4:2:0) or ~= 2x
    // (4:2:2, chroma full height).
    void encrypt_ycc420(const float& pixel_p, const std::string& password, int sample_bits = 8, int chroma_bits = 0, int mode = CS_MODE_YCC420, const std::vector<CsHrSample>& hr_detail = {});

    void encrypt_hf_focus(const float& pixel_p, const std::string& password,
        int tile_size, int lod_min, int sample_bits = 8, int chroma_bits = 0);
    void encrypt_ycc420_hf(const float& pixel_p, const std::string& password,
        int tile_size, int sample_bits = 8, int chroma_bits = 0, int mode = CS_MODE_YCC420_HF);

    // Per-segment encryption (mode 10): the mask file is an 8-bit label map
    // (pixel value = class id) or a BGR color mask matched to the scene
    // palette; rates[c] in [0,1] is the sampling rate of class c (ids beyond
    // the list score 0). pixel_p scales every rate globally. Rates are
    // quantized to u8 and shipped with a PNG of the label map in the body
    // prefix, so decrypt regenerates the identical positions with no flags.
    void encrypt_segmask(const float& pixel_p, const std::string& password,
        const std::string& mask_path, const std::vector<float>& rates,
        int sample_bits = 8, int chroma_bits = 0, int seg_blend = 0);

    // Per-segment luma + uniform chroma (modes 11/12): same seg-mask luma
    // budget as encrypt_segmask, each chroma plane sampled uniformly on its
    // coarse grid (mode selects 4:2:0 vs 4:2:2). Body: [seg prefix+PNG]
    // [Y x mY][Cr x mC][Cb x mC]; decrypt auto-detects.
    void encrypt_ycc420_segmask(const float& pixel_p, const std::string& password,
        const std::string& mask_path, const std::vector<float>& rates,
        int sample_bits = 8, int chroma_bits = 0, int mode = CS_MODE_YCC420_SEG,
        int seg_blend = 0);

    // YCC 4:2:0 with LOD-driven luma (mode 3 + shipped lod bytes; mode 7 for
    // 4:2:2): same LOD sources as encrypt_adaptive (Laplacian / two-pass /
    // spectral-DCT / LLM regions + smoothing) drive the luma budget; chroma stays uniform
    // coarse-grid.
    // Body layout: [lod x lod_bytes][Y x mY_written][Cr][Cb].
    void encrypt_ycc420_adaptive(const float& pixel_p, const std::string& password,
        int tile_size, int lod_min, int weight_base = 256, bool two_pass = false,
        float pilot_ratio = 0.05f, const std::string& regions_json = "",
        float region_blend = 0.5f, float lod_smooth = 0.0f, int sample_bits = 8, int chroma_bits = 0, int lod_full = 0, int mode = CS_MODE_YCC420, bool spectral_lod = false, float lod_gamma = 1.0f, int lod_cutoff = 0);    // Periodic tile-based encryption: generate random pattern within one tile,
    // then repeat across the image. tile_size should divide the image dimensions.
    void encrypt_periodic(const float& pixel_p, const std::string& password, int tile_size, int sample_bits = 8, int chroma_bits = 0);

    // Adaptive periodic encryption: per-tile LOD bytes drive sample counts
    // (high detail -> more measurements). lod_min floors the 8-bit score so
    // flat tiles are never starved. adaptive_strength in [0,1] scales LOD bias
    // (0=uniform, 0.5=default, 1=max). Writes mode=2 + lod + weight_base.
    void encrypt_adaptive(const float& pixel_p, const std::string& password, int tile_size,
        int lod_min, int weight_base = 256, bool two_pass = false, float pilot_ratio = 0.05f,
        const std::string& regions_json = "", float region_blend = 0.5f, float lod_smooth = 0.0f,
        int sample_bits = 8, int chroma_bits = 0, int lod_full = 0, bool spectral_lod = false,
        float lod_gamma = 1.0f, int lod_cutoff = 0);

    void writeEncryptedImageToDisk(const std::string& output_path);

    cv::Size get_size();

    // Build source | binary mask | cyan overlay from the indices used by the
    // last encrypt_* call. Returns false when no indices are available.
    bool build_sampling_mask_view(cv::Mat& out) const;

    // Build under/over-sampled tile canvases from the indices used by the
    // last encrypt_* call (see cs_build_sample_class_views). Writes
    // <prefix>_under.png / <prefix>_over.png; no-ops with a warning when
    // there is nothing to classify. No GUI (headless-safe).
    void present_sample_classes(const std::string& prefix, int tile_size) const;

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
    static int encrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, float compression_ratio = 1.0f, int tile_size = 0, cv::Mat* encrypted_out = nullptr, bool adaptive = false, int lod_min = 32, float adaptive_strength = 0.5f, bool show_mask = false, bool full_res = false, bool two_pass = false, float pilot_ratio = 0.05f, const std::string& regions_json = "", float region_blend = 0.5f, float lod_smooth = 0.0f, bool ycc420 = false, bool hf_focus = false,     int sample_bits = 8, int chroma_bits = 0, int lod_full = 0, bool ycc422 = false, bool spectral = false, const std::string& sample_split = "", int hr_detail = 0, float lod_gamma = 1.0f, int lod_cutoff = 0, const std::string& seg_mask = "", const std::string& seg_rates = "", int seg_blend = 0);

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

// Per-segment helpers (free functions for test access): parse_seg_rates
// splits "0.02,0.6,..." into per-class [0,1] rates (1..255 entries);
// cs_segmask_resolve_ids maps a mask file image to 8-bit class ids at the
// pipeline geometry (grayscale passes through, BGR matches the scene
// palette by nearest color, everything resizes with INTER_NEAREST).
std::vector<float> parse_seg_rates(const std::string& s);
cv::Mat cs_segmask_resolve_ids(const cv::Mat& mask_raw, int rows, int cols);

// Per-tile spectral LOD bytes (free function for test access): each tile's
// gray DCT energy whitened by the image's global radial spectral decay,
// min-max normalized with lod_min floor. Same grid layout as
// compute_adaptive_lod, so counts/container/decrypt are unchanged.
std::vector<uint8_t> compute_spectral_lod(const cv::Mat& img, int tile_size, int lod_min);

// Classify tiles by sample density: bins (rix, riy) into a tile_size grid
// over rows x cols and splits tile counts against their mean (under: count
// < mean, over: count > mean; ties go in neither canvas). Renders two
// full-geometry BGR canvases: source tile pixels at original positions for
// the class, black elsewhere, each labeled with its tile count. Pure
// function of inputs (deterministic); returns false on degenerate input
// (empty image/indices, tile_size < 1, no grid cells).
bool cs_build_sample_class_views(const cv::Mat& input,
    const std::vector<int>& rix, const std::vector<int>& riy,
    int tile_size, cv::Mat& under, cv::Mat& over);

/** @brief encrypts a given image and writes the result to disk
@param input_path : path to input image
@param output_path : path to output image
@param password : password to be used for encryption and decryption
@param compression_ratio : compression ratio given as a value from 0 to 1, lower value means higher compression
*/

#endif
