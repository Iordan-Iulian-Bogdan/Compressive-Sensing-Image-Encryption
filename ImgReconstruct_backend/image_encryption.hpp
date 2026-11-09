#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"
#include <functional>

class encrypt_image : CSencryption
{
private:
    float bm = 0.0f;
    // keep_key: the next container reuses the previous salt + key (no PBKDF2, same
    // sample draws), so a re-encrypt with anchors ships the samples they were chosen on.
    bool keep_key = false;
    uint8_t kept_salt[CS_SALT_BYTES] = {};

    // Container build steps (see .cpp).
    void derive_container_key(const std::string& password, uint8_t salt[CS_SALT_BYTES]);
    uint8_t* alloc_container(int total, const uint8_t salt[CS_SALT_BYTES]);
    static void stamp_sample_bits(uint8_t* buf, int sample_bits, int& chroma_bits);
    void finish_container(int total, int sample_bits, int chroma_bits,
        const std::vector<cs_body_section>& sections);

public:
    encrypt_image(const cv::Mat& input, bool global_image);

    encrypt_image() {}

    // Call before an encrypt_*: its key survives the container, and every later encrypt_* reuses
    // that salt + key (see keep_key) until release_key() wipes it.
    void reuse_key() { keep_key = true; }
    void release_key();
    // Like reuse_key, with a salt + key derived elsewhere (a multi-scale container's layers share its key).
    void use_key(const uint8_t salt[CS_SALT_BYTES], const uint8_t key[32]);

    // Native size parity recorded in the container (the input was padded to even by one replicated
    // row / column per odd side; decrypt crops it back).
    void set_odd_parity(int p) { odd_parity = p; }

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    // Per-tile mini-container (plain header + the samples at the given positions) for the decoder's tile solves.
    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    // Per-segment luma + uniform 4:2:0 chroma (mode 11) from resolved ids:
    // rates[c] in [0,1] is the luma sampling rate of class c, scaled by pixel_p.
    // Body: [seg prefix + label PNG][Y x mY][Cr x mC][Cb x mC]; decrypt
    // regenerates the positions from the shipped prefix. A non-null anchor plan
    // switches to CS_MODE_YCC420_SEG_HRDETAIL (mode 19) with the compact anchor
    // payload (see cs_anchor_payload). chroma_masked (multi-scale layers): chroma
    // drawn inside the label support (seg prefix version 2).
    void encrypt_ycc420_segmask_ids(const float& pixel_p, const std::string& password,
        const cv::Mat& ids, const std::vector<float>& rates,
        int sample_bits = 8, int chroma_bits = 0, const CsAnchorPlan* anchors = nullptr, bool chroma_masked = false);

    // Build source | binary mask | cyan overlay from the indices used by the
    // last encrypt_* call. Returns false when no indices are available.
    bool build_sampling_mask_view(cv::Mat& out) const;
    // The luma sample positions of the last encrypt_* call (rows, cols).
    void sample_positions(std::vector<int>& rows, std::vector<int>& cols) const;

    // --show-mask: write <output>.mask.png and open a window until a key is
    // pressed (skipped when headless/no window support). output_path may be
    // empty (in-memory encrypt); then only the window is used.
    void present_sampling_mask(const std::string& output_path) const;

    /** @brief encrypts a photo into the single-scale production container (mode 19) and writes it to disk:
    --seg-auto classes split into --seg-tile detail levels, 4:2:0 chroma, and HR anchors placed by a closed loop.
    @param compression_ratio : --ratio in (0.001, 1.0]
    @param seg_auto : k-means classes (2..32)
    @param hr_detail : anchor count (1..CS_HRDETAIL_MAX_SAMPLES)
    @param sample_bits, chroma_bits : --sample-bits; sample_bits 0 = the default 6-bit luma / 8-bit chroma at the
        8-bit byte budget of --ratio
    @param show_mask : after encrypt render and show the sampling mask
    @param anchor_decode : decodes a container to BGR at decrypt output geometry; the anchors go to the worst
        blocks of that decode
    */
    static int encrypt_image_tiled(const std::string& input_path, const std::string& output_path,
        const std::string& password, float compression_ratio, int seg_auto, int hr_detail, int sample_bits,
        int chroma_bits, bool show_mask, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode);
};

// Built-in scene segmentation (--seg-auto): k-means in Lab on a capped
// working copy, classes sorted by size desc, tiny components reassigned to
// the neighborhood mode, upscaled NEAREST to rows x cols. Deterministic
// (fixed OpenCV RNG seed, state restored). No models or downloads.
void cs_seg_auto_ids(const cv::Mat& bgr, int classes, int rows, int cols, cv::Mat& ids);
// Auto per-class rates from mean Sobel gradient energy (detail proxy):
// r = 0.08 + 0.92*(d/dmax)^0.75 over the classes present in ids; flat
// images (dmax == 0) get uniform 1.0.
std::vector<float> cs_seg_auto_rates(const cv::Mat& bgr, const cv::Mat& ids, int classes);
// Multi-scale container (mode 20, the default production encode): the encoder plans every 120 px tile's scale
// (2x..6x) and luma rate for the budget of a production container at ratio, and writes one seg layer (mode 11) per
// scale in use. hr_detail with anchor_decode: the 2x layer gets the production closed-loop anchors, hr_detail x its
// share of the tiles. Writes the container to output_path; returns 0 on success.
// show_mask (--show-mask): every layer's luma samples on the 2x layer's grid (W/2 x H/2; a sample of the s-x layer
// at its pixel's centre, so coarse tiles show sparse dots), as <output>.mask.png and a window like single-scale.
int cs_encrypt_multiscale(const std::string& input_path, const std::string& output_path, const std::string& password,
    int hr_detail, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode, float ratio, bool show_mask = false);

// Per-tile detail split (--seg-tile, the "seg8-tile" container): every 64x64
// tile gets an LOD from its mean |Laplacian| (min-max to [32, 255]) bucketed
// into 8 levels; label = class * 8 + level, rate = class rate x level weight
// (256 + bin-centre LOD) / the class's pixel-weighted mean weight, so each
// class keeps its share. The rates are then scaled by one factor s, solved so
// that sum(pixels x min(1, s x rate)) = pixel_p x all pixels: the same luma
// budget as a uniform/adaptive container at that ratio (busy labels cap at 1
// and the rest keep rising). ids (K -> 8K labels) and rates (final, encrypt
// at ratio 1) in place. Needs classes <= 32.
void cs_seg_tile_refine(const cv::Mat& bgr, cv::Mat& ids, std::vector<float>& rates, int classes,
    float pixel_p);

#endif
