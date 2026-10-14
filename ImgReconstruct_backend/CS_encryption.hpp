#ifndef CS_ENCRYPTION_HPP
#define CS_ENCRYPTION_HPP


#include "helper_functions.hpp"
#include "crypto_utils.hpp"

/** @brief deterministic per-tile sample budget from LOD bytes.
 * Pure function shared by encrypt and decrypt: same (lod, m, geometry,
 * weight_base) always yields the same counts. Largest-remainder allocation
 * with weight = weight_base + lod (scaled by tile capacity), then clamp to
 * each tile's pixel capacity while keeping the total exactly m.
 * weight_base controls aggression: high = near-uniform, 1 = max LOD bias.
 * full_threshold > 0 (--lod-full) additionally guarantees whole tiles: tiles
 * with lod >= threshold take their full pixel capacity first (greedy fit in
 * lod-desc/index-asc order against m_total), the rest split the remainder.
 * cutoff > 0 (--adaptive-cutoff) hard-excludes tiles with lod < cutoff
 * (count 0, surviving tiles absorb the budget up to their capacity; the
 * total may come in under m_total when survivors saturate). Cutoff wins
 * over full_threshold on overlap. Decrypt recomputes identical counts, so
 * the cutoff travels in the authenticated header with the other parameters.
 */
std::vector<int> cs_compute_tile_sample_counts(const std::vector<uint8_t>& lod, int m_total,
    int rows, int cols, int tile_size, int weight_base = 256, int full_threshold = 0,
    int cutoff = 0);

void cs_hf_thumb_weights(const cv::Mat& thumbnail, std::vector<float>& weights,
    int& weight_cols, int& weight_rows);

// Compact triangular filtering of local rates. CV_16U probability map;
// support is exactly width pixels per axis, not an unbounded Gaussian.
cv::Mat cs_segmask_density(const cv::Mat& labels,
    const std::vector<uint8_t>& rates_q, int width);

class CSencryption {
protected:
    cv::Mat encrypted_img, input_img;
    cv::Mat decrypted_img;
    cv::Size org_size;
    int rows, cols, m, n;
    std::vector<int> ri_x, ri_y;

    // 256-bit key material derived by PBKDF2-HMAC-SHA256(password, salt).
    // Valid after cs_parse_header succeeded on a v2 container (decrypt side)
    // or after encrypt() derived it (encrypt side). Wiped in the destructor.
    uint8_t cs_key[32] = {};
    bool cs_key_valid = false;

    void returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, const uint8_t key[32]);

    // Periodic tile-based sampling: one random per-tile pattern, replicated
    // across the image. samples_per_tile positions are drawn inside one
    // tile_size x tile_size block; the pattern repeats over the whole image.
    void returnPeriodicIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int samples_per_tile, const uint8_t key[32], int tile_size);

    // Adaptive periodic sampling: per-tile counts from LOD bytes (see
    // cs_compute_tile_sample_counts), then a distinct in-tile draw per tile
    // in row-major order. Fully deterministic from (key, lod, m, geometry,
    // weight_base, full_threshold, cutoff).
    void returnAdaptiveIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
        const std::vector<uint8_t>& lod, int m_total, const uint8_t key[32], int tile_size,
        int weight_base = 256, int full_threshold = 0, int cutoff = 0);

    // Per-segment sampling (mode 10): class ids from an 8-bit label map
    // (xm x ym), per-class sample counts from quantized rates
    // (count_c = round(pix_c * q_c / 255), ids >= rates_q.size() score 0).
    // One shared RNG stream draws classes in ascending id order, then one
    // global shuffle fixes the measurement order — fully deterministic from
    // (key, labels, rates) so encryption and decryption agree.
    // seg_blend (0..128, pixels): smooth_blend selects compact local density
    // smoothing and per-pixel draws. False regenerates the older flat boosted
    // bands. W == 0 always runs the original class-budget sampler.
    void returnSegMaskIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
        const cv::Mat& labels, const std::vector<uint8_t>& rates_q, const uint8_t key[32],
        int seg_blend = 0, bool smooth_blend = false);
    void returnHfWeightedIndices(std::vector<int>& ri_x, std::vector<int>& ri_y,
        int rows, int cols, const std::vector<uint8_t>& lod, int m_total,
        const uint8_t key[32], int tile_size, int weight_base,
        const std::vector<float>& weights, int weight_cols, int weight_rows,
        float pixel_floor = 8.0f);

    // Sampling mode state (decrypt side: restored from the container header;
    // encrypt side: set by encrypt_*). CS_MODE_* from crypto_utils.hpp.
    int sampling_mode = CS_MODE_RANDOM;
    int periodic_tile = 0;
    int periodic_samples = 0;
    int adaptive_base = 256;         // mode 2: LOD weight_base from header
    int lod_full_threshold = 0;      // --lod-full detail guarantee from header (0 = off)
    int adaptive_cutoff = 0;         // --adaptive-cutoff hard floor from header (0 = off)
    cv::Mat thumb_seed;

    // Mode 4/9 (CS_MODE_*_HRDETAIL): exact native-luma anchors carried as
    // a trailing CTR-encrypted body section. Filled by the decrypt-side
    // payload parse; applied as post-composite luma replacements.
    std::vector<CsHrSample> hr_samples;
    bool hr_detail = false;
    uint32_t hr_count = 0;
    size_t hr_applied = 0;

    // Mode 3/7 (CS_MODE_YCC420/CS_MODE_YCC422): coarse chroma sampling
    // positions on the mode's chroma grid + per-plane chroma sample count.
    // half-resolution chroma grid + per-plane chroma sample count. Filled by
    // returnYcc420Indices; ri_x/ri_y keep the full-res luma positions.
    std::vector<int> ri_cx1, ri_cy1, ri_cx2, ri_cy2;
    int m_chroma = 0;

    // YCC 4:2:0 index generation: m_luma full-res luma positions from the key
    // plus m_chroma coarse-grid positions per chroma plane from tagged key
    // copies (see cs_ycc420_channel_key). Fully deterministic from
    // (key, m_luma, geometry) so encryption and decryption agree.
    void returnYcc420Indices(int xm, int ym, int m_luma, const uint8_t key[32], int mode = CS_MODE_YCC420);

    // Coarse-grid chroma half: fills ri_cx1/ri_cy1 (Cr), ri_cx2/ri_cy2 (Cb)
    // and m_chroma from (key, m_luma, geometry). Shared by the uniform-luma
    // path above and the LOD-luma path (whose luma draw differs).
    void returnYcc420ChromaIndices(int xm, int ym, int m_luma, const uint8_t key[32], int mode = CS_MODE_YCC420);

public:
    static int params;

    ~CSencryption() {
        if (cs_key_valid) {
            cs_wipe(cs_key, sizeof(cs_key));
            cs_key_valid = false;
        }
    }
};

#endif
