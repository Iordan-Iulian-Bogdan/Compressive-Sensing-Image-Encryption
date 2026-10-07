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
 */
std::vector<int> cs_compute_tile_sample_counts(const std::vector<uint8_t>& lod, int m_total,
    int rows, int cols, int tile_size, int weight_base = 256, int full_threshold = 0);

void cs_hf_thumb_weights(const cv::Mat& thumbnail, std::vector<float>& weights,
    int& weight_cols, int& weight_rows);

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
    // weight_base, full_threshold).
    void returnAdaptiveIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
        const std::vector<uint8_t>& lod, int m_total, const uint8_t key[32], int tile_size,
        int weight_base = 256, int full_threshold = 0);
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
    cv::Mat thumb_seed;

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
