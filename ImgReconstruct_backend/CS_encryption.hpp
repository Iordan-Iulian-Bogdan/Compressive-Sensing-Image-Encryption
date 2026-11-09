#ifndef CS_ENCRYPTION_HPP
#define CS_ENCRYPTION_HPP


#include "helper_functions.hpp"
#include "crypto_utils.hpp"

// Chroma support of a label map (seg prefix version 2): CV_8U at the mode's
// chroma geometry, 255 where any luma pixel of the chroma pixel's block has a
// label with rate > 0. Labels at luma (pipeline) geometry rows x cols.
cv::Mat cs_seg_chroma_support(const cv::Mat& labels, const std::vector<uint8_t>& rates_q, int mode);

class CSencryption {
protected:
    cv::Mat encrypted_img, input_img;
    cv::Size org_size;
    int odd_parity = 0;   // native size parity (cs_build_metadata): bit 0 odd height, bit 1 odd width
    int rows, cols, m, n;
    std::vector<int> ri_x, ri_y;

    // 256-bit key material derived by PBKDF2-HMAC-SHA256(password, salt).
    // Valid after cs_parse_header succeeded on a v2 container (decrypt side)
    // or after encrypt() derived it (encrypt side). Wiped in the destructor.
    uint8_t cs_key[32] = {};
    bool cs_key_valid = false;

    void returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, const uint8_t key[32]);

    // Per-segment sampling: class ids from an 8-bit label map (xm x ym),
    // per-class sample counts from quantized rates
    // (count_c = round(pix_c * q_c / 255), ids >= rates_q.size() score 0).
    // One shared RNG stream draws classes in ascending id order (partial
    // Fisher-Yates over each class's pixel pool), then one global shuffle
    // fixes the measurement order — fully deterministic from (key, labels,
    // rates) so encryption and decryption agree.
    void returnSegMaskIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
        const cv::Mat& labels, const std::vector<uint8_t>& rates_q, const uint8_t key[32]);

    // Sampling mode state (decrypt side: restored from the container header;
    // encrypt side: set by encrypt_*). CS_MODE_* from crypto_utils.hpp.
    int sampling_mode = CS_MODE_RANDOM;

    // Mode 19 (CS_MODE_YCC420_SEG_HRDETAIL): exact native-luma anchors carried
    // as a trailing CTR-encrypted body section. Filled by the decrypt-side
    // payload parse; enter the cs-sr solve as data terms.
    std::vector<CsHrSample> hr_samples;
    bool hr_detail = false;
    uint32_t hr_count = 0;
    size_t hr_applied = 0;

    // 4:2:0 chroma sampling positions on the half-resolution chroma grid +
    // per-plane chroma sample count, filled by returnYcc420ChromaIndices;
    // ri_x/ri_y keep the full-res luma positions.
    std::vector<int> ri_cx1, ri_cy1, ri_cx2, ri_cy2;
    int m_chroma = 0;
    // Non-empty (CV_8U at the chroma geometry): the chroma draws pick only its
    // nonzero pixels (seg prefix version 2, cs_seg_chroma_support).
    cv::Mat chroma_support;

    // Chroma draws: fills ri_cx1/ri_cy1 (Cr), ri_cx2/ri_cy2 (Cb) and m_chroma
    // from tagged key copies (see cs_ycc420_channel_key). Fully deterministic
    // from (key, m_luma, geometry, chroma_support) so encryption and
    // decryption agree.
    void returnYcc420ChromaIndices(int xm, int ym, int m_luma, const uint8_t key[32], int mode = CS_MODE_YCC420);

public:
    ~CSencryption() {
        if (cs_key_valid) {
            cs_wipe(cs_key, sizeof(cs_key));
            cs_key_valid = false;
        }
    }
};

#endif
