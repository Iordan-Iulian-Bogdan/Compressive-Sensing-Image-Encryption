#ifndef CS_ENCRYPTION_HPP
#define CS_ENCRYPTION_HPP


#include "helper_functions.hpp"
#include "crypto_utils.hpp"


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

    // New: periodic tile-based sampling - generate random pattern within one tile,
    // then tile it across the entire image.
    void returnPeriodicIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, const uint8_t key[32], int tile_size);

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