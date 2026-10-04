#include "crypto_utils.hpp"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

// ---------------------------------------------------------------------------
// Crypto primitives are provided by the platform backend selected below:
//   - Windows  : CNG (BCrypt) — PBKDF2-HMAC-SHA256, HMAC-SHA256, AES, RNG
//   - other OS : OpenSSL      — same primitives via libcrypto
// Everything above this ifdef (metadata, header encode/parse, seal/verify)
// is backend-independent.
// ---------------------------------------------------------------------------

#ifdef _WIN32

#include <windows.h>
#include <bcrypt.h>

#pragma comment(lib, "bcrypt.lib")

#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
#endif

namespace {

void throw_nt(NTSTATUS st, const char* what) {
    if (st != STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (CNG status 0x" +
            std::to_string((unsigned long)st) + ")");
    }
}

struct AlgHandle {
    BCRYPT_ALG_HANDLE h = NULL;
    explicit AlgHandle(const wchar_t* alg, DWORD flags = 0) {
        throw_nt(BCryptOpenAlgorithmProvider(&h, alg, NULL, flags), "BCryptOpenAlgorithmProvider");
    }
    ~AlgHandle() { if (h) BCryptCloseAlgorithmProvider(h, 0); }
    AlgHandle(const AlgHandle&) = delete;
    AlgHandle& operator=(const AlgHandle&) = delete;
};

const DWORD sha256_obj_len = [] {
    // object length must be queried with the SAME flags used to create the
    // hash: an HMAC hash object stores the key state and is larger
    AlgHandle alg(L"SHA256", BCRYPT_ALG_HANDLE_HMAC_FLAG | BCRYPT_HASH_REUSABLE_FLAG);
    DWORD len = 0, cb = 0;
    throw_nt(BCryptGetProperty(alg.h, BCRYPT_OBJECT_LENGTH, (PUCHAR)&len, sizeof(len), &cb, 0),
        "BCryptGetProperty(OBJECT_LENGTH)");
    return len;
}();

// Incremental HMAC-SHA256 so header and body are MACed in one pass.
class HmacSha256 {
public:
    explicit HmacSha256(const uint8_t key[32]) {
        NTSTATUS st = BCryptOpenAlgorithmProvider(&alg_, L"SHA256", NULL,
            BCRYPT_ALG_HANDLE_HMAC_FLAG | BCRYPT_HASH_REUSABLE_FLAG);
        if (st != STATUS_SUCCESS) throw std::runtime_error("BCryptOpenAlgorithmProvider(HMAC) failed");
        st = BCryptCreateHash(alg_, &h_, obj_.data(), (ULONG)obj_.size(), (PUCHAR)key, 32, 0);
        if (st != STATUS_SUCCESS) {
            BCryptCloseAlgorithmProvider(alg_, 0);
            throw std::runtime_error("BCryptCreateHash(HMAC) failed");
        }
    }
    ~HmacSha256() {
        if (h_) BCryptDestroyHash(h_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }
    HmacSha256(const HmacSha256&) = delete;
    HmacSha256& operator=(const HmacSha256&) = delete;

    void update(const uint8_t* data, size_t len) {
        while (len > 0) {
            ULONG chunk = (ULONG)std::min<size_t>(len, 0x40000000UL);
            throw_nt(BCryptHashData(h_, (PUCHAR)data, chunk, 0), "BCryptHashData");
            data += chunk;
            len -= chunk;
        }
    }

    void finish(uint8_t out[32]) {
        throw_nt(BCryptFinishHash(h_, out, 32, 0), "BCryptFinishHash");
    }

private:
    BCRYPT_ALG_HANDLE alg_ = NULL;
    BCRYPT_HASH_HANDLE h_ = NULL;
    std::vector<uint8_t> obj_ = std::vector<uint8_t>(sha256_obj_len);
};

} // namespace

bool cs_random_bytes(uint8_t* out, size_t len) {
    try {
        AlgHandle rng(L"RNG");
        // chunk to DWORD range for very large requests
        while (len > 0) {
            ULONG chunk = (ULONG)std::min<size_t>(len, 0x40000000UL);
            if (BCryptGenRandom(rng.h, out, chunk, 0) != STATUS_SUCCESS) return false;
            out += chunk;
            len -= chunk;
        }
        return true;
    }
    catch (...) {
        return false;
    }
}

bool cs_derive_key(const std::string& password, const uint8_t* salt, uint8_t key_out[32]) {
    if (password.empty()) return false;
    try {
        AlgHandle alg(L"SHA256", BCRYPT_ALG_HANDLE_HMAC_FLAG);
        NTSTATUS st = BCryptDeriveKeyPBKDF2(alg.h,
            (PUCHAR)password.data(), (ULONG)password.size(),
            (PUCHAR)salt, CS_SALT_BYTES, CS_PBKDF2_ITERATIONS,
            key_out, 32, 0);
        return st == STATUS_SUCCESS;
    }
    catch (...) {
        return false;
    }
}

bool cs_aes256_ctr_xor(uint8_t* data, size_t len, const uint8_t key[32], const uint8_t counter[16]) {
    try {
        AlgHandle alg(L"AES");
        throw_nt(BCryptSetProperty(alg.h, BCRYPT_CHAINING_MODE,
            (PUCHAR)BCRYPT_CHAIN_MODE_ECB, sizeof(BCRYPT_CHAIN_MODE_ECB), 0),
            "BCryptSetProperty(ECB)");
        BCRYPT_KEY_HANDLE hk = NULL;
        throw_nt(BCryptGenerateSymmetricKey(alg.h, &hk, NULL, 0, (PUCHAR)key, 32, 0),
            "BCryptGenerateSymmetricKey");

        uint8_t ctr[16], ks[16];
        std::memcpy(ctr, counter, 16);
        bool ok = true;
        ULONG done = 0;
        for (size_t off = 0; off < len && ok; off += 16) {
            // ECB single-block encrypt of the counter block = keystream
            ok = BCryptEncrypt(hk, ctr, 16, NULL, NULL, 0, ks, 16, &done, 0) == STATUS_SUCCESS;
            if (!ok) break;
            const size_t n = std::min<size_t>(16, len - off);
            for (size_t i = 0; i < n; i++) {
                data[off + i] ^= ks[i];
            }
            // big-endian increment of the counter block
            for (int b = 15; b >= 0; --b) {
                if (++ctr[b] != 0) break;
            }
        }
        cs_wipe(ctr, sizeof(ctr));
        cs_wipe(ks, sizeof(ks));
        BCryptDestroyKey(hk);
        return ok;
    }
    catch (...) {
        return false;
    }
}

#else // !_WIN32 -> OpenSSL backend

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>

namespace {

// Incremental HMAC-SHA256 (same interface as the CNG version above).
class HmacSha256 {
public:
    explicit HmacSha256(const uint8_t key[32]) {
        ctx_ = HMAC_CTX_new();
        if (!ctx_) throw std::runtime_error("HMAC_CTX_new failed");
        if (HMAC_Init_ex(ctx_, key, 32, EVP_sha256(), NULL) != 1) {
            throw std::runtime_error("HMAC_Init_ex failed");
        }
    }
    ~HmacSha256() {
        if (ctx_) HMAC_CTX_free(ctx_);
    }
    HmacSha256(const HmacSha256&) = delete;
    HmacSha256& operator=(const HmacSha256&) = delete;

    void update(const uint8_t* data, size_t len) {
        if (HMAC_Update(ctx_, data, len) != 1) throw std::runtime_error("HMAC_Update failed");
    }

    void finish(uint8_t out[32]) {
        unsigned int out_len = 0;
        if (HMAC_Final(ctx_, out, &out_len) != 1 || out_len != 32) {
            throw std::runtime_error("HMAC_Final failed");
        }
    }

private:
    HMAC_CTX* ctx_ = nullptr;
};

} // namespace

bool cs_random_bytes(uint8_t* out, size_t len) {
    return len == 0 || RAND_bytes(out, (int)len) == 1;
}

bool cs_derive_key(const std::string& password, const uint8_t* salt, uint8_t key_out[32]) {
    if (password.empty()) return false;
    return PKCS5_PBKDF2_HMAC(
        password.data(), (int)password.size(),
        salt, CS_SALT_BYTES, CS_PBKDF2_ITERATIONS,
        EVP_sha256(), 32, key_out) == 1;
}

bool cs_aes256_ctr_xor(uint8_t* data, size_t len, const uint8_t key[32], const uint8_t counter[16]) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return false;
    bool ok = EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, counter) == 1;
    if (ok) {
        int out_len = 0;
        ok = EVP_EncryptUpdate(ctx, data, &out_len, data, (int)len) == 1 && out_len == (int)len;
    }
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

#endif // _WIN32

// ---------------------------------------------------------------------------
// Backend-independent container format code
// ---------------------------------------------------------------------------

std::string cs_build_metadata(int m, int rows, int cols, int org_h, int org_w) {
    std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" +
        std::to_string(org_h) + "|" + std::to_string(org_w);
    if (text.size() > CS_METADATA_BYTES) {
        throw std::runtime_error("metadata too large for header (image dimensions out of range)");
    }
    return std::string(CS_METADATA_BYTES - text.size(), '-') + text;
}

namespace {

// Parse the '-'-padded "m|rows|cols|h|w" metadata string. Returns false on
// malformed content instead of throwing, so header errors surface uniformly.
bool parse_metadata_fields(const uint8_t* meta, int (&vals)[5]) {
    std::string text((const char*)meta, CS_METADATA_BYTES);
    const size_t first = text.find_first_not_of('-');
    if (first == std::string::npos) return false;
    text = text.substr(first);

    int vi = 0;
    std::string cur;
    for (char c : text) {
        if (c == '|') {
            if (vi >= 5) return false;
            try { vals[vi++] = std::stoi(cur); } catch (...) { return false; }
            cur.clear();
        }
        else if (c == '-' || (c >= '0' && c <= '9')) {
            cur.push_back(c);
        }
        else {
            return false;
        }
    }
    if (vi != 4) return false;
    try { vals[vi++] = std::stoi(cur); } catch (...) { return false; }
    return true;
}

} // namespace

bool cs_write_header(uint8_t* buf, size_t buf_bytes, const std::string& metadata, const uint8_t key[32], const uint8_t salt[CS_SALT_BYTES]) {
    if (buf_bytes < CS_HEADER_BYTES) return false;
    if (metadata.size() != CS_METADATA_BYTES) return false;

    uint8_t iv[CS_IV_BYTES], scratch_meta[CS_METADATA_BYTES];
    if (!cs_random_bytes(iv, CS_IV_BYTES)) return false;

    buf[CS_OFF_VERSION] = 2;
    std::memcpy(buf + CS_OFF_SALT, salt, CS_SALT_BYTES);
    std::memcpy(buf + CS_OFF_IV, iv, CS_IV_BYTES);
    std::memset(buf + CS_OFF_TAG, 0, CS_TAG_BYTES);
    std::memset(buf + CS_OFF_PAD, 0, CS_HEADER_BYTES - CS_OFF_PAD);

    // encrypt-then-MAC: the metadata plaintext only ever lives in scratch
    std::memcpy(scratch_meta, metadata.data(), CS_METADATA_BYTES);
    const bool ok = cs_aes256_ctr_xor(scratch_meta, CS_METADATA_BYTES, key, iv);
    if (ok) std::memcpy(buf + CS_OFF_META, scratch_meta, CS_METADATA_BYTES);
    cs_wipe(scratch_meta, sizeof(scratch_meta));
    cs_wipe(iv, sizeof(iv));
    return ok;
}

void cs_write_header_plain(uint8_t* buf, size_t buf_bytes, const std::string& metadata) {
    if (buf_bytes < CS_HEADER_BYTES || metadata.size() != CS_METADATA_BYTES) {
        throw std::runtime_error("cs_write_header_plain: bad buffer or metadata");
    }
    buf[CS_OFF_VERSION] = 1;
    std::memset(buf + CS_OFF_SALT, 0, CS_SALT_BYTES);
    std::memset(buf + CS_OFF_IV, 0, CS_IV_BYTES);
    std::memcpy(buf + CS_OFF_META, metadata.data(), CS_METADATA_BYTES);
    std::memset(buf + CS_OFF_TAG, 0, CS_TAG_BYTES);
    std::memset(buf + CS_OFF_PAD, 0, CS_HEADER_BYTES - CS_OFF_PAD);
}

void cs_pack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out) {
    if (count == 0) return;
    if (bits >= 8) {
        std::memcpy(out, in, count);
        return;
    }
    const int maxv = (1 << bits) - 1;
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t q = (uint32_t)((in[i] * maxv + 127) / 255);
        acc = (acc << (uint32_t)bits) | q;
        acc_bits += bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            out[o++] = (uint8_t)((acc >> acc_bits) & 0xFF);
        }
    }
    if (acc_bits > 0) {
        out[o++] = (uint8_t)((acc << (8 - acc_bits)) & 0xFF);
    }
    (void)o;
}

void cs_unpack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out) {
    if (count == 0) return;
    if (bits >= 8) {
        std::memcpy(out, in, count);
        return;
    }
    const int maxv = (1 << bits) - 1;
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t i = 0;
    for (size_t k = 0; k < count; ++k) {
        while (acc_bits < bits) {
            acc = (acc << 8) | in[i++];
            acc_bits += 8;
        }
        acc_bits -= bits;
        const uint32_t q = (acc >> acc_bits) & (uint32_t)maxv;
        acc &= (acc_bits > 0) ? ((1u << acc_bits) - 1u) : 0u;
        out[k] = (uint8_t)((q * 255 + (uint32_t)(maxv / 2)) / (uint32_t)maxv);
    }
}

void cs_pack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out) {
    if (luma_bits == chroma_bits) {
        cs_pack_samples(in, samples * 3, luma_bits, out);
        return;
    }
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < samples * 3; ++i) {
        const int bits = (i % 3 == 1) ? luma_bits : chroma_bits; // G is luma-weighted
        const int maxv = (1 << bits) - 1;
        const uint32_t q = (uint32_t)((in[i] * maxv + 127) / 255);
        acc = (acc << (uint32_t)bits) | q;
        acc_bits += bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            out[o++] = (uint8_t)((acc >> acc_bits) & 0xFF);
        }
    }
    if (acc_bits > 0) out[o] = (uint8_t)((acc << (8 - acc_bits)) & 0xFF);
}

void cs_unpack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out) {
    if (luma_bits == chroma_bits) {
        cs_unpack_samples(in, samples * 3, luma_bits, out);
        return;
    }
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t i = 0;
    for (size_t k = 0; k < samples * 3; ++k) {
        const int bits = (k % 3 == 1) ? luma_bits : chroma_bits;
        const int maxv = (1 << bits) - 1;
        while (acc_bits < bits) {
            acc = (acc << 8) | in[i++];
            acc_bits += 8;
        }
        acc_bits -= bits;
        const uint32_t q = (acc >> acc_bits) & (uint32_t)maxv;
        acc &= (acc_bits > 0) ? ((1u << acc_bits) - 1u) : 0u;
        out[k] = (uint8_t)((q * 255 + (uint32_t)(maxv / 2)) / (uint32_t)maxv);
    }
}

bool cs_seal_header(uint8_t* buf, size_t buf_bytes, const uint8_t key[32]) {
    if (buf_bytes < CS_HEADER_BYTES) return false;
    try {
        HmacSha256 mac(key);
        mac.update(buf, CS_MAC_PREFIX_BYTES);
        // packed writers (marker 0xA7) also cover the pad (mode/geometry +
        // depths); legacy writers leave the pad outside the tag.
        if (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] == CS_SAMPLE_BITS_MARKER) {
            mac.update(buf + CS_OFF_PAD, CS_HEADER_BYTES - CS_OFF_PAD);
        }
        mac.update(buf + CS_HEADER_BYTES, buf_bytes - CS_HEADER_BYTES);
        uint8_t full[32];
        mac.finish(full);
        std::memcpy(buf + CS_OFF_TAG, full, CS_TAG_BYTES);
        cs_wipe(full, sizeof(full));
        return true;
    }
    catch (...) {
        return false;
    }
}

bool cs_verify_header(const uint8_t* buf, size_t buf_bytes, const uint8_t key[32]) {
    if (buf_bytes < CS_HEADER_BYTES) return false;
    try {
        HmacSha256 mac(key);
        mac.update(buf, CS_MAC_PREFIX_BYTES);
        if (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] == CS_SAMPLE_BITS_MARKER) {
            mac.update(buf + CS_OFF_PAD, CS_HEADER_BYTES - CS_OFF_PAD);
        }
        mac.update(buf + CS_HEADER_BYTES, buf_bytes - CS_HEADER_BYTES);
        uint8_t full[32];
        mac.finish(full);
        // constant-time compare of the truncated tag
        uint8_t diff = 0;
        for (int i = 0; i < CS_TAG_BYTES; i++) {
            diff |= (uint8_t)(full[i] ^ buf[CS_OFF_TAG + i]);
        }
        cs_wipe(full, sizeof(full));
        return diff == 0;
    }
    catch (...) {
        return false;
    }
}

bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, const std::string& password, CsHeaderInfo& out) {
    out = CsHeaderInfo{};
    if (buf_bytes < CS_HEADER_BYTES) {
        out.legacy = true;
        return false;
    }

    // v1: ephemeral in-memory tile containers (plaintext metadata, no MAC)
    if (buf[CS_OFF_VERSION] == 1) {
        int vals[5];
        if (!parse_metadata_fields(buf + CS_OFF_META, vals)) return false;
        out.version = 1;
        out.m = vals[0]; out.rows = vals[1]; out.cols = vals[2];
        out.org_h = vals[3]; out.org_w = vals[4];
        return true;
    }

    // v2: authenticated container
    if (buf[CS_OFF_VERSION] != 2) {
        out.legacy = true;
        return false;
    }

    if (!cs_derive_key(password, buf + CS_OFF_SALT, out.key)) {
        out.auth_failed = true;
        return false;
    }
    out.key_valid = true;

    if (!cs_verify_header(buf, buf_bytes, out.key)) {
        out.auth_failed = true;
        cs_wipe(out.key, sizeof(out.key));
        out.key_valid = false;
        return false;
    }

    uint8_t meta[CS_METADATA_BYTES];
    std::memcpy(meta, buf + CS_OFF_META, CS_METADATA_BYTES);
    if (!cs_aes256_ctr_xor(meta, CS_METADATA_BYTES, out.key, buf + CS_OFF_IV)) {
        cs_wipe(meta, sizeof(meta));
        cs_wipe(out.key, sizeof(out.key));
        out.key_valid = false;
        out.auth_failed = true;
        return false;
    }

    int vals[5];
    const bool ok = parse_metadata_fields(meta, vals);
    cs_wipe(meta, sizeof(meta));
    if (!ok) {
        cs_wipe(out.key, sizeof(out.key));
        out.key_valid = false;
        return false;
    }

    out.version = 2;
    out.m = vals[0]; out.rows = vals[1]; out.cols = vals[2];
    out.org_h = vals[3]; out.org_w = vals[4];

    // sampling-mode flags live in the authenticated pad region:
    // [CS_OFF_PAD] = mode (0 random, 1 periodic, 2 adaptive), followed by
    // tile_size and (mode 1) samples_per_tile as 16-bit little-endian values.
    // Mode 2 stores lod byte count in the second slot as a sanity check.
    const uint8_t mode = buf[CS_OFF_PAD];
    if (mode == CS_MODE_YCC420) {
        // luma/chroma-split container. Pure-uniform mode 3 leaves the tile
        // fields zero; the LOD-luma variant reuses the adaptive pad layout
        // (tile_size, lod byte count, weight_base). Accept the LOD variant
        // only when the lod count matches the geometry, otherwise fall back
        // to uniform (tolerant read of pre-LOD containers).
        out.sampling_mode = mode;
        const int tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        const int lodc = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        if (tile > 0 && lodc == cs_lod_bytes(out.rows, out.cols, tile)) {
            out.periodic_tile = tile;
            out.periodic_samples = lodc;
            const int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
                (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
            out.adaptive_base = (base > 0) ? base : 256;
        }
    }
    else if (mode == CS_MODE_YCC420_HF) {
        out.sampling_mode = mode;
        out.periodic_tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        const int lodc = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        const int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
            (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
        out.adaptive_base = base > 0 ? base : 65535;
        out.thumb_len = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN] |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1] << 8) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2] << 16) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3] << 24);
        if (out.periodic_tile <= 0 || lodc != 0) {
            out.legacy = true;
            return false;
        }
    }
    else if (mode == CS_MODE_PERIODIC || mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) {
        out.sampling_mode = mode;
        out.periodic_tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        out.periodic_samples = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        if (mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) {
            int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
                (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
            out.adaptive_base = (base > 0) ? base : 256; // 0 = pre-strength default
        }
        if (out.periodic_tile <= 0 ||
            (mode == CS_MODE_PERIODIC && out.periodic_samples <= 0) ||
            ((mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) &&
                out.periodic_samples != cs_lod_bytes(out.rows, out.cols, out.periodic_tile))) {
            out.sampling_mode = CS_MODE_RANDOM;
            out.periodic_tile = 0;
            out.periodic_samples = 0;
            out.adaptive_base = 256;
            out.thumb_len = 0;
        }
        if (mode == CS_MODE_HF_FOCUS && out.sampling_mode == mode) {
            out.thumb_len = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN] |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1] << 8) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2] << 16) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3] << 24);
        }
    }
    // sample bit-depth travels in the pad for every mode: 0 = legacy writer
    // (zeroed pad) decodes as 8; anything outside 1..8 is a corrupt header.
    {
        if (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] != 0 &&
            buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] != CS_SAMPLE_BITS_MARKER) {
            return false;
        }
        const int sb = buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS];
        if (sb == 0) out.sample_bits = 8;
        else if (sb >= 1 && sb <= 8) out.sample_bits = sb;
        else return false;
        const int cb = buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA];
        if (cb == 0) out.sample_bits_chroma = out.sample_bits;
        else if (cb >= 1 && cb <= 8) out.sample_bits_chroma = cb;
        else return false;
        // Marker + 8/8 is the default writer output (depths always stamped,
        // body left unpacked): plain 8-bit. The marker keeps the pad
        // (mode/geometry) under MAC in every mode.
    }
    // --lod-full detail guarantee travels in the pad for adaptive-family
    // modes; 0 = off. Only honored from stamped (MAC-covered) headers so a
    // legacy zeroed pad always decodes as off.
    out.full_threshold = (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] == CS_SAMPLE_BITS_MARKER)
        ? buf[CS_OFF_PAD + CS_OFF_LOD_FULL] : 0;
    return true;
}

void cs_wipe(void* p, size_t n) {
    volatile uint8_t* v = (volatile uint8_t*)p;
    while (n--) {
        *v++ = 0;
    }
}
