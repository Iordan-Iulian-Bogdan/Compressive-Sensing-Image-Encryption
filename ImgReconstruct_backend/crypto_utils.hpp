#ifndef CS_CRYPTO_UTILS_HPP
#define CS_CRYPTO_UTILS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Security design (v2 container format)
//
// Key derivation:
//   The passphrase is run through PBKDF2-HMAC-SHA256 (CS_PBKDF2_ITERATIONS
//   iterations, 16-byte random salt) to produce a 256-bit key. The key is the
//   single root secret: it seeds both the header cipher/MAC and the index
//   RNG. There is no secondary "seed funnel".
//
// Header (v2), 32 pixels = 96 bytes at the start of the encrypted container:
//   [0]      version byte (2)
//   [1..16]  salt (16, random)
//   [17..32] AES-CTR IV (16, random)
//   [33..64] metadata (32 bytes: "m|rows|cols|h|w" '-' padded) XOR AES-256-CTR
//   [65..80] HMAC-SHA256 tag truncated to 16 bytes (encrypt-then-MAC)
//   [81..95] random padding
//
//   The tag covers header bytes [0..64] plus the whole measurement body that
//   follows the header, so any tampering with either is detected.
//
// Random-index generation:
//   MT19937 is seeded via std::seed_seq from the 256-bit key; the Fisher-
//   Yates shuffle seed is the first 4 key bytes. NOTE: MT19937 is NOT
//   cryptographically secure (624-output state recovery); this is documented
//   in the README. The KDF keeps offline guessing expensive, but the stream
//   itself must not be trusted against state-recovery attacks.
//
// Zeroization: derived keys are wiped (cs_wipe) after sealing on the encrypt
//   side and in the CSencryption destructor on the decrypt side.
//
// Legacy (pre-v2) containers have an 11-pixel plaintext header and no
//   version byte; they are rejected with a clear error (re-encrypt required).
// ---------------------------------------------------------------------------

#define CS_HEADER_PIXELS 32          // v2 header size in container pixels
#define CS_HEADER_BYTES (CS_HEADER_PIXELS * 3)
#define CS_METADATA_BYTES 32
#define CS_SALT_BYTES 16
#define CS_IV_BYTES 16
#define CS_TAG_BYTES 16
#define CS_PBKDF2_ITERATIONS 200000

// Offsets inside the 96-byte v2 header
#define CS_OFF_VERSION 0
#define CS_OFF_SALT 1
#define CS_OFF_IV 17
#define CS_OFF_META 33
#define CS_OFF_TAG 65
#define CS_OFF_PAD 81
#define CS_MAC_PREFIX_BYTES 65       // bytes [0..64] covered by the tag prefix

// Fills out with cryptographically random bytes (Windows CNG RNG).
bool cs_random_bytes(uint8_t* out, size_t len);

// Derives a 256-bit key from the passphrase + salt using
// PBKDF2-HMAC-SHA256 (Windows CNG). Returns false on CNG failure.
bool cs_derive_key(const std::string& password, const uint8_t* salt, uint8_t key_out[32]);

// Builds the 32-byte '-'-padded metadata string "m|rows|cols|h|w".
// Throws std::runtime_error if the unpadded text exceeds 32 bytes.
std::string cs_build_metadata(int m, int rows, int cols, int org_h, int org_w);

// Writes a v2 authenticated header into buf (first CS_HEADER_BYTES bytes of a
// linear CV_8UC3 container). salt must have been generated beforehand (the
// caller derives the key from it). The tag region is zeroed; call
// cs_seal_header AFTER the measurement body has been written to buf.
// Returns false if buf is too small or metadata is bad.
bool cs_write_header(uint8_t* buf, size_t buf_bytes, const std::string& metadata, const uint8_t key[32], const uint8_t salt[CS_SALT_BYTES]);

// v1 header for ephemeral in-memory containers (tile re-encryption): plaintext
// metadata, version byte 1, no MAC. The tile path has no password context and
// never leaves process memory.
void cs_write_header_plain(uint8_t* buf, size_t buf_bytes, const std::string& metadata);

// Computes HMAC-SHA256(key, header[0..64] || body[CS_HEADER_BYTES..buf_bytes))
// truncated to CS_TAG_BYTES and stores it at CS_OFF_TAG.
bool cs_seal_header(uint8_t* buf, size_t buf_bytes, const uint8_t key[32]);

// Recomputes the seal and compares it to the stored tag. Returns false on any
// mismatch (tampered header or body, or wrong key).
bool cs_verify_header(const uint8_t* buf, size_t buf_bytes, const uint8_t key[32]);

// Sampling modes recorded in the authenticated pad region (CS_OFF_PAD)
#define CS_MODE_RANDOM    0            // uniform random global indices (pad = 0)
#define CS_MODE_PERIODIC  1            // one pattern repeated per tile (pad = 1)
#define CS_MODE_ADAPTIVE  2            // LOD-based per-tile counts (pad = 2)
#define CS_MODE_YCC420    3            // luma/chroma-split 4:2:0 sampling (pad = 3)
#define CS_MODE_HF_FOCUS  5            // encrypted thumbnail + HF-weighted BGR sampling
#define CS_MODE_YCC420_HF 6            // HF-weighted luma + uniform coarse chroma
#define CS_MODE_YCC422    7            // luma/chroma-split 4:2:2 sampling (pad = 7)
#define CS_MODE_YCC422_HF 8            // HF-weighted luma + uniform 4:2:2 chroma (pad = 8)
#define CS_MODE_BGR_HRDETAIL 4         // uniform random BGR + HR-detail payload (pad = 4)
#define CS_MODE_YCC420_HRDETAIL 9      // uniform YCC420 + HR-detail payload (pad = 9)
#define CS_MODE_SEGMASK 10             // per-segment BGR sampling from a label map (pad = 10)
#define CS_MODE_YCC420_SEG 11          // seg-mask luma + uniform 4:2:0 chroma (pad = 11)
#define CS_MODE_YCC422_SEG 12          // seg-mask luma + uniform 4:2:2 chroma (pad = 12)

// Per-segment sampling (mode 10) body prefix, little-endian, right after the
// 32-pixel header:
//   [0]      nclasses (u8, 1..255)
//   [1..n]   per-class quantized rates (u8: count_c = round(pix_c * q/255))
//   [1+n..]  label-map PNG length (u32)
//   [...]    label-map PNG bytes (8-bit grayscale class ids, pipeline geometry)
// Measurements (3 bytes each, BGR) follow, padded to whole pixels. Rates are
// quantized AFTER the global --ratio scale on encrypt so both sides derive
// identical counts from the shipped bytes; decrypt needs no extra flags.
struct CsSegPrefix {
    uint8_t nclasses = 0;
    std::vector<uint8_t> rates_q;
    uint32_t seglen = 0;
    size_t total = 0; // 1 + nclasses + 4 + seglen
};
inline bool cs_segmask_parse_prefix(const uint8_t* body, size_t body_bytes, CsSegPrefix& out) {
    out = CsSegPrefix{};
    if (body_bytes < 5) return false;
    const uint8_t n = body[0];
    if (n == 0) return false;
    const size_t head = (size_t)1 + n + 4;
    if (body_bytes < head) return false;
    out.nclasses = n;
    out.rates_q.assign(body + 1, body + 1 + n);
    out.seglen = (uint32_t)body[1 + n] |
        ((uint32_t)body[2 + n] << 8) |
        ((uint32_t)body[3 + n] << 16) |
        ((uint32_t)body[4 + n] << 24);
    if (out.seglen == 0 || out.seglen > (1u << 24)) return false;
    if (head + out.seglen > body_bytes) return false;
    out.total = head + out.seglen;
    return true;
}
// Quantize a [0,1] sampling rate to the shipped u8 (both sides derive counts
// from these bytes, so encrypt must quantize before drawing, not after).
inline uint8_t cs_segmask_quantize_rate(float r) {
    if (!(r > 0.0f)) return 0;
    if (r >= 1.0f) return 255;
    return (uint8_t)std::lround(r * 255.0f);
}

// HR-detail payload (--hr-detail): K native-resolution luma anchors stored
// as a trailing body section, 6 bytes each (u32 LE output-geometry linear
// index + u8 luma + 1 zero pad byte = 2 container pixels), AES-256-CTR
// encrypted with the container key and a domain-separated counter
// (IV with counter[0] ^= CS_HRDETAIL_TAG). Length is implicit:
// (body_bytes - measurement_bytes) / CS_HRDETAIL_SAMPLE_BYTES. Covered by
// the container HMAC like the measurement body. Decrypt applies the anchors
// as exact post-composite luma replacements.
#define CS_HRDETAIL_TAG 0x48
#define CS_HRDETAIL_SAMPLE_BYTES 6
#define CS_HRDETAIL_MAX_SAMPLES 65536
// Offset from CS_OFF_PAD for the modes 4/9 HR sample count (u32 LE,
// 1..CS_HRDETAIL_MAX_SAMPLES). Aliases the HF thumbnail-length field,
// which is unused outside modes 5/6/8. MAC-covered via the stamped marker.
#define CS_OFF_HR_COUNT 7

// One HR-detail anchor: linear index into the decrypt output geometry
// (row-major, output width stride) + native luma value.
struct CsHrSample {
    uint32_t idx = 0;
    uint8_t y = 0;
};

// Adaptive pad layout (mode = 2), all little-endian:
//   [0]      mode (= 2)
//   [1..2]   tile_size
//   [3..4]   lod byte count (sanity vs geometry)
//   [5..6]   weight_base (uint16): w = tile_pixels * (weight_base + lod)
//            0 is treated as the default 256 on parse (pre-strength containers)
#define CS_OFF_ADAPTIVE_BASE 5         // offset from CS_OFF_PAD
#define CS_OFF_HF_THUMBLEN 7          // modes 5/6/8: encrypted thumbnail length (u32 LE)
#define CS_OFF_SAMPLE_BITS 11         // offset from CS_OFF_PAD: bits per stored
                                      // sample value, 1..8 (0 = legacy 8-bit)
#define CS_OFF_SAMPLE_BITS_CHROMA 12  // chroma bits; 0 = legacy/same as luma
#define CS_OFF_SAMPLE_BITS_MARKER 13  // authenticated bit-packing extension marker
#define CS_SAMPLE_BITS_MARKER 0xA7
#define CS_OFF_LOD_FULL 14          // offset from CS_OFF_PAD: --lod-full
                                    // detail threshold 0..255 (0 = off)
#define CS_OFF_ADAPTIVE_CUTOFF 7    // offset from CS_OFF_PAD: --adaptive-cutoff
                                    // hard LOD floor 0..255 (0 = off). Aliases
                                    // the HF-thumbnail/HR-count u32 field,
                                    // unused by adaptive-family modes (2 and
                                    // LOD-luma mode 3); honored only from
                                    // stamped (MAC-covered) headers.

// Maps --adaptive-strength in [0, 1] to the integer weight_base.
//   0.0 -> 65535 (essentially uniform: density ratio ~1.004)
//   0.5 -> 256   (tuned default: density ratio ~1.8x)
//   1.0 -> 1     (max LOD bias: density ratio ~7.8x)
inline int cs_adaptive_base_from_strength(double strength) {
    if (!(strength > 0.0)) return 65535;
    if (strength >= 1.0) return 1;
    const double b = std::exp(std::log(65535.0) * (1.0 - strength));
    int v = (int)std::lround(b);
    if (v < 1) v = 1;
    if (v > 65535) v = 65535;
    return v;
}

// LOD contrast exponent (--adaptive-gamma): reshapes shipped detail bytes
// as lod_min + round(((v - lod_min) / (255 - lod_min))^gamma * (255 - lod_min)).
// gamma == 1 is identity; > 1 pushes mid-detail down (stronger edge focus,
// past what --adaptive-strength 1.0 + --adaptive-floor 0 can express);
// < 1 flattens toward uniform. Floor and ceiling are fixed points.
// Applied encrypt-side to the final shipped bytes (after scoring, region
// blend, and smoothing), so decrypt recomputes identical counts with no
// format change. No-op when gamma == 1 or the range is degenerate.
inline void cs_apply_lod_gamma(std::vector<uint8_t>& lod, int lod_min, double gamma) {
    if (!(gamma > 0.0) || gamma == 1.0) return;
    const double den = 255.0 - (double)lod_min;
    if (den <= 0.0) return;
    for (size_t i = 0; i < lod.size(); ++i) {
        if (lod[i] <= lod_min) continue; // floor fixed, sub-floor preserved
        const double t = ((double)lod[i] - (double)lod_min) / den;
        const double tc = t > 1.0 ? 1.0 : t;
        int v = lod_min + (int)std::lround(std::pow(tc, gamma) * den);
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        lod[i] = (uint8_t)v;
    }
}

// lod region: tiles_x * tiles_y bytes after the 32-pixel header, padded to a
// whole number of container pixels so measurements stay pixel-aligned
inline int cs_lod_tile_count(int rows, int cols, int tile_size) {
    if (tile_size <= 0) return 0;
    const int tx = (cols + tile_size - 1) / tile_size;
    const int ty = (rows + tile_size - 1) / tile_size;
    return tx * ty;
}
inline int cs_lod_bytes(int rows, int cols, int tile_size) {
    return cs_lod_tile_count(rows, cols, tile_size);
}
inline int cs_lod_pixels(int rows, int cols, int tile_size) {
    const int b = cs_lod_bytes(rows, cols, tile_size);
    return (b + 2) / 3;               // ceil(b / 3)
}

// YCC 4:2:0 split sampling (mode = 3): luma is sampled at full resolution
// (m_luma positions) while each chroma plane is first downsampled 2x and
// then sampled on its own coarse grid. The per-plane chroma count derives
// deterministically from the stored luma budget so no extra header fields
// are needed — both sides must use cs_ycc420_chroma_count. Container body
// layout after the 32-pixel header is raw bytes:
//   [Y x m_luma][Cr x m_chroma][Cb x m_chroma], zero-padded to whole pixels.
// At the same --ratio this stores ~1.5 bytes/px instead of 3 (BGR modes).
//
// YCC 4:2:2 (modes 7/8) is the same construction with chroma downsampled
// horizontally only (full height, half width): ~2 bytes/px, sharper chroma
// edges than 4:2:0 at a larger container. cs_chroma_dims/cs_chroma_count
// parameterize all four split modes (3/6 halve both axes, 7/8 halve width
// only); the cs_ycc420_* names remain for older 4:2:0-only call sites.
inline void cs_chroma_dims(int mode, int rows, int cols, int& crows, int& ccols) {
    if (mode == CS_MODE_YCC422 || mode == CS_MODE_YCC422_HF) {
        crows = rows;
        ccols = (cols + 1) / 2;
    } else {
        crows = (rows + 1) / 2;
        ccols = (cols + 1) / 2;
    }
}
inline int cs_chroma_count(int mode, int m_luma, int rows, int cols) {
    int crows, ccols;
    cs_chroma_dims(mode, rows, cols, crows, ccols);
    if (rows <= 0 || cols <= 0) return 0;
    int mc = (int)(((long long)m_luma * crows * ccols) / ((long long)rows * cols));
    if (mc < 1) mc = 1;
    const int cap = crows * ccols;
    if (mc > cap) mc = cap;
    return mc;
}
// Any luma/chroma-split container (uniform, adaptive-luma, HF-luma, or
// HR-detail).
inline bool cs_is_ycc(int mode) {
    return mode == CS_MODE_YCC420 || mode == CS_MODE_YCC422 ||
        mode == CS_MODE_YCC420_HF || mode == CS_MODE_YCC422_HF ||
        mode == CS_MODE_YCC420_HRDETAIL ||
        mode == CS_MODE_YCC420_SEG || mode == CS_MODE_YCC422_SEG;
}
// Any container carrying a trailing HR-detail payload.
inline bool cs_is_hrdetail(int mode) {
    return mode == CS_MODE_BGR_HRDETAIL || mode == CS_MODE_YCC420_HRDETAIL;
}
inline bool cs_is_ycc422(int mode) {
    return mode == CS_MODE_YCC422 || mode == CS_MODE_YCC422_HF ||
        mode == CS_MODE_YCC422_SEG;
}
inline bool cs_is_ycc_hf(int mode) {
    return mode == CS_MODE_YCC420_HF || mode == CS_MODE_YCC422_HF;
}
inline void cs_ycc420_chroma_dims(int rows, int cols, int& crows, int& ccols) {
    cs_chroma_dims(CS_MODE_YCC420, rows, cols, crows, ccols);
}
inline int cs_ycc420_chroma_count(int m_luma, int rows, int cols) {
    return cs_chroma_count(CS_MODE_YCC420, m_luma, rows, cols);
}
// Domain separation for the three per-plane RNG streams: luma uses the key
// as-is (tag 0), chroma planes use tagged copies. Deterministic on both
// sides; the tags are fixed constants, not secrets.
inline void cs_ycc420_channel_key(const uint8_t in[32], uint8_t tag, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i) out[i] = in[i];
    out[0] ^= tag;
    out[16] ^= tag;
}
#define CS_YCC420_TAG_CR 0xA5
#define CS_YCC420_TAG_CB 0x5A

struct CsHeaderInfo {
    int m = 0, rows = 0, cols = 0, org_h = 0, org_w = 0;
    int version = 0;                 // 1 = plain v1, 2 = authenticated v2
    uint8_t key[32] = {};            // derived key (v2 only)
    bool key_valid = false;
    bool auth_failed = false;        // v2 tag mismatch / CNG failure
    bool legacy = false;             // pre-v2 container (no version byte)
    int sampling_mode = CS_MODE_RANDOM; // CS_MODE_* from the pad byte
    int periodic_tile = 0;           // > 0: tile size (periodic or adaptive)
    int periodic_samples = 0;        // mode 1: samples per tile; mode 2: lod byte count (sanity)
    int adaptive_base = 256;         // mode 2: weight_base (w = cap * (base + lod))
    uint32_t thumb_len = 0;          // modes 5/6: encrypted thumbnail bytes in body
    bool hr_detail = false;          // modes 4/9: trailing HR-detail payload present
    uint32_t hr_count = 0;           // modes 4/9: HR sample count (payload = 6 * count bytes)
    int sample_bits = 8;             // bits per stored sample value (pad+11, 0 = legacy 8)
    int sample_bits_chroma = 8;      // Cr/Cb (pad+12, 0 = same as luma)
    int full_threshold = 0;          // --lod-full detail guarantee (pad+14, 0 = off)
    int adaptive_cutoff = 0;         // --adaptive-cutoff hard floor (pad+7, 0 = off)
};

// Sample bit-depth reduction (--sample-bits): measurement values are
// plaintext in secret order, so the sequence is incompressible as stored —
// fewer bits per value is the structural size lever (besides ratio/planes).
// Quantize maps 8-bit values onto N bits (rounded), dequantize scales back;
// N = 8 is the identity in both directions. Pack/unpack lay values out
// MSB-first, each section independently whole-byte padded (pad bits zero).
inline int cs_sample_bits_valid(int bits) { return bits >= 1 && bits <= 8; }
inline size_t cs_packed_bytes(size_t count, int bits) {
    if (count == 0 || bits >= 8) return count;
    return (count * (size_t)bits + 7) / 8;
}
inline size_t cs_packed_bgr_bytes(size_t samples, int luma_bits, int chroma_bits) {
    return (samples * (size_t)(luma_bits + 2 * chroma_bits) + 7) / 8;
}
inline uint8_t cs_quantize_sample(uint8_t v, int bits) {
    if (bits >= 8) return v;
    const int maxv = (1 << bits) - 1;
    return (uint8_t)((v * maxv + 127) / 255);
}
inline uint8_t cs_dequantize_sample(uint8_t q, int bits) {
    if (bits >= 8) return q;
    const int maxv = (1 << bits) - 1;
    return (uint8_t)((q * 255 + maxv / 2) / maxv);
}

// Bit-level sample codec (MSB-first; out must hold cs_packed_bytes /
// cs_packed_bgr_bytes). The _bgr variants interleave per-value depths: G
// uses luma_bits, B/R use chroma_bits (equal depths collapse to the plain
// path, keeping the legacy stream bit-identical).
void cs_pack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);
void cs_unpack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);
void cs_pack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out);
void cs_unpack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out);

// One packed body section: raw byte offset + value count + plane depth(s).
// interleaved_bgr = flat BGR triplets (G = luma depth, B/R = chroma depth);
// otherwise a single plane at bits. Sections must be ascending; everything
// before the first section copies verbatim (header + lod/thumbnail prefixes).
struct cs_body_section {
    size_t offset;
    size_t count;
    int bits;
    int chroma_bits;
    bool interleaved_bgr;
};

// Parses (+ authenticates and decrypts for v2) the header from the raw bytes
// of a linear CV_8UC3 container. password may be empty for v1 tile containers.
// Returns false for legacy containers, undecryptable v2 headers or failed
// authentication (check out.auth_failed / out.legacy for the reason).
bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, const std::string& password, CsHeaderInfo& out);

// In-place AES-256-CTR stream XOR (Windows CNG). counter is the 16-byte big-
// endian initial counter block; it is wiped before return on success.
bool cs_aes256_ctr_xor(uint8_t* data, size_t len, const uint8_t key[32], const uint8_t counter[16]);

// Cryptographically wipes a memory range (volatile write loop).
void cs_wipe(void* p, size_t n);

#endif
