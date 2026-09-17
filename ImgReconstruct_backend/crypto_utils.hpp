#ifndef CS_CRYPTO_UTILS_HPP
#define CS_CRYPTO_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <string>

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

struct CsHeaderInfo {
    int m = 0, rows = 0, cols = 0, org_h = 0, org_w = 0;
    int version = 0;                 // 1 = plain v1, 2 = authenticated v2
    uint8_t key[32] = {};            // derived key (v2 only)
    bool key_valid = false;
    bool auth_failed = false;        // v2 tag mismatch / CNG failure
    bool legacy = false;             // pre-v2 container (no version byte)
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
