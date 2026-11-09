#ifndef IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
#define IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_

#include "image_tiles.hpp"
#include "cs_wavelet.hpp"
#include "crypto_utils.hpp"
#include "cs_gpu.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct indices {
    std::vector<int> ri_x_g, ri_y_g;
    // Averaged measurements (T-means): group lists in tile-local flat
    // pixel indices. rg_off has ng+1 entries; group g covers
    // rg_px[rg_off[g] .. rg_off[g+1]) with true-average weights rg_w[]
    // (0.25f). Empty = legacy point samples (implicit weight 1).
    // Point-mode tiles keep ri_x_g/ri_y_g; grouped tiles leave them
    // empty and set these instead (counts derive from ng).
    std::vector<int> rg_off, rg_px;
    // True-average row weights (1/group-size, i.e. 0.25f): group
    // predictions equal the stored means exactly.
    std::vector<float> rg_w;
};

// Grouped-measurement view for the FISTA data term: ng groups, flat
// tile-local pixel indices and true-average row weights (1/group-size,
// so predictions equal the stored means; the fixed FISTA step stays
// stable because the true Lipschitz only shrinks vs point rows).
// nullptr/empty = legacy point samples. The pointed-to vectors (tile
// lists) outlive the solve.
struct cs_meas_groups {
    const int* off = nullptr;
    const int* px = nullptr;
    const float* w = nullptr;
    int ng = 0;
};

// Noise sigma estimate of a BGR tile in gray levels: Immerkaer Laplacian
// (sigma = sqrt(pi/2) * mean|Lo| / 6) per 32px block, MEDIAN over blocks.
// The median separates grain from texture — texture raises some blocks
// while flat blocks stay at the noise floor, so textured-but-clean tiles
// read low and uniformly grainy tiles read high. Single block for small
// tiles. Returns 0 on degenerate input. Pure function (deterministic).
double cs_estimate_noise_sigma(const cv::Mat& bgr);

// Container section packing for --sample-bits (bit format in
// crypto_utils.hpp): sections record raw offset/count and their plane depth
// (or interleaved BGR luma/chroma depths) in ascending order; everything
// before the first section copies verbatim (header + lod/thumbnail raw
// prefixes); each measurement section packs independently, whole-byte
// padded; trailing slack stays zero. The packed Mat is square (same
// next-perfect-square convention as the raw paths). Unpack mirrors it;
// both throw std::runtime_error on truncation.
cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<cs_body_section>& sections);
cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<std::pair<size_t, size_t>>& sections, int bits);
size_t cs_unpacked_body_bytes(const std::vector<cs_body_section>& sections);
size_t cs_unpacked_body_bytes(const std::vector<std::pair<size_t, size_t>>& sections);
void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<cs_body_section>& sections, uint8_t* raw);
void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<std::pair<size_t, size_t>>& sections, int bits, uint8_t* raw);

std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols);

/** @brief single-channel reweighted-L1 FISTA (exact proximal soft-threshold).
`iterations` is in L-BFGS units and is scaled internally (each outer
reweight gets clamp(iterations*4,16,40) FISTA steps); `reweights` counts
outer passes (first unweighted). DCT/10 warm start in, pixel plane
0..255 out after the tail scale.
*/
// Default sink for the optional next_ref chain output (a temporary cannot
// bind to the non-const reference; never written unless copy_next_ref).
inline cv::Mat& cs_null_mat() { static cv::Mat m; return m; }
void reconstruct_color_channel_fista(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cs_null_mat(),
    float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f,
    bool use_onsager = false, const cs_meas_groups* groups = nullptr);

// Automatic regularization from the Params-test sweep (bird photo, DCT,
// wscale 2, tiles 64/96/128 + full-res spot checks; see Params-test/).
// coef: seg (SEGMASK) follows 0.045*ratio^-0.75 (sparse regions need a
// strong prior); uniform random follows 0.01875/ratio. Both scale by a
// tile-grid factor (num_tiles/64)^0.15 in [0.5, 1.2] — always for seg
// (one global coef compromises across density variation), only below
// ratio 0.5 for uniform (dense uniform tiles are self-similar). tv:
// 0.01 for seg below ratio 0.5 (smoothing helps starved regions), else 0.
// Returns < 0 when the mode is not covered (YCC/adaptive/periodic/HF/
// T-tetromino: caller keeps legacy behavior). Ratio is clamped to
// [0.05, 1.0] (validated 0.15-0.5); coef to [0.01, 0.3]. Recommend
// --per-tile-coef for seg multi-tile solves (measured +1-2 dB).
float cs_auto_reg_coefficient(int sampling_mode, float ratio, int num_tiles);
float cs_auto_reg_tv(int sampling_mode, float ratio);

// Block-aware 2D DCT on a CV_32F plane: dct_block <= 0 is the legacy
// whole-plane transform (bit-identical); >0 applies an independent
// orthonormal DCT per BxB block (edge blocks clipped to the plane).
// Forward and inverse stay exact adjoints either way, so every FISTA
// fixed point only moves via the sparser block prior. All solver-domain
// conversions (warm starts, tails, RED round trips) must use these —
// never raw cv::dct — so analysis and synthesis agree on the basis.
void cs_dct_forward(cv::Mat& plane, int dct_block);
void cs_dct_inverse(cv::Mat& plane, int dct_block);

/** @brief HR-anchor context for the CS-SR fused solve (--hr-detail).
Exact native-luma anchors (global output-geometry indices) enter the HR
coefficient solve as extra data terms beside the LR samples, so the anchor
neighborhoods improve too — not just the anchor pixels (post-hoc
replacement still pins those exactly afterwards). Anchors outside this
tile map to nothing here (neighbor tiles own them). Empty anchors, zero
stride, or zero weight disable the term bit-identically. */
struct CsHrSrCtx {
    const std::vector<CsHrSample>* anchors = nullptr;
    int stride = 0;     // output-geometry row stride (image width)
    int ox = 0, oy = 0; // this tile's origin in output geometry
    float weight = 1.0f;
};

/** @brief CS super-resolution 2x refinement of one solved LR tile (CPU, DCT).
Luminance-only PnP-ADMM (rho 0.1, admm_iters outer passes, clamped [1, 50]):
the x-update is a sparse DCT FISTA solve so the 2x2-box downsample of the HR
luma fits the LR luma samples (BGR samples combined with the BGR2YCrCb
weights) and the exact-HR anchors in hr_sr, with optional smoothed TV on the
HR grid; the z-update is a sigma-matched fastNlMeans prox (identity with
CS_ADMM_PROX_OFF=1). The warm start and the Cr/Cb planes are the AVIR
upscale of the tile, so one HR solve runs instead of three. Falls back to
AVIR when the sample set is empty or the tile is degenerate. lr_tile is the
solved CV_8UC3 tile; hr_out is the 2x CV_8UC3 result. */
void cs_sr_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int fista_iters, float wscale, int admm_iters,
    cv::Mat& hr_out, const CsHrSrCtx& hr_sr = CsHrSrCtx());

/** @brief One tile for cs_sr_gpu_batch: the inputs cs_sr_upscale_tile_2x takes. */
struct CsSrBatchTile {
    const cv::Mat* lr_tile;              // solved LR tile, CV_8UC3
    const cv::Mat* meas;                 // tile container (header + BGR samples)
    const std::vector<int>* ri_x;
    const std::vector<int>* ri_y;
    float coef, tv;
    int iterations, fista_iters;
    float wscale;
    CsHrSrCtx hr_sr;                     // anchors + this tile's origin (ox, oy) in output geometry
    cv::Mat* hr_out;                     // receives the HR tile
};

/** @brief Batched GPU form of cs_sr_upscale_tile_2x: AVIR warm start, YCrCb, luma ADMM and the
BGR merge for every eligible tile in one GPU pass (gpu_sr.h), grouped by ADMM parameters. done[k] = 1
when tile k's hr_out was produced; the rest are left to the per-tile path. Without the GPU build or
with CS_GPU_ADMM off nothing is done. CS_GPU_SR_VERIFY=<n> compares every n-th tile's warm start,
NLM strength and final tile with the CPU functions and prints the differences. `prep`, when given,
is cs_sr_gpu_batch_prepare's result for the same tiles (null = prepared here). */
struct CsSrBatchPrep;
void cs_sr_gpu_batch(std::vector<CsSrBatchTile>& tiles, int admm_iters, std::vector<char>& done,
    CsSrBatchPrep* prep = nullptr);

/** @brief The host part of cs_sr_gpu_batch that does not read the LR pixels (samples, anchor lists, ADMM groups), so a
decode can run it while the LR solve is still on the GPU. The tiles' lr_tile must already have their final size and
type; cs_sr_gpu_batch drops a tile whose size changed. Null when the batch would not run. */
std::shared_ptr<CsSrBatchPrep> cs_sr_gpu_batch_prepare(const std::vector<CsSrBatchTile>& tiles, int admm_iters);

/** Compact HR anchors (mode 19): which 16x16 blocks hold anchors, how many,
and the exact luma they come from. Positions are not stored; both sides draw
them from (flags, key). */
#define CS_ANCHOR_BLOCK 16
struct CsAnchorPlan {
    cv::Mat flags;   // CV_8U, (H / 16) x (W / 16), 1 = block holds anchors
    cv::Mat luma;    // CV_8U Y plane at decrypt output geometry (W x H)
    int count = 0;   // anchors to draw (clamped to the flagged pixel count)
};

/** @brief closed-loop block choice: flag the worst 5% of 16x16 blocks by
MSE between the original and a decode of its container (same size, BGR
8-bit), or more when `count` anchors would fill them past 74% (production's
fill: 450k anchors on 5% of a 12.2 MP photo's blocks), so the share grows
with the count (--hr-scale). Empty Mat when the inputs mismatch or are
smaller than a block. */
cv::Mat cs_error_anchor_flags(const cv::Mat& orig_bgr, const cv::Mat& dec_bgr, int count = 0);

/** @brief plaintext mode-19 anchor payload:
[u32 K][u16 block][u32 W][u32 H][u32 L][L bytes PNG of flags][K luma bytes].
Positions are K distinct pixels of the flagged blocks drawn from the key
(cs_anchor_positions), so only the luma bytes grow with K. Throws on an
empty plan or a PNG encode failure. */
std::vector<uint8_t> cs_anchor_payload(const CsAnchorPlan& plan, const uint8_t key[32]);

/** @brief AES-256-CTR over a mode-19 payload in place (counter = container
IV with counter[0] ^= CS_HRDETAIL_TAG, as the 6-byte payload). Symmetric. */
bool cs_anchor_payload_xor(uint8_t* p, size_t len, const uint8_t key[32], const uint8_t iv[16]);

/** @brief CTR-decrypt + parse a mode-19 payload and regenerate the anchors
(idx in W x H output geometry, ascending). False on any inconsistency. */
bool cs_read_anchor_payload(const uint8_t* src, size_t len, const uint8_t key[32],
    const uint8_t iv[16], std::vector<CsHrSample>& out);

/** @brief apply HR-detail anchors to a BGR image in place: converts to
YCrCb, replaces the Y plane at each in-range anchor index with the stored
value, converts back. Out-of-range indices are skipped (e.g. a half-res
decrypt of a half-res-pipeline container). Anchors whose exact Y plus the
image chroma falls outside BGR gamut (clipped on the way back, Y drifts)
are reverted instead of injecting clip artifacts. No-op on empty
input/payload. Returns the number of anchors applied. */
size_t cs_apply_hr_detail(cv::Mat& bgr, const std::vector<CsHrSample>& s);

#endif  // IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
