#ifndef IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
#define IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_

#include "image_tiles.hpp"
#include "cs_wavelet.hpp"
#include "crypto_utils.hpp"
#include "cs_gpu.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#define MANUAL_PARAM  1
#define AUTO_PARAM  2

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

// Per-tile measurement count for regularization scaling: grouped tiles
// count groups, point tiles count pixels.
inline int cs_tile_meas_count(const indices& idx) {
    if (!idx.rg_off.empty()) return (int)idx.rg_off.size() - 1;
    return (int)idx.ri_x_g.size();
}

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

int nextClosestDivisible(int x, int y);

// Per-tile regularization scaling (--per-tile-coef, --per-tile-tv): tiles
// with fewer measurements than average get a stronger prior and vice versa,
// scale = sqrt(m_mean / m_tile) clamped to [0.25x, 4x]. Degenerate inputs
// (empty tile, non-positive mean) return the base value unchanged. tv shares
// the law via its own flag; per-tile TV can draw the tile grid on smooth
// gradients, so the two are independently opt-in.
inline float cs_per_tile_coef(float coef, int m_tile, double m_mean) {
    if (m_tile <= 0 || m_mean <= 0.0 || coef <= 0.0f) return coef;
    double f = std::sqrt(m_mean / (double)m_tile);
    if (f < 0.25) f = 0.25;
    if (f > 4.0) f = 4.0;
    return (float)(coef * f);
}

// Grain-adaptive TV strength (--grain-tv): ROF lambda from measured tile
// noise over the reference (2x excess; 0 at/below reference). Calibrated:
// +1dB-class gain on sigma-25 synthetic noise (higher lambdas plateau
// there), clean content untouched.
inline float cs_grain_tv_lambda(double sigma, float ref) {
    if (!(ref > 0.0f) || !(sigma > (double)ref)) return 0.0f;
    return (float)(2.0 * (sigma - (double)ref));
}

// Noise sigma estimate of a BGR tile in gray levels: Immerkaer Laplacian
// (sigma = sqrt(pi/2) * mean|Lo| / 6) per 32px block, MEDIAN over blocks.
// The median separates grain from texture — texture raises some blocks
// while flat blocks stay at the noise floor, so textured-but-clean tiles
// read low and uniformly grainy tiles read high. Single block for small
// tiles. Returns 0 on degenerate input. Pure function (deterministic).
double cs_estimate_noise_sigma(const cv::Mat& bgr);

// ROF TV denoise of one BGR tile, luma-only (chroma passes through):
// Chambolle projection, lambda in pixel units, fixed iteration budget.
// Deterministic; no-op on degenerate input. Used by --grain-tv to smooth
// only tiles whose measured grain exceeds the reference.
void cs_tv_denoise_tile(cv::Mat& bgr, float lambda, int iters = 100);

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

/** @brief solver selector for the tile reconstruction (see --solver).
FISTA (1) solves each channel independently with reweighted-L1 FISTA;
FISTA_JOINT (2) solves all three channels together with SOMP-structured
joint (group-L2,1) sparsity, i.e. one common DCT support shared across
R/G/B plus reweighting.
*/
enum CsSolver {
    CS_SOLVER_FISTA = 1,
    CS_SOLVER_FISTA_JOINT = 2
};

int cs_solver_from_name(const std::string& name, int& out);

/** @brief solve-stage profiler (opt-in, env CS_PROFILE=1).
Atomic nanosecond accumulators shared by decrypt_tiles and the solver
cores. cs_solveprof_reset()/cs_solveprof_dump() bracket a decrypt_tiles
call; dump prints one "profile[solve ...]" stderr line. Inactive until
reset sees the env var, so normal runs pay only a predictable branch. */
struct cs_solve_profile {
    bool on = false;
    // decrypt_tiles structure
    std::atomic<long long> warm_ns{ 0 };      // build_neighbor_warm_start
    std::atomic<long long> ctor_ns{ 0 };      // decrypt_image(tile) construction
    std::atomic<long long> call_ns{ 0 };      // decrypt(): channel solves + merge/convert
    std::atomic<long long> wave_wall_ns{ 0 }; // sum of per-wave wall times
    std::atomic<long long> wave_cap_ns{ 0 };  // wall x usable threads (capacity; work/cap = efficiency)
    std::atomic<long long> tiles{ 0 };
    std::atomic<long long> samples{ 0 };      // sum of m across channel-wrapper calls
    // channel wrapper (fista/joint share these buckets)
    std::atomic<long long> wrap_setup_ns{ 0 }, wrap_core_ns{ 0 }, wrap_rw_ns{ 0 }, wrap_tail_ns{ 0 };
    std::atomic<long long> wrap_calls{ 0 }, rw_passes{ 0 };
    // FISTA core
    std::atomic<long long> fs_grad_ns{ 0 }, fs_shrink_ns{ 0 }, fs_mom_ns{ 0 };
    std::atomic<long long> fs_iters{ 0 };
};
extern cs_solve_profile g_solveprof;
void cs_solveprof_reset();
void cs_solveprof_dump(int num_tiles, int num_threads, long long wall_ns);
inline long long cs_solveprof_ns_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

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

// Block-aware 2D DCT on a CV_32F plane: dct_block <= 0 is the legacy
// whole-plane transform (bit-identical); >0 applies an independent
// orthonormal DCT per BxB block (edge blocks clipped to the plane).
// Forward and inverse stay exact adjoints either way, so every FISTA
// fixed point only moves via the sparser block prior. All solver-domain
// conversions (warm starts, tails, RED round trips) must use these —
// never raw cv::dct — so analysis and synthesis agree on the basis.
void cs_dct_forward(cv::Mat& plane, int dct_block);
void cs_dct_inverse(cv::Mat& plane, int dct_block);

// Second Y-only CPU FISTA pass with a smoothed sample-density confidence map.
// plane is CV_32F [0,255] in and out; measured pixels retain their data term.
void cs_refine_y_confidence(const cv::Mat& measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& plane, float strength, float coef, float tv,
    int iterations, int fista_iters, int basis, float wscale);

// Detect mixed-density reconstruction tiles; uniform tiles need one solve.
bool cs_y_mixed_density(const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    int rows, int cols);

// Post-composite segment mask: strong Sobel edges from the stable high-TV
// image are closed/flood-filled and hue-connected where supported by samples.
// Hard BGR selection: white takes the low-TV composite, black the high-TV
// composite. When weight_out is non-null it receives this exact binary mask
// as CV_32F (0 or 1), for saving to PNG.
void cs_blend_y_dual_tv(const cv::Mat& low_bgr, const cv::Mat& high_bgr,
    const cv::Mat& sample_mask, cv::Mat& blended, cv::Mat* weight_out = nullptr);

// Classify solved tiles by dual-TV blend weight: per-tile white fraction of
// the binary weight mask (white = low-TV solve chosen) split against the
// mean tile fraction (above -> lowtv canvas, below -> hightv, ties in
// neither). Renders two full-geometry BGR canvases from the composite at
// the tiles' rects, black elsewhere, each labeled with its tile count. Pure
// function of inputs (deterministic); returns false on degenerate input
// (geometry/type mismatch, empty grids, nothing to split).
bool cs_build_dual_tv_class_views(const cv::Mat& composite, const cv::Mat& weight,
    const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    cv::Mat& lowtv, cv::Mat& hightv);

// Pixel-exact cutouts of a dual-TV blend: lowcut keeps the low-candidate
// pixels where the binary weight is 1, highcut keeps the blended pixels
// where it is 0 (== the high candidate there, by the blend contract);
// black elsewhere. Accepts CV_32F 0/1 or CV_8U 0/255 weights. Pure function
// of inputs (deterministic); returns false on degenerate input (geometry/
// type mismatch, empty mats).
bool cs_build_dual_tv_cutouts(const cv::Mat& blended, const cv::Mat& low,
    const cv::Mat& weight, cv::Mat& lowcut, cv::Mat& highcut);

/** @brief two-phase per-channel FISTA: submit all channels of a tile (or
wave) before waiting for any of them, so the GPU worker's batch queue stays
full and its pipelined kernels overlap. Begin prepares the problem and
enqueues it in GPU mode (no wait); in CPU mode it runs the full solve+tail
inline. End blocks for the GPU result and runs the same IDCT+255 tail.
Channels are independent: each warm-starts from the neighbor field (no
cross-channel copy chain), which is what lets them be in flight together.
*/
struct fista_channel_task {
    cs_gpu::FistaProblem pb {};
    std::vector<float> b;
    int basis = CS_BASIS_DCT;
    float wscale = 2.0f;
    const cs_meas_groups* groups = nullptr;  // grouped (averaged) measurements, else null
    cs_gpu::FistaHandle h = nullptr;  // null => already done (CPU/fallback)
};
void fista_channel_begin(fista_channel_task& t, const cv::Mat& pixel_measurements,
    const int& channel, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations,
    cv::Mat& ref, float tv = 0.0f, int reweights = 2, int fista_iters = 0,
    int basis = CS_BASIS_DCT, float wscale = 2.0f, bool use_onsager = false,
    const cs_meas_groups* groups = nullptr);
void fista_channel_end(fista_channel_task& t, cv::Mat& ref);

/** @brief SOMP-structured joint RGB solve: stacked group-L2,1 FISTA with
reweighting. refs[3] hold per-channel DCT/10 warm starts in, solved pixel
planes (CV_32F, 0..255, full tile size) out, ready for cv::merge.
*/
void reconstruct_image_fista_joint(const cv::Mat& pixel_measurements, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat refs[3], float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f,
    const cs_meas_groups* groups = nullptr);

/** @brief true for the CS super-resolution upscaler backends ("cs", "cs-sr").
Selected via --photo-upscaler; handled as a fused 2x path in the decrypt
pipeline (like "avir", not a subprocess pipe). */
bool cs_is_superres_backend(const std::string& backend);

/** @brief true for the single-stage SR backend ("sr-direct").
Selected via --photo-upscaler; handled as a fused 2x path in the decrypt
pipeline (like "avir", not a subprocess pipe). */
bool cs_is_srdirect_backend(const std::string& backend);

/** @brief true for the untrained-prior SR backend ("dip").
Same fused 2x handling as cs-sr (luma-only solve, ycc via luma path). */
bool cs_is_dip_backend(const std::string& backend);

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
    float red_radius = 0.0f; // anchor-proximity RED radius in HR px (0 = uniform RED)
};

/** @brief CS super-resolution 2x refinement of one solved LR tile (CPU, DCT).
Luminance-only: a sparse DCT correction around an AVIR warm start is solved
so the 2x2-box downsample of the HR luma fits the original LR luma samples
(BGR samples combined with the BGR2YCrCb weights), with a quadratic AVIR
anchor and optional smoothed TV on the HR grid. (An explicit iterative
back-projection closure was trialed and removed: pulling toward the noisy
LR solve hurt SSIM with no PSNR gain; the anchor already provides dense
consistency.) Chroma comes straight from the AVIR upscale, so one HR solve
runs instead of three. red in [0,1] adds a RED-lite proximal step per
outer pass (0 = off, bit-identical): "nlmeans" blends luma toward
fastNlMeans, "dncnn"/"scunet"/"restormer" blend the BGR tile toward the
ONNX model output (red_model overrides the per-denoiser default path).
red_decay in [0,1] schedules the blend per pass (HQS-style: 1.0 keeps all
passes identical, <1 decays geometrically). Falls back to AVIR when the basis is
not DCT, the sample set is empty, or the tile is degenerate. coef/tv/iterations/reweights/
basis/wscale mirror the LR solve conventions. lr_tile is the solved CV_8UC3
tile; hr_out is the 2x CV_8UC3 result. hr_sr carries tile-mapped HR-detail
anchors as in-solve data terms (weight 0 = post-hoc replacement only). */
// Default DnCNN color model (repo-vendored) for the dncnn RED denoiser.
#define CS_DNCNN_DEFAULT_MODEL "models/dncnn/dncnn_color.onnx"
// Default blind color models for the scunet/restormer RED denoisers (not
// vendored: models/ is gitignored; see README for download sources).
#define CS_SCUNET_DEFAULT_MODEL "models/scunet/scunet_color_real_psnr.onnx"
#define CS_RESTORMER_DEFAULT_MODEL "models/restormer/restormer_color.onnx"

// Blind ONNX image-denoiser kinds sharing one RGB-NCHW clean-output
// contract (BGR in/out at the API boundary): dncnn (tiny, 512px runner
// tiles) vs scunet/restormer (large, 256px runner tiles).
enum cs_onnx_kind {
    CS_ONNX_DNCNN = 0,
    CS_ONNX_SCUNET = 1,
    CS_ONNX_RESTORMER = 2
};

// Map a --red-denoiser name to its kind (-1 when not an ONNX denoiser).
inline int cs_onnx_kind_for(const std::string& denoiser) {
    std::string s = denoiser;
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    if (s == "dncnn") return CS_ONNX_DNCNN;
    if (s == "scunet") return CS_ONNX_SCUNET;
    if (s == "restormer") return CS_ONNX_RESTORMER;
    return -1;
}

// Resolve the model file for a RED denoiser: explicit --red-model wins,
// else the dncnn path for dncnn (back-compat with --dncnn-model), else the
// per-kind default. Empty when the denoiser needs no model (nlmeans).
inline std::string cs_red_model_for(const std::string& denoiser,
    const std::string& dncnn_model, const std::string& red_model) {
    if (!red_model.empty()) return red_model;
    const int kind = cs_onnx_kind_for(denoiser);
    if (kind == CS_ONNX_DNCNN)
        return dncnn_model.empty() ? std::string(CS_DNCNN_DEFAULT_MODEL) : dncnn_model;
    if (kind == CS_ONNX_SCUNET) return std::string(CS_SCUNET_DEFAULT_MODEL);
    if (kind == CS_ONNX_RESTORMER) return std::string(CS_RESTORMER_DEFAULT_MODEL);
    return std::string();
}

void cs_sr_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int reweights, int fista_iters, int basis, float wscale,
    cv::Mat& hr_out, float red = 0.0f,
    const std::string& red_denoiser = "nlmeans",
    const std::string& dncnn_model = CS_DNCNN_DEFAULT_MODEL,
    bool sr_cascade = false, const std::string& red_model = "",
    float red_decay = 1.0f, const CsHrSrCtx& hr_sr = CsHrSrCtx());

/** @brief single-stage SR 2x refinement of one tile (CPU, DCT): the HR luma
is solved directly against the tile's LR samples (cs_sr_direct_luma, no
LR-solve warm start) and merged with the AVIR chroma, so one HR solve runs
instead of three. sr_cascade selects the warm start: false = cold zeros
with no anchor (init_mode 0), true = multiscale cascade chain
(cs_sr_cascade_init, init_mode 2, anchored). Falls back to AVIR when the basis is
not DCT, the sample set is empty, or the tile is degenerate. YCC
containers refine luma-only against the tile's LR luma samples (the BGR2Y
weights collapse to 1.0 on the gray-as-BGR luma container); chroma comes
from the AVIR upscale inside. */
void cs_sr_direct_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, float tv, int iterations, int reweights, int fista_iters, int basis, float wscale,
    cv::Mat& hr_out, bool sr_cascade = false);

/** @brief untrained-prior (Deep Decoder style) 2x refinement of one solved
LR tile (CPU): a tiny randomly-initialized generator (nearest-upsample +
per-pixel channel mixes + leaky ReLU, final sigmoid; underparameterized vs
the HR pixels) is fit by Adam against the tile's LR luma samples (2x2-box
forward model) plus an AVIR anchor, starting from a fixed seed, with early
stopping on the data-loss plateau (the DIP regularizer: structure fits
before noise, so grainy tiles stop before fitting speckles; iters is the
max budget). No
training data, no weights, no basis involved. Luma-only solve merged with
the AVIR chroma (same contract as the cs-sr path). Falls back to AVIR on
degenerate input or a non-finite result. Deterministic. */
void cs_dip_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& hr_out, float anchor_w = 0.05f, int iters = 400, float step = 0.04f);

/** @brief true when the DnCNN model loads and runs (cached per path).
Thread-safe; loads lazily on first call. */
bool cs_dncnn_available(const std::string& model_path);

/** @brief denoise one BGR 8U image with the DnCNN color model (tiled
internally so activation memory stays bounded). Input/output [0,255].
Returns false when the model cannot be loaded or run (caller falls back).
Thread-safe; deterministic on CPU. */
bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path);

/** @brief same RGB-NCHW clean-output contract for any blind ONNX color
denoiser (scunet/restormer share the DnCNN runner with smaller tiles).
kind selects messages and tile geometry (see cs_onnx_kind). */
bool cs_onnx_available(const std::string& model_path, int kind);
bool cs_onnx_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int kind);

/** @brief single-stage HR solve for experiments (see .cpp): full
HR-coefficient FISTA against the LR samples with no LR-solve warm start.
b/rix/riy/m are the LR samples; hr_out holds Hr=2*Lr floats in [0,1]
(w caller-allocated). init_mode 0 = zeros, 1 = AVIR demosaic (+anchor when
anchor_w > 0). VERDICT (measured): a 4x-budget cold solve beats two-stage
+2.5 dB on small synthetic tiles but loses on real photos (warm-start
economics dominate; a production --sr-direct path was built on this entry
and removed after measuring worse on IMG_3690 in every configuration).
The cascade init below answers it: cascade+1x matches the 4x-budget solve,
so a production retry should warm-start from the cascade, not from zeros.
Kept as the experiment harness. */
void cs_sr_direct_luma(const float* b, const int* rix, const int* riy, int m,
    int Lr, int Lc, float coef, float tv, int iterations, int reweights,
    int fista_iters, float wscale, int init_mode, float anchor_w,
    float* hr_out, const float* warm_px = nullptr);

/** @brief multiscale cascade warm start for single-stage SR (see .cpp):
bins the LR samples into progressively coarser cells, runs one short
unweighted DCT-FISTA pass per level coarse-to-fine, and bilinearly
upscales each solution to seed the next. hr_init holds Hr=2*Lr [0,1]
pixels (caller-allocated). Pure function of the samples (deterministic);
fista_iters overrides the fixed per-level budget (clamped [4,64]). */
void cs_sr_cascade_init(const float* b, const int* rix, const int* riy, int m,
    int Lr, int Lc, float coef, int fista_iters, float wscale, float* hr_init);

/** @brief anchor-proximity RED map for one HR tile: per-pixel factor in
[0,1] scaling the RED-lite blend. Exact Euclidean distance from each pixel
to the nearest tile-local anchor (openCV distance transform), mapped as
1-exp(-d^2/2r^2): 0 at anchor pixels (exact truth needs no denoising pull),
~0.39 at one radius, ~0.86 at two. Empty when the anchor list is empty or
radius <= 0 (caller keeps legacy uniform RED). Deterministic. */
cv::Mat cs_hr_red_map(int Hr, int Wc, const std::vector<int>& anchor_idx, float radius);

/** @brief select up to count HR-detail anchors from a native BGR image.
Resizes the reference to out_w x out_h when it differs (decrypt output
geometry), ranks pixels by Sobel gradient magnitude on luma with per-cell
(16x16) top-2 bucketing for spatial spread, and keeps the global top
`count`. Returns fewer when the image holds fewer candidates. Empty when
count <= 0 or the image is degenerate. Deterministic. */
std::vector<CsHrSample> cs_select_hr_detail(const cv::Mat& native_bgr,
    int out_w, int out_h, int count);

/** @brief serialize + AES-256-CTR encrypt an HR-detail payload into dst
(CS_HRDETAIL_SAMPLE_BYTES per sample; dst must hold at least that many
bytes). Counter is the container IV with counter[0] ^= CS_HRDETAIL_TAG.
Throws std::runtime_error on CNG failure. */
void cs_write_hr_payload(uint8_t* dst, const std::vector<CsHrSample>& s,
    const uint8_t key[32], const uint8_t iv[16]);

/** @brief CTR-decrypt + parse an HR-detail payload. len must be a positive
multiple of CS_HRDETAIL_SAMPLE_BYTES with at most CS_HRDETAIL_MAX_SAMPLES
samples; returns false otherwise (caller treats it as a corrupt body).
Counter mirrors cs_write_hr_payload. */
bool cs_read_hr_payload(const uint8_t* src, size_t len, const uint8_t key[32],
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
