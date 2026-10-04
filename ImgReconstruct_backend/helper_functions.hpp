#ifndef IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
#define IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_

#include "lbfgs.hpp"
#include "image_tiles.hpp"
#include "cs_wavelet.hpp"
#include "crypto_utils.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#define MANUAL_PARAM  1
#define AUTO_PARAM  2

struct indices {
    std::vector<int> ri_x_g, ri_y_g;
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

inline void updateAxb2AndComputeFx(float* x_copy, const int* ri_x, const int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n);
inline void eval_g(float* Axb2, float* g, int n);
inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n);
float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
);
int progress(
    void* instance,
    const float* x,
    const float* g,
    const float fx,
    const float xnorm,
    const float gnorm,
    const float step,
    int n,
    int k,
    int ls
);
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols);

void reconstruct_color_channel(const cv::Mat& measurement, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cv::Mat(), float tv = 0.0f);

/** @brief solver selector for the tile reconstruction (see --solver).
OWLQN (0) is the legacy liblbfgs path; FISTA (1) solves each channel
independently with reweighted-L1 FISTA; FISTA_JOINT (2) solves all three
channels together with SOMP-structured joint (group-L2,1) sparsity, i.e.
one common DCT support shared across R/G/B plus reweighting; ADMM (3) is
the consensus-ADMM per-channel solve of the same reweighted-L1 objective
(exact quadratic x-update + exact soft-threshold z-update instead of the
FISTA proximal gradient).
*/
enum CsSolver {
    CS_SOLVER_OWLQN = 0,
    CS_SOLVER_FISTA = 1,
    CS_SOLVER_FISTA_JOINT = 2,
    CS_SOLVER_ADMM = 3
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
    // channel wrapper (fista/joint/admm share these buckets)
    std::atomic<long long> wrap_setup_ns{ 0 }, wrap_core_ns{ 0 }, wrap_rw_ns{ 0 }, wrap_tail_ns{ 0 };
    std::atomic<long long> wrap_calls{ 0 }, rw_passes{ 0 };
    // ADMM core
    std::atomic<long long> admm_setup_ns{ 0 }, admm_rhs_ns{ 0 }, admm_x_ns{ 0 }, admm_cg_ns{ 0 }, admm_z_ns{ 0 }, admm_dual_ns{ 0 };
    std::atomic<long long> admm_iters{ 0 };
    // FISTA core
    std::atomic<long long> fs_grad_ns{ 0 }, fs_shrink_ns{ 0 }, fs_mom_ns{ 0 };
    std::atomic<long long> fs_iters{ 0 };
    // OWL-QN path (reconstruct_color_channel + evaluate)
    std::atomic<long long> owl_setup_ns{ 0 }, owl_lbfgs_ns{ 0 }, owl_tail_ns{ 0 };
    std::atomic<long long> owl_eval_idct_ns{ 0 }, owl_eval_data_ns{ 0 }, owl_eval_dct_ns{ 0 };
    std::atomic<long long> owl_evals{ 0 };
};
extern cs_solve_profile g_solveprof;
void cs_solveprof_reset();
void cs_solveprof_dump(int num_tiles, int num_threads, long long wall_ns);
inline long long cs_solveprof_ns_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

/** @brief single-channel reweighted-L1 FISTA (exact proximal soft-threshold
instead of the OWL-QN pseudo-gradient). `iterations` is in L-BFGS units and
is scaled internally (each outer reweight gets clamp(iterations*4,16,40)
FISTA steps); `reweights` counts outer passes (first unweighted).
Same ref convention as reconstruct_color_channel (DCT/10 warm start in,
pixel plane 0..255 out after the tail scale).
*/
void reconstruct_color_channel_fista(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cv::Mat(),
    float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f);

/** @brief SOMP-structured joint RGB solve: stacked group-L2,1 FISTA with
reweighting. refs[3] hold per-channel DCT/10 warm starts in, solved pixel
planes (CV_32F, 0..255, full tile size) out, ready for cv::merge.
*/
void reconstruct_image_fista_joint(const cv::Mat& pixel_measurements, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat refs[3], float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f);

/** @brief single-channel consensus ADMM: same reweighted-L1 objective, same
budget mapping and same ref convention as reconstruct_color_channel_fista
(DCT/10 warm start in, pixel plane 0..255 out after the tail scale), but a
different optimizer -- consensus ADMM (exact quadratic x-update, exact
soft-threshold z-update, residual-balanced rho) instead of proximal FISTA.
`iterations` is in L-BFGS units and maps through the same helper
(clamp(iterations*4,16,40) ADMM steps per reweight pass; a positive
admm_iters sets the per-pass count directly).
*/
void reconstruct_color_channel_admm(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cv::Mat(),
    float tv = 0.0f, int reweights = 2, int admm_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f);

/** @brief solves ONE chroma plane at half resolution (4:2:0-style subsampling).
The unknown is a (rows+1)/2 x (cols+1)/2 DCT plane; the forward operator
IDCTs it, upsamples it back to the full tile grid and gathers the scattered
full-res measurements there. The gradient path downsamples the full-res
residual (2x2 area average) before the DCT. ref holds the coarse warm-start
(crows x ccols, DCT / 10 domain) and is left holding the solved plane as a
full-res pixel plane (CV_32F, 0..255) so cv::merge can consume it.
*/
void reconstruct_color_channel_subchroma(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref);

/** @brief objective/gradient for the subsampled-chroma solve: forward =
IDCT(coarse) -> upsample -> gather at full-res positions; gradient =
downsample(residual) -> DCT -> 2*s. */
float evaluate_coarse(void* instance, const float* x, eval_data data, float* g, const int n, const float step);

#endif  // IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
