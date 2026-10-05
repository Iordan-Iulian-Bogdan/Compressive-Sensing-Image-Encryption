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
    float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f);

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
    cs_gpu::FistaHandle h = nullptr;  // null => already done (CPU/fallback)
};
void fista_channel_begin(fista_channel_task& t, const cv::Mat& pixel_measurements,
    const int& channel, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations,
    cv::Mat& ref, float tv = 0.0f, int reweights = 2, int fista_iters = 0,
    int basis = CS_BASIS_DCT, float wscale = 2.0f);
void fista_channel_end(fista_channel_task& t, cv::Mat& ref);

/** @brief SOMP-structured joint RGB solve: stacked group-L2,1 FISTA with
reweighting. refs[3] hold per-channel DCT/10 warm starts in, solved pixel
planes (CV_32F, 0..255, full tile size) out, ready for cv::merge.
*/
void reconstruct_image_fista_joint(const cv::Mat& pixel_measurements, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat refs[3], float tv = 0.0f, int reweights = 2, int fista_iters = 0, int basis = CS_BASIS_DCT, float wscale = 2.0f);

#endif  // IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
