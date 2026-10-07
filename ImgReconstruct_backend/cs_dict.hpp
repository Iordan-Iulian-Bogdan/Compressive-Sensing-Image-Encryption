#ifndef IMGRECONSTRUCT_BACKEND_CS_DICT_HPP_
#define IMGRECONSTRUCT_BACKEND_CS_DICT_HPP_

// Coupled patch dictionaries for super-resolution (Yang et al., TIP 2010).
//
// A coupled pair (Dl, Dh) shares sparse codes across resolutions: an LR
// patch's gradient features are coded against Dl with OMP, and the same
// code synthesizes the HR patch through Dh. Training stacks zero-meaned
// HR patches and normalized LR gradient features ([Xh/rh; Yl/rl]) and runs
// the resurrected K-SVD machinery (OMP coding + SVD atom updates, dead-atom
// reinit) on the concatenation, then splits and renormalizes the halves.
// File format CSD2 (replaces the single-dict CSD1, which nothing reads):
//   magic "CSD2" | int version=1 | int atoms, plr, phr, feat |
//   float Dh[atoms*phr*phr] | float Dl[atoms*feat]
// with phr == 2*plr and feat == 4*plr*plr (4 gradient filters).
//
// Conventions: luma-only (Y channel, [0,1] float); HR patches zero-meaned,
// LR mean re-added at synthesis; Dl atoms unit L2 norm, Dh counter-scaled
// so Dh*alpha stays in patch units.

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct cs_coupled_dict {
    int atoms = 0;
    int plr = 8;                 // LR patch width (HR patch is 2x)
    int phr = 16;
    int feat = 256;              // LR gradient-feature dim (4 filters)
    std::vector<float> Dh;       // atoms * phr * phr, atom-major
    std::vector<float> Dl;       // atoms * feat, atom-major, unit norm
};

// Orthogonal matching pursuit for one signal: sparse code over dict atoms.
// D is atoms x dim atom-major (row a at D[a*dim]). At most target_sparsity
// nonzeros (hard-capped at 16: the Cholesky solver below is 16x16).
// Pure function of inputs; deterministic.
void cs_omp_encode_vec(const float* y, const float* D, int dim, int atoms,
    int target_sparsity, float* code_out);

// Gram matrix G = D^T D (atoms x atoms, symmetric) for Batch-OMP.
void cs_gram_matrix(const float* D, int dim, int atoms, float* G);

// Batch-OMP with a precomputed Gram matrix: same greedy selections as
// cs_omp_encode_vec; the caller amortizes G over many signals sharing one
// dictionary (one G per K-SVD iteration / inference tile). D is used only
// for the initial D^T y correlations.
void cs_omp_encode_gram(const float* y, const float* D, const float* G,
    int dim, int atoms, int target_sparsity, float* code_out);

// K-SVD joint training on concatenated [Xh; Yl] pairs (dim = hdim + fdim).
// pairs holds npairs stacked vectors; out.D holds the joint atoms on return
// (caller splits). Prints per-iteration diagnostics to stdout.
bool cs_ksvd_train(const float* pairs, long long npairs, int dim,
    int atoms, int iters, int target_sparsity,
    std::vector<float>& out_D);

// Joint coupled training from HR photos: HR 2*plr patches (stride plr,
// zero-meaned) paired with gradient features of the box-downsampled LR
// image. Caps at max_pairs (uniform stride sampling). Deterministic given
// the same images.
bool cs_train_coupled_dictionary(const std::vector<cv::Mat>& images, int atoms,
    int iters, int max_pairs, int plr, cs_coupled_dict& out);

bool cs_save_coupled_dictionary(const std::string& path, const cs_coupled_dict& d);
bool cs_load_coupled_dictionary(const std::string& path, cs_coupled_dict& d);

// Cached load for the decrypt path: loads once per path, reuses after.
// Returns null (after one warning) when the file cannot be loaded.
const cs_coupled_dict* cs_sr_dict_cached(const std::string& path);

// Dict-based 2x refinement of one solved LR tile (CPU): luma patches are
// gradient-feature coded against Dl with OMP and synthesized through Dh
// with overlap averaging, then globally back-projected against the dense
// solved LR luma. Chroma comes from the AVIR upscale (same contract as the
// FISTA cs-sr path). Falls back to AVIR when dict is null/mismatched or
// the tile is degenerate. dict must outlive the call (use cs_sr_dict_cached).
void cs_sr_upscale_dict_2x(const cv::Mat& lr_tile, const cs_coupled_dict* dict,
    cv::Mat& hr_out);

#endif  // IMGRECONSTRUCT_BACKEND_CS_DICT_HPP_
