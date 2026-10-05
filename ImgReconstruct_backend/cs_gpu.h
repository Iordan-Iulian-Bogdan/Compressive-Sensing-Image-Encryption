#pragma once
// GPU (HIP) solve offload for the reweighted-L1 FISTA tile solvers
// (per-channel and SOMP-structured joint).
//
// cs_gpu.h is plain C++: callers see no HIP headers. The implementation lives
// in cs_gpu.hip (compiled with hipcc, linked into both ImgReconstruct_backend
// and cs_tests). Callers enqueue batched work through fista_solve_blocking,
// which blocks until p.x holds the solved coefficients. When the device is
// unavailable (or fails mid-flight) every problem is completed on the CPU
// via the registered fallback, so results stay correct either way.
//
// Contract is DCT-basis (callers gate CDF97 to the CPU path); the GPU
// reproduces the corresponding CPU core (cs_fista_core_single /
// cs_fista_core_joint) including the IRLS reweight schedule.

namespace cs_gpu {

// FISTA problem: inner = FISTA steps per pass (already mapped by the caller
// via cs_fista_inner_iters). joint = 1 selects the SOMP-structured group-
// L2,1 variant over stacked [x0|x1|x2] (b1/b2/x1/x2 then carry the G/B
// planes; single-channel callers leave them null).
struct FistaProblem {
    const float* b;    // m measurements (channel samples in [0,1])
    const int* rix;    // m measurement row indices (tile-local)
    const int* riy;    // m measurement column indices (tile-local)
    int m;
    int rows, cols;    // tile grid; n = rows * cols
    float* x;          // n host floats: warm start in, solution out
    float lambda;      // L1 weight (per-coefficient threshold scale)
    int inner;         // FISTA steps per pass (already clamped by caller)
    int reweights;     // IRLS passes (already clamped by the caller)
    const float* b1 = nullptr;
    const float* b2 = nullptr;
    float* x1 = nullptr;
    float* x2 = nullptr;
    int joint = 0;
    float tv = 0.0f;  // isotropic TV weight (DCT only); tv = 0 selects the
                      // sampled-gradient fast path, tv > 0 the full-pix TV
                      // branch (same cs_fista_grad math as the CPU)
};

// Runs on the GPU worker thread when the device fails mid-flight; must
// recompute p.x from scratch exactly like the CPU path (DCT basis).
using FistaFallbackFn = void (*)(const FistaProblem& p);
void register_cpu_fallback_fista(FistaFallbackFn fn);

bool enabled();
void set_enabled(bool on);   // probes the device once, logs what it found
bool available();

// Enqueue p (batched with concurrent callers) and block until p.x holds the
// solved coefficients: reweighted-L1 proximal gradient with Nesterov
// momentum -- the CPU cs_fista_core_single / cs_fista_core_joint contract.
void fista_solve_blocking(const FistaProblem& p);

// Two-phase variant: submit enqueues without waiting and returns an opaque
// handle (null when the work already completed inline -- device off or dead,
// CPU fallback ran inside the call). Wait blocks until p.x holds the result
// and consumes the handle. Submitting several independent problems first
// keeps the worker's batch queue full so its pipelined kernel launch stays
// busy instead of draining between dependent-looking calls.
using FistaHandle = void*;
FistaHandle fista_solve_submit(const FistaProblem& p);
void fista_solve_wait(FistaHandle h);

}  // namespace cs_gpu
