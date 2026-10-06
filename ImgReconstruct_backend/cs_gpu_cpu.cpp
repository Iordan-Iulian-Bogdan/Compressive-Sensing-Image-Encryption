// CPU-only implementation of the cs_gpu interface (no HIP/ROCm).
//
// Used by build_cpu.bat instead of cs_gpu.hip: every solve runs the
// registered CPU fallback inline (registered by helper_functions.cpp via
// register_cpu_fallback_fista, exactly as in the HIP build), enabled()
// stays false so callers take the CPU path, and set_enabled(true) degrades
// with the same "staying on CPU" warning the HIP build prints when no
// device is present. Results are identical to --device cpu on the HIP
// build; only the offload path is missing.
#include <cstdio>

#include "cs_gpu.h"

namespace cs_gpu {
namespace {
FistaFallbackFn g_fallback = nullptr;
}  // namespace

void register_cpu_fallback_fista(FistaFallbackFn fn) { g_fallback = fn; }

bool enabled() { return false; }

void set_enabled(bool on) {
    if (on) {
        std::fprintf(stderr, "[gpu] CPU-only build: no HIP device, staying on CPU\n");
    }
}

bool available() { return false; }

void fista_solve_blocking(const FistaProblem& p) {
    if (g_fallback) g_fallback(p);
}

FistaHandle fista_solve_submit(const FistaProblem& p) {
    if (g_fallback) g_fallback(p);
    return nullptr;  // null => completed inline (see cs_gpu.h contract)
}

void fista_solve_wait(FistaHandle /*h*/) {}

}  // namespace cs_gpu
