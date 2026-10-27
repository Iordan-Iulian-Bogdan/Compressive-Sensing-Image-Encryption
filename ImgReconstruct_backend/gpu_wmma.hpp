#pragma once
// RDNA3 WMMA helpers shared by the HIP kernels (include from .hip files).
//
// v_wmma_f32_16x16x16_f16 (wave32), D = A * B + C on 16 x 16 tiles:
//   A fragment: lane l holds row l % 16, its 16 k values (lanes 16..31 repeat lanes 0..15)
//   B fragment: lane l holds column l % 16, its 16 k values
//   C / D: lane l holds column l % 16, rows 2 i + l / 16 in element i (i = 0..7)
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>

namespace gpu_wmma {

typedef _Float16 half_t;
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

__device__ inline f8 mma(const h16& a, const h16& b, const f8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// 16 consecutive halves (16-byte aligned).
__device__ inline h16 ld16(const half_t* p) {
    const h8 lo = *(const h8*)p, hi = *(const h8*)(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ inline f8 splat(float v) { f8 c; for (int i = 0; i < 8; ++i) c[i] = v; return c; }
// Value of lane l ^ M (M < 16, within the lane's row of 16; DPP row_xmask).
template <int M>
__device__ inline float dpp_xor(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | M, 0xf, 0xf, false));
}
// WMMA needs every lane active, but the compiler may sink a product whose result is only used
// under a lane-divergent branch into that branch. Pinning the result keeps the product before it.
__device__ inline void pin(f8& c) { asm volatile("" : "+v"(c)); }

// fp32 -> fp16 bits, round to nearest even (host side, for packing weights and tables).
inline std::uint16_t to_half(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t s = (x >> 16) & 0x8000u;
    const int e = (int)((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t m = x & 0x7fffffu;
    if (e >= 31) return (std::uint16_t)(s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (std::uint16_t)s;
        m |= 0x800000u;
        const int sh = 14 - e;
        std::uint32_t r = m >> sh;
        const std::uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (std::uint16_t)(s | r);
    }
    std::uint32_t r = ((std::uint32_t)e << 10) | (m >> 13);
    const std::uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (std::uint16_t)(s | r);
}
// fp16 bits -> fp32 (host side; finite values).
inline float from_half(std::uint16_t h) {
    const std::uint32_t s = (std::uint32_t)(h & 0x8000u) << 16;
    int e = (h >> 10) & 0x1f;
    std::uint32_t m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = s;
        else {   // subnormal: normalise
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            x = s | ((std::uint32_t)(e - 15 + 127) << 23) | ((m & 0x3ffu) << 13);
        }
    } else {
        x = s | ((std::uint32_t)(e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

}  // namespace gpu_wmma
