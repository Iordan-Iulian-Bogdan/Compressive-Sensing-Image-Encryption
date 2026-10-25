// GPU compositing of the solved HR tiles (see gpu_composite.h).
#include "gpu_composite.h"
#include "gpu_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

namespace gpu_composite {
namespace {

using namespace gpu_common;

constexpr int kMaxCover = 4;     // tiles covering one column / row of the canvas, at most

struct BlendTile {
    int x, y, w, h;
    long long off;               // pixel offset of the tile (BGR bytes at 3*off, weights at off)
};

// Per output pixel: sum over the covering tiles (rows ascending, then columns ascending, the
// CPU loop order) of v * w with the tile's own weight map, then divide.
__global__ void k_blend(const BlendTile* T, const int* tile_of, int ncol_grid, const int* colj, const int* rowi,
    const float* WT, const std::uint8_t* SRC, int W, int H, std::uint8_t* OUT) {
    // Built with -ffp-contract=off (build_gpu.bat): a fused acc + v*w would round once and no longer
    // match OpenCV's separate multiply and add (verified bit-exact against blendTilesWithImage).
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= W * H) return;
    const int py = p / W, px = p % W;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, ws = 0.f;
    for (int ri = 0; ri < kMaxCover; ++ri) {
        const int i = rowi[py * kMaxCover + ri];
        if (i < 0) break;
        for (int cj = 0; cj < kMaxCover; ++cj) {
            const int j = colj[px * kMaxCover + cj];
            if (j < 0) break;
            const int t = tile_of[i * ncol_grid + j];
            if (t < 0) continue;
            const BlendTile B = T[t];
            const int rr = py - B.y, cc = px - B.x;
            if (rr < 0 || cc < 0 || rr >= B.h || cc >= B.w) continue;
            const long long q = B.off + (long long)rr * B.w + cc;
            const float w = WT[q];
            const std::uint8_t* s = SRC + 3 * q;
            a0 = __fadd_rn(a0, __fmul_rn((float)s[0], w));
            a1 = __fadd_rn(a1, __fmul_rn((float)s[1], w));
            a2 = __fadd_rn(a2, __fmul_rn((float)s[2], w));
            ws = __fadd_rn(ws, w);
        }
    }
    const float sw = fmaxf(ws, 1e-6f);
    auto cvt = [](float v) { int r = __float2int_rn(v); return (std::uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r)); };
    std::uint8_t* o = OUT + 3 * (long long)p;
    o[0] = cvt(__fdiv_rn(a0, sw));
    o[1] = cvt(__fdiv_rn(a1, sw));
    o[2] = cvt(__fdiv_rn(a2, sw));
}

}  // namespace

bool blend_tiles(const std::vector<TileRef>& tiles, int W, int H, std::uint8_t* out_bgr) {
    if (tiles.empty() || W <= 0 || H <= 0) return false;
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev <= 0) return false;
    // Regular grid check: x/w per column j, y/h per row i.
    int ni = 0, nj = 0;
    for (const TileRef& t : tiles) { ni = std::max(ni, t.i + 1); nj = std::max(nj, t.j + 1); }
    std::vector<int> xs(nj, -1), ws(nj, -1), ys(ni, -1), hs(ni, -1), tile_of((size_t)ni * nj, -1);
    for (size_t k = 0; k < tiles.size(); ++k) {
        const TileRef& t = tiles[k];
        if (t.x < 0 || t.y < 0 || t.x + t.w > W || t.y + t.h > H) return false;   // clipping not needed here
        if (xs[t.j] < 0) { xs[t.j] = t.x; ws[t.j] = t.w; } else if (xs[t.j] != t.x || ws[t.j] != t.w) return false;
        if (ys[t.i] < 0) { ys[t.i] = t.y; hs[t.i] = t.h; } else if (ys[t.i] != t.y || hs[t.i] != t.h) return false;
        tile_of[(size_t)t.i * nj + t.j] = (int)k;
    }
    std::vector<int> colj((size_t)W * kMaxCover, -1), rowi((size_t)H * kMaxCover, -1);
    for (int j = 0; j < nj; ++j) {
        if (xs[j] < 0) continue;
        for (int c = xs[j]; c < xs[j] + ws[j]; ++c) {
            int s = 0; while (s < kMaxCover && colj[(size_t)c * kMaxCover + s] >= 0) ++s;
            if (s == kMaxCover) return false;
            colj[(size_t)c * kMaxCover + s] = j;   // j ascending by construction
        }
    }
    for (int i = 0; i < ni; ++i) {
        if (ys[i] < 0) continue;
        for (int r = ys[i]; r < ys[i] + hs[i]; ++r) {
            int s = 0; while (s < kMaxCover && rowi[(size_t)r * kMaxCover + s] >= 0) ++s;
            if (s == kMaxCover) return false;
            rowi[(size_t)r * kMaxCover + s] = i;
        }
    }
    std::vector<BlendTile> bt(tiles.size());
    long long total = 0;
    for (size_t k = 0; k < tiles.size(); ++k) {
        const TileRef& t = tiles[k];
        BlendTile b{};
        b.x = t.x; b.y = t.y; b.w = t.w; b.h = t.h; b.off = total;
        bt[k] = b;
        total += (long long)t.w * t.h;
    }
    std::vector<std::uint8_t> src((size_t)(3 * total));
    std::vector<float> wt((size_t)total);
    for (size_t k = 0; k < tiles.size(); ++k) {
        const size_t n = (size_t)tiles[k].w * tiles[k].h;
        std::copy(tiles[k].bgr, tiles[k].bgr + 3 * n, src.begin() + 3 * bt[k].off);
        std::copy(tiles[k].weights, tiles[k].weights + n, wt.begin() + bt[k].off);
    }

    DevBuf<BlendTile> dT; DevBuf<int> dTO, dC, dR; DevBuf<float> dW; DevBuf<std::uint8_t> dS, dO;
    if (!dT.alloc(bt.size()) || !dTO.alloc(tile_of.size()) || !dC.alloc(colj.size()) || !dR.alloc(rowi.size()) ||
        !dW.alloc(wt.size()) || !dS.alloc(src.size()) || !dO.alloc((size_t)3 * W * H)) {
        std::fprintf(stderr, "gpu_composite: device allocation failed\n");
        return false;
    }
    HIP_TRY(hipMemcpy(dT.p, bt.data(), bt.size() * sizeof(BlendTile), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dTO.p, tile_of.data(), tile_of.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dC.p, colj.data(), colj.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dR.p, rowi.data(), rowi.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dW.p, wt.data(), wt.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dS.p, src.data(), src.size(), hipMemcpyHostToDevice));
    k_blend<<<dim3((unsigned)(((long long)W * H + kThreads - 1) / kThreads)), dim3(kThreads)>>>(
        dT.p, dTO.p, nj, dC.p, dR.p, dW.p, dS.p, W, H, dO.p);
    HIP_TRY(hipGetLastError());
    HIP_TRY(hipDeviceSynchronize());
    HIP_TRY(hipMemcpy(out_bgr, dO.p, (size_t)3 * W * H, hipMemcpyDeviceToHost));
    return true;
}

}  // namespace gpu_composite
