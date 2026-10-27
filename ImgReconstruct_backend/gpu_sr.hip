// Batched GPU cs-sr tile stage (see gpu_sr.h).
#include "gpu_sr.h"
#include "gpu_avir_tables.hpp"
#include "gpu_common.hpp"
#include "gpu_swin.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

namespace gpu_sr {
namespace {

using namespace gpu_common;

// Per-tile record: LR geometry and offsets into the LR byte buffer, the HR planes and the
// AVIR table buffer.
struct SrDev {
    int R, C;      // LR rows / cols (HR is 2R x 2C)
    int lr_off;    // LR pixel offset (bytes = 3 * lr_off in the BGR buffer)
    int hr_off;    // HR pixel offset (same as the gpu_admm TileLayout offset)
    int ar, ac;    // offsets of A_r (2R x R) and A_c (2C x C) in the table buffer
    int pad[2];
};

__device__ inline int sat8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
// OpenCV 8-bit BGR -> Y (yuv_shift 14, CV_DESCALE)
__device__ inline int ycc_y(int b, int g, int r) { return (b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14; }

// NLM strength per tile (one block per tile), as cs_sr_prox_nlm_h(cs_estimate_noise_sigma):
// Immerkaer Laplacian of the 8-bit luma, mean |.| per 32x32 block of the interior (blocks
// narrower than 8 skipped), times sqrt(pi/2)/6, upper median; h = clamp(1.6*sigma, 3, 30).
__global__ void k_nlm_h(const SrDev* sd, const std::uint8_t* LR, float* H) {
    const SrDev T = sd[blockIdx.x];
    const std::uint8_t* p = LR + 3 * (size_t)T.lr_off;
    auto Y = [&](int i, int j) { const std::uint8_t* q = p + 3 * ((size_t)i * T.C + j); return ycc_y(q[0], q[1], q[2]); };
    __shared__ double blocks[64];
    int nb = 0;
    const int BS = 32;
    if (T.R >= 4 && T.C >= 4) {
        for (int y0 = 1; y0 + 1 < T.R; y0 += BS)
            for (int x0 = 1; x0 + 1 < T.C; x0 += BS) {
                const int w = min(BS, T.C - 1 - x0), h = min(BS, T.R - 1 - y0);
                if (w < 8 || h < 8) continue;
                double s = 0.0;
                for (int e = threadIdx.x; e < w * h; e += kThreads) {
                    const int i = y0 + e / w, j = x0 + e % w;
                    const int lap = Y(i - 1, j - 1) - 2 * Y(i - 1, j) + Y(i - 1, j + 1)
                        - 2 * Y(i, j - 1) + 4 * Y(i, j) - 2 * Y(i, j + 1)
                        + Y(i + 1, j - 1) - 2 * Y(i + 1, j) + Y(i + 1, j + 1);
                    s += (double)(lap < 0 ? -lap : lap);
                }
                s = block_sum(s);
                if (threadIdx.x == 0 && nb < 64) blocks[nb] = 1.2533141373155001 / 6.0 * (s / (double)(w * h));
                ++nb;
            }
    }
    if (threadIdx.x != 0) return;
    nb = min(nb, 64);
    double sigma = 0.0;
    if (nb > 0) {
        for (int a = 1; a < nb; ++a) {             // insertion sort (a handful of blocks)
            const double v = blocks[a]; int k = a - 1;
            while (k >= 0 && blocks[k] > v) { blocks[k + 1] = blocks[k]; --k; }
            blocks[k + 1] = v;
        }
        sigma = blocks[nb / 2];                    // nth_element at size/2
    }
    if (!(sigma > 0.0)) sigma = 3.0;
    float h = (float)(1.6 * sigma);
    H[blockIdx.x] = h < 3.0f ? 3.0f : (h > 30.0f ? 30.0f : h);
}

// AVIR pass 1: T[c][i][q] = sum_j X[i][j][c] * A_c[q][j]   (R x 2C per channel)
__global__ void k_avir_h(const SrDev* sd, const float* tab, const std::uint8_t* LR, float* TMP) {
    const SrDev T = sd[blockIdx.y / 3];
    const int c = blockIdx.y % 3;
    const int e = blockIdx.x * kThreads + threadIdx.x;
    const int W2 = 2 * T.C;
    if (e >= T.R * W2) return;
    const int i = e / W2, q = e % W2;
    const std::uint8_t* x = LR + 3 * ((size_t)T.lr_off + (size_t)i * T.C) + c;
    const float* a = tab + T.ac + (size_t)q * T.C;
    float s = 0.0f;
    for (int j = 0; j < T.C; ++j) s += (float)x[3 * j] * a[j];
    TMP[(size_t)6 * T.lr_off + (size_t)c * T.R * W2 + e] = s;
}

// AVIR pass 2: F[c][p][q] = sum_i A_r[p][i] * T[c][i][q]   (2R x 2C per channel, planar)
__global__ void k_avir_v(const SrDev* sd, const float* tab, const float* TMP, float* F) {
    const SrDev T = sd[blockIdx.y / 3];
    const int c = blockIdx.y % 3;
    const int e = blockIdx.x * kThreads + threadIdx.x;
    const int W2 = 2 * T.C, n = 4 * T.R * T.C;
    if (e >= n) return;
    const int pr = e / W2, q = e % W2;
    const float* t = TMP + (size_t)6 * T.lr_off + (size_t)c * T.R * W2 + q;
    const float* a = tab + T.ar + (size_t)pr * T.R;
    float s = 0.0f;
    for (int i = 0; i < T.R; ++i) s += a[i] * t[(size_t)i * W2];
    F[(size_t)3 * T.hr_off + (size_t)c * n + e] = s;
}

__device__ inline float avir_round(float d) { return d < 0.f ? -(float)(int)(0.5f - d) : (float)(int)(d + 0.5f); }

// AVIR CImageResizerDithererErrdINL (TrMul 1, PkOut 255) per tile and channel, in scan order:
// one thread walks the 2R x 2C plane; D (2C + 2 floats per thread) is the next-row error row.
__global__ void k_dither(const SrDev* sd, int nt, float* F, float* DBUF, int dstride, std::uint8_t* HR8) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= 3 * nt) return;
    const SrDev T = sd[g / 3];
    const int c = g % 3, H2 = 2 * T.R, W2 = 2 * T.C, n = H2 * W2;
    float* line = F + (size_t)3 * T.hr_off + (size_t)c * n;
    float* d = DBUF + (size_t)g * dstride + 1;   // d[-1] is the discarded slot
    for (int j = -1; j <= W2; ++j) d[j] = 0.f;
    std::uint8_t* out = HR8 + (size_t)3 * T.hr_off + c;
    for (int i = 0; i < H2; ++i) {
        float* L = line + (size_t)i * W2;
        for (int j = 0; j < W2; ++j) { L[j] += d[j]; d[j] = 0.f; }
        for (int j = 0; j < W2; ++j) {
            const float z0 = avir_round(L[j]);
            const float noise = L[j] - z0;
            const float v = fminf(fmaxf(z0, 0.f), 255.f);
            out[3 * ((size_t)i * W2 + j)] = (std::uint8_t)(int)v;
            if (j < W2 - 1) {
                const float nm1 = noise * 0.364842f;
                L[j + 1] += nm1;
                d[j - 1] += noise * 0.207305f;
                d[j] += nm1;
                d[j + 1] += noise * 0.063011f;
            } else {
                d[j - 1] += noise * 0.207305f;
                d[j] += noise * 0.364842f;
            }
        }
    }
}

// k_dither on an anti-diagonal wavefront: one block per tile and channel, one thread per row; row
// i handles column s - 2i on step s, so a pixel's carries from the row above (columns j-1..j+1)
// are complete before it runs. Each row has its own carry row in LDS (slot 0 = column -1), and a
// pixel adds (F + carry from above) + carry from the left, the order of the serial scan, so the
// output is bit-identical to k_dither.
__global__ void k_dither_wave(const SrDev* sd, float* F, std::uint8_t* HR8) {
    extern __shared__ float dsh[];
    const SrDev T = sd[blockIdx.x / 3];
    const int c = blockIdx.x % 3, H2 = 2 * T.R, W2 = 2 * T.C, n = H2 * W2, ds = W2 + 2;
    const int i = threadIdx.x;
    for (int e = threadIdx.x; e < H2 * ds; e += blockDim.x) dsh[e] = 0.f;
    __syncthreads();
    const float* line = F + (size_t)3 * T.hr_off + (size_t)c * n + (size_t)i * W2;
    std::uint8_t* out = HR8 + (size_t)3 * T.hr_off + c + (size_t)3 * i * W2;
    const float* d = dsh + (size_t)i * ds + 1;
    float* dn = (i + 1 < H2) ? dsh + (size_t)(i + 1) * ds + 1 : nullptr;   // the last row's carries are dropped
    float left = 0.f;   // the left pixel's noise
    const int steps = W2 + 2 * (H2 - 1);
    for (int s = 0; s < steps; ++s) {
        const int j = s - 2 * i;
        if (i < H2 && j >= 0 && j < W2) {
            // fused like k_dither, where the carry's multiply and add sit in one iteration
            float L = line[j] + d[j];
            if (j > 0) L = __fmaf_rn(left, 0.364842f, L);
            const float z0 = avir_round(L);
            const float noise = L - z0;
            const float v = fminf(fmaxf(z0, 0.f), 255.f);
            out[3 * j] = (std::uint8_t)(int)v;
            left = noise;
            if (dn) {
                dn[j - 1] = __fmaf_rn(noise, 0.207305f, dn[j - 1]);
                dn[j] = __fmaf_rn(noise, 0.364842f, dn[j]);
                if (j < W2 - 1) dn[j + 1] = __fmaf_rn(noise, 0.063011f, dn[j + 1]);
            }
        }
        __syncthreads();
    }
}

// BGR -> YCrCb (OpenCV 8-bit fixed point); Z = Y * (1/255) is the ADMM start (as
// init_ycc_chs[0].convertTo(CV_32F, 1/255)); Cr/Cb kept for the output merge.
__global__ void k_bgr2ycc(const SrDev* sd, const std::uint8_t* HR8, float* Z, std::uint8_t* CR, std::uint8_t* CB) {
    const SrDev T = sd[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= 4 * T.R * T.C) return;
    const std::uint8_t* q = HR8 + 3 * ((size_t)T.hr_off + i);
    const int b = q[0], g = q[1], r = q[2];
    const int y = ycc_y(b, g, r);
    const int delta = 128 << 14;
    CR[T.hr_off + i] = (std::uint8_t)sat8(((r - y) * 11682 + delta + (1 << 13)) >> 14);
    CB[T.hr_off + i] = (std::uint8_t)sat8(((b - y) * 9241 + delta + (1 << 13)) >> 14);
    Z[T.hr_off + i] = (float)sat8(y) * (1.0f / 255.0f);
}

// Final tile: y8 = saturate(round-half-even(z * 255)) (convertTo CV_8U), merged with the AVIR
// Cr/Cb and converted YCrCb -> BGR (OpenCV 8-bit fixed point).
__global__ void k_finish(const SrDev* sd, const float* Z, const std::uint8_t* CR, const std::uint8_t* CB, std::uint8_t* OUT) {
    const SrDev T = sd[blockIdx.y];
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= 4 * T.R * T.C) return;
    const int o = T.hr_off + i;
    const int y = sat8(__float2int_rn(Z[o] * 255.0f));
    const int cr = (int)CR[o] - 128, cb = (int)CB[o] - 128;
    std::uint8_t* q = OUT + 3 * (size_t)o;
    q[0] = (std::uint8_t)sat8(y + ((cb * 29049 + (1 << 13)) >> 14));
    q[1] = (std::uint8_t)sat8(y + ((cb * -5636 + cr * -11698 + (1 << 13)) >> 14));
    q[2] = (std::uint8_t)sat8(y + ((cr * 22987 + (1 << 13)) >> 14));
}

}  // namespace

bool run(std::vector<Tile>& tiles, const gpu_admm::Params& prm, bool swin_prior) {
    if (tiles.empty()) return true;
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev <= 0) return false;
    const int nt = (int)tiles.size();

    // Layout: LR bytes, HR planes (same order/offsets as gpu_admm's TileLayout), AVIR tables
    // (none with the Swin2SR prior, which needs no AVIR buffers either).
    std::vector<SrDev> sd((size_t)nt);
    std::vector<float> table;
    std::map<std::pair<int, int>, std::pair<int, int>> tab_off;   // (R, C) -> (ar, ac)
    long long lr_total = 0, hr_total = 0;
    int max_hr = 0, max_tmp = 0, max_c = 0, max_r = 0;
    for (int t = 0; t < nt; ++t) {
        const Tile& P = tiles[(size_t)t];
        auto key = std::make_pair(P.lr_rows, P.lr_cols);
        auto it = tab_off.find(key);
        if (it == tab_off.end() && !swin_prior) {
            const std::vector<float>* Ar = nullptr; const std::vector<float>* Ac = nullptr;
            if (!cs_avir_2x_tables(P.lr_rows, P.lr_cols, Ar, Ac)) return false;
            const int ar = (int)table.size(); table.insert(table.end(), Ar->begin(), Ar->end());
            const int ac = (int)table.size(); table.insert(table.end(), Ac->begin(), Ac->end());
            it = tab_off.emplace(key, std::make_pair(ar, ac)).first;
        }
        SrDev d{};
        d.R = P.lr_rows; d.C = P.lr_cols;
        d.lr_off = (int)lr_total; d.hr_off = (int)hr_total;
        if (!swin_prior) { d.ar = it->second.first; d.ac = it->second.second; }
        sd[(size_t)t] = d;
        lr_total += (long long)P.lr_rows * P.lr_cols;
        hr_total += 4LL * P.lr_rows * P.lr_cols;
        max_hr = std::max(max_hr, 4 * P.lr_rows * P.lr_cols);
        max_tmp = std::max(max_tmp, 2 * P.lr_rows * P.lr_cols);
        max_c = std::max(max_c, P.lr_cols);
        max_r = std::max(max_r, P.lr_rows);
    }
    if (3 * hr_total > 2000000000LL) return false;
    std::vector<std::uint8_t> hlr((size_t)3 * lr_total);
    parallel_for(nt, [&](int t) {
        std::copy(tiles[(size_t)t].lr_bgr, tiles[(size_t)t].lr_bgr + (size_t)3 * tiles[(size_t)t].lr_rows * tiles[(size_t)t].lr_cols,
            hlr.begin() + (size_t)3 * sd[(size_t)t].lr_off);
    });
    const int dstride = 2 * max_c + 2;

    DevBuf<SrDev> dsd; DevBuf<float> dtab, dTMP, dF, dD, dZ, dH; DevBuf<std::uint8_t> dLR, dHR8, dCR, dCB, dOUT;
    if (!dsd.alloc((size_t)nt) || !dtab.alloc(table.size()) || !dTMP.alloc(swin_prior ? 0 : (size_t)6 * lr_total) ||
        !dF.alloc(swin_prior ? 0 : (size_t)3 * hr_total) || !dD.alloc(swin_prior ? 0 : (size_t)3 * nt * dstride) || !dZ.alloc((size_t)hr_total) ||
        !dH.alloc((size_t)nt) || !dLR.alloc((size_t)3 * lr_total) || !dHR8.alloc((size_t)3 * hr_total) ||
        !dCR.alloc((size_t)hr_total) || !dCB.alloc((size_t)hr_total) || !dOUT.alloc((size_t)3 * hr_total)) {
        std::fprintf(stderr, "gpu_sr: device allocation failed\n");
        return false;
    }
    HIP_TRY(hipMemcpy(dsd.p, sd.data(), (size_t)nt * sizeof(SrDev), hipMemcpyHostToDevice));
    if (!table.empty()) HIP_TRY(hipMemcpy(dtab.p, table.data(), table.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_TRY(hipMemcpy(dLR.p, hlr.data(), hlr.size(), hipMemcpyHostToDevice));

    const dim3 blk(kThreads);
    // NLM strength per tile (needed on the host for the ADMM problems)
    k_nlm_h<<<dim3((unsigned)nt), blk>>>(dsd.p, dLR.p, dH.p);
    if (swin_prior) {
        // Swin2SR 2x of every tile, straight into the 8-bit BGR warm start
        std::vector<gpu_swin::Image> ims((size_t)nt);
        for (int t = 0; t < nt; ++t) ims[(size_t)t] = { sd[(size_t)t].R, sd[(size_t)t].C, sd[(size_t)t].lr_off, sd[(size_t)t].hr_off };
        if (!gpu_swin::run(ims, dLR.p, dHR8.p)) return false;
    } else {
        // AVIR 2x: two separable passes, then the scan-order dither into 8-bit BGR
        k_avir_h<<<dim3((unsigned)((max_tmp + kThreads - 1) / kThreads), (unsigned)(3 * nt)), blk>>>(dsd.p, dtab.p, dLR.p, dTMP.p);
        k_avir_v<<<dim3((unsigned)((max_hr + kThreads - 1) / kThreads), (unsigned)(3 * nt)), blk>>>(dsd.p, dtab.p, dTMP.p, dF.p);
        const int wave_threads = (2 * max_r + 31) / 32 * 32;
        const size_t wave_lds = (size_t)2 * max_r * (2 * max_c + 2) * sizeof(float);
        if (wave_threads <= 1024 && wave_lds <= 65536)
            k_dither_wave<<<dim3((unsigned)(3 * nt)), dim3((unsigned)wave_threads), wave_lds>>>(dsd.p, dF.p, dHR8.p);
        else
            k_dither<<<dim3((unsigned)((3 * nt + 63) / 64)), dim3(64)>>>(dsd.p, nt, dF.p, dD.p, dstride, dHR8.p);
    }
    // YCrCb; the luma becomes the ADMM start
    const dim3 hgrid((unsigned)((max_hr + kThreads - 1) / kThreads), (unsigned)nt);
    k_bgr2ycc<<<hgrid, blk>>>(dsd.p, dHR8.p, dZ.p, dCR.p, dCB.p);
    HIP_TRY(hipGetLastError());
    std::vector<float> hh((size_t)nt);
    HIP_TRY(hipMemcpy(hh.data(), dH.p, (size_t)nt * sizeof(float), hipMemcpyDeviceToHost));

    // optional verification outputs
    bool dbg = false;
    for (const Tile& P : tiles) dbg = dbg || P.dbg_avir || P.dbg_nlm_h;
    if (dbg) {
        std::vector<std::uint8_t> hav((size_t)3 * hr_total);
        HIP_TRY(hipMemcpy(hav.data(), dHR8.p, hav.size(), hipMemcpyDeviceToHost));
        for (int t = 0; t < nt; ++t) {
            Tile& P = tiles[(size_t)t];
            if (P.dbg_avir) std::copy(hav.begin() + (size_t)3 * sd[(size_t)t].hr_off,
                hav.begin() + (size_t)3 * (sd[(size_t)t].hr_off + 4 * P.lr_rows * P.lr_cols), P.dbg_avir);
            if (P.dbg_nlm_h) *P.dbg_nlm_h = hh[(size_t)t];
        }
    }

    // luma PnP-ADMM with z resident on the device
    std::vector<gpu_admm::Problem> probs((size_t)nt);
    for (int t = 0; t < nt; ++t) {
        const Tile& P = tiles[(size_t)t];
        gpu_admm::Problem q{ 2 * P.lr_rows, 2 * P.lr_cols, P.b, P.rix, P.riy, P.m, nullptr, P.lambda, hh[(size_t)t] };
        q.hr_val = P.hr_val;
        q.tv = P.tv;
        probs[(size_t)t] = q;
    }
    if (!gpu_admm::solve_pnp_admm_device(probs, prm, dZ.p)) return false;

    // 8-bit luma + AVIR chroma -> BGR, back to the host tiles
    k_finish<<<hgrid, blk>>>(dsd.p, dZ.p, dCR.p, dCB.p, dOUT.p);
    HIP_TRY(hipGetLastError());
    // Output tiles laid out back to back in one buffer (the batched caller's pool) take one copy.
    bool packed = true;
    for (int t = 0; t < nt && packed; ++t) packed = tiles[(size_t)t].hr_bgr == tiles[0].hr_bgr + (size_t)3 * sd[(size_t)t].hr_off;
    if (packed) {
        HIP_TRY(hipMemcpy(tiles[0].hr_bgr, dOUT.p, (size_t)3 * hr_total, hipMemcpyDeviceToHost));
        return true;
    }
    std::vector<std::uint8_t> hout((size_t)3 * hr_total);
    HIP_TRY(hipMemcpy(hout.data(), dOUT.p, hout.size(), hipMemcpyDeviceToHost));
    parallel_for(nt, [&](int t) {
        Tile& P = tiles[(size_t)t];
        std::copy(hout.begin() + (size_t)3 * sd[(size_t)t].hr_off,
            hout.begin() + (size_t)3 * (sd[(size_t)t].hr_off + 4 * P.lr_rows * P.lr_cols), P.hr_bgr);
    });
    return true;
}

}  // namespace gpu_sr
