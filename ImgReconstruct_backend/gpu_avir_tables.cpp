// AVIR 2x matrices for the GPU tile pipeline (see gpu_avir_tables.hpp).
#include "gpu_avir_tables.hpp"
#include "avir.h"

#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace {

typedef avir::fpclass_def<float, float, avir::CImageResizerDithererErrdINL<float>> fpclass_dith;

struct Tables { std::vector<float> Ar, Ac; };

// AVIR float in/out (no dither) of a 3-channel interleaved image with X1 in channel 0.
std::vector<float> avir_ch0(const std::vector<float>& X1, int R, int C) {
    std::vector<float> in((size_t)R * C * 3, 0.f), out((size_t)4 * R * C * 3);
    for (size_t k = 0; k < X1.size(); ++k) in[k * 3] = X1[k];
    avir::CImageResizer<fpclass_dith> rs(8);   // same resizer as cs_upscale_2x_avir
    rs.resizeImage(in.data(), C, R, 0, out.data(), 2 * C, 2 * R, 3, 0);
    std::vector<float> o((size_t)4 * R * C);
    for (size_t k = 0; k < o.size(); ++k) o[k] = out[k * 3];
    return o;
}

// X = 1 e_j^T gives (A_r 1) A_c[:,j]^T and X = e_i 1^T gives A_r[:,i] (A_c 1)^T. The measured
// columns carry a = (A_c 1)[C] and b = (A_r 1)[R], and a*b = AVIR(ones)[R][C] is divided out
// of A_c so that A_r X A_c^T reproduces AVIR.
void extract(int R, int C, Tables& T) {
    T.Ar.assign((size_t)2 * R * R, 0.f);
    T.Ac.assign((size_t)2 * C * C, 0.f);
    const std::vector<float> s = avir_ch0(std::vector<float>((size_t)R * C, 1.f), R, C);
    const double ab = s[(size_t)R * 2 * C + C];
    for (int j = 0; j < C; ++j) {
        std::vector<float> X((size_t)R * C, 0.f);
        for (int i = 0; i < R; ++i) X[(size_t)i * C + j] = 1.f;
        const std::vector<float> o = avir_ch0(X, R, C);
        for (int q = 0; q < 2 * C; ++q) T.Ac[(size_t)q * C + j] = (float)(o[(size_t)R * 2 * C + q] / ab);
    }
    for (int i = 0; i < R; ++i) {
        std::vector<float> X((size_t)R * C, 0.f);
        for (int j = 0; j < C; ++j) X[(size_t)i * C + j] = 1.f;
        const std::vector<float> o = avir_ch0(X, R, C);
        for (int p = 0; p < 2 * R; ++p) T.Ar[(size_t)p * R + i] = o[(size_t)p * 2 * C + C];
    }
}

std::mutex g_mu;
std::map<std::pair<int, int>, std::unique_ptr<Tables>> g_cache;

}  // namespace

bool cs_avir_2x_tables(int R, int C, const std::vector<float>*& Ar, const std::vector<float>*& Ac) {
    if (R < 1 || C < 1) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    auto& slot = g_cache[{ R, C }];
    if (!slot) {
        slot = std::make_unique<Tables>();
        extract(R, C, *slot);
    }
    Ar = &slot->Ar;
    Ac = &slot->Ac;
    return true;
}
