// Coupled patch dictionaries for super-resolution (Yang et al., TIP 2010).
//
// K-SVD/OMP core resurrected from the removed --dict path (commit 6675cee),
// generalized to arbitrary signal dims; the coupled training and the SR
// inference below are new. Luma-only, [0,1] float throughout.

#include "cs_dict.hpp"
#include "image_tiles.hpp" // cs_upscale_2x_avir (warm start + chroma source)

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>

void cs_gram_matrix(const float* D, int dim, int atoms, float* G) {
    for (int a = 0; a < atoms; ++a) {
        const float* da = &D[(size_t)a * dim];
        for (int b = 0; b <= a; ++b) {
            const float* db = &D[(size_t)b * dim];
            double dot = 0.0;
            for (int p = 0; p < dim; ++p) dot += (double)da[p] * db[p];
            G[(size_t)a * atoms + b] = G[(size_t)b * atoms + a] = (float)dot;
        }
    }
}

void cs_omp_encode_vec(const float* y, const float* D, int dim, int atoms,
    int target_sparsity, float* code_out)
{
    if (dim <= 0 || atoms <= 0) return;
    if (target_sparsity > 16) target_sparsity = 16; // Cholesky solver is 16x16
    if (target_sparsity < 1) target_sparsity = 1;
    std::memset(code_out, 0, sizeof(float) * (size_t)atoms);

    // Batch-OMP (Rubinstein et al.): callers working in bulk should hoist
    // the Gram matrix via cs_gram_matrix + cs_omp_encode_gram (one G per
    // dictionary, not per signal). This wrapper keeps the simple signature
    // for one-off calls (tests, single patches).
    std::vector<float> G((size_t)atoms * atoms);
    cs_gram_matrix(D, dim, atoms, G.data());
    cs_omp_encode_gram(y, D, G.data(), dim, atoms, target_sparsity, code_out);
}

// Batch-OMP with a precomputed Gram matrix G = D^T D (atoms x atoms).
// Same greedy selections as cs_omp_encode_vec; caller amortizes G over
// many signals sharing one dictionary (D is used only for the initial
// D^T y correlations).
void cs_omp_encode_gram(const float* y, const float* D, const float* G,
    int dim, int atoms, int target_sparsity, float* code_out)
{
    if (dim <= 0 || atoms <= 0 || !G || !D) return;
    if (target_sparsity > 16) target_sparsity = 16;
    if (target_sparsity < 1) target_sparsity = 1;
    std::memset(code_out, 0, sizeof(float) * (size_t)atoms);
    std::vector<float> alpha((size_t)atoms, 0.0f);
    for (int a = 0; a < atoms; ++a) {
        const float* da = &D[(size_t)a * dim];
        double dot = 0.0;
        for (int p = 0; p < dim; ++p) dot += (double)da[p] * y[p];
        alpha[(size_t)a] = (float)dot;
    }

    std::vector<char> used((size_t)atoms, 0);
    int support[16];
    int n = 0;
    float L[16][16] = {};
    float c[16] = {};
    for (int s = 0; s < target_sparsity && n < atoms && n < 16; ++s) {
        int best = -1;
        float best_corr = 0.0f;
        for (int a = 0; a < atoms; ++a) {
            if (used[(size_t)a]) continue;
            const float mag = std::fabs(alpha[(size_t)a]);
            if (best < 0 || mag > best_corr) { best_corr = mag; best = a; }
        }
        if (best < 0) break;
        // Cholesky row for the enlarged support: solve L w = G[best][S]
        float wrow[16] = {};
        bool bad = false;
        for (int i = 0; i < n; ++i) {
            double acc = G[(size_t)best * atoms + support[i]];
            for (int m = 0; m < i; ++m) acc -= (double)L[i][m] * wrow[m];
            if (!(std::fabs(L[i][i]) > 1e-12)) { bad = true; break; }
            wrow[i] = (float)(acc / L[i][i]);
        }
        if (bad) break;
        double diag = G[(size_t)best * atoms + best];
        for (int m = 0; m < n; ++m) diag -= (double)wrow[m] * wrow[m];
        if (!(diag > 1e-12)) break; // linearly dependent: support is complete
        for (int i = 0; i < n; ++i) L[n][i] = wrow[i];
        L[n][n] = (float)std::sqrt(diag);
        used[(size_t)best] = 1;
        support[n++] = best;
        // forward/back substitution for the coefficients on the support
        float y1[16] = {};
        for (int i = 0; i < n; ++i) {
            double acc = alpha[(size_t)support[i]];
            for (int m = 0; m < i; ++m) acc -= (double)L[i][m] * y1[m];
            y1[i] = (float)(acc / L[i][i]);
        }
        for (int i = n - 1; i >= 0; --i) {
            double acc = y1[i];
            for (int m = i + 1; m < n; ++m) acc -= (double)L[m][i] * c[m];
            c[i] = (float)(acc / L[i][i]);
        }
        // alpha <- D^T(y - D_S c_S) = alpha - G[:,S] c
        for (int a = 0; a < atoms; ++a) {
            if (used[(size_t)a]) continue;
            double acc = 0.0;
            for (int r = 0; r < n; ++r) acc += (double)G[(size_t)a * atoms + support[r]] * c[r];
            alpha[(size_t)a] -= (float)acc;
        }
    }
    for (int r = 0; r < n; ++r) code_out[support[r]] = c[r];
}

bool cs_ksvd_train(const float* pairs, long long npairs, int dim,
    int atoms, int iters, int target_sparsity,
    std::vector<float>& out_D)
{
    if (!pairs || npairs <= 0 || dim <= 0 || atoms <= 0 || atoms > 4096) return false;
    if (npairs < atoms) return false;
    if (iters < 1) iters = 1;

    // deterministic init: evenly spaced training vectors, unit norm
    out_D.assign((size_t)atoms * dim, 0.0f);
    for (int a = 0; a < atoms; ++a) {
        float* atom = &out_D[(size_t)a * dim];
        const long long src = ((long long)a * npairs) / atoms;
        std::memcpy(atom, &pairs[(size_t)src * dim], sizeof(float) * (size_t)dim);
        double norm = 0.0;
        for (int p = 0; p < dim; ++p) norm += (double)atom[p] * atom[p];
        norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
        for (int p = 0; p < dim; ++p) atom[p] = (float)(atom[p] / norm);
    }

    cv::RNG rng(12345);
    std::vector<float> codes((size_t)atoms * (size_t)npairs, 0.0f);
    std::vector<float> G((size_t)atoms * atoms);
    for (int iter = 0; iter < iters; ++iter) {
        const auto t0 = std::chrono::steady_clock::now();
        std::fill(codes.begin(), codes.end(), 0.0f);
        cs_gram_matrix(out_D.data(), dim, atoms, G.data());
        #pragma omp parallel for schedule(static)
        for (long long p = 0; p < npairs; ++p) {
            cs_omp_encode_gram(&pairs[(size_t)p * dim], out_D.data(), G.data(), dim, atoms,
                target_sparsity, &codes[(size_t)p * atoms]);
        }
        const auto t1 = std::chrono::steady_clock::now();
        {
            long long nnz = 0;
            for (size_t i = 0; i < codes.size(); ++i) if (codes[i] != 0.0f) ++nnz;
            std::printf("  ksvd iter %d: nonzero codes = %lld (of %lld)\n",
                iter, nnz, (long long)codes.size());
            std::fflush(stdout);
        }
        for (int a = 0; a < atoms; ++a) {
            std::vector<std::pair<float, long long>> users;
            for (long long p = 0; p < npairs; ++p) {
                const float c = codes[(size_t)p * atoms + a];
                if (c != 0.0f) users.push_back({ std::fabs(c), p });
            }
            if (users.empty()) {
                const int rp = rng.uniform(0, (int)(npairs - 1));
                float* atom = &out_D[(size_t)a * dim];
                std::memcpy(atom, &pairs[(size_t)rp * dim], sizeof(float) * (size_t)dim);
                double norm = 0.0;
                for (int q = 0; q < dim; ++q) norm += (double)atom[q] * atom[q];
                norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
                for (int q = 0; q < dim; ++q) atom[q] = (float)(atom[q] / norm);
                continue;
            }
            if (users.size() > 256) {
                std::nth_element(users.begin(), users.begin() + 255, users.end());
                users.resize(256);
            }
            const int nu = (int)users.size();
            cv::Mat E(nu, dim, CV_32F);
            for (int u = 0; u < nu; ++u) {
                std::memcpy(E.ptr<float>(u), &pairs[users[(size_t)u].second * dim],
                    sizeof(float) * (size_t)dim);
                for (int a2 = 0; a2 < atoms; ++a2) {
                    if (a2 == a) continue;
                    const float c2 = codes[users[(size_t)u].second * atoms + a2];
                    if (c2 == 0.0f) continue;
                    const float* atom2 = &out_D[(size_t)a2 * dim];
                    float* row = E.ptr<float>(u);
                    for (int q = 0; q < dim; ++q) row[q] -= atom2[q] * c2;
                }
            }
            cv::Mat w, uu, vt;
            // economy SVD: only the leading triplet (w0, u0, vt0) is used
            cv::SVD::compute(E, w, uu, vt, cv::SVD::MODIFY_A);
            float* atom = &out_D[(size_t)a * dim];
            for (int q = 0; q < dim; ++q) atom[q] = vt.at<float>(0, q);
            double norm = 0.0;
            for (int q = 0; q < dim; ++q) norm += (double)atom[q] * atom[q];
            norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
            for (int q = 0; q < dim; ++q) atom[q] = (float)(atom[q] / norm);
            for (int u = 0; u < nu; ++u) {
                codes[users[(size_t)u].second * atoms + a] = w.at<float>(0) * uu.at<float>(u, 0);
            }
        }
        const double iter_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  ksvd iter %d done in %.1fs\n", iter, iter_s);
        std::fflush(stdout);
    }
    return true;
}

// Yang-style LR gradient features: [-1,0,1] and [1,0,-2,0,1] separably in
// both orientations over the whole plane (BORDER_REPLICATE so every patch
// position is valid). feats[4] are CV_32F, same size as gray.
static void cs_dict_grad_features(const cv::Mat& gray, cv::Mat feats[4]) {
    const cv::Mat k1 = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat k1t = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    const cv::Mat k2 = (cv::Mat_<float>(1, 5) << 1, 0, -2, 0, 1);
    const cv::Mat k2t = (cv::Mat_<float>(5, 1) << 1, 0, -2, 0, 1);
    const cv::Mat one = (cv::Mat_<float>(1, 1) << 1);
    cv::sepFilter2D(gray, feats[0], CV_32F, k1, one, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(gray, feats[1], CV_32F, one, k1t, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(gray, feats[2], CV_32F, k2, one, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(gray, feats[3], CV_32F, one, k2t, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

bool cs_train_coupled_dictionary(const std::vector<cv::Mat>& images, int atoms,
    int iters, int max_pairs, int plr, cs_coupled_dict& out)
{
    if (images.empty() || atoms <= 0 || atoms > 4096 || plr < 4 || plr > 16) return false;
    const int phr = 2 * plr;
    const int hdim = phr * phr;
    const int feat = 4 * plr * plr;
    const int dim = hdim + feat;

    // gather HR/LR pairs: HR phr window (stride plr, zero-meaned) with the
    // gradient features of the box-downsampled LR window at half coords.
    // Flat patches (no gradient energy) carry no detail info: skipped, like
    // the zero-mean collapse guard in the single-dict trainer.
    std::vector<float> Xh, Yl;
    Xh.reserve(4096);
    Yl.reserve(4096);
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const cv::Mat& img = images[idx];
        if (img.empty() || img.channels() != 3) continue;
        cv::Mat gray, gray_f;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
        if (gray_f.rows < phr || gray_f.cols < phr) continue;
        cv::Mat lr;
        cv::resize(gray_f, lr, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::Mat feats[4];
        cs_dict_grad_features(lr, feats);
            for (int y = 0; y + phr <= gray_f.rows; y += plr) {
            for (int x = 0; x + phr <= gray_f.cols; x += plr) {
                const int ly = y / 2, lx = x / 2; // y,x multiples of plr: exact halves
                double fnorm = 0.0;
                // feature norm check first (cheap reject before copying)
                for (int f = 0; f < 4; ++f) {
                    for (int r = 0; r < plr; ++r) {
                        const float* frow = feats[f].ptr<float>(ly + r);
                        for (int c = 0; c < plr; ++c) {
                            const double v = frow[lx + c];
                            fnorm += v * v;
                        }
                    }
                }
                if (!(fnorm > 1e-8)) continue;
                double sum = 0.0;
                const size_t base = Xh.size();
                Xh.resize(base + (size_t)hdim);
                for (int r = 0; r < phr; ++r) {
                    const float* grow = gray_f.ptr<float>(y + r);
                    for (int c = 0; c < phr; ++c) {
                        const float v = grow[x + c];
                        Xh[base + (size_t)r * phr + c] = v;
                        sum += v;
                    }
                }
                const float mean = (float)(sum / hdim);
                for (int q = 0; q < hdim; ++q) Xh[base + (size_t)q] -= mean;
                const size_t fbase = Yl.size();
                Yl.resize(fbase + (size_t)feat);
                for (int f = 0; f < 4; ++f) {
                    for (int r = 0; r < plr; ++r) {
                        const float* frow = feats[f].ptr<float>(ly + r);
                        std::memcpy(&Yl[fbase + ((size_t)f * plr + r) * plr],
                            &frow[lx], sizeof(float) * (size_t)plr);
                    }
                }
            }
        }
    }
    long long npairs = (long long)Xh.size() / hdim;
    if (npairs < atoms || npairs <= 0) return false;
    if (npairs > max_pairs) {
        // uniform stride subsample (deterministic)
        const long long step = npairs / max_pairs + 1;
        std::vector<float> Xs, Ys;
        Xs.reserve((size_t)max_pairs * hdim);
        Ys.reserve((size_t)max_pairs * feat);
        long long kept = 0;
        for (long long p = 0; p < npairs && kept < max_pairs; ++p) {
            if (p % step != 0) continue;
            Xs.insert(Xs.end(), Xh.begin() + p * hdim, Xh.begin() + (p + 1) * hdim);
            Ys.insert(Ys.end(), Yl.begin() + p * feat, Yl.begin() + (p + 1) * feat);
            ++kept;
        }
        Xh.swap(Xs);
        Yl.swap(Ys);
        npairs = kept;
    }
    if (npairs < atoms) return false;

    // energy-balance the halves (gradient magnitudes << pixel residuals),
    // then stack [Xh/rh; Yl/rl] and run one K-SVD (Yang concatenated form).
    double rh = 0.0, rl = 0.0;
    for (size_t i = 0; i < Xh.size(); ++i) rh += (double)Xh[i] * Xh[i];
    for (size_t i = 0; i < Yl.size(); ++i) rl += (double)Yl[i] * Yl[i];
    rh = std::sqrt(rh / (Xh.size() > 0 ? Xh.size() : 1));
    rl = std::sqrt(rl / (Yl.size() > 0 ? Yl.size() : 1));
    if (!(rh > 1e-9)) rh = 1.0;
    if (!(rl > 1e-9)) rl = 1.0;
    std::vector<float> pairs((size_t)npairs * dim);
    for (long long p = 0; p < npairs; ++p) {
        float* dst = &pairs[(size_t)p * dim];
        const float* sh = &Xh[(size_t)p * hdim];
        const float* sl = &Yl[(size_t)p * feat];
        for (int q = 0; q < hdim; ++q) dst[q] = (float)(sh[q] / rh);
        for (int q = 0; q < feat; ++q) dst[hdim + q] = (float)(sl[q] / rl);
    }

    std::printf("  gathered %lld pairs (%d HR-dim + %d feat-dim)\n", npairs, hdim, feat);
    std::fflush(stdout);
    std::vector<float> Djoint;
    constexpr int train_sparsity = 8;
    if (!cs_ksvd_train(pairs.data(), npairs, dim, atoms, iters, train_sparsity, Djoint)) {
        return false;
    }

    // split + renormalize: Dl atoms to unit norm, Dh counter-scaled per atom
    // so Dh*alpha stays in patch units for codes fit on normalized Dl.
    out.atoms = atoms;
    out.plr = plr;
    out.phr = phr;
    out.feat = feat;
    out.Dh.assign((size_t)atoms * hdim, 0.0f);
    out.Dl.assign((size_t)atoms * feat, 0.0f);
    for (int a = 0; a < atoms; ++a) {
        const float* jh = &Djoint[(size_t)a * dim];
        const float* jl = &Djoint[(size_t)a * dim + hdim];
        double nl = 0.0;
        for (int q = 0; q < feat; ++q) nl += (double)jl[q] * jl[q];
        nl = std::sqrt(nl);
        if (!(nl > 1e-9)) {
            // degenerate LR half: fall back to a normalized HR-only atom so
            // the slot stays usable (flat response, mean-driven synthesis)
            double nh = 0.0;
            for (int q = 0; q < hdim; ++q) nh += (double)jh[q] * jh[q];
            nh = std::sqrt(nh > 1e-12 ? nh : 1e-12);
            for (int q = 0; q < hdim; ++q) out.Dh[(size_t)a * hdim + q] = (float)(jh[q] / nh * rh);
            for (int q = 0; q < feat; ++q) out.Dl[(size_t)a * feat + q] = 0.0f;
            out.Dl[(size_t)a * feat] = 1.0f;
            continue;
        }
        for (int q = 0; q < feat; ++q) out.Dl[(size_t)a * feat + q] = (float)(jl[q] / nl);
        const float back = (float)(rh / (rl * nl));
        for (int q = 0; q < hdim; ++q) out.Dh[(size_t)a * hdim + q] = jh[q] * back;
    }
    return true;
}

bool cs_save_coupled_dictionary(const std::string& path, const cs_coupled_dict& d) {
    if (d.atoms <= 0 || d.Dh.size() != (size_t)d.atoms * d.phr * d.phr ||
        d.Dl.size() != (size_t)d.atoms * d.feat) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) return false;
    const char magic[4] = { 'C', 'S', 'D', '2' };
    const int version = 1;
    f.write(magic, 4);
    f.write((const char*)&version, sizeof(int));
    f.write((const char*)&d.atoms, sizeof(int));
    f.write((const char*)&d.plr, sizeof(int));
    f.write((const char*)&d.phr, sizeof(int));
    f.write((const char*)&d.feat, sizeof(int));
    f.write((const char*)d.Dh.data(), (std::streamsize)(d.Dh.size() * sizeof(float)));
    f.write((const char*)d.Dl.data(), (std::streamsize)(d.Dl.size() * sizeof(float)));
    return (bool)f.good();
}

bool cs_load_coupled_dictionary(const std::string& path, cs_coupled_dict& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[4];
    int version = 0;
    f.read(magic, 4);
    f.read((char*)&version, sizeof(int));
    if (std::memcmp(magic, "CSD2", 4) != 0 || version != 1) return false;
    int atoms = 0, plr = 0, phr = 0, feat = 0;
    f.read((char*)&atoms, sizeof(int));
    f.read((char*)&plr, sizeof(int));
    f.read((char*)&phr, sizeof(int));
    f.read((char*)&feat, sizeof(int));
    if (atoms <= 0 || atoms > 4096 || plr < 4 || plr > 16 ||
        phr != 2 * plr || feat != 4 * plr * plr) return false;
    d.atoms = atoms;
    d.plr = plr;
    d.phr = phr;
    d.feat = feat;
    d.Dh.resize((size_t)atoms * phr * phr);
    d.Dl.resize((size_t)atoms * feat);
    f.read((char*)d.Dh.data(), (std::streamsize)(d.Dh.size() * sizeof(float)));
    f.read((char*)d.Dl.data(), (std::streamsize)(d.Dl.size() * sizeof(float)));
    if (!f.good() || d.Dh.empty() || d.Dl.empty()) return false;
    return true;
}

const cs_coupled_dict* cs_sr_dict_cached(const std::string& path) {
    static std::mutex m;
    static std::map<std::string, cs_coupled_dict> cache;
    static std::map<std::string, bool> failed;
    std::lock_guard<std::mutex> lk(m);
    auto ok = cache.find(path);
    if (ok != cache.end()) return &ok->second;
    if (failed.find(path) != failed.end()) return nullptr;
    cs_coupled_dict d;
    if (!cs_load_coupled_dictionary(path, d)) {
        failed[path] = true;
        std::fprintf(stderr, "Warning: --sr-dict: cannot load '%s'; using AVIR\n", path.c_str());
        return nullptr;
    }
    auto it = cache.emplace(path, std::move(d)).first;
    return &it->second;
}

// 2x box downsample (local copy: helper_functions' version is TU-private).
static void cs_dict_downsample_2x(const float* hr, float* lr, int Lr, int Lc) {
    const int Wc = Lc * 2;
    for (int i = 0; i < Lr; ++i) {
        for (int j = 0; j < Lc; ++j) {
            const int h0 = (i * 2) * Wc + (j * 2);
            lr[i * Lc + j] = 0.25f * (hr[h0] + hr[h0 + 1] + hr[h0 + Wc] + hr[h0 + Wc + 1]);
        }
    }
}

struct cs_dict_patch_code {
    float mean = 0.0f;
    float fscale = 0.0f; // ||yl|| of the (unnormalized) LR feature vector
    int idx[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    float coef[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
};

void cs_sr_upscale_dict_2x(const cv::Mat& lr_tile, const cs_coupled_dict* dict,
    cv::Mat& hr_out)
{
    hr_out.release();
    if (lr_tile.empty() || lr_tile.type() != CV_8UC3) return;
    if (!dict || dict->atoms <= 0 || dict->plr != 8 || dict->phr != 16) {
        cs_upscale_2x_avir(lr_tile, hr_out);
        return;
    }
    const int Lr = lr_tile.rows, Lc = lr_tile.cols;
    const int plr = dict->plr, phr = dict->phr, feat = dict->feat;
    if (Lr < plr || Lc < plr) { cs_upscale_2x_avir(lr_tile, hr_out); return; }
    const int Hr = Lr * 2, Wc = Lc * 2;

    // luma working planes (Yang operates on luminance)
    cv::Mat gray;
    cv::cvtColor(lr_tile, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
    cv::Mat feats[4];
    cs_dict_grad_features(gray_f, feats);

    // AVIR HR image: chroma source (kept) + fallback
    cv::Mat init_hr;
    cs_upscale_2x_avir(lr_tile, init_hr);
    if (init_hr.empty() || init_hr.rows != Hr || init_hr.cols != Wc) return;

    // patch grid: stride plr/2 with a clamped last row/column for exact cover
    std::vector<int> ys, xs;
    for (int y = 0; y + plr <= Lr; y += plr / 2) ys.push_back(y);
    for (int x = 0; x + plr <= Lc; x += plr / 2) xs.push_back(x);
    if (ys.empty() || xs.empty()) { init_hr.copyTo(hr_out); return; }
    if (ys.back() != Lr - plr) ys.push_back(Lr - plr);
    if (xs.back() != Lc - plr) xs.push_back(Lc - plr);

    // Training codes use sparsity 8 on the joint vector; inference must
    // match it on the LR half (fewer atoms underfit the features and the
    // synthesis collapses toward the mean).
    constexpr int S = 8; // inference sparsity (must match cs_dict_patch_code)
    std::vector<cs_dict_patch_code> codes(ys.size() * xs.size());
    // Dl is fixed for the tile: one Gram matrix for all patches
    std::vector<float> G((size_t)dict->atoms * dict->atoms);
    cs_gram_matrix(dict->Dl.data(), feat, dict->atoms, G.data());

    // phase 1 (parallel): OMP-code every patch against Dl. Flat patches
    // (no gradient energy) skip coding: mean-only synthesis is exact there.
    #pragma omp parallel for schedule(static)
    for (long long t = 0; t < (long long)codes.size(); ++t) {
        const int ty = ys[(size_t)(t / xs.size())];
        const int tx = xs[(size_t)(t % xs.size())];
        cs_dict_patch_code pc;
        double mean = 0.0;
        for (int r = 0; r < plr; ++r) {
            const float* grow = gray_f.ptr<float>(ty + r);
            for (int c = 0; c < plr; ++c) mean += grow[tx + c];
        }
        pc.mean = (float)(mean / (plr * plr));
        std::vector<float> yv((size_t)feat, 0.0f);
        double fnorm = 0.0;
        for (int f = 0; f < 4; ++f) {
            for (int r = 0; r < plr; ++r) {
                const float* frow = feats[f].ptr<float>(ty + r);
                for (int c = 0; c < plr; ++c) {
                    const float v = frow[tx + c];
                    yv[((size_t)f * plr + r) * plr + c] = v;
                    fnorm += (double)v * v;
                }
            }
        }
        if (fnorm > 1e-8) {
            pc.fscale = (float)std::sqrt(fnorm);
            const float n = (float)(1.0 / pc.fscale);
            for (int q = 0; q < feat; ++q) yv[(size_t)q] *= n;
            std::vector<float> cc((size_t)dict->atoms, 0.0f);
            cs_omp_encode_gram(yv.data(), dict->Dl.data(), G.data(), feat, dict->atoms, S, cc.data());
            int got = 0;
            for (int a = 0; a < dict->atoms && got < S; ++a) {
                if (cc[(size_t)a] != 0.0f) { pc.idx[got] = a; pc.coef[got] = cc[(size_t)a]; ++got; }
            }
        }
        codes[(size_t)t] = pc;
    }

    // phase 2 (serial, deterministic order): splat Dh syntheses + average
    cv::Mat canvas(Hr, Wc, CV_32F, cv::Scalar(0));
    cv::Mat weight(Hr, Wc, CV_32F, cv::Scalar(0));
    std::vector<float> hp((size_t)phr * phr);
    for (size_t t = 0; t < codes.size(); ++t) {
        const int ty = ys[t / xs.size()];
        const int tx = xs[t % xs.size()];
        const cs_dict_patch_code& pc = codes[t];
        std::fill(hp.begin(), hp.end(), pc.mean);
        for (int s = 0; s < S; ++s) {
            if (pc.idx[s] < 0) continue;
            const float* atom = &dict->Dh[(size_t)pc.idx[s] * hp.size()];
            // restore the feature magnitude removed by unit-norm coding
            const float c = pc.coef[s] * pc.fscale;
            for (size_t q = 0; q < hp.size(); ++q) hp[q] += atom[q] * c;
        }
        const int hy = ty * 2, hx = tx * 2;
        for (int r = 0; r < phr; ++r) {
            float* crow = canvas.ptr<float>(hy + r);
            float* wrow = weight.ptr<float>(hy + r);
            for (int c = 0; c < phr; ++c) {
                crow[hx + c] += hp[(size_t)r * phr + c];
                wrow[hx + c] += 1.0f;
            }
        }
    }
    for (int i = 0; i < Hr; ++i) {
        float* crow = canvas.ptr<float>(i);
        const float* wrow = weight.ptr<float>(i);
        for (int j = 0; j < Wc; ++j) {
            crow[j] = wrow[j] > 0.0f ? crow[j] / wrow[j] : crow[j];
        }
    }

    // global back-projection against the dense solved LR luma (same box
    // model as the FISTA path): 3 Landweber steps, gain 2 (stable: <= 8).
    // Load-bearing here (unlike the FISTA path): the patch canvas needs the
    // dense consistency pull, verified by ablation (off: -4.5 dB).
    {
        std::vector<float> lr_buf((size_t)Lr * Lc), corr((size_t)Hr * Wc);
        float* hr = (float*)canvas.data;
        for (int it = 0; it < 3; ++it) {
            cs_dict_downsample_2x(hr, lr_buf.data(), Lr, Lc);
            for (int i = 0; i < Lr; ++i) {
                const float* trow = gray_f.ptr<float>(i);
                for (int j = 0; j < Lc; ++j) {
                    lr_buf[(size_t)i * Lc + j] = trow[j] - lr_buf[(size_t)i * Lc + j];
                }
            }
            for (int i = 0; i < Lr; ++i) {
                for (int j = 0; j < Lc; ++j) {
                    const float v = 0.5f * lr_buf[(size_t)i * Lc + j]; // 0.25 adjoint x gain 2
                    const int h0 = (i * 2) * Wc + (j * 2);
                    corr[h0] = v; corr[h0 + 1] = v;
                    corr[h0 + Wc] = v; corr[h0 + Wc + 1] = v;
                }
            }
            for (int i = 0; i < Hr * Wc; ++i) hr[i] += corr[i];
        }
    }

    cv::Mat out_y;
    canvas.convertTo(out_y, CV_8U, 255.0);
    cv::Mat init_ycc;
    cv::cvtColor(init_hr, init_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ycc;
    cv::split(init_ycc, ycc);
    ycc[0] = out_y;
    cv::Mat merged;
    cv::merge(ycc, merged);
    cv::cvtColor(merged, hr_out, cv::COLOR_YCrCb2BGR);
}
