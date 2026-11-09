#include "CS_encryption.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <xmmintrin.h>


void CSencryption::returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, const uint8_t key[32]) {

    // MT19937 seeded from the 256-bit PBKDF2 key via std::seed_seq. There is
    // no secondary seed funnel; the passphrase -> key mapping is the KDF.
    // NOTE: MT19937 is not cryptographically secure, see README.
    std::seed_seq seq(key, key + 32);
    std::mt19937 generator(seq);

    // the Fisher-Yates shuffles below are keyed off the first 4 key bytes so
    // encryption and decryption derive identical index orderings
    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));

    const int total = xm * ym;
    if (numOfIndices < 0 || numOfIndices > total) {
        throw std::runtime_error("requested sample count exceeds the number of available pixels");
    }

    // partial Fisher-Yates selects EXACTLY numOfIndices distinct, uniformly
    // random positions. The previous Bernoulli pass (uniform < pixel_p) could
    // undershoot numOfIndices on unlucky seeds, leaving the tail of ri_x/ri_y
    // filled with duplicate (0,0) entries.
    std::vector<int> pool(total);
    std::iota(pool.begin(), pool.end(), 0);
    for (int k = 0; k < numOfIndices; ++k) {
        std::uniform_int_distribution<int> pick(k, total - 1);
        const int j = pick(generator);
        std::swap(pool[k], pool[j]);
        ri_x[k] = pool[k] / ym;
        ri_y[k] = pool[k] % ym;
    }

    shuffle(ri_x, shuffle_seed);
    shuffle(ri_y, shuffle_seed);
}

cv::Mat cs_seg_chroma_support(const cv::Mat& labels, const std::vector<uint8_t>& rates_q, int mode) {
    if (labels.empty() || labels.type() != CV_8U)
        throw std::runtime_error("chroma support: invalid label map");
    int crows, ccols;
    cs_chroma_dims(mode, labels.rows, labels.cols, crows, ccols);
    cv::Mat sup = cv::Mat::zeros(crows, ccols, CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        const uint8_t* l = labels.ptr<uint8_t>(y);
        uint8_t* s = sup.ptr<uint8_t>((std::min)(crows - 1, (int)((long long)y * crows / labels.rows)));
        for (int x = 0; x < labels.cols; ++x)
            if (l[x] < rates_q.size() && rates_q[l[x]] > 0) s[(std::min)(ccols - 1, (int)((long long)x * ccols / labels.cols))] = 255;
    }
    return sup;
}

void CSencryption::returnSegMaskIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
    const cv::Mat& labels, const std::vector<uint8_t>& rates_q, const uint8_t key[32])
{
    if (labels.empty() || labels.rows != xm || labels.cols != ym || labels.type() != CV_8U) {
        throw std::runtime_error("seg-mask sampling: label map geometry mismatch");
    }

    std::seed_seq seq(key, key + 32);
    std::mt19937 generator(seq);

    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));

    ri_x.clear();
    ri_y.clear();

    const int nclass = (int)rates_q.size();
    auto want_of = [](size_t pix, uint8_t q) {
        if (pix == 0 || q == 0) return (size_t)0;
        const size_t want = (size_t)std::llround((double)pix * (double)q / 255.0);
        return want > pix ? pix : want;
    };
    auto draw_pool = [&](std::vector<int>& pool, uint8_t q) {
        const size_t pix = pool.size(), want = want_of(pix, q);
        // The picks depend on the stream only, not on the pool: a batch of them is drawn first (same calls, same
        // order), so the swaps can prefetch the pool entries they will touch.
        constexpr size_t kBatch = 256, kAhead = 16;
        size_t js[kBatch];
        for (size_t k0 = 0; k0 < want; k0 += kBatch) {
            const size_t n = (std::min)(kBatch, want - k0);
            for (size_t t = 0; t < n; ++t) {
                std::uniform_int_distribution<size_t> pick(k0 + t, pix - 1);
                js[t] = pick(generator);
            }
            for (size_t t = 0; t < n; ++t) {
                if (t + kAhead < n) _mm_prefetch((const char*)&pool[js[t + kAhead]], _MM_HINT_T0);
                const size_t k = k0 + t;
                std::swap(pool[k], pool[js[t]]);
                const int lin = pool[k];
                ri_x.push_back(lin / ym);
                ri_y.push_back(lin % ym);
            }
        }
    };

    // The pools of every class in one raster pass (each pool keeps raster order),
    // then the draws in ascending class order from the shared stream.
    std::vector<std::vector<int>> pools((size_t)nclass);
    for (int r = 0; r < xm; ++r) {
        const uint8_t* lrow = labels.ptr<uint8_t>(r);
        for (int col = 0; col < ym; ++col) {
            const int c = lrow[col];
            if (c >= nclass) continue;
            pools[(size_t)c].push_back(r * ym + col);
        }
    }
    size_t total_want = 0;
    for (size_t c = 0; c < rates_q.size(); ++c)
        total_want += want_of(pools[c].size(), rates_q[c]);
    ri_x.reserve(total_want);
    ri_y.reserve(total_want);
    for (size_t c = 0; c < rates_q.size(); ++c) {
        const uint8_t qc = rates_q[c];
        if (qc == 0) continue;
        draw_pool(pools[c], qc);
    }

    // the same keyed permutation on both lists, one per thread
    std::thread shuffle_y([&] { shuffle(ri_y, shuffle_seed); });
    shuffle(ri_x, shuffle_seed);
    shuffle_y.join();
}

// Coarse-grid chroma draws with domain-separated keys (Cr 0xA5, Cb 0x5A).
// m_chroma derives deterministically via cs_chroma_count for the mode.
void CSencryption::returnYcc420ChromaIndices(int xm, int ym, int m_luma, const uint8_t key[32], int mode) {
    int crows, ccols;
    cs_chroma_dims(mode, xm, ym, crows, ccols);
    m_chroma = cs_chroma_count(mode, m_luma, xm, ym);

    ri_cx1.resize((size_t)m_chroma);
    ri_cy1.resize((size_t)m_chroma);
    ri_cx2.resize((size_t)m_chroma);
    ri_cy2.resize((size_t)m_chroma);

    uint8_t key_cr[32], key_cb[32];
    cs_ycc420_channel_key(key, CS_YCC420_TAG_CR, key_cr);
    cs_ycc420_channel_key(key, CS_YCC420_TAG_CB, key_cb);

    // Seg prefix version 2: partial Fisher-Yates over the support pixels (raster order) from the plane's key.
    std::vector<int> pool;
    if (!chroma_support.empty()) {
        if (chroma_support.rows != crows || chroma_support.cols != ccols || chroma_support.type() != CV_8U)
            throw std::runtime_error("chroma support: geometry mismatch (internal error)");
        for (int r = 0; r < crows; ++r) {
            const uint8_t* s = chroma_support.ptr<uint8_t>(r);
            for (int c = 0; c < ccols; ++c) if (s[c]) pool.push_back(r * ccols + c);
        }
        if ((int)pool.size() < m_chroma) throw std::runtime_error("chroma support smaller than the chroma count");
    }
    const auto draw = [&](std::vector<int>& rx, std::vector<int>& ry, const uint8_t* k) {
        if (pool.empty()) { returnRandomIndices(rx, ry, crows, ccols, m_chroma, k); return; }
        std::vector<int> p(pool);
        std::seed_seq seq(k, k + 32);
        std::mt19937 gen(seq);
        for (int i = 0; i < m_chroma; ++i) {
            std::uniform_int_distribution<size_t> pick((size_t)i, p.size() - 1);
            std::swap(p[(size_t)i], p[pick(gen)]);
            rx[(size_t)i] = p[(size_t)i] / ccols;
            ry[(size_t)i] = p[(size_t)i] % ccols;
        }
    };

    // the two planes draw from independent keys: Cb on a second thread
    std::exception_ptr cb_error;
    std::thread cb([&] {
        try { draw(ri_cx2, ri_cy2, key_cb); }
        catch (...) { cb_error = std::current_exception(); }
    });
    try { draw(ri_cx1, ri_cy1, key_cr); }
    catch (...) { cb.join(); throw; }
    cb.join();
    if (cb_error) std::rethrow_exception(cb_error);

    cs_wipe(key_cr, sizeof(key_cr));
    cs_wipe(key_cb, sizeof(key_cb));
}
