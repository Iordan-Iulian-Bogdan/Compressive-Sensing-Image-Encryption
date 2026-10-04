#include "CS_encryption.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

int CSencryption::params = AUTO_PARAM;

std::vector<int> cs_compute_tile_sample_counts(const std::vector<uint8_t>& lod, int m_total,
    int rows, int cols, int tile_size, int weight_base, int full_threshold)
{
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }
    if (m_total < 0) {
        throw std::runtime_error("adaptive sampling: negative measurement count");
    }
    if (weight_base < 1) weight_base = 1;
    if (weight_base > 65535) weight_base = 65535;
    const int tiles_x = (cols + tile_size - 1) / tile_size; // columns
    const int tiles_y = (rows + tile_size - 1) / tile_size; // rows
    const int nt = tiles_x * tiles_y;
    if ((int)lod.size() != nt) {
        throw std::runtime_error("adaptive sampling: lod length does not match tile grid");
    }

    std::vector<int> cap(nt);
    std::vector<long long> w(nt);
    for (int tr = 0; tr < tiles_y; ++tr) {
        for (int tc = 0; tc < tiles_x; ++tc) {
            const int i = tr * tiles_x + tc;
            const int row0 = tr * tile_size;
            const int col0 = tc * tile_size;
            const int tile_w = (tile_size < cols - col0) ? tile_size : cols - col0;
            const int tile_h = (tile_size < rows - row0) ? tile_size : rows - row0;
            cap[i] = tile_w * tile_h;
            // weight = capacity * (weight_base + lod): density ratio is
            // (base+255)/(base+32). base is the aggression knob — high
            // (e.g. 65535) is near-uniform, 1 is max LOD bias, 256 is the
            // tuned default (--adaptive-strength 0.5). Capacity keeps total
            // allocation proportional to tile size.
            w[i] = (long long)cap[i] * ((long long)weight_base + (long long)lod[i]);
        }
    }

    std::vector<int> counts(nt, 0);
    std::vector<char> is_full(nt, 0);
    long long pool_budget = m_total;
    if (full_threshold > 0 && m_total > 0) {
        // --lod-full: qualifying tiles (lod >= threshold) take their whole
        // pixel capacity first, greedy fit in (lod desc, index asc) order
        // against the budget; the pool below splits whatever remains
        std::vector<int> order;
        order.reserve((size_t)nt);
        for (int i = 0; i < nt; ++i)
            if (lod[i] >= full_threshold) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (lod[a] != lod[b]) return lod[a] > lod[b];
            return a < b;
        });
        long long cost = 0;
        for (int i : order) {
            if (cost + cap[i] <= m_total) {
                is_full[i] = 1;
                counts[i] = cap[i];
                cost += cap[i];
            }
        }
        pool_budget = m_total - cost;
    }

    // pool = tiles still splitting the (remaining) budget via the standard
    // largest-remainder machinery; with threshold 0 this is every tile and
    // the allocation below is bit-identical to the legacy path
    std::vector<int> pool;
    pool.reserve((size_t)nt);
    for (int i = 0; i < nt; ++i)
        if (!is_full[i]) pool.push_back(i);
    const int np = (int)pool.size();
    if (np == 0 || pool_budget <= 0) {
        return counts;
    }
    long long poolW = 0;
    for (int i : pool) poolW += w[i];
    if (poolW == 0) {
        return counts;
    }

    // largest-remainder proportional split: floor(budget * w_i / poolW),
    // then hand the leftover one-by-one in (remainder desc, index asc)
    // order — fully deterministic, no floating point
    long long assigned = 0;
    std::vector<std::pair<long long, int>> frac;
    frac.reserve((size_t)np);
    for (int i : pool) {
        const long long num = pool_budget * w[i];
        counts[i] = (int)(num / poolW);
        assigned += counts[i];
        frac.emplace_back(num % poolW, i);
    }
    std::sort(frac.begin(), frac.end(), [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    long long left = pool_budget - assigned;
    for (int k = 0; left > 0; --left, ++k) {
        counts[frac[(size_t)(k % np)].second]++;
    }

    // Clamp to per-tile capacity and redistribute until sum(counts) reaches
    // target = min(budget, pool capacity). (History: an earlier version
    // broke out of the remainder pass on the first full tile, leaving the
    // sum short under skewed LOD; the loop below re-seats the excess.)
    long long pool_cap = 0;
    for (int i : pool) pool_cap += cap[i];
    const long long target = (std::min)(pool_budget, pool_cap);

    for (int pass = 0; pass < 2 * np + 16; ++pass) {
        long long total = 0;
        for (int i : pool) {
            if (counts[i] > cap[i]) counts[i] = cap[i];
            total += counts[i];
        }
        if (total == target) break;

        if (total > target) {
            // strip from highest-index pool tiles until exact (deterministic)
            for (int k = np - 1; k >= 0 && total > target; --k) {
                const int i = pool[(size_t)k];
                if (counts[i] > 0) {
                    const int take = (std::min)(counts[i], (int)(total - target));
                    counts[i] -= take;
                    total -= take;
                }
            }
            break;
        }

        // total < target: place the shortfall among pool tiles that still
        // have headroom — largest-remainder by weight, then one-by-one
        const long long need = target - total;
        long long headroom_w = 0;
        for (int i : pool) {
            if (counts[i] < cap[i]) headroom_w += w[i];
        }
        if (headroom_w == 0) break;  // every pool tile full

        long long given = 0;
        std::vector<std::pair<long long, int>> fr;
        fr.reserve((size_t)np);
        for (int i : pool) {
            if (counts[i] >= cap[i]) continue;
            const long long num = need * w[i];
            const long long give = num / headroom_w;
            const long long g = (std::min)(give, (long long)(cap[i] - counts[i]));
            counts[i] += (int)g;
            given += g;
            fr.emplace_back(num % headroom_w, i);
        }
        std::sort(fr.begin(), fr.end(), [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        long long rem = need - given;
        if (rem > 0 && !fr.empty()) {
            size_t idx = 0;
            int stagnant = 0;
            while (rem > 0 && stagnant < (int)fr.size()) {
                const int i = fr[idx % fr.size()].second;
                if (counts[i] < cap[i]) {
                    counts[i]++;
                    --rem;
                    stagnant = 0;
                }
                else {
                    ++stagnant;
                }
                ++idx;
            }
        }
    }
    return counts;
}

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

// Periodic tile-based sampling: generate random indices within one canonical
// tile, then replicate that pattern across the entire image. Every tile of
// the image carries the SAME sampling positions (identical local offsets).
// samples_per_tile positions are drawn inside one tile_size x tile_size block.
// The result is fully deterministic from (key, geometry) so encryption and
// decryption regenerate identical index sets.
void CSencryption::returnPeriodicIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int samples_per_tile, const uint8_t key[32], int tile_size) {

    std::seed_seq seq(key, key + 32);
    std::mt19937 generator(seq);

    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));

    const int tile_pixels = tile_size * tile_size;
    if (tile_size <= 0) {
        throw std::runtime_error("periodic sampling: tile size must be positive");
    }
    if (samples_per_tile < 1 || samples_per_tile > tile_pixels) {
        throw std::runtime_error("requested samples per tile exceeds the tile's pixel count");
    }

    // draw samples_per_tile distinct positions inside the canonical tile
    std::vector<int> tile_pool(tile_pixels);
    std::iota(tile_pool.begin(), tile_pool.end(), 0);
    std::vector<int> tile_ri_x(samples_per_tile), tile_ri_y(samples_per_tile);
    for (int k = 0; k < samples_per_tile; ++k) {
        std::uniform_int_distribution<int> pick(k, tile_pixels - 1);
        const int j = pick(generator);
        std::swap(tile_pool[k], tile_pool[j]);
        tile_ri_x[k] = tile_pool[k] / tile_size;
        tile_ri_y[k] = tile_pool[k] % tile_size;
    }

    // replicate the pattern across the whole image; edge tiles keep only the
    // positions that fall inside the image (deterministic on both sides)
    const int tiles_x = (xm + tile_size - 1) / tile_size;
    const int tiles_y = (ym + tile_size - 1) / tile_size;

    ri_x.clear();
    ri_y.clear();
    ri_x.reserve((size_t)samples_per_tile * tiles_x * tiles_y);
    ri_y.reserve((size_t)samples_per_tile * tiles_x * tiles_y);

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            const int base_x = tx * tile_size;
            const int base_y = ty * tile_size;
            const int tile_w = (tile_size < (xm - base_x)) ? tile_size : (xm - base_x);
            const int tile_h = (tile_size < (ym - base_y)) ? tile_size : (ym - base_y);

            for (int k = 0; k < samples_per_tile; ++k) {
                const int local_x = tile_ri_x[k];
                const int local_y = tile_ri_y[k];
                if (local_x < tile_w && local_y < tile_h) {
                    ri_x.push_back(base_x + local_x);
                    ri_y.push_back(base_y + local_y);
                }
            }
        }
    }

    // deterministic measurement ordering (same on both sides)
    shuffle(ri_x, shuffle_seed);
    shuffle(ri_y, shuffle_seed);
}

// Adaptive periodic sampling: per-tile sample counts come from the LOD byte
// array (high detail -> more samples), each tile draws its own in-tile
// pattern from the shared RNG in row-major tile order, then one global
// shuffle fixes the measurement order. Deterministic from (key, lod, m, geom,
// weight_base).
void CSencryption::returnAdaptiveIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
    const std::vector<uint8_t>& lod, int m_total, const uint8_t key[32], int tile_size,
    int weight_base, int full_threshold)
{
    if (tile_size <= 0) {
        throw std::runtime_error("adaptive sampling: tile size must be positive");
    }

    std::seed_seq seq(key, key + 32);
    std::mt19937 generator(seq);

    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));

    const std::vector<int> counts = cs_compute_tile_sample_counts(lod, m_total, xm, ym, tile_size, weight_base, full_threshold);
    // same grid as cs_compute_tile_sample_counts: tiles_cols across ym (cols),
    // tiles_rows across xm (rows), lod index = tr * tiles_cols + tc
    const int tiles_cols = (ym + tile_size - 1) / tile_size;
    const int tiles_rows = (xm + tile_size - 1) / tile_size;

    // exact index count (capacity clamp may drop a few when m > total pixels)
    int total_draw = 0;
    for (int c : counts) total_draw += c;

    ri_x.clear();
    ri_y.clear();
    ri_x.reserve((size_t)total_draw);
    ri_y.reserve((size_t)total_draw);

    for (int tr = 0; tr < tiles_rows; ++tr) {
        for (int tc = 0; tc < tiles_cols; ++tc) {
            const int i = tr * tiles_cols + tc;
            const int s = counts[i];
            if (s <= 0) continue;

            const int row0 = tr * tile_size;
            const int col0 = tc * tile_size;
            const int tile_w = (tile_size < ym - col0) ? tile_size : ym - col0; // extent in cols
            const int tile_h = (tile_size < xm - row0) ? tile_size : xm - row0; // extent in rows
            const int tile_pixels = tile_w * tile_h;
            if (s > tile_pixels) continue;  // defensive; counts already capped

            // partial Fisher-Yates over this tile only (same draw style as
            // periodic/random so the key stream advances identically both sides)
            std::vector<int> pool((size_t)tile_pixels);
            std::iota(pool.begin(), pool.end(), 0);
            for (int k = 0; k < s; ++k) {
                std::uniform_int_distribution<int> pick(k, tile_pixels - 1);
                const int j = pick(generator);
                std::swap(pool[(size_t)k], pool[(size_t)j]);
                const int local = pool[(size_t)k];
                ri_x.push_back(row0 + local / tile_w);
                ri_y.push_back(col0 + local % tile_w);
            }
        }
    }

    shuffle(ri_x, shuffle_seed);
    shuffle(ri_y, shuffle_seed);
}

// YCC 4:2:0 index generation (mode 3): m_luma full-res luma positions from
// the key plus m_chroma coarse-grid positions per chroma plane from tagged
// key copies. Fills ri_x/ri_y (luma) and ri_cx1/ri_cy1 (Cr), ri_cx2/ri_cy2
// (Cb); m_chroma derives deterministically via cs_ycc420_chroma_count.
void CSencryption::returnYcc420Indices(int xm, int ym, int m_luma, const uint8_t key[32]) {
    ri_x.resize((size_t)m_luma);
    ri_y.resize((size_t)m_luma);
    returnRandomIndices(ri_x, ri_y, xm, ym, m_luma, key);
    returnYcc420ChromaIndices(xm, ym, m_luma, key);
}

// Coarse-grid chroma draws with domain-separated keys (Cr 0xA5, Cb 0x5A).
// m_chroma derives deterministically via cs_ycc420_chroma_count.
void CSencryption::returnYcc420ChromaIndices(int xm, int ym, int m_luma, const uint8_t key[32]) {
    int crows, ccols;
    cs_ycc420_chroma_dims(xm, ym, crows, ccols);
    m_chroma = cs_ycc420_chroma_count(m_luma, xm, ym);

    ri_cx1.resize((size_t)m_chroma);
    ri_cy1.resize((size_t)m_chroma);
    ri_cx2.resize((size_t)m_chroma);
    ri_cy2.resize((size_t)m_chroma);

    uint8_t key_cr[32], key_cb[32];
    cs_ycc420_channel_key(key, CS_YCC420_TAG_CR, key_cr);
    cs_ycc420_channel_key(key, CS_YCC420_TAG_CB, key_cb);

    returnRandomIndices(ri_cx1, ri_cy1, crows, ccols, m_chroma, key_cr);
    returnRandomIndices(ri_cx2, ri_cy2, crows, ccols, m_chroma, key_cb);

    cs_wipe(key_cr, sizeof(key_cr));
    cs_wipe(key_cb, sizeof(key_cb));
}

void cs_hf_thumb_weights(const cv::Mat& thumbnail, std::vector<float>& weights,
                         int& weight_cols, int& weight_rows) {
    weights.clear();
    weight_cols = 0;
    weight_rows = 0;
    if (thumbnail.empty() || thumbnail.channels() != 3) return;
    cv::Mat gray, laplacian, absolute;
    cv::cvtColor(thumbnail, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_32F);
    absolute = cv::abs(laplacian);
    if (!absolute.isContinuous()) absolute = absolute.clone();
    weight_cols = absolute.cols;
    weight_rows = absolute.rows;
    const float* data = reinterpret_cast<const float*>(absolute.data);
    weights.assign(data, data + static_cast<size_t>(weight_cols) * weight_rows);
}

void CSencryption::returnHfWeightedIndices(
    std::vector<int>& out_rows, std::vector<int>& out_cols, int image_rows,
    int image_cols, const std::vector<uint8_t>& lod, int total_samples,
    const uint8_t key[32], int tile_size, int weight_base,
    const std::vector<float>& weights, int weight_cols, int weight_rows,
    float pixel_floor) {
    if (tile_size <= 0 || weights.empty() || weight_cols <= 0 || weight_rows <= 0) {
        throw std::runtime_error("HF-focus sampling requires a valid tile and thumbnail");
    }
    if (pixel_floor <= 0.0f) pixel_floor = 1.0f;

    std::seed_seq seed(key, key + 32);
    std::mt19937 generator(seed);
    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));
    const std::vector<int> counts = cs_compute_tile_sample_counts(
        lod, total_samples, image_rows, image_cols, tile_size, weight_base);
    const int tile_cols = (image_cols + tile_size - 1) / tile_size;
    const int tile_rows = (image_rows + tile_size - 1) / tile_size;
    int sample_capacity = 0;
    for (int count : counts) sample_capacity += count;
    out_rows.clear();
    out_cols.clear();
    out_rows.reserve(static_cast<size_t>(sample_capacity));
    out_cols.reserve(static_cast<size_t>(sample_capacity));

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> keys;
    std::vector<int> order;
    for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
        for (int tile_col = 0; tile_col < tile_cols; ++tile_col) {
            const int tile_index = tile_row * tile_cols + tile_col;
            const int sample_count = counts[tile_index];
            if (sample_count <= 0) continue;
            const int row0 = tile_row * tile_size;
            const int col0 = tile_col * tile_size;
            const int tile_width = (std::min)(tile_size, image_cols - col0);
            const int tile_height = (std::min)(tile_size, image_rows - row0);
            const int pixel_count = tile_width * tile_height;
            if (sample_count >= pixel_count) {
                for (int i = 0; i < pixel_count; ++i) {
                    uniform(generator);
                    out_rows.push_back(row0 + i / tile_width);
                    out_cols.push_back(col0 + i % tile_width);
                }
                continue;
            }
            keys.resize(static_cast<size_t>(pixel_count));
            for (int i = 0; i < pixel_count; ++i) {
                const int local_row = i / tile_width;
                const int local_col = i % tile_width;
                const int thumb_row = (std::min)(weight_rows - 1,
                    (row0 + local_row) * weight_rows / image_rows);
                const int thumb_col = (std::min)(weight_cols - 1,
                    (col0 + local_col) * weight_cols / image_cols);
                const double weight = pixel_floor + weights[
                    static_cast<size_t>(thumb_row) * weight_cols + thumb_col];
                keys[static_cast<size_t>(i)] = std::pow(uniform(generator), 1.0 / weight);
            }
            order.resize(static_cast<size_t>(pixel_count));
            std::iota(order.begin(), order.end(), 0);
            std::nth_element(order.begin(), order.begin() + sample_count, order.end(),
                [&](int a, int b) {
                    return keys[static_cast<size_t>(a)] > keys[static_cast<size_t>(b)];
                });
            for (int i = 0; i < sample_count; ++i) {
                const int local = order[static_cast<size_t>(i)];
                out_rows.push_back(row0 + local / tile_width);
                out_cols.push_back(col0 + local % tile_width);
            }
        }
    }
    shuffle(out_rows, shuffle_seed);
    shuffle(out_cols, shuffle_seed);
}
