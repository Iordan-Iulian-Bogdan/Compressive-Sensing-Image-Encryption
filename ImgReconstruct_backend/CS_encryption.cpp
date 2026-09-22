#include "CS_encryption.hpp"
#include <cstring>
#include <numeric>
#include <stdexcept>

int CSencryption::params = AUTO_PARAM;

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
// tile, then replicate that pattern across the entire image. This ensures
// the sampling pattern is periodic - identical sampling positions within
// each tile of the image.
void CSencryption::returnPeriodicIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, const uint8_t key[32], int tile_size) {

    // MT19937 seeded from the 256-bit PBKDF2 key via std::seed_seq.
    std::seed_seq seq(key, key + 32);
    std::mt19937 generator(seq);

    // Fisher-Yates shuffle for the tile pattern, keyed off first 4 key bytes
    unsigned int shuffle_seed;
    std::memcpy(&shuffle_seed, key, sizeof(shuffle_seed));

    const int tile_pixels = tile_size * tile_size;
    if (numOfIndices < 0 || numOfIndices > tile_size * tile_size) {
        throw std::runtime_error("requested sample count exceeds the number of available pixels within one tile");
    }

    // Generate random indices within ONE canonical tile
    std::vector<int> tile_pool(tile_size * tile_size);
    std::iota(tile_pool.begin(), tile_pool.end(), 0);
    std::vector<int> tile_ri_x(numOfIndices), tile_ri_y(numOfIndices);

    for (int k = 0; k < numOfIndices; ++k) {
        std::uniform_int_distribution<int> pick(k, tile_size * tile_size - 1);
        const int j = pick(generator);
        std::swap(tile_pool[k], tile_pool[j]);
        tile_ri_x[k] = tile_pool[k] / tile_size;
        tile_ri_y[k] = tile_pool[k] % tile_size;
    }

    // Fisher-Yates shuffle the tile indices using the shuffle seed
    shuffle(tile_ri_x, shuffle_seed);
    shuffle(tile_ri_y, shuffle_seed);

    // Now tile this pattern across the entire image
    const int tiles_x = (xm + tile_size - 1) / tile_size;
    const int tiles_y = (ym + tile_size - 1) / tile_size;

    ri_x.clear();
    ri_y.clear();
    ri_x.reserve(numOfIndices * tiles_x * tiles_y);
    ri_y.reserve(numOfIndices * tiles_y * tiles_x);

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            const int base_x = tx * tile_size;
            const int base_y = ty * tile_size;
            int tile_w = (tile_size < (xm - base_x)) ? tile_size : (xm - base_x);
            int tile_h = (tile_size < (ym - base_y)) ? tile_size : (ym - base_y);

            for (int k = 0; k < numOfIndices; ++k) {
                const int local_x = tile_ri_x[k];
                const int local_y = tile_ri_y[k];
                if (local_x < tile_w && local_y < tile_h) {
                    ri_x.push_back(base_x + local_x);
                    ri_y.push_back(base_y + local_y);
                }
            }
        }
    }

    // Final shuffle with the shuffle seed for deterministic ordering
    shuffle(ri_x, shuffle_seed);
    shuffle(ri_y, shuffle_seed);
}
