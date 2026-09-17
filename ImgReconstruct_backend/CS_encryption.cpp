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
