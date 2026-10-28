// Parallel PNG writer (see png_fast.hpp).
#include "png_fast.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <zlib.h>

namespace {

void put_be32(std::uint8_t* p, std::uint32_t v) {
    p[0] = (std::uint8_t)(v >> 24); p[1] = (std::uint8_t)(v >> 16); p[2] = (std::uint8_t)(v >> 8); p[3] = (std::uint8_t)v;
}

// One chunk: length, type, data, CRC-32 of type + data.
bool write_chunk(std::FILE* f, const char* type, const std::uint8_t* data, std::uint32_t n, std::uint32_t crc) {
    std::uint8_t head[8], tail[4];
    put_be32(head, n);
    std::copy(type, type + 4, head + 4);
    put_be32(tail, crc);
    return std::fwrite(head, 1, 8, f) == 8 && (n == 0 || std::fwrite(data, 1, n, f) == n) && std::fwrite(tail, 1, 4, f) == 4;
}

std::uint32_t chunk_crc(const char* type, const std::uint8_t* data, std::uint32_t n) {
    uLong c = crc32(0L, (const Bytef*)type, 4);
    // crc32(c, NULL, 0) returns zlib's initial value (0), not c: IEND has no data
    return n ? (std::uint32_t)crc32(c, data, n) : (std::uint32_t)c;
}

}  // namespace

bool cs_write_png_parallel(const std::string& path, const cv::Mat& img) {
    if (img.empty() || img.type() != CV_8UC3) return false;
    const int W = img.cols, H = img.rows;
    const size_t row_bytes = 1 + (size_t)3 * W;
    const int rows_per_strip = 64;
    const int ns = (H + rows_per_strip - 1) / rows_per_strip;

    // Per strip: SUB-filtered RGB rows -> raw deflate (sync-flushed, or finished for the last strip).
    std::vector<std::vector<std::uint8_t>> out((size_t)ns);
    std::vector<uLong> adler((size_t)ns);
    std::vector<uLong> raw_len((size_t)ns);
    std::vector<char> ok((size_t)ns, 0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < ns; ++s) {
        const int y0 = s * rows_per_strip, y1 = (std::min)(H, y0 + rows_per_strip);
        std::vector<std::uint8_t> raw((size_t)(y1 - y0) * row_bytes);
        for (int y = y0; y < y1; ++y) {
            const std::uint8_t* src = img.ptr<std::uint8_t>(y);
            std::uint8_t* d = raw.data() + (size_t)(y - y0) * row_bytes;
            d[0] = 1;   // SUB
            std::uint8_t prev[3] = { 0, 0, 0 };
            for (int x = 0; x < W; ++x) {
                const std::uint8_t rgb[3] = { src[3 * x + 2], src[3 * x + 1], src[3 * x] };
                for (int c = 0; c < 3; ++c) { d[1 + 3 * x + c] = (std::uint8_t)(rgb[c] - prev[c]); prev[c] = rgb[c]; }
            }
        }
        adler[(size_t)s] = adler32(adler32(0L, Z_NULL, 0), raw.data(), (uInt)raw.size());
        raw_len[(size_t)s] = (uLong)raw.size();
        z_stream zs{};
        if (deflateInit2(&zs, 1, Z_DEFLATED, -15, 8, Z_RLE) != Z_OK) continue;
        std::vector<std::uint8_t>& o = out[(size_t)s];
        o.resize(deflateBound(&zs, (uLong)raw.size()) + 16);
        zs.next_in = raw.data(); zs.avail_in = (uInt)raw.size();
        zs.next_out = o.data(); zs.avail_out = (uInt)o.size();
        const bool last = s + 1 == ns;
        const int rc = deflate(&zs, last ? Z_FINISH : Z_SYNC_FLUSH);
        if ((last ? rc == Z_STREAM_END : rc == Z_OK) && zs.avail_in == 0) {
            o.resize(o.size() - zs.avail_out);
            ok[(size_t)s] = 1;
        }
        deflateEnd(&zs);
    }
    for (char c : ok) if (!c) return false;

    // zlib header on the first strip, combined Adler-32 after the last; one IDAT per strip.
    uLong a = adler[0];
    for (int s = 1; s < ns; ++s) a = adler32_combine(a, adler[(size_t)s], (z_off_t)raw_len[(size_t)s]);
    out[0].insert(out[0].begin(), { 0x78, 0x01 });
    std::uint8_t tail[4];
    put_be32(tail, (std::uint32_t)a);
    out.back().insert(out.back().end(), tail, tail + 4);
    std::vector<std::uint32_t> crc((size_t)ns);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < ns; ++s)
        crc[(size_t)s] = chunk_crc("IDAT", out[(size_t)s].data(), (std::uint32_t)out[(size_t)s].size());

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    static const std::uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    std::uint8_t ihdr[13];
    put_be32(ihdr, (std::uint32_t)W);
    put_be32(ihdr + 4, (std::uint32_t)H);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;   // 8 bit, RGB, deflate, adaptive filters, no interlace
    bool good = std::fwrite(sig, 1, 8, f) == 8 && write_chunk(f, "IHDR", ihdr, 13, chunk_crc("IHDR", ihdr, 13));
    for (int s = 0; good && s < ns; ++s)
        good = write_chunk(f, "IDAT", out[(size_t)s].data(), (std::uint32_t)out[(size_t)s].size(), crc[(size_t)s]);
    good = good && write_chunk(f, "IEND", nullptr, 0, chunk_crc("IEND", nullptr, 0));
    good = (std::fclose(f) == 0) && good;
    return good;
}

bool cs_write_image(const std::string& path, const cv::Mat& img) {
    std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : std::string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (ext == ".png" && img.type() == CV_8UC3 && cs_write_png_parallel(path, img)) return true;
    return cv::imwrite(path, img);
}
