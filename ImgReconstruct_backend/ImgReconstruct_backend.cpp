#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <iostream>

namespace {

void print_usage(const char* exe) {
    std::cout <<
        "Compressive-sensing image encryption\n"
        "\n"
        "Usage:\n"
        "  " << exe << " encrypt <input.png> <output.png> [options]\n"
        "  " << exe << " decrypt <input.png> <output.png> [options]\n"
        "  " << exe << " roundtrip <input.png> <output.png> [options]\n"
        "\n"
"Options:\n"
        "  --password <pw>       passphrase (else $CS_PASSWORD, else interactive prompt)\n"
        "  --ratio <f>           encryption sampling ratio in (0.05, 1.0]; default 1.0\n"
        "  --tiles <n>           decrypt tile count (manual mode only)\n"
        "  --overlap <n>         decrypt tile overlap in (24, 96) (manual mode only)\n"
        "  --iterations <n>      decrypt solver iterations (manual mode only)\n"
        "  --threads <n>         decrypt worker threads (manual mode only)\n"
        "  --coef <f>            decrypt solver coefficient in (0.01, 0.05) (manual mode only)\n"
        "  --manual              force manual parameter mode (auto is the default and\n"
        "                        derives tiles/overlap/iterations/coef from the image)\n"
        "  --no-preview          disable the live decryption preview window\n"
        "  --denoise             apply a final non-local-means denoise pass\n"
        "                        (smooths solver noise, blurs fine detail)\n"
        "  --tv <f>              total-variation fusion weight for the solver\n"
        "                        (>0 enables TV, smoother edges, helps at high\n"
        "                        compression; 0 = off)\n"
        "  --dict <file>         solve with a learned K-SVD patch dictionary\n"
        "                        instead of the DCT basis (see the trainer tool)\n"
        "  --periodic            use periodic tile-based sampling (repeat a random\n"
        "                        tile pattern across the image)\n"
        "  --tile-size <n>       tile size for periodic mode (default 64, must divide image dims)\n"
        "\n"
        "  roundtrip encrypts the input and then decrypts the in-memory result\n"
        "\n"
        "Examples:\n"
        "  " << exe << " encrypt photo.png photo.enc.png\n"
        "  " << exe << " decrypt photo.enc.png photo.dec.png --password \"my secret\"\n"
        "  " << exe << " roundtrip photo.png photo.dec.png --password \"my secret\"\n"
        "  " << exe << " encrypt photo.png photo.enc.png --ratio 0.5 --periodic --tile-size 64\n";
}

std::string read_password() {
    std::string env;
    const char* env_pw = std::getenv("CS_ENCRYPTION_PASSWORD");
    if (env_pw && env_pw[0]) {
        std::cout << "Using passphrase from CS_ENCRYPTION_PASSWORD" << std::endl;
        return env_pw;
    }
    std::cout << "Enter passphrase: " << std::flush;
    std::string pw;
    std::getline(std::cin, pw);
    return pw;
}

bool parse_int(const char* s, int& out) {
    try { size_t pos; long v = std::stol(s, &pos); if (pos != std::strlen(s)) return false; out = (int)v; return true; }
    catch (...) { return false; }
}

bool parse_float(const char* s, float& out) {
    try { size_t pos; float v = std::stof(s, &pos); if (pos != std::strlen(s)) return false; out = v; return true; }
    catch (...) { return false; }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 64; // usage error
    }

    const std::string mode = argv[1];
    if (mode != "encrypt" && mode != "decrypt" && mode != "roundtrip") {
        std::cerr << "Error: unknown mode '" << mode << "'" << std::endl;
        print_usage(argv[0]);
        return 64;
    }

    if (argc < 4) {
        std::cerr << "Error: missing input/output paths" << std::endl;
        print_usage(argv[0]);
        return 64;
    }

    const std::string input = argv[2];
    const std::string output = argv[3];

    std::string password;
    bool have_password = false;
    float ratio = 1.0f;
    int tiles = 24, overlap = 24, iterations = 5, threads = 8;
    float coef = 0.01f;
    float tv_lambda = 0.0f;
    std::string dict_path;
    bool manual = false;
    bool show_preview = true;
    bool denoise = false;
    bool periodic = false;
    int tile_size = 64;

    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires a value (" << what << ")" << std::endl;
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--password") {
            const char* v = next("passphrase text");
            if (!v) return 64;
            password = v; have_password = true;
        }
        else if (a == "--ratio") {
            const char* v = next("floating point");
            if (!v || !parse_float(v, ratio)) return 64;
        }
        else if (a == "--tiles") {
            const char* v = next("integer");
            if (!v || !parse_int(v, tiles)) return 64;
            manual = true;
        }
        else if (a == "--overlap") {
            const char* v = next("integer");
            if (!v || !parse_int(v, overlap)) return 64;
            manual = true;
        }
        else if (a == "--iterations") {
            const char* v = next("integer");
            if (!v || !parse_int(v, iterations)) return 64;
            manual = true;
        }
        else if (a == "--threads") {
            const char* v = next("integer");
            if (!v || !parse_int(v, threads)) return 64;
            manual = true;
        }
        else if (a == "--coef") {
            const char* v = next("floating point");
            if (!v || !parse_float(v, coef)) return 64;
            manual = true;
        }
        else if (a == "--manual") {
            manual = true;
        }
        else if (a == "--no-preview") {
            show_preview = false;
        }
        else if (a == "--denoise") {
            denoise = true;
        }
        else if (a == "--tv") {
            const char* v = next("floating point");
            if (!v || !parse_float(v, tv_lambda)) return 64;
            manual = true;
        }
        else if (a == "--dict") {
            const char* v = next("dictionary file");
            if (!v) return 64;
            dict_path = v;
        }
        else if (a == "--periodic") {
            periodic = true;
            manual = true;
        }
        else if (a == "--tile-size") {
            const char* v = next("integer");
            if (!v || !parse_int(v, tile_size)) return 64;
            manual = true;
        }
        else {
            std::cerr << "Error: unknown option '" << a << "'" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
    }

    if (!have_password) {
        password = read_password();
    }

    if (password.size() < 10) {
        std::cerr << "Error: password should be at least 10 characters" << std::endl;
        return 64;
    }

    CSencryption::params = manual ? MANUAL_PARAM : AUTO_PARAM;

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

try {
        if (mode == "encrypt") {
            if (periodic) {
                rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, tile_size);
            } else {
                rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio);
            }
        }
        else if (mode == "decrypt") {
            rc = decrypt_image::decrypt_image_tiled(input, output, password,
                tiles, overlap, iterations, threads, coef, show_preview, denoise, tv_lambda, dict_path);
        }
else { // roundtrip: decrypt the in-memory encrypted image, no disk roundtrip
            cv::Mat encrypted;
            rc = encrypt_image::encrypt_image_tiled(input, output, password, ratio, 64, &encrypted);
            if (rc == 0) {
                rc = decrypt_image::decrypt_image_tiled(encrypted, output, password,
                    tiles, overlap, iterations, threads, coef, show_preview, denoise, tv_lambda, dict_path);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 3;
    }

    if (rc != 0) {
        std::cerr << (mode == "decrypt" ? "Decryption" : "Encryption") << " failed (code " << rc << ")" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    return 0;
}



