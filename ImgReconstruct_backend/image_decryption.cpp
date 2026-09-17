#include "image_decryption.hpp"
#include "image_encryption.hpp"
#include <cstring>

decrypt_image::decrypt_image(const std::string input_path, const std::string& password) {
    parse_container_header(cv::imread(input_path, cv::IMREAD_COLOR), password);
}

decrypt_image::decrypt_image(cv::Mat input, const std::string& password) {
    parse_container_header(input, password);
}

void decrypt_image::parse_container_header(const cv::Mat& input, const std::string& password) {
    if (input.empty()) {
        throw std::runtime_error("failed to load encrypted image");
    }

    input.copyTo(encrypted_img);
    encrypted_img = encrypted_img.reshape(0, encrypted_img.total());

    CsHeaderInfo info;
    if (!cs_parse_header(encrypted_img.data, encrypted_img.total() * 3, password, info)) {
        if (info.legacy) {
            throw std::runtime_error("legacy container format: this image was encrypted with an older version, please re-encrypt");
        }
        if (info.auth_failed) {
            throw std::runtime_error("authentication failed: wrong password or corrupted/tampered image");
        }
        throw std::runtime_error("failed to parse encrypted image header");
    }

    m = info.m;
    rows = info.rows;
    cols = info.cols;
    org_size.height = info.org_h;
    org_size.width = info.org_w;

    if (info.key_valid) {
        std::memcpy(cs_key, info.key, sizeof(cs_key));
        cs_key_valid = true;
        cs_wipe(info.key, sizeof(info.key));
    }
}

void decrypt_image::decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out) {

    // c1/c2 get half the iterations: later channels are warm-started from the previous
    // channel's solution, so they need far fewer iterations to converge; capping them
    // harder trims nearly-free solver time (~30% less solver work at same quality target)
    int num_iterations_offset = (num_iterations / 2);

    // Reverted to per-channel saves: A merged single lbfgs over all 3 channels measured
    // ~2.3x slower wall time (~7s vs ~3s) because channels must share convergence/step
    // (30 iters x 3 planes instead of staggered 30/20/20). If re-enabled prefer
    // reconstruct_image_packed below.
    reconstruct_color_channel(encrypted_img, 0, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[0], true, ref[1]);
    reconstruct_color_channel(encrypted_img, 1, coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[1], true, ref[2]);
    reconstruct_color_channel(encrypted_img, 2, coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[2], false);

    cv::merge(ref, 3, out);
    out.convertTo(out, CV_8UC3);
}

void decrypt_image::get_mat(cv::Mat& dest) {
    decrypted_img.copyTo(dest);
}

cv::Mat decrypt_image::get_mat() {
    return decrypted_img.clone();
}

void decrypt_image::writeDecryptedImageToDisk(std::string output_path, bool remove_noise, int noise_level) {

    cv::imwrite(output_path, decrypted_img);

    if (remove_noise) {
        cv::Mat decrypted_img_noisless = cv::imread(output_path, cv::IMREAD_COLOR);

        cv::fastNlMeansDenoisingColored(decrypted_img_noisless, decrypted_img_noisless, noise_level);
        cv::imwrite(output_path, decrypted_img_noisless);
    }
}

void decrypt_image::get_sampled_mat(cv::Mat& sampled_mat, cv::Mat& masked_mat) {
    if (!cs_key_valid) {
        throw std::runtime_error("derived key not available: header must be parsed with the password first");
    }

    sampled_mat = cv::Mat(rows, cols, CV_8UC3);
    masked_mat = cv::Mat(rows, cols, CV_8UC3);

    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, cs_key);

    for (int i = CS_HEADER_PIXELS; i < ri_x.size() + CS_HEADER_PIXELS - 32; i += 32) {
        // I think this helps with cache hits
        for (int j = 0; j < 32; ++j) {
            sampled_mat.at<cv::Vec3b>(ri_x[i - CS_HEADER_PIXELS + j], ri_y[i - CS_HEADER_PIXELS + j]) = encrypted_img.at<cv::Vec3b>(i + j);
            masked_mat.at<cv::Vec3b>(ri_x[i - CS_HEADER_PIXELS + j], ri_y[i - CS_HEADER_PIXELS + j]) = cv::Vec3b(1, 1, 1);
        }
    }
}

float decrypt_image::get_compression_ratio() {
    return float(m) / (rows * cols);
}

cv::Size decrypt_image::get_org_size() {
    return org_size;
}

// decrypts tiles in parallel
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::string> processing_order, int iterations, cv::Size tile_size, float coef) {

    // we use a reference image as the initial solution
    // this helps speed up convergence
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);

    // the loop is driven by processing_order instead of a squared grid
    // dimension: it already holds exactly one entry per tile, so non-square
    // tile layouts no longer silently break
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < (int)processing_order.size(); k++) {
        try {
            std::vector<std::string> splitText = splitString(processing_order[k], '_');
            int tile_i = std::stoi(splitText[0]);
            int tile_j = std::stoi(splitText[1]);

            if (tile_i < 0 || tile_j < 0 ||
                tile_i >= (int)mats_in.size() || tile_j >= (int)mats_in[tile_i].size() ||
                tile_i >= (int)indices.size() || tile_j >= (int)indices[tile_i].size() ||
                tile_i >= (int)mats_out.size() || tile_j >= (int)mats_out[tile_i].size()) {
                #pragma omp critical
                {
                    std::cerr << "Tile (" << tile_i << ", " << tile_j << ") is outside the tile grid, skipping" << std::endl;
                }
                continue;
            }

            // it's faster to make a copy of the original reference rather than create one every time
            cv::Mat copied[3];

            // renamed to ch: this counter used to shadow the parsed tile
            // index 'i' above, which made the loop very easy to misread
            for (size_t ch = 0; ch < ref.size() && ch < 3; ++ch) {
                copied[ch] = ref[ch].clone();
            }

            decrypt_image dimgs = decrypt_image(mats_in[tile_i][tile_j]);
            dimgs.decrypt(copied, indices[tile_i][tile_j].ri_x_g, indices[tile_i][tile_j].ri_y_g, iterations, coef, mats_out[tile_i][tile_j]);
        }
        catch (const std::exception& e) {
            // an exception escaping an OpenMP region terminates the process;
            // log it and degrade this one tile gracefully instead
            #pragma omp critical
            {
                std::cerr << "Tile decryption failed: " << e.what() << std::endl;
            }
        }
    }
}

int decrypt_image::decrypt_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& password,
    int num_tiles,
    int overlap,
    int iterations,
    int nun_threads,
    float coef,
    bool show_preview
) {
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    return decrypt_image_tiled(encrypted_img_g, output_path, password, num_tiles, overlap, iterations, nun_threads, coef, show_preview);
}

int decrypt_image::decrypt_image_tiled(
    const cv::Mat& encrypted_img_in,
    const std::string& output_path,
    const std::string& password,
    int num_tiles,
    int overlap,
    int iterations,
    int nun_threads,
    float coef,
    bool show_preview
) {
    const cv::Mat& encrypted_img_g = encrypted_img_in;
    try {
        if (CSencryption::params == MANUAL_PARAM) {
            if (overlap < 24 || overlap > 96) {
                throw std::runtime_error("Overlap is outside the acceptable range of (24, 96)");
            }

            if (num_tiles < 24) {
                throw std::runtime_error("Number of tiles is less than 24");
            }

            if (coef < 0.01f || coef > 0.05f) {
                throw std::runtime_error("Coef is outside of the acceptable range of (0.01, 0.05)");
            }

            if (nun_threads < 1) {
                throw std::runtime_error("Number of threads is less than 1");
            }

            if (CSencryption::params != AUTO_PARAM && CSencryption::params != MANUAL_PARAM) {
                throw std::runtime_error("Parameters type is incorrect");
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }    
    // header parse + PBKDF2 key derivation + MAC verification happen here;
    // a wrong password or a tampered container is rejected before any solve
    decrypt_image dimgs;
    try {
        dimgs = decrypt_image(encrypted_img_g, password);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
    cv::Size org_size = dimgs.get_org_size();

    if (CSencryption::params == AUTO_PARAM) {
        iterations = (1.0f / dimgs.get_compression_ratio()) * 10;

        if (iterations > 30) {
            iterations = 30;
        }

        if (dimgs.get_compression_ratio() < 0.5f) {
            coef = 0.045f;
        }
        else
        {
            coef = (1.0f / dimgs.get_compression_ratio()) * 0.01875f;
        }
        num_tiles = 24;
        overlap = 24;
        nun_threads = omp_get_max_threads();
    }

    //int N_reconfigured = tiles;
    std::string windowName = output_path.empty() ? "decrypted tiles" : output_path;
    try {
        std::vector<std::vector<cv::Mat>> encrypted_image_tiles;
        std::vector<std::vector<cv::Mat>> decrypted_image_tiles;
        decrypted_image_tiles.resize(num_tiles, std::vector<cv::Mat>(num_tiles));
        std::vector<std::vector<indices>> indices_reconfigured(num_tiles, std::vector<indices>(num_tiles));
        std::vector<std::vector<TileCoord>> coordinates;
        cv::Mat sampled_mat;
        cv::Mat masked_mat;

        // extracting all the sampled pixels (sampled_mat) and thier coordinates (masked_mat)
        dimgs.get_sampled_mat(sampled_mat, masked_mat);

        // we split the sampled image into tiles
        // every tile overlaps with other neighboring tiles  
        // this is done because otherwise the titles wont quite match with eachother along the borders
        // this becomes more obvious as the numer of samples goes down, aka more compression 
        splitImageIntoTiles(sampled_mat, encrypted_image_tiles, coordinates, num_tiles, overlap);

        // we will decrypt the tiles in a spiral order from the middle
        // this is done just beacuse it looks "better" this way
        const std::vector<std::string> processing_order = spiralOrder(num_tiles);

        cv::Size tile_size;
        int estimated_number_of_samples = 0;

        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {

                std::vector<int> ri_x_g, ri_y_g;

                // reserving space to avoid realocations
                ri_x_g.reserve(estimated_number_of_samples);
                ri_y_g.reserve(estimated_number_of_samples);

                int base_row = i * (encrypted_image_tiles[i][j].rows - overlap);
                int base_col = j * (encrypted_image_tiles[i][j].cols - overlap);

                for (int q = 0; q < encrypted_image_tiles[i][j].rows; q++) {
                    for (int k = 0; k < encrypted_image_tiles[i][j].cols; k++) {
                        if (masked_mat.at<cv::Vec3b>(base_row + q, base_col + k) == cv::Vec3b(1, 1, 1)) {
                            ri_x_g.push_back(q);
                            ri_y_g.push_back(k);
                        }
                    }
                }

                // we use the the number of previous samples as estimations, 
                // all tiles are going to have roughly the same number of samples
                estimated_number_of_samples = ri_x_g.size();

                decrypted_image_tiles[i][j] = cv::Mat::zeros(encrypted_image_tiles[i][j].rows, encrypted_image_tiles[i][j].cols, CV_8UC3);
                encrypt_image img(encrypted_image_tiles[i][j], false);

                indices_reconfigured[i][j] = { ri_x_g, ri_y_g };
                img.encrypt(ri_x_g, ri_y_g);
                encrypted_image_tiles[i][j] = img.get_mat();
                tile_size = img.get_size();
            }
        }

        cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);

    // this updates and displays the image as it is being decrypted
    // (opt-in: --no-preview on the CLI, and always skipped in headless tests)
    display disp;
    if (show_preview) {
        disp.display_image(windowName, reconstructed, coordinates, decrypted_image_tiles);
    }

    std::thread decrypt_tiles_thread;
    decrypt_tiles_thread = std::thread(decrypt_tiles, nun_threads, std::ref(encrypted_image_tiles), std::ref(indices_reconfigured),
        std::ref(decrypted_image_tiles), std::ref(processing_order), iterations, tile_size, coef);
    decrypt_tiles_thread.join();

    if (show_preview) {
        disp.stop_display();
    }

        reconstructed = reconstructImage(decrypted_image_tiles, coordinates);

        // blending the overlapping tiles together for better quality
        reconstructed = blendTilesWithImage(decrypted_image_tiles, coordinates, reconstructed, 0.5f);
        cv::resize(reconstructed, reconstructed, org_size);

        // sharpening the image to bring out more detail
        cv::Mat sharpened;
        float sharpness = 0.3f;
        sharpenImage(reconstructed, sharpened, sharpness);

        // upscaling the image
        cv::Mat encrypted_img_upscaled = cv::Mat::zeros(sharpened.rows * 2, sharpened.cols * 2, CV_8UC3);
        typedef avir::fpclass_def< float, float,
            avir::CImageResizerDithererErrdINL< float > > fpclass_dith;
        avir::CImageResizer< fpclass_dith > ImageResizer(8);
        ImageResizer.resizeImage(sharpened.data, sharpened.cols, sharpened.rows, 0, encrypted_img_upscaled.data, sharpened.cols * 2, sharpened.rows * 2, 3, 0);
        
        cv::imwrite(output_path, encrypted_img_upscaled);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: decryption pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    return 0;
}

