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

void decrypt_image::decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, bool ycrcb, bool chroma_sub, float tv) {

    // Experiment notes (kept so the findings aren't re-litigated):
    // 1. Packed 3-channel solve: a merged single lbfgs over all 3 channels
    //    measured ~2.3x slower wall time (~7s vs ~3s) because channels must
    //    share convergence/step (30 iters x 3 planes instead of staggered
    //    30/20/20). If re-enabled prefer reconstruct_image_packed below.
    // 2. YCrCb solve: the first attempt (15.6 dB, looked hopeless) was later
    //    traced to a channel-ordering bug -- the manual BGR->YCrCb conversion
    //    packed (Cb, Cr, Y) while cvtColor/warm-strips use (Y, Cr, Cb), so the
    //    solver got a permuted measurement space. Re-tested correctly on top
    //    of wavefront warm-starts (chroma strips from neighbors, half chroma
    //    l1, no luma->chroma forwarding): 26.71 / 30.29 dB synthetic / photo,
    //    i.e. a tie with BGR (26.77 / 30.39) at strictly higher cost
    //    (O(m) re-encoding per tile + extra cvtColor). BGR stays the default;
    //    ycrcb=true keeps the path alive for future tuning experiments.
    // 3. Chroma subsampling (4:2:0-style: chroma solved at half resolution,
    //    see reconstruct_color_channel_subchroma): 25.18 / 28.73 dB
    //    synthetic/photo (-1.6/-1.7 dB vs full-res BGR) for only ~7% faster
    //    wall time on the 20MP photo and slightly slower on small images --
    //    the per-eval resize overhead eats the 4x unknown reduction. Off by
    //    default; a viable opt-in when speed matters more than color accuracy
    //    (e.g. previews). Revisit only after making evaluate_coarse
    //    allocation-free (the residual buffer is reallocated per lbfgs eval).
    int num_iterations_offset = (num_iterations / 2);

    if (ycrcb) {
        // YCrCb solve on top of wavefront warm-starts: luma carries most of
        // the structure while the chroma planes are much smoother, so the
        // chroma l1 penalty is softened (half coefficient). The stored
        // measurements are sampled BGR pixels; they are re-encoded to YCrCb
        // once here (O(m)) and the solved planes are merged back.
        cv::Mat ycc_measurements = encrypted_img.clone();
        const int measurement_pixels = std::min<int>((int)ycc_measurements.total(), m - 1 + CS_HEADER_PIXELS);
        for (int i = CS_HEADER_PIXELS; i < measurement_pixels; i++) {
            const cv::Vec3b bgr = ycc_measurements.at<cv::Vec3b>(i);
            // same coefficients as cv::BGR2YCrCb; packed (Y, Cr, Cb) to match
            // cv::COLOR_BGR2YCrCb / COLOR_YCrCb2BGR channel order
            const float y  = 0.299f * bgr[2] + 0.587f * bgr[1] + 0.114f * bgr[0];
            const float cr = (bgr[2] - y) * 0.713f + 128.0f;
            const float cb = (bgr[0] - y) * 0.564f + 128.0f;
            ycc_measurements.at<cv::Vec3b>(i) = cv::Vec3b(
                cv::saturate_cast<uchar>(y + 0.5f),
                cv::saturate_cast<uchar>(cr + 0.5f),
                cv::saturate_cast<uchar>(cb + 0.5f));
        }

        // Y gets full iterations and no in-tile forwarding: the chroma solves
        // warm-start from the neighbor strips' CHROMA content (x0[1]/x0[2]) and
        // from each other (Cr -> Cb), not from luma-like planes.
        const float chroma_coef = coef * 0.5f;
        if (chroma_sub) {
            // 4:2:0-style subsampling: chroma solved at half resolution --
            // 4x fewer unknowns per chroma plane; content is smooth so the
            // upsampled result is expected to be near-lossless. x0[1]/x0[2]
            // hold coarse chroma warm-starts from the neighbor strips.
            const int chroma_iters = num_iterations - num_iterations_offset;
            reconstruct_color_channel(ycc_measurements, 0, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[0], false, ref[1]);
            reconstruct_color_channel_subchroma(ycc_measurements, 1, chroma_coef, rows, cols, ri_x_g, ri_y_g, chroma_iters, ref[1]);
            reconstruct_color_channel_subchroma(ycc_measurements, 2, chroma_coef, rows, cols, ri_x_g, ri_y_g, chroma_iters, ref[2]);
        }
        else {
            reconstruct_color_channel(ycc_measurements, 0, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[0], false, ref[1], tv);
            reconstruct_color_channel(ycc_measurements, 1, chroma_coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[1], true, ref[2], tv);
            reconstruct_color_channel(ycc_measurements, 2, chroma_coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[2], false, ref[2], tv);
        }

        cv::merge(ref, 3, out);
        out.convertTo(out, CV_8UC3);
        cv::cvtColor(out, out, cv::COLOR_YCrCb2BGR);
        return;
    }

    // BGR per-channel solve (best measured configuration so far):
    // c1/c2 get half the iterations: later channels are warm-started from the previous
    // channel's solution, so they need far fewer iterations to converge; capping them
    // harder trims nearly-free solver time (~30% less solver work at same quality target)
    reconstruct_color_channel(encrypted_img, 0, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[0], true, ref[1], tv);
    reconstruct_color_channel(encrypted_img, 1, coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[1], true, ref[2], tv);
    reconstruct_color_channel(encrypted_img, 2, coef, rows, cols, ri_x_g, ri_y_g, num_iterations - num_iterations_offset, ref[2], false, ref[2], tv);
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

// Builds the initial solver solution for one tile from its already-solved
// west/north neighbors: their overlap strips contain actual reconstructed
// image content for this tile's border region, a much stronger prior than
// the generic reference. Pixel-domain copy first, then the same DCT / 10
// convention createRefSolutions uses, so reconstruct_color_channel accepts
// it unchanged.
// With ycrcb = true the strips (and the fallback refs) are produced in the
// YCrCb domain instead, to match a YCrCb solve. With chroma_sub the chroma
// warm-start planes are additionally downsampled to half resolution so they
// match the subsampled-chroma solver's unknown space.
static void build_neighbor_warm_start(cv::Mat refs[3],
    const std::vector<cv::Mat>& generic_refs,
    const std::vector<std::vector<cv::Mat>>& solved,
    const std::vector<std::vector<TileCoord>>& coordinates,
    int i, int j, bool ycrcb, bool chroma_sub)
{
    const cv::Rect t_rect(coordinates[i][j].x, coordinates[i][j].y, solved[i][j].cols, solved[i][j].rows);
    cv::Mat pix(t_rect.height, t_rect.width, CV_32FC3, cv::Scalar(0, 0, 0)); // pixel domain, [0,1]
    bool any_neighbor = false;

    const int west[2] = { i, j - 1 };
    const int north[2] = { i - 1, j };
    for (int n = 0; n < 2; ++n) {
        const int ni = west[0] + (north[0] - west[0]) * n; // (i, j-1) then (i-1, j)
        const int nj = west[1] + (north[1] - west[1]) * n;
        if (ni < 0 || nj < 0 || solved[ni][nj].empty()) continue;
        const cv::Rect n_rect(coordinates[ni][nj].x, coordinates[ni][nj].y, solved[ni][nj].cols, solved[ni][nj].rows);
        const cv::Rect inter = n_rect & t_rect;
        if (inter.width <= 0 || inter.height <= 0) continue;

        const cv::Rect dst_local(inter.x - t_rect.x, inter.y - t_rect.y, inter.width, inter.height);
        const cv::Rect src_local(inter.x - n_rect.x, inter.y - n_rect.y, inter.width, inter.height);
        cv::Mat strip = solved[ni][nj](src_local).clone();
        if (ycrcb) {
            // the solved tiles hold BGR pixels; re-encode to YCrCb so the
            // warm start lives in the same domain as the solve
            cv::cvtColor(strip, strip, cv::COLOR_BGR2YCrCb);
        }
        cv::Mat strip32;
        strip.convertTo(strip32, CV_32FC3, 1.0 / 255.0);
        strip32.copyTo(pix(dst_local));
        any_neighbor = true;
    }

    if (!any_neighbor) {
        // first wave (or failed neighbors): generic reference as before
        if (ycrcb) {
            // Y gets the generic image-like ref; chroma planes start from a
            // neutral flat-chroma solution (their statistics are very different)
            refs[0] = generic_refs[0].clone();
            cv::Size chroma_size = chroma_sub
                ? cv::Size((generic_refs[0].cols + 1) / 2, (generic_refs[0].rows + 1) / 2)
                : generic_refs[0].size();
            refs[1] = cv::Mat(chroma_size, CV_32F, cv::Scalar(0.5f));
            cv::dct(refs[1], refs[1], 0);
            refs[1] /= 10.0f;
            refs[2] = refs[1].clone();
        }
        else {
            for (int ch = 0; ch < 3; ++ch) {
                refs[ch] = generic_refs[ch].clone();
            }
        }
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(pix, planes);
    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat plane = planes[ch];
        if (chroma_sub && ch > 0) {
            // match the subsampled-chroma solver's unknown space
            cv::resize(plane, plane, cv::Size((plane.cols + 1) / 2, (plane.rows + 1) / 2), 0, 0, cv::INTER_AREA);
        }
        cv::dct(plane, plane, 0);
        plane /= 10.0f;
        refs[ch] = plane;
    }
}

// decrypts tiles in wavefront (anti-diagonal) order
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, bool ycrcb, bool chroma_sub, float tv) {

    // we use a reference image as the initial solution for tiles without
    // neighbors (first wave); this helps speed up convergence
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);

    // wavefront over the tile grid: wave w = i + j. Tiles on the same
    // anti-diagonal share no west or north neighbors, so every tile of a
    // wave runs in parallel while warm-starting from tiles of earlier
    // waves only -- no ordering races, and roughly 2N-1 waves instead of
    // num_tiles^2 independent solves.
    for (int w = 0; w <= 2 * (num_tiles - 1); ++w) {
        const int i_lo = (std::max)(0, w - (num_tiles - 1));
        const int i_hi = (std::min)(num_tiles - 1, w);
        const int wave_count = i_hi - i_lo + 1;

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int t = 0; t < wave_count; ++t) {
            try {
                const int i = i_lo + t;
                const int j = w - i;

                cv::Mat x0[3];
                build_neighbor_warm_start(x0, ref, mats_out, coordinates, i, j, ycrcb, chroma_sub);

                decrypt_image dimgs = decrypt_image(mats_in[i][j]);
                dimgs.decrypt(x0, indices[i][j].ri_x_g, indices[i][j].ri_y_g, iterations, coef, mats_out[i][j], ycrcb, chroma_sub, tv);
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
    bool show_preview,
    bool denoise,
    float tv
) {
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    return decrypt_image_tiled(encrypted_img_g, output_path, password, num_tiles, overlap, iterations, nun_threads, coef, show_preview, denoise, tv);
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
    bool show_preview,
    bool denoise,
    float tv
) {
    const cv::Mat& encrypted_img_g = encrypted_img_in;
    try {
        if (CSencryption::params == MANUAL_PARAM) {
            if (overlap < 24 || overlap > 96) {
                throw std::runtime_error("Overlap is outside the acceptable range of (24, 96)");
            }

            if (num_tiles < 8) {
                throw std::runtime_error("Number of tiles is less than 8");
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
        // iterations: the solver saturates by ~5 steps across compression
        // ratios 0.25..1.0 (tests/bench_iterations.cpp: the PSNR plateau
        // starts at 5 everywhere, time scales linearly beyond it), so the
        // old 10*(1/ratio) formula (capped 30) overshot by 2-6x. Keep a
        // knee-proportional allocation with a cap of 8.
        iterations = (int)std::ceil(5.0f / dimgs.get_compression_ratio());
        if (iterations > 8) {
            iterations = 8;
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

        cv::Size tile_size;
        int estimated_number_of_samples = 0;

        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {

                std::vector<int> ri_x_g, ri_y_g;

                // reserving space to avoid realocations
                ri_x_g.reserve(estimated_number_of_samples);
                ri_y_g.reserve(estimated_number_of_samples);

                // tile origin as produced by splitImageIntoTiles: the last
                // row/column tiles are anchored to the image edge there, so
                // the origin cannot be re-derived arithmetically here
                const int base_row = coordinates[i][j].y;
                const int base_col = coordinates[i][j].x;

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
    // chroma_sub = true runs the subsampled-chroma solve (requires ycrcb);
    // measured (wavefront, half chroma l1): 25.18 / 28.73 dB synthetic/photo
    // vs 26.77 / 30.39 dB full-res BGR, for only ~7% faster wall time on the
    // 20MP photo (3.8s vs 4.1s) and slightly slower on small images -- the
    // resize-heavy coarse operators eat the 4x unknown reduction. Off by
    // default; kept as an opt-in speed/quality trade via the flags.
    decrypt_tiles_thread = std::thread(decrypt_tiles, nun_threads, std::ref(encrypted_image_tiles), std::ref(indices_reconfigured),
        std::ref(decrypted_image_tiles), std::cref(coordinates), num_tiles, overlap, iterations, tile_size, coef, /*ycrcb*/ false, /*chroma_sub*/ false, tv);
    decrypt_tiles_thread.join();

    if (show_preview) {
        disp.stop_display();
    }

        reconstructed = reconstructImage(decrypted_image_tiles, coordinates);

        // cosine-feathered compositing across the overlapping tile borders:
        // ramp width == overlap means non-overlapped interiors keep full
        // weight and the overlap zones sum to a smooth transition (fewer
        // visible seams than the legacy 0.5 alpha blend)
        reconstructed = blendTilesWithImage(decrypted_image_tiles, coordinates, reconstructed, 0.5f, overlap);
        cv::resize(reconstructed, reconstructed, org_size);

        // optional final denoise (cv::photo): smooths solver noise but blurs
        // fine detail, so it is opt-in
        if (denoise) {
            cv::fastNlMeansDenoisingColored(reconstructed, reconstructed, 3.0f, 3.0f, 7, 21);
        }

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


