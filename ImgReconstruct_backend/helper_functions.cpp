#include "helper_functions.hpp"

int nextClosestDivisible(const int& x, const int& y) {
    // Ensure y is not zero to avoid division by zero error
    if (y == 0) {
        throw std::invalid_argument("y must not be zero");
    }

    // Find the next multiple of y greater than x
    int nextMultiple = ((x + y - 1) / y) * y;

    return nextMultiple;
}

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates) {
    if (tiles.empty() || coordinates.empty() ||
        tiles.size() != coordinates.size() ||
        tiles[0].size() != coordinates[0].size()) {
        return cv::Mat();
    }

    int tileCountN = tiles.size();

    // Calculate output image size from the covered extents of every tile;
    // empty/skipped slots (default {0,0} + 0-size) must not drag maxX/maxY
    // down or invent a zero-size canvas
    int maxX = 0, maxY = 0;
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            if (tiles[i][j].empty()) continue;
            int rightEdge = coordinates[i][j].x + tiles[i][j].cols;
            int bottomEdge = coordinates[i][j].y + tiles[i][j].rows;
            maxX = max(maxX, rightEdge);
            maxY = max(maxY, bottomEdge);
        }
    }
    if (maxX <= 0 || maxY <= 0) {
        return cv::Mat();
    }

    // Create output image
    cv::Mat output(maxY, maxX, tiles[0][0].empty() ? CV_8UC3 : tiles[0][0].type(), cv::Scalar(0));

    // Copy tiles to their original positions
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            if (!tiles[i][j].empty()) {
                cv::Rect roi(coordinates[i][j].x,
                    coordinates[i][j].y,
                    tiles[i][j].cols,
                    tiles[i][j].rows);
                tiles[i][j].copyTo(output(roi));
            }
        }
    }

    return output;
}

std::vector<cv::Mat> splitMat(cv::Mat& image, int M, int N)
{
    int width = image.cols / M;
    int height = image.rows / N;
    int width_last_column = width + (image.cols % width);
    int height_last_row = height + (image.rows % height);

    std::vector<cv::Mat> result;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            cv::Rect roi(width * j,
                height * i,
                (j == (M - 1)) ? width_last_column : width,
                (i == (N - 1)) ? height_last_row : height);

            result.push_back(image(roi));
        }
    }

    return result;
}

inline void updateAxb2AndComputeFx(float* x_copy, const int* ri_x, const int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n) {
    __m256 fx_vec = _mm256_setzero_ps();  // Accumulator for fx

    int i = 0;
    for (; i <= n - 8; i += 8) {
        // Gather indices (aka coordinates of sampled pixels)
        int idx[8];
        for (int k = 0; k < 8; k++) {
            idx[k] = ri_x[i + k] * cols + ri_y[i + k];
        }

        // Load x_copy values using gather
        __m256 x_val = _mm256_i32gather_ps(x_copy, _mm256_load_si256((__m256i*) & idx[0]), 4);

        // Load b values (measurment aka sampled tile)
        __m256 b_val = _mm256_load_ps(&b[i]);

        // Compute differences
        __m256 diff = _mm256_sub_ps(x_val, b_val);

        // Accumulate fx (diff * diff)
        fx_vec = _mm256_fmadd_ps(diff, diff, fx_vec);

        // Store differences to Axb2_vec
        alignas(32) float temp[8];
        _mm256_store_ps(temp, diff);
        for (int k = 0; k < 8; k++) {
            Axb2_vec[idx[k]] = temp[k];
        }
    }

    // Handle remaining elements
    float fx_temp = 0.0f;
    for (; i < n; ++i) {
        int idx = ri_x[i] * cols + ri_y[i];
        float diff = x_copy[idx] - b[i];
        fx_temp += diff * diff;
        Axb2_vec[idx] = diff;
    }

    // Reduce fx_vec to scalar
    __m128 hi = _mm256_extractf128_ps(fx_vec, 1);
    __m128 lo = _mm256_castps256_ps128(fx_vec);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    fx = _mm_cvtss_f32(sum) + fx_temp;
}

inline void eval_g(float* Axb2, float* g, int n) {
    __m256 scalar = _mm256_set1_ps(2.0f); // Set scalar to 2.0f
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&Axb2[i]); 
        _mm256_store_ps(&g[i], _mm256_mul_ps(vecData, scalar));  // Multiply and store
    }

    // Process remaining elements
    for (; i < n; ++i) {
        g[i] = Axb2[i] * 2.0f;
    }
}

inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n) {
    __m256 factor = _mm256_set1_ps(0.0f);
    int i = 0;
    // Process multiples of 8
    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&x_copy[i], vecData);  // Copy to x_copy
        _mm256_store_ps(&Axb2_vec[i], factor); // Set Axb2_vec to 0
    }

    // Process remaining elements
    for (; i < n; ++i) {
        x_copy[i] = x[i];
        Axb2_vec[i] = 0.0f;
    }
}


// here we are basically evaluating the objective function
// as well as evaluating the error
// looks very unreadable because I tried to optimize it as much as possible
// DCTs are the limiting performance factor
float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
)
{
    float fx = 0;
    copy_x(data.x_copy, (float*)x, data.Axb2, n);
    cv::Mat Ax(data.rows, data.cols, CV_32F, data.x_copy);
    dct(Ax, Ax, cv::DCT_INVERSE);
    updateAxb2AndComputeFx(data.x_copy, data.ri_x, data.ri_y, data.Axb2, data.b, data.cols, fx, data.m);
    cv::Mat Axb2(data.rows, data.cols, CV_32F, data.Axb2);
    dct(Axb2, Axb2);
    eval_g(data.Axb2, g, n);

    // Optional total-variation fusion (smoothed isotropic TV on the
    // pixel-domain plane, which data.x_copy holds at this point):
    //   fx += tv_lambda * sum(sqrt(dx^2 + dy^2 + eps^2) - eps)
    //   gradient back-projected through the DCT and added to g.
    // The gradient is accumulated into the Axb2 buffer, which is free after
    // eval_g consumed its DCT of the data residual.
    if (data.tv_lambda > 0.0f) {
        const int rows = data.rows, cols = data.cols;
        const float eps = 1e-3f; // normalized-domain smoothing (x in [0,1])
        float* xpix = data.x_copy;
        float* tvgrad = data.Axb2;
        float phi = 0.0f;
        std::memset(tvgrad, 0, sizeof(float) * n);
        for (int i = 0; i < rows; ++i) {
            const bool has_down = i + 1 < rows;
            for (int j = 0; j < cols; ++j) {
                const int idx = i * cols + j;
                const float a = has_down ? xpix[idx + cols] - xpix[idx] : 0.0f;
                const float b = (j + 1 < cols) ? xpix[idx + 1] - xpix[idx] : 0.0f;
                const float d = std::sqrt(a * a + b * b + eps * eps);
                phi += d - eps;
                const float ua = a / d;
                const float vb = b / d;
                // phi = sum(sqrt(a^2+b^2)); dphi/dx[p,q] = -div(u, v)
                tvgrad[idx] -= ua + vb;
                if (has_down) {
                    tvgrad[idx + cols] += ua;
                }
                if (j + 1 < cols) {
                    tvgrad[idx + 1] += vb;
                }
            }
        }
        cv::Mat tvgrad_m(data.rows, data.cols, CV_32F, tvgrad);
        dct(tvgrad_m, tvgrad_m);
        for (int i = 0; i < n; ++i) {
            g[i] += data.tv_lambda * tvgrad[i];
        }
        fx += data.tv_lambda * phi;
    }

    return fx;
}

// prints out convergence metrics with every iterations
// this is more for debugging purposes, it's not necesarry to be called
int progress(
    void* instance,
    const float* x,
    const float* g,
    const float fx,
    const float xnorm,
    const float gnorm,
    const float step,
    int n,
    int k,
    int ls
)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");

    return 0;
}

// this function creates initial solutions for each color channel
// we use a generic reference image to create them
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols) {
    cv::Mat ref = cv::imread("ref.png", cv::IMREAD_COLOR);

    // the warm-start asset is optional; fall back to neutral gray if missing
    // instead of crashing inside cv::resize on an empty Mat
    if (ref.empty()) {
        ref = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128));
    }
    else {
        // resizing to accomodate the size of the tiles
        cv::resize(ref, ref, cv::Size(rows, cols));
    }

    std::vector<cv::Mat> c;
    cv::split(ref, c);

    for (int i = 0; i < 3; i++) {
        c[i].convertTo(c[i], CV_32F);
        c[i] = c[i] / 255.0f;
        cv::dct(c[i], c[i], 0);
        c[i] = c[i] / 10.0f;
    }

    return c;
}

// reconstructs a color channel using LBFGS
void reconstruct_color_channel(const cv::Mat& pixel_measurements, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref, float tv) {

    int n = rows * cols; // size of solution (size of vectorized image)
    float fx;
    /* Initialize the parameters for the optimization. */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // this tells lbfgs to do OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;
    int lbfgs_ret;
    std::vector<float> b;

    // reserving space to avoid realocations
    b.reserve(ri_x.size());

    //auto update_progress = progress;
    lbfgs_progress_t update_progress = NULL;
    eval_data data;
    std::vector<float> Axb2(n);
    std::vector<float> x_copy(n);

    // extracting pixel measurements from encrypted image
    for (int i = CS_HEADER_PIXELS; i < ri_x.size() + CS_HEADER_PIXELS && i < pixel_measurements.total(); i++) {
        b.push_back(pixel_measurements.at<cv::Vec3b>(i)[k] / 255.0f);
    }

    // sometimes the number of sampled pixels in a tile wont be exactly ri_x.size()
    // so we just make the rest of them 0
    for (int i = b.size(); i < ri_x.size(); i++) {
        b.push_back(0.0f);
    }

    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.m = ri_x.size();
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = rows;
    data.cols = cols;
    data.tv_lambda = tv;

    // LBFGS optimization
    lbfgs_ret = lbfgs(n, (float*)ref.data, data, &fx, evaluate, update_progress, NULL, &param);

    // we are copying the current solution to the next solution for faster convergence
    if (copy_next_ref) {
        int i;
        for (i = 0; i <= next_ref.total() - 8; i += 8) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&next_ref.data[i]), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ref.data[i])));
        }

        for (; i < next_ref.total(); i++) {
            next_ref.data[i] = ref.data[i];
        }
    }

    cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref.data);
    dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
    AtAxb2 = AtAxb2 * 255.0f;
}

float evaluate_coarse(void* instance, const float* x, eval_data data, float* g, const int n, const float step)
{
    float fx = 0;

    // x: coarse-plane DCT coefficients -> pixel plane (coarse)
    copy_x(data.x_copy, (float*)x, data.Axb2, n);
    cv::Mat C(data.rows, data.cols, CV_32F, data.x_copy);
    cv::dct(C, C, cv::DCT_INVERSE);

    // forward operator: upsample the coarse chroma plane to the full tile
    // grid and gather the scattered full-res measurements there
    cv::Mat full;
    cv::resize(C, full, cv::Size(data.full_cols, data.full_rows), 0, 0, cv::INTER_LINEAR);

    const int mm = data.m;
    cv::Mat R(data.full_rows, data.full_cols, CV_32F, cv::Scalar(0));
    float* rp = (float*)R.data;
    for (int k = 0; k < mm; ++k) {
        const int r = data.ri_x[k], c = data.ri_y[k];
        const float diff = full.at<float>(r, c) - data.b[k];
        R.at<float>(r, c) = diff;
        fx += diff * diff;
    }

    // gradient: 2x2 area-average the full-res residual back to the coarse
    // grid, then DCT it and scale
    cv::resize(R, C, cv::Size(data.cols, data.rows), 0, 0, cv::INTER_AREA);
    cv::dct(C, C, 0);
    eval_g((float*)C.data, g, n);

    return fx;
}

void reconstruct_color_channel_subchroma(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref)
{
    const int crows = (rows + 1) / 2, ccols = (cols + 1) / 2;
    const int nC = crows * ccols;
    float fx;

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;

    // measurements for this channel at the scattered full-res positions
    std::vector<float> b;
    b.reserve(ri_x.size());
    for (int i = CS_HEADER_PIXELS; i < (int)ri_x.size() + CS_HEADER_PIXELS && i < pixel_measurements.total(); i++) {
        b.push_back(pixel_measurements.at<cv::Vec3b>(i)[channel] / 255.0f);
    }
    for (int i = (int)b.size(); i < (int)ri_x.size(); i++) {
        b.push_back(0.0f);
    }

    eval_data data;
    std::vector<float> Axb2(nC), x_copy(nC);
    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.m = (int)ri_x.size();
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = crows;      // coarse unknown dims
    data.cols = ccols;
    data.full_rows = rows;  // full-res measurement grid
    data.full_cols = cols;

    lbfgs(nC, (float*)ref.data, data, &fx, evaluate_coarse, NULL, NULL, &param);

    // solved coarse DCT plane -> pixel plane -> upsample to the full tile size
    cv::Mat C(crows, ccols, CV_32F, ref.data);
    cv::dct(C, C, cv::DCT_INVERSE);
    C *= 255.0f;
    cv::resize(C, ref, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Patch-dictionary (K-SVD) machinery
// ---------------------------------------------------------------------------

static float cs_dict_lipschitz(const cs_dictionary& d);

void cs_omp_encode(const float* patch_values, const cs_dictionary& dict, int target_sparsity, float* coefficients_out) {
    const int P = dict.patch * dict.patch;
    const int A = dict.atoms;
    std::memset(coefficients_out, 0, sizeof(float) * A);

    float residual[64 * 64];
    const int Pcap = P <= 64 * 64 ? P : 64 * 64;
    std::memcpy(residual, patch_values, sizeof(float) * Pcap);
    bool used[4096] = {};
    int support[64];
    int support_len = 0;

    for (int s = 0; s < target_sparsity && support_len < A; ++s) {
        int best = -1;
        float best_corr = 0.0f;
        for (int a = 0; a < A; ++a) {
            if (used[a]) continue;
            const float* atom = &dict.D[(size_t)a * P];
            float corr = 0.0f;
            for (int p = 0; p < P; ++p) {
                corr += atom[p] * residual[p];
            }
            const float mag = std::fabs(corr);
            if (best < 0 || mag > best_corr) {
                best_corr = mag;
                best = a;
            }
        }
        if (best < 0) break;
        used[best] = true;
        support[support_len++] = best;

        // least-squares refit of the coefficients on the selected support:
        // normal equations G = Ds^T Ds, rhs = Ds^T residual (small system,
        // solved by Gaussian elimination -- no per-patch LAPACK calls)
        float G[16][16] = {};
        float rhs[16] = {};
        for (int r = 0; r < support_len; ++r) {
            const float* atom_r = &dict.D[(size_t)support[r] * P];
            for (int s = 0; s <= r; ++s) {
                const float* atom_s = &dict.D[(size_t)support[s] * P];
                float dot = 0.0f;
                for (int p = 0; p < P; ++p) dot += atom_r[p] * atom_s[p];
                G[r][s] = dot;
                G[s][r] = dot;
            }
            float dot = 0.0f;
            for (int p = 0; p < P; ++p) dot += atom_r[p] * residual[p];
            rhs[r] = dot;
        }
        float csol[16] = {};
        for (int col = 0; col < support_len; ++col) {
            int piv = col;
            for (int row = col + 1; row < support_len; ++row) {
                if (std::fabs(G[row][col]) > std::fabs(G[piv][col])) piv = row;
            }
            if (std::fabs(G[piv][col]) < 1e-12f) break;
            if (piv != col) {
                // swap rows AND columns (the system is symmetric) and keep the
                // support permutation in sync, so csol[r] still corresponds to
                // support[r] after elimination
                for (int c2 = 0; c2 < support_len; ++c2) std::swap(G[piv][c2], G[col][c2]);
                std::swap(rhs[piv], rhs[col]);
                for (int row = 0; row < support_len; ++row) std::swap(G[row][piv], G[row][col]);
                std::swap(support[piv], support[col]);
            }
            for (int row = col + 1; row < support_len; ++row) {
                const float f = G[row][col] / G[col][col];
                for (int c2 = col; c2 < support_len; ++c2) G[row][c2] -= f * G[col][c2];
                rhs[row] -= f * rhs[col];
            }
        }
        for (int row = support_len - 1; row >= 0; --row) {
            float acc = rhs[row];
            for (int c2 = row + 1; c2 < support_len; ++c2) acc -= G[row][c2] * csol[c2];
            csol[row] = G[row][row] != 0.0f ? acc / G[row][row] : 0.0f;
        }
        for (int r = 0; r < support_len; ++r) {
            coefficients_out[support[r]] = csol[r];
        }

        // residual = patch - D_S * c_S
        for (int p = 0; p < P; ++p) {
            float syn = 0.0f;
            for (int r = 0; r < support_len; ++r) {
                syn += dict.D[(size_t)support[r] * P + p] * coefficients_out[support[r]];
            }
            residual[p] = patch_values[p] - syn;
        }
    }
}

bool cs_train_dictionary(const std::vector<cv::Mat>& images, int atoms, int ksvd_iters,
    int max_patches, cs_dictionary& out)
{
    const int PATCH = 8;
    const int P = PATCH * PATCH;
    if (images.empty() || atoms <= 0 || atoms > 4096) return false;

    // gather 8x8 patches (zero-meaned) sampled across the WHOLE image —
    // sampling only the first rows would make the training data homogeneous
    // and the dictionary degenerate
    std::vector<std::vector<std::pair<int, int>>> per_image(images.size());
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const auto& img = images[idx];
        if (img.empty()) continue;
        for (int y = 0; y + PATCH <= img.rows; y += 4) {
            for (int x = 0; x + PATCH <= img.cols; x += 4) {
                per_image[idx].push_back({ y, x });
            }
        }
    }
    long long total_slots = 0;
    for (const auto& pv : per_image) total_slots += (long long)pv.size() * 3;
    const long long cap = max_patches;
    const long long step_slots = total_slots > cap ? total_slots / cap + 1 : 1;

    std::vector<float> Y;
    long long gathered = 0, slot = 0;
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const auto& img = images[idx];
        for (const auto& pt : per_image[idx]) {
            for (int ch = 0; ch < 3; ++ch, ++slot) {
                if (gathered >= cap) break;
                if (slot % step_slots != 0) continue;
                float sum = 0.0f;
                const size_t base = Y.size();
                Y.resize(Y.size() + P);
                for (int r = 0; r < PATCH; ++r) {
                    for (int c = 0; c < PATCH; ++c) {
                        Y[base + (size_t)r * PATCH + c] = img.at<cv::Vec3b>(pt.first + r, pt.second + c)[ch] / 255.0f;
                        sum += Y[base + (size_t)r * PATCH + c];
                    }
                }
                // zero-mean the patch: the shared brightness component would
                // collapse every SVD update toward the mean patch
                const float mean = sum / P;
                for (int q = 0; q < P; ++q) Y[base + (size_t)q] -= mean;
                ++gathered;
            }
            if (gathered >= cap) break;
        }
        if (gathered >= cap) break;
    }
    const long long P_total = gathered;
    if (P_total < atoms) return false;

    // initialize atoms from deterministic slices of the training set, unit norm
    out.patch = PATCH;
    out.atoms = atoms;
    out.D.assign((size_t)atoms * P, 0.0f);
    for (int a = 0; a < atoms; ++a) {
        float* atom = &out.D[(size_t)a * P];
        const long long src = ((long long)a * P_total) / atoms;
        std::memcpy(atom, &Y[(size_t)src * P], sizeof(float) * P);
        float norm = 0.0f;
        for (int p = 0; p < P; ++p) norm += atom[p] * atom[p];
        norm = std::sqrt((std::max)(norm, 1e-12f));
        for (int p = 0; p < P; ++p) atom[p] /= norm;
    }

    const int K = 10; // target sparsity per patch
    cv::RNG rng(12345);
    std::vector<float> codes((size_t)atoms * P_total, 0.0f);

    for (int iter = 0; iter < ksvd_iters; ++iter) {
        // ---- sparse coding: OMP per training patch (parallel, thread-safe) ----
        std::fill(codes.begin(), codes.end(), 0.0f);
        #pragma omp parallel for schedule(dynamic)
        for (long long p = 0; p < P_total; ++p) {
            cs_omp_encode(&Y[(size_t)p * P], out, K, &codes[(size_t)p * atoms]);
        }
        {
            // training diagnostics: how many codes came out non-degenerate?
            long long nnz = 0;
            for (size_t i = 0; i < codes.size(); ++i) if (codes[i] != 0.0f) nnz++;
            std::printf("  ksvd iter %d: nonzero codes = %lld (of %lld)\n", iter, nnz, (long long)codes.size());
        }

        // ---- atom update: SVD refit per atom over its users ----
        for (int a = 0; a < atoms; ++a) {
            std::vector<std::pair<float, long long>> users; // |coef|, patch idx
            for (long long p = 0; p < P_total; ++p) {
                const float c = codes[(size_t)p * atoms + a];
                if (c != 0.0f) users.push_back({ std::fabs(c), p });
            }
            if (users.empty()) {
                // dead atom: reinitialize from a random training patch
                const int rp = rng.uniform(0, (int)P_total - 1);
                float* atom = &out.D[(size_t)a * P];
                std::memcpy(atom, &Y[(size_t)rp * P], sizeof(float) * P);
                float norm = 0.0f;
                for (int q = 0; q < P; ++q) norm += atom[q] * atom[q];
                norm = std::sqrt((std::max)(norm, 1e-12f));
                for (int q = 0; q < P; ++q) atom[q] /= norm;
                continue;
            }
            // cap the user set so the SVD stays cheap
            if (users.size() > 256) {
                std::nth_element(users.begin(), users.begin() + 255, users.end());
                users.resize(256);
            }
            const int nu = (int)users.size();

            // E (nu x P): row u = user's training patch minus every OTHER
            // atom's contribution; the best rank-1 fit's RIGHT singular
            // vector (vt row 0) becomes the new atom
            cv::Mat E(nu, P, CV_32F);
            for (int u = 0; u < nu; ++u) {
                std::memcpy(E.ptr<float>(u), &Y[(size_t)users[u].second * P], sizeof(float) * P);
                for (int a2 = 0; a2 < atoms; ++a2) {
                    if (a2 == a) continue;
                    const float c2 = codes[(size_t)users[u].second * atoms + a2];
                    if (c2 == 0.0f) continue;
                    const float* atom2 = &out.D[(size_t)a2 * P];
                    for (int q = 0; q < P; ++q) {
                        E.at<float>(u, q) -= atom2[q] * c2;
                    }
                }
            }
            cv::Mat w, uu, vt;
            cv::SVD::compute(E, w, uu, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
            float* atom = &out.D[(size_t)a * P];
            for (int q = 0; q < P; ++q) {
                atom[q] = vt.at<float>(0, q);   // right singular vector (P-dim)
            }
            float norm = 0.0f;
            for (int q = 0; q < P; ++q) norm += atom[q] * atom[q];
            norm = std::sqrt((std::max)(norm, 1e-12f));
            for (int q = 0; q < P; ++q) atom[q] /= norm;
            for (int u = 0; u < nu; ++u) {
                codes[(size_t)users[u].second * atoms + a] = w.at<float>(0) * uu.at<float>(u, 0);
            }
        }
        // diagnostic: atom distinctness after the update pass
        {
            double d01 = 0.0, d12 = 0.0;
            for (int p = 0; p < P; ++p) {
                const double x = out.D[(size_t)0 * P + p] - out.D[(size_t)1 * P + p];
                const double y = out.D[(size_t)1 * P + p] - out.D[(size_t)2 * P + p];
                d01 += x * x;
                d12 += y * y;
            }
            std::printf("  ksvd iter %d: |d0-d1|^2 = %.6f, |d1-d2|^2 = %.6f\n", iter, d01, d12);
        }
    }
    out.lipschitz = cs_dict_lipschitz(out);
    return true;
}

// lambda_max(D^T D) via power iteration (used as the FISTA step constant)
static float cs_dict_lipschitz(const cs_dictionary& d) {
    const int P = d.patch * d.patch;
    const int A = d.atoms;
    std::vector<float> v((size_t)A, 1.0f / std::sqrt((float)A));
    float lam = 1.0f;
    for (int it = 0; it < 24; ++it) {
        // syn = D v
        std::vector<float> syn((size_t)P, 0.0f);
        for (int a = 0; a < A; ++a) {
            const float ca = v[(size_t)a];
            if (ca == 0.0f) continue;
            const float* atom = &d.D[(size_t)a * P];
            for (int p = 0; p < P; ++p) syn[(size_t)p] += atom[p] * ca;
        }
        // w = D^T syn
        std::vector<float> w((size_t)A, 0.0f);
        for (int a = 0; a < A; ++a) {
            const float* atom = &d.D[(size_t)a * P];
            float dot = 0.0f;
            for (int p = 0; p < P; ++p) dot += atom[p] * syn[(size_t)p];
            w[(size_t)a] = dot;
        }
        float n2 = 0.0f;
        for (int a = 0; a < A; ++a) n2 += w[(size_t)a] * w[(size_t)a];
        lam = std::sqrt(n2);
        if (lam <= 1e-9f) return 1.0f;
        for (int a = 0; a < A; ++a) v[(size_t)a] = w[(size_t)a] / lam;
    }
    return lam;
}

bool cs_save_dictionary(const std::string& path, const cs_dictionary& d) {    std::ofstream f(path, std::ios::binary);
    if (!f.good()) return false;
    const char magic[4] = { 'C', 'S', 'D', '1' };
    f.write(magic, 4);
    f.write((const char*)&d.atoms, sizeof(int));
    f.write((const char*)&d.patch, sizeof(int));
    f.write((const char*)d.D.data(), (std::streamsize)(d.D.size() * sizeof(float)));
    return f.good();
}

bool cs_load_dictionary(const std::string& path, cs_dictionary& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "CSD1", 4) != 0) return false;
    f.read((char*)&d.atoms, sizeof(int));
    f.read((char*)&d.patch, sizeof(int));
    if (d.atoms <= 0 || d.atoms > 4096 || d.patch < 2 || d.patch > 64) return false;
    d.D.resize((size_t)d.atoms * d.patch * d.patch);
    f.read((char*)d.D.data(), (std::streamsize)(d.D.size() * sizeof(float)));
    if (!(f.good() && !d.D.empty())) return false;
    d.lipschitz = cs_dict_lipschitz(d);
    return true;
}

float evaluate_dict(void* instance, const float* x, eval_data data, float* g, const int n, const float step)
{
    const int patch = data.patch;
    const int P = patch * patch;
    const int atoms = data.dict_atoms;
    const int patchRows = (data.rows + patch - 1) / patch;
    const int patchCols = (data.cols + patch - 1) / patch;

    float fx = 0.0f;
    std::memset(g, 0, sizeof(float) * n);

    // synthesize the tile plane from the concatenated patch coefficients
    float* plane = data.x_copy;
    std::memset(plane, 0, sizeof(float) * (size_t)data.rows * data.cols);
    for (int pr = 0; pr < patchRows; ++pr) {
        for (int pc = 0; pc < patchCols; ++pc) {
            const float* c = x + ((size_t)pr * patchCols + pc) * atoms;
            for (int py = 0; py < patch; ++py) {
                for (int px = 0; px < patch; ++px) {
                    const int ty = pr * patch + py;
                    const int tx = pc * patch + px;
                    if (ty >= data.rows || tx >= data.cols) continue;
                    float syn = 0.0f;
                    const int pidx = py * patch + px;
                    const float* Dcol = data.dict_D + pidx; // strided access per atom
                    for (int a = 0; a < atoms; ++a) {
                        syn += Dcol[(size_t)a * P] * c[a];
                    }
                    plane[(size_t)ty * data.cols + tx] = syn;
                }
            }
        }
    }

    // residual at the measured positions
    float* resid = data.Axb2;
    std::memset(resid, 0, sizeof(float) * (size_t)data.rows * data.cols);
    for (int k = 0; k < data.m; ++k) {
        const int r = data.ri_x[k], c = data.ri_y[k];
        const float diff = plane[(size_t)r * data.cols + c] - data.b[k];
        resid[(size_t)r * data.cols + c] = diff;
        fx += diff * diff;
    }

    // gradient: per patch, g_p = 2 * D^T * resid_p
    for (int pr = 0; pr < patchRows; ++pr) {
        for (int pc = 0; pc < patchCols; ++pc) {
            float* gpatch = g + ((size_t)pr * patchCols + pc) * atoms;
            for (int py = 0; py < patch; ++py) {
                const int ty = pr * patch + py;
                for (int px = 0; px < patch; ++px) {
                    const int tx = pc * patch + px;
                    if (ty >= data.rows || tx >= data.cols) continue;
                    const float rv = resid[(size_t)ty * data.cols + tx];
                    if (rv == 0.0f) continue;
                    const int pidx = py * patch + px;
                    for (int a = 0; a < atoms; ++a) {
                        gpatch[a] += 2.0f * data.dict_D[(size_t)a * P + pidx] * rv;
                    }
                }
            }
        }
    }

    return fx;
}

/** @brief orthogonal matching pursuit against a SUB-SAMPLED observation of one
patch: only the measured positions (their dictionary rows) are observable.
Selects up to target_sparsity atoms by correlation magnitude on the measured
rows, refits coefficients by normal equations on the same rows, and finally
synthesizes the FULL patch from the sparse code. */
void cs_omp_encode_measured(const int* pos, const float* vals, int count,
    const cs_dictionary& dict, int target_sparsity, float* coefficients_out, float* patch_out)
{
    const int P = dict.patch * dict.patch;
    const int A = dict.atoms;
    std::memset(coefficients_out, 0, sizeof(float) * A);

    const int M = count < P ? count : P;
    float resid[64 * 64];
    std::memcpy(resid, vals, sizeof(float) * count);
    bool used[4096] = {};
    int support[64];
    int support_len = 0;
    if (target_sparsity > 64) target_sparsity = 64;
    if (target_sparsity > count) target_sparsity = count;

    for (int s = 0; s < target_sparsity; ++s) {
        int best = -1;
        float best_corr = 0.0f;
        for (int a = 0; a < A; ++a) {
            if (used[a]) continue;
            float corr = 0.0f;
            for (int m = 0; m < M; ++m) {
                corr += dict.D[(size_t)a * P + pos[m]] * resid[m];
            }
            const float mag = std::fabs(corr);
            if (best < 0 || mag > best_corr) {
                best_corr = mag;
                best = a;
            }
        }
        if (best < 0) break;
        used[best] = true;
        support[support_len++] = best;

        // normal equations on the MEASURED rows only
        float G[16][16] = {};
        float rhs[16] = {};
        for (int r = 0; r < support_len; ++r) {
            const float* atom_r = &dict.D[(size_t)support[r] * P];
            for (int s2 = 0; s2 <= r; ++s2) {
                const float* atom_s = &dict.D[(size_t)support[s2] * P];
                float dot = 0.0f;
                for (int m = 0; m < M; ++m) dot += atom_r[pos[m]] * atom_s[pos[m]];
                G[r][s2] = dot;
                G[s2][r] = dot;
            }
            float dot = 0.0f;
            for (int m = 0; m < M; ++m) dot += atom_r[pos[m]] * resid[m];
            rhs[r] = dot;
        }
        float csol[16] = {};
        for (int col = 0; col < support_len; ++col) {
            int piv = col;
            for (int row = col + 1; row < support_len; ++row) {
                if (std::fabs(G[row][col]) > std::fabs(G[piv][col])) piv = row;
            }
            if (std::fabs(G[piv][col]) < 1e-10f) break;
            if (piv != col) {
                // symmetric elimination: swap rows AND columns, keep support in sync
                for (int c2 = 0; c2 < support_len; ++c2) std::swap(G[piv][c2], G[col][c2]);
                std::swap(rhs[piv], rhs[col]);
                for (int row = 0; row < support_len; ++row) std::swap(G[row][piv], G[row][col]);
                std::swap(support[piv], support[col]);
            }
            for (int row = col + 1; row < support_len; ++row) {
                const float f = G[row][col] / G[col][col];
                for (int c2 = col; c2 < support_len; ++c2) G[row][c2] -= f * G[col][c2];
                rhs[row] -= f * rhs[col];
            }
        }
        for (int row = support_len - 1; row >= 0; --row) {
            float acc = rhs[row];
            for (int c2 = row + 1; c2 < support_len; ++c2) acc -= G[row][c2] * csol[c2];
            csol[row] = G[row][row] != 0.0f ? acc / G[row][row] : 0.0f;
        }
        for (int r = 0; r < support_len; ++r) {
            coefficients_out[support[r]] = csol[r];
        }

        // residual update over measured positions
        for (int m = 0; m < M; ++m) {
            float syn = 0.0f;
            for (int r = 0; r < support_len; ++r) {
                syn += dict.D[(size_t)support[r] * P + pos[m]] * csol[r];
            }
            resid[m] = vals[m] - syn;
        }
    }

    // full patch synthesis from the sparse code
    for (int p = 0; p < P; ++p) {
        float syn = 0.0f;
        const float* Dcol = dict.D.data() + p;
        for (int a = 0; a < A; ++a) {
            syn += Dcol[(size_t)a * P] * coefficients_out[a];
        }
        patch_out[p] = syn;
    }
}

void reconstruct_color_channel_dict(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, const cs_dictionary& dict, cv::Mat& ref_out)
{
    // Classic K-SVD compressed-sensing decoder: every 8x8 patch of the tile is
    // coded independently with OMP against its OWN measured subset of
    // positions (sparse codes are recoverable from far fewer measurements
    // than a dense DCT plane needs), then synthesized back onto the full
    // patch. Fully data-parallel per patch; no global iterative solve.
    (void)param_c;    // OMP is parameter-free (sparsity is fixed at training time)
    (void)iterations; // OMP runs a fixed number of selections

    const int patch = dict.patch;
    const int P = patch * patch;
    const int patchRows = (rows + patch - 1) / patch;
    const int patchCols = (cols + patch - 1) / patch;
    const int nPatches = patchRows * patchCols;
    const int m = (int)ri_x.size();

    std::vector<float> b((size_t)m, 0.0f);
    for (int i = CS_HEADER_PIXELS; i < m + CS_HEADER_PIXELS && i < pixel_measurements.total(); i++) {
        b[i - CS_HEADER_PIXELS] = pixel_measurements.at<cv::Vec3b>(i)[channel] / 255.0f;
    }

    // bucket measurement indices per patch
    std::vector<std::vector<int>> per_patch((size_t)nPatches);
    for (int k = 0; k < m; ++k) {
        const int pr = ri_x[k] / patch;
        const int pc = ri_y[k] / patch;
        per_patch[(size_t)pr * patchCols + pc].push_back(k);
    }

    if (std::getenv("CS_DICT_DEBUG")) {
        std::printf("[dict] tile %dx%d m=%d patches=%dx%d; b[0..3] = %.4f %.4f %.4f %.4f\n",
            rows, cols, m, patchRows, patchCols, b[0], b[1], b[2], b[3]);
        std::printf("[dict] ri[0..3] = (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
            ri_x[0], ri_y[0], ri_x[1], ri_y[1], ri_x[2], ri_y[2], ri_x[3], ri_y[3]);
    }

    ref_out.create(rows, cols, CV_32F);
    float* plane = (float*)ref_out.data;
    std::memset(plane, 0, sizeof(float) * (size_t)rows * cols);

    #pragma omp parallel for schedule(dynamic)
    for (int pi = 0; pi < nPatches; ++pi) {
        const auto& ks = per_patch[(size_t)pi];
        const int M = (int)ks.size();
        if (M < 4) continue; // not enough evidence for this patch

        int pos[64 * 64];
        float vals[64 * 64];
        const int Mcap = M < 64 * 64 ? M : 64 * 64;
        float mean = 0.0f;
        for (int i = 0; i < Mcap; ++i) {
            const int k = ks[i];
            pos[i] = (ri_x[k] % patch) * patch + (ri_y[k] % patch);
            vals[i] = b[k];
            mean += b[k];
        }
        mean /= (float)Mcap;
        for (int i = 0; i < Mcap; ++i) {
            vals[i] -= mean; // the dictionary models zero-mean patches
        }

        float coeffs[4096] = {};
        float patch_rec[64 * 64] = {};
        cs_omp_encode_measured(pos, vals, Mcap, dict, 10, coeffs, patch_rec);
        for (int p = 0; p < P; ++p) {
            patch_rec[p] += mean; // restore the brightness level
        }
        if (std::getenv("CS_DICT_DEBUG") && pi == 0) {
            double e = 0.0;
            for (int i = 0; i < Mcap; ++i) {
                const float syn = patch_rec[pos[i]];
                e += (syn - (vals[i] + mean)) * (syn - (vals[i] + mean));
            }
            std::printf("[dict] patch0: M=%d mean=%.4f vals[0..3]=%.4f %.4f %.4f %.4f rec[0]=%.4f residEnergy=%.4f\n",
                Mcap, mean, vals[0] + mean, vals[1] + mean, vals[2] + mean, patch_rec[0], e);
        }

        const int pr = pi / patchCols;
        const int pc = pi % patchCols;
        for (int py = 0; py < patch; ++py) {
            const int ty = pr * patch + py;
            if (ty >= rows) continue;
            for (int px = 0; px < patch; ++px) {
                const int tx = pc * patch + px;
                if (tx >= cols) continue;
                // scale to the 0..255 pixel domain the merge/convert stage
                // expects (the DCT path multiplies by 255 the same way)
                plane[(size_t)ty * cols + tx] = patch_rec[py * patch + px] * 255.0f;
            }
        }
    }
}

float evaluate_stacked(void* instance, const float* x, eval_data data, float* g, const int nTotal, const float step)
{    float fx = 0;
    const int planes = 3;
    const int n = nTotal / planes;

    // one fused copy: planes for c0/c1/c2 laid out back to back
    copy_x(data.x_copy, (float*)x, data.Axb2, nTotal);

    cv::Mat Ax;
    for (int pi = 0; pi < planes; ++pi) {
        float* xp = data.x_copy + (size_t)pi * n;
        float* ap = data.Axb2 + (size_t)pi * n;

        // gradient needs the pixel-domain residual: IDCT of current solution plane
        Ax = cv::Mat(data.rows, data.cols, CV_32F, xp);
        dct(Ax, Ax, cv::DCT_INVERSE);

        updateAxb2AndComputeFx(xp, data.ri_x, data.ri_y, ap, data.b + (size_t)pi * n, data.cols, fx, data.m);

        cv::Mat Axb2M(data.rows, data.cols, CV_32F, ap);
        dct(Axb2M, Axb2M);
        eval_g(ap, g + (size_t)pi * n, n);
    }

    return fx;
}

void reconstruct_image_packed(const cv::Mat& pixel_measurements, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat refs[3], cv::Mat& out)
{
    int n = rows * cols;
    const int m = (int)ri_x.size();
    const int nTotal = 3 * n;
    float fx;

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // OWL-QN, same coefficient for all three channels
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;

    // initial solution = warm-starting all three planes from the given reference solutions
    std::vector<float> x(nTotal);
    for (int pi = 0; pi < 3; ++pi) {
        std::memcpy(&x[(size_t)pi * n], refs[pi].data, (size_t)n * sizeof(float));
    }

    // one fused measurement-extraction pass over the byte-vectorized encrypted tile:
    // writes interleaved-per-plane b[c*m + i] instead of looping over the encrypted tile 3 times
    std::vector<float> b(3 * m, 0.0f);
    int cnt = 0;
    for (int i = CS_HEADER_PIXELS; i < m + CS_HEADER_PIXELS && i < pixel_measurements.total(); ++i, ++cnt) {
        const cv::Vec3b v = pixel_measurements.at<cv::Vec3b>(i);
        b[cnt] = v[0] / 255.0f;
        b[m + cnt] = v[1] / 255.0f;
        b[2 * m + cnt] = v[2] / 255.0f;
    }
    // remaining entries already zero-initialized (same fallback the per-channel version had)

    eval_data data;
    std::vector<float> Axb2(nTotal);
    std::vector<float> x_copy(nTotal);
    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.m = m;
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = rows;
    data.cols = cols;

    float _fx_unused;
    const int lbfgs_ret = lbfgs(nTotal, x.data(), data, &_fx_unused, evaluate_stacked, NULL, NULL, &param);
    (void)lbfgs_ret;

    // copy each solved plane back to its ref buffer and run the final IDCT + 255 scaling,
    // identical to the per-channel tail so callers can carry on with cv::merge as before
    for (int pi = 0; pi < 3; ++pi) {
        std::memcpy(refs[pi].data, &x[(size_t)pi * n], (size_t)n * sizeof(float));
        cv::Mat AtAxb2(rows, cols, CV_32F, refs[pi].data);
        dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
        AtAxb2 = AtAxb2 * 255.0f;
    }
}

std::vector<std::string> splitString(const std::string& str, const char& delimiter) {
    std::vector<std::string> result;
    std::string temp;
    for (char c : str) {
        if (c == delimiter) {
            if (!temp.empty()) {
                result.push_back(temp);
                temp.clear();
            }
        }
        else {
            temp.push_back(c);
        }
    }
    // Add the last substring if there is any
    if (!temp.empty()) {
        result.push_back(temp);
    }
    return result;
}

std::string removeCharacter(const std::string& str, const char& ch) {
    std::string result;
    for (char c : str) {
        if (c != ch) {
            result.push_back(c);
        }
    }
    return result;
}

void storeStringInColorMat(const std::string& text, cv::Mat& colorMat) {
    // Ensure the colorMat is large enough to hold the string
    int rows = (text.size() / 3) + 1;
    int cols = 1;
    colorMat = cv::Mat::zeros(rows, cols, CV_8UC3);

    // Encode the string into the Mat
    for (int i = 0; i < text.size(); ++i) {
        int row = i / 3;
        int channel = i % 3;
        colorMat.at<cv::Vec3b>(row, 0)[channel] = static_cast<uchar>(text[i]);
    }
}

std::string retrieveStringFromColorMat(const cv::Mat& colorMat) {
    std::string text;

    // Decode the Mat back into a string
    for (int i = 0; i < colorMat.rows; ++i) {
        for (int channel = 0; channel < 3; ++channel) {
            uchar value = colorMat.at<cv::Vec3b>(i, 0)[channel];
            if (value != 0) {
                text.push_back(static_cast<char>(value));
            }
        }
    }

    return text;
}


std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image, const int& tile_width, const int& tile_height, const int& rows, const int& cols) {
    std::vector<cv::Mat> tiles;

    // Iterate over each tile position and extract the tile from the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Rect roi(j * tile_width, i * tile_height, tile_width, tile_height);
            tiles.push_back(image(roi).clone());
        }
    }

    return tiles;
}

std::vector<std::string> spiralOrder(const int& tiles) {

    std::vector<std::vector<std::string>> matrix(tiles, std::vector<std::string>(tiles));

    int k = 0;
    for (int i = 0; i < tiles; i++) {
        for (int j = 0; j < tiles; j++) {
            matrix[i][j] = std::to_string(i) + "_" + std::to_string(j);
        }
    }

    std::vector<std::string> result;
    int m = matrix.size();
    if (m == 0) return result;
    int n = matrix[0].size();

    int startRow = m / 2, startCol = n / 2; // start from the middle
    int dir = 0; // 0 = up, 1 = left, 2 = down, 3 = right
    int steps = 1, stepCount = 0;

    int row = startRow, col = startCol;
    result.push_back(matrix[row][col]);

    while (result.size() < m * n) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < steps; ++j) {
                if (dir == 0) --row;
                else if (dir == 1) --col;
                else if (dir == 2) ++row;
                else ++col;

                if (row >= 0 && row < m && col >= 0 && col < n) {
                    result.push_back(matrix[row][col]);
                }
            }
            dir = (dir + 1) % 4;
        }
        ++steps;
    }

    return result;
}

void splitImageIntoTiles(const cv::Mat& inputImage,
    std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates,
    const int& tileCountN,
    const int& overlap) {
    // Input validation
    if (inputImage.empty() || tileCountN <= 0 || overlap < 0) {
        return;
    }

    int height = inputImage.rows;
    int width = inputImage.cols;

    // Calculate tile dimensions considering overlap. Ceiling (not flooring)
    // guarantees the natural stride reaches the image edge: with a floored
    // tile size the accumulated truncation could exceed the stride, leaving
    // uncovered strips between the last tiles (and the previous edge-anchor
    // workaround itself opened holes for tiny strides).
    int tileWidth = (int)std::ceil((width + (tileCountN - 1) * (double)overlap) / tileCountN);
    int tileHeight = (int)std::ceil((height + (tileCountN - 1) * (double)overlap) / tileCountN);

    // Resize vectors to N x N
    tiles.resize(tileCountN, std::vector<cv::Mat>(tileCountN));
    coordinates.resize(tileCountN, std::vector<TileCoord>(tileCountN));

    // Split image into tiles
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            // Calculate tile position
            int x = j * (tileWidth - overlap);
            int y = i * (tileHeight - overlap);

            // Clamp tiles that overrun the right/bottom edge so their outer
            // edge lands exactly on the image boundary (full coverage, no
            // gaps); tiny images where the tile exceeds the image are skipped
            if (x + tileWidth > width && width >= tileWidth) {
                x = width - tileWidth;
            }
            if (y + tileHeight > height && height >= tileHeight) {
                y = height - tileHeight;
            }
            if (x < 0) x = 0;
            if (y < 0) y = 0;

            // Adjust for edges
            int currentWidth = tileWidth;
            int currentHeight = tileHeight;

            if (x + tileWidth > width) {
                currentWidth = width - x;
            }
            if (y + tileHeight > height) {
                currentHeight = height - y;
            }

            // Ensure valid coordinates
            if (x < 0 || y < 0 || x >= width || y >= height) {
                continue;
            }

            // Extract tile
            cv::Rect roi(x, y, currentWidth, currentHeight);
            tiles[i][j] = inputImage(roi).clone();

            // Store coordinates
            coordinates[i][j] = { x, y };
        }
    }
}


cv::Mat blendTilesWithImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& targetImage,
    float alpha,
    int feather) {
    // Input validation
    if (tiles.empty() || coordinates.empty() ||
        tiles.size() != coordinates.size() ||
        tiles[0].size() != coordinates[0].size() ||
        targetImage.empty()) {
        return cv::Mat();
    }

    // Check if target image has valid dimensions
    int tileCountN = tiles.size();
    int maxX = targetImage.cols;
    int maxY = targetImage.rows;

    if (feather > 0) {
        // cosine-feathered compositing: every tile contributes a weighted sum,
        // the weight ramping from 1 in the tile interior to 0 at the borders
        // over `feather` pixels. Accumulate color*weight and weight, then
        // normalize; overlapped regions get exactly the sum of the tiles'
        // ramped contributions (order-independent, no visible seams even at
        // low overlap).
        cv::Mat acc(maxY, maxX, CV_32FC3, cv::Scalar(0, 0, 0));
        cv::Mat wsum(maxY, maxX, CV_32FC1, cv::Scalar(0));

        for (int i = 0; i < tileCountN; i++) {
            for (int j = 0; j < tileCountN; j++) {
                if (tiles[i][j].empty()) continue;
                const int th = tiles[i][j].rows, tw = tiles[i][j].cols;
                const int x = coordinates[i][j].x, y = coordinates[i][j].y;
                // clamp into the target instead of dropping the tile: an
                // OOB discard leaves a black gap when tile geometry is off
                int cx = x, cy = y, cw = tw, ch = th;
                if (cx < 0) { cw += cx; cx = 0; }
                if (cy < 0) { ch += cy; cy = 0; }
                if (cx + cw > maxX) cw = maxX - cx;
                if (cy + ch > maxY) ch = maxY - cy;
                if (cw <= 0 || ch <= 0) continue;
                const cv::Mat tile_roi = tiles[i][j](cv::Rect(0, 0, cw, ch));
                const int x0 = cx, y0 = cy;

                // per-tile weight map: 0.5*(1 - cos(pi * d / feather)) with d =
                // distance to the nearest tile border, clamped to [0, feather]
                cv::Mat w(ch, cw, CV_32FC1);
                for (int r = 0; r < ch; r++) {
                    const int dy = (std::min)(r, th - 1 - r);
                    for (int c = 0; c < cw; c++) {
                        const int dx = (std::min)(c, tw - 1 - c);
                        const int d = (std::min)(dx, dy);
                        const int dc = (std::min)(d, feather);
                        w.at<float>(r, c) = 0.5f * (1.0f - std::cos(CV_PI * dc / (float)feather));
                    }
                }

                cv::Mat tile_f;
                tile_roi.convertTo(tile_f, CV_32FC3);

                cv::Rect roi(x0, y0, cw, ch);
                cv::Mat acc_roi = acc(roi);
                cv::Mat wsum_roi = wsum(roi);

                std::vector<cv::Mat> w3 = { w, w, w };
                cv::Mat w3c;
                cv::merge(w3, w3c);
                cv::Mat contrib;
                cv::multiply(tile_f, w3c, contrib);
                cv::add(acc_roi, contrib, acc_roi);
                cv::add(wsum_roi, w, wsum_roi);
            }
        }

        // normalize by accumulated weight (guard fully-uncovered pixels)
        cv::Mat safe_w;
        (cv::max)(wsum, 1e-6f, safe_w);
        std::vector<cv::Mat> acc_planes;
        cv::split(acc, acc_planes);
        for (auto& plane : acc_planes) {
            cv::divide(plane, safe_w, plane);
        }
        cv::merge(acc_planes, acc);

        cv::Mat out;
        acc.convertTo(out, CV_8UC3);
        return out;
    }

    // Create a copy of the target image as base
    cv::Mat output = targetImage.clone();

    // Validate alpha value
    alpha = max(0.0f, min(1.0f, alpha));  // Clamp between 0 and 1

    // Blend each tile with the target image
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            if (!tiles[i][j].empty()) {
                // Get tile dimensions and position
                int tileWidth = tiles[i][j].cols;
                int tileHeight = tiles[i][j].rows;
                int x = coordinates[i][j].x;
                int y = coordinates[i][j].y;

                // clamp into the target instead of discarding the tile
                int cx = x, cy = y, cw = tileWidth, ch = tileHeight;
                if (cx < 0) { cw += cx; cx = 0; }
                if (cy < 0) { ch += cy; cy = 0; }
                if (cx + cw > maxX) cw = maxX - cx;
                if (cy + ch > maxY) ch = maxY - cy;
                if (cw <= 0 || ch <= 0) continue;
                const cv::Mat src = tiles[i][j](cv::Rect(0, 0, cw, ch));

                // Define ROI in output image
                cv::Rect roi(cx, cy, cw, ch);
                cv::Mat outputROI = output(roi);

                // Ensure compatible types
                if (src.type() != outputROI.type()) {
                    continue;
                }

                // Perform alpha blending
                // outputROI = alpha * tile + (1 - alpha) * outputROI
                addWeighted(src, alpha, outputROI, 1.0f - alpha, 0.0f, outputROI);
            }
        }
    }

    return output;
}

static unsigned long long fastmix64(unsigned long long s) {
    s += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void shuffle(std::vector<int>& data, unsigned seed) {
    if (data.size() < 2) return;
    unsigned long long state = (unsigned long long)seed * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
    for (size_t i = data.size() - 1; i > 0; --i) {
        size_t j = fastmix64(state) % (i + 1); // biased for i+1 > 2^64/2^32, negligible for shuffling pixel indices
        std::swap(data[i], data[j]);
    }
}

void reverseShuffle(std::vector<int>& data, unsigned seed) {
    if (data.size() < 2) return;
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    shuffle(indices, seed);

    // Use indices to reconstruct the original order
    std::vector<int> original(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        original[indices[i]] = data[i];
    }
    data = std::move(original);
}

void sharpenImage(const cv::Mat& input, cv::Mat& output, float sharpness) {

    output = input.clone();

    float kernel_data[] = {
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0
    };
    cv::Mat kernel(3, 3, CV_32F, kernel_data);
    cv::filter2D(input, output, -1, kernel);

    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(0, 0), 3);
    cv::addWeighted(input, 1.0 + sharpness, blurred, -sharpness, 0, output);
}


