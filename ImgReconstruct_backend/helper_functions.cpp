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

    // Calculate output image size
    int maxX = 0, maxY = 0;
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            int rightEdge = coordinates[i][j].x + tiles[i][j].cols;
            int bottomEdge = coordinates[i][j].y + tiles[i][j].rows;
            maxX = max(maxX, rightEdge);
            maxY = max(maxY, bottomEdge);
        }
    }

    // Create output image
    cv::Mat output(maxY, maxX, tiles[0][0].type(), cv::Scalar(0));

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
void reconstruct_color_channel(const cv::Mat& pixel_measurements, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref) {

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

    // Calculate tile dimensions considering overlap
    int tileWidth = (width + (tileCountN - 1) * overlap) / tileCountN;
    int tileHeight = (height + (tileCountN - 1) * overlap) / tileCountN;

    // Resize vectors to N x N
    tiles.resize(tileCountN, std::vector<cv::Mat>(tileCountN));
    coordinates.resize(tileCountN, std::vector<TileCoord>(tileCountN));

    // Split image into tiles
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            // Calculate tile position
            int x = j * (tileWidth - overlap);
            int y = i * (tileHeight - overlap);

            // Integer truncation of tileWidth can leave the last column short
            // of the right image edge (a uncovered black strip up to ~4% of
            // the image). Anchor the last tile to the edge so coverage is
            // complete for any width/height/overlap combination.
            if (j == tileCountN - 1 && width >= tileWidth) {
                x = width - tileWidth;
            }
            if (i == tileCountN - 1 && height >= tileHeight) {
                y = height - tileHeight;
            }

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
                if (x < 0 || y < 0 || x + tw > maxX || y + th > maxY) continue;

                // per-tile weight map: 0.5*(1 - cos(pi * d / feather)) with d =
                // distance to the nearest tile border, clamped to [0, feather]
                cv::Mat w(th, tw, CV_32FC1);
                for (int r = 0; r < th; r++) {
                    const int dy = (std::min)(r, th - 1 - r);
                    for (int c = 0; c < tw; c++) {
                        const int dx = (std::min)(c, tw - 1 - c);
                        const int d = (std::min)(dx, dy);
                        const int dc = (std::min)(d, feather);
                        w.at<float>(r, c) = 0.5f * (1.0f - std::cos(CV_PI * dc / (float)feather));
                    }
                }

                cv::Mat tile_f;
                tiles[i][j].convertTo(tile_f, CV_32FC3);

                cv::Rect roi(x, y, tw, th);
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

                // Ensure tile fits within output image
                if (x < 0 || y < 0 || x + tileWidth > maxX || y + tileHeight > maxY) {
                    continue;
                }

                // Define ROI in output image
                cv::Rect roi(x, y, tileWidth, tileHeight);
                cv::Mat outputROI = output(roi);

                // Ensure compatible types
                if (tiles[i][j].type() != outputROI.type()) {
                    continue;
                }

                // Perform alpha blending
                // outputROI = alpha * tile + (1 - alpha) * outputROI
                addWeighted(tiles[i][j], alpha, outputROI, 1.0f - alpha, 0.0f, outputROI);
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