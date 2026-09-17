#include <stdio.h>
#include "lbfgs.hpp"
#include "crypto_utils.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <omp.h>
#include <iomanip>
#include <list>
#include <numeric>
#include "avir.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>   // AVX2/FMA gather/scatter kernels
#endif

#if defined(_WIN32)
#include <windows.h>     // desktop metrics for the live preview window
#endif


#define MANUAL_PARAM  1
#define AUTO_PARAM  2

struct TileCoord {
    int x;
    int y;
};

struct indices {
    std::vector<int> ri_x_g, ri_y_g;
};

int nextClosestDivisible(const int& x, const int& y);

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates);
std::vector<cv::Mat> splitMat(cv::Mat& image, int M, int N);
inline void updateAxb2AndComputeFx(float* x_copy, const int* ri_x, const int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n);
inline void eval_g(float* Axb2, float* g, int n);
inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n);
float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
);
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
);
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols);

/** @brief reconstructs all 3 color channels inside ONE lbfgs run (dimension 3n).
Fuses the measurement extraction, the residual gathers and the gradient pass across channels,
so per solver iteration the vec machinery runs once instead of three times.
DCT kernels remain 6 real cv::dct calls (cv::dct rejects multi-channel input), but the
fits/vector overhead and 3 channel launches collapse into one.
*/
void reconstruct_image_packed(const cv::Mat& pixel_measurements, const float& param_c, const int& rows, const int& cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat refs[3], cv::Mat& out);

/** @brief fused objective/gradient for the stacked 3-channel solve.
x has dimension 3n laid out as [plane c0 | plane c1 | plane c2] (row-major n = rows*cols each).
b, Axb2, x_copy, g follow the same stacked layout.
*/
float evaluate_stacked(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
);

void reconstruct_color_channel(const cv::Mat& measurement, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cv::Mat());
std::vector<std::string> splitString(const std::string& str, const char& delimiter);
std::string removeCharacter(const std::string& str, const char& ch);
void storeStringInColorMat(const std::string& text, cv::Mat& colorMat);
std::string retrieveStringFromColorMat(const cv::Mat& colorMat);
std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image, const int& tile_width, const int& tile_height, const int& rows, const int& cols);
std::vector<std::string> spiralOrder(const int& tiles);
void splitImageIntoTiles(const cv::Mat& inputImage,
    std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates,
    const int& tileCountN,
    const int& overlap);
/** @brief merges tiles back into an image.
@param alpha : legacy sequential alpha blend weight (used when feather <= 0)
@param feather : if > 0, tiles are composited with a cosine-feathered weight
                 ramp of this many pixels at every tile border instead of the
                 sequential alpha blend; weights sum to 1 across overlaps, so
                 the result is order-independent and seam-free
*/
cv::Mat blendTilesWithImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& targetImage,
    float alpha,
    int feather = 0);

struct display {
    std::thread display_image_thread;
    bool continue_displaying = true;

    void display_output(std::string windowName, cv::Mat& image, const std::vector<std::vector<TileCoord>>& coordinates, const std::vector<std::vector<cv::Mat>>& image_tiles) const{
#if defined(_WIN32)
        RECT desktop;
        const HWND hDesktop = GetDesktopWindow();
        GetWindowRect(hDesktop, &desktop);
        int horizontal = desktop.right;
        int vertical = desktop.bottom;
#else
        // no Win32 desktop metrics; OpenCV will fit the window itself
        int horizontal = 1280;
        int vertical = 720;
#endif
        double ratio = double(image.cols) / double(image.rows);
        double scale = 0.5;

        while (continue_displaying) {
            cv::waitKey(33);
            image = reconstructImage(image_tiles, coordinates);
            cv::Mat aux = image.clone();
            cv::resize(aux, aux, cv::Size(scale * horizontal, scale * vertical * ratio));
            cv::imshow(windowName, aux);
        }
    }

    void display_image(const std::string& windowName, cv::Mat& reconstructed, const std::vector<std::vector<TileCoord>>& coordinates, const std::vector<std::vector<cv::Mat>>& image_tiles) {
        display_image_thread = std::thread(&display::display_output, this, windowName, std::ref(reconstructed), std::ref(coordinates), std::ref(image_tiles));
    }

    void stop_display() {
        continue_displaying = false;
        display_image_thread.join();
    }
};

void shuffle(std::vector<int>& data, unsigned seed);
void reverseShuffle(std::vector<int>& data, unsigned seed);
void sharpenImage(const cv::Mat& input, cv::Mat& output, float sharpness);