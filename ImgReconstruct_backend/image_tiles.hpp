#ifndef IMGRECONSTRUCT_BACKEND_IMAGE_TILES_HPP_
#define IMGRECONSTRUCT_BACKEND_IMAGE_TILES_HPP_

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct TileCoord {
  int x;
  int y;
};

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
                         const std::vector<std::vector<TileCoord>>& coordinates);
std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image,
                                         int tile_width, int tile_height,
                                         int rows, int cols);
std::vector<std::string> spiralOrder(int tiles);
void splitImageIntoTiles(
    const cv::Mat& input_image, std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates, int tile_count,
    int overlap);
cv::Mat blendTilesWithImage(
    const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& target_image, float alpha, int feather = 0);

void shuffle(std::vector<int>& data, unsigned seed);
void reverseShuffle(std::vector<int>& data, unsigned seed);

// 2x upscale of one CV_8UC3 tile with the vendored AVIR resampler. No-op
// for empty or non-8UC3 input.
void cs_upscale_2x_avir(const cv::Mat& src, cv::Mat& dst);

#endif  // IMGRECONSTRUCT_BACKEND_IMAGE_TILES_HPP_
