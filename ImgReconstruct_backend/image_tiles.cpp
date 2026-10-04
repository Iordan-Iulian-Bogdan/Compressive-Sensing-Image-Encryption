#include "image_tiles.hpp"

#include "avir.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>

cv::Mat reconstructImage(
    const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates) {
  if (tiles.empty() || coordinates.empty() ||
      tiles.size() != coordinates.size() ||
      tiles[0].size() != coordinates[0].size()) {
    return cv::Mat();
  }

  const int tile_count = static_cast<int>(tiles.size());
  int max_x = 0;
  int max_y = 0;
  for (int i = 0; i < tile_count; ++i) {
    for (int j = 0; j < tile_count; ++j) {
      if (tiles[i][j].empty()) continue;
      max_x = (std::max)(max_x, coordinates[i][j].x + tiles[i][j].cols);
      max_y = (std::max)(max_y, coordinates[i][j].y + tiles[i][j].rows);
    }
  }
  if (max_x <= 0 || max_y <= 0) return cv::Mat();

  const int type = tiles[0][0].empty() ? CV_8UC3 : tiles[0][0].type();
  cv::Mat output(max_y, max_x, type, cv::Scalar(0));
  for (int i = 0; i < tile_count; ++i) {
    for (int j = 0; j < tile_count; ++j) {
      if (tiles[i][j].empty()) continue;
      const cv::Rect roi(coordinates[i][j].x, coordinates[i][j].y,
                         tiles[i][j].cols, tiles[i][j].rows);
      tiles[i][j].copyTo(output(roi));
    }
  }
  return output;
}

std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image,
                                         int tile_width, int tile_height,
                                         int rows, int cols) {
  std::vector<cv::Mat> tiles;
  tiles.reserve(static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const cv::Rect roi(j * tile_width, i * tile_height, tile_width,
                         tile_height);
      tiles.push_back(image(roi).clone());
    }
  }
  return tiles;
}

std::vector<std::string> spiralOrder(int tile_count) {
  std::vector<std::string> result;
  if (tile_count <= 0) return result;

  std::vector<std::vector<std::string>> matrix(
      tile_count, std::vector<std::string>(tile_count));
  for (int i = 0; i < tile_count; ++i) {
    for (int j = 0; j < tile_count; ++j) {
      matrix[i][j] = std::to_string(i) + "_" + std::to_string(j);
    }
  }

  const int start_row = tile_count / 2;
  const int start_col = tile_count / 2;
  int direction = 0;  // up, left, down, right
  int steps = 1;
  int row = start_row;
  int col = start_col;
  result.reserve(static_cast<size_t>(tile_count) * tile_count);
  result.push_back(matrix[row][col]);

  while (result.size() < static_cast<size_t>(tile_count) * tile_count) {
    for (int turn = 0; turn < 2; ++turn) {
      for (int step = 0; step < steps; ++step) {
        if (direction == 0) {
          --row;
        } else if (direction == 1) {
          --col;
        } else if (direction == 2) {
          ++row;
        } else {
          ++col;
        }
        if (row >= 0 && row < tile_count && col >= 0 && col < tile_count) {
          result.push_back(matrix[row][col]);
        }
      }
      direction = (direction + 1) % 4;
    }
    ++steps;
  }
  return result;
}

void splitImageIntoTiles(
    const cv::Mat& input_image, std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates, int tile_count,
    int overlap) {
  if (input_image.empty() || tile_count <= 0 || overlap < 0) return;

  const int height = input_image.rows;
  const int width = input_image.cols;
  const int tile_width = static_cast<int>(std::ceil(
      (width + (tile_count - 1) * static_cast<double>(overlap)) / tile_count));
  const int tile_height = static_cast<int>(std::ceil(
      (height + (tile_count - 1) * static_cast<double>(overlap)) / tile_count));

  tiles.resize(tile_count, std::vector<cv::Mat>(tile_count));
  coordinates.resize(tile_count, std::vector<TileCoord>(tile_count));
  for (int i = 0; i < tile_count; ++i) {
    for (int j = 0; j < tile_count; ++j) {
      int x = j * (tile_width - overlap);
      int y = i * (tile_height - overlap);
      if (x + tile_width > width && width >= tile_width) x = width - tile_width;
      if (y + tile_height > height && height >= tile_height) y = height - tile_height;
      x = (std::max)(0, x);
      y = (std::max)(0, y);

      const int current_width = (std::min)(tile_width, width - x);
      const int current_height = (std::min)(tile_height, height - y);
      if (x >= width || y >= height || current_width <= 0 || current_height <= 0) {
        continue;
      }
      const cv::Rect roi(x, y, current_width, current_height);
      tiles[i][j] = input_image(roi).clone();
      coordinates[i][j] = {x, y};
    }
  }
}

cv::Mat blendTilesWithImage(
    const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& target_image, float alpha, int feather) {
  if (tiles.empty() || coordinates.empty() ||
      tiles.size() != coordinates.size() ||
      tiles[0].size() != coordinates[0].size() || target_image.empty()) {
    return cv::Mat();
  }

  const int tile_count = static_cast<int>(tiles.size());
  const int max_x = target_image.cols;
  const int max_y = target_image.rows;
  if (feather > 0) {
    cv::Mat accum(max_y, max_x, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight_sum(max_y, max_x, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < tile_count; ++i) {
      for (int j = 0; j < tile_count; ++j) {
        if (tiles[i][j].empty()) continue;
        const int tile_height = tiles[i][j].rows;
        const int tile_width = tiles[i][j].cols;
        const int x = coordinates[i][j].x;
        const int y = coordinates[i][j].y;
        int clipped_x = x;
        int clipped_y = y;
        int clipped_width = tile_width;
        int clipped_height = tile_height;
        if (clipped_x < 0) {
          clipped_width += clipped_x;
          clipped_x = 0;
        }
        if (clipped_y < 0) {
          clipped_height += clipped_y;
          clipped_y = 0;
        }
        clipped_width = (std::min)(clipped_width, max_x - clipped_x);
        clipped_height = (std::min)(clipped_height, max_y - clipped_y);
        if (clipped_width <= 0 || clipped_height <= 0) continue;

        const cv::Mat tile_roi =
            tiles[i][j](cv::Rect(0, 0, clipped_width, clipped_height));
        cv::Mat weights(clipped_height, clipped_width, CV_32FC1);
        for (int r = 0; r < clipped_height; ++r) {
          const int dy = (std::min)(r, tile_height - 1 - r);
          for (int c = 0; c < clipped_width; ++c) {
            const int dx = (std::min)(c, tile_width - 1 - c);
            const int distance = (std::min)((std::min)(dx, dy), feather);
            weights.at<float>(r, c) = static_cast<float>(
                0.5 * (1.0 - std::cos(CV_PI * distance / feather)));
          }
        }

        cv::Mat tile_float;
        tile_roi.convertTo(tile_float, CV_32FC3);
        const cv::Rect roi(clipped_x, clipped_y, clipped_width, clipped_height);
        cv::Mat accum_roi = accum(roi);
        cv::Mat weight_roi = weight_sum(roi);
        std::vector<cv::Mat> weight_channels(3, weights);
        cv::Mat weights_3c;
        cv::merge(weight_channels, weights_3c);
        cv::Mat contribution;
        cv::multiply(tile_float, weights_3c, contribution);
        cv::add(accum_roi, contribution, accum_roi);
        cv::add(weight_roi, weights, weight_roi);
      }
    }

    cv::Mat safe_weights;
    (cv::max)(weight_sum, 1e-6f, safe_weights);
    std::vector<cv::Mat> accum_planes;
    cv::split(accum, accum_planes);
    for (cv::Mat& plane : accum_planes) cv::divide(plane, safe_weights, plane);
    cv::merge(accum_planes, accum);
    cv::Mat output;
    accum.convertTo(output, CV_8UC3);
    return output;
  }

  cv::Mat output = target_image.clone();
  alpha = (std::max)(0.0f, (std::min)(1.0f, alpha));
  for (int i = 0; i < tile_count; ++i) {
    for (int j = 0; j < tile_count; ++j) {
      if (tiles[i][j].empty()) continue;
      const int x = coordinates[i][j].x;
      const int y = coordinates[i][j].y;
      int clipped_x = x;
      int clipped_y = y;
      int clipped_width = tiles[i][j].cols;
      int clipped_height = tiles[i][j].rows;
      if (clipped_x < 0) {
        clipped_width += clipped_x;
        clipped_x = 0;
      }
      if (clipped_y < 0) {
        clipped_height += clipped_y;
        clipped_y = 0;
      }
      clipped_width = (std::min)(clipped_width, max_x - clipped_x);
      clipped_height = (std::min)(clipped_height, max_y - clipped_y);
      if (clipped_width <= 0 || clipped_height <= 0) continue;

      const cv::Mat source =
          tiles[i][j](cv::Rect(0, 0, clipped_width, clipped_height));
      cv::Mat output_roi = output(cv::Rect(clipped_x, clipped_y,
                                           clipped_width, clipped_height));
      if (source.type() != output_roi.type()) continue;
      cv::addWeighted(source, alpha, output_roi, 1.0f - alpha, 0.0,
                      output_roi);
    }
  }
  return output;
}

namespace {
uint64_t FastMix64(uint64_t state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}
}  // namespace

void shuffle(std::vector<int>& data, unsigned seed) {
  if (data.size() < 2) return;
  uint64_t state = static_cast<uint64_t>(seed) * 0x2545F4914F6CDD1DULL +
                   0x9E3779B97F4A7C15ULL;
  for (size_t i = data.size() - 1; i > 0; --i) {
    const size_t j = FastMix64(state) % (i + 1);
    std::swap(data[i], data[j]);
  }
}

void reverseShuffle(std::vector<int>& data, unsigned seed) {
  if (data.size() < 2) return;
  std::vector<int> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);
  shuffle(indices, seed);
  std::vector<int> original(data.size());
  for (size_t i = 0; i < data.size(); ++i) original[indices[i]] = data[i];
  data = std::move(original);
}

void cs_upscale_2x_avir(const cv::Mat& src, cv::Mat& dst) {
  if (src.empty() || src.type() != CV_8UC3) return;
  dst.create(src.rows * 2, src.cols * 2, CV_8UC3);
  typedef avir::fpclass_def<float, float,
                           avir::CImageResizerDithererErrdINL<float>>
      fpclass_dith;
  avir::CImageResizer<fpclass_dith> resizer(8);
  resizer.resizeImage(src.data, src.cols, src.rows, 0, dst.data, dst.cols,
                      dst.rows, 3, 0);
}
