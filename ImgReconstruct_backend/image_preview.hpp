#ifndef IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_
#define IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_

#include "image_tiles.hpp"

#include <atomic>
#include <string>
#include <thread>

// Owns the optional live preview thread used while image tiles are solved.
class ImagePreview {
 public:
  ImagePreview() = default;
  ~ImagePreview();

  ImagePreview(const ImagePreview&) = delete;
  ImagePreview& operator=(const ImagePreview&) = delete;

  void Start(const std::string& window_name, cv::Mat& reconstructed,
             const std::vector<std::vector<TileCoord>>& coordinates,
             const std::vector<std::vector<cv::Mat>>& image_tiles);
  void Stop();

 private:
  void Run(const std::string& window_name, cv::Mat& reconstructed,
           const std::vector<std::vector<TileCoord>>& coordinates,
           const std::vector<std::vector<cv::Mat>>& image_tiles);

  std::atomic<bool> stop_requested_{true};
  std::thread thread_;
};

#endif  // IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_
