#ifndef IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_
#define IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_

#include "image_tiles.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

// Owns the optional live preview thread used while image tiles are solved.
// grid_mutex (optional) serializes the per-poll reconstructImage against
// concurrent tile writes (HR grid: solver/upscaler workers replace cell
// headers, which is not an atomic op for readers). The LR path passes
// nullptr (pre-existing behavior).
class ImagePreview {
  public:
   ImagePreview() = default;
   ~ImagePreview();

   ImagePreview(const ImagePreview&) = delete;
   ImagePreview& operator=(const ImagePreview&) = delete;

   void Start(const std::string& window_name, cv::Mat& reconstructed,
              const std::vector<std::vector<TileCoord>>& coordinates,
              const std::vector<std::vector<cv::Mat>>& image_tiles,
              std::mutex* grid_mutex = nullptr);
   void Stop();

  private:
   void Run(const std::string& window_name, cv::Mat& reconstructed,
            const std::vector<std::vector<TileCoord>>& coordinates,
            const std::vector<std::vector<cv::Mat>>& image_tiles,
            std::mutex* grid_mutex);

   std::atomic<bool> stop_requested_{true};
   std::thread thread_;
};

#endif  // IMGRECONSTRUCT_BACKEND_IMAGE_PREVIEW_HPP_
