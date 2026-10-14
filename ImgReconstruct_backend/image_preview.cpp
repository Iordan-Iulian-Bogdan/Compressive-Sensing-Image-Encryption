#include "image_preview.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#endif

ImagePreview::~ImagePreview() { Stop(); }

void ImagePreview::Start(
    const std::string& window_name, cv::Mat& reconstructed,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const std::vector<std::vector<cv::Mat>>& image_tiles,
    std::mutex* grid_mutex,
    std::function<cv::Mat(const cv::Mat&)> transform) {
  Stop();
  stop_requested_.store(false, std::memory_order_release);
  thread_ = std::thread(&ImagePreview::Run, this, window_name,
                        std::ref(reconstructed), std::cref(coordinates),
                        std::cref(image_tiles), grid_mutex, std::move(transform));
}

void ImagePreview::Stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ImagePreview::Run(
    const std::string& window_name, cv::Mat& reconstructed,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const std::vector<std::vector<cv::Mat>>& image_tiles,
    std::mutex* grid_mutex,
    std::function<cv::Mat(const cv::Mat&)> transform) {
#if defined(_WIN32)
  RECT desktop;
  const HWND desktop_window = GetDesktopWindow();
  GetWindowRect(desktop_window, &desktop);
  const int horizontal = desktop.right;
  const int vertical = desktop.bottom;
#else
  // OpenCV sizes the window itself on platforms without desktop metrics.
  const int horizontal = 1280;
  const int vertical = 720;
#endif
  constexpr double kPreviewScale = 0.5;

  while (!stop_requested_.load(std::memory_order_acquire)) {
    try {
      cv::waitKey(33);
      if (grid_mutex) {
        std::lock_guard<std::mutex> lk(*grid_mutex);
        reconstructed = reconstructImage(image_tiles, coordinates);
      } else {
        reconstructed = reconstructImage(image_tiles, coordinates);
      }
      // Progressive grids start all-empty (a tile appears only once
      // solved): nothing to show yet, keep the window hidden rather than
      // resizing an empty Mat (which would throw inside this thread).
      if (reconstructed.empty()) continue;
      cv::Mat preview = reconstructed.clone();
      if (transform) preview = transform(preview);
      if (preview.empty()) continue;
      // Aspect from the current frame: the bound canvas may start empty
      // (shared multi-pass previews), when a one-time ratio would be NaN.
      double aspect_ratio = 1.0;
      if (preview.rows > 0) {
        aspect_ratio = static_cast<double>(preview.cols) / preview.rows;
        if (!(aspect_ratio > 0.0)) aspect_ratio = 1.0;
      }
      const int show_w = (std::max)(1, static_cast<int>(kPreviewScale * horizontal));
      const int show_h = (std::max)(1, static_cast<int>(show_w / aspect_ratio));
      cv::resize(preview, preview, cv::Size(show_w, show_h));
      cv::imshow(window_name, preview);
    } catch (const cv::Exception& e) {
      // A preview failure (headless backend, closed window) must never kill
      // a solve: report once and retire this thread.
      std::cerr << "preview stopped: " << e.what() << std::endl;
      break;
    }
  }
}
