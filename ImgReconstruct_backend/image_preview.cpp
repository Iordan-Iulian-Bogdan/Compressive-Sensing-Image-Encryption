#include "image_preview.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#if defined(_WIN32)
#include <windows.h>
#endif

ImagePreview::~ImagePreview() { Stop(); }

void ImagePreview::Start(
    const std::string& window_name, cv::Mat& reconstructed,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const std::vector<std::vector<cv::Mat>>& image_tiles) {
  Stop();
  stop_requested_.store(false, std::memory_order_release);
  thread_ = std::thread(&ImagePreview::Run, this, window_name,
                        std::ref(reconstructed), std::cref(coordinates),
                        std::cref(image_tiles));
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
    const std::vector<std::vector<cv::Mat>>& image_tiles) {
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
  const double aspect_ratio =
      static_cast<double>(reconstructed.cols) / reconstructed.rows;
  constexpr double kPreviewScale = 0.5;

  while (!stop_requested_.load(std::memory_order_acquire)) {
    cv::waitKey(33);
    reconstructed = reconstructImage(image_tiles, coordinates);
  cv::Mat preview = reconstructed.clone();
  cv::resize(preview, preview,
               cv::Size(static_cast<int>(kPreviewScale * horizontal),
                        static_cast<int>(kPreviewScale * vertical * aspect_ratio)));
    cv::imshow(window_name, preview);
  }
}
