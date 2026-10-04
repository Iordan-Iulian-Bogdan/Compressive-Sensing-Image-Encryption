#include "photo_upscaler.hpp"

#include "image_tiles.hpp"

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

void UpscaleAvirInplace(cv::Mat& tile) {
  cv::Mat up;
  cs_upscale_2x_avir(tile, up);
  if (!up.empty()) tile = std::move(up);
}

long CurrentProcessId() {
#if defined(_WIN32)
  return static_cast<long>(GetCurrentProcessId());
#else
  return static_cast<long>(getpid());
#endif
}

// One batched nunif run over the grid. Returns true when every tile was
// replaced with its 2x waifu2x output; false (with per-tile AVIR fallback
// already applied) otherwise.
bool UpscaleWaifu2xBatch(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt) {
  struct Job {
    int i;
    int j;
  };
  std::vector<Job> jobs;
  const int n = static_cast<int>(tiles.size());
  for (int i = 0; i < n; ++i) {
    if (static_cast<int>(tiles[i].size()) != n) return false;
    for (int j = 0; j < n; ++j) {
      const cv::Mat& tile = tiles[i][j];
      if (!tile.empty()) {
        if (tile.type() != CV_8UC3) return false;
        jobs.push_back({i, j});
      }
    }
  }
  if (jobs.empty()) return true;

  static std::atomic<long> sequence{0};
  std::filesystem::path root;
  try {
    root = std::filesystem::temp_directory_path() /
           ("cs_waifu2x_" + std::to_string(CurrentProcessId()) + "_" +
            std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(root / "in");
    std::filesystem::create_directories(root / "out");
  } catch (const std::exception& e) {
    std::cerr << "Warning: --photo-upscaler waifu2x: temp dir failed ("
              << e.what() << "); falling back to AVIR" << std::endl;
    return false;
  }

  // Best-effort temp cleanup on every exit path below.
  struct TempGuard {
    std::filesystem::path root;
    ~TempGuard() {
      try {
        std::filesystem::remove_all(root);
      } catch (...) {
      }
    }
  } guard{root};

  auto name = [](int i, int j) {
    return "tile_" + std::to_string(i) + "_" + std::to_string(j) + ".png";
  };
  for (const Job& job : jobs) {
    try {
      if (!cv::imwrite((root / "in" / name(job.i, job.j)).string(),
                       tiles[job.i][job.j])) {
        std::cerr << "Warning: --photo-upscaler waifu2x: tile write failed; "
                     "falling back to AVIR"
                  << std::endl;
        return false;
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: --photo-upscaler waifu2x: tile write failed ("
                << e.what() << "); falling back to AVIR" << std::endl;
      return false;
    }
  }

  const std::string cmd = opt.waifu2x_cmd + " -i \"" + (root / "in").string() +
                          "\" -o \"" + (root / "out").string() + "\" " +
                          cs_waifu2x_effective_args(opt);
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::cerr << "Warning: --photo-upscaler waifu2x: command failed (code "
              << rc << "); falling back to AVIR" << std::endl;
    return false;
  }

  int bad = 0;
  for (const Job& job : jobs) {
    cv::Mat& tile = tiles[job.i][job.j];
    cv::Mat up;
    try {
      up = cv::imread((root / "out" / name(job.i, job.j)).string(),
                      cv::IMREAD_COLOR);
    } catch (...) {
    }
    if (up.empty() || up.type() != CV_8UC3 || up.cols != tile.cols * 2 ||
        up.rows != tile.rows * 2) {
      UpscaleAvirInplace(tile);  // per-tile fallback keeps the 2x invariant
      ++bad;
    } else {
      tile = std::move(up);
    }
  }
  if (bad > 0) {
    std::cerr << "Warning: --photo-upscaler waifu2x: " << bad << " of "
              << jobs.size() << " tiles unusable; AVIR fallback applied"
              << std::endl;
  }
  return true;
}

}  // namespace

std::string cs_waifu2x_effective_args(const CsPhotoUpscalerOptions& opt) {
  if (!opt.waifu2x_args.empty()) return opt.waifu2x_args;  // explicit override
  std::string method = opt.waifu2x_method;
  if (method != "scale" && method != "noise_scale") {
    std::cerr << "Warning: --waifu2x-method must be scale or noise_scale; "
                 "using scale"
              << std::endl;
    method = "scale";
  }
  int noise = opt.waifu2x_noise;
  if (noise < 0 || noise > 3) {
    std::cerr << "Warning: --waifu2x-noise must be in [0, 3]; using 0"
              << std::endl;
    noise = 0;
  }
  return "--style photo --method " + method + " -n " + std::to_string(noise) +
         " -g -1";
}

void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt) {
  if (opt.backend == "waifu2x") {
    if (UpscaleWaifu2xBatch(tiles, opt)) return;
    // Batch failed (warning already printed): fall through to AVIR.
    for (auto& row : tiles)
      for (auto& tile : row)
        if (!tile.empty() && tile.type() == CV_8UC3) UpscaleAvirInplace(tile);
    return;
  }
  if (opt.backend != "avir") {
    std::cerr << "Warning: unknown --photo-upscaler '" << opt.backend
              << "'; using AVIR" << std::endl;
  }
  for (auto& row : tiles)
    for (auto& tile : row)
      if (!tile.empty() && tile.type() == CV_8UC3) UpscaleAvirInplace(tile);
}
