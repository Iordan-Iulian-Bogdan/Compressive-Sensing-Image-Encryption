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

// One batched subprocess run over the grid (nunif waifu2x or an ncnn-vulkan
// binary: same -i/-o dir contract). Returns true when every tile was
// replaced with its 2x output; false (with per-tile AVIR fallback already
// applied) otherwise.
bool UpscaleSubprocessBatch(std::vector<std::vector<cv::Mat>>& tiles,
                            const std::string& cmd,
                            const std::string& args,
                            const char* tag) {
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
           ("cs_upscale_" + std::to_string(CurrentProcessId()) + "_" +
            std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(root / "in");
    std::filesystem::create_directories(root / "out");
  } catch (const std::exception& e) {
    std::cerr << "Warning: --photo-upscaler " << tag << ": temp dir failed ("
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
        std::cerr << "Warning: --photo-upscaler " << tag
                  << ": tile write failed; falling back to AVIR" << std::endl;
        return false;
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: --photo-upscaler " << tag
                << ": tile write failed (" << e.what()
                << "); falling back to AVIR" << std::endl;
      return false;
    }
  }

  const std::string full =
      cmd + " -i \"" + (root / "in").string() + "\" -o \"" +
      (root / "out").string() + "\" " + args;
  const int rc = std::system(full.c_str());
  if (rc != 0) {
    std::cerr << "Warning: --photo-upscaler " << tag << ": command failed (code "
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
    std::cerr << "Warning: --photo-upscaler " << tag << ": " << bad << " of "
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

// Shared method/noise mapping for the ncnn CLIs: "noise_scale" selects the
// denoise level, anything else (i.e. "scale") disables denoise (-1).
int ncnn_noise_level(const CsPhotoUpscalerOptions& opt, const char* tag) {
  if (opt.waifu2x_method == "noise_scale") {
    int noise = opt.waifu2x_noise;
    if (noise < 0 || noise > 3) {
      std::cerr << "Warning: --waifu2x-noise must be in [0, 3] for " << tag
                << "; using 0" << std::endl;
      noise = 0;
    }
    return noise;
  }
  if (opt.waifu2x_method != "scale") {
    std::cerr << "Warning: --waifu2x-method must be scale or noise_scale "
                 "for "
              << tag << "; using scale" << std::endl;
  }
  return -1;
}

std::string cs_waifu2x_ncnn_effective_args(const CsPhotoUpscalerOptions& opt) {
  if (!opt.waifu2x_ncnn_args.empty()) return opt.waifu2x_ncnn_args;
  return "-n " + std::to_string(ncnn_noise_level(opt, "waifu2x-ncnn")) +
         " -s 2";
}

std::string cs_realcugan_effective_args(const CsPhotoUpscalerOptions& opt) {
  if (!opt.realcugan_args.empty()) return opt.realcugan_args;
  return "-n " + std::to_string(ncnn_noise_level(opt, "realcugan")) +
         " -s 2 -m " + cs_realcugan_model_dir(opt);
}

#if defined(_WIN32)
static std::string exe_sibling_dir(const std::string& cmd) {
  // First whitespace-delimited token, unquoted: the executable.
  std::string exe;
  {
    // simple quote-aware split
    bool in_q = false;
    for (char c : cmd) {
      if (c == '"' || c == '\'') {
        in_q = !in_q;
        continue;
      }
      if (!in_q && (c == ' ' || c == '\t')) break;
      exe.push_back(c);
    }
  }
  char full[MAX_PATH] = {0};
  auto directory_of = [](const std::string& path) {
    const size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? std::string() : path.substr(0, p);
  };
  if (exe.find_first_of("/\\") != std::string::npos) return directory_of(exe);
  DWORD n = SearchPathA(nullptr, exe.c_str(), ".exe", MAX_PATH, full, nullptr);
  if (n > 0 && n < MAX_PATH) return directory_of(full);
  return std::string();
}
#else
static std::string exe_sibling_dir(const std::string& cmd) {
  (void)cmd;
  return std::string();
}
#endif

std::string cs_realcugan_model_dir(const CsPhotoUpscalerOptions& opt) {
  std::string m = opt.realcugan_model;
  if (m == "se" || m == "pro" || m == "nose") m = "models-" + m;
  const bool looks_like_path =
      m.find_first_of("/\\") != std::string::npos ||
      (m.size() > 1 && m[1] == ':');
  if (looks_like_path) return opt.realcugan_model;  // explicit path: untouched
  // short name: prefer the models shipped next to the binary ...
  const std::string dir = exe_sibling_dir(opt.realcugan_cmd);
  auto exists_dir = [](const std::string& p) {
    std::error_code ec;
    return !p.empty() && std::filesystem::is_directory(p, ec);
  };
  if (!dir.empty() && exists_dir(dir + "/" + m)) return dir + "/" + m;
  // ... then the current directory; otherwise hand the name through and let
  // the tool resolve it (preserves old behavior for CWD-relative layouts).
  if (exists_dir(m)) return m;
  return m;
}

void avir_fallback(std::vector<std::vector<cv::Mat>>& tiles) {
  for (auto& row : tiles)
    for (auto& tile : row)
      if (!tile.empty() && tile.type() == CV_8UC3) UpscaleAvirInplace(tile);
}

void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt) {
  if (opt.backend == "waifu2x") {
    if (UpscaleSubprocessBatch(tiles, opt.waifu2x_cmd,
                               cs_waifu2x_effective_args(opt), "waifu2x"))
      return;
    avir_fallback(tiles);  // warning already printed
    return;
  }
  if (opt.backend == "waifu2x-ncnn") {
    if (UpscaleSubprocessBatch(tiles, opt.waifu2x_ncnn_cmd,
                               cs_waifu2x_ncnn_effective_args(opt),
                               "waifu2x-ncnn"))
      return;
    avir_fallback(tiles);
    return;
  }
  if (opt.backend == "realcugan") {
    if (UpscaleSubprocessBatch(tiles, opt.realcugan_cmd,
                               cs_realcugan_effective_args(opt), "realcugan"))
      return;
    avir_fallback(tiles);
    return;
  }
  if (opt.backend != "avir") {
    std::cerr << "Warning: unknown --photo-upscaler '" << opt.backend
              << "'; using AVIR" << std::endl;
  }
  avir_fallback(tiles);
}
