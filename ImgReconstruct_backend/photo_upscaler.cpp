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
         " -s 2 -m " + cs_waifu2x_ncnn_model_dir(opt);
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

std::string cs_waifu2x_ncnn_model_dir(const CsPhotoUpscalerOptions& opt) {
  const std::string& m = opt.waifu2x_ncnn_model;
  const bool looks_like_path =
      m.find_first_of("/\\") != std::string::npos ||
      (m.size() > 1 && m[1] == ':');
  if (looks_like_path) return m;  // explicit path: untouched
  // plain name: prefer the models shipped next to the binary ...
  const std::string dir = exe_sibling_dir(opt.waifu2x_ncnn_cmd);
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

#if __has_include(<opencv2/dnn_superres.hpp>)
#include <opencv2/dnn_superres.hpp>
#endif
#include <mutex>
#include <set>

namespace {
// Warn-once registry for FSRCNN model problems (a missing model would
// otherwise spam one warning per tile across the whole grid).
void cs_fsrcnn_warn_once(const std::string& model, const std::string& what) {
  static std::mutex m;
  static std::set<std::string> warned;
  std::lock_guard<std::mutex> lk(m);
  if (warned.insert(model).second) {
    std::cerr << "Warning: --photo-upscaler fsrcnn: model '" << model
              << "' unusable (" << what << "); falling back to AVIR" << std::endl;
  }
}
}  // namespace

bool cs_fsrcnn_upscale_2x(const cv::Mat& src, cv::Mat& dst,
                          const std::string& model) {
  dst.release();
  if (src.empty() || src.type() != CV_8UC3) return false;
#if !__has_include(<opencv2/dnn_superres.hpp>)
  (void)model;
  return false;
#else
  // Small-input guard: dnn_superres FSRCNN crashes (AV) on tiny tiles of
  // this OpenCV build below ~32px (observed at 16x24, fine at 48x32).
  // AVIR fallback preserves the contract; production tiles are larger
  // except under extreme manual --tiles counts on small images.
  if (src.rows < 32 || src.cols < 32) return false;
  // One session per worker thread: dnn_superres instances are not documented
  // thread-safe, while fused tile workers run concurrently. FSRCNN weights
  // are ~40KB, so per-thread sessions are cheap; the model loads once per
  // thread and is reused for all its tiles.
  // NOTE: a DnnSuperResImpl that threw from readModel/upsample is NEVER
  // reused: OpenCV's importer is not exception-safe to retry on the same
  // instance (second load after a failed first load corrupts the heap and
  // crashes later in unrelated code). Every (re)load allocates fresh.
  struct Slot {
    std::string path;
    cv::Ptr<cv::dnn_superres::DnnSuperResImpl> net;
    bool ready = false;
  };
  thread_local Slot slot;
  if (!slot.ready || slot.path != model) {
    slot.net.release();
    slot.ready = false;
    slot.path.clear();
    try {
      slot.net = cv::dnn_superres::DnnSuperResImpl::create();
      slot.net->readModel(model);
      slot.net->setModel("fsrcnn", 2);
      slot.net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      slot.net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
      cs_fsrcnn_warn_once(model, e.what());
      slot.net.release();
      return false;
    }
    slot.path = model;
    slot.ready = true;
  }
  try {
    slot.net->upsample(src, dst);
  } catch (const cv::Exception& e) {
    cs_fsrcnn_warn_once(model, e.what());
    slot.net.release();
    slot.ready = false;
    slot.path.clear();
    return false;
  }
  if (dst.empty() || dst.type() != CV_8UC3 || dst.cols != src.cols * 2 ||
      dst.rows != src.rows * 2) {
    dst.release();
    return false;
  }
  return true;
#endif
}

// Single-tile variant of UpscaleSubprocessBatch: one temp dir, one input,
// one 2x output. Same contract style (true = backend produced a valid 2x
// tile, false = caller should AVIR-fallback).
bool UpscaleSubprocessSingle(const cv::Mat& src, cv::Mat& dst,
                             const std::string& cmd, const std::string& args,
                             const char* tag) {
  dst.release();
  if (src.empty() || src.type() != CV_8UC3) return false;

  static std::atomic<long> single_sequence{0};
  std::filesystem::path root;
  try {
    root = std::filesystem::temp_directory_path() /
           ("cs_upscale1_" + std::to_string(CurrentProcessId()) + "_" +
            std::to_string(single_sequence.fetch_add(1)));
    std::filesystem::create_directories(root / "in");
    std::filesystem::create_directories(root / "out");
  } catch (const std::exception& e) {
    std::cerr << "Warning: --photo-upscaler " << tag << ": temp dir failed ("
              << e.what() << "); falling back to AVIR" << std::endl;
    return false;
  }

  struct TempGuard {
    std::filesystem::path root;
    ~TempGuard() {
      try {
        std::filesystem::remove_all(root);
      } catch (...) {
      }
    }
  } guard{root};

  try {
    if (!cv::imwrite((root / "in" / "tile.png").string(), src)) {
      std::cerr << "Warning: --photo-upscaler " << tag
                << ": tile write failed; falling back to AVIR" << std::endl;
      return false;
    }
  } catch (const std::exception& e) {
    std::cerr << "Warning: --photo-upscaler " << tag << ": tile write failed ("
              << e.what() << "); falling back to AVIR" << std::endl;
    return false;
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

  cv::Mat up;
  try {
    up = cv::imread((root / "out" / "tile.png").string(), cv::IMREAD_COLOR);
  } catch (...) {
  }
  if (up.empty() || up.type() != CV_8UC3 || up.cols != src.cols * 2 ||
      up.rows != src.rows * 2) {
    std::cerr << "Warning: --photo-upscaler " << tag
              << ": tile output unusable; falling back to AVIR" << std::endl;
    return false;
  }
  dst = std::move(up);
  return true;
}

void cs_upscale_tiles_2x(std::vector<std::vector<cv::Mat>>& tiles,
                         const CsPhotoUpscalerOptions& opt) {
  if (opt.backend == "fsrcnn") {
    // In-process neural upscale, one session per worker thread inside the
    // helper; per-tile AVIR fallback keeps the 2x invariant on misses.
    for (auto& row : tiles)
      for (auto& tile : row) {
        if (tile.empty() || tile.type() != CV_8UC3) continue;
        cv::Mat up;
        if (cs_fsrcnn_upscale_2x(tile, up, opt.fsrcnn_model))
          tile = std::move(up);
        else
          UpscaleAvirInplace(tile);
      }
    return;
  }
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

bool cs_upscale_one_tile_2x(const cv::Mat& src_lr, cv::Mat& dst_hr,
                            const CsPhotoUpscalerOptions& opt) {
  dst_hr.release();
  if (src_lr.empty() || src_lr.type() != CV_8UC3) return false;
  if (opt.backend == "avir") {
    cs_upscale_2x_avir(src_lr, dst_hr);
    return !dst_hr.empty();
  }
  if (opt.backend == "fsrcnn") {
    if (cs_fsrcnn_upscale_2x(src_lr, dst_hr, opt.fsrcnn_model)) return true;
    cs_upscale_2x_avir(src_lr, dst_hr);  // warning already printed
    return false;
  }
  const char* tag = nullptr;
  std::string cmd, args;
  if (opt.backend == "waifu2x") {
    tag = "waifu2x";
    cmd = opt.waifu2x_cmd;
    args = cs_waifu2x_effective_args(opt);
  } else if (opt.backend == "waifu2x-ncnn") {
    tag = "waifu2x-ncnn";
    cmd = opt.waifu2x_ncnn_cmd;
    args = cs_waifu2x_ncnn_effective_args(opt);
  } else if (opt.backend == "realcugan") {
    tag = "realcugan";
    cmd = opt.realcugan_cmd;
    args = cs_realcugan_effective_args(opt);
  } else {
    std::cerr << "Warning: unknown --photo-upscaler '" << opt.backend
              << "'; using AVIR" << std::endl;
    cs_upscale_2x_avir(src_lr, dst_hr);
    return false;
  }
  if (UpscaleSubprocessSingle(src_lr, dst_hr, cmd, args, tag)) return true;
  cs_upscale_2x_avir(src_lr, dst_hr);  // warning already printed
  return false;
}
