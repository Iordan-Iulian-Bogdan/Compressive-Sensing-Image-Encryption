// ONNX Runtime denoisers and SR priors for the CS backend: DnCNN, SCUNet and
// Restormer denoisers plus the ONNX 2x SR prior. Moved out of helper_functions.cpp;
// the public entry points are declared in helper_functions.hpp.
#include "helper_functions.hpp"
#include "image_tiles.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <onnxruntime_cxx_api.h>
#ifdef CS_ONNX_DML
#include <dml_provider_factory.h>   // DirectML ORT build (see build_gpu.bat)
#endif
#endif

// Set before the first ONNX session is created. A multi-tile decrypt calls
// Run() from many OpenMP threads at once, so each run then uses one intra-op
// thread instead of all of them competing for the session's shared pool.
static std::atomic<bool> g_tile_parallel{ false };
void cs_onnx_set_tile_parallel(bool on) { g_tile_parallel = on; }

#ifdef _WIN32
// In-process DnCNN color denoiser (repo-vendored ONNX model) for the dncnn
// RED branch. Sessions are cached per path (load once, warn once); Run is
// thread-safe so fused tile workers may share one session. Model contract
// (verified by unit test, not assumed): RGB planar [0,1] in, CLEAN image
// out (not residual) — a clean input comes back nearly unchanged, so the
// first implementation's residual assumption failed loudly here and was
// flipped.
struct CsDncnnEntry {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cs-dncnn" };
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
    bool dml = false;      // DirectML sessions do not allow concurrent Run()
    std::mutex run_mu;
    std::string path;      // model file (for the shape-folded DML sessions)
    std::map<std::pair<int, int>, std::unique_ptr<Ort::Session>> folded;   // tile size -> folded session
    std::mutex fold_mu;    // guards 'folded' and 'dml_dyn' (a prefetch thread builds sessions too)
    // SR prior on DirectML with shape folding: 'session' is then only a cheap CPU
    // session used to validate the model and read its I/O names; the dynamic
    // DirectML session ('dml_dyn') is built lazily, only if folding fails.
    bool meta_only = false;
    std::unique_ptr<Ort::Session> dml_dyn;
};
// CS_GPU_ONNX_FOLD=0 turns the shape-folded DirectML sessions off.
static bool cs_fold_env_on() { const char* v = std::getenv("CS_GPU_ONNX_FOLD"); return !(v && v[0] == '0'); }
static std::mutex g_dncnn_mutex;
static std::map<std::string, std::unique_ptr<CsDncnnEntry>> g_dncnn_cache;
static std::map<std::string, bool> g_dncnn_failed;

// The ORT DLL is delay-loaded and resolved from PATH: a stale system DLL
// older than the vendored headers makes GetApi(ORT_API_VERSION) return null,
// and constructing Ort::Env on it AV-crashes (not an Ort::Exception, so the
// loader below cannot catch it). Probe the API table first so a
// version-skewed runtime degrades to "model unavailable" instead of taking
// the process down.
static bool cs_ort_api_available() {
    try {
        const OrtApiBase* base = OrtGetApiBase();
        if (!base || !base->GetApi) return false;
        return base->GetApi(ORT_API_VERSION) != nullptr;
    }
    catch (...) {
        return false;
    }
}

static CsDncnnEntry* cs_dncnn_entry(const std::string& path, const char* tag) {
    std::lock_guard<std::mutex> lk(g_dncnn_mutex);
    auto it = g_dncnn_cache.find(path);
    if (it != g_dncnn_cache.end()) return it->second.get();
    if (g_dncnn_failed.find(path) != g_dncnn_failed.end()) return nullptr;
    if (!cs_ort_api_available()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                "Warning: onnxruntime.dll is older than the build headers "
                "(API %d unavailable); DnCNN denoiser disabled\n",
                ORT_API_VERSION);
        }
        g_dncnn_failed[path] = true;
        return nullptr;
    }
    try {
        auto e = std::make_unique<CsDncnnEntry>();
        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (g_tile_parallel) opt.SetIntraOpNumThreads(1);
#ifdef CS_ONNX_DML
        // CS_GPU_ONNX=1: run the model on the GPU through DirectML (adapter 0).
        // DML requires sequential execution and no memory-pattern planning.
        const bool want_dml = [] { const char* g = std::getenv("CS_GPU_ONNX"); return g && g[0] == '1'; }();
        if (want_dml && cs_fold_env_on() && std::strcmp(tag, "SR") == 0) {
            e->dml = true;   // no session here at all: see meta_only
            e->meta_only = true;
        }
        else if (want_dml) {
            try {
                opt.DisableMemPattern();
                opt.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(opt, 0));
                e->dml = true;
            }
            catch (const Ort::Exception& ex) {
                std::fprintf(stderr, "Warning: DirectML unavailable (%s); %s runs on CPU\n", ex.what(), tag);
            }
        }
#endif
        e->path = path;
        if (e->meta_only) {
            // Creating even a bare CPU session for this 19k-node graph takes ~4.7 s, so none
            // is made: only check the file is readable (keeps the fail-fast contract) and let
            // the folded / dynamic DirectML session supply the I/O names when it is built.
            std::ifstream probe(path, std::ios::binary);
            if (!probe) throw Ort::Exception("cannot open the model file", ORT_NO_SUCHFILE);
        }
        else {
            const std::wstring wpath(path.begin(), path.end());
            const auto t_sess0 = std::chrono::steady_clock::now();
            e->session = std::make_unique<Ort::Session>(e->env, wpath.c_str(), opt);
            if (const char* pr = std::getenv("CS_GPU_PROF"); pr && pr[0] == '1')
                std::fprintf(stderr, "[gpu-prof] %s session create: %.0f ms%s\n", tag, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sess0).count(), e->dml ? " (DirectML)" : "");
            Ort::AllocatorWithDefaultOptions alloc;
            e->in_name = e->session->GetInputNameAllocated(0, alloc).get();
            e->out_name = e->session->GetOutputNameAllocated(0, alloc).get();
            auto ti = e->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                ti.GetDimensionsCount() != 4) {
                throw Ort::Exception("dncnn: expected float NCHW input", ORT_INVALID_ARGUMENT);
            }
        }
        if (e->dml) std::fprintf(stderr, "[onnx] %s: DirectML execution provider enabled\n", tag);
        auto r = g_dncnn_cache.emplace(path, std::move(e));
        return r.first->second.get();
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: %s model '%s' failed to load: %s\n",
            tag, path.c_str(), ex.what());
        g_dncnn_failed[path] = true;
        return nullptr;
    }
}

static bool cs_dncnn_run_tile(CsDncnnEntry* e, const float* rgb01, float* clean,
    int h, int w)
{
    int64_t shape[4] = { 1, 3, h, w };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem,
        const_cast<float*>(rgb01), (size_t)3 * h * w, shape, 4);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    std::vector<Ort::Value> out;
    try {
        std::unique_lock<std::mutex> run_lk(e->run_mu, std::defer_lock);
        if (e->dml) run_lk.lock();
        out = e->session->Run(Ort::RunOptions{ nullptr }, inn, &in, 1, outn, 1);
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: onnx-denoiser run failed: %s\n", ex.what());
        return false;
    }
    if (out.size() != 1 || !out[0].IsTensor()) return false;
    const std::vector<int64_t> dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] != h || dims[3] != w) return false;
    const float* src = out[0].GetTensorData<float>();
    std::memcpy(clean, src, sizeof(float) * (size_t)3 * h * w);
    return true;
}

bool cs_dncnn_available(const std::string& model_path) {
    if (model_path.empty()) return false;
    return cs_dncnn_entry(model_path, "DnCNN") != nullptr;
}

// Shared tiled RGB-NCHW clean-output runner (see header contract): BGR in,
// BGR out, feathered overlap compositing, per-path session cache. tile/ov
// bound activation memory per model size (DnCNN 512/48; SCUNet/Restormer
// 256/32).
static bool cs_onnx_denoise_tiled(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, const char* tag, int tile, int ov, bool pad128)
{
    dst_bgr.release();
    if (src_bgr.empty() || src_bgr.type() != CV_8UC3) return false;
    if (tile < 32 || ov < 8 || ov * 2 >= tile) return false;
    CsDncnnEntry* e = cs_dncnn_entry(model_path, tag);
    if (!e) return false;
    // Swin-window models (SCUNet/Restormer) reshape downsampled grids into
    // fixed 8-wide windows down to the /16 bottleneck: every model run is
    // padded to multiples of 128 below (replicate edge), so odd edge tiles
    // run instead of failing Reshape/Gather nodes. DnCNN is fully
    // convolutional: unpadded (bit-identical legacy path).
    const cv::Mat& src = src_bgr;
    const int H0 = src_bgr.rows, W0 = src_bgr.cols;
    const int H = src.rows, W = src.cols;
    (void)pad128;
    cv::Mat rgb;
    cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chs;
    cv::split(rgb, chs);
    const int T = tile, OV = ov;
    std::vector<int> ys, xs;
    for (int y = 0; y < H; y += T - OV) ys.push_back(y);
    for (int x = 0; x < W; x += T - OV) xs.push_back(x);
    if (ys.back() + T < H) ys.push_back(H - T);
    if (xs.back() + T < W) xs.push_back(W - T);
    cv::Mat acc(H, W, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight(H, W, CV_32F, cv::Scalar(0));
    std::vector<float> in, rs;
    for (size_t t = 0; t < ys.size() * xs.size(); ++t) {
        const int y0 = ys[t / xs.size()], x0 = xs[t % xs.size()];
        const int th = (std::min)(T, H - y0), tw = (std::min)(T, W - x0);
        // Swin-window models need multiples of 128 per run: replicate-pad
        // the run tile (DnCNN runs unpadded). Only the valid th x tw
        // region feeds the feathered accumulation below.
        const int ph = pad128 ? ((th + 127) / 128) * 128 : th;
        const int pw = pad128 ? ((tw + 127) / 128) * 128 : tw;
        in.assign((size_t)3 * ph * pw, 0.0f);
        rs.assign((size_t)3 * ph * pw, 0.0f);
        for (int c = 0; c < 3; ++c) {
            float* dst = &in[(size_t)c * ph * pw];
            for (int r = 0; r < ph; ++r) {
                const int sr = y0 + ((r < th) ? r : th - 1);
                const uint8_t* srow = chs[(size_t)c].ptr<uint8_t>(sr) + x0;
                for (int k = 0; k < pw; ++k) {
                    const int sk = (k < tw) ? k : tw - 1;
                    dst[(size_t)r * pw + k] = srow[sk] * (1.0f / 255.0f);
                }
            }
        }
        if (!cs_dncnn_run_tile(e, in.data(), rs.data(), ph, pw)) return false;
        for (int r = 0; r < th; ++r) {
            const int gy = y0 + r;
            cv::Vec3f* arow = acc.ptr<cv::Vec3f>(gy);
            float* wrow = weight.ptr<float>(gy);
            // linear feather over OV at interior tile borders; full weight
            // where the tile touches the image edge (no neighbor there)
            const int dt = (y0 == 0) ? OV : r;
            const int db = (y0 + th == H) ? OV : th - 1 - r;
            const float wy = (std::min)(1.0f, (std::min)(dt, db) / (float)OV);
            for (int k = 0; k < tw; ++k) {
                const int gx = x0 + k;
                const int dl = (x0 == 0) ? OV : k;
                const int dr = (x0 + tw == W) ? OV : tw - 1 - k;
                const float wx = (std::min)(1.0f, (std::min)(dl, dr) / (float)OV);
                const float wt = wx * wy;
                const size_t o = (size_t)r * pw + k;
                // clean-output convention: the model output IS the denoised
                // estimate (convertTo saturates to [0,255] at the end).
                const cv::Vec3f clean_rgb(
                    rs[o],
                    rs[(size_t)ph * pw + o],
                    rs[(size_t)2 * ph * pw + o]);
                arow[gx] += clean_rgb * wt;
                wrow[gx] += wt;
            }
        }
    }
    cv::Mat clean_rgb(H, W, CV_32FC3);
    for (int i = 0; i < H; ++i) {
        const cv::Vec3f* arow = acc.ptr<cv::Vec3f>(i);
        const float* wrow = weight.ptr<float>(i);
        cv::Vec3f* crow = clean_rgb.ptr<cv::Vec3f>(i);
        for (int j = 0; j < W; ++j) {
            const float w = wrow[j] > 0.0f ? wrow[j] : 1.0f;
            crow[j] = arow[j] / w;
        }
    }
    cv::Mat clean_bgr;
    cv::cvtColor(clean_rgb, clean_bgr, cv::COLOR_RGB2BGR);
    clean_bgr.convertTo(dst_bgr, CV_8U, 255.0);
    if (dst_bgr.rows != H0 || dst_bgr.cols != W0 || dst_bgr.type() != CV_8UC3) {
        dst_bgr.release();
        return false;
    }
    return true;
}

static void cs_onnx_geom_for_kind(int kind, const char*& tag, int& tile, int& ov) {
    if (kind == CS_ONNX_SCUNET) { tag = "SCUNet"; tile = 256; ov = 32; return; }
    if (kind == CS_ONNX_RESTORMER) { tag = "Restormer"; tile = 256; ov = 32; return; }
    tag = "DnCNN"; tile = 512; ov = 48;
}

bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path)
{
    return cs_onnx_denoise_tiled(src_bgr, dst_bgr, model_path, "DnCNN", 512, 48, false);
}

bool cs_onnx_available(const std::string& model_path, int kind) {
    if (model_path.empty()) return false;
    const char* tag; int tile, ov;
    cs_onnx_geom_for_kind(kind, tag, tile, ov);
    return cs_dncnn_entry(model_path, tag) != nullptr;
}

bool cs_onnx_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int kind)
{
    const char* tag; int tile, ov;
    cs_onnx_geom_for_kind(kind, tag, tile, ov);
    return cs_onnx_denoise_tiled(src_bgr, dst_bgr, model_path, tag, tile, ov,
        kind != CS_ONNX_DNCNN);
}

// ONNX photo-SR runner (Real-ESRGAN/SwinIR contract): same session cache as
// the denoisers, but the single output is scale-fold larger in both axes, so
// a run returns the clean HR plane (assumed [0,1]; written back [0,255]).
#ifdef CS_ONNX_DML
// DirectML runs are latency-bound and serialized per session, so concurrent
// tile workers hand their input to a batcher: same-size tiles are stacked into
// one N x 3 x h x w run (the Swin2SR export has a dynamic batch axis).
namespace {
struct SrJob {
    const float* in; int h, w;
    std::vector<float>* out; int* oh; int* ow;
    bool done = false, ok = false;
};
std::mutex g_srb_mu;
std::condition_variable g_srb_cv;
std::vector<SrJob*> g_srb_queue;
bool g_srb_busy = false;
const int kSrBatchMax = 32;
const int kSrBatchTarget = [] { const char* v = std::getenv("CS_GPU_ONNX_BATCH"); return v ? std::atoi(v) : 16; }();
const int kSrBatchWaitMs = 30;

// Opt-in (CS_GPU_PROF=1) totals of the batched DirectML runs, printed at exit.
struct SrBatchStats {
    std::atomic<long long> runs{0}, tiles{0}, ns{0};
    std::map<int, long long> hist;   // batch size -> count (guarded by g_srb_mu)
    std::map<std::pair<int, int>, long long> sizes;   // tile h,w -> tiles
    ~SrBatchStats() {
        const char* p = std::getenv("CS_GPU_PROF");
        if (!p || p[0] != '1' || runs.load() == 0) return;
        std::fprintf(stderr, "[gpu-prof] onnx-dml: %lld runs, %lld tiles (%.1f avg), run time %.1f s (%.1f ms/run, %.1f ms/tile)\n",
            (long long)runs.load(), (long long)tiles.load(), (double)tiles.load() / runs.load(),
            ns.load() * 1e-9, ns.load() * 1e-6 / runs.load(), ns.load() * 1e-6 / (double)tiles.load());
        std::fprintf(stderr, "[gpu-prof]   batch-size histogram:");
        for (auto& kv : hist) std::fprintf(stderr, " %d:%lld", kv.first, kv.second);
        std::fprintf(stderr, "\n[gpu-prof]   tile sizes (h x w: tiles):");
        for (auto& kv : sizes) std::fprintf(stderr, " %dx%d:%lld", kv.first.first, kv.first.second, kv.second);
        std::fprintf(stderr, "\n");
    }
} g_srb_stats;

// The Swin2SR export recomputes its shape/attention-mask bookkeeping (thousands
// of Shape/Range/Expand/Slice/Where nodes) on every run, and DirectML does not
// constant-fold them. Fixing batch/height/width and optimizing once on the CPU
// provider folds them away (19k -> 2k nodes, ~2.6x faster, outputs identical),
// so the DML session is built from that folded graph, one per tile size.
// CS_GPU_ONNX_FOLD=0 disables it (dynamic-shape session, batches up to 32).
const int kFoldBatch = [] { const char* v = std::getenv("CS_GPU_ONNX_FOLD_N"); return v ? std::atoi(v) : 2; }();
// CS_GPU_ONNX_PRECISION=fp16 uses the sibling "<model>_fp16.onnx" (scripts/make_swin2sr_fp16.py)
// for the folded DML sessions; the fp32 model stays the default.
const bool kFp16 = [] { const char* v = std::getenv("CS_GPU_ONNX_PRECISION"); return v && std::string(v) == "fp16"; }();

Ort::Session* cs_dml_folded_session(CsDncnnEntry* e, int h, int w) {
    if (!cs_fold_env_on() || kFoldBatch < 1) return nullptr;
    std::lock_guard<std::mutex> lk(e->fold_mu);   // a concurrent prefetch may be building this size
    const std::pair<int, int> key{ h, w };
    auto it = e->folded.find(key);
    if (it != e->folded.end()) return it->second.get();
    const auto t_fold0 = std::chrono::steady_clock::now();
    std::unique_ptr<Ort::Session> made;
    const char* precision = "fp32";
    bool cached = false;
    std::error_code ec;
    try {
        // The folded graph of a given model/size/precision never changes, so it is
        // cached on disk (temp dir); a hit skips the ~1.5 s CPU folding pass.
        std::string src_path = e->path;
        if (kFp16) {
            const std::filesystem::path p(e->path);
            const std::filesystem::path q = p.parent_path() / (p.stem().string() + "_fp16" + p.extension().string());
            if (std::filesystem::exists(q, ec)) src_path = q.string();
            else std::fprintf(stderr, "Warning: %s not found; using the fp32 model\n", q.string().c_str());
        }
        precision = (src_path == e->path) ? "fp32" : "fp16";
        const auto src_size = std::filesystem::file_size(src_path, ec);
        const auto src_time = std::filesystem::last_write_time(src_path, ec).time_since_epoch().count();
        const size_t tag_hash = std::hash<std::string>()(src_path + "|" + std::to_string((long long)src_size) + "|" +
            std::to_string((long long)src_time) + "|" + Ort::GetVersionString() + "|" + std::to_string(kFoldBatch) + "|" +
            std::to_string(h) + "x" + std::to_string(w));
        const std::filesystem::path cache_dir = std::filesystem::temp_directory_path(ec) / "cs_fold_cache";
        std::filesystem::create_directories(cache_dir, ec);
        const std::filesystem::path folded_path = cache_dir / ("swin_" + std::to_string(tag_hash) + ".onnx");
        bool hit = std::filesystem::exists(folded_path, ec);
        if (!hit) {
            const std::filesystem::path part = cache_dir / ("swin_" + std::to_string(tag_hash) + "." +
                std::to_string((long long)reinterpret_cast<intptr_t>(e)) + ".part");
            {
                Ort::SessionOptions fo;
                fo.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
                fo.SetIntraOpNumThreads(1);
                fo.SetOptimizedModelFilePath(part.wstring().c_str());
                Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(fo, "batch_size", kFoldBatch));
                Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(fo, "num_channels", 3));
                Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(fo, "height", h));
                Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(fo, "width", w));
                const std::wstring wpath(src_path.begin(), src_path.end());
                Ort::Session cpu(e->env, wpath.c_str(), fo);   // writes the folded model
            }
            std::filesystem::rename(part, folded_path, ec);   // atomic publish; a lost race is harmless
            if (ec) { std::filesystem::remove(part, ec); }
        }
        Ort::SessionOptions dopt;
        dopt.DisableMemPattern();
        dopt.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(dopt, 0));
        made = std::make_unique<Ort::Session>(e->env, folded_path.wstring().c_str(), dopt);
        const auto shp = made->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shp.size() != 4 || shp[0] != kFoldBatch || shp[1] != 3 || shp[2] != h || shp[3] != w) made.reset();   // not folded as expected
        else if (e->in_name.empty()) {
            Ort::AllocatorWithDefaultOptions alloc;
            e->in_name = made->GetInputNameAllocated(0, alloc).get();
            e->out_name = made->GetOutputNameAllocated(0, alloc).get();
        }
        if (hit) cached = true;
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: folded DML session for %dx%d failed (%s); using the dynamic one\n", h, w, ex.what());
        made.reset();
    }
    if (const char* pr = std::getenv("CS_GPU_PROF"); pr && pr[0] == '1')
        std::fprintf(stderr, "[gpu-prof] folded session create (%s): %.0f ms\n", cached ? "cached folded graph + DML load" : "fold + DML load", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fold0).count());
    if (made) std::fprintf(stderr, "[onnx] SR %dx%d: folded DirectML session (%s), batch %d\n", h, w, precision, kFoldBatch);
    return (e->folded[key] = std::move(made)).get();
}

// The dynamic-shape DirectML session: the entry's own session normally, or (when
// the entry only holds a metadata session) one built here on first use.
Ort::Session* cs_dml_dynamic_session(CsDncnnEntry* e) {
    if (!e->meta_only) return e->session.get();
    std::lock_guard<std::mutex> lk(e->fold_mu);
    if (!e->dml_dyn) {
        try {
            Ort::SessionOptions dopt;
            dopt.DisableMemPattern();
            dopt.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(dopt, 0));
            const std::wstring wpath(e->path.begin(), e->path.end());
            e->dml_dyn = std::make_unique<Ort::Session>(e->env, wpath.c_str(), dopt);
            if (e->in_name.empty()) {
                Ort::AllocatorWithDefaultOptions alloc;
                e->in_name = e->dml_dyn->GetInputNameAllocated(0, alloc).get();
                e->out_name = e->dml_dyn->GetOutputNameAllocated(0, alloc).get();
            }
        }
        catch (const Ort::Exception& ex) {
            std::fprintf(stderr, "Warning: dynamic DirectML session failed: %s\n", ex.what());
        }
    }
    return e->dml_dyn.get();
}

// One N-sample run of a (possibly folded) session; fills the jobs' outputs.
bool cs_sr_run_chunk(CsDncnnEntry* e, Ort::Session* sess, int n_batch, std::vector<SrJob*>& g, size_t first, int m) {
    const int h = g[0]->h, w = g[0]->w;
    const size_t per = (size_t)3 * h * w;
    std::vector<float> in((size_t)n_batch * per);
    for (int i = 0; i < n_batch; ++i)   // pad a short final chunk by repeating its last sample
        std::memcpy(&in[(size_t)i * per], g[first + (size_t)(std::min)(i, m - 1)]->in, sizeof(float) * per);
    int64_t shape[4] = { n_batch, 3, h, w };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tin = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    try {
        const auto prof_t0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> res = sess->Run(Ort::RunOptions{ nullptr }, inn, &tin, 1, outn, 1);
        g_srb_stats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t0).count();
        ++g_srb_stats.runs; g_srb_stats.tiles += (long long)m; ++g_srb_stats.hist[m]; g_srb_stats.sizes[{ h, w }] += (long long)m;
        if (res.size() != 1 || !res[0].IsTensor()) return false;
        const std::vector<int64_t> d = res[0].GetTensorTypeAndShapeInfo().GetShape();
        if (d.size() != 4 || d[0] != n_batch || d[1] != 3 || d[2] <= 0 || d[3] <= 0) return false;
        const size_t oper = (size_t)3 * d[2] * d[3];
        const float* src = res[0].GetTensorData<float>();
        for (int i = 0; i < m; ++i) {
            SrJob* j = g[first + (size_t)i];
            j->out->assign(src + (size_t)i * oper, src + (size_t)(i + 1) * oper);
            *j->oh = (int)d[2];
            *j->ow = (int)d[3];
            j->ok = true;
        }
        return true;
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: onnx-sr batched run failed: %s\n", ex.what());
        return false;
    }
}

void cs_sr_run_group(CsDncnnEntry* e, std::vector<SrJob*>& g) {
    std::lock_guard<std::mutex> lk(e->run_mu);
    const int n = (int)g.size();
    if (Ort::Session* fs = cs_dml_folded_session(e, g[0]->h, g[0]->w)) {
        for (int s = 0; s < n; s += kFoldBatch)
            cs_sr_run_chunk(e, fs, kFoldBatch, g, (size_t)s, (std::min)(kFoldBatch, n - s));
        return;
    }
    if (Ort::Session* dyn = cs_dml_dynamic_session(e)) cs_sr_run_chunk(e, dyn, n, g, 0, n);
}

bool cs_sr_run_batched(CsDncnnEntry* e, const float* rgb01, int h, int w,
    std::vector<float>& out, int& out_h, int& out_w)
{
    SrJob job{ rgb01, h, w, &out, &out_h, &out_w };
    const auto enq = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(g_srb_mu);
    g_srb_queue.push_back(&job);
    g_srb_cv.notify_all();
    while (!job.done) {
        const bool waited = std::chrono::steady_clock::now() - enq > std::chrono::milliseconds(kSrBatchWaitMs);
        if (!g_srb_busy && !g_srb_queue.empty() && ((int)g_srb_queue.size() >= kSrBatchTarget || waited)) {
            std::vector<SrJob*> batch;
            batch.swap(g_srb_queue);
            g_srb_busy = true;
            lk.unlock();
            std::map<std::pair<int, int>, std::vector<SrJob*>> by_size;
            for (SrJob* j : batch) by_size[{ j->h, j->w }].push_back(j);
            for (auto& kv : by_size)
                for (size_t s = 0; s < kv.second.size(); s += kSrBatchMax) {
                    std::vector<SrJob*> chunk(kv.second.begin() + s,
                        kv.second.begin() + (std::min)(kv.second.size(), s + (size_t)kSrBatchMax));
                    cs_sr_run_group(e, chunk);
                }
            lk.lock();
            g_srb_busy = false;
            for (SrJob* j : batch) j->done = true;
            g_srb_cv.notify_all();
        }
        else {
            g_srb_cv.wait_for(lk, std::chrono::milliseconds(2));
        }
    }
    return job.ok;
}
}  // namespace
#endif

static bool cs_onnx_sr_run_tile(CsDncnnEntry* e, const float* rgb01,
    int h, int w, std::vector<float>& out, int& out_h, int& out_w)
{
#ifdef CS_ONNX_DML
    if (e->dml) return cs_sr_run_batched(e, rgb01, h, w, out, out_h, out_w);
#endif
    int64_t shape[4] = { 1, 3, h, w };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem,
        const_cast<float*>(rgb01), (size_t)3 * h * w, shape, 4);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    std::vector<Ort::Value> res;
    try {
        std::unique_lock<std::mutex> run_lk(e->run_mu, std::defer_lock);
        if (e->dml) run_lk.lock();
        res = e->session->Run(Ort::RunOptions{ nullptr }, inn, &in, 1, outn, 1);
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: onnx-sr run failed: %s\n", ex.what());
        return false;
    }
    if (res.size() != 1 || !res[0].IsTensor()) return false;
    const std::vector<int64_t> dims = res[0].GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3) return false;
    if (dims[2] <= 0 || dims[3] <= 0) return false;
    out_h = (int)dims[2];
    out_w = (int)dims[3];
    const float* src = res[0].GetTensorData<float>();
    out.assign(src, src + (size_t)3 * out_h * out_w);
    return true;
}

// Tiled ONNX SR: RGB planar [0,1] tiles in, scale-fold clean RGB out, feathered
// overlap compositing on the HR canvas (mirrors cs_onnx_denoise_tiled). The
// scale is detected from the first run and must stay an integer multiple.
// True when the LR sub-tile lies in one segment (min label == max label). An
// empty seg_roi disables the gate, so the prior always runs.
static bool cs_seg_single_label(const cv::Mat& seg_roi, int y0, int x0, int th, int tw) {
    if (seg_roi.empty()) return false;
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(seg_roi(cv::Rect(x0, y0, tw, th)), &mn, &mx);
    return mn == mx;
}

static bool cs_onnx_sr_tiled(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out, const cv::Mat& seg_roi,
    const std::function<bool(int, int, int, int, cv::Mat&)>& single_patch)
{
    dst_bgr.release();
    scale_out = 0;
    if (src_bgr.empty() || src_bgr.type() != CV_8UC3) return false;
    const int H = src_bgr.rows, W = src_bgr.cols;
    const int T = 256, OV = 16;
    if (src_bgr.total() == 0 || H < 4 || W < 4) return false;
    CsDncnnEntry* e = cs_dncnn_entry(model_path, "SR");
    if (!e) return false;
    cv::Mat rgb;
    cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chs;
    cv::split(rgb, chs);
    std::vector<int> ys, xs;
    for (int y = 0; y < H; y += T - OV) ys.push_back(y);
    for (int x = 0; x < W; x += T - OV) xs.push_back(x);
    if (ys.back() > H - T && H > T) ys.back() = H - T;
    if (xs.back() > W - T && W > T) xs.back() = W - T;
    int scale = 0, OH = 0, OW = 0;
    size_t n_avir = 0;
    cv::Mat acc, weight;
    std::vector<float> in, out;
    for (size_t t = 0; t < ys.size() * xs.size(); ++t) {
        const int y0 = ys[t / xs.size()], x0 = xs[t % xs.size()];
        const int th = (std::min)(T, H - y0), tw = (std::min)(T, W - x0);
        int oh = 0, ow = 0;
        if (cs_seg_single_label(seg_roi, y0, x0, th, tw)) {
            ++n_avir;
            // Single-segment sub-tile: AVIR patch in place of the ONNX prior,
            // as planar RGB [0,1] like the model output.
            cv::Mat hr_bgr;
            // Caller's patch (cs-sr on the sub-tile) when given; AVIR otherwise.
            // clone(): AVIR reads rows as packed, a ROI view is not packed.
            const bool got = single_patch &&
                single_patch(y0, x0, th, tw, hr_bgr) &&
                hr_bgr.rows == 2 * th && hr_bgr.cols == 2 * tw;
            if (!got) cs_upscale_2x_avir(src_bgr(cv::Rect(x0, y0, tw, th)).clone(), hr_bgr);
            if (hr_bgr.rows != 2 * th || hr_bgr.cols != 2 * tw) return false;
            cv::Mat hr_rgb, hr_f;
            cv::cvtColor(hr_bgr, hr_rgb, cv::COLOR_BGR2RGB);
            hr_rgb.convertTo(hr_f, CV_32FC3, 1.0 / 255.0);
            std::vector<cv::Mat> planes;
            cv::split(hr_f, planes);
            oh = 2 * th; ow = 2 * tw;
            out.resize((size_t)3 * oh * ow);
            for (int c = 0; c < 3; ++c)
                std::memcpy(&out[(size_t)c * oh * ow], planes[(size_t)c].ptr<float>(0),
                    sizeof(float) * (size_t)oh * ow);
        }
        else {
            in.assign((size_t)3 * th * tw, 0.0f);
            for (int c = 0; c < 3; ++c) {
                float* d = &in[(size_t)c * th * tw];
                const int cr = c;
                for (int r = 0; r < th; ++r) {
                    const uint8_t* srow = chs[(size_t)cr].ptr<uint8_t>(y0 + r) + x0;
                    for (int k = 0; k < tw; ++k)
                        d[(size_t)r * tw + k] = srow[k] * (1.0f / 255.0f);
                }
            }
            if (!cs_onnx_sr_run_tile(e, in.data(), th, tw, out, oh, ow)) return false;
        }
        if (oh % th != 0 || ow % tw != 0 || oh / th != ow / tw) return false;
        const int s = oh / th;
        if (s < 1 || s > 8) return false;
        if (scale == 0) {
            scale = s; OH = H * scale; OW = W * scale;
            acc = cv::Mat::zeros(OH, OW, CV_32FC3);
            weight = cv::Mat::zeros(OH, OW, CV_32F);
        }
        else if (s != scale) return false;
        for (int r = 0; r < oh; ++r) {
            const int gy = y0 * scale + r;
            if (gy < 0 || gy >= OH) continue;
            cv::Vec3f* arow = acc.ptr<cv::Vec3f>(gy);
            float* wrow = weight.ptr<float>(gy);
            const float lr_y = (float)r / (float)scale;
            const int dt = (y0 == 0) ? OV : (int)lr_y;
            const int db = (y0 + th == H) ? OV : (int)((float)(th - 1) - lr_y);
            const float wy = (std::min)(1.0f, (std::min)(dt, db) / (float)OV);
            for (int k = 0; k < ow; ++k) {
                const int gx = x0 * scale + k;
                if (gx < 0 || gx >= OW) continue;
                const float lr_x = (float)k / (float)scale;
                const int dl = (x0 == 0) ? OV : (int)lr_x;
                const int dr = (x0 + tw == W) ? OV : (int)((float)(tw - 1) - lr_x);
                const float wx = (std::min)(1.0f, (std::min)(dl, dr) / (float)OV);
                const float wt = wx * wy;
                const size_t o = (size_t)r * ow + k;
                arow[gx] += cv::Vec3f(out[o], out[(size_t)oh * ow + o],
                    out[(size_t)2 * oh * ow + o]) * wt;
                wrow[gx] += wt;
            }
        }
    }
    if (scale == 0) return false;
    if (!seg_roi.empty())
        std::cout << "seg-gate: prior on " << (ys.size() * xs.size() - n_avir) << " of "
                  << (ys.size() * xs.size()) << " sub-tiles" << std::endl;
    cv::Mat clean_rgb(OH, OW, CV_32FC3);
    for (int i = 0; i < OH; ++i) {
        const cv::Vec3f* arow = acc.ptr<cv::Vec3f>(i);
        const float* wrow = weight.ptr<float>(i);
        cv::Vec3f* crow = clean_rgb.ptr<cv::Vec3f>(i);
        for (int j = 0; j < OW; ++j) {
            const float w = wrow[j] > 0.0f ? wrow[j] : 1.0f;
            crow[j] = arow[j] / w;
        }
    }
    cv::Mat clean_bgr;
    cv::cvtColor(clean_rgb, clean_bgr, cv::COLOR_RGB2BGR);
    clean_bgr.convertTo(dst_bgr, CV_8U, 255.0);
    scale_out = scale;
    return true;
}

bool cs_onnx_sr_available(const std::string& model_path) {
    if (model_path.empty()) return false;
    return cs_dncnn_entry(model_path, "SR") != nullptr;
}

void cs_onnx_sr_prefetch(const std::string& model_path, int h, int w) {
#ifdef CS_ONNX_DML
    CsDncnnEntry* e = model_path.empty() ? nullptr : cs_dncnn_entry(model_path, "SR");
    if (!e || !e->meta_only) return;
    // Futures of std::async block in their destructor, so a prefetch that is
    // still running at exit is joined, never abandoned.
    static std::mutex m;
    static std::vector<std::future<void>> pending;
    std::lock_guard<std::mutex> lk(m);
    pending.push_back(std::async(std::launch::async, [e, h, w] { cs_dml_folded_session(e, h, w); }));
#else
    (void)model_path; (void)h; (void)w;
#endif
}

bool cs_onnx_sr_upscale_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out, const cv::Mat& seg_roi,
    const std::function<bool(int, int, int, int, cv::Mat&)>& single_patch)
{
    return cs_onnx_sr_tiled(src_bgr, dst_bgr, model_path, scale_out, seg_roi, single_patch);
}
bool cs_onnx_api_ok() { return cs_ort_api_available(); }
#else
bool cs_onnx_api_ok() { return false; }
bool cs_dncnn_available(const std::string& model_path) {
    (void)model_path;
    return false;
}
bool cs_dncnn_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    return false;
}
bool cs_onnx_available(const std::string& model_path, int kind) {
    (void)model_path;
    (void)kind;
    return false;
}
bool cs_onnx_denoise_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int kind)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    (void)kind;
    return false;
}
bool cs_onnx_sr_available(const std::string& model_path) {
    (void)model_path;
    return false;
}
void cs_onnx_sr_prefetch(const std::string& model_path, int h, int w) {
    (void)model_path; (void)h; (void)w;
}
bool cs_onnx_sr_upscale_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out, const cv::Mat& seg_roi,
    const std::function<bool(int, int, int, int, cv::Mat&)>& single_patch)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    (void)seg_roi;
    (void)single_patch;
    scale_out = 0;
    return false;
}
#endif
