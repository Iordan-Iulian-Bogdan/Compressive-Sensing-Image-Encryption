// ONNX Runtime denoisers and SR priors for the CS backend: DnCNN, SCUNet and
// Restormer denoisers plus the ONNX 2x SR prior. Moved out of helper_functions.cpp;
// the public entry points are declared in helper_functions.hpp.
#include "helper_functions.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <onnxruntime_cxx_api.h>
#endif

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
};
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
        const std::wstring wpath(path.begin(), path.end());
        e->session = std::make_unique<Ort::Session>(e->env, wpath.c_str(), opt);
        Ort::AllocatorWithDefaultOptions alloc;
        e->in_name = e->session->GetInputNameAllocated(0, alloc).get();
        e->out_name = e->session->GetOutputNameAllocated(0, alloc).get();
        auto ti = e->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            ti.GetDimensionsCount() != 4) {
            throw Ort::Exception("dncnn: expected float NCHW input", ORT_INVALID_ARGUMENT);
        }
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
static bool cs_onnx_sr_run_tile(CsDncnnEntry* e, const float* rgb01,
    int h, int w, std::vector<float>& out, int& out_h, int& out_w)
{
    int64_t shape[4] = { 1, 3, h, w };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem,
        const_cast<float*>(rgb01), (size_t)3 * h * w, shape, 4);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    std::vector<Ort::Value> res;
    try {
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
static bool cs_onnx_sr_tiled(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out)
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
    cv::Mat acc, weight;
    std::vector<float> in, out;
    for (size_t t = 0; t < ys.size() * xs.size(); ++t) {
        const int y0 = ys[t / xs.size()], x0 = xs[t % xs.size()];
        const int th = (std::min)(T, H - y0), tw = (std::min)(T, W - x0);
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
        int oh = 0, ow = 0;
        if (!cs_onnx_sr_run_tile(e, in.data(), th, tw, out, oh, ow)) return false;
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

bool cs_onnx_sr_upscale_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out)
{
    return cs_onnx_sr_tiled(src_bgr, dst_bgr, model_path, scale_out);
}
#else
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
bool cs_onnx_sr_upscale_bgr(const cv::Mat& src_bgr, cv::Mat& dst_bgr,
    const std::string& model_path, int& scale_out)
{
    (void)src_bgr;
    dst_bgr.release();
    (void)model_path;
    scale_out = 0;
    return false;
}
#endif
