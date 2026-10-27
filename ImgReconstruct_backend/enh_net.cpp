// Restoration CNN (--enhance); see enh_net.hpp.
#include "enh_net.hpp"
#include "helper_functions.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <onnxruntime_cxx_api.h>
#ifdef CS_ONNX_DML
#include <dml_provider_factory.h>   // DirectML ORT build (see build_gpu.bat)
#endif
#endif
#ifdef CS_GPU_TILES
#include "gpu_enh.h"
#endif

namespace {
constexpr float kResScale = 64.0f;    // anchor residual scale (enh_inputs.RES_SCALE)
constexpr float kSresScale = 8.0f;    // LR sample residual scale (enh_inputs.SRES_SCALE)
constexpr double kBlurLr = 4.0;       // density blur on the LR grid (enh_inputs.BLUR_LR)
constexpr int kTile = 1024, kPad = 64;   // tiles + context (train_enh_fresh.enhance_full)

// BT.601 luma of an 8-bit BGR image, float (enh_inputs.luma).
cv::Mat luma32(const cv::Mat& bgr) {
    cv::Mat y(bgr.size(), CV_32F);
    cv::parallel_for_(cv::Range(0, bgr.rows), [&](const cv::Range& rr) {
        for (int r = rr.start; r < rr.end; ++r) {
            const cv::Vec3b* s = bgr.ptr<cv::Vec3b>(r);
            float* d = y.ptr<float>(r);
            for (int c = 0; c < bgr.cols; ++c)
                d[c] = 0.299f * s[c][2] + 0.587f * s[c][1] + 0.114f * s[c][0];
        }
    });
    return y;
}

float clip1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }
}  // namespace

void cs_enh_planes(const cv::Mat& bgr, const CsEnhInputs& in, int n, std::vector<cv::Mat>& P) {
    const int H = bgr.rows, W = bgr.cols;
    P.assign((size_t)n, cv::Mat());
    for (auto& p : P) p = cv::Mat::zeros(H, W, CV_32F);
    const cv::Mat lum = luma32(bgr);
    const bool have_mask = in.y_mask && !in.y_mask->empty();

    // 0: local sampling density, centred
    if (have_mask) {
        cv::Mat m255 = (*in.y_mask > 0), b, up;          // 0 / 255
        cv::GaussianBlur(m255, b, cv::Size(0, 0), kBlurLr, kBlurLr, cv::BORDER_REPLICATE);   // PIL extends edges
        cv::resize(b, up, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        up.convertTo(P[0], CV_32F, 1.0 / 255.0, -0.5);
    }
    else P[0].setTo(-0.5f);
    // 1, 2: anchor indicator and residual
    if (in.anchors) {
        float* ind = P[1].ptr<float>(0); float* res = P[2].ptr<float>(0);
        const float* L = lum.ptr<float>(0);
        const size_t N = (size_t)H * W;
        for (const CsHrSample& s : *in.anchors) {
            if ((size_t)s.idx >= N) continue;
            ind[s.idx] = 1.0f;
            res[s.idx] = clip1(((float)s.y - L[s.idx]) / kResScale);
        }
    }
    // 3, 4: LR sample mask and measurement residual (sample - 2x2 mean of the decoded luma), each on its 2x2 block
    if (n >= 5 && have_mask && in.y_samp && !in.y_samp->empty()) {
        const cv::Mat& M = *in.y_mask; const cv::Mat& S = *in.y_samp;
        const int h = (std::min)(M.rows, H / 2), w = (std::min)(M.cols, W / 2);
        cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& rr) {
            for (int i = rr.start; i < rr.end; ++i) {
                const uint8_t* m = M.ptr<uint8_t>(i); const uint8_t* s = S.ptr<uint8_t>(i);
                const float* l0 = lum.ptr<float>(2 * i); const float* l1 = lum.ptr<float>(2 * i + 1);
                float* k0 = P[3].ptr<float>(2 * i); float* k1 = P[3].ptr<float>(2 * i + 1);
                float* r0 = P[4].ptr<float>(2 * i); float* r1 = P[4].ptr<float>(2 * i + 1);
                for (int j = 0; j < w; ++j) {
                    if (!m[j]) continue;
                    const float box = (l0[2 * j] + l0[2 * j + 1] + l1[2 * j] + l1[2 * j + 1]) * 0.25f;
                    const float r = clip1(((float)s[j] - box) / kSresScale);
                    k0[2 * j] = k0[2 * j + 1] = k1[2 * j] = k1[2 * j + 1] = 1.0f;
                    r0[2 * j] = r0[2 * j + 1] = r1[2 * j] = r1[2 * j + 1] = r;
                }
            }
        });
    }
}

#ifdef _WIN32
namespace {
struct EnhSession {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cs-enhance" };
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
    int channels = 0;
    bool dml = false;
};
std::mutex g_enh_mu;
std::map<std::string, std::unique_ptr<EnhSession>> g_enh_cache;

EnhSession* enh_session(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_enh_mu);
    if (auto it = g_enh_cache.find(path); it != g_enh_cache.end()) return it->second.get();
    if (!cs_onnx_api_ok()) { std::fprintf(stderr, "Warning: --enhance: onnxruntime unavailable\n"); return nullptr; }
    try {
        auto e = std::make_unique<EnhSession>();
        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef CS_ONNX_DML
        if (const char* g = std::getenv("CS_GPU_ONNX"); g && g[0] == '1') {
            try {
                opt.DisableMemPattern();
                opt.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(opt, 0));
                e->dml = true;
            }
            catch (const Ort::Exception& ex) {
                std::fprintf(stderr, "Warning: DirectML unavailable (%s); --enhance runs on the CPU\n", ex.what());
            }
        }
#endif
        const std::wstring wpath(path.begin(), path.end());
        e->session = std::make_unique<Ort::Session>(e->env, wpath.c_str(), opt);
        Ort::AllocatorWithDefaultOptions alloc;
        e->in_name = e->session->GetInputNameAllocated(0, alloc).get();
        e->out_name = e->session->GetOutputNameAllocated(0, alloc).get();
        const auto ti = e->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto shp = ti.GetShape();
        if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shp.size() != 4 || (shp[1] != 6 && shp[1] != 8))
            throw Ort::Exception("expected float NCHW input with 6 or 8 channels", ORT_INVALID_ARGUMENT);
        e->channels = (int)shp[1];
        return g_enh_cache.emplace(path, std::move(e)).first->second.get();
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: --enhance model '%s' failed to load: %s\n", path.c_str(), ex.what());
        return nullptr;
    }
}

// torch 'reflect' (no edge repeat) index for i in [0, n + pad)
inline int reflect(int i, int n) { return i < n ? i : 2 * (n - 1) - i; }

void dump_planes(const char* prefix, const cv::Mat& bgr, const std::vector<cv::Mat>& P) {
    cv::imwrite(std::string(prefix) + "_in.png", bgr);
    for (size_t k = 0; k < P.size(); ++k) {
        if (FILE* f = std::fopen((std::string(prefix) + "_p" + std::to_string(k) + ".f32").c_str(), "wb")) {
            for (int r = 0; r < P[k].rows; ++r) std::fwrite(P[k].ptr<float>(r), sizeof(float), (size_t)P[k].cols, f);
            std::fclose(f);
        }
    }
}
}  // namespace

// ONNX Runtime path: CPU-built planes, tiles packed on the host, DirectML or CPU inference.
static bool enhance_onnx(cv::Mat& bgr, const std::string& model_path, const CsEnhInputs& in) {
    const auto ts = std::chrono::steady_clock::now();
    EnhSession* e = enh_session(model_path);
    if (!e) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double t_pack = 0.0, t_run = 0.0, t_unpack = 0.0;
    const int H = bgr.rows, W = bgr.cols, C = e->channels, nx = C - 3;
    std::vector<cv::Mat> P;
    cs_enh_planes(bgr, in, nx, P);
    const double t_planes = ms(t0, std::chrono::steady_clock::now());
    if (const char* d = std::getenv("CS_ENH_DUMP"); d && *d) dump_planes(d, bgr, P);   // diagnostic, not timed
    const auto t1 = std::chrono::steady_clock::now();

    cv::Mat out(H, W, CV_8UC3);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* inn[1] = { e->in_name.c_str() };
    const char* outn[1] = { e->out_name.c_str() };
    std::vector<float> buf;
    int tiles = 0;
    try {
        for (int y0 = 0; y0 < H; y0 += kTile)
            for (int x0 = 0; x0 < W; x0 += kTile) {
                const int ya = (std::max)(y0 - kPad, 0), xa = (std::max)(x0 - kPad, 0);
                const int yb = (std::min)(y0 + kTile + kPad, H), xb = (std::min)(x0 + kTile + kPad, W);
                const int th = yb - ya, tw = xb - xa, Th = th + ((4 - th % 4) % 4), Tw = tw + ((4 - tw % 4) % 4);
                const auto tp = std::chrono::steady_clock::now();
                buf.assign((size_t)C * Th * Tw, 0.0f);
                const size_t plane = (size_t)Th * Tw;
                cv::parallel_for_(cv::Range(0, Th), [&](const cv::Range& rr) {
                    for (int r = rr.start; r < rr.end; ++r) {
                        const int sr = ya + reflect(r, th);
                        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(sr);
                        for (int c = 0; c < Tw; ++c) {
                            const int sc = xa + reflect(c, tw);
                            const size_t o = (size_t)r * Tw + c;
                            buf[o] = src[sc][2] / 255.0f;               // RGB order
                            buf[plane + o] = src[sc][1] / 255.0f;
                            buf[2 * plane + o] = src[sc][0] / 255.0f;
                            for (int k = 0; k < nx; ++k) buf[(3 + k) * plane + o] = P[(size_t)k].ptr<float>(sr)[sc];
                        }
                    }
                });
                const int64_t shape[4] = { 1, C, Th, Tw };
                Ort::Value tin = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shape, 4);
                const auto tr = std::chrono::steady_clock::now();
                auto res = e->session->Run(Ort::RunOptions{ nullptr }, inn, &tin, 1, outn, 1);
                const auto tu = std::chrono::steady_clock::now();
                t_pack += ms(tp, tr); t_run += ms(tr, tu);
                const float* y = res[0].GetTensorData<float>();
                const int hh = (std::min)(kTile, H - y0), ww = (std::min)(kTile, W - x0);
                const int oy = y0 - ya, ox = x0 - xa;
                for (int r = 0; r < hh; ++r) {
                    cv::Vec3b* d = out.ptr<cv::Vec3b>(y0 + r);
                    for (int c = 0; c < ww; ++c) {
                        const size_t o = (size_t)(oy + r) * Tw + (ox + c);
                        for (int ch = 0; ch < 3; ++ch) {
                            const float v = (std::min)(1.0f, (std::max)(0.0f, y[(size_t)ch * plane + o]));
                            d[x0 + c][2 - ch] = (uint8_t)std::nearbyint(v * 255.0f);   // round half to even, as torch
                        }
                    }
                }
                t_unpack += ms(tu, std::chrono::steady_clock::now());
                ++tiles;
            }
    }
    catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "Warning: --enhance run failed (%s); image left as decoded\n", ex.what());
        return false;
    }
    bgr = out;
    const auto te = std::chrono::steady_clock::now();
    std::printf("[enhance] %s: %d extra planes, %d tiles on %s; session %.0f ms, planes %.0f, pack %.0f, run %.0f, "
        "unpack %.0f; total %.0f ms (%.0f without the session)\n", model_path.c_str(), nx, tiles, e->dml ? "DirectML" : "CPU",
        ms(ts, t0), t_planes, t_pack, t_run, t_unpack, ms(ts, te) - ms(t0, t1) + t_planes, ms(t0, te) - ms(t0, t1) + t_planes);
    return true;
}
#else
static bool enhance_onnx(cv::Mat&, const std::string&, const CsEnhInputs&) {
    std::fprintf(stderr, "Warning: --enhance needs the Windows ONNX Runtime build; image left as decoded\n");
    return false;
}
#endif

#ifdef CS_GPU_TILES
// HIP path (gpu_enh.hip): planes, network and 8-bit output on the GPU. Used when the weight file exported next to the
// model (<stem>.enh.bin, or the .enh.bin itself) loads; CS_GPU_ENH=0 forces the ONNX Runtime path.
static bool enhance_hip(cv::Mat& bgr, const std::string& model_path, const CsEnhInputs& in) {
    if (const char* g = std::getenv("CS_GPU_ENH"); g && g[0] == '0') return false;
    const bool is_bin = model_path.size() > 8 && model_path.compare(model_path.size() - 8, 8, ".enh.bin") == 0;
    const std::string wp = is_bin ? model_path : gpu_enh::weights_path(model_path);
    if (!gpu_enh::load(wp)) return false;
    static_assert(sizeof(gpu_enh::Anchor) == sizeof(CsHrSample) && offsetof(gpu_enh::Anchor, idx) == offsetof(CsHrSample, idx) &&
        offsetof(gpu_enh::Anchor, y) == offsetof(CsHrSample, y), "gpu_enh::Anchor must mirror CsHrSample");
    const auto u8 = [](const cv::Mat* m) { return m && !m->empty() && m->type() == CV_8UC1 && m->isContinuous(); };
    const bool lr = u8(in.y_mask) && (!in.y_samp || (u8(in.y_samp) && in.y_samp->size() == in.y_mask->size()));
    const cv::Mat src = bgr.isContinuous() ? bgr : bgr.clone();
    gpu_enh::Inputs gi{};
    gi.bgr = src.data; gi.H = src.rows; gi.W = src.cols;
    if (lr) {
        gi.y_mask = in.y_mask->data; gi.y_samp = in.y_samp ? in.y_samp->data : nullptr;
        gi.h = in.y_mask->rows; gi.w = in.y_mask->cols;
    }
    if (in.anchors && !in.anchors->empty()) {
        gi.anchors = reinterpret_cast<const gpu_enh::Anchor*>(in.anchors->data());
        gi.n_anchors = (int)in.anchors->size();
    }
    cv::Mat out(src.size(), CV_8UC3);
    double ms[3] = {};
    if (!gpu_enh::run(gi, out.data, ms)) {
        std::fprintf(stderr, "Warning: --enhance HIP run failed; trying ONNX Runtime\n");
        return false;
    }
    bgr = out;
    std::printf("[enhance] %s: HIP, %d extra planes; upload + planes %.0f ms, network %.0f ms, download %.0f ms\n",
        wp.c_str(), gpu_enh::extra_planes(), ms[0], ms[1], ms[2]);
    return true;
}
#endif

bool cs_enhance_net(cv::Mat& bgr, const std::string& model_path, const CsEnhInputs& in) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;
#ifdef CS_GPU_TILES
    if (enhance_hip(bgr, model_path, in)) return true;
#endif
    return enhance_onnx(bgr, model_path, in);
}
