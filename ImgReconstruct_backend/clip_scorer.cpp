#include "clip_scorer.hpp"

#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>

namespace {

// Flat-JSON parser: {"token": id, ...} as in HF CLIP vocab.json
bool parse_json_int_map(const std::string& path,
                        std::unordered_map<std::string, int>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    size_t i = 0;
    auto skip_ws = [&] {
        while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i;
    };
    auto parse_string = [&](std::string& res) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i; res.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char n = s[i + 1];
                if (n == 'u' && i + 5 < s.size()) {
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i + 2 + k];
                        cp <<= 4;
                        if (h>='0'&&h<='9') cp |= unsigned(h-'0');
                        else if (h>='a'&&h<='f') cp |= unsigned(h-'a'+10);
                        else if (h>='A'&&h<='F') cp |= unsigned(h-'A'+10);
                        else return false;
                    }
                    if (cp < 0x80) res += char(cp);
                    else if (cp < 0x800) {
                        res += char(0xC0 | (cp >> 6));
                        res += char(0x80 | (cp & 0x3F));
                    } else {
                        res += char(0xE0 | (cp >> 12));
                        res += char(0x80 | ((cp >> 6) & 0x3F));
                        res += char(0x80 | (cp & 0x3F));
                    }
                    i += 6;
                } else { i += 2; res += n; }
            } else res += s[i++];
        }
        if (i >= s.size()) return false;
        ++i; return true;
    };
    skip_ws();
    if (i >= s.size() || s[i] != '{') return false;
    ++i;
    while (true) {
        skip_ws();
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        std::string key;
        if (!parse_string(key)) return false;
        skip_ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i; skip_ws();
        bool neg = false;
        if (i < s.size() && s[i] == '-') { neg = true; ++i; }
        if (i >= s.size() || !std::isdigit((unsigned char)s[i])) return false;
        int v = 0;
        while (i < s.size() && std::isdigit((unsigned char)s[i])) {
            v = v * 10 + (s[i] - '0'); ++i;
        }
        out[key] = neg ? -v : v;
        skip_ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        return false;
    }
}

// CLIP bytes_to_unicode (openai/CLIP simple_tokenizer.py)
const unsigned int* byte_encoder() {
    static unsigned int map[256];
    static bool init = false;
    if (!init) {
        std::vector<unsigned int> bs, cs;
        for (unsigned int c = 33; c <= 126; ++c) bs.push_back(c);
        for (unsigned int c = 161; c <= 172; ++c) bs.push_back(c);
        for (unsigned int c = 174; c <= 255; ++c) bs.push_back(c);
        cs = bs;
        unsigned int n = 0;
        for (unsigned int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t k = 0; k < bs.size(); ++k) map[bs[k]] = cs[k];
        init = true;
    }
    return map;
}

std::string utf8_encode(unsigned int cp) {
    std::string o;
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) {
        o += char(0xC0 | (cp >> 6));
        o += char(0x80 | (cp & 0x3F));
    } else {
        o += char(0xE0 | (cp >> 12));
        o += char(0x80 | ((cp >> 6) & 0x3F));
        o += char(0x80 | (cp & 0x3F));
    }
    return o;
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

// Whitespace split; spaces are separators (this vocab's merges carry no
// space-marker, so word-initial pieces carry no prefix).
std::vector<std::string> pre_tokenize(const std::string& lower) {
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = lower.size();
    while (i < n) {
        while (i < n && std::isspace((unsigned char)lower[i])) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !std::isspace((unsigned char)lower[j])) ++j;
        pieces.push_back(lower.substr(i, j - i));
        i = j;
    }
    return pieces;
}

void l2_normalize(float* p, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += double(p[i]) * p[i];
    s = std::sqrt(s);
    if (s > 1e-12)
        for (int i = 0; i < n; ++i) p[i] = float(p[i] / s);
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

// GPT-2-style BPE over a char list whose last element carries the </w>
// suffix: each round merges ALL occurrences of the lowest-rank pair.
std::vector<std::string> bpe_split(std::vector<std::string> word,
                                   const std::vector<std::string>& merge_pairs) {
    // merge_pairs indexed by rank: "left\u0001right"
    if (word.size() <= 1) return word;
    auto rank_of = [&](const std::string& a, const std::string& b) -> int {
        std::string key = a + '\x01' + b;
        for (size_t r = 0; r < merge_pairs.size(); ++r)
            if (merge_pairs[r] == key) return int(r);
        return -1;
    };
    while (word.size() > 1) {
        int best_rank = -1;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            const int r = rank_of(word[i], word[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) best_rank = r;
        }
        if (best_rank < 0) break;
        std::vector<std::string> merged;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && rank_of(word[i], word[i + 1]) == best_rank) {
                merged.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                merged.push_back(word[i]);
                i += 1;
            }
        }
        word.swap(merged);
    }
    return word;
}

} // namespace

struct ClipScorer::Net {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "clip"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    std::vector<std::string> in_names_s;
    std::vector<std::string> out_names_s;
    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
};

// first 2D [1,N] float output -> out (requires N == 512)
static int extract_embed_512(std::vector<Ort::Value>& outs, float* out512, std::string& err) {
    for (auto& o : outs) {
        if (!o.IsTensor()) continue;
        auto info = o.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
        auto sh = info.GetShape();
        if (sh.size() == 2 && sh[0] == 1 && sh[1] == 512) {
            const float* d = o.GetTensorData<float>();
            for (int i = 0; i < 512; ++i) out512[i] = d[i];
            l2_normalize(out512, 512);
            return 0;
        }
    }
    err = "no [1,512] float output from CLIP model";
    return -1;
}

bool ClipScorer::load_tokenizer(const std::string& dir) {
    if (!parse_json_int_map(dir + "/vocab.json", token_to_id_)) {
        error_ = "failed to parse vocab.json";
        return false;
    }
    auto it_s = token_to_id_.find("<|startoftext|>");
    auto it_e = token_to_id_.find("<|endoftext|>");
    if (it_s == token_to_id_.end()) it_s = token_to_id_.find("startoftext");
    if (it_e == token_to_id_.end()) it_e = token_to_id_.find("endoftext");
    if (it_e == token_to_id_.end()) {
        // last resort: scan
        for (auto& kv : token_to_id_)
            if (kv.first.find("endoftext") != std::string::npos) { it_e = token_to_id_.find(kv.first); break; }
    }
    if (it_s == token_to_id_.end() || it_e == token_to_id_.end()) {
        error_ = "missing CLIP special tokens in vocab";
        return false;
    }
    sot_id_ = it_s->second;
    eot_id_ = it_e->second;

    std::ifstream min(dir + "/merges.txt");
    if (!min) { error_ = "failed to open merges.txt"; return false; }
    std::string line;
    std::getline(min, line); // #version header
    while (std::getline(min, line)) {
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merge_pairs_.push_back(line.substr(0, sp) + '\x01' + line.substr(sp + 1));
    }
    return true;
}

std::vector<int> ClipScorer::tokenize(const std::string& text) const {
    std::vector<int> ids;
    ids.push_back(sot_id_);
    std::string lower = to_lower_ascii(text);
    const unsigned int* enc = byte_encoder();
    for (const auto& piece : pre_tokenize(lower)) {
        // byte-encode, suffix last char with </w>, BPE, look up each part
        std::vector<std::string> chars;
        for (unsigned char b : piece) chars.push_back(utf8_encode(enc[b]));
        if (chars.empty()) continue;
        chars.back() += "</w>";
        for (const auto& part : bpe_split(chars, merge_pairs_)) {
            auto it = token_to_id_.find(part);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
                continue;
            }
            // fall back to single chars (all byte tokens + </w> forms in vocab)
            std::string core = part;
            bool has_w = core.size() >= 4 &&
                core.compare(core.size() - 4, 4, "</w>") == 0;
            if (has_w) core.erase(core.size() - 4);
            std::vector<std::string> chs = split_utf8(core);
            for (size_t k = 0; k < chs.size(); ++k) {
                std::string c = chs[k];
                if (has_w && k + 1 == chs.size()) c += "</w>";
                auto ci = token_to_id_.find(c);
                if (ci != token_to_id_.end()) ids.push_back(ci->second);
            }
        }
        if (int(ids.size()) >= ctx_len_ - 1) break;
    }
    ids.push_back(eot_id_);
    ids.resize(ctx_len_, 0);
    return ids;
}

bool ClipScorer::load_nets(const std::string& dir) {
#ifdef _WIN32
    // onnxruntime.dll is delay-loaded: probe it explicitly so a missing DLL
    // becomes a clean error instead of a delay-load SEH crash
    if (!LoadLibraryA("onnxruntime.dll")) {
        error_ = "onnxruntime.dll not found: copy third_party/onnxruntime/lib/onnxruntime.dll next to the exe (see scripts/download_clip.ps1)";
        return false;
    }
#endif
    try {
        // prefer Qdrant embedding exports (projection included, [1,512]);
        // fall back to Xenova submodule exports
        auto pick = [&](const char* a, const char* b) {
            std::ifstream fa(dir + "/" + a, std::ios::binary);
            if (fa) return dir + "/" + a;
            return dir + "/" + b;
        };
        const std::string vpath = pick("qdrant_vision.onnx", "vision_model.onnx");
        const std::string tpath = pick("qdrant_text.onnx", "text_model.onnx");
        vision_ = new Net();
        text_ = new Net();
        vision_->opts.SetIntraOpNumThreads(4);
        text_->opts.SetIntraOpNumThreads(4);
        const std::wstring wv(vpath.begin(), vpath.end());
        const std::wstring wt(tpath.begin(), tpath.end());
        vision_->session = Ort::Session(vision_->env, wv.c_str(), vision_->opts);
        text_->session = Ort::Session(text_->env, wt.c_str(), text_->opts);
        Ort::AllocatorWithDefaultOptions alloc;
        auto fill = [&](Net* n) {
            const size_t ni = n->session.GetInputCount();
            for (size_t i = 0; i < ni; ++i) {
                Ort::AllocatedStringPtr s = n->session.GetInputNameAllocated(i, alloc);
                n->in_names_s.emplace_back(s.get());
            }
            const size_t no = n->session.GetOutputCount();
            for (size_t i = 0; i < no; ++i) {
                Ort::AllocatedStringPtr s = n->session.GetOutputNameAllocated(i, alloc);
                n->out_names_s.emplace_back(s.get());
            }
            for (auto& s : n->in_names_s) n->in_names.push_back(s.c_str());
            for (auto& s : n->out_names_s) n->out_names.push_back(s.c_str());
        };
        fill(vision_);
        fill(text_);
    } catch (const Ort::Exception& e) {
        error_ = std::string("ONNX load failed: ") + e.what();
        return false;
    }
    return true;
}

float ClipScorer::text_embed(const std::vector<int>& ids, float* out512) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> tids(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) tids[i] = ids[i];
        std::vector<int64_t> shape{1, (int64_t)ids.size()};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, tids.data(), tids.size(), shape.data(), shape.size()));
        std::vector<const char*> in_n;
        in_n.push_back(text_->in_names[0]);
        std::vector<int64_t> mask;
        if (text_->in_names.size() > 1) {
            // attention mask: 1 through EOT inclusive, 0 for padding
            mask.assign(ids.size(), 0);
            size_t eot_pos = ids.size() - 1;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == eot_id_) { eot_pos = i; break; }
            for (size_t i = 0; i <= eot_pos; ++i) mask[i] = 1;
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem, mask.data(), mask.size(), shape.data(), shape.size()));
            in_n.push_back(text_->in_names[1]);
        }
        std::vector<Ort::Value> outs = text_->session.Run(Ort::RunOptions{nullptr},
            in_n.data(), inputs.data(), inputs.size(),
            text_->out_names.data(), text_->out_names.size());
        if (extract_embed_512(outs, out512, error_) != 0) return -1.f;
        return 0.f;
    } catch (const Ort::Exception& e) {
        error_ = std::string("text forward failed: ") + e.what();
        return -1.f;
    }
}

float ClipScorer::vision_embed(const cv::Mat& bgr, float* out512) {
    if (bgr.empty()) { error_ = "empty image"; return -1.f; }
    try {
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);
        cv::Mat f;
        resized.convertTo(f, CV_32FC3, 1.0 / 255.0);
        // CHW tensor with CLIP mean/std normalization
        static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
        static const float stdv[3] = {0.26862954f, 0.26862954f, 0.26862954f};
        std::vector<float> px(3 * 224 * 224);
        for (int y = 0; y < 224; ++y) {
            const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < 224; ++x) {
                for (int c = 0; c < 3; ++c)
                    px[c * 224 * 224 + y * 224 + x] = (row[x][c] - mean[c]) / stdv[c];
            }
        }
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape{1, 3, 224, 224};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, px.data(), px.size(), shape.data(), shape.size()));
        std::vector<Ort::Value> outs = vision_->session.Run(Ort::RunOptions{nullptr},
            vision_->in_names.data(), inputs.data(), inputs.size(),
            vision_->out_names.data(), vision_->out_names.size());
        if (extract_embed_512(outs, out512, error_) != 0) return -1.f;
        return 0.f;
    } catch (const Ort::Exception& e) {
        error_ = std::string("vision forward failed: ") + e.what();
        return -1.f;
    }
}

bool ClipScorer::load(const std::string& model_dir) {
    ready_ = false;
    error_.clear();
    std::string dir = model_dir;
    if (dir.empty()) {
        const char* env = std::getenv("CS_CLIP_DIR");
        dir = (env && *env) ? std::string(env) : std::string("models/clip");
    }
    if (!load_tokenizer(dir)) return false;
    if (!load_nets(dir)) return false;
    ready_ = true;
    return true;
}

std::string ClipScorer::debug_ids(const std::string& text) const {
    std::string o = "[merges=" + std::to_string(merge_pairs_.size()) + "] ";
    for (int id : tokenize(text)) {
        if (!o.empty() && (int)o.size() < 400) o += ' ';
        if ((int)o.size() >= 400) break;
        o += std::to_string(id);
    }
    return o;
}

float ClipScorer::score(const cv::Mat& bgr, const std::string& description) {
    if (!ready_) return -2.f;
    std::vector<int> ids = tokenize(description);
    float te[512], ve[512];
    if (text_embed(ids, te) < 0) return -1.f;
    if (vision_embed(bgr, ve) < 0) return -1.f;
    double dot = 0;
    for (int i = 0; i < 512; ++i) dot += double(te[i]) * ve[i];
    return float(dot);
}
