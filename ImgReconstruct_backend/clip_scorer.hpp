// CLIP zero-shot scoring for restart selection: encode a candidate image and
// a text description into the shared CLIP embedding space (OpenCV DNN, ONNX)
// and return their cosine similarity. Forward-only — no gradients, so it can
// rank decrypt restarts without touching the L-BFGS objective.
#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class ClipScorer {
public:
    // Loads vision_model.onnx, text_model.onnx, vocab.json and merges.txt
    // from model_dir (empty -> "models/clip" or $CS_CLIP_DIR).
    bool load(const std::string& model_dir);

    bool ready() const { return ready_; }

    // Cosine similarity in [-1, 1] between the BGR image and description.
    // Returns -2 if not ready, -1 on a failed forward pass.
    float score(const cv::Mat& bgr, const std::string& description);

    const std::string& error() const { return error_; }

    // debug: space-separated token ids for a description (tokenizer check)
    std::string debug_ids(const std::string& text) const;

private:
    bool load_tokenizer(const std::string& dir);
    bool load_nets(const std::string& dir);
    std::vector<int> tokenize(const std::string& text) const;
    float text_embed(const std::vector<int>& ids, float* out512);
    float vision_embed(const cv::Mat& bgr, float* out512);

    struct Net;
    Net* vision_ = nullptr;
    Net* text_ = nullptr;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> merge_pairs_;
    bool ready_ = false;
    std::string error_;
    int sot_id_ = 49406;
    int eot_id_ = 49407;
    int ctx_len_ = 77;
};
