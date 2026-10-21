// Batching front end of the GPU ADMM stage (see gpu_admm_broker.hpp).
#include "gpu_admm_broker.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

namespace {

struct Job {
    gpu_admm::Problem p;
    gpu_admm::Params prm;
    bool done = false, ok = false;
};

// A job that finds no full batch waits this long for others before it closes
// the batch itself (tail waves have fewer tiles than worker threads).
const int kMaxWaitMs = [] {
    const char* e = std::getenv("CS_GPU_ADMM_WAIT_MS");
    return e ? std::atoi(e) : 30;
}();

std::mutex g_mu;
std::condition_variable g_cv;
std::vector<Job*> g_queue;
bool g_busy = false;
std::atomic<int> g_target{ omp_get_max_threads() };   // batch closes at this many waiting jobs

struct Stats {
    std::atomic<long long> tiles{0}, batches{0}, failed{0};
    std::atomic<double> gpu_ms{0.0};
    ~Stats() {
        if (batches.load() > 0)
            std::fprintf(stderr, "[gpu-admm] %lld tiles in %lld batches (%.1f avg), gpu %.1f s, %lld fell back to CPU\n",
                (long long)tiles.load(), (long long)batches.load(),
                (double)tiles.load() / (double)batches.load(), gpu_ms.load() * 1e-3, (long long)failed.load());
    }
} g_stats;

void run_batch(const std::vector<Job*>& batch) {
    using Key = std::tuple<int, int, float, float, float>;
    std::map<Key, std::vector<Job*>> groups;
    for (Job* j : batch)
        groups[Key(j->prm.admm, j->prm.inner, j->prm.rho, j->prm.lip, j->prm.wscale)].push_back(j);
    for (auto& g : groups) {
        std::vector<gpu_admm::Problem> probs;
        probs.reserve(g.second.size());
        for (Job* j : g.second) probs.push_back(j->p);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = gpu_admm::solve_pnp_admm(probs, g.second.front()->prm);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        g_stats.gpu_ms.store(g_stats.gpu_ms.load() + ms);   // leader-only writer
        ++g_stats.batches;
        g_stats.tiles += (long long)probs.size();
        if (!ok) g_stats.failed += (long long)probs.size();
        for (Job* j : g.second) j->ok = ok;
    }
}

}  // namespace

bool cs_gpu_admm_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("CS_GPU_ADMM");
        return e && e[0] == '1' && gpu_admm::available();
    }();
    return on;
}

int cs_gpu_admm_threads(int num_tiles, int num_threads) {
    // Also raised for the batched DirectML prior (CS_GPU_ONNX=1): its workers wait too.
    const char* onnx = std::getenv("CS_GPU_ONNX");
    if (!cs_gpu_admm_enabled() && !(onnx && onnx[0] == '1')) return num_threads;
    int cap = 64;
    if (const char* w = std::getenv("CS_GPU_WORKERS")) cap = std::atoi(w);   // tuning override
    const int want = (std::max)(num_threads, (std::min)(num_tiles, cap));
    g_target = want;
    return want;
}

bool cs_gpu_admm_run(const gpu_admm::Problem& p, const gpu_admm::Params& prm) {
    if (!gpu_admm::supported(p.rows, p.cols)) return false;
    const int target = g_target;
    Job job;
    job.p = p;
    job.prm = prm;
    const auto enq = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(g_mu);
    g_queue.push_back(&job);
    g_cv.notify_all();
    while (!job.done) {
        const bool waited = std::chrono::steady_clock::now() - enq > std::chrono::milliseconds(kMaxWaitMs);
        if (!g_busy && !g_queue.empty() && ((int)g_queue.size() >= target || waited)) {
            std::vector<Job*> batch;
            batch.swap(g_queue);
            g_busy = true;
            lk.unlock();
            run_batch(batch);
            lk.lock();
            g_busy = false;
            for (Job* j : batch) j->done = true;
            g_cv.notify_all();
        } else {
            g_cv.wait_for(lk, std::chrono::milliseconds(2));
        }
    }
    return job.ok;
}
