#pragma once
// Collects PnP-ADMM tile jobs from concurrent decrypt threads into batches for
// the GPU (gpu_admm.h). Plain C++; built only in the GPU build (-DCS_GPU_TILES).
#include "gpu_admm.h"

/** @brief True when GPU ADMM is requested (env CS_GPU_ADMM=1) and a device is present. */
bool cs_gpu_admm_enabled();

/** @brief Worker-thread count for a wavefront of num_tiles tiles. Workers wait
inside cs_gpu_admm_run, so with GPU ADMM on one thread per tile of a wave (up to
64) is used to fill the batches; otherwise num_threads is returned unchanged. */
int cs_gpu_admm_threads(int num_tiles, int num_threads);

/** @brief Runs one tile's ADMM loop on the GPU. Blocks until the batch that
holds the job has finished; the batch closes when every worker thread has a job
waiting or after a few milliseconds. On success p.z holds the final HR plane.
Returns false when the tile or the GPU call failed: the caller then runs the
CPU loop (p.z is untouched in that case). */
bool cs_gpu_admm_run(const gpu_admm::Problem& p, const gpu_admm::Params& prm);
