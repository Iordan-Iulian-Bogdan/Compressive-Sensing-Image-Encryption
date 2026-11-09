// Test suite for the CS image encryption backend.
// Covers: crypto primitives (KDF determinism, seal/verify, tamper detection),
// index shuffle determinism, tiling helpers, and a full encrypt->decrypt
// roundtrip scored by PSNR + SSIM.
#include "test_common.hpp"

int g_failures = 0;

int main(int argc, char* argv[]) {
    // Large photos trip CL_MEM_OBJECT_ALLOCATION_FAILURE on this host's
    // OpenCL runtime during heavy GEMM; CPU path is deterministic for tests.
    cv::ocl::setUseOpenCL(false);
    if (argc == 2 && std::string(argv[1]) == "--y-confidence-only") {
        return g_failures ? 1 : 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--grain-only") {
        return g_failures ? 1 : 0;
    }

    test_crypto();
    test_shuffle();
    test_tile_helpers();
    test_roundtrip();
    test_odd_size_roundtrip();
    test_hr_detail();
    test_hr_sr_anchors();
    test_enh_planes();
    test_sample_bits();
    test_seg_auto();
    test_encoder_geometry();
    test_auto_reg_params();
    test_gpu_fista_agreement();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}

