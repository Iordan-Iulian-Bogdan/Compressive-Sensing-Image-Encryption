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
        test_y_confidence_prior();
        test_y_dual_tv_blend();
        test_y_dual_tv_color_segment();
        test_dual_tv_split();
        return g_failures ? 1 : 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--grain-only") {
        test_grain_tv();
        return g_failures ? 1 : 0;
    }

    test_crypto();
    test_shuffle();
    test_tile_helpers();
    test_roundtrip();
    test_fista_solvers();
    test_wavelet_basis();
    test_adaptive_counts();
    test_sample_split();
    test_grain_tv();
    test_adaptive_roundtrip();
    test_spectral_lod();
    test_lod_full_roundtrip();
    test_ycc420_roundtrip();
    test_ycc422_roundtrip();
    test_tsampling();
    test_twopass_roundtrip();
    test_regions_roundtrip();
    test_photo_roundtrip();
    test_photo_upscaler();
    test_cs_super_resolution();
    test_sr_direct();
    test_coupled_dict();
    test_dncnn_red();
    test_onnx_denoisers();
    test_onsager();
    test_hf_focus_roundtrip();
    test_adaptive_gamma();
    test_adaptive_cutoff();
    test_anchor_weighted_red();
    test_edge_scores();
    test_hr_detail();
    test_hr_sr_anchors();
    test_hf_weighted_spread();
    test_adaptive_tile_grid_limit();
    test_sample_bits();
    test_y_confidence_prior();
    test_y_dual_tv_blend();
    test_y_dual_tv_color_segment();
    test_dual_tv_split();
    test_seg_auto();
    test_encoder_geometry();
    test_auto_reg_params();
    test_per_tile_coef();
    test_gpu_fista_agreement();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}

