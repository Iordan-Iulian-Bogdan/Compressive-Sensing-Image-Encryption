#pragma once
// Internal FISTA/DCT kernels shared by the solver (helper_functions.cpp) and the
// super-resolution backends (superres.cpp). Not part of the public backend API.
#include "helper_functions.hpp"

float cs_tv_grad_phi(const float* pix, float* tvgrad, int rows, int cols);
int cs_fista_inner_iters(int iterations, int fista_iters);
void cs_extract_channel_measurements(const cv::Mat& pixel_measurements, int channel, int m, std::vector<float>& b);
void cs_dct_forward(cv::Mat& plane, int dct_block);
void cs_dct_inverse(cv::Mat& plane, int dct_block);
void cs_fista_core_single(float* x, const float* b, const int* rix, const int* riy, int m, int rows, int cols, const float* w, float lambda, float tv_lambda, int iters, const cs_fista_basis& bx, std::vector<float>& y, std::vector<float>& x_prev, std::vector<float>& grad, std::vector<float>& pix, std::vector<float>& sc, bool use_onsager = false, const cs_meas_groups* groups = nullptr);
