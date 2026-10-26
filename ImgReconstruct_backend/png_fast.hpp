#pragma once
// Parallel PNG writer for the decoder's final 8-bit BGR image. Needs a zlib, so only the GPU build
// compiles it (build_gpu.bat links AOCL-Compression's and defines CS_FAST_PNG for the callers).
#include <opencv2/core.hpp>
#include <string>

/** @brief Writes img (CV_8UC3, BGR) to path as an RGB PNG with OpenCV's default encoding choices
(SUB filter on every row, deflate level 1, RLE strategy), but deflates row strips on all threads:
each strip is a byte-aligned piece of one zlib stream in its own IDAT chunk, and the Adler-32 of the
strips is combined. Lossless; any PNG decoder reads it. Returns false (nothing usable written) when
img is not CV_8UC3 or a zlib/file call fails, so the caller can fall back to cv::imwrite. */
bool cs_write_png_parallel(const std::string& path, const cv::Mat& img);

/** @brief cv::imwrite, except that .png paths with a CV_8UC3 image go through
cs_write_png_parallel (falling back to cv::imwrite if it fails). */
bool cs_write_image(const std::string& path, const cv::Mat& img);
