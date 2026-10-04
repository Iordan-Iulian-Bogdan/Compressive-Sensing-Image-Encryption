#include "cs_wavelet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kCdf97A = -1.586134342f;
constexpr float kCdf97B = -0.052980118f;
constexpr float kCdf97G = 0.882911075f;
constexpr float kCdf97D = 0.443506852f;
constexpr float kCdf97K = 1.230174105f;

int ExtensionIndex(int index, int length) {
  if (length <= 1) return 0;
  while (index < 0 || index >= length) {
    if (index < 0) {
      index = -index;
    } else {
      index = 2 * (length - 1) - index;
    }
  }
  return index;
}

void LiftingStep(const float* src, float* dst, int length, float coefficient,
                 bool odd) {
  for (int j = 0; j < length; ++j) {
    float value = src[j];
    if (((j & 1) != 0) == odd) {
      value += coefficient *
               (src[ExtensionIndex(j - 1, length)] +
                src[ExtensionIndex(j + 1, length)]);
    }
    dst[j] = value;
  }
}

void Forward1D(float* values, int length, float* temp0, float* temp1) {
  if (length <= 1) return;
  LiftingStep(values, temp0, length, kCdf97A, true);
  LiftingStep(temp0, temp1, length, kCdf97B, false);
  LiftingStep(temp1, temp0, length, kCdf97G, true);
  LiftingStep(temp0, temp1, length, kCdf97D, false);
  const int low_count = (length + 1) / 2;
  int low_index = 0;
  int high_index = low_count;
  for (int j = 0; j < length; ++j) {
    if ((j & 1) == 0) {
      values[low_index++] = temp1[j] / kCdf97K;
    } else {
      values[high_index++] = temp1[j] * kCdf97K;
    }
  }
}

void Inverse1D(float* values, int length, float* temp0, float* temp1) {
  if (length <= 1) return;
  const int low_count = (length + 1) / 2;
  for (int j = 0; j < length; ++j) {
    temp0[j] = (j & 1) == 0 ? values[j / 2] * kCdf97K
                             : values[low_count + j / 2] / kCdf97K;
  }
  LiftingStep(temp0, temp1, length, -kCdf97D, false);
  LiftingStep(temp1, temp0, length, -kCdf97G, true);
  LiftingStep(temp0, temp1, length, -kCdf97B, false);
  LiftingStep(temp1, values, length, -kCdf97A, true);
}

void AdjointLiftingStep(const float* input, float* output, int length,
                        float coefficient, bool odd) {
  for (int j = 0; j < length; ++j) output[j] = input[j];
  for (int j = 0; j < length; ++j) {
    if (((j & 1) != 0) == odd) {
      output[ExtensionIndex(j - 1, length)] += coefficient * input[j];
      output[ExtensionIndex(j + 1, length)] += coefficient * input[j];
    }
  }
}

void SynthesisAdjoint1D(float* values, int length, float* temp0,
                        float* temp1) {
  if (length <= 1) return;
  AdjointLiftingStep(values, temp0, length, -kCdf97A, true);
  AdjointLiftingStep(temp0, temp1, length, -kCdf97B, false);
  AdjointLiftingStep(temp1, temp0, length, -kCdf97G, true);
  AdjointLiftingStep(temp0, temp1, length, -kCdf97D, false);
  const int low_count = (length + 1) / 2;
  for (int j = 0; j < length; ++j) {
    if ((j & 1) == 0) {
      values[j / 2] = temp1[j] * kCdf97K;
    } else {
      values[low_count + j / 2] = temp1[j] / kCdf97K;
    }
  }
}

void GetPyramidSizes(int rows, int cols, int levels, int* heights,
                     int* widths) {
  int height = rows;
  int width = cols;
  for (int level = 0; level < levels; ++level) {
    heights[level] = height;
    widths[level] = width;
    height = (height + 1) / 2;
    width = (width + 1) / 2;
  }
}

void FillWaveletWeights(std::vector<float>& weights, int rows, int cols,
                        int levels, float finest_band_weight) {
  weights.assign(static_cast<size_t>(rows) * cols, 1.0f);
  if (levels <= 0 || finest_band_weight <= 0.0f ||
      std::fabs(finest_band_weight - 1.0f) < 1e-9f) {
    return;
  }

  int heights[8];
  int widths[8];
  GetPyramidSizes(rows, cols, levels, heights, widths);
  for (int level = 0; level < levels; ++level) {
    const int fineness = levels - 1 - level;
    const double exponent = levels > 1
                                ? static_cast<double>(fineness) / (levels - 1)
                                : 1.0;
    const float multiplier = static_cast<float>(
        std::pow(static_cast<double>(finest_band_weight), exponent));
    const int low_height = (heights[level] + 1) / 2;
    const int low_width = (widths[level] + 1) / 2;
    for (int row = low_height; row < heights[level]; ++row) {
      for (int col = 0; col < widths[level]; ++col) {
        weights[static_cast<size_t>(row) * cols + col] = multiplier;
      }
    }
    for (int row = 0; row < low_height; ++row) {
      for (int col = low_width; col < widths[level]; ++col) {
        weights[static_cast<size_t>(row) * cols + col] = multiplier;
      }
    }
  }
}

}  // namespace

int cs_basis_from_name(const std::string& name, int& out) {
  std::string normalized;
  normalized.reserve(name.size());
  for (char value : name) {
    if (value >= 'A' && value <= 'Z') value = static_cast<char>(value - 'A' + 'a');
    normalized.push_back(value);
  }
  if (normalized == "dct" || normalized == "0") {
    out = CS_BASIS_DCT;
    return 0;
  }
  if (normalized == "wavelet" || normalized == "cdf97" ||
      normalized == "cdf-9/7" || normalized == "w97" || normalized == "1") {
    out = CS_BASIS_CDF97;
    return 0;
  }
  return -1;
}

int cs_dwt_levels(int rows, int cols) {
  int minimum_dimension = (std::min)(rows, cols);
  int levels = 0;
  while (minimum_dimension >= 16 && levels < 4) {
    minimum_dimension = (minimum_dimension + 1) / 2;
    ++levels;
  }
  return (std::max)(1, levels);
}

cs_fista_basis cs_make_basis(int rows, int cols, int basis_id, float wscale) {
  cs_fista_basis basis;
  basis.basis = basis_id == CS_BASIS_CDF97 ? CS_BASIS_CDF97 : CS_BASIS_DCT;
  basis.levels = basis.basis == CS_BASIS_CDF97 ? cs_dwt_levels(rows, cols) : 0;
  basis.lip = 2.0f;
  if (basis.basis == CS_BASIS_CDF97) {
    FillWaveletWeights(basis.wscale, rows, cols, basis.levels, wscale);
  } else {
    basis.wscale.assign(static_cast<size_t>(rows) * cols, 1.0f);
  }
  return basis;
}

void cs_dwt_forward(float* data, int rows, int cols, int levels) {
  if (levels <= 0 || rows <= 0 || cols <= 0) return;
  levels = (std::min)(levels, 8);
  const int max_dimension = (std::max)(rows, cols);
  std::vector<float> scratch(static_cast<size_t>(3) * max_dimension);
  float* temp0 = scratch.data();
  float* temp1 = scratch.data() + max_dimension;
  float* column = scratch.data() + 2 * static_cast<size_t>(max_dimension);
  int height = rows;
  int width = cols;
  for (int level = 0; level < levels; ++level) {
    for (int row = 0; row < height; ++row) {
      Forward1D(data + static_cast<size_t>(row) * cols, width, temp0, temp1);
    }
    for (int col = 0; col < width; ++col) {
      for (int row = 0; row < height; ++row) {
        column[row] = data[static_cast<size_t>(row) * cols + col];
      }
      Forward1D(column, height, temp0, temp1);
      for (int row = 0; row < height; ++row) {
        data[static_cast<size_t>(row) * cols + col] = column[row];
      }
    }
    height = (height + 1) / 2;
    width = (width + 1) / 2;
  }
}

void cs_dwt_inverse(float* data, int rows, int cols, int levels) {
  if (levels <= 0 || rows <= 0 || cols <= 0) return;
  levels = (std::min)(levels, 8);
  int heights[8];
  int widths[8];
  GetPyramidSizes(rows, cols, levels, heights, widths);
  const int max_dimension = (std::max)(rows, cols);
  std::vector<float> scratch(static_cast<size_t>(3) * max_dimension);
  float* temp0 = scratch.data();
  float* temp1 = scratch.data() + max_dimension;
  float* column = scratch.data() + 2 * static_cast<size_t>(max_dimension);
  for (int level = levels - 1; level >= 0; --level) {
    const int height = heights[level];
    const int width = widths[level];
    for (int col = 0; col < width; ++col) {
      for (int row = 0; row < height; ++row) {
        column[row] = data[static_cast<size_t>(row) * cols + col];
      }
      Inverse1D(column, height, temp0, temp1);
      for (int row = 0; row < height; ++row) {
        data[static_cast<size_t>(row) * cols + col] = column[row];
      }
    }
    for (int row = 0; row < height; ++row) {
      Inverse1D(data + static_cast<size_t>(row) * cols, width, temp0, temp1);
    }
  }
}

void cs_dwt_synth_adjoint(float* data, int rows, int cols, int levels) {
  if (levels <= 0 || rows <= 0 || cols <= 0) return;
  levels = (std::min)(levels, 8);
  int heights[8];
  int widths[8];
  GetPyramidSizes(rows, cols, levels, heights, widths);
  const int max_dimension = (std::max)(rows, cols);
  std::vector<float> scratch(static_cast<size_t>(3) * max_dimension);
  float* temp0 = scratch.data();
  float* temp1 = scratch.data() + max_dimension;
  float* column = scratch.data() + 2 * static_cast<size_t>(max_dimension);
  for (int level = 0; level < levels; ++level) {
    const int height = heights[level];
    const int width = widths[level];
    for (int row = 0; row < height; ++row) {
      SynthesisAdjoint1D(data + static_cast<size_t>(row) * cols, width, temp0,
                         temp1);
    }
    for (int col = 0; col < width; ++col) {
      for (int row = 0; row < height; ++row) {
        column[row] = data[static_cast<size_t>(row) * cols + col];
      }
      SynthesisAdjoint1D(column, height, temp0, temp1);
      for (int row = 0; row < height; ++row) {
        data[static_cast<size_t>(row) * cols + col] = column[row];
      }
    }
  }
}
