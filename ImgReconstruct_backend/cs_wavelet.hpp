#ifndef IMGRECONSTRUCT_BACKEND_CS_WAVELET_HPP_
#define IMGRECONSTRUCT_BACKEND_CS_WAVELET_HPP_

#include <string>
#include <vector>

enum CsBasis {
  CS_BASIS_DCT = 0,
  CS_BASIS_CDF97 = 1,
};

struct cs_fista_basis {
  int basis = CS_BASIS_DCT;
  int levels = 0;
  float lip = 2.0f;
  std::vector<float> wscale;
};

int cs_basis_from_name(const std::string& name, int& out);
int cs_dwt_levels(int rows, int cols);
cs_fista_basis cs_make_basis(int rows, int cols, int basis_id,
                             float wscale = 2.0f);
void cs_dwt_forward(float* data, int rows, int cols, int levels);
void cs_dwt_inverse(float* data, int rows, int cols, int levels);
void cs_dwt_synth_adjoint(float* data, int rows, int cols, int levels);

#endif  // IMGRECONSTRUCT_BACKEND_CS_WAVELET_HPP_
