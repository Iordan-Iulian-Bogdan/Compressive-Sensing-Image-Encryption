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
  // DCT block size (--dct-block): 0 = legacy whole-plane DCT, >0 =
  // independent orthonormal DCT per BxB block (edge blocks clipped).
  int dct_block = 0;
};

// Process-wide default DCT block size for cs_make_basis (decrypt/solver
// only; the container format is unchanged). Set once from --dct-block.
int cs_default_dct_block();
void cs_set_default_dct_block(int block);

// Process-wide power-of-two-canvas mode (--pad-radix): the per-channel DCT
// tile solver pads the solve plane to the next power of two per dimension,
// transforms there (cv::dct fast radix path) and crops back. Decrypt/solver
// only, CPU only; the container format is unchanged. Set once from
// --pad-radix (or the CS_PAD_RADIX env override for experiments).
int cs_default_pad_radix();
void cs_set_default_pad_radix(int on);

int cs_basis_from_name(const std::string& name, int& out);
int cs_dwt_levels(int rows, int cols);
cs_fista_basis cs_make_basis(int rows, int cols, int basis_id,
                             float wscale = 2.0f, int dct_block = -1);
void cs_dwt_forward(float* data, int rows, int cols, int levels);
void cs_dwt_inverse(float* data, int rows, int cols, int levels);
void cs_dwt_synth_adjoint(float* data, int rows, int cols, int levels);

#endif  // IMGRECONSTRUCT_BACKEND_CS_WAVELET_HPP_
