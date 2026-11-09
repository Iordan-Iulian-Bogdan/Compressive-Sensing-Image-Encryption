#ifndef IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
#define IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_

#include <string>

// Decode options of the 2x stage (cs-sr with PnP-ADMM, cs_sr_upscale_tile_2x in helper_functions.hpp) and the
// restoration nets after it.
struct CsPhotoUpscalerOptions {
  // Tile overlap of the automatic decrypt grid (the encoder's anchor pass: 8).
  int auto_overlap = 24;
  // ADMM outer iterations (--sr-pnp-iters, clamped [1,50]).
  int sr_pnp_iters = 8;
  // HR-anchor data-term weight for the cs-sr solve (0 = off: post-hoc
  // replacement only). Anchors are exact native luma, so 1.0 pulls each
  // anchor pixel at full weight beside the LR samples.
  float sr_hr_weight = 1.0f;
  // Restoration CNN (--enhance <model.onnx>, empty = off): same-size U-Net on
  // the finished ycc decode, fed the decoder-side planes (sample density,
  // anchors, LR samples) it was trained with; see enh_net.hpp. Runs before
  // the post-hoc anchor replacement.
  std::string enhance_model;
  // Second unrolled stage (--enhance2 <model.onnx>, decrypt only, empty =
  // off): the same kind of net run again on the projected output of the
  // first, its planes rebuilt against that output, then projected again.
  std::string enhance2_model;
  // Low-ratio first stage (--enhance-low <model.onnx>, decrypt only, empty =
  // off): replaces enhance_model on containers with few luma sample bits per
  // output pixel (cs_enhance_pick in image_decryption.cpp).
  std::string enhance_low_model;
};

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
