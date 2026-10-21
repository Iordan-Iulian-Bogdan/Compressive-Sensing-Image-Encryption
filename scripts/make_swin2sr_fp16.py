"""Builds the fp16 Swin2SR prior (model_fp16.onnx) next to the fp32 model.

    python scripts/make_swin2sr_fp16.py [models/swin2sr/onnx/model.onnx [out.onnx]]

Both files are kept; the decoder picks the fp16 one with
CS_GPU_ONNX_PRECISION=fp16 (DirectML only, see README).

A plain float16 conversion destroys this model (19 dB against fp32): the final
LayerNorm squares token deviations of up to ~500, i.e. 2.5e5, which overflows
fp16 (max 65504). LayerNorm is scale invariant, so the input of that last
LayerNorm is multiplied by 1/16 first; the fp32 result is unchanged (130 dB
against the original) and the fp16 model then matches fp32 at ~60-65 dB.

Needs: pip install onnx onnxruntime (onnxruntime.transformers.float16 does the
conversion).
"""
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16

FINAL_LN_MEAN = "/swin2sr/layernorm/ReduceMean"
FINAL_LN_SUB = "/swin2sr/layernorm/Sub"
PRESCALE = 1.0 / 16.0


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "swin2sr", "onnx", "model.onnx")
    root, ext = os.path.splitext(src)
    dst = sys.argv[2] if len(sys.argv) > 2 else root + "_fp16" + ext
    model = onnx.load(src)
    nodes = {n.name: n for n in model.graph.node}
    if FINAL_LN_MEAN not in nodes or FINAL_LN_SUB not in nodes:
        print("final LayerNorm nodes not found: this is not the expected Swin2SR export", file=sys.stderr)
        return 1
    mean, sub = nodes[FINAL_LN_MEAN], nodes[FINAL_LN_SUB]
    if mean.input[0] != sub.input[0]:
        print("unexpected final LayerNorm wiring", file=sys.stderr)
        return 1
    model.graph.initializer.append(numpy_helper.from_array(np.array(PRESCALE, dtype=np.float32), "ln_prescale"))
    mul = helper.make_node("Mul", [mean.input[0], "ln_prescale"], ["ln_prescaled"], name="ln_prescale_mul")
    model.graph.node.insert(list(model.graph.node).index(mean), mul)
    mean.input[0] = "ln_prescaled"
    sub.input[0] = "ln_prescaled"
    fp16 = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    onnx.save(fp16, dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB; fp32 original {os.path.getsize(src) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
