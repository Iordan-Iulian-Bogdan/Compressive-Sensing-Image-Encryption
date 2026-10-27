"""Export the Swin2SR-lightweight x2 ONNX weights to the flat file the HIP engine loads (gpu_swin.hip),
and check a reference forward pass (PyTorch, same math as the engine) against ONNX Runtime.

Usage: py -3.12 scripts/export_swin2sr_weights.py [models/swin2sr/onnx/model.onnx] [--check]

Output: <model dir>/model.swin2sr.bin, little endian:
  u32 magic 'SW2S', u32 version 1, u32 embed_dim, u32 stages, u32 depth, u32 heads, u32 window, u32 mlp_hidden
  then float32 arrays in this order (shapes in brackets, Linear weights stored [in][out]):
  mean[3]
  first_conv.w[60][3][3][3], first_conv.b[60]
  embed.proj.w[60][60], embed.proj.b[60], embed.ln.w[60], embed.ln.b[60]
  per stage s, per layer l:
    q.w[60][60] q.b[60] k.w[60][60] v.w[60][60] v.b[60]
    logit_scale[heads] (already exp(min(x, log 100)))
    rel_bias[heads][64][64] (16*sigmoid(cpb_mlp(table))[index], precomputed)
    proj.w[60][60] proj.b[60] ln_before.w[60] ln_before.b[60]
    fc1.w[60][120] fc1.b[120] fc2.w[120][60] fc2.b[60] ln_after.w[60] ln_after.b[60]
  per stage: conv.w[60][60][3][3] conv.b[60] patch.w[60][60] patch.b[60]
  final ln.w[60] ln.b[60], conv_after_body.w[60][60][3][3] .b[60], upsample.w[12][60][3][3] .b[12]
"""
import math, os, struct, sys
import numpy as np
import onnx
from onnx import numpy_helper

MEAN = np.array([0.4488, 0.4371, 0.4040], np.float32)


def load(path):
    m = onnx.load(path)
    g = m.graph
    W = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    producer = {o: n for n in g.node for o in n.output}

    def add_of(bias_name):
        for n in consumers.get(bias_name, []):
            if n.op_type == "Add":
                return n
        raise KeyError(bias_name)

    def matmul_weight_feeding(add_node):
        for i in add_node.input:
            p = producer.get(i)
            if p is not None and p.op_type == "MatMul" and p.input[1] in W:
                return p
        raise KeyError(add_node.name)

    nstage = 1 + max(int(k.split("stages.")[1].split(".")[0]) for k in W if "stages." in k)
    depth = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in W if "layers." in k)
    P = {}
    for s in range(nstage):
        for l in range(depth):
            pre = f"swin2sr.encoder.stages.{s}.layers.{l}."
            mq = matmul_weight_feeding(add_of(pre + "attention.self.query.bias"))
            mv = matmul_weight_feeding(add_of(pre + "attention.self.value.bias"))
            # key: the other bias-free MatMul on the same input as query
            mk = [n for n in consumers[mq.input[0]] if n.op_type == "MatMul" and n is not mq and n is not mv
                  and n.input[1] in W and W[n.input[1]].shape == (60, 60)]
            assert len(mk) == 1, (s, l, len(mk))
            mk = mk[0]
            c1 = matmul_weight_feeding(add_of(pre + "attention.self.continuous_position_bias_mlp.0.bias"))
            relu = [n for n in consumers[add_of(pre + "attention.self.continuous_position_bias_mlp.0.bias").output[0]] if n.op_type == "Relu"][0]
            c2 = [n for n in consumers[relu.output[0]] if n.op_type == "MatMul"][0]
            mo = matmul_weight_feeding(add_of(pre + "attention.output.dense.bias"))
            f1 = matmul_weight_feeding(add_of(pre + "intermediate.dense.bias"))
            f2 = matmul_weight_feeding(add_of(pre + "output.dense.bias"))
            P[(s, l)] = dict(
                q_w=W[mq.input[1]], q_b=W[pre + "attention.self.query.bias"],
                k_w=W[mk.input[1]],
                v_w=W[mv.input[1]], v_b=W[pre + "attention.self.value.bias"],
                logit_scale=W[pre + "attention.self.logit_scale"].reshape(-1),
                c1_w=W[c1.input[1]], c1_b=W[pre + "attention.self.continuous_position_bias_mlp.0.bias"],
                c2_w=W[c2.input[1]],
                o_w=W[mo.input[1]], o_b=W[pre + "attention.output.dense.bias"],
                lnb_w=W[pre + "layernorm_before.weight"], lnb_b=W[pre + "layernorm_before.bias"],
                f1_w=W[f1.input[1]], f1_b=W[pre + "intermediate.dense.bias"],
                f2_w=W[f2.input[1]], f2_b=W[pre + "output.dense.bias"],
                lna_w=W[pre + "layernorm_after.weight"], lna_b=W[pre + "layernorm_after.bias"])
    return W, P, nstage, depth


def rel_bias(p, win=8, heads=6):
    """16*sigmoid(cpb_mlp(relative_coords_table))[relative_position_index] -> [heads][win*win][win*win]."""
    r = np.arange(-(win - 1), win, dtype=np.float32)
    t = np.stack(np.meshgrid(r, r, indexing="ij"), -1)        # [2w-1][2w-1][2]
    t = t / (win - 1) * 8
    t = np.sign(t) * np.log2(np.abs(t) + 1.0) / math.log2(8)
    h = np.maximum(t.reshape(-1, 2) @ p["c1_w"] + p["c1_b"], 0) @ p["c2_w"]   # [(2w-1)^2][heads]
    c = np.stack(np.meshgrid(np.arange(win), np.arange(win), indexing="ij")).reshape(2, -1)
    rel = (c[:, :, None] - c[:, None, :]).transpose(1, 2, 0) + (win - 1)
    idx = rel[:, :, 0] * (2 * win - 1) + rel[:, :, 1]
    b = h[idx.reshape(-1)].reshape(win * win, win * win, heads).transpose(2, 0, 1)
    return (16.0 / (1.0 + np.exp(-b))).astype(np.float32)


def export(path, W, P, nstage, depth):
    out = os.path.splitext(path)[0] + ".swin2sr.bin"
    arrs = [MEAN, W["swin2sr.first_convolution.weight"], W["swin2sr.first_convolution.bias"],
            W["swin2sr.embeddings.patch_embeddings.projection.weight"].reshape(60, 60).T,
            W["swin2sr.embeddings.patch_embeddings.projection.bias"],
            W["swin2sr.embeddings.patch_embeddings.layernorm.weight"], W["swin2sr.embeddings.patch_embeddings.layernorm.bias"]]
    for s in range(nstage):
        for l in range(depth):
            p = P[(s, l)]
            arrs += [p["q_w"], p["q_b"], p["k_w"], p["v_w"], p["v_b"],
                     np.exp(np.minimum(p["logit_scale"], math.log(100.0))).astype(np.float32), rel_bias(p),
                     p["o_w"], p["o_b"], p["lnb_w"], p["lnb_b"], p["f1_w"], p["f1_b"], p["f2_w"], p["f2_b"],
                     p["lna_w"], p["lna_b"]]
        pre = f"swin2sr.encoder.stages.{s}."
        arrs += [W[pre + "conv.weight"], W[pre + "conv.bias"],
                 W[pre + "patch_embed.projection.weight"].reshape(60, 60).T, W[pre + "patch_embed.projection.bias"]]
    arrs += [W["swin2sr.layernorm.weight"], W["swin2sr.layernorm.bias"],
             W["swin2sr.conv_after_body.weight"], W["swin2sr.conv_after_body.bias"],
             W["upsample.conv.weight"], W["upsample.conv.bias"]]
    with open(out, "wb") as f:
        f.write(struct.pack("<8I", 0x53325753, 1, 60, nstage, depth, 6, 8, 120))
        for a in arrs:
            f.write(np.ascontiguousarray(a, np.float32).tobytes())
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    return out


def reference(x, W, P, nstage, depth):
    """PyTorch forward of the engine's math, x: [N][3][H][W] in [0,1], H and W multiples of 8."""
    import torch
    import torch.nn.functional as F
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    N, _, H, Wd = x.shape
    win, sh, heads, C = 8, 4, 6, 60
    x = (T(x) - T(MEAN).view(1, 3, 1, 1))
    emb = F.conv2d(x, T(W["swin2sr.first_convolution.weight"]), T(W["swin2sr.first_convolution.bias"]), padding=1)
    h = F.conv2d(emb, T(W["swin2sr.embeddings.patch_embeddings.projection.weight"]), T(W["swin2sr.embeddings.patch_embeddings.projection.bias"]))
    h = h.flatten(2).transpose(1, 2)
    h = F.layer_norm(h, (C,), T(W["swin2sr.embeddings.patch_embeddings.layernorm.weight"]), T(W["swin2sr.embeddings.patch_embeddings.layernorm.bias"]), 1e-5)
    # shifted-window mask (-100 where labels differ, added twice as in the reference)
    img = torch.zeros(1, H, Wd, 1)
    cnt = 0
    for hs in (slice(0, -win), slice(-win, -sh), slice(-sh, None)):
        for ws in (slice(0, -win), slice(-win, -sh), slice(-sh, None)):
            img[:, hs, ws, :] = cnt; cnt += 1
    mw = img.view(1, H // win, win, Wd // win, win, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win)
    am = mw.unsqueeze(1) - mw.unsqueeze(2)
    am = am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)
    nw = am.shape[0]
    for s in range(nstage):
        res = h
        for l in range(depth):
            p = P[(s, l)]
            shift = sh if l % 2 else 0
            sc = h
            t = h.view(N, H, Wd, C)
            if shift: t = torch.roll(t, (-shift, -shift), (1, 2))
            t = t.view(N, H // win, win, Wd // win, win, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win, C)
            q = (t @ T(p["q_w"]) + T(p["q_b"])).view(-1, 64, heads, 10).transpose(1, 2)
            k = (t @ T(p["k_w"])).view(-1, 64, heads, 10).transpose(1, 2)
            v = (t @ T(p["v_w"]) + T(p["v_b"])).view(-1, 64, heads, 10).transpose(1, 2)
            a = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            a = a * T(np.exp(np.minimum(p["logit_scale"], math.log(100.0)))).view(1, heads, 1, 1)
            a = a + T(rel_bias(p)).unsqueeze(0)
            if shift:
                a = a.view(-1, nw, heads, 64, 64) + 2 * am.unsqueeze(1).unsqueeze(0)
                a = a.view(-1, heads, 64, 64)
            a = a.softmax(-1)
            o = (a @ v).transpose(1, 2).reshape(-1, 64, C)
            o = o @ T(p["o_w"]) + T(p["o_b"])
            o = o.view(N, H // win, Wd // win, win, win, C).permute(0, 1, 3, 2, 4, 5).reshape(N, H, Wd, C)
            if shift: o = torch.roll(o, (shift, shift), (1, 2))
            o = o.reshape(N, H * Wd, C)
            h = sc + F.layer_norm(o, (C,), T(p["lnb_w"]), T(p["lnb_b"]), 1e-5)
            m = F.gelu(h @ T(p["f1_w"]) + T(p["f1_b"])) @ T(p["f2_w"]) + T(p["f2_b"])
            h = h + F.layer_norm(m, (C,), T(p["lna_w"]), T(p["lna_b"]), 1e-5)
        pre = f"swin2sr.encoder.stages.{s}."
        img_ = h.transpose(1, 2).reshape(N, C, H, Wd)
        img_ = F.conv2d(img_, T(W[pre + "conv.weight"]), T(W[pre + "conv.bias"]), padding=1)
        img_ = F.conv2d(img_, T(W[pre + "patch_embed.projection.weight"]), T(W[pre + "patch_embed.projection.bias"]))
        h = img_.flatten(2).transpose(1, 2) + res
    h = F.layer_norm(h, (C,), T(W["swin2sr.layernorm.weight"]), T(W["swin2sr.layernorm.bias"]), 1e-5)
    img_ = h.transpose(1, 2).reshape(N, C, H, Wd)
    img_ = F.conv2d(img_, T(W["swin2sr.conv_after_body.weight"]), T(W["swin2sr.conv_after_body.bias"]), padding=1) + emb
    up = F.pixel_shuffle(F.conv2d(img_, T(W["upsample.conv.weight"]), T(W["upsample.conv.bias"]), padding=1), 2)
    return (up + T(MEAN).view(1, 3, 1, 1)).numpy()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "models", "swin2sr", "onnx", "model.onnx")
    W, P, nstage, depth = load(path)
    print(f"stages {nstage}, depth {depth}")
    export(path, W, P, nstage, depth)
    if "--check" in sys.argv:
        import onnxruntime as ort
        rng = np.random.default_rng(1)
        x = rng.random((2, 3, 48, 56)).astype(np.float32)
        ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x})[0]
        mine = reference(x, W, P, nstage, depth)
        d = np.abs(ref - mine)
        print(f"reference vs ORT CPU: max |diff| {d.max():.3e}, mean {d.mean():.3e}, shape {mine.shape}")
