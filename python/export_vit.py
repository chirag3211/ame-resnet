#!/usr/bin/env python3
"""Export a ViT-B/16 with L layers (seeded random weights, randomized head) to linalg-on-tensors MLIR.
usage: export_vit.py L [OUTDIR] [--orig]
  default : Linear and attention projections use PRE-TRANSPOSED weights, so the IR has no
            linalg.transpose of a constant (those ran as scalar copies on every forward).
            The modified model is checked against the original torchvision model (same weights).
  --orig  : plain torchvision model (what produced the 858.33M / 4953.64M measurements).
Writes OUTDIR/vit.mlir, input.bin, golden_logits.bin  (OUTDIR default build_vit<L>[p])."""
import sys, os, copy
import torch, torch.nn as nn, torchvision
from torch_mlir import fx

args = [a for a in sys.argv[1:] if not a.startswith("--")]
orig = "--orig" in sys.argv
L = int(args[0]) if args else 2
out = args[1] if len(args) > 1 else f"build_vit{L}" + ("" if orig else "p")


class PLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.wt = nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        self.b = nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        s = x.shape
        y = torch.mm(x.reshape(-1, s[-1]), self.wt) + self.b
        return y.reshape(*s[:-1], -1)


class PAttn(nn.Module):
    """Same math as nn.MultiheadAttention(batch_first=True) for B=1, with pre-transposed projections."""
    def __init__(self, mha):
        super().__init__()
        self.h = mha.num_heads
        self.wqkv = nn.Parameter(mha.in_proj_weight.detach().t().contiguous(), requires_grad=False)
        self.bqkv = nn.Parameter(mha.in_proj_bias.detach().clone(), requires_grad=False)
        self.wo = nn.Parameter(mha.out_proj.weight.detach().t().contiguous(), requires_grad=False)
        self.bo = nn.Parameter(mha.out_proj.bias.detach().clone(), requires_grad=False)

    def forward(self, q, k, v, need_weights=False):
        B, S, E = q.shape
        assert B == 1
        hd = E // self.h
        qkv = torch.mm(q.reshape(S, E), self.wqkv) + self.bqkv
        qq, kk, vv = qkv.split(E, dim=1)
        qq = qq.reshape(S, self.h, hd).permute(1, 0, 2) * (hd ** -0.5)
        kk = kk.reshape(S, self.h, hd).permute(1, 2, 0)
        vv = vv.reshape(S, self.h, hd).permute(1, 0, 2)
        a = torch.softmax(torch.bmm(qq, kk), dim=-1)
        o = torch.bmm(a, vv).permute(1, 0, 2).reshape(S, E)
        return (torch.mm(o, self.wo) + self.bo).reshape(1, S, E), None


def pretranspose(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(mod, cname, PAttn(child))
            elif isinstance(child, nn.Linear):
                setattr(mod, cname, PLinear(child))
    return model


torch.manual_seed(0)
m = torchvision.models.vision_transformer.VisionTransformer(
    image_size=224, patch_size=16, num_layers=L, num_heads=12,
    hidden_dim=768, mlp_dim=3072, num_classes=1000).eval()
torch.nn.init.normal_(m.heads.head.weight, std=0.02)
torch.nn.init.normal_(m.heads.head.bias, std=0.02)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = m(x)
    if not orig:
        m = pretranspose(copy.deepcopy(m)).eval()
        y2 = m(x)
        d = float((y2 - y).abs().max())
        print("pre-transposed vs original: max abs diff", d)
        assert d < 1e-4, "pre-transposed model differs from the original"
    os.makedirs(out, exist_ok=True)
    open(f"{out}/vit.mlir", "w").write(str(fx.export_and_import(m, x, output_type="linalg-on-tensors", func_name="forward")))
x.numpy().tofile(f"{out}/input.bin"); y.numpy().tofile(f"{out}/golden_logits.bin")
print("top1", int(y.argmax()), "logit absmax", float(y.abs().max()), "->", out)