#!/usr/bin/env python3
"""Moonshine ENCODER (raw audio, 16 kHz) with seeded random weights -> build_moonshine_<size>_enc<sec>s[p][w]/.
usage: export_moonshine.py SIZE [--samples 160000] [--orig] [--weights-as-args]     SIZE in tiny|base
default: Linears pre-transposed (checked against the original). Config values for base are from memory
(hidden 416, 8 layers, 8 heads, ffn 1664, head-dim padded to a multiple of 8): CHECK against the model card."""
import argparse, copy, sys, torch
from transformers import MoonshineConfig, MoonshineModel
import ame_export as X

ap = argparse.ArgumentParser()
ap.add_argument("size", nargs="?", default="tiny")
ap.add_argument("--samples", type=int, default=160000)
ap.add_argument("--orig", action="store_true")
ap.add_argument("--weights-as-args", action="store_true")
a = ap.parse_args()
torch.manual_seed(0)
kw = {}
if a.size == "base":
    kw = dict(hidden_size=416, intermediate_size=1664, encoder_num_hidden_layers=8, decoder_num_hidden_layers=8,
              encoder_num_attention_heads=8, decoder_num_attention_heads=8, encoder_num_key_value_heads=8,
              decoder_num_key_value_heads=8, pad_head_dim_to_multiple_of=8)
cfg = MoonshineConfig(**kw)
cfg._attn_implementation = "eager"
enc = MoonshineModel(cfg).encoder.eval()
for m_ in enc.modules():
    m_._non_persistent_buffers_set.clear()          # torch-mlir needs every buffer in the state dict


class Wrap(torch.nn.Module):
    def __init__(s, e): super().__init__(); s.e = e
    def forward(s, x): return s.e(x).last_hidden_state


x = torch.randn(1, a.samples)
m = Wrap(enc)
with torch.no_grad():
    y = m(x)
    if not a.orig:
        m = X.pretranspose_linears(copy.deepcopy(m)).eval()
        d_ = float((m(x) - y).abs().max())
        print("pre-transposed vs original: max abs diff", d_)
        assert d_ < 1e-4, "pre-transposed model differs"
name = f"build_moonshine_{a.size}_enc{a.samples // 16000}s" + ("" if a.orig else "p") + ("w" if a.weights_as_args else "")
X.export(name, m, (x,), y, weight_args=a.weights_as_args)
