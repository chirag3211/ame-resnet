#!/usr/bin/env python3
"""Whisper ENCODER (seeded random weights, 30 s of mel = 3000 frames) -> build_whisper_<size>_enc[p]/.
usage: export_whisper.py SIZE [--orig] [--frames N]   SIZE in tiny|base|small|large-v3-turbo
--frames N: mel frames (default 3000 = 30 s); the encoder then has N/2 positions (reduced-length test if N < 3000)
default: Linear layers pre-transposed (checked against the original model); --orig keeps nn.Linear.
Sizes below are from memory of the public configs; check against the model cards."""
import sys, copy, torch
from transformers import WhisperConfig, WhisperModel
import ame_export as X

CFG = {  # d_model, enc_layers, dec_layers, heads, ffn, mel bins, vocab
    "tiny":            (384, 4, 4, 6, 1536, 80, 51865),
    "base":            (512, 6, 6, 8, 2048, 80, 51865),
    "small":           (768, 12, 12, 12, 3072, 80, 51865),
    "large-v3-turbo":  (1280, 32, 4, 20, 5120, 128, 51866),
}
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("size", nargs="?", default="tiny")
ap.add_argument("--orig", action="store_true")
ap.add_argument("--frames", type=int, default=3000)
ap.add_argument("--weights-as-args", action="store_true", help="weights become function arguments, loaded from <dir>/weights.bin at run time")
a = ap.parse_args()
size, orig, frames = a.size, a.orig, a.frames
assert frames % 2 == 0
d, el, dl, h, ffn, mel, vocab = CFG[size]
torch.manual_seed(0)
cfg = WhisperConfig(d_model=d, encoder_layers=el, decoder_layers=dl, encoder_attention_heads=h,
                    decoder_attention_heads=h, encoder_ffn_dim=ffn, decoder_ffn_dim=ffn,
                    num_mel_bins=mel, vocab_size=vocab, max_source_positions=frames // 2)
cfg._attn_implementation = "eager"
enc = WhisperModel(cfg).encoder.eval()


class Wrap(torch.nn.Module):
    def __init__(s, e): super().__init__(); s.e = e
    def forward(s, x): return s.e(x).last_hidden_state


x = torch.randn(1, mel, frames)
m = Wrap(enc)
with torch.no_grad():
    y = m(x)
    if not orig:
        m = X.pretranspose_linears(copy.deepcopy(m)).eval()
        d_ = float((m(x) - y).abs().max())
        print("pre-transposed vs original: max abs diff", d_)
        assert d_ < 1e-4, "pre-transposed model differs"
X.export(f"build_whisper_{size}_enc" + ("" if frames == 3000 else str(frames)) + ("" if orig else "p") + ("w" if a.weights_as_args else ""), m, (x,), y, weight_args=a.weights_as_args)
