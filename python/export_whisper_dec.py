#!/usr/bin/env python3
"""Whisper DECODER STEP: one new token, self-attention KV cache of --past tokens, cached cross-attention K/V of the
1500-frame encoder output -> logits [1,1,vocab]. Seeded random weights. Linears pre-transposed; the K caches are
stored transposed ([H,hd,T]) so no per-step activation transpose is needed. A step module is checked numerically
against the HF decoder run on a full sequence (assert < 1e-3) before export.
usage: export_whisper_dec.py SIZE [--past 32] [--enc-len 1500]      -> build_whisper_<size>_dec<past>p/
Sizes from memory of the public configs; check against model cards."""
import argparse, copy, sys, torch, torch.nn as nn, torch.nn.functional as F
from transformers import WhisperConfig, WhisperModel
import ame_export as X

CFG = {  # d_model, enc_layers, dec_layers, heads, ffn, mel bins, vocab
    "tiny":            (384, 4, 4, 6, 1536, 80, 51865),
    "base":            (512, 6, 6, 8, 2048, 80, 51865),
    "small":           (768, 12, 12, 12, 3072, 80, 51865),
    "large-v3-turbo":  (1280, 32, 4, 20, 5120, 128, 51866),
}
ap = argparse.ArgumentParser()
ap.add_argument("size", nargs="?", default="tiny")
ap.add_argument("--past", type=int, default=32)
ap.add_argument("--enc-len", type=int, default=1500)
ap.add_argument("--weights-as-args", action="store_true", help="weights become function arguments, loaded from <dir>/weights.bin at run time")
a = ap.parse_args()
d, el, dl, h, ffn, mel, vocab = CFG[a.size]
T, TE = a.past, a.enc_len
torch.manual_seed(0)
cfg = WhisperConfig(d_model=d, encoder_layers=el, decoder_layers=dl, encoder_attention_heads=h,
                    decoder_attention_heads=h, encoder_ffn_dim=ffn, decoder_ffn_dim=ffn,
                    num_mel_bins=mel, vocab_size=vocab)
cfg._attn_implementation = "eager"
hf = WhisperModel(cfg).decoder.eval()
H, hd = h, d // h


class DecStep(nn.Module):
    def __init__(self, dec, pos):
        super().__init__()
        self.dec = dec
        self.scale = hd ** -0.5
        self.proj_t = nn.Parameter(dec.embed_tokens.weight.detach().t().contiguous(), requires_grad=False)  # [d,V]
        self.pos = nn.Parameter(dec.embed_positions.weight.detach()[pos:pos + 1].clone(), requires_grad=False)  # [1,d]

    def mha(self, m, x, Kt, V, new):
        q = (m.q_proj(x) * self.scale).reshape(1, H, hd).permute(1, 0, 2)          # [H,1,hd]
        if new:
            k = m.k_proj(x).reshape(1, H, hd).permute(1, 2, 0)                      # [H,hd,1]
            v = m.v_proj(x).reshape(1, H, hd).permute(1, 0, 2)                      # [H,1,hd]
            Kt = torch.cat([Kt, k], dim=2); V = torch.cat([V, v], dim=1)
        p = torch.softmax(torch.bmm(q, Kt), dim=-1)                                 # [H,1,T]
        return m.out_proj(torch.bmm(p, V).permute(1, 0, 2).reshape(1, d))

    def forward(self, ids, *c):
        x = F.embedding(ids, self.dec.embed_tokens.weight).reshape(1, d) + self.pos
        for i, L in enumerate(self.dec.layers):
            sk, sv, ck, cv = c[4 * i:4 * i + 4]
            x = x + self.mha(L.self_attn, L.self_attn_layer_norm(x), sk, sv, True)
            x = x + self.mha(L.encoder_attn, L.encoder_attn_layer_norm(x), ck, cv, False)
            x = x + L.fc2(F.gelu(L.fc1(L.final_layer_norm(x))))
        return torch.mm(self.dec.layer_norm(x), self.proj_t).reshape(1, 1, -1)


with torch.no_grad():
    # reference: HF decoder on a full sequence of T+1 tokens; caches for the first T tokens derived from its hidden states
    seq = torch.randint(0, vocab, (1, T + 1)); E = torch.randn(1, TE, d) * 0.5
    out = hf(input_ids=seq, encoder_hidden_states=E, output_hidden_states=True, use_cache=False)
    ref = (out.last_hidden_state[:, -1] @ hf.embed_tokens.weight.t()).reshape(1, 1, -1)
    caches = []
    for i, L in enumerate(hf.layers):
        hin = L.self_attn_layer_norm(out.hidden_states[i][:, :T])
        caches += [L.self_attn.k_proj(hin).reshape(T, H, hd).permute(1, 2, 0).contiguous(),
                   L.self_attn.v_proj(hin).reshape(T, H, hd).permute(1, 0, 2).contiguous(),
                   L.encoder_attn.k_proj(E).reshape(TE, H, hd).permute(1, 2, 0).contiguous(),
                   L.encoder_attn.v_proj(E).reshape(TE, H, hd).permute(1, 0, 2).contiguous()]
    ids = seq[:, -1:].contiguous()
    step = DecStep(X.pretranspose_linears(copy.deepcopy(hf)), T).eval()
    y = step(ids, *caches)
    err = float((y - ref).abs().max())
    print("step vs HF full-sequence decoder: max abs diff", err, " |ref|max", float(ref.abs().max()))
    assert err < 1e-3, "decoder step differs from the HF decoder"
X.export(f"build_whisper_{a.size}_dec{T}p" + ("w" if a.weights_as_args else ""), step, (ids, *caches), y, weight_args=a.weights_as_args)
