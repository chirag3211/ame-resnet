import torch, re, sys
from transformers import MoonshineConfig, MoonshineModel
from torch_mlir import fx
n = int(sys.argv[1]) if len(sys.argv) > 1 else 160000          # audio samples (16 kHz): 160000 = 10 s
try:
    cfg = MoonshineConfig(); cfg._attn_implementation = "eager"
    enc = MoonshineModel(cfg).encoder.eval()
    class W(torch.nn.Module):
        def __init__(s, e): super().__init__(); s.e = e
        def forward(s, x): return s.e(x).last_hidden_state
    for m_ in enc.modules(): m_._non_persistent_buffers_set.clear()   # torch-mlir needs buffers in the state dict
    x = torch.randn(1, n)
    y = W(enc)(x)
    t = str(fx.export_and_import(W(enc), x, output_type="linalg-on-tensors", func_name="forward"))
    print("out", list(y.shape), len(t)//1000000, "MB")
    print(sorted(set(re.findall(r'(?:linalg|math|tensor)\.[a-z_0-9]+', t))))
    print(re.findall(r'linalg\.conv_1d_ncw_fcw[^\n]{0,260}', t)[:3])
except Exception as e:
    print("FAIL", repr(e)[:500])
