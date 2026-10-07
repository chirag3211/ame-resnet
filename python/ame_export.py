"""Shared helpers for exporting HF/torch models to linalg-on-tensors for the AME pipeline."""
import copy, json, os
import torch, torch.nn as nn


class PLinear(nn.Module):
    """nn.Linear with the weight stored pre-transposed ([in,out]) so the IR has no transpose of a constant."""
    def __init__(self, lin):
        super().__init__()
        self.wt = nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        self.b = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        s = x.shape
        y = torch.mm(x.reshape(-1, s[-1]), self.wt)
        if self.b is not None:
            y = y + self.b
        return y.reshape(*s[:-1], -1)


def pretranspose_linears(model):
    """Replace every nn.Linear in `model` (recursively) by PLinear. Returns the model."""
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, name, PLinear(child))
    return model


class WeightArgs(nn.Module):
    """Wrap `model` so every parameter and buffer becomes a trailing function argument (weights as arguments):
    forward(*inputs, *weights) = functional_call(model, weights, inputs). The MLIR then holds no weight constants."""
    def __init__(self, model, n_inputs):
        super().__init__()
        self.model, self.n = model, n_inputs
        self.names = [k for k, _ in model.named_parameters()] + [k for k, _ in model.named_buffers()]
        self.tensors = [v.detach() for _, v in model.named_parameters()] + [v.detach() for _, v in model.named_buffers()]

    def forward(self, *args):
        ins, ws = args[:self.n], args[self.n:]
        return torch.func.functional_call(self.model, dict(zip(self.names, ws)), tuple(ins))


def export(outdir, model, inputs, golden, check_against=None, tol=1e-4, weight_args=False):
    """Write outdir/{resnet18.mlir (symlink name used by the Makefile), input.bin, golden_logits.bin, model.json}."""
    from torch_mlir import fx
    os.makedirs(outdir, exist_ok=True)
    warrays = None
    if weight_args:
        wa = WeightArgs(model, len(inputs))
        model, warrays = wa, [t.numpy() for t in wa.tensors]
        call_inputs = tuple(inputs) + tuple(wa.tensors)
    else:
        call_inputs = tuple(inputs)
    with torch.no_grad():
        text = str(fx.export_and_import(model, *call_inputs, output_type="linalg-on-tensors", func_name="forward"))
    open(f"{outdir}/model.mlir", "w").write(text)
    link = f"{outdir}/resnet18.mlir"          # the Makefile still names things resnet18.*
    if os.path.lexists(link):
        os.remove(link)
    os.symlink("model.mlir", link)
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ame_pack
    ame_pack.pack([t.detach().numpy() for t in inputs], list(golden.shape), f"{outdir}/input.bin", f"{outdir}/model.json",
                  warrays=warrays, wbinpath=f"{outdir}/weights.bin", wfile_for_driver=f"{outdir}/weights.bin")
    golden.detach().numpy().astype("float32").tofile(f"{outdir}/golden_logits.bin")
    # driver generated from model.json
    here = os.path.dirname(os.path.abspath(__file__))
    os.system(f'python3 {here}/gen_driver.py {outdir}/model.json {outdir}/main_gen.c')
    print(f"wrote {outdir}: {len(text)//1000000} MB mlir, {len(inputs)} input(s), {0 if warrays is None else len(warrays)} weight arg(s), out {list(golden.shape)}")
