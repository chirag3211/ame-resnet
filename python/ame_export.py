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


def export(outdir, model, inputs, golden, check_against=None, tol=1e-4):
    """Write outdir/{resnet18.mlir (symlink name used by the Makefile), input.bin, golden_logits.bin, model.json}."""
    from torch_mlir import fx
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        text = str(fx.export_and_import(model, *inputs, output_type="linalg-on-tensors", func_name="forward"))
    open(f"{outdir}/model.mlir", "w").write(text)
    link = f"{outdir}/resnet18.mlir"          # the Makefile still names things resnet18.*
    if os.path.lexists(link):
        os.remove(link)
    os.symlink("model.mlir", link)
    assert len(inputs) == 1, "single-input driver for now"
    inputs[0].detach().numpy().astype("float32").tofile(f"{outdir}/input.bin")
    golden.detach().numpy().astype("float32").tofile(f"{outdir}/golden_logits.bin")
    json.dump({"in_shape": list(inputs[0].shape), "out_shape": list(golden.shape)}, open(f"{outdir}/model.json", "w"))
    # driver generated from model.json
    here = os.path.dirname(os.path.abspath(__file__))
    os.system(f'python3 {here}/gen_driver.py {outdir}/model.json {outdir}/main_gen.c')
    print(f"wrote {outdir}: {len(text)//1000000} MB mlir, in {list(inputs[0].shape)} out {list(golden.shape)}")
