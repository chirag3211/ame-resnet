import sys, os, torch, torchvision
from torch_mlir import fx
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2
out = sys.argv[2] if len(sys.argv) > 2 else f"build_vit{L}"
torch.manual_seed(0)
m = torchvision.models.vision_transformer.VisionTransformer(
    image_size=224, patch_size=16, num_layers=L, num_heads=12,
    hidden_dim=768, mlp_dim=3072, num_classes=1000).eval()
torch.nn.init.normal_(m.heads.head.weight, std=0.02)
torch.nn.init.normal_(m.heads.head.bias, std=0.02)
x = torch.randn(1, 3, 224, 224)
os.makedirs(out, exist_ok=True)
with torch.no_grad():
    y = m(x)
    open(f"{out}/vit.mlir", "w").write(str(fx.export_and_import(m, x, output_type="linalg-on-tensors", func_name="forward")))
x.numpy().tofile(f"{out}/input.bin"); y.numpy().tofile(f"{out}/golden_logits.bin")
print("top1", int(y.argmax()), "logit absmax", float(y.abs().max()))
