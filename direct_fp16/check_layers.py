#!/usr/bin/env python3
"""Compare the C network's dumped tensors and final softmax against the TF graph (resnet-50_v2.pb), image N.
   usage: check_layers.py <resnet-50_v2.pb> <imagenet_pic_data.bin> <dump_dir> <run_output.txt> <N> [ref_softmax_fp16.bin]
   dump_dir: AME16_DUMP_DIR of a host run; run_output.txt: its stdout (for the `result N ...` line)."""
import sys, numpy as np, tensorflow as tf
pb, picbin, dumpdir, runtxt, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
pics = np.fromfile(picbin, dtype=np.float16).reshape(-1, 224, 224, 3)
gd = tf.compat.v1.GraphDef(); gd.ParseFromString(open(pb, "rb").read())
g = tf.Graph()
names = ["resnet_model/max_pooling2d/MaxPool", "resnet_model/Mean"] + [f"resnet_model/Relu{'' if k == 0 else '_' + str(k)}" for k in range(49)] + [f"resnet_model/add{'' if k == 0 else '_' + str(k)}" for k in range(16)]
with g.as_default():
    tf.compat.v1.import_graph_def(gd, name="")
    with tf.compat.v1.Session(graph=g) as sess:
        outs = sess.run([g.get_tensor_by_name(n + ":0") for n in names] + [g.get_tensor_by_name("softmax_tensor_fp16:0")],
                        {g.get_tensor_by_name("input_tensor:0"): np.repeat(pics[N:N + 1], 64, 0)})
ref = {n: o[0].astype(np.float32).ravel() for n, o in zip(names, outs[:-1])}
print(f"{'tensor':<12}{'elems':>9}{'rel_rms':>11}{'max|d|':>10}{'max|tf|':>10}")
worst = (0, "")
for n in names:
    short = n.split("/")[-1] if "Relu" in n or "add" in n or "Mean" in n else "MaxPool"
    fname = short + "_0" if short in ("Relu", "add") else short          # the C dumps number from 0: Relu_0, add_0
    mine = np.fromfile(f"{dumpdir}/{fname}.bin", dtype=np.float16).astype(np.float32)
    r = ref[n]; assert mine.size == r.size, (short, mine.size, r.size)
    rr = float(np.linalg.norm(mine - r) / max(np.linalg.norm(r), 1e-30))
    if rr > worst[0]: worst = (rr, short)
    if short in ("MaxPool", "Mean", "Relu", "Relu_3", "Relu_12", "Relu_24", "Relu_36", "Relu_48") or short.startswith("add") and short in ("add", "add_3", "add_7", "add_13", "add_15"):
        print(f"{short:<12}{r.size:>9}{rr:>11.2e}{np.abs(mine - r).max():>10.3g}{np.abs(r).max():>10.3g}")
print(f"worst layer rel_rms: {worst[0]:.2e} at {worst[1]}")
line = [l for l in open(runtxt) if l.startswith("result")][0].split()
assert int(line[1]) == N
mine = np.array([int(x) for x in line[2].strip(",").split(",")], dtype=np.uint16).view(np.float16).astype(np.float32)
tfp = outs[-1][0].astype(np.float32)
print(f"softmax: top1 mine={mine.argmax()} tf={tfp.argmax()} | max|dp|={np.abs(mine - tfp).max():.3e} L1={np.abs(mine - tfp).sum():.3e} sum(mine)={mine.sum():.4f}"
      f" | top5 mine={np.argsort(-mine)[:5].tolist()} tf={np.argsort(-tfp)[:5].tolist()}")
if len(sys.argv) > 6:
    rf = np.fromfile(sys.argv[6], dtype=np.float16).reshape(-1, 1001)[N].astype(np.float32)
    print(f"vs stored ref_softmax_fp16.bin[{N}]: top1 {rf.argmax()}, max|d|={np.abs(mine - rf).max():.3e}; stored-vs-live TF max|d|={np.abs(tfp - rf).max():.3e}")
ok = mine.argmax() == tfp.argmax() and np.abs(mine - tfp).max() < 1e-2 + 1e-2 * np.abs(tfp).max()
print("PASS (top-1 equal, max|dp| within atol=rtol=1e-2, riscv-dnn's own tolerance)" if ok else "CHECK (see numbers above)")
