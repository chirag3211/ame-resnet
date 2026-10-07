"""numpy-only helpers: pack several input arrays into one input.bin and describe them in model.json."""
import json
import numpy as np

DT = {"float32": ("f32", "float", 4), "int64": ("i64", "int64_t", 8)}


def pack(arrays, out_shape, binpath, jsonpath, warrays=None, wbinpath=None, wfile_for_driver=None):
    """arrays: list of numpy arrays (float32 or int64). Each is padded to a multiple of 64 bytes in input.bin."""
    specs, blob, off = [], bytearray(), 0
    for a in arrays:
        a = np.ascontiguousarray(a)
        assert a.dtype.name in DT, f"unsupported dtype {a.dtype}"
        raw = a.tobytes()
        pad = (-len(raw)) % 64
        specs.append({"shape": list(a.shape), "dtype": a.dtype.name, "offset": off, "bytes": len(raw)})
        blob += raw + b"\0" * pad
        off += len(raw) + pad
    open(binpath, "wb").write(bytes(blob))
    meta = {"inputs": specs, "out_shape": list(out_shape)}
    if warrays is not None:
        wspecs, off = [], 0
        with open(wbinpath, "wb") as f:                     # weights are streamed out, never held twice in memory
            for a in warrays:
                a = np.ascontiguousarray(a)
                assert a.dtype.name == "float32", f"weights must be float32, got {a.dtype}"
                raw = a.tobytes(); pad = (-len(raw)) % 64
                wspecs.append({"shape": list(a.shape), "offset": off, "bytes": len(raw)})
                f.write(raw + b"\0" * pad); off += len(raw) + pad
        meta["weights"] = {"file": wfile_for_driver or wbinpath, "total": off, "specs": wspecs}
    json.dump(meta, open(jsonpath, "w"))
    return specs
