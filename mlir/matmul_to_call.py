#!/usr/bin/env python3
"""Rewrite memref-level `linalg.matmul` ops into calls to external functions.

    linalg.matmul ins(%a, %b : TA, TB) outs(%c : TC)
      ->  func.call @ame_mm_<f32|i8>_<n>(%a, %b, %c) : (TA, TB, TC) -> ()

Each distinct (TA,TB,TC) gets its own private declaration, using the REAL
operand types, so no memref.cast is needed. The callee bodies are the C shims
produced by gen_shims.py, which forward to ame_matmul_f32 / ame_matmul_i8.

Text-based on purpose (robust to MLIR version differences in pass names).
Run it AFTER bufferization, BEFORE lowering to LLVM.
Only handles rank-2 `linalg.matmul`; `linalg.batch_matmul` is reported, not rewritten.
"""
import re
import sys


def split_top(s, sep=","):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def match_paren(s, i):
    """s[i] == '(' -> index of its matching ')'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced parentheses")


def elem_kind(ta, tb, tc):
    if "xf32" in ta and "xf32" in tb and "xf32" in tc:
        return "f32"
    if "xi8" in ta and "xi8" in tb and "xi32" in tc:
        return "i8"
    raise ValueError(f"unsupported matmul element types: {ta} {tb} {tc}")


def main(src_path, dst_path):
    text = open(src_path).read()
    decls = {}   # (kind, ta, tb, tc) -> name
    counter = 0
    pieces = []
    pos = 0
    # the op may span several lines when hand-written, so parse the whole text
    for m in re.finditer(r"\blinalg\.(batch_matmul|matmul)\b(?=\s*(?:\{[^}]*\}\s*)?ins\()", text):
        prefix = "ame_bmm" if m.group(1) == "batch_matmul" else "ame_mm"
        if m.start() < pos:
            continue
        ins_open = text.index("ins(", m.end()) + 3
        ins_close = match_paren(text, ins_open)
        outs_open = text.index("outs(", ins_close) + 4
        outs_close = match_paren(text, outs_open)

        ins_vals, ins_types = text[ins_open + 1: ins_close].split(" : ", 1)
        outs_vals, outs_types = text[outs_open + 1: outs_close].split(" : ", 1)
        a, b = [v.strip() for v in split_top(ins_vals)]
        (c,) = [v.strip() for v in split_top(outs_vals)]
        ta, tb = split_top(ins_types)
        (tc,) = split_top(outs_types)

        kind = elem_kind(ta, tb, tc)
        key = (prefix, kind, ta, tb, tc)
        if key not in decls:
            decls[key] = f"{prefix}_{kind}_{counter}"
            counter += 1
        name = decls[key]
        pieces.append(text[pos: m.start()])
        pieces.append(f"func.call @{name}({a}, {b}, {c}) : ({ta}, {tb}, {tc}) -> ()")
        pos = outs_close + 1
    pieces.append(text[pos:])
    out = "".join(pieces)


    decl_text = "".join(
        f"  func.func private @{n}({ta}, {tb}, {tc})\n"
        for (_pf, kind, ta, tb, tc), n in decls.items()
    )
    if decl_text:
        if re.search(r"^\s*module\b", out, re.M):
            res = out.find("{-#")                  # trailing resource blob (torch-mlir weights)
            end = res if res != -1 else len(out)
            idx = out[:end].rstrip().rfind("}")    # closing brace of the module
            out = out[:idx] + decl_text + out[idx:]
        else:
            out = "module {\n" + out + "\n" + decl_text + "}\n"
    open(dst_path, "w").write(out)
    print(f"rewrote {counter} distinct matmul signature(s) -> {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: matmul_to_call.py in.mlir out.mlir")
    main(sys.argv[1], sys.argv[2])
