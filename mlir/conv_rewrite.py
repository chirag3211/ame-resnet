#!/usr/bin/env python3
"""Rewrite linalg.conv_2d_nchw_fchw and linalg.conv_1d_ncw_fcw (tensor form) into:
  5D linalg.generic gather  [C,KH,KW,OH,OW]   (no div/mod in any map)
  -> tensor.collapse_shape  [C*KH*KW, OH*OW]
  -> linalg.matmul with collapsed weights [F, C*KH*KW], outs = collapsed conv init
  -> tensor.expand_shape    [1,F,OH,OW]
Input is assumed already padded (tensor.pad before the conv). N must be 1, f32 only.
1D: gather [C,KW,OW] -> collapse [C*KW, OW] -> matmul with weights [F, C*KW] -> expand [1,F,OW].
Usage: conv_rewrite.py IN.mlir OUT.mlir [--old-expand]
  --old-expand: omit `output_shape` on tensor.expand_shape (older MLIR). Auto-detected
                if IN already contains an expand_shape without output_shape.
Exits non-zero if any conv is left or the count is not what was matched.
"""
import re
import sys

V = r'%[\w$.#-]+'
def T(n):
    return r'tensor<(?P<' + n + r'>[\dx]+)xf32>'

CONV = re.compile(
    r'^(?P<ind>\s*)(?P<res>' + V + r') = linalg\.conv_2d_nchw_fchw\s*(?P<attrs>\{.*?\})?\s*'
    r'ins\((?P<x>' + V + r'),\s*(?P<w>' + V + r')\s*:\s*' + T('xt') + r',\s*' + T('wt') + r'\)\s*'
    r'outs\((?P<o>' + V + r')\s*:\s*' + T('ot') + r'\)\s*->\s*' + T('rt') + r'\s*$')


CONV1 = re.compile(
    r'^(?P<ind>\s*)(?P<res>' + V + r') = linalg\.conv_1d_ncw_fcw\s*(?P<attrs>\{.*?\})?\s*'
    r'ins\((?P<x>' + V + r'),\s*(?P<w>' + V + r')\s*:\s*' + T('xt') + r',\s*' + T('wt') + r'\)\s*'
    r'outs\((?P<o>' + V + r')\s*:\s*' + T('ot') + r'\)\s*->\s*' + T('rt') + r'\s*$')


def dims(s):
    return [int(v) for v in s.split('x')]


def pair(attrs, name):
    m = re.search(name + r'\s*=\s*dense<(\[[^\]]*\]|\d+)>', attrs or '')
    if not m:
        return (1, 1)
    v = [int(a) for a in re.findall(r'\d+', m.group(1))]
    return (v[0], v[-1])


def tt(*d):
    return 'tensor<' + 'x'.join(str(i) for i in d) + 'xf32>'


def emit(m, n, old_expand):
    ind, res = m['ind'], m['res']
    x, w, o = m['x'], m['w'], m['o']
    N, C, H, W = dims(m['xt'])
    F, C2, KH, KW = dims(m['wt'])
    N2, F2, OH, OW = dims(m['ot'])
    assert m['rt'] == m['ot'], 'result type != outs type'
    assert N == 1 and N2 == 1, 'only N=1 supported'
    assert C == C2 and F == F2, 'channel mismatch'
    sh, sw = pair(m['attrs'], 'strides')
    dh, dw = pair(m['attrs'], 'dilations')
    assert (dh, dw) == (1, 1), 'dilation != 1 not supported'
    assert H >= (OH - 1) * sh + KH and W >= (OW - 1) * sw + KW, 'input too small for conv geometry'
    ck, ohw = C * KH * KW, OH * OW
    p = f'%cr{n}'
    ih = f'd3 * {sh} + d1' if sh != 1 else 'd3 + d1'
    iw = f'd4 * {sw} + d2' if sw != 1 else 'd4 + d2'
    t_x, t_w = tt(N, C, H, W), tt(F, C, KH, KW)
    t_g, t_gc = tt(C, KH, KW, OH, OW), tt(ck, ohw)
    t_wc, t_o2, t_o = tt(F, ck), tt(F, ohw), tt(N, F, OH, OW)
    shape = '' if old_expand else f' output_shape [{N}, {F}, {OH}, {OW}]'
    L = [
        f'{ind}{p}_e = tensor.empty() : {t_g}',
        f'{ind}{p}_g = linalg.generic {{indexing_maps = ['
        f'affine_map<(d0, d1, d2, d3, d4) -> (0, d0, {ih}, {iw})>, '
        f'affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], '
        f'iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}} '
        f'ins({x} : {t_x}) outs({p}_e : {t_g}) {{',
        f'{ind}^bb0(%in: f32, %out: f32):',
        f'{ind}  linalg.yield %in : f32',
        f'{ind}}} -> {t_g}',
        f'{ind}{p}_gc = tensor.collapse_shape {p}_g [[0, 1, 2], [3, 4]] : {t_g} into {t_gc}',
        f'{ind}{p}_wc = tensor.collapse_shape {w} [[0], [1, 2, 3]] : {t_w} into {t_wc}',
        f'{ind}{p}_oc = tensor.collapse_shape {o} [[0, 1], [2, 3]] : {t_o} into {t_o2}',
        f'{ind}{p}_mm = linalg.matmul ins({p}_wc, {p}_gc : {t_wc}, {t_gc}) outs({p}_oc : {t_o2}) -> {t_o2}',
        f'{ind}{res} = tensor.expand_shape {p}_mm [[0, 1], [2, 3]]{shape} : {t_o2} into {t_o}',
    ]
    return '\n'.join(L)


def emit1d(m, n, old_expand):
    ind, res = m['ind'], m['res']
    x, w, o = m['x'], m['w'], m['o']
    N, C, W = dims(m['xt'])
    F, C2, KW = dims(m['wt'])
    N2, F2, OW = dims(m['ot'])
    assert m['rt'] == m['ot'], 'result type != outs type'
    assert N == 1 and N2 == 1, 'only N=1 supported'
    assert C == C2 and F == F2, 'channel mismatch'
    sw = pair(m['attrs'], 'strides')[0]
    assert pair(m['attrs'], 'dilations')[0] == 1, 'dilation != 1 not supported'
    assert W >= (OW - 1) * sw + KW, 'input too small for conv geometry'
    ck = C * KW
    p = f'%cr{n}'
    iw = f'd2 * {sw} + d1' if sw != 1 else 'd2 + d1'
    t_x, t_w = tt(N, C, W), tt(F, C, KW)
    t_g, t_gc = tt(C, KW, OW), tt(ck, OW)
    t_wc, t_o2, t_o = tt(F, ck), tt(F, OW), tt(N, F, OW)
    shape = '' if old_expand else f' output_shape [{N}, {F}, {OW}]'
    L = [
        f'{ind}{p}_e = tensor.empty() : {t_g}',
        f'{ind}{p}_g = linalg.generic {{indexing_maps = ['
        f'affine_map<(d0, d1, d2) -> (0, d0, {iw})>, '
        f'affine_map<(d0, d1, d2) -> (d0, d1, d2)>], '
        f'iterator_types = ["parallel", "parallel", "parallel"]}} '
        f'ins({x} : {t_x}) outs({p}_e : {t_g}) {{',
        f'{ind}^bb0(%in: f32, %out: f32):',
        f'{ind}  linalg.yield %in : f32',
        f'{ind}}} -> {t_g}',
        f'{ind}{p}_gc = tensor.collapse_shape {p}_g [[0, 1], [2]] : {t_g} into {t_gc}',
        f'{ind}{p}_wc = tensor.collapse_shape {w} [[0], [1, 2]] : {t_w} into {t_wc}',
        f'{ind}{p}_oc = tensor.collapse_shape {o} [[0, 1], [2]] : {t_o} into {t_o2}',
        f'{ind}{p}_mm = linalg.matmul ins({p}_wc, {p}_gc : {t_wc}, {t_gc}) outs({p}_oc : {t_o2}) -> {t_o2}',
        f'{ind}{res} = tensor.expand_shape {p}_mm [[0, 1], [2]]{shape} : {t_o2} into {t_o}',
    ]
    return '\n'.join(L)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 2:
        sys.exit(__doc__)
    src = open(args[0]).read().split('\n')
    old = '--old-expand' in sys.argv or any(
        'tensor.expand_shape' in l and 'output_shape' not in l for l in src)
    out, n = [], 0
    for line in src:
        m = CONV.match(line) if 'linalg.conv_2d_nchw_fchw' in line else None
        m1 = CONV1.match(line) if 'linalg.conv_1d_ncw_fcw' in line else None
        if m:
            out.append(emit(m, n, old))
            n += 1
        elif m1:
            out.append(emit1d(m1, n, old))
            n += 1
        else:
            out.append(line)
    text = '\n'.join(out)
    left = text.count('linalg.conv_2d_nchw_fchw') + text.count('linalg.conv_1d_ncw_fcw')
    open(args[1], 'w').write(text)
    print(f'rewrote {n} convs, {left} left, old_expand={old}')
    if left:
        sys.exit(f'ERROR: {left} conv(s) not matched (dynamic shape / odd formatting?)')


if __name__ == '__main__':
    main()
