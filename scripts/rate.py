#!/usr/bin/env python
"""
Solver-independent rate accounting.

Input : a run dump dir with layers.*.npz, each holding
          codes : uint8 [d_out, d_in]   integer codes (alphabet 2**bits)
          scale : fp32  [d_out, n_groups]
          zero  : uint8 [d_out, n_groups]
          bits  : int   (code alphabet width)
        (optional) refine : packed bool, free : packed bool
Output: per-tensor and model-level bpw:
          nominal = bits + (16 + zero_bits)/g
          ideal   = Shannon entropy of the code stream + table + scales + zeros
          huffman = one 2**bits-symbol Huffman table per tensor + table bits + scales + zeros
Zero bits: the coarse zero width (stored as `zero_bits` in the dump: b for native W_b and for nested b->b+1; older dumps
without it: 3 for native W3 / nested 3->4, else = bits).
"""
import argparse, glob, heapq, json, math, os
import numpy as np


def huffman_lengths(counts):
    """Code lengths of an optimal prefix code for the given symbol counts (0-count symbols get length 0)."""
    syms = [(c, i) for i, c in enumerate(counts) if c > 0]
    if len(syms) == 1:
        return {syms[0][1]: 1}
    heap = [(c, idx, [i]) for idx, (c, i) in enumerate(syms)]
    heapq.heapify(heap)
    lengths = {i: 0 for _, i in syms}
    uid = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for i in s1 + s2:
            lengths[i] += 1
        heapq.heappush(heap, (c1 + c2, uid, s1 + s2))
        uid += 1
    return lengths


def rate_tensor(codes, scale, zero, bits, group_size, zero_bits=None, n_qparams=None):
    n = codes.size
    nsym = 2 ** bits
    counts = np.bincount(codes.reshape(-1), minlength=nsym)[:nsym].astype(np.int64)
    p = counts / n
    nz = p > 0
    ent_bits = float(-(counts[nz] * np.log2(p[nz])).sum())
    L = huffman_lengths(counts)
    huff_bits = float(sum(counts[i] * L[i] for i in L))
    table_bits = nsym * 4                      # one 4-bit code length per symbol
    n_groups = scale.size if n_qparams is None else int(n_qparams)   # per-tensor dumps: 1 (scale, zero) set
    if zero_bits is None:   # old dumps without an explicit zero width: 3-bit coarse zero for W3 / nested 3->4, else native
        zero_bits = 3 if (bits == 3 or _is_nested_like(zero)) else bits
    side_bits = n_groups * (16 + zero_bits)
    return dict(
        n=int(n), n_groups=int(n_groups), zero_bits=int(zero_bits),
        entropy_per_code=ent_bits / n,
        huffman_per_code=huff_bits / n,
        nominal_bpw=bits + side_bits / n,
        ideal_bpw=(ent_bits + table_bits + side_bits) / n,
        huffman_bpw=(huff_bits + table_bits + side_bits) / n,
        hist=counts.tolist(),
    )


def _is_nested_like(zero):
    # zero stored as coarse 3-bit zero (0..7) on the nested path; native W4 zeros go up to 15
    return int(zero.max()) <= 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--zero_bits", type=int, default=None, help="override side-info zero width (default: 3 for W3/nested, 4 for native W4)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "layers.*.npz")))
    assert files, f"no layer dumps in {args.run_dir}"
    per_layer = {}
    tot = dict(n=0, nominal=0.0, ideal=0.0, huffman=0.0, refined=0, free=0, odd=0)
    for f in files:
        d = np.load(f)
        codes, scale, zero, bits = d["codes"], d["scale"], d["zero"], int(d["bits"])
        r = rate_tensor(codes, scale, zero, bits, args.group_size, int(d["zero_bits"]) if "zero_bits" in d else None,
                        int(d["n_qparams"]) if "n_qparams" in d else None)
        if args.zero_bits is not None and r["zero_bits"] != args.zero_bits:
            delta = (args.zero_bits - r["zero_bits"]) * r["n_groups"] / r["n"]
            r["ideal_bpw"] += delta; r["huffman_bpw"] += delta
            r["nominal_bpw"] += (args.zero_bits - r["zero_bits"]) / args.group_size
            r["zero_bits"] = args.zero_bits
        if "refine" in d:
            r["refined_frac"] = float(np.unpackbits(d["refine"])[: codes.size].mean())
            r["free_frac"] = float(np.unpackbits(d["free"])[: codes.size].mean())
            tot["refined"] += r["refined_frac"] * codes.size
            tot["free"] += r["free_frac"] * codes.size
        r["odd_frac"] = float((codes % 2 == 1).mean()) if bits == 4 else None
        key = os.path.basename(f)[:-4]
        per_layer[key] = r
        tot["n"] += r["n"]
        tot["nominal"] += r["nominal_bpw"] * r["n"]
        tot["ideal"] += r["ideal_bpw"] * r["n"]
        tot["huffman"] += r["huffman_bpw"] * r["n"]
    n = tot["n"]
    model = dict(
        n_weights=n, n_tensors=len(files),
        nominal_bpw=tot["nominal"] / n, ideal_bpw=tot["ideal"] / n, huffman_bpw=tot["huffman"] / n,
        refined_frac=(tot["refined"] / n) if tot["refined"] else None,
        free_frac=(tot["free"] / n) if tot["free"] else None,
    )
    out = dict(model=model, per_layer=per_layer)
    with open(os.path.join(args.run_dir, "rate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(model, indent=1))


if __name__ == "__main__":
    main()
