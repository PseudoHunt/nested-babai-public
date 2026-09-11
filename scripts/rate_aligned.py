#!/usr/bin/env python
"""
Zero-aligned rate accounting: entropy-code the zero-relative symbol (q - z_g) instead of the raw code q.
z_g is already stored per group, so the decoder recovers q = sym + z_g exactly; this only changes the encoder.
Symbols: native b-bit: q - z in [-(2^b-1), 2^b-1]; nested b->b+1: q4 - z_f (z_f = 2 z_c) in [-(2^(b+1)-2), 2^(b+1)-1].
One Huffman table per tensor over the (2^(b+1)-1)-symbol alphabet + table bits + the same scale/zero side info.
Usage: rate_aligned.py RUN_DIR [RUN_DIR ...]  -> prints raw vs aligned ideal/Huffman bpw, writes rate_aligned.json per run.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rate import huffman_lengths


def aligned_tensor(codes, scale, zero, bits, zero_bits, group_size=128, n_qparams=None):
    n = codes.size
    z = zero.astype(np.int64)
    if zero_bits < bits:            # nested path: stored zero is the coarse one, dequant zero is 2*z_c
        z = 2 * z
    sym = codes.astype(np.int64) - np.repeat(z, codes.shape[1] // z.shape[1], axis=1)
    off = 2 ** bits - 1
    nsym = 2 * off + 1
    counts = np.bincount((sym + off).reshape(-1), minlength=nsym)[:nsym]
    p = counts / n
    nz = p > 0
    ent = float(-(counts[nz] * np.log2(p[nz])).sum())
    L = huffman_lengths(counts)
    huff = float(sum(counts[i] * L[i] for i in L))
    table = nsym * 4
    side = (scale.size if n_qparams is None else int(n_qparams)) * (16 + zero_bits)
    return dict(entropy_per_code=ent / n, huffman_per_code=huff / n,
                ideal_bpw=(ent + table + side) / n, huffman_bpw=(huff + table + side) / n, n=int(n),
                hist=counts.tolist(), offset=int(off))


def main():
    print(f"{'run':22s} {'raw ideal':>9s} {'raw Huff':>9s} {'aligned ideal':>13s} {'aligned Huff':>12s} {'code H raw':>10s} {'code H al':>9s}")
    for run in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(run, "layers.*.npz")))
        raw = json.load(open(os.path.join(run, "rate.json")))["model"]
        tot = dict(n=0, ideal=0.0, huff=0.0, ent=0.0)
        per = {}
        ent_raw = 0.0
        rawpl = json.load(open(os.path.join(run, "rate.json")))["per_layer"]
        for f in files:
            d = np.load(f)
            zb = int(d["zero_bits"]) if "zero_bits" in d else (3 if int(d["bits"]) in (3, 4) and int(d["zero"].max()) <= 7 else int(d["bits"]))
            r = aligned_tensor(d["codes"], d["scale"], d["zero"], int(d["bits"]), zb, n_qparams=int(d["n_qparams"]) if "n_qparams" in d else None)
            key = os.path.basename(f)[:-4]
            per[key] = r
            tot["n"] += r["n"]; tot["ideal"] += r["ideal_bpw"] * r["n"]; tot["huff"] += r["huffman_bpw"] * r["n"]
            tot["ent"] += r["entropy_per_code"] * r["n"]; ent_raw += rawpl[key]["entropy_per_code"] * r["n"]
        n = tot["n"]
        model = dict(ideal_bpw=tot["ideal"] / n, huffman_bpw=tot["huff"] / n, entropy_per_code=tot["ent"] / n,
                     raw_ideal_bpw=raw["ideal_bpw"], raw_huffman_bpw=raw["huffman_bpw"])
        json.dump(dict(model=model, per_layer=per), open(os.path.join(run, "rate_aligned.json"), "w"), indent=1)
        print(f"{os.path.basename(run):22s} {raw['ideal_bpw']:9.3f} {raw['huffman_bpw']:9.3f} {model['ideal_bpw']:13.3f} {model['huffman_bpw']:12.3f} {ent_raw/n:10.3f} {model['entropy_per_code']:9.3f}")


if __name__ == "__main__":
    main()
