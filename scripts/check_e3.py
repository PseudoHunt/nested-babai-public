#!/usr/bin/env python
"""Phase-1 gate (a): the coarse-only endpoint run must equal the native W_b run: codes == 2*codes_Wb, identical scale/zero, identical PPL.
usage: check_e3.py NESTED_TAG NATIVE_TAG"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
e3, w3 = sys.argv[1] if len(sys.argv) > 1 else "e3_s0", sys.argv[2] if len(sys.argv) > 2 else "w3_s0"
de, dw = [os.path.join(ROOT, "results", "runs", t) for t in (e3, w3)]
bad = 0
n_code_mism = 0; n_tot = 0; max_scale_diff = 0.0; n_zero_mism = 0
for f in sorted(glob.glob(os.path.join(dw, "layers.*.npz"))):
    a, b = np.load(os.path.join(de, os.path.basename(f))), np.load(f)
    assert int(a["bits"]) == int(b["bits"]) + 1, (int(a["bits"]), int(b["bits"]))
    m = int((a["codes"].astype(np.int32) != 2 * b["codes"].astype(np.int32)).sum())
    n_code_mism += m; n_tot += a["codes"].size
    max_scale_diff = max(max_scale_diff, float(np.abs(a["scale"] - b["scale"]).max()))
    n_zero_mism += int((a["zero"] != b["zero"]).sum())
    if m or (a["scale"] != b["scale"]).any() or (a["zero"] != b["zero"]).any():
        bad += 1
        print(f"  MISMATCH {os.path.basename(f)}: codes {m}, scale maxdiff {np.abs(a['scale']-b['scale']).max():.3g}, zeros {(a['zero']!=b['zero']).sum()}")
re = json.load(open(os.path.join(de, "result.json")))["results"]
rw = json.load(open(os.path.join(dw, "result.json")))["results"]
print(f"{e3} vs {w3}: code mismatches {n_code_mism}/{n_tot} ({n_code_mism/n_tot:.2e}), max |scale diff| {max_scale_diff:.3g}, zero mismatches {n_zero_mism}, tensors with any mismatch {bad}/112")
print(f"PPL wiki2 {re['wikitext2']} vs {rw['wikitext2']}   c4-new {re['c4-new']} vs {rw['c4-new']}")
print("GATE (a):", "PASS" if (n_code_mism == 0 and max_scale_diff == 0 and n_zero_mism == 0 and re['wikitext2'] == rw['wikitext2']) else "CHECK")
