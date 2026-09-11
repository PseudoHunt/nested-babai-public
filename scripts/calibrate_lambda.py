#!/usr/bin/env python
"""
λ for arm C from the model-wide CDF of gain/N_l logged by an arm-B (or any nested) run with --log_percol.
Rate only: pick λ so that the expected refined fraction equals --target_p (elements with gain/N_l > λ).
Optionally a secant step from two previous (λ, huffman_bpw) points to a target bpw.
"""
import argparse, glob, json, os
import numpy as np


def load_gains(run_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(run_dir, "layers.*.npz"))):
        d = np.load(f)
        assert "gain_over_N" in d, f"{f} has no per-element gains (run with --log_percol)"
        parts.append(d["gain_over_N"].reshape(-1))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--target_p", type=float, default=None, help="expected refined fraction")
    ap.add_argument("--secant", nargs=5, type=float, default=None, metavar=("L0", "BPW0", "L1", "BPW1", "TARGET"),
                    help="secant step on (λ -> bpw) to hit TARGET (uses log λ)")
    args = ap.parse_args()
    if args.secant:
        l0, b0, l1, b1, tgt = args.secant
        x0, x1 = np.log(l0), np.log(l1)
        x = x1 + (tgt - b1) * (x1 - x0) / (b1 - b0)
        print(json.dumps(dict(lambda_next=float(np.exp(x)))))
        return
    g = load_gains(args.run_dir).astype(np.float32)
    n = g.size
    pos = float((g > 0).mean())
    print(f"{n/1e6:.1f}M elements, gain>0 fraction {pos:.4f}, max gain/N {g.max():.4g}")
    qs = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    print("quantiles of gain/N:", {q: float(np.quantile(g, q)) for q in qs})
    if args.target_p is not None:
        lam = float(np.quantile(g, 1 - args.target_p))
        real = float((g > lam).mean())
        print(json.dumps(dict(target_p=args.target_p, lam=lam, expected_refined_frac=real)))


if __name__ == "__main__":
    main()
