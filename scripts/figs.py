#!/usr/bin/env python
"""Figures (seed 0) into results/figs/. Usage: figs.py --A run_A --B run_B --C run_C [--E3 .. --E4 .. --W3 .. --W4 ..]"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
FIGS = os.path.join(ROOT, "results", "figs")


def layer_files(tag):
    return sorted(glob.glob(os.path.join(RUNS, tag, "layers.*.npz")))


def layer_key(f):
    return os.path.basename(f)[7:-4]  # strip 'layers.' and '.npz'


def unpack(d, key, n):
    return np.unpackbits(d[key])[:n].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    for a in ["A", "B", "C", "E3", "E4", "W3", "W4"]:
        ap.add_argument(f"--{a}", default=None)
    ap.add_argument("--out", default=None, help="output dir (default results/figs)")
    args = ap.parse_args()
    global FIGS
    if args.out:
        FIGS = args.out
    os.makedirs(FIGS, exist_ok=True)
    arms = {a: getattr(args, a) for a in ["A", "B", "C"] if getattr(args, a)}

    # ---- per-layer refined fraction + free fraction + Jaccard(A,B)
    stats = {a: [] for a in arms}
    free = []
    jacc = []
    names = []
    ref_files = layer_files(next(iter(arms.values())))
    for f in ref_files:
        k = layer_key(f)
        names.append(k)
        masks = {}
        for a, tag in arms.items():
            d = np.load(os.path.join(RUNS, tag, os.path.basename(f)))
            n = d["codes"].size
            m = unpack(d, "refine", n)
            masks[a] = m
            stats[a].append(m.mean())
            if a == list(arms)[0]:
                free.append(unpack(d, "free", n).mean())
        if "A" in masks and "B" in masks:
            inter = (masks["A"] & masks["B"]).sum()
            union = (masks["A"] | masks["B"]).sum()
            jacc.append(inter / max(union, 1))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 4))
    w = 0.8 / max(len(arms), 1)
    for i, a in enumerate(arms):
        ax.bar(x + i * w, 100 * np.array(stats[a]), w, label=f"arm {a}")
    ax.set_xticks(x + w); ax.set_xticklabels(names, rotation=90, fontsize=5); ax.set_ylabel("refined %"); ax.legend(); ax.set_title("per-layer refined fraction")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "refined_frac_per_layer.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(x, 100 * np.array(free)); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("% with e_c == e_f"); ax.set_title("'free' coarse elements (already on an even fine bin)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "free_frac_per_layer.png"), dpi=150); plt.close(fig)

    if jacc:
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.bar(x, jacc); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=5); ax.set_ylabel("Jaccard(A, B)")
        ax.set_title("overlap of Fisher mask (A) and in-loop refined set (B)")
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "jaccard_A_B_per_layer.png"), dpi=150); plt.close(fig)
        json.dump(dict(zip(names, map(float, jacc))), open(os.path.join(FIGS, "jaccard_A_B.json"), "w"), indent=1)

    # ---- gain histograms (log x), one panel per block-type, from the B run (needs --log_percol)
    tagB = arms.get("B") or arms.get("C")
    if tagB:
        files = layer_files(tagB)
        types = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.reshape(-1)
        bins = np.logspace(-9, 2, 100)
        for ti, t in enumerate(types):
            ax = axes[ti]
            for f in files:
                if not f.endswith(t + ".npz"):
                    continue
                d = np.load(f)
                if "gain_over_N" not in d:
                    continue
                g = d["gain_over_N"].astype(np.float32).reshape(-1)
                g = g[g > 0]
                blk = int(os.path.basename(f).split(".")[1])
                ax.hist(g, bins=bins, histtype="step", color=plt.cm.viridis(blk / 16), lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("gain / N_l (>0 only)")
        axes[-1].axis("off")
        fig.suptitle(f"gain/N_l histograms per layer (colour = block index), run {tagB}")
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "gain_hist_per_layer.png"), dpi=130); plt.close(fig)

    # ---- rate–distortion scatter
    pts = []
    for tag in sorted(os.listdir(RUNS)):
        rj, rt = os.path.join(RUNS, tag, "result.json"), os.path.join(RUNS, tag, "rate.json")
        if os.path.exists(rj) and os.path.exists(rt):
            R = json.load(open(rj)); M = json.load(open(rt))["model"]
            arm = R["config"].get("nested_arm", "none")
            arm = f"W{R['config']['w_bits']}" if arm == "none" else arm
            pts.append((tag, arm, R["config"]["seed"], M["huffman_bpw"], R["results"]["wikitext2"]))
    if pts:
        fig, ax = plt.subplots(figsize=(7, 5))
        cols = {"W3": "k", "W4": "k", "E3": "gray", "E4": "gray", "A": "C0", "B": "C1", "C": "C2", "D": "C3"}
        for tag, arm, seed, bpw, ppl in pts:
            ax.scatter(bpw, ppl, c=cols.get(arm, "C4"), marker="o" if seed == 0 else "x", s=30)
            if seed == 0:
                ax.annotate(tag, (bpw, ppl), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Huffman bpw (incl. scales/zeros)"); ax.set_ylabel("wiki2 PPL"); ax.set_title("rate–distortion (o seed 0, x seeds 1/2)")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rate_distortion.png"), dpi=150); plt.close(fig)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
