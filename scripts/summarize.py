#!/usr/bin/env python
"""Collect results/runs/*/{result.json,rate.json} into results/runs_table.md and refresh the auto block in SUMMARY.md."""
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
SUMMARY = os.path.join(ROOT, "results", "SUMMARY.md")


def load_runs():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        rj, rt = os.path.join(d, "result.json"), os.path.join(d, "rate.json")
        if not os.path.exists(rj):
            continue
        R = json.load(open(rj))
        cfg, res = R["config"], R["results"]
        rate = json.load(open(rt))["model"] if os.path.exists(rt) else {}
        arm = cfg.get("nested_arm", "none")
        b = cfg["w_bits"]
        if arm == "none":
            arm = f"native W{b}"
        else:
            arm = {"E3": f"E{b} (coarse only)", "E4": f"E{b+1} (all fine)"}.get(arm, arm) + f" [{b}→{b+1}]"
        sel = "—"
        if cfg.get("nested_arm") in ("A", "B"):
            sel = f"p={cfg['nested_p']}"
        elif cfg.get("nested_arm") in ("C", "D"):
            sel = f"λ={cfg['nested_lambda']:.4g}"
        elif cfg.get("nested_arm") == "E3":
            sel = "λ=∞"
        elif cfg.get("nested_arm") == "E4":
            sel = "λ=0"
        rows.append(dict(
            tag=os.path.basename(d), arm=arm, sel=sel, seed=cfg["seed"],
            nominal=rate.get("nominal_bpw"), ideal=rate.get("ideal_bpw"), huffman=rate.get("huffman_bpw"),
            refined=rate.get("refined_frac"), free=rate.get("free_frac"),
            wiki2=res.get("wikitext2"), c4=res.get("c4-new"), time=res.get("time"),
        ))
    return rows


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(rows):
    hdr = "| run | Arm | p / λ | seed | refined % | nominal bpw | ideal bpw | Huffman bpw | wiki2 PPL | c4-new PPL | time (s) |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += f"| {r['tag']} | {r['arm']} | {r['sel']} | {r['seed']} | {fmt(None if r['refined'] is None else 100*r['refined'], 2)} | {fmt(r['nominal'])} | {fmt(r['ideal'])} | {fmt(r['huffman'])} | {fmt(r['wiki2'])} | {fmt(r['c4'])} | {fmt(r['time'], 0)} |\n"
    return hdr + body


def main():
    rows = load_runs()
    t = table(rows)
    open(os.path.join(ROOT, "results", "runs_table.md"), "w").write(t)
    json.dump(rows, open(os.path.join(ROOT, "results", "runs.json"), "w"), indent=1)
    if os.path.exists(SUMMARY):
        s = open(SUMMARY).read()
        new = "<!-- AUTO-RUNS-START -->\n" + t + "<!-- AUTO-RUNS-END -->"
        if "<!-- AUTO-RUNS-START -->" in s:
            s = re.sub(r"<!-- AUTO-RUNS-START -->.*?<!-- AUTO-RUNS-END -->", lambda m: new, s, flags=re.S)
        else:
            s += "\n\n## All runs (auto)\n\n" + new + "\n"
        open(SUMMARY, "w").write(s)
    print(t)


if __name__ == "__main__":
    main()
