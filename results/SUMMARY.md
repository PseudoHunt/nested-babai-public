# nested-babai — SUMMARY

**Status:** Phases 0–2 done; Phase 3 done for seeds 0 and 1 (seed 2 deferred by the PI on 2026-09-11 in favour of the
2→3-bit experiment, which is done: seed 0, C 36.1 < B 37.3 ≪ A 71.1 at matched rate). **Interim verdict on 2 seeds: PASS** — μ_C − μ_A = **−0.120** wiki2 PPL at matched
Huffman rate (threshold −max(2σ, 0.05) = −0.05, pooled σ = 0.010); μ_B − μ_A = −0.088 (also beyond the threshold).
Paper draft in `paper/`. Extra experiments on PI request: 2→3 bit (in-loop wins by 2×), zero-alignment coding (−0.03 bpw only), per-tensor scale (mixed, see section).

## §6 result table (A100, TurboBoA g=128, matched Huffman rate R1 = 3.071 ± 0.02)
| Arm | p / λ | seed | nominal bpw | ideal bpw | Huffman bpw | wiki2 PPL | c4-new PPL | quant time |
|---|---|---|---|---|---|---|---|---|
| native W3 | — | 0 | 3.148 | 2.966 | 2.996 | 12.011 | 20.115 | 1059 s |
| native W4 | — | 0 | 4.156 | 3.875 | 3.902 | 10.224 | 15.428 | 892 s |
| E3 | ∞ | 0 | 4.148 | 2.966 | 2.996 | 12.011 | 20.115 | 1205 s |
| E4 | 0 | 0 | 4.148 | 3.946 | 3.982 | 10.533 | 16.346 | 1092 s |
| A | 1 % | 0 / 1 | 4.148 | 3.008 / 3.008 | 3.0706 / 3.0705 | 11.855 / 11.863 → **μ 11.859 ± 0.006** | 19.703 / 19.621 → 19.662 | 1234 / 1227 s |
| B | p = 1 % | 0 / 1 | 4.148 | 3.027 / 3.027 | 3.0756 / 3.0755 | 11.782 / 11.761 → **μ 11.772 ± 0.015** | 19.513 / 19.434 → 19.474 | 1456 / — s |
| C | λ = 0.0759 | 0 / 1 | 4.148 | 3.031 / 3.031 | 3.0858 / 3.0853 | 11.743 / 11.735 → **μ 11.739 ± 0.006** | 19.410 / 19.242 → 19.326 | 1383 / — s |
| D | — | — | | | | not run (Phase 4) | | |
| A-ref | 1 % | — | | | 2.6 (different solver, the reference pipeline) | closer to W4 than W3 | | |

Decision rule (brief §4, applied to the two available seeds; σ = pooled seed-to-seed std of A/B/C = 0.0097):
PASS requires μ_C − μ_A ≤ −max(2σ, 0.05) = −0.05 → **−0.120: PASS**. B: −0.088, also a clear win. Rate drift across seeds at
fixed p / λ is ≤ 0.001 bpw for all three arms (C's realized refined fraction 0.996 % → 0.984 %). c4-new agrees (C −0.34,
B −0.19 vs A). The seed-2 runs (`A_p1_s2`, `B_p1_s2`, `C_l0.0759_s2`, ~70 min) would complete the pre-registered 3-seed
rule; the Fisher for seed 2 is not yet computed. **`r_i` handling:** `r_i = [(U_subᵀU_sub)⁻¹]_ii` from the solver's own
`U_out_sub⁻¹` for q/k (TurboBoA's two-sided layers, 16-row chunks), `r_i = 1` for every one-sided layer (v/o/MLP).

## Phase 0 (seed 0)
| run | GPU | wiki2 PPL | c4-new PPL | nominal bpw | ideal bpw | Huffman bpw | quant time |
|---|---|---|---|---|---|---|---|
| native W3 g128 (`w3_s0`) | **A100** | **12.011** | 20.115 | 3.148 | 2.966 | 2.996 | 1059 s |
| native W4 g128 (`w4_s0`) | **A100** | **10.224** | 15.428 | 4.156 | 3.875 | 3.902 | 892 s |
| native W3 g128 (`w3_s0_a30`, cross-check only) | A30 | 12.034 | 20.146 | 3.148 | 2.966 | 2.996 | 1307 s |

A100 vs A30 on the identical config/seed: rate accounting identical to 4 decimals, PPL differs by 0.023 (cuBLAS/kernel
differences) — this is the reason every arm is run on the one A100. A100 timing: column loops ≈ 37 s/block as on the A30
(the per-column Python loop is latency-bound, not FLOP-bound), Hessian stage ≈ 30 s/block → ~66 s/block, ~18 min quant + ~3 min eval per run.

Timing breakdown (A30): column loops 37 s/block (q_proj 8.3, k_proj 8.0, down_proj 10.5, others 2–3 s) = 597 s total; the remaining ~710 s is the Hessian stage (`consider_dX` doubles the block forwards, `block_v` builds a per-head `X AᵀA Xᵀ` value Hessian from a `[32,2048,2048]` tensor per sample, all fp32 with TF32 hard-disabled in TurboBoA's code) plus the block forward that produces the next block's inputs.

Rate-script note: native W3 lands at 2.996 Huffman bpw (8-symbol code entropy 2.84 bits + 0.156 side info), slightly above the brief's guessed 2.6–2.9 range — the Hessian-weighted grid search uses the full 3-bit range so the code histogram is close to uniform; the accounting itself is verified on synthetic data (dequant(codes) reproduces the stored weights exactly). Native W4 likewise lands at 3.902 Huffman bpw (16-symbol code entropy 3.72 + 0.16 side info + table), above the brief's guessed 3.3–3.6 for the same reason (its per-tensor histogram is a broad bell over all 16 codes, e.g. layer-0 down_proj min/max bin ratio ≈ 1:6.5). Gate (c) is read as: the accounting is right (verified on synthetic data, W3 and W4 sit 0.9 bpw apart as expected); the brief's ranges assumed peakier histograms than TurboBoA's Hessian-weighted scale search produces.

## Phase 1 (seed 0) — nested endpoints through the modified solver
| run | wiki2 PPL | c4-new PPL | nominal bpw | ideal bpw | Huffman bpw | free % | quant time |
|---|---|---|---|---|---|---|---|
| E3 (`e3_s0`, never refine) | **12.011** | 20.115 | 4.148 | 2.966 | 2.996 | 51.4 | 1205 s |
| E4 (`e4_s0`, always refine) | **10.533** | 16.346 | 4.148 | 3.946 | 3.982 | 51.6 | 1092 s |

Gate (a) **PASS**: `scripts/check_e3.py e3_s0 w3_s0` → `codes_E3 == 2·codes_W3` for all 973,078,528 weights, max |scale diff| 0, 0 zero mismatches, 0/112 tensors differ; wiki2 12.011 = 12.011, c4-new 20.115 = 20.115. The patched column loop is therefore an exact no-op when nothing is refined (rate accounting on the 16-symbol alphabet with all-even codes also reproduces 2.9964 bpw to 4 decimals — the 16-entry Huffman table costs only 32 extra bits per tensor). "Free" = coarse elements whose fine code is already `2·q_c` (`e_c == e_f`): 51.4 % model-wide — refinement can only ever change the other ~49 %. Gate (b): E4 = 10.533 vs native W4 = 10.224, a gap of **+0.31 wiki2 PPL** (+0.9 c4-new): E4 rounds on the fine grid `s_c/2, 2 z_c` derived from the 3-bit Hessian-weighted search, whereas native W4 searches its own 4-bit scale; the 3-bit-optimal scale is slightly too coarse for a 16-level grid. This is the expected small gap and it defines the ceiling of the nested scheme (no arm can beat E4). E4's Huffman rate is 3.982 vs W4's 3.902 because the derived fine grid spreads codes more evenly over the 16 symbols (code entropy 3.79 vs 3.72). E3's extra ~150 s of wall time is the per-column Python logging on the nested path (gain/refine/free bookkeeping), not solver work.

## Phase 2 (seed 0) — Fisher mask vs in-loop gain, matched on Huffman bpw
| run | refined % | odd-code % | wiki2 PPL | c4-new PPL | ideal bpw | Huffman bpw | quant time |
|---|---|---|---|---|---|---|---|
| A, Fisher top-1 % per tensor (`A_p1_s0`) | 1.00 | 0.46 | **11.855** | 19.703 | 3.008 | **3.071 = R1** | 1234 s |
| B, in-loop top-k per column, p=1 % (`B_p1_s0`) | 1.00 | 1.00 | **11.782** | 19.513 | 3.027 | 3.076 (R1 + 0.005) | 1456 s |
| C, in-loop global λ = 0.0759 (`C_l0.0759_s0`) | 0.996 | 0.996 | **11.743** | 19.410 | 3.031 | 3.086 (R1 + 0.015) | 1383 s |
| seed 1: A / B / C | 1.00 / 1.00 / 0.984 | | 11.863 / 11.761 / 11.735 | 19.621 / 19.434 / 19.242 | | 3.071 / 3.076 / 3.085 | |

Fisher (`results/fisher/fisher_s0.pt`, 25 s on the A100, 12.7 GB peak): bf16 model, fp32 `Σ_n grad_n²` over the 128 seed-0
calibration sequences, transformer-block linears only. Arm A refines exactly 1.000 % of every tensor, but only 0.46 % of the
codes end up odd: 54 % of the Fisher picks are "free" elements (their fine code is already `2·q_c`), so a static mask spends
half its budget on elements where refinement changes nothing. Arms B/C require `gain > 0`, so at the same refined fraction
they produce ~2× the odd codes; p / λ are therefore matched to R1 on Huffman bpw, not to A's nominal 1 %.

**Arm B at p = 1 % lands at 3.076 Huffman bpw = R1 + 0.005**, inside the ±0.02 window, so no p adjustment was needed. Phase-2
gates: B's PPL (11.782) lies between E3 (12.011) and E4 (10.533) ✓; realized refined fraction 1.000 % ✓; 100 % of refined
elements have gain > 0 ✓ (by construction). Seed 0: **B beats A by 0.073 wiki2 PPL** (0.19 c4-new) at matched Huffman rate.
Caveat to carry into the verdict: B stores 1.00 % odd codes vs A's 0.46 %, so B's *ideal* (Shannon) rate is 0.019 bpw above
A's (3.027 vs 3.008) while the Huffman rates agree to 0.005 — integer Huffman code lengths do not resolve a 0.5 % change in
the odd-symbol mass (all 8 odd symbols get the same length either way). The brief fixes Huffman as the matching axis; both
numbers are reported.

**Jaccard(A-mask, B-mask)** per layer (`results/figs/jaccard_A_B.json`): mean **0.009**, median 0.006, max 0.047 — the
Fisher mask and the in-loop gain pick essentially disjoint sets (chance overlap at 1 % would be ≈ 0.005). Both nevertheless
reduce PPL by 0.16–0.23: the Fisher picks the globally sensitive weights (half of which are already exactly representable),
the in-loop rule picks the weights whose *current* compensated residual sits mid-way between two coarse levels.
Figures (seed 0) in `results/figs/`: `refined_frac_per_layer.png`, `free_frac_per_layer.png`, `jaccard_A_B_per_layer.png`,
`gain_hist_per_layer.png`, `rate_distortion.png`.

## Phase 3 — λ calibration (rate only) and the seed sweep
λ for arm C came from the model-wide CDF of `gain/N_ℓ` logged by `B_p1_s0` (`scripts/calibrate_lambda.py --target_p 0.01`):
973.1 M elements, 48.6 % with gain > 0, quantiles of gain/N_ℓ: 90 % 0.0213, 99 % **0.0759**, 99.9 % 0.231 → **λ = 0.0759**
(expected refined fraction 1.000 %). One C run realized 0.996 % refined and **3.086 Huffman bpw = R1 + 0.015**, inside the
±0.02 window, so no secant adjustment was used (0 of the ≤ 2 allowed). No PPL was looked at when choosing λ.

Allocation under the global λ (refined % by layer type, seed 0): q_proj 0.004, k_proj 0.006, v_proj 1.03, o_proj 1.45,
gate_proj 0.32, up_proj 1.11, down_proj 1.76 (per-block o_proj/down_proj reach 2.5–3.9 % in blocks 6–12; block 15 up/down
get ≈ 0.1 %). Arm B, by construction, refines exactly 1 % of every layer. The global criterion therefore takes the budget
away from q/k almost entirely and spends it on o_proj/down_proj — and still ends 0.04 PPL below B and 0.11 below A. The q/k
gain/N_ℓ values are in the solver's own two-sided loss units (`r_i` from `U_out_sub⁻¹`, `N_ℓ` with the same `H_out`), so this
is a property of the criterion, not a units mismatch; Phase 4(c) (`N_ℓ = 1`) is the designed probe of whether the
normalisation is what makes this work.

## 2→3-bit experiment (seed 0, PI request 2026-09-11)
Same construction one level down: coarse 2-bit grid (codes 0..3, 2-bit zero) nested in a derived 3-bit grid (`s_f = s_c/2`,
`z_f = 2 z_c`, stored codes 0..7). The patch was generalised to `coarse b → fine b+1`, b ∈ {2, 3} (the 3-bit path is
unchanged — `E3` arm at b = 2 means "coarse only", `E4` means "all fine"); `scripts/test_nested_bits.py` checks on synthetic
one- and two-sided layers that the coarse-only endpoint is bit-identical to the stock solver at both widths and that
`s_f (q − z_f)` reproduces the stored weights. Runs (tags `b2_*`): native W2, coarse-only endpoint, all-fine endpoint,
A (Fisher s0, p = 1 %), B (p = 1 %, gains logged), then C at λ from B's CDF. Rate side info: 16-bit scale + 2-bit zero.

**Gate (b) at 2→3 is far from met**: the all-fine endpoint (every element on the derived 3-bit grid) gives wiki2 **18.36** at
3.100 Huffman bpw, against native W3's 12.011 at 2.996 — a **+6.3 PPL** gap, versus +0.31 for the 3→4 case. The reason is
the scale: the 2-bit Hessian-weighted search picks a heavily clipped range (a 4-level grid must trade clipping for
resolution), and halving that scale gives a 3-bit grid with the *same clipped range* and no way to recover the outliers.
So at 2→3 the nesting ceiling is a bad 3-bit model; any 1 %-refined arm sits between 109 (W2) and 18.4, not near native
W3. This bounds what the 2-bit experiment can show: it tests whether the in-loop rule still beats Fisher *inside* the
nested family, not whether nested 2→3 is a competitive 3-bit method (it is not, without a fine-grid scale search, which the
brief forbids because it breaks the nesting).

At matched Huffman rate (2.343 vs 2.346) the in-loop rule wins by a wide margin at 2 bits: **B 37.3 vs A 71.1** wiki2
(c4 93.6 vs 138.8). Caveat: with an 8-symbol alphabet whose 4 even symbols carry ≥ 99 % of the mass, Huffman lengths are
integer-limited and the Huffman rate hardly moves with the odd fraction (Shannon: A 2.140, B 2.164 — B is +0.024 bpw
in ideal rate). A B run at p = 0.45 % (A's odd-code fraction) gives the Shannon-matched comparison: at *strictly lower* rate than A on
both axes (ideal 2.133 vs 2.140, Huffman 2.338 vs 2.343) B still reaches **43.3 vs 71.1** wiki2, so the 2-bit win is not
a rate-matching artefact. **2-bit verdict (seed 0):** at matched Huffman rate C 36.1 < B 37.3 ≪ A 71.1 (E2 109.2, all-fine
ceiling 18.4); same ordering and the same allocation pattern as at 3 bits (C gives q/k ≈ 0.005 %, o/up/down 1.3–1.5 %),
Jaccard(A,B) ≈ 0.01. Absolute quality is poor for the reason above (the family is dominated by native W3 at ≈ 3 bpw), so
the 2-bit result supports the selection-rule claim, not a competitive 2-bit operating point. Figures in `results/figs_2bit/`.

| run | wiki2 PPL | c4-new PPL | ideal bpw | Huffman bpw | refined % | odd % |
|---|---|---|---|---|---|---|
| native W2 (`b2_w2_s0`) | 109.22 | 233.46 | 2.099 | 2.141 | — | — |
| coarse-only endpoint (`b2_e2_s0`) — **bit-identical to W2** (gate a PASS: 0/973M code mismatches) | 109.22 | 233.46 | 2.141 | 2.141 | 0 | 0 |
| all-fine endpoint (`b2_e3_s0`, derived 3-bit grid `s_c/2`) | **18.36** | 31.30 | 3.064 | 3.100 | 100 | 50.0 |
| A, Fisher top-1 % (`b2_A_p1_s0`) | **71.05** | 138.82 | 2.140 | **2.343 = R1(2-bit)** | 1.00 | 0.45 |
| B, in-loop top-k, p = 1 % (`b2_B_p1_s0`) | **37.31** | 93.57 | 2.164 | 2.346 (R1 + 0.003) | 1.00 | 1.00 |
| C, in-loop global λ = 0.2585 (`b2_C_l0.2585_s0`) | **36.05** | 91.46 | 2.165 | 2.349 (R1 + 0.006) | 0.99 | 0.99 |
| B, p = 0.45 % = A's odd fraction (`b2_B_p045_s0`, Shannon-matched) | **43.30** | 125.89 | 2.133 | 2.338 | 0.45 | 0.45 |

## Why the code entropy is high — zero-point alignment test (PI hypothesis, 2026-09-11)
Hypothesis: per-group zero points differ, so the per-tensor histogram is a mixture of peaks at different positions and
looks flat. Test (`scripts/rate_aligned.py`, encoder-only change: Huffman-code the zero-relative symbol `q − z_g`; `z_g` is
already stored side info, so decoding is exact): 

| run | raw code H | zero-aligned code H | raw Huffman bpw | aligned Huffman bpw | per-group conditional H (floor) |
|---|---|---|---|---|---|
| native W3 | 2.817 | 2.793 | 2.996 | 3.005* | 2.72 |
| native W4 | 3.718 | 3.688 | 3.902 | 3.868 | |
| A p=1 % | 2.859 | 2.835 | 3.071 | 3.038 | |
| B p=1 % | 2.878 | 2.858 | 3.076 | 3.056 | |
| C λ=0.0759 | 2.882 | 2.860 | 3.086 | 3.063 | |
| native W2 | 1.958 | 1.925 | 2.141 | 2.223* | 1.81 |
| 2-bit A / B / C | 2.00 / 2.02 / 2.02 | 1.97 / 2.00 / 1.99 | 2.343 / 2.346 / 2.349 | 2.245 / 2.257 / 2.262 | |

(* the aligned alphabet has 2·2^b − 1 symbols, so the integer Huffman lengths are slightly worse for native W2/W3 even though
the entropy is lower.) Alignment recovers only **0.02–0.03 bits/code (≈ 0.03 bpw)**, because the zero points are *already*
aligned: on the 3-bit grid 98 % of all groups have `z_c ∈ {3, 4}` (2-bit: 99.9 % have `z ∈ {1, 2}`) — the asymmetric search
lands at the centre because weight distributions are symmetric. The aligned per-group histogram is a bell, but a **broad**
one: TurboBoA's Hessian-weighted scale search chooses a tight range, ± 2.1 row-σ for 3-bit (step ≈ 0.6 σ, 5.5 % of codes
clipped at each end) and ± 1.6 σ for 2-bit (19 % clipped at each end), so the 8 (resp. 4) levels are all well populated.
The per-group conditional entropy H(q | group) — the floor for *any* coder that adapts to the group — is 2.72 bits for W3
(1.81 for W2), i.e. at most 0.1 bits/code below what we report now. Conclusion: the entropy is set by the quantizer's
scale choice (a distortion-optimal, rate-blind trade-off), not by the coder. Lowering it materially requires an
entropy-constrained scale search (larger scale → peakier histogram → lower rate at some MSE cost), which is a change to the
quantizer, outside the current brief — a candidate follow-up, not a bug in the accounting.

## Per-tensor scale experiment (3→4 bit, seed 0, PI request 2026-09-11)
One (scale, zero) per weight matrix (`--per_tensor --group_size -1`, no `refine_qparam`): tensor-wide min/max, the solver's
shrink grid search scored by the Hessian-weighted error summed over all rows, fixed (s, z) in the column loop. Side info
≈ 0 bpw, so rate = code entropy. Unit test: coarse-only endpoint bit-identical to the native per-tensor solve.

| run | wiki2 PPL | c4-new PPL | ideal bpw | Huffman bpw | refined % |
|---|---|---|---|---|---|
| native W3 per-tensor (`pt_w3_s0`) | **65.03** | 294.9 | 2.454 | 2.520 | — |
| A, Fisher top-1 % (`pt_A_p1_s0`) | **41.98** | 176.4 | 2.485 | **2.550 = R1(pt)** | 1.00 (odd 0.49 %, free 51.1 %) |
| B, in-loop top-k, p = 1 % (`pt_B_p1_s0`) | **43.80** | **145.2** | 2.503 | 2.567 (R1 + 0.017) | 1.00 |

Per-tensor verdict (seed 0, matched Huffman rate): **mixed** — Fisher is 1.8 PPL better on wiki2 (41.98 vs 43.80) while
in-loop is 31 PPL better on c4-new (145.2 vs 176.4); B's Shannon rate is +0.018 bpw above A's. Both are far from usable
(native g=128 W3 is 12.0), and at PPL ≈ 40 a single seed cannot separate a 1.8-PPL wiki2 difference from calibration
noise, whereas the c4 gap is large. The in-loop rule's advantage is therefore not universal at every operating point: at
g=128 (3→4 and 2→3) it wins on both metrics; with one scale per matrix the wiki2 ordering flips while c4 still favours
in-loop. Arm C, endpoints, native W4 and seeds 1/2 were not run for this configuration (time budget).

Per-tensor 3-bit is far worse than g=128 (65.0 vs 12.0): one range per matrix cannot serve outlier rows. Its code entropy
is *lower* (2.45 vs 2.82 bits/code) because the single wide scale makes the histogram peaky — a clean illustration that a
low-entropy stream is not the same as a good quantizer.

## Setup (fixed for all runs)
- Model `unsloth/Llama-3.2-1B` (byte-identical mirror of `meta-llama/Llama-3.2-1B`), bf16. FP16/bf16 wiki2 PPL (ctx 2048) = **9.751** (paper 9.74).
- Solver: TurboBoA at its README defaults for Llama + `g=128`: `--block_v --n_quant_rows 16 --consider_dX --alpha .25 --adaptive_qparam --refine_qparam --qparam_comput Hessian`, damping 1 %, `act_order_col` (forced off by TurboBoA for `group_size != -1`) and `act_order_row` off. `refine_qparam` is a no-op for grouped quantization in the released code ("NOT supported for group-wise quantization yet"), so no scale re-search happens after the column loop starts.
- Calibration: wikitext2 train, 128 × 2048, TurboBoA sampler, seeds 0/1/2. Eval: wikitext2 test (ctx 2048) + c4-new.
- GPU: 1× A100-PCIE-40GB (driver 595.58), torch 2.6.0+cu124, transformers 4.53.0. (Scaffold + A30 cross-check were done on an A30 24 GB.)

## Result table (§6)
(auto-filled below from `results/runs/*`; hand-curated main table follows once Phase 3 is done.)

## Where the pieces live in TurboBoA (implementation notes §8)
- (i) qparam search: `TurboBoA.gptq()` → `quantizer.find_params_H(W[..., g, :], H_in[block], search=True)` — Hessian-weighted grid search over shrink factors, per group, on the row-compensated but column-uncompensated group. Unchanged; runs on the 3-bit grid (`--w_bits 3`), `s_f = s_c/2`, `z_f = 2 z_c` derived.
- (ii) rounding: `fake_quantize(w, scale_group, zero_group, maxq)` in the column loop → replaced by `NestedRounder.round_column` when `--nested_arm != none`.
- (iii) error propagation: `err = (w − q) / U_in[j,j]`, `update = err @ U_in[j, j:] − w @ P[j, j:]` — untouched; `q` is the dequant of the single stored code. Hence `d_j = 1 / U_in[j,j]²` where `U_in = chol(H_in⁻¹, upper)`.
- **Row weight `r_i`**: TurboBoA is two-sided for **q_proj / k_proj** (H_out = RoPE-rotated covariance of the other projection's output, per head), not for v/o as the brief assumed; v_proj gets a per-kv-head block-diagonal `H_in = X Aᵀ A Xᵀ` (one-sided), o_proj and MLP are one-sided. Rows of q/k are quantized in chunks of `n_quant_rows = 16` per head; the solver computes `U_out_sub⁻¹` for the chunk. The exact OBS loss weight of a chunk's error is `tr( (U_subᵀ U_sub)⁻¹ E H_in Eᵀ )`, so we use `r_i = [(U_subᵀ U_sub)⁻¹]_ii = Σ_k (U_sub⁻¹)_ik²` (row sums of squares of the already-computed `U_out_sub_inv`), which reduces to `1/U_out[i,i]²` for `n_quant_rows = 1`. `r_i = 1` for all one-sided layers.
- `N_ℓ` = the solver's own loss functional on `W` itself / numel: `Σ_h tr(W_h H_in,h W_hᵀ)` (one-sided) or `Σ_h tr(H_out,h W_h H_in W_hᵀ)` (q/k), with the damped Hessians the solver uses.

## Prior art (§7) — unchanged from the brief; see CLAUDE_TASK.md §7.


## All runs (auto)

<!-- AUTO-RUNS-START -->
| run | Arm | p / λ | seed | refined % | nominal bpw | ideal bpw | Huffman bpw | wiki2 PPL | c4-new PPL | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_p1_s0 | A [3→4] | p=0.01 | 0 | 1.00 | 4.148 | 3.008 | 3.071 | 11.855 | 19.703 | 1234 |
| A_p1_s1 | A [3→4] | p=0.01 | 1 | 1.00 | 4.148 | 3.008 | 3.071 | 11.863 | 19.621 | 1227 |
| B_p1_s0 | B [3→4] | p=0.01 | 0 | 1.00 | 4.148 | 3.027 | 3.076 | 11.782 | 19.513 | 1456 |
| B_p1_s1 | B [3→4] | p=0.01 | 1 | 1.00 | 4.148 | 3.027 | 3.075 | 11.761 | 19.434 | 1275 |
| C_l0.0759_s0 | C [3→4] | λ=0.07593 | 0 | 1.00 | 4.148 | 3.031 | 3.086 | 11.743 | 19.410 | 1383 |
| C_l0.0759_s1 | C [3→4] | λ=0.07593 | 1 | 0.98 | 4.148 | 3.030 | 3.085 | 11.735 | 19.242 | 1228 |
| b2_A_p1_s0 | A [2→3] | p=0.01 | 0 | 1.00 | 3.141 | 2.140 | 2.343 | 71.052 | 138.817 | 1301 |
| b2_B_p045_s0 | B [2→3] | p=0.0045 | 0 | 0.45 | 3.141 | 2.133 | 2.337 | 43.301 | 125.894 | 1332 |
| b2_B_p1_s0 | B [2→3] | p=0.01 | 0 | 1.00 | 3.141 | 2.164 | 2.346 | 37.305 | 93.568 | 1546 |
| b2_C_l0.2585_s0 | C [2→3] | λ=0.2585 | 0 | 0.99 | 3.141 | 2.165 | 2.349 | 36.053 | 91.455 | 1295 |
| b2_e2_s0 | E2 (coarse only) [2→3] | λ=∞ | 0 |  | 3.141 | 2.099 | 2.141 | 109.221 | 233.458 | 1278 |
| b2_e3_s0 | E3 (all fine) [2→3] | λ=0 | 0 | 100.00 | 3.141 | 3.064 | 3.100 | 18.360 | 31.304 | 1275 |
| b2_w2_s0 | native W2 | — | 0 |  | 2.141 | 2.099 | 2.141 | 109.221 | 233.458 | 1139 |
| e3_s0 | E3 (coarse only) [3→4] | λ=∞ | 0 |  | 4.148 | 2.966 | 2.996 | 12.011 | 20.115 | 1205 |
| e4_s0 | E4 (all fine) [3→4] | λ=0 | 0 | 100.00 | 4.148 | 3.946 | 3.982 | 10.533 | 16.346 | 1092 |
| pt_A_p1_s0 | A [3→4] | p=0.01 | 0 | 1.00 | 4.000 | 2.485 | 2.550 | 41.978 | 176.365 | 987 |
| pt_B_p1_s0 | B [3→4] | p=0.01 | 0 | 1.00 | 4.000 | 2.503 | 2.567 | 43.798 | 145.209 | 1204 |
| pt_w3_s0 | native W3 | — | 0 |  | 3.000 | 2.454 | 2.520 | 65.031 | 294.899 | 805 |
| w3_s0 | native W3 | — | 0 |  | 3.148 | 2.966 | 2.996 | 12.011 | 20.115 | 1059 |
| w3_s0_a30 | native W3 | — | 0 |  | 3.148 | 2.966 | 2.996 | 12.034 | 20.146 | 1307 |
| w4_s0 | native W4 | — | 0 |  | 4.156 | 3.875 | 3.902 | 10.224 | 15.428 | 892 |
<!-- AUTO-RUNS-END -->
