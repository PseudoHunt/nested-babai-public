# CLAUDE.md — read this first (you have no other context)

You are continuing a research experiment that was started on another machine. **Everything you need is in this repo.**
Read, in this order, before touching anything:

1. `CLAUDE_TASK.md` — the full task brief (goal, definitions, algorithm, arms, phases/gates, decision rule, prior art,
   implementation notes, "do not" list, deliverables). It is the authority. Re-read §9 ("Do not") before every design choice.
2. This file — what has been done, what was decided and why, how the code is organised, exact next steps.
3. `results/SUMMARY.md` — the live results document you must update after **every** run (table + one-paragraph verdict).
4. `HANDOFF.md` — environment setup commands and the run order.

## 0. The project in three sentences
We quantise Llama-3.2-1B with TurboBoA (a GPTQ-family PTQ solver) to 3-bit weights on a grid whose codes are embedded in a
4-bit code space (coarse code = 2·q3, same scale), and "refine" a small fraction (~1 %) of elements to the 4-bit grid, so the
stored stream is 4-bit codes with low entropy that Huffman-codes to ~2.6–3.0 bits/weight. Today that ~1 % is chosen by a
static diagonal-Fisher pass (SqueezeLLM style, needs backprop) — **arm A**. The experiment asks whether choosing it **inside
the solver's column loop**, from the solver's own second-order error weight on the error-compensated weight
(`gain = (e_c² − e_f²)·d_j·r_i`, refine iff gain is in the top-k per column [arm B] or `gain/N_ℓ > λ` globally [arm C]),
matches or beats the Fisher mask **at matched entropy-coded bits/weight**. The outcome is decided by the rule in
`CLAUDE_TASK.md` §4 Phase 3 (PASS / PARTIAL / FAIL) over calibration seeds 0, 1, 2.

## 1. What has been done so far (previous machine: 1× A30 24 GB)
- Repo scaffold, vendored TurboBoA (`turboboa/`, CC BY-NC 4.0) with a **flag-guarded** patch (see §3). `--nested_arm none`
  (the default) is the unmodified solver.
- Unit tests on synthetic layers (one-sided, per-head-`H_in` like v_proj, two-sided row-chunked like q/k): **E3 is
  bit-identical to stock W3** (weights, codes, scales); E4 error sits between W3 and W4; `s_f·(q4 − z_f)` reproduces the stored
  weights exactly; arms A/B/C/D run with correct shapes; ~52 % of elements are "free" (`q_f == 2·q_c`).
- FP16/bf16 baseline: wiki2 PPL **9.751** (paper 9.74), c4-new 14.02 → model + eval path are correct.
- Phase 0, native W3 g128 seed 0 (A30): wiki2 **12.034**, c4-new 20.146, nominal 3.148 / ideal 2.966 / Huffman **2.996** bpw,
  quant time 1307 s. Treat this as a cross-check only — **rerun it on the A100** because cuBLAS kernels differ across GPUs
  and all arms must be compared on one device.
- Nothing else has run. No Fisher yet. No nested arms on the real model yet.

## 2. Decisions already made (with reasons) — do not re-litigate, but do report them in SUMMARY.md
- **Model**: `unsloth/Llama-3.2-1B` (ungated mirror, byte-identical tensors to `meta-llama/Llama-3.2-1B`; FP PPL matches the
  paper). Local path must contain lowercase `llama` (TurboBoA's `get_model` checks `'llama' in path`, and its cache names use
  `path.split('/')[-1].split('-')[0]`).
- **Solver config** ("TurboBoA's Llama defaults", README command + g=128): `--w_bits 3 --group_size 128 --block_v
  --n_quant_rows 16 --consider_dX --alpha .25 --adaptive_qparam --refine_qparam --qparam_comput Hessian`, damping 1 %,
  `act_order_col` off (TurboBoA forces it off for `group_size != -1`), `act_order_row` off. This is baked into
  `scripts/run.sh`. Do not change any of it (§9).
- Facts about the released TurboBoA code that matter: `refine_qparam` prints "NOT supported for group-wise quantization yet"
  and is a **no-op** with g=128 (so no scale re-search happens after the column loop starts — §8 item 2 is moot);
  `adaptive_qparam` is effectively always on for the two-sided path because `gptq()` re-searches qparams per row-chunk on the
  row-compensated weights regardless of the flag. The qparam search is a **Hessian-weighted** grid search
  (`qparam_comput=Hessian`, the default), not plain MSE — we use it as-is, on the 3-bit grid only.
- **`d_j = 1 / U_in[j,j]²`** where `U_in = chol(H_in⁻¹, upper)` — the solver computes `err = (w − q)/U_in[j,j]` and the
  per-element loss increment is `err²`. Found at `turboboa/quantizers/turboboa.py`, `gptq()`.
- **`r_i` (row weight)**: TurboBoA is two-sided for **q_proj and k_proj** (H_out = per-head, RoPE-rotated covariance of the
  *other* projection's output), **not** for v/o as the brief guessed. v_proj gets a per-kv-head block-diagonal
  `H_in = X AᵀA Xᵀ` (one-sided); o_proj and MLP are one-sided. For q/k, rows are quantised in chunks of 16 per head and the
  solver already forms `U_sub⁻¹` (`U_out_sub_inv`). The exact OBS loss weight of a chunk's error matrix E is
  `tr((U_subᵀU_sub)⁻¹ · E H_in Eᵀ)`, so **`r_i = [(U_subᵀU_sub)⁻¹]_ii = Σ_k (U_sub⁻¹)_ik²`** (row sums of squares of
  `U_out_sub_inv`; reduces to `1/U_out[i,i]²` for `n_quant_rows = 1`). `r_i = 1` for all one-sided layers. This is
  implemented; it must be **stated in SUMMARY.md** (brief §1).
- **`N_ℓ`** = the solver's own loss functional applied to W itself / numel: `Σ_h tr(W_h H_in,h W_hᵀ)` (one-sided) or
  `Σ_h tr(H_out,h W_h H_in W_hᵀ)` (q/k), using the damped Hessians the solver uses (so `gain/N_ℓ` is dimensionless in every
  layer, including two-sided ones).
- **Arm B top-k**: per column, over all `n_heads × rows` entries present in the `gptq()` call (whole tensor for one-sided
  layers; a 16-row × n_heads chunk for q/k), with `k_j = ⌊p·n·(j+1)⌋ − ⌊p·n·j⌋` (integer accumulator, so the mean is exactly p),
  and `gain > 0` required.
- **Arm A mask**: per-tensor top-p by Fisher (`fisher_topk_mask` in `turboboa/quantize.py`), p = 0.01 default (brief says
  "confirm per-tensor vs global with the PI; default per-tensor").
- **Fisher**: `scripts/fisher.py`, bf16 model (its native dtype; the brief says FP16 — bf16 is what TurboBoA loads and the
  difference is immaterial, note it), fp32 `Σ_n grad_n²` accumulation over the 128 seed-0 calibration sequences, loss = mean
  next-token CE with labels = inputs, only transformer-block linear weights.
- **Rate accounting** (`scripts/rate.py`, solver-independent): per tensor one Huffman table over the 2^bits alphabet
  (8 symbols for native W3, 16 otherwise) + table bits (2^bits × 4) + 16-bit scale and 3-bit coarse zero per group
  (4-bit zero for native W4). Nominal = bits + (16 + zero_bits)/128. Reports nominal / ideal (Shannon) / Huffman; **match arms
  on Huffman**. Native W3 came out at 2.996 Huffman (code entropy 2.84 + 0.156 side info), slightly above the brief's guessed
  2.6–2.9 — the Hessian-weighted grid search gives a near-uniform 3-bit histogram. The accounting is verified; report it.
- **Matching rule**: R1 = Huffman bpw of arm A at p = 1 % (seed 0). B adjusts p, C adjusts λ, to R1 ± 0.02. λ / p are tuned on
  **rate only, never on PPL** (§9).
- Eval: TurboBoA's `evaluate()` reports wikitext2 (the decision metric) and c4-new (extra, keep it).

## 3. The code patch — file by file (all guarded by `--nested_arm`)
- `turboboa/quantizers/nested.py` — `NestedRounder`: `begin_layer(W, H_in, H_out, mask, fisher)` computes `N_ℓ`;
  `round_column(w, s_c, z_c, d_j, r_i, col, sub)` computes `q_c, q_f, deq_c, deq_f, e_c, e_f, gain`, applies SELECT for the arm,
  returns `deq, q4, gain, refine, free`. This is the only place the rounding rule lives.
- `turboboa/quantizers/turboboa.py` — `__init__` reads `opts['nested']`; `quant()` calls `begin_layer`, and after the solve
  stores integer codes (`q4` on the nested path, `q` on the stock path), `scale`, `zero`, `refine`, `free`, `gain/N_ℓ`, `N` into
  `self.dump`; `gptq()` calls `round_column` instead of `fake_quantize` when nested (error propagation line untouched) and
  collects per-element logs; `turboboa()` passes `row_w = (U_out_sub_inv**2).sum(-1)` per chunk.
- `turboboa/quantize.py` — attaches the Fisher mask/weights per layer (arms A/D), times each layer, and `save_layer_dump()`
  writes `results/runs/<tag>/layers.<block>.<name>.npz` (+ `layers.jsonl` row) for **every** run, nested or not.
- `turboboa/utils/process_args.py` — new args `--nested_arm {none,E3,E4,A,B,C,D} --nested_p --nested_lambda --fisher_path
  --dump_dir --log_percol`; builds the `NestedRounder`.
- `turboboa/main.py` — loads the Fisher dict for A/D, writes `result.json` (config + PPLs + time) into `--dump_dir`.
- `scripts/run.sh TAG SEED WBITS [extra]` → `results/runs/TAG/` then `rate.py` then `summarize.py`.
  `scripts/rate.py`, `scripts/fisher.py`, `scripts/calibrate_lambda.py` (λ from the arm-B gain CDF; `--secant` for the
  rate-only adjustment), `scripts/figs.py` (all §5 figures into `results/figs/`), `scripts/summarize.py` (refreshes the
  auto table between `<!-- AUTO-RUNS-START/END -->` in SUMMARY.md and writes `results/runs_table.md`).
- Per-run outputs to commit: `result.json`, `rate.json`, `layers.jsonl`, `log.txt`. The `.npz` dumps are git-ignored
  (GBs) — keep them locally, they feed `figs.py` and `calibrate_lambda.py`.

## 4. Exact next steps (see HANDOFF.md §1 for environment, §2 for the commands)
Phase 0: `w3_s0`, `w4_s0` → gate: both finish, time known. Phase 1: `e3_s0` (must equal `w3_s0` — check codes == 2·codes_W3,
identical scale/zero, identical PPL), `e4_s0` (PPL ≤ W4 + small margin; report the gap), rate ranges. Phase 2: Fisher →
`A_p1_s0` (defines R1) → `B` at p to hit R1 → figures (refined fraction per layer A/B, free fraction, Jaccard(A,B), gain
histograms, rate–distortion). Phase 3: λ from B's gain CDF → `C` at R1 (≤ 2 secant adjustments) → seeds 1, 2 for A, B, C at
the seed-0 p/λ (report bpw drift) → fill the §6 table → apply the §4 decision rule → verdict in SUMMARY.md. Phase 4 only if
PASS/PARTIAL: arm D (3 seeds), R2 point (p ≈ 4 %) for A/B/C seed 0, C with `N_ℓ = 1` seed 0. Then **stop**.

## 5. Working rules
- Commit + push after **every** run (`git add -A && git commit && git push`); update `results/SUMMARY.md` every time
  (table + verdict paragraph). `gh` is authenticated as PseudoHunt on the old machine; on the new one use whatever git
  credentials are available, the remote is `https://github.com/PseudoHunt/nested-babai` (private).
- Report progress to the user in short messages: what ran, the number, what is next.
- Never edit `scripts/run.sh` while a run is in flight (bash reads scripts incrementally → spurious syntax errors).
- If `datasets` complains "Couldn't find cache for allenai/c4 for config …" in offline mode, run once online to build
  `turboboa/cache/testloader_llama_c4-new_2048.cache`; afterwards everything is read from TurboBoA's own cache files.
- Do not run 3B/8B, do not change solver defaults, do not tune on PPL, do not re-search scales on the 4-bit grid,
  do not average coarse/fine errors (all §9).
