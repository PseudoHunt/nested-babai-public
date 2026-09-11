# CLAUDE_TASK — In-loop nested-lattice refinement inside the TurboBoA column loop

Self-contained brief. Assume zero prior context. Read fully before touching the repo.

---

## 0. Goal in one paragraph

We have a working mixed-precision PTQ pipeline where ~1% of weights are stored at 4-bit and the rest at 3-bit, with the 3-bit weights embedded in the 4-bit code space (code = 2 × q3, shared scale). The resulting 4-bit code stream has low entropy, so Huffman coding brings the stored size down to ~2.6 bits/weight while perplexity sits closer to pure W4 than pure W3. Today the 1% is chosen by a **static** SqueezeLLM diagonal-Fisher pass (needs a backward pass over the full model) run **before** quantization. This task replaces that with an **in-loop** decision made inside the GPTQ/TurboBoA column solve, using the solver's own second-order error weight and the **error-compensated** weight at the moment it is quantized. Score = (e_c² − e_f²) × solver weight, refine iff score > λ. No gradients, no extra calibration pass. The experiment decides whether this in-loop rule matches or beats the static Fisher mask at the same entropy-coded bits/weight.

If it beats Fisher by more than seed noise → method paper. If it only matches → still useful (removes the gradient pass) but goes into the BoA dissection paper as a section. If it loses → fold the negative into the dissection paper.

---

## 1. Definitions (use these names in code)

- **Model**: Llama-3.2-1B (HF `meta-llama/Llama-3.2-1B`, gated; token in `HF_TOKEN`). Do not use larger models in this task.
- **Solver**: TurboBoA, `github.com/SamsungLabs/TurboBoA` (supports OPT and Llama). Default calibration, damping, block size, act-order, and group size **g = 128** exactly as its Llama scripts do. Do not change these.
- **Coarse grid Λc** (3-bit): per group of 128 input columns, scale `s_c`, zero `z_c ∈ {0..7}`. `q_c = clamp(round(w/s_c) + z_c, 0, 7)`, `deq_c = s_c·(q_c − z_c)`.
- **Fine grid Λf** (4-bit): `s_f = s_c / 2`, `z_f = 2·z_c`. `q_f = clamp(round(w/s_f) + z_f, 0, 15)`, `deq_f = s_f·(q_f − z_f)`.
- **Stored code**: always a 4-bit integer `q4 ∈ {0..15}`. Coarse weights store `q4 = 2·q_c` (even codes). Fine weights store `q4 = q_f`. Dequant for every weight is `s_f·(q4 − z_f)`. One scale and one zero per group; **no per-element mask is stored** — the mask is implicit in the code stream.
- **Scale/zero search**: run TurboBoA's existing qparam search (MSE grid search, and its adaptive/refine step if the script enables it) **on the 3-bit grid only**, on the original (uncompensated) group. Derive `s_f, z_f` from that. Never re-search on the 4-bit grid; that would break the nesting.
- **Errors** at column `j`, row `i`, on the compensated weight `w`: `e_c = w − deq_c`, `e_f = w − deq_f`.
- **Solver weight `d_j`**: the quantity TurboBoA/GPTQ already uses to turn a rounding error into a loss increment for column `j`. In vanilla GPTQ this is `1 / L[j,j]²` where `L` is the Cholesky factor of `H_in⁻¹` (the code computes `err = (w − q) / L[j,j]`, so per-element loss is `(w − q)² / L[j,j]²`). **Locate the exact expression in TurboBoA's solver and use it as-is**, so the gain is in the solver's own loss units.
- **Row weight `r_i`**: for layers where TurboBoA uses an attention-aware row metric `H_out` (v_proj / o_proj and any others its code treats two-sidedly), the per-element loss also carries a row factor `1 / [H_out⁻¹]_ii` (Kronecker structure ⇒ diagonal of the inverse is the product). If the solver exposes it, use it. If TurboBoA handles row coupling only by quantizing `n_quant_rows` rows jointly and exposes no per-row weight, set `r_i = 1` and **say so in the summary**. `r_i = 1` for all identity-`H_out` layers (all MLP layers, q/k if TurboBoA treats them one-sidedly).
- **Gain**: `gain_ij = (e_c,ij² − e_f,ij²) · d_j · r_i`. Non-negative by construction except for clamp edge cases; treat negative as 0.
- **Layer normalizer `N_ℓ`** (arm C only): `N_ℓ = tr(W_ℓ H_in Wᵀ_ℓ) / (d_row·d_col)`, computed once per layer from the unquantized `W_ℓ` and the same `H_in` the solver builds. Gain and `N_ℓ` are in the same units, so `gain/N_ℓ` is dimensionless.
- **Nominal bpw**: `3 + p_eff` where `p_eff` = fraction of weights stored on odd codes... no — nominal bpw is not used for matching. Only the next line is.
- **Entropy-coded bpw (the rate axis, the only one used for matching)**: per tensor, histogram the `q4` stream, build one Huffman table (16 symbols), coded bits + table bits (≤ 16 code lengths × 4 bits); add group scales (16 bits each) and coarse zeros (3 bits each). Sum over all quantized tensors, divide by weight count. Also report the ideal Shannon entropy of each stream (same accounting, entropy instead of Huffman lengths). Report both; match arms on the Huffman number.

---

## 2. The algorithm (what changes in the solver)

Only the rounding rule inside the column loop changes. Everything else — Hessian build, Cholesky, act-order, block updates, TurboBoA's dX correction and adaptive qparams — stays exactly as shipped.

```
for each layer ℓ (in TurboBoA's order):
    build H_in (and H_out where TurboBoA does)
    per group: search (s_c, z_c) on the 3-bit grid; s_f = s_c/2; z_f = 2·z_c
    N_ℓ = tr(W H_in Wᵀ)/(d_row·d_col)              # arm C only
    for each column j in solver order (compensated W):
        w    = W[:, j]
        q_c, deq_c, q_f, deq_f, e_c, e_f  as defined above
        d_j  = solver's OBS weight for column j
        gain = (e_c² − e_f²) · d_j · r            # vector over rows
        refine = SELECT(gain, arm)                 # boolean vector over rows
        q4   = where(refine, q_f, 2·q_c)
        deq  = s_f · (q4 − z_f)
        err  = (w − deq) / L[j,j]                  # the SINGLE chosen value's error
        propagate err exactly as the solver already does
        log gain (float16), refine (bool), for this column
```

`SELECT` per arm:

- **Arm A (static Fisher)**: `refine[i] = M[i, j]` where `M` is a precomputed per-tensor mask: top `p` fraction of elements by SqueezeLLM diagonal Fisher `F_ij = Σ_n (∂L_n/∂w_ij)²` over the same calibration set. `p = 0.01` per tensor. (Confirm with the PI whether the existing pipeline's 1% is per-tensor or global; default per-tensor.)
- **Arm B (in-loop, fixed per-layer p)**: `refine[i] = gain[i] > 0 and gain[i] in top-k of this column`, `k = round(p · d_row)`. If TurboBoA processes rows in chunks (`n_quant_rows`), take top-k within the chunk with `k` scaled to the chunk size.
- **Arm C (in-loop, global λ)**: `refine[i] = gain[i] / N_ℓ > λ`, one `λ` for the whole model.
- **Arm D (hybrid, optional)**: as C but `gain[i] ← gain[i] · F_ij / mean(F_ℓ)`.
- **Endpoints**: **E3** = never refine (`λ = ∞`); **E4** = always refine (`λ = 0`). Both run through this same modified solver.

Do **not** propagate an average of the coarse and fine errors (that is MatGPTQ's rule for a different objective). Propagate the error of the value actually stored.

### Calibrating λ (arm C) without repeated full runs

1. From arm B's logged gains (already normalized by `N_ℓ`), build the model-wide CDF of `gain/N_ℓ` over all elements.
2. Pick `λ` at the quantile that gives expected refined fraction = target `p`.
3. Run arm C once. Measure realized entropy-coded bpw.
4. If bpw is outside target ± 0.02, adjust `λ` by one secant step on (λ → bpw) and rerun. At most two adjustments. Budget ≤ 3 full runs per seed for calibration; reuse the calibrated `λ` across seeds (re-check bpw per seed; report drift).

λ is calibrated on **rate only**, never on test perplexity.

---

## 3. Arms and matching

All arms are compared at **matched entropy-coded bpw** (target ± 0.02). Target rate **R1** = the bpw that arm A produces at `p = 1%` (seed 0). Arms B and C adjust `p` / `λ` to land on R1. If time allows, a second point **R2** at arm A `p = 4%`.

| Arm | Selection | Where decided | Extra cost |
|---|---|---|---|
| E3 | none | — | none |
| E4 | all | — | none |
| A | Fisher top-p, static | before solver | one full backward pass |
| B | gain top-k per column | in loop | ~0 |
| C | gain/N_ℓ > λ, global | in loop | ~0 (+ λ calibration runs) |
| D | (gain·F) > λ | in loop | backward pass |
| A-ref | the existing (reference) pipeline number | — | reference row only, different solver |

Seeds = calibration sample seeds `0, 1, 2` (TurboBoA's sampler; change only the seed). Same seed ⇒ same calibration set for every arm.

---

## 4. Phases, gates, budget (total target 15–25 GPU-hours on one 24–48 GB GPU)

**Phase 0 — baseline reproduction (~1 GPU-h).**
Set up TurboBoA. Run its Llama scripts on Llama-3.2-1B at W3 g128 and W4 g128, seed 0. Record wiki2 PPL (ctx 2048) and wall-clock per run. If the repo has Llama-3.2-1B reference numbers, match them within noise; if not, record ours as the reference. **Gate**: both runs complete; per-run time known.

**Phase 1 — nested grid + rate script (~1 GPU-h).**
Implement the modified rounding rule and logging. Run E3 and E4 (seed 0). Implement the entropy-coded bpw script and run it on E3, E4, and the Phase-0 W3/W4 checkpoints.
**Gates**: (a) E3 must equal native W3 to within seed-0 reproducibility (it should be nearly bit-identical if the 3-bit qparam search is the same path — if it is not, find out why before continuing). (b) E4 PPL must be ≤ native W4 + small margin (E4 uses `s_c/2`, not a native 4-bit search, so a small gap is expected; report it). (c) Rate script gives bpw for native W3/W4 in the ranges ~2.6–2.9 and ~3.3–3.6 respectively; if not, the accounting is wrong.

**Phase 2 — pipeline sanity (~3 GPU-h).**
Compute Fisher once (seed-0 calibration set, FP16 model, accumulate squared grads per linear weight; fp32 accumulator ≈ 4 GB). Run arm A at `p = 1%` (defines R1). Run arm B at `p = 1%`, adjust `p` if needed to hit R1. Seed 0 only.
**Gates**: B's PPL lies between E3 and E4; realized refined fraction ≈ target; `gain > 0` for the large majority of refined elements; logs and figures produced (see §5). Report the overlap between B's refined set and A's mask (Jaccard per layer) — this is diagnostic, not a gate.

**Phase 3 — the kill gate (~8 GPU-h).**
Calibrate `λ` for arm C (§2). Run A, B, C at R1 for seeds 0, 1, 2. Fill the table in §6.
**Decision** (μ = mean over 3 seeds, σ = pooled std of A/B/C):
- **PASS (method paper)**: `μ_C − μ_A ≤ −max(2σ, 0.05)` wiki2 PPL at matched bpw.
- **PARTIAL (dissection-paper section)**: `|μ_B − μ_A| ≤ σ` or `|μ_C − μ_A| ≤ σ` — in-loop matches Fisher at zero gradient cost.
- **FAIL**: both B and C worse than A by more than σ.

**Phase 4 — only if Phase 3 is PASS or PARTIAL (~4 GPU-h).**
(a) Arm D at R1, 3 seeds. (b) Second rate point R2 (`p ≈ 4%`) for A, B, C, seed 0 only, to draw the two-point rate–distortion line against E3/E4. (c) Arm C with `N_ℓ = 1` (no normalization), seed 0, to show whether normalization is what makes global λ work.

Stop after Phase 4 regardless of outcome. Do not start 3B/8B runs.

---

## 5. Logging and figures (write to `results/`)

Per run: config JSON, per-layer table (refined fraction, ideal entropy, Huffman bpw, `N_ℓ`, wall time), model-level bpw (nominal, ideal, Huffman), wiki2 PPL, seed.
Per column (arms B/C/D, seed 0 only, float16 to keep size down): gain vector and refine mask, saved per layer as `.npz`.
Figures (seed 0): per-layer refined fraction for A vs B vs C; gain histogram per layer (log x); fraction of coarse elements with `e_c == e_f` (already on an even bin — these are the "free" weights Fisher cannot see); Jaccard(A-mask, B-mask) per layer; rate–distortion scatter (bpw vs PPL) with E3, E4, native W3/W4, A, B, C.

`results/SUMMARY.md` holds the current table and a one-paragraph verdict, updated after **every** run. Push to the repo after every run; do not batch.

---

## 6. Result table template

| Arm | p / λ | seed | nominal bpw | ideal bpw | Huffman bpw | wiki2 PPL | time |
|---|---|---|---|---|---|---|---|
| native W3 | — | 0 | 3.16 | | | | |
| native W4 | — | 0 | 4.16 | | | | |
| E3 | ∞ | 0 | | | | | |
| E4 | 0 | 0 | | | | | |
| A | 1% | 0,1,2 | | | | | |
| B | p→R1 | 0,1,2 | | | | | |
| C | λ→R1 | 0,1,2 | | | | | |
| D | λ→R1 | 0,1,2 | | | | | |
| A-ref | 1% | — | | | 2.6 (Huffman vs 4-bit: 35%) | closer to W4 than W3 | — |

(nominal bpw column includes scale/zero overhead at g=128: +0.16.)

---

## 7. Prior art — what this must be different from (for SUMMARY.md and later the paper)

- **MatGPTQ** (Kleinegger, Crnčević, Alistarh, Feb 2026, `github.com/IST-DASLab/MatGPTQ`). Read on 11 Sept 2026. Mechanism: a single master-bit-width code per element, chosen by brute force over all 2^c candidates to minimize the λ_r-weighted **sum** of reconstruction errors across every target bit-width in R (e.g. {3,4,8}); MSB slicing gives the lower-bit models; the error propagated through the GPTQ update is the **average** over target bit-widths; heterogeneous precision is **per-layer** via EvoPress search; solver is one-sided GPTQ. Their MSB slice `S(q,r) = round(q/2^{c−r})·2^{c−r}` is the same nested lattice as ours (3-from-4 = even codes). What they do **not** do: choose a bit-width per element, keep a per-element mask, or use the nested structure for entropy coding. So the paper does not shrink; MatGPTQ-EP at ~3.0–3.25 bpw becomes a baseline row at paper stage (not in this task).
- **SpQR**: detects outliers **in-loop** from the OBS error inside GPTQ, but stores them FP16 sparse with an explicit index. We keep everything in one code stream with one scale, and our score is `e_c² − e_f²`, not `e_c²`.
- **SqueezeLLM**: sensitivity-based mixed precision, static diagonal Fisher, outside the solver. This is arm A.
- **NestQuant**: nested E8 lattice, fixed for the whole tensor, no per-element selection.
- **Our square**: per-element, in-loop, on the compensated residual, two-sided where the solver is two-sided, one global λ for allocation, mask implicit in an entropy-coded stream.

---

## 8. Implementation notes for TurboBoA

1. First find the three things in the solver: (i) the qparam search call, (ii) the rounding call inside the column loop, (iii) the line that turns `(w − q)` into the propagated error. Only (i) gets restricted to 3-bit; only (ii) is replaced; (iii) is untouched.
2. Check whether TurboBoA's `consider_dX` and adaptive-qparam refinement touch the grid after the loop starts. If the refinement step re-searches scales after some columns are quantized, it must re-search on the **3-bit** grid and re-derive `s_f, z_f`; refined elements' stored codes stay valid because dequant is always through `s_f, z_f`.
3. Row factor `r_i`: search for where `H_out` (or the attention-aware Hessian) enters the error weighting for v/o projections. If the diagonal of its inverse is available (or cheaply computable once per layer), use it. Otherwise `r_i = 1` and document.
4. Keep the modified solver behind a flag so native W3/W4 runs are unchanged.
5. Fisher (arm A): FP16 forward, fp32 gradient accumulation, `F += grad²` per linear weight, over the same 128 calibration sequences; skip embeddings and lm_head (they are not quantized here).
6. Rate script must be solver-independent: input = dict of `q4` int tensors + group scales/zeros; output = the three bpw numbers. Run it on native W3/W4 too (native W3 codes are 0..7 — treat them as an 8-symbol alphabet for its own Huffman table; that is the fair rate for pure W3).

---

## 9. Do not

- Do not tune `λ` or `p` on wiki2 test perplexity. Rate only.
- Do not change act-order, damping, block size, calibration data, or group size from TurboBoA defaults.
- Do not average coarse/fine errors in propagation.
- Do not re-search scales on the 4-bit grid.
- Do not add MatGPTQ's "push-up" bit trick; our coarse codes are exactly `2·q_c`.
- Do not report compression "vs uncompressed 4-bit". Report bpw.
- Do not run anything larger than Llama-3.2-1B.
- Do not open a fourth live direction. Arms A–D and the R2 point are the whole task.

---

## 10. Deliverables

- Private repo (the PI names it; suggested `nested-babai`), or a branch of the existing BoA-extension repo. Structure: `turboboa/` (submodule or vendored, with the flag-guarded patch), `scripts/` (one script per arm + `rate.py` + `fisher.py`), `results/` (per-run JSON, `SUMMARY.md`, `figs/`), `CLAUDE_TASK.md` (this file).
- `results/SUMMARY.md` with the §6 table filled, the Phase-3 verdict stated as PASS / PARTIAL / FAIL with the numbers, and the `r_i` handling stated explicitly.
- Push after every run. Report progress in short messages: what ran, the number, what is next.
