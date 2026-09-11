# nested-babai — in-loop nested-lattice mixed precision for entropy-coded 3-bit LLM weights

Quantise Llama-3.2-1B with TurboBoA (a GPTQ-family PTQ solver) to a 3-bit grid nested in a 4-bit grid
(coarse code = 2·q3, same scale/zero), and refine ~1 % of the elements to the 4-bit grid. The stored stream is one
4-bit code stream with an implicit mask, entropy-coded to ≈ 3.07 bits/weight including all side information.
The refinement decision is made **inside the solver's column loop** from the solver's own second-order error weight on the
error-compensated weight (`gain = (e_c² − e_f²)·d_j·r_i`, two-sided where the solver is two-sided), and is compared at
matched entropy-coded rate with a static SqueezeLLM-style diagonal-Fisher mask.

**Result (Llama-3.2-1B, TurboBoA g=128, matched Huffman rate ≈ 3.07–3.09 bpw, wiki2 PPL, seeds 0/1):**
Fisher mask 11.859 · in-loop top-k 11.772 · in-loop global λ **11.739** (pure 3-bit 12.011, all-4-bit-grid 10.533).
Also in `results/SUMMARY.md`: the 2→3-bit experiment (in-loop 36.1 vs Fisher 71.1 wiki2 at matched rate), a zero-aligned
entropy-coding analysis (`scripts/rate_aligned.py`), and a per-tensor-scale experiment (`--per_tensor --group_size -1`:
1 % refinement takes per-tensor W3 from 65.0 to 42.0 wiki2 for +0.03 bpw).

## Layout
- `turboboa/` — vendored [TurboBoA](https://github.com/SamsungLabs/TurboBoA) (CC BY-NC 4.0, see `turboboa/LICENSE`) with a
  flag-guarded patch: `--nested_arm {none,E3,E4,A,B,C,D}`; `none` is the unmodified solver. The rounding rule lives in
  `turboboa/quantizers/nested.py`.
- `scripts/` — `run.sh` (one run), `fisher.py` (diagonal Fisher for arm A/D), `rate.py` / `rate_aligned.py` (solver-independent
  entropy-coded rate accounting), `calibrate_lambda.py` (rate-only λ calibration for arm C), `figs.py`, `summarize.py`,
  `check_e3.py` (bit-identity gate), `test_nested_bits.py` (synthetic unit test, 2- and 3-bit coarse grids).
- `results/` — per-run `result.json`, `rate.json`, `layers.jsonl`, `log.txt` under `results/runs/<tag>/`, figures, and
  `SUMMARY.md` (tables, gates, verdicts). Per-layer code dumps (`.npz`) are not committed.
- `CLAUDE_TASK.md` — the experiment brief; `CLAUDE.md` / `HANDOFF.md` — working notes and the exact run order.

## Reproduce
See `HANDOFF.md` §1 (environment: torch 2.6.0+cu124, transformers 4.53.0, datasets<4; `unsloth/Llama-3.2-1B` in a local
directory whose path contains lowercase `llama`; edit `env.sh`) and §2 (run order). Each run is
`scripts/run.sh TAG SEED WBITS [--nested_arm ...]`, ≈ 20 min quant + 3 min eval on one A100 40 GB.

## License
`turboboa/` is Samsung Labs' TurboBoA under CC BY-NC 4.0 (non-commercial, attribution). The license for the additions in
this repository (`scripts/`, the nested patch, results) has not been set yet — all rights reserved until it is.
