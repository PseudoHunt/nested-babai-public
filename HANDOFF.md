# HANDOFF — continue on the A100 40 GB

**Start with `CLAUDE.md`** (full context: goal, what is done, decisions, code map, next steps) and `CLAUDE_TASK.md` (the brief).
This file is only the environment recipe and the command list.

State when this was written (2026-09-11, A30 session): repo scaffold complete, solver patch unit-tested
(E3 bit-identical to stock W3 on one-sided, per-head-H_in and two-sided/row-chunked layers), Phase 0 W3 s0 run on the A30
(wiki2 12.034). Nothing else has been run. Nothing is running on the A30 any more.

## 1. Environment (≈10 min)
```bash
git clone https://github.com/PseudoHunt/nested-babai && cd nested-babai
python3 -m venv venv && source venv/bin/activate && pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # cu124 wheels run fine on cu12/13 drivers
pip install "transformers==4.53.0" "datasets<4" accelerate matplotlib scipy numpy pandas safetensors sentencepiece protobuf
```
Edit `env.sh`: `source` the venv, set `HF_HOME` to wherever you want the HF cache, `NB_ROOT` to the repo root, and
`LLAMA_PATH` to a **local directory whose path contains lowercase `llama`** (TurboBoA's `get_model` checks `'llama' in path`
and names its calibration cache after `path.split('/')[-1].split('-')[0]`). E.g.
```bash
huggingface-cli download unsloth/Llama-3.2-1B --local-dir /data/models/llama-3.2-1b     # ungated, byte-identical to meta-llama/Llama-3.2-1B
#   (or meta-llama/Llama-3.2-1B with HF_TOKEN set — same tensors)
```
Datasets: `wikitext` (wikitext-2-raw-v1, train+test) and `allenai/c4` `en/c4-validation.00000-of-00008.json.gz` are pulled by
TurboBoA on first use; leave `HF_HUB_OFFLINE` unset in `env.sh` for the first run, then TurboBoA caches
`turboboa/cache/testloader_llama_{wikitext2,c4-new}_2048.cache` and `calib_llama_wikitext2_128_2048_{seed}.cache`.
The calibration sampler is deterministic per seed (verified byte-identical against caches from an earlier machine).

Sanity: `cd turboboa && python main.py --llm_path $LLAMA_PATH --eval_fp` → wikitext2 9.751, c4-new 14.02.

## 2. Run order (each run: `scripts/run.sh TAG SEED WBITS [args]`; ~1 min model load + quant + ~2 min eval)
Do **not** edit `scripts/run.sh` while a run is in flight (bash reads scripts incrementally).
```bash
scripts/run.sh w3_s0 0 3                                   # Phase 0 (rerun on the A100 so all numbers share one GPU)
scripts/run.sh w4_s0 0 4
scripts/run.sh e3_s0 0 3 --nested_arm E3                   # Phase 1: gate (a) E3 == w3_s0 (codes/scales identical, PPL identical)
scripts/run.sh e4_s0 0 3 --nested_arm E4                   #          gate (b) E4 PPL <= W4 + small margin
python scripts/fisher.py --calib_cache turboboa/cache/calib_llama_wikitext2_128_2048_0.cache --out results/fisher/fisher_s0.pt   # Phase 2, ~5 min, ~17 GB
scripts/run.sh A_p1_s0 0 3 --nested_arm A --nested_p 0.01 --fisher_path $NB_ROOT/results/fisher/fisher_s0.pt    # defines R1 = its Huffman bpw
scripts/run.sh B_p1_s0 0 3 --nested_arm B --nested_p 0.01 --log_percol                                 # adjust p to hit R1 ± 0.02
python scripts/calibrate_lambda.py results/runs/B_p1_s0 --target_p <realized p of B at R1>             # λ for arm C (rate only)
scripts/run.sh C_l<λ>_s0 0 3 --nested_arm C --nested_lambda <λ> --log_percol                           # ≤2 secant adjustments: calibrate_lambda.py --secant L0 BPW0 L1 BPW1 R1
python scripts/figs.py --A A_p1_s0 --B B_p1_s0 --C C_..._s0                                            # Phase-2 figures + Jaccard
# Phase 3: seeds 1,2 for A, B, C at the seed-0 p / λ (report bpw drift). Then the §4 decision rule.
```
`scripts/summarize.py` (called by run.sh) refreshes the auto table in `results/SUMMARY.md`; commit + push after every run
(`results/runs/<tag>/{result.json,rate.json,layers.jsonl,log.txt}`; the per-layer `.npz` dumps are git-ignored, keep them local
for figs/λ-calibration).

## 3. Gates to check before Phase 2
- `python - <<'EOF'` compare `results/runs/e3_s0` vs `results/runs/w3_s0`: `codes_E3 == 2*codes_W3`, identical `scale`/`zero`, identical wiki2 PPL.
- W4 Huffman bpw in ~3.3–3.6, W3 ≈ 3.0 (see SUMMARY note), E4 ≈ W4 rate.

## 4. Things already decided (see SUMMARY.md "Where the pieces live")
- `d_j = 1/U_in[j,j]²`, `r_i = [(U_subᵀU_sub)⁻¹]_ii` for q/k (TurboBoA's two-sided layers), 1 elsewhere; `N_ℓ` = solver loss functional on W / numel.
- Arm B uses a per-column accumulator `k_j = ⌊p·n·(j+1)⌋ − ⌊p·n·j⌋` over the `n_heads × rows` entries present in the gptq call (so the average is exactly p), and requires gain > 0.
- Arm A mask: per-tensor top-p by Fisher (`fisher_topk_mask` in `turboboa/quantize.py`).
