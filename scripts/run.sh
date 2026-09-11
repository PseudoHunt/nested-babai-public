#!/usr/bin/env bash
# usage: scripts/run.sh TAG SEED WBITS [extra turboboa args...]
#   TAG   : run name -> results/runs/TAG
#   SEED  : calibration seed (TurboBoA sampler)
#   WBITS : 3 or 4 (native); nested arms use 3 + --nested_arm ...
# TurboBoA default Llama config (README defaults + g=128): block_v, n_quant_rows 16, consider_dX (alpha .25),
# adaptive_qparam, refine_qparam (no-op for grouped), qparam_comput Hessian, damping 1%, act-order off (forced off for g!=-1).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
TAG=$1; SEED=$2; WBITS=$3; shift 3
GROUP=${GROUP:-128}                                              # -1 = per-channel (one scale per row); add --per_tensor for one scale per tensor
QPARAM_FLAGS=${QPARAM_FLAGS:---adaptive_qparam --refine_qparam}  # per-tensor runs: QPARAM_FLAGS=--adaptive_qparam (refine_qparam re-fits per-row scales)
OUT=$NB_ROOT/results/runs/$TAG
mkdir -p "$OUT"
cd $NB_ROOT/turboboa
echo "[run.sh] $TAG seed=$SEED w=$WBITS group=$GROUP qparam='$QPARAM_FLAGS' extra: $*" | tee "$OUT/log.txt"
python -u main.py --llm_path "$LLAMA_PATH" --w_bits "$WBITS" --group_size "$GROUP" --block_v --n_quant_rows 16 \
  --consider_dX $QPARAM_FLAGS --qparam_comput Hessian --seed "$SEED" \
  --dump_dir "$OUT" "$@" 2>&1 | grep --line-buffered -v -E "^(Loss:|-{20,}|NOT supported)" | tee -a "$OUT/log.txt"
python $NB_ROOT/scripts/rate.py "$OUT" | tee -a "$OUT/log.txt"
python $NB_ROOT/scripts/summarize.py
