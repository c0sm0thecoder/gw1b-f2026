#!/usr/bin/env bash
# Team 3, E8: 350M AdamW vs Muon on 8 A100s (bf16 for both), one run each at the learning rate predicted by the
# 50M-100M-200M trend (E1, E1b, E5). Spends A100 hours, which are capped (300 per team, production run included).
#
#   bash team3/experiments/e8_350m_a100.sh <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [total_tokens]     # 2 runs
#   ... add --submit to queue.
#
# total_tokens defaults to 7e9 (20 tokens/parameter): about 48 A100-hours for AdamW and 53 for Muon at 35% MFU
# (`gw1b budget --config configs/team3/e8_350m.yaml --tokens 7e9 --gpu a100 --n-gpus 8`). 3.5e9 (10 tokens/parameter)
# halves that. Precision differs from the V100 sizes (bf16 vs float32), so 350M is compared within itself (Muon vs
# AdamW at 350M) and enters the scale trend as a ratio, never as raw losses pooled with the V100 runs.
source "$(dirname "$0")/_common.sh"
[[ $# -eq 4 || $# -eq 5 ]] || { echo "usage: $0 <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [total_tokens] [--submit]" >&2; exit 2; }
TOKENS="${5:-7e9}"
COMMON="--gpus 8 --gpu-type a100 --time 10:00:00 --config configs/team3/e8_350m.yaml --set run.total_tokens=$TOKENS"

run() {  # run <optimizer> <lr> <wd>
  emit $COMMON --set run.name="team3-e8-$1-lr$2-wd$3-tok$TOKENS-s0" --set optim.name="$1" \
    --set optim.lr="$2" --set optim.weight_decay="$3" $(seed_args 0)
}

run adamw "$1" "$2"
run muon "$3" "$4"
finish
