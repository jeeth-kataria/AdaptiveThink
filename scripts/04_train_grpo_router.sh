#!/usr/bin/env bash
# Day 4: full GRPO router training
set -e
SEED=${1:-0}
mkdir -p outputs/router-seed${SEED} logs
python src/adaptivethink/router/train_grpo.py \
  --data data/gsm8k_train_labelled.jsonl \
  --verifier-ckpt outputs/verifier-400m/best.pt \
  --output-dir outputs/router-seed${SEED} \
  --steps 1500 --batch 1 --group-size 8 \
  --lr 1e-5 --kl-beta 5e-3 --max-seq-len 2048 \
  --seed ${SEED} 2>&1 | tee logs/grpo_seed${SEED}.log
