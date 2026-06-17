#!/usr/bin/env bash
# Day 4: full GRPO router training
set -e
SEED=${1:-0}
mkdir -p outputs/router-seed${SEED} logs

# ── Mandatory smoke-test gate ─────────────────────────────────────────────────
# 50-step sanity run. Must pass before launching the 36-hour full run.
# Skip with SKIP_SMOKE=1 bash scripts/04_train_grpo_router.sh if already verified.
if [ "${SKIP_SMOKE}" != "1" ]; then
  echo "=== Running smoke test (50 steps). Set SKIP_SMOKE=1 to bypass. ==="
  bash scripts/00_smoke_grpo.sh
  echo "=== Smoke test PASSED. Launching full run. ==="
fi

# ── Ensure GRPO data file exists ──────────────────────────────────────────────
if [ ! -f "data/gsm8k_train_labelled.jsonl" ]; then
  echo "=== data/gsm8k_train_labelled.jsonl not found — building from teacher_labels ==="
  python scripts/02b_build_grpo_data.py
fi

python src/adaptivethink/router/train_grpo.py \
  --data data/gsm8k_train_labelled.jsonl \
  --verifier-ckpt outputs/verifier-400m/best.pt \
  --output-dir outputs/router-seed${SEED} \
  --steps 1500 --batch 1 --group-size 8 \
  --lr 1e-5 --kl-beta 5e-3 --max-seq-len 2048 \
  --seed ${SEED} 2>&1 | tee logs/grpo_seed${SEED}.log
