#!/usr/bin/env bash
# Day 2: generate teacher labels for verifier distillation
set -e
python -m adaptivethink.data.loaders --dump-pool data/verifier_pool.jsonl 2>/dev/null || \
python -c "
import json, sys
sys.path.insert(0, 'src')
from adaptivethink.data.loaders import build_verifier_pool, build_verifier_eval
pool = build_verifier_pool()
eval_set = build_verifier_eval()
import pathlib; pathlib.Path('data').mkdir(exist_ok=True)
with open('data/verifier_pool.jsonl','w') as f:
    [f.write(json.dumps(r)+'\n') for r in pool]
with open('data/verifier_eval.jsonl','w') as f:
    [f.write(json.dumps(r)+'\n') for r in eval_set]
print(f'Pool: {len(pool)} | Eval: {len(eval_set)}')
"

python src/adaptivethink/data/teacher_labels.py \
  --pool data/verifier_pool.jsonl \
  --out data/teacher_labels.jsonl \
  --db data/teacher_cache.sqlite \
  --provider "${TEACHER_PROVIDER:-deepinfra}" \
  --max-cost-usd 50
