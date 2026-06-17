# ── Cell 6: Download + build verifier pool ───────────────────────────────────
from pathlib import Path
import json

DATA_DIR = _AT_DATA_DIR  # set by Cell 2

pool_path = DATA_DIR / "verifier_pool.jsonl"
eval_path = DATA_DIR / "verifier_eval_labelled.jsonl"

if pool_path.exists() and eval_path.exists():
    n_pool = sum(1 for _ in pool_path.open())
    n_eval = sum(1 for _ in eval_path.open())
    print(f"✅ Pool already exists: {n_pool} items. Eval: {n_eval} items. Skipping download.")
else:
    print("Building verifier pool (12k items) — requires dataset downloads (~2 min)...")
    from adaptivethink.data.loaders import build_verifier_pool, build_verifier_eval

    pool = build_verifier_pool(seed=0)
    eval_set = build_verifier_eval(seed=42)

    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("w") as f:
        for it in pool: f.write(json.dumps(it) + "\n")
    with eval_path.open("w") as f:
        for it in eval_set: f.write(json.dumps(it) + "\n")

    print(f"✅ Pool: {len(pool)} items -> {pool_path}")
    print(f"✅ Eval: {len(eval_set)} items -> {eval_path}")

# Verify no overlap between pool questions and eval questions
pool_qs = {json.loads(l)["question"] for l in pool_path.open()}
eval_qs = {json.loads(l)["question"] for l in eval_path.open()}
overlap = pool_qs & eval_qs
if overlap:
    print(f"  ⚠️  {len(overlap)} overlap questions found — check loader splits")
else:
    print("  ✅ No overlap between pool and eval set.")
