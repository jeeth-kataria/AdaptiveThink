# ── Cell 8: Build GRPO training data ─────────────────────────────────────────
import subprocess, sys, json
from pathlib import Path

DATA_DIR = _AT_DATA_DIR
grpo_path   = DATA_DIR / "gsm8k_train_labelled.jsonl"
labels_path = DATA_DIR / "teacher_labels.jsonl"

if grpo_path.exists():
    n = sum(1 for _ in grpo_path.open())
    print(f"✅ GRPO data already exists: {n} items. Skipping.")
else:
    cmd = [sys.executable, "scripts/02b_build_grpo_data.py",
           "--teacher-labels", str(labels_path),
           "--out", str(grpo_path)]
    if not labels_path.exists():
        cmd.append("--fallback-stub")
        print("⚠️  teacher_labels.jsonl absent — building stub (d=0.5). "
              "Run Cell 7 for real labels.")
    subprocess.run(cmd, check=True)

# ── Distribution sanity gate ─────────────────────────────────────────────────
items = [json.loads(l) for l in grpo_path.open()]
ds = [float(it.get("difficulty", 0.5)) for it in items]
mid_frac = sum(1 for d in ds if 0.4 < d < 0.6) / len(ds)

print(f"\n  Items: {len(items)}  |  mid-band (0.4–0.6): {mid_frac:.1%}")

if all(d == 0.5 for d in ds):
    print("  ⚠️  All difficulties are 0.5 (stub mode). "
          "The verifier gating term is DEAD until real labels are used.")
elif mid_frac > 0.70:
    print("  ⚠️  >70% of items near 0.5 — verifier gating will be weak.")
    print("  Recommendation: re-run teacher labelling with a more bimodal prompt.")
else:
    print("  ✅ Difficulty distribution looks informative.")

import builtins; builtins._AT_GRPO_DATA = str(grpo_path)
