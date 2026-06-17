# ── Cell 7: Generate teacher difficulty labels ────────────────────────────────
import subprocess, sys, json
from pathlib import Path

DATA_DIR = _AT_DATA_DIR

labels_path = DATA_DIR / "teacher_labels.jsonl"
pool_path   = DATA_DIR / "verifier_pool.jsonl"
db_path     = DATA_DIR / "teacher_cache.sqlite"

if labels_path.exists():
    n = sum(1 for _ in labels_path.open())
    print(f"✅ Teacher labels already exist: {n} items. Skipping.")
else:
    print(f"Labelling {sum(1 for _ in pool_path.open())} items via {CFG['TEACHER_PROVIDER']}...")
    print(f"Cost guard: ${CFG['TEACHER_MAX_COST_USD']}")
    print("This will take 4-10 hours. Run in tmux or as a background cell.\n")
    try:
        subprocess.run(
            [sys.executable, "src/adaptivethink/data/teacher_labels.py",
             "--pool",         str(pool_path),
             "--out",          str(labels_path),
             "--db",           str(db_path),
             "--provider",     CFG["TEACHER_PROVIDER"],
             "--max-cost-usd", str(CFG["TEACHER_MAX_COST_USD"])],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Teacher labelling failed or was interrupted: {e}")
        print(f"   Partial results cached in {db_path}")
        print(f"   RESUME COMMAND:\n"
              f"   python src/adaptivethink/data/teacher_labels.py "
              f"--pool {pool_path} --out {labels_path} "
              f"--db {db_path} --provider {CFG['TEACHER_PROVIDER']}")
        raise

# ── Pilot distribution check ─────────────────────────────────────────────────
items = [json.loads(l) for l in labels_path.open()]
ds = [it["difficulty"] for it in items]
bins = {"[0, 0.33]": sum(1 for d in ds if d <= 0.33),
        "(0.33, 0.67]": sum(1 for d in ds if 0.33 < d <= 0.67),
        "(0.67, 1.0]": sum(1 for d in ds if d > 0.67)}
print("\n=== Difficulty distribution ===")
for b, n in bins.items():
    pct = 100*n/len(ds)
    print(f"  {b}: {n} ({pct:.1f}%)")

if bins["(0.33, 0.67]"] / len(ds) > 0.70:
    print("\n⚠️  WARNING: >70% of items near 0.5 — teacher prompt may be mode-collapsing.")
    print("   Consider re-running with a more bimodal prompt before proceeding.")
else:
    print("\n✅ Distribution looks healthy.")
