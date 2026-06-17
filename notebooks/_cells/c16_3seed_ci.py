# ── Cell 16: 3-seed aggregation — mean ± 95% CI ───────────────────────────────
import subprocess, sys, json, csv
from pathlib import Path
import statistics, math

OUTPUTS_DIR  = _AT_OUTPUTS_DIR
BENCHMARKS   = ["gsm8k", "mmlu", "strategyqa"]
N            = CFG["EVAL_N_PER_BENCH"]
seed_dirs    = [_AT_SEED0_DIR, _AT_SEED1_DIR, _AT_SEED2_DIR]

def _eval_seed(seed, adapter_dir, tag_prefix):
    out = f"results/router_seed{seed}_full.json"
    if Path(out).exists():
        print(f"  ⏭️  seed {seed} full eval already exists.")
        return json.load(open(out))
    cmd = [sys.executable, "eval/run_benchmarks.py",
           "--benchmark", "all",
           "--adapter", str(adapter_dir),
           "--verifier-ckpt", str(OUTPUTS_DIR / "verifier-400m" / "best.pt"),
           "--route-mode", "model",
           "--tag", f"{tag_prefix}_seed{seed}",
           "--out", out,
           "--seed", str(seed)]
    if N: cmd += ["--n", str(N)]
    subprocess.run(cmd, check=True)
    return json.load(open(out))

print("=== Full eval across all 3 seeds ===")
seed_results = [_eval_seed(s, d, "router") for s, d in enumerate(seed_dirs)]

def _ci95(vals):
    n = len(vals)
    if n < 2: return 0.0
    se = statistics.stdev(vals) / math.sqrt(n)
    return 1.96 * se  # ~95% CI for n=3

# ── Build aggregation table ───────────────────────────────────────────────────
rows = []
print("\n=== 3-seed results (mean ± 95% CI) ===")
print(f"{'Benchmark':<14} {'Pass@1':>8} {'±CI':>7} {'think_rate':>11} {'avg_tok':>9}")
print("-" * 55)
for bench in BENCHMARKS:
    acc_vals  = [r["benchmarks"][bench]["pass@1"]    for r in seed_results if bench in r["benchmarks"]]
    tr_vals   = [r["benchmarks"][bench]["think_rate"] for r in seed_results if bench in r["benchmarks"]]
    tok_vals  = [r["benchmarks"][bench]["avg_tokens"] for r in seed_results if bench in r["benchmarks"]]
    if not acc_vals: continue
    mean_acc = statistics.mean(acc_vals)
    ci       = _ci95(acc_vals)
    mean_tr  = statistics.mean(tr_vals)
    mean_tok = statistics.mean(tok_vals)
    print(f"{bench:<14} {mean_acc:>8.3f} {ci:>7.3f} {mean_tr:>11.2f} {mean_tok:>9.0f}")
    rows.append({"benchmark": bench, "mean_pass1": round(mean_acc,4),
                 "ci95": round(ci,4), "mean_think_rate": round(mean_tr,4),
                 "mean_avg_tokens": round(mean_tok,1)})

# ── Save CSV ──────────────────────────────────────────────────────────────────
Path("results/ablations").mkdir(parents=True, exist_ok=True)
csv_path = "results/ablations/3seed_ci_table.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\n✅ Saved to {csv_path}")

# Store best checkpoint (highest mean GSM8K acc)
gsm_accs = [r["benchmarks"].get("gsm8k",{}).get("pass@1", 0) for r in seed_results]
best_seed = gsm_accs.index(max(gsm_accs))
import builtins; builtins._AT_BEST_SEED_DIR = seed_dirs[best_seed]
print(f"   Best seed: {best_seed} (GSM8K={max(gsm_accs):.3f}) → used for GGUF export.")
