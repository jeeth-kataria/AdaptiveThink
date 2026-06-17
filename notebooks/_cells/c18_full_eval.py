# ── Cell 18: Full benchmark eval — all baselines ─────────────────────────────
import subprocess, sys
from pathlib import Path

OUTPUTS_DIR  = _AT_OUTPUTS_DIR
best_adapter = str(_AT_BEST_SEED_DIR)
verifier_ckpt = str(OUTPUTS_DIR / "verifier-400m" / "best.pt")
N = CFG["EVAL_N_PER_BENCH"]

BASELINES = [
    # (tag, extra_args)
    ("baseline",        ["--route-mode", "always_think"]),
    ("no_cot",          ["--route-mode", "never_think",
                         "--adapter", best_adapter, "--verifier-ckpt", verifier_ckpt]),
    ("ours_full",       ["--adapter", best_adapter, "--verifier-ckpt", verifier_ckpt,
                         "--route-mode", "model"]),
    ("ours_router_only",["--adapter", best_adapter, "--route-mode", "threshold"]),
]
# Add TTRL point if it ran
if CFG["RUN_TTRL"] and Path("results/ttrl.json").exists():
    BASELINES.append(("ours_full_ttrl",
                      ["--adapter", str(_AT_TTRL_DIR),
                       "--verifier-ckpt", verifier_ckpt, "--route-mode", "model"]))

for tag, extra in BASELINES:
    out = f"results/{tag}.json"
    if Path(out).exists():
        print(f"  ⏭️  {tag} already evaluated.")
        continue
    print(f"  Evaluating {tag} ...")
    cmd = [sys.executable, "eval/run_benchmarks.py",
           "--benchmark", "all",
           "--tag", tag,
           "--out", out,
           "--seed", "0"]
    if N: cmd += ["--n", str(N)]
    cmd += extra
    subprocess.run(cmd, check=True)
    print(f"  ✅ {tag} done.")

print("\n✅ All baselines evaluated.")
