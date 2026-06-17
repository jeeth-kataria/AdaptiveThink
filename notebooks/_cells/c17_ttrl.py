# ── Cell 17: TTRL add-on (optional — gated by CFG['RUN_TTRL']) ───────────────
import subprocess, sys, traceback, json
from pathlib import Path

if not CFG["RUN_TTRL"]:
    print("⏭️  RUN_TTRL=False — skipping TTRL. Set CFG['RUN_TTRL']=True in Cell 0 to enable.")
else:
    OUTPUTS_DIR  = _AT_OUTPUTS_DIR
    best_adapter = _AT_BEST_SEED_DIR
    ttrl_out     = OUTPUTS_DIR / "ttrl-seed0"

    if ttrl_out.exists() and any(ttrl_out.iterdir()):
        print(f"✅ TTRL output already exists at {ttrl_out}. Skipping.")
    else:
        print(f"=== TTRL add-on ({CFG['TTRL_STEPS']} steps on {CFG['TTRL_N_ITEMS']} MMLU items) ===")
        try:
            subprocess.run(
                [sys.executable, "src/adaptivethink/ttrl/ttrl.py",
                 "--adapter",    str(best_adapter),
                 "--output-dir", str(ttrl_out),
                 "--n",          str(CFG["TTRL_N_ITEMS"]),
                 "--steps",      str(CFG["TTRL_STEPS"]),
                 "--group-size", "8",
                 "--kl-beta",    "1e-3",
                 "--lambda-tok", "1e-3",
                 "--seed",       "0"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            tb = traceback.format_exc()
            print(f"\n❌ TTRL failed:\n{tb}")
            print(f"   RESUME: re-run this cell — TTRL will resume from {ttrl_out}/checkpoint-*/")
            print("   NOTE: TTRL is optional. If it fails, proceed to Cell 18 without it.")
            # Don't hard-abort — TTRL is labelled optional in plan.md
            print("   Continuing without TTRL result...")
        else:
            # Quick eval to see if TTRL helped
            out_json = "results/ttrl.json"
            subprocess.run(
                [sys.executable, "eval/run_benchmarks.py",
                 "--benchmark", "mmlu",
                 "--adapter",   str(ttrl_out),
                 "--verifier-ckpt", str(OUTPUTS_DIR / "verifier-400m" / "best.pt"),
                 "--route-mode", "model",
                 "--tag", "ours_full_ttrl",
                 "--out", out_json,
                 "--n", "200"],
                check=True,
            )
            ttrl_acc = json.load(open(out_json))["benchmarks"]["mmlu"]["pass@1"]
            base_acc = json.load(open("results/baseline_quick.json"))["benchmarks"].get("mmlu", {}).get("pass@1", "N/A")
            print(f"\n  TTRL MMLU pass@1: {ttrl_acc:.3f}  (baseline: {base_acc})")
            if isinstance(base_acc, float) and ttrl_acc > base_acc + 0.01:
                print("  ✅ TTRL helped — add ours_full_ttrl point to Pareto.")
            else:
                print("  ℹ️  TTRL did not improve MMLU — will report negative result in report.")

    import builtins; builtins._AT_TTRL_DIR = ttrl_out
