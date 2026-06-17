# ── Cell 13: Seed-0 Pareto check + Case A/B/C/D decision ─────────────────────
import subprocess, sys, json
from pathlib import Path

OUTPUTS_DIR = _AT_OUTPUTS_DIR
seed0_dir   = _AT_SEED0_DIR
N           = CFG["EVAL_N_PER_BENCH"]

Path("results").mkdir(exist_ok=True)

def _quick_eval(tag, adapter=None, route_mode="always_think"):
    out = f"results/{tag}.json"
    if Path(out).exists():
        print(f"  ⏭️  {tag} already evaluated.")
        return json.load(open(out))
    cmd = [sys.executable, "eval/run_benchmarks.py",
           "--benchmark", "gsm8k",
           "--route-mode", route_mode,
           "--tag", tag,
           "--out", out,
           "--n", str(N or 200)]
    if adapter:
        cmd += ["--adapter", adapter,
                "--verifier-ckpt", str(OUTPUTS_DIR / "verifier-400m" / "best.pt")]
    subprocess.run(cmd, check=True)
    return json.load(open(out))

print("=== Quick eval: baseline vs seed-0 router (GSM8K only) ===")
baseline = _quick_eval("baseline_quick", route_mode="always_think")
router   = _quick_eval("router_seed0_quick", adapter=str(seed0_dir),
                        route_mode="model")

b_acc  = baseline["benchmarks"]["gsm8k"]["pass@1"]
r_acc  = router["benchmarks"]["gsm8k"]["pass@1"]
b_tok  = baseline["benchmarks"]["gsm8k"]["avg_tokens"]
r_tok  = router["benchmarks"]["gsm8k"]["avg_tokens"]
tok_red = (b_tok - r_tok) / b_tok

print(f"\n  Baseline  — acc={b_acc:.3f}  avg_tok={b_tok:.0f}")
print(f"  Router    — acc={r_acc:.3f}  avg_tok={r_tok:.0f}")
print(f"  Token reduction: {tok_red:.1%}  |  Accuracy delta: {r_acc-b_acc:+.3f}")

# ── Case decision ─────────────────────────────────────────────────────────────
if r_acc >= b_acc - 0.01 and tok_red >= 0.20:
    case = "A"
    print("\n🟢 CASE A — Router dominates (≥20% token reduction at ≤1% accuracy loss).")
    print("   Action: lock hyperparams, proceed to seeds 1+2 (Cells 14-15).")
elif tok_red >= 0.10:
    case = "B"
    print("\n🟡 CASE B — Close but not dominating. Try λ_tok sweep.")
    print("   Action: optionally change CFG['GRPO_LAMBDA_TOK'] and rerun this seed.")
    print("   Then proceed to seeds 1+2.")
elif r_acc < b_acc - 0.05:
    case = "C"
    print("\n🔴 CASE C — Router worse than baseline. Likely causes:")
    print("   (1) Verifier d not informative — check ρ from Cell 9.")
    print("   (2) Reward parsing bug — run: python -c \"from adaptivethink.router.reward import compute_rewards; help(compute_rewards)\"")
    print("   (3) think_rate collapsed — check wandb.")
    print("   Do NOT proceed to seeds 1+2 until root cause is fixed.")
else:
    case = "D"
    print("\n🔴 CASE D — Total failure. Falling back to AdaptThink-style internal routing.")
    print("   See plan.md §6 Case D. Document honestly in the report.")

import builtins; builtins._AT_PARETO_CASE = case
print(f"\n📌 Read the case above, then confirm to continue → run Cell 14.")
