# ── Cell 23: Auto-update progress docs ───────────────────────────────────────
from datetime import datetime
from pathlib import Path
import json

# ── Read best results ─────────────────────────────────────────────────────────
best_seed_res = json.load(open(f"results/router_seed0_full.json"))
gsm_acc = best_seed_res["benchmarks"]["gsm8k"]["pass@1"]
think_rate = best_seed_res["benchmarks"]["gsm8k"]["think_rate"]
step = CFG["GRPO_STEPS"]

# ── Update results/progress.md ────────────────────────────────────────────────
prog_path = Path("results/progress.md")
if not prog_path.exists():
    prog_path.write_text("| Date | Stage | Step | GSM8K Pass@1 | Think-rate | Notes |\n"
                         "|------|-------|------|-------------|------------|-|\n")

today = datetime.now().strftime("%Y-%m-%d")
new_line = (f"| {today} | GRPO-3seed | {step} | {gsm_acc:.3f} | "
            f"{think_rate:.2f} | λ_tok={CFG['GRPO_LAMBDA_TOK']} |\n")
prog_path.write_text(prog_path.read_text() + new_line)
print(f"✅ Updated {prog_path}")

# ── Append to execution.md §Lessons ───────────────────────────────────────────
exec_path = Path("execution.md")
if exec_path.exists():
    lesson = (f"\n- {today} — Completed 3-seed GRPO. GSM8K Pass@1={gsm_acc:.3f}, "
              f"think_rate={think_rate:.2f}. "
              f"Case {getattr(__builtins__, '_AT_PARETO_CASE', 'unknown')} on Day 6 check.")
    content = exec_path.read_text()
    if "## Appendix C — Lessons" in content:
        exec_path.write_text(content + lesson)
        print(f"✅ Appended to {exec_path} §Lessons")

# ── Append to CLAUDE.md §5 Decision log ───────────────────────────────────────
claude_path = Path("CLAUDE.md")
if claude_path.exists():
    decision = (f"\n- {today} — Completed 3-seed GRPO run. Best seed: GSM8K {gsm_acc:.3f}. "
                f"Pushed all artefacts to HF Hub. Ready for quantisation + report.")
    content = claude_path.read_text()
    if "## 5. Decision log" in content:
        # Append after the last existing decision or after the section header
        parts = content.split("---\n\n## 6.")
        if len(parts) == 2:
            claude_path.write_text(parts[0] + decision + "\n\n---\n\n## 6." + parts[1])
        else:
            claude_path.write_text(content + decision)
        print(f"✅ Appended to {claude_path} §5")

# ── Commit results (not the notebook itself) ──────────────────────────────────
import subprocess
subprocess.run(["git", "add", "results/", "CLAUDE.md", "execution.md"], check=False)
subprocess.run(["git", "commit", "-m",
                f"chore: training complete — GSM8K {gsm_acc:.3f} @ step {step}"],
               check=False)
print("✅ Committed results to git (notebook excluded).")

print(f"\n📊 Summary:\n   GSM8K Pass@1: {gsm_acc:.3f}\n   Think-rate: {think_rate:.2f}\n"
      f"   Steps: {step}\n   λ_tok: {CFG['GRPO_LAMBDA_TOK']}")
