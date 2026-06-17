# ── Cell 9: Train 400M difficulty verifier (Stage 1, ~6h on T4) ──────────────
import subprocess, sys, traceback, os
from pathlib import Path

DATA_DIR    = _AT_DATA_DIR
OUTPUTS_DIR = _AT_OUTPUTS_DIR
verifier_out = OUTPUTS_DIR / "verifier-400m" / "best.pt"

if verifier_out.exists():
    print(f"✅ Verifier checkpoint already exists at {verifier_out}. Skipping training.")
else:
    print("=== Training 400M difficulty verifier ===")
    print(f"   Estimated time: ~6h on T4, ~2h on A100")
    print(f"   Wandb: https://wandb.ai/{CFG['WANDB_PROJECT']}\n")
    try:
        subprocess.run(
            [sys.executable, "src/adaptivethink/verifier/train.py",
             "--train",  str(DATA_DIR / "teacher_labels.jsonl"),
             "--eval",   str(DATA_DIR / "verifier_eval_labelled.jsonl"),
             "--out",    str(verifier_out),
             "--epochs", str(CFG["VERIFIER_EPOCHS"]),
             "--batch",  str(CFG["VERIFIER_BATCH"]),
             "--lr",     str(CFG["VERIFIER_LR"])],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        tb = traceback.format_exc()
        print(f"\n❌ Verifier training failed:\n{tb}")
        # last.pt is saved every epoch by train.py — resume is automatic on rerun
        print(f"   RESUME COMMAND (auto-resumes from last checkpoint):")
        print(f"   Re-run this cell — train.py will pick up from last.pt automatically.")
        try:
            import wandb
            if wandb.run:
                wandb.log({"error/verifier_train": str(e)})
                wandb.finish(exit_code=1)
        except Exception:
            pass
        raise

# ── Acceptance gate ───────────────────────────────────────────────────────────
print("\n=== Evaluating verifier on held-out set ===")
import torch
from adaptivethink.verifier.model import DifficultyVerifier, load_verifier
from transformers import AutoTokenizer
from scipy.stats import spearmanr
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tok = load_verifier(str(verifier_out), device)

eval_items = [json.loads(l) for l in (DATA_DIR / "verifier_eval_labelled.jsonl").open()]
questions = [it["question"] for it in eval_items]
gt_labels = [float(it["difficulty"]) for it in eval_items]

scores = model.score(questions, tok, device=device)
rho = spearmanr(scores, gt_labels).statistic
print(f"   Spearman ρ = {rho:.4f}  (target ≥ {CFG['VERIFIER_WARN_RHO']})")

if rho < CFG["VERIFIER_MIN_RHO"]:
    print(f"   ⚠️  ρ={rho:.3f} < {CFG['VERIFIER_MIN_RHO']} — continuing for pilot")
elif rho < CFG["VERIFIER_WARN_RHO"]:
    print(f"   ⚠️  ρ={rho:.3f} < {CFG['VERIFIER_WARN_RHO']} — verifier is marginal. "
          "GRPO will proceed but Stage-2 novelty (verifier gating) may be weak.")
else:
    print(f"   ✅ Verifier accepted (ρ={rho:.4f}).")

import builtins; builtins._AT_VERIFIER_CKPT = str(verifier_out)
