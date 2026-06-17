# ── Cell 24b: Git backup after training ──────────────────────────────────────
import subprocess
import os
from pathlib import Path

print("=== Git Backup After Training ===")

# Only backup results, not the massive checkpoint files
backup_paths = [
    "results/",
    "logs/",
    "CLAUDE.md",
    "execution.md",
]

try:
    # Check if we have changes
    status = subprocess.run(["git", "status", "--porcelain"], 
                          capture_output=True, text=True, check=True)
    
    if not status.stdout.strip():
        print("✅ No changes to commit - already backed up")
    else:
        # Add only the safe files
        for path in backup_paths:
            if Path(path).exists():
                subprocess.run(["git", "add", path], check=False)
        
        # Get training summary for commit message
        try:
            import json
            results = json.load(open("results/router_seed0_full.json"))
            gsm_acc = results["benchmarks"]["gsm8k"]["pass@1"]
            commit_msg = f"results: Training complete - GSM8K {gsm_acc:.3f} @ {CFG['GRPO_STEPS']} steps"
        except:
            commit_msg = f"results: Training complete @ {CFG['GRPO_STEPS']} steps"
        
        # Commit
        result = subprocess.run(["git", "commit", "-m", commit_msg],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Committed: {commit_msg}")
            
            # Push to GitHub
            print("\n🔄 Pushing to GitHub...")
            push_result = subprocess.run(["git", "push", "origin", "main"],
                                        capture_output=True, text=True)
            
            if push_result.returncode == 0:
                print("✅ Results backed up to GitHub!")
                print("   View at: https://github.com/jeeth-kataria/AdaptiveThink")
            else:
                print(f"⚠️  Push failed: {push_result.stderr}")
                print("   Run manually: git push origin main")
        else:
            print("ℹ️  No new changes to commit")

except subprocess.CalledProcessError as e:
    print(f"⚠️  Git backup failed: {e}")
    print("   Your results are still safe locally in results/ and on HF Hub")
    print("   Manual backup: git add results/ && git commit -m 'results' && git push")

print("\n✅ Training artifacts saved:")
print(f"   - Local: results/, logs/")
print(f"   - HuggingFace Hub: https://huggingface.co/{CFG['HF_ORG']}")
print(f"   - WandB: https://wandb.ai/jeeth-kataria/{CFG['WANDB_PROJECT']}")
