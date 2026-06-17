# ── Cell 12: GRPO seed 0 (1500 steps, ~36h) ──────────────────────────────────
import subprocess, sys, os, traceback, json
from pathlib import Path

OUTPUTS_DIR = _AT_OUTPUTS_DIR

def _run_grpo(seed: int):
    """Launch one GRPO seed. Handles resume, exception logging, wandb."""
    out_dir = OUTPUTS_DIR / f"router-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / f"grpo_seed{seed}.log"

    # Log git SHA + pip freeze to a file for this run
    import subprocess as _sp
    git_sha = _sp.run(["git","rev-parse","HEAD"], capture_output=True, text=True).stdout.strip()
    (out_dir / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha,
        "seed": seed,
        "lambda_tok": CFG["GRPO_LAMBDA_TOK"],
        "kl_beta": CFG["GRPO_KL_BETA"],
    }))

    cmd = [
        sys.executable, "src/adaptivethink/router/train_grpo.py",
        "--data",          str(_AT_DATA_DIR / "gsm8k_train_labelled.jsonl"),
        "--verifier-ckpt", str(_AT_OUTPUTS_DIR / "verifier-400m" / "best.pt"),
        "--output-dir",    str(out_dir),
        "--steps",         str(CFG["GRPO_STEPS"] or 0),
        "--group-size",    str(CFG["GRPO_GROUP_SIZE"] or 0),
        "--max-seq-len",   str(CFG["GRPO_MAX_SEQ_LEN"] or 0),
        "--lr",            str(CFG["GRPO_LR"]),
        "--kl-beta",       str(CFG["GRPO_KL_BETA"]),
        "--lambda-tok",    str(CFG["GRPO_LAMBDA_TOK"]),
        "--lambda-obey",   str(CFG["GRPO_LAMBDA_OBEY"]),
        "--seed",          str(seed),
    ]

    print(f"\n=== GRPO seed={seed} | output: {out_dir} ===")
    print(f"    Wandb: https://wandb.ai/{CFG['WANDB_PROJECT']}/runs/grpo_seed{seed}")
    print(f"    Log:   {log_path}\n")

    try:
        with log_path.open("w") as log_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
            proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception as e:
        tb = traceback.format_exc()
        # Checkpoint is saved every 50 steps by TRL — at most 50 steps lost
        print(f"\n❌ GRPO seed={seed} failed:\n{tb}")
        print(f"\n   RESUME COMMAND:")
        print(f"   SKIP_SMOKE=1 bash scripts/04_train_grpo_router.sh {seed}")
        print(f"   (train_grpo.py will auto-resume from {out_dir}/checkpoint-*/)")
        try:
            import wandb
            if wandb.run: wandb.log({"error/grpo": str(e)}); wandb.finish(exit_code=1)
        except Exception:
            pass
        raise

    print(f"\n✅ GRPO seed={seed} complete.")
    return out_dir

# Make helper available to cells 13-15
import builtins; builtins._run_grpo = _run_grpo

# ── Run seed 0 ────────────────────────────────────────────────────────────────
_seed0_dir = _run_grpo(0)
import builtins; builtins._AT_SEED0_DIR = _seed0_dir
