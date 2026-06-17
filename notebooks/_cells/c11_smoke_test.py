# ── Cell 11: Smoke test gate ──────────────────────────────────────────────────
import subprocess, sys, os

if os.environ.get("SKIP_SMOKE") == "1":
    print("⏭️  SKIP_SMOKE=1 — bypassing smoke test.")
else:
    print("=== 50-step GRPO smoke test ===")
    result = subprocess.run(["bash", "scripts/00_smoke_grpo.sh"],
                            capture_output=True, text=True)
    out = result.stdout + result.stderr
    print(out[-3000:] if len(out) > 3000 else out)

    if result.returncode != 0:
        if "out of memory" in out.lower():
            raise RuntimeError(
                "OOM in smoke test.\n"
                "Fix: set CFG['GRPO_GROUP_SIZE']=4 and CFG['GRPO_MAX_SEQ_LEN']=1024 in Cell 0."
            )
        if "TypeError" in out or "unexpected keyword" in out:
            raise RuntimeError(
                "GRPOConfig API mismatch — check trl.__version__ == '0.12.1'."
            )
        raise RuntimeError("Smoke test failed. See output above. Fix before the 36h run.")

    assert "SMOKE TEST PASSED" in out, "Smoke test exited 0 but success string missing."
    print("\n✅ Smoke test passed.")
