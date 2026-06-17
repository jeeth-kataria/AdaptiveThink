# ── Cell 5: Unit test gate — MUST PASS before any training ───────────────────
import subprocess, sys

print("=== Running unit tests ===")
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_reward.py",
     "tests/test_eval.py",
     "tests/test_verifier.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=False,   # stream output live
)

assert result.returncode == 0, (
    "ABORT — unit tests failed. Fix all failures before proceeding. "
    "A bug in the reward function found now saves 36 wasted GPU hours."
)
print("\n✅ All unit tests passed. Safe to proceed to training.")
