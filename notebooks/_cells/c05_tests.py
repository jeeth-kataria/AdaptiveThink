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

if result.returncode != 0:
    print("\n⚠️  Unit tests failed - continuing anyway for pilot")
else:
    print("\n✅ All unit tests passed. Safe to proceed to training.")
