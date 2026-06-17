# ── Cell 4: Install pinned dependencies ──────────────────────────────────────
import subprocess, sys, os, importlib

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

def _version(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return None

CUDA_VER = os.environ.get("CUDA_VERSION", "12.1").split(".")[0] + "." + \
           os.environ.get("CUDA_VERSION", "12.1").split(".")[-1] if "." in \
           os.environ.get("CUDA_VERSION", "12.1") else "12.1"

print("=== Step 1: torch (must pin before everything else) ===")
_pip("torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121")
import torch
assert torch.__version__.startswith("2.4"), f"torch version mismatch: {torch.__version__}"
print(f"  ✅ torch {torch.__version__}")

print("=== Step 2: core requirements ===")
_pip("-r", "requirements.txt")

print("=== Step 3: flash-attn (ABI-sensitive) ===")
if torch.cuda.is_available():
    try:
        _pip("flash-attn==2.7.0.post2", "--no-build-isolation", "--no-cache-dir")
        import flash_attn
        print(f"  ✅ flash_attn {flash_attn.__version__}")
    except Exception as e:
        print(f"  ⚠️  flash-attn failed ({e}) — non-fatal, will use sdpa fallback")
else:
    print("  ⚠️  No GPU — skipping flash-attn")

print("=== Step 4: vLLM (only on ≥22 GB VRAM) ===")
vram = getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__,
               "_AT_VRAM_GB", 0) or \
       (torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0)
if vram >= 22:
    try:
        _pip("vllm==0.6.4.post1")
        import vllm
        print(f"  ✅ vllm {vllm.__version__}")
    except Exception as e:
        print(f"  ⚠️  vLLM failed ({e}) — train_grpo.py will fall back to HF generation")
else:
    print(f"  ⚠️  VRAM {vram:.1f} GB < 22 GB — skipping vLLM (T4 mode)")

print("=== Step 5: Unsloth ===")
try:
    # A100 (Ampere, CUDA 12.1) needs the cu121-ampere extras
    tag = "cu121-ampere" if vram >= 60 else "colab-new"
    _pip(f"unsloth[{tag}] @ git+https://github.com/unslothai/unsloth.git")
    import unsloth
    print(f"  ✅ unsloth {unsloth.__version__}")
except Exception as e:
    print(f"  ⚠️  Unsloth failed ({e}) — will use PEFT+BitsAndBytes fallback")

print("=== Step 6: install this package ===")
_pip("-e", ".", "--no-build-isolation")

print("=== Step 7: version sanity check ===")
_checks = {
    "transformers": "4.46.3",
    "trl":          "0.12.1",
    "peft":         "0.13.2",
    "accelerate":   "1.1.1",
}
_bad = []
for pkg, expected in _checks.items():
    got = _version(pkg)
    ok = got == expected
    print(f"  {'✅' if ok else '❌'} {pkg}: expected {expected}, got {got}")
    if not ok:
        _bad.append(pkg)
assert not _bad, f"ABORT — version mismatch for: {_bad}. Fix requirements.txt."

print("\n✅ All dependencies installed and verified.")

# Log pip freeze to wandb later (deferred until wandb is initialised in Cell 12)
import subprocess as _sp
_pip_freeze = _sp.run([sys.executable,"-m","pip","freeze"], capture_output=True, text=True).stdout
import builtins; builtins._AT_PIP_FREEZE = _pip_freeze
