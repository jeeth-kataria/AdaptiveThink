# ── Cell 4: Install pinned dependencies ──────────────────────────────────────
import subprocess, sys, os, importlib
from pathlib import Path

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

def _version(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return None

print("=== Step 1: torch ===")
try:
    import torch
    if torch.__version__.startswith("2.5") or torch.__version__.startswith("2.4"):
        print(f"  ✅ torch {torch.__version__} already installed")
    else:
        _pip("torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121")
        import torch
        print(f"  ✅ torch {torch.__version__}")
except ImportError:
    _pip("torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121")
    import torch
    print(f"  ✅ torch {torch.__version__}")

print("\n=== Step 2: core ML libraries ===")
_pip("transformers==4.45.2", "accelerate==0.34.2", "peft==0.12.0", "bitsandbytes==0.43.3", "trl==0.11.4")

print("\n=== Step 3: data & utilities ===")
_pip("datasets==3.1.0", "scipy==1.14.1", "numpy==1.26.4", "sentencepiece==0.2.0")
_pip("wandb==0.18.7", "huggingface-hub==0.26.5", "openai==1.54.0", "python-dotenv==1.0.1", "pytest==8.3.4")

print("\n=== Step 4: flash-attn (optional) ===")
if torch.cuda.is_available():
    try:
        _pip("flash-attn==2.7.0.post2", "--no-build-isolation", "--no-cache-dir")
        print("  ✅ flash-attn")
    except Exception:
        print("  ⚠️  flash-attn failed — using sdpa fallback")
else:
    print("  ⏭️  No GPU — skipping")

print("\n=== Step 5: unsloth (optional) ===")
vram = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
try:
    tag = "cu121-ampere" if vram >= 60 else "colab-new"
    _pip(f"unsloth[{tag}] @ git+https://github.com/unslothai/unsloth.git")
    print("  ✅ unsloth")
except Exception:
    print("  ⚠️  unsloth failed — using PEFT fallback")

print("\n=== Step 6: install package ===")
_pip("-e", ".", "--no-build-isolation")

print("\n=== Step 7: verify ===")
for pkg in ["torch", "transformers", "trl", "peft", "bitsandbytes"]:
    v = _version(pkg)
    print(f"  ✅ {pkg:15s} {v}")

print("\n✅ All dependencies installed.")

import subprocess as _sp
_pip_freeze = _sp.run([sys.executable,"-m","pip","freeze"], capture_output=True, text=True).stdout
import builtins; builtins._AT_PIP_FREEZE = _pip_freeze
