#!/usr/bin/env bash
# scripts/01_setup.sh — works on Colab T4, Vast.ai RTX 4090, local Linux
set -e

# Create required directories up front so no script ever fails with ENOENT
mkdir -p data logs results/pareto results/ablations results/figures results/onDevice outputs

# Detect environment
CUDA_VER=$(python -c "import torch; print(torch.version.cuda or '0')" 2>/dev/null || echo "0")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "CPU")
echo "=== Environment: CUDA ${CUDA_VER} | GPU: ${GPU_NAME} ==="

# Base packages (fast, no build)
pip install -q \
  transformers==4.46.3 \
  accelerate==1.1.1 \
  peft==0.13.2 \
  bitsandbytes==0.44.1 \
  trl==0.12.1 \
  datasets==3.1.0 \
  wandb==0.18.7 \
  huggingface-hub==0.26.5 \
  sentencepiece==0.2.0 \
  openai==1.54.0 \
  python-dotenv==1.0.1 \
  scipy==1.14.1 \
  numpy==1.26.4

# flash-attn — skip on CPU or if CUDA < 11.8
if python -c "import torch; assert torch.cuda.is_available() and float(torch.version.cuda) >= 11.8" 2>/dev/null; then
  echo "=== Installing flash-attn ==="
  pip install -q flash-attn --no-build-isolation || echo "WARNING: flash-attn failed (non-fatal)"
else
  echo "=== Skipping flash-attn (no compatible CUDA) ==="
fi

# vLLM — only on 4090/A100 (>=22 GB); skip on T4
VRAM_GB=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
if python -c "assert float('${VRAM_GB}') >= 22" 2>/dev/null; then
  echo "=== Installing vLLM (${VRAM_GB} GB VRAM detected) ==="
  pip install -q vllm==0.6.4.post1 || echo "WARNING: vllm failed"
else
  echo "=== Skipping vLLM (${VRAM_GB} GB VRAM < 22 GB — T4 mode) ==="
fi

# Unsloth
echo "=== Installing Unsloth ==="
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || \
  pip install -q unsloth || echo "WARNING: Unsloth failed — will use PEFT fallback"

# Install this package in editable mode
pip install -q -e . 2>/dev/null || true

# Sanity check
echo "=== Sanity check ==="
python -c "
import torch, transformers, peft, trl, datasets
print(f'torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}')
try: import unsloth; print('unsloth: ok')
except: print('unsloth: NOT installed (PEFT fallback will be used)')
try: import vllm; print('vllm: ok')
except: print('vllm: NOT installed (HF generation fallback will be used)')
try: import flash_attn; print(f'flash_attn {flash_attn.__version__}')
except: print('flash_attn: NOT installed (non-fatal)')
"

pip freeze > requirements.lock.txt
echo "=== Setup complete. requirements.lock.txt written. ==="

# ── HF Hub pre-creation check ─────────────────────────────────────────────────
# Repos must exist BEFORE training starts; push_to_hub errors at step 50
# will kill a 36-hour run. Run this once and confirm.
if [ -n "${HF_TOKEN}" ]; then
  echo "=== Checking HF Hub repos ==="
  python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repos = [
    "statezero/verifier-400m",
    "statezero/router-1p5b-seed0",
    "statezero/router-1p5b-seed1",
    "statezero/router-1p5b-seed2",
    "statezero/difficulty-labels",
]
for repo in repos:
    try:
        api.repo_info(repo_id=repo)
        print(f"  ✓ {repo} exists")
    except Exception:
        print(f"  + creating {repo}")
        repo_type = "dataset" if "labels" in repo else "model"
        api.create_repo(repo_id=repo, repo_type=repo_type, private=True, exist_ok=True)
        print(f"  ✓ {repo} created")
PYEOF
else
  echo "=== HF_TOKEN not set — skipping HF repo check (set it in .env before training) ==="
fi
