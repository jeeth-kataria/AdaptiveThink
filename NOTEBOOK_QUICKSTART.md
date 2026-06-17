# AdaptiveThink Notebook Quick Start

## ✅ Fixed Dependency Issues

The notebook is now **completely self-contained** and ready to run on Ubuntu GPU machines (CUDA 12.x).

### What was fixed:
- ✅ `torch==2.5.1` (stable with CUDA 12.x)
- ✅ `bitsandbytes==0.43.3` (0.44.1 doesn't exist)
- ✅ Removed vLLM (conflicts resolved)
- ✅ Self-contained Cell 4 (no external setup.sh)
- ✅ Graceful fallbacks for optional dependencies

## 🚀 How to Run

### On your GPU machine (Ubuntu + CUDA 12.x):

1. **Open Jupyter**
   ```bash
   cd /path/to/ENNOVATE
   jupyter notebook notebooks/AdaptiveThink_Train.ipynb
   ```

2. **Edit Cell 0** - Add your API keys:
   - `GROQ_API_KEY_1` (already filled)
   - `WANDB_API_KEY` (already filled)
   - `HF_TOKEN` (already filled)

3. **Run All Cells** - Click "Cell" → "Run All"

That's it! Cell 4 will automatically install all dependencies.

## 📊 What to expect:

- **Cell 1**: Environment check (GPU, CUDA, VRAM)
- **Cell 4**: Dependencies install (~5-10 min)
  - torch, transformers, peft, trl, etc.
  - flash-attn (optional, may fail - non-fatal)
  - unsloth (optional, may fail - non-fatal)
- **Cell 5**: Unit tests (must pass)
- **Cells 6-8**: Data preparation
- **Cell 9**: Verifier training (~6h on your GPU)
- **Cells 12-15**: GRPO training (~36h per seed)

## ⚠️ Known warnings (safe to ignore):

- `flash-attn failed` → Will use slower SDPA attention (works fine)
- `unsloth failed` → Will use PEFT+BitsAndBytes fallback (works fine)
- `Version mismatch` warnings → As long as major.minor match, it's OK

## 🆘 If Cell 4 fails:

Check the error message:
- **Out of memory**: Not a dependency issue - reduce batch size in Cell 0
- **CUDA not found**: `nvidia-smi` to verify GPU is visible
- **bitsandbytes error**: Make sure CUDA 12.x is installed

## 📝 No other setup needed:

- ❌ Don't run `quick_start.sh`
- ❌ Don't run `scripts/01_setup.sh`
- ✅ Just run the notebook!
