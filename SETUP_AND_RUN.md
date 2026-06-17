# 🚀 AdaptiveThink - Complete Setup & Run Guide

## Prerequisites
- Ubuntu machine with GPU
- Conda/Miniforge installed
- Git installed

---

## Part 1: Environment Setup (5 minutes)

### Step 1: Clone & Create Environment
```bash
# Navigate to your workspace
cd /media/ccps/EXT3

# Clone (if not already cloned)
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink

# Pull latest fixes
git pull origin main

# Create conda environment with Python 3.11
conda create -n adaptivethink python=3.11 -y

# Activate environment
conda activate adaptivethink
```

### Step 2: Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Setup
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.5.1+cu121
CUDA: True
GPU: NVIDIA RTX A6000
```

---

## Part 2: Get API Keys (5 minutes)

### 1. Groq Keys (FREE - need 2 for fallback)
- Go to: https://console.groq.com/keys
- Click "Create API Key" 
- Copy first key (starts with `gsk_...`)
- Click "Create API Key" again
- Copy second key
- Save both somewhere

### 2. WandB Key
- Go to: https://wandb.ai/authorize
- Copy the key (starts with `wandb_v1_...` or similar)

### 3. HuggingFace Token
- Go to: https://huggingface.co/settings/tokens
- Click "New token"
- Type: "Write"
- Copy the token (starts with `hf_...`)

---

## Part 3: Run Training

### Step 1: Open Notebook in VS Code
```bash
# Make sure you're in the AdaptiveThink directory
cd /media/ccps/EXT3/AdaptiveThink

# Open in VS Code
code notebooks/AdaptiveThink_Train.ipynb
```

### Step 2: Select Kernel
- Click **"Select Kernel"** (top-right corner)
- Choose **"Python Environments"**
- Select **"adaptivethink"**

### Step 3: Edit Cell 0 (Paste Your Keys)
```python
CFG = {
    "GROQ_API_KEY_1":    "gsk_YOUR_FIRST_KEY_HERE",    # ⬅️ PASTE
    "GROQ_API_KEY_2":    "gsk_YOUR_SECOND_KEY_HERE",   # ⬅️ PASTE
    "WANDB_API_KEY":     "wandb_v1_YOUR_KEY_HERE",     # ⬅️ PASTE
    "HF_TOKEN":          "hf_YOUR_TOKEN_HERE",         # ⬅️ PASTE
    
    # Keep these settings for pilot run
    "TEACHER_PROVIDER":  "groq",
    "GRPO_STEPS":        200,    # 200 = pilot (4h), 1500 = full (36h)
    "HF_ORG":            "Jeeeeet123",
    "WANDB_PROJECT":     "adaptivethink-samsung",
    ...
}
```

### Step 4: Run Training
- Click **"Run All"** button at the very top of the notebook
- Or press: `Ctrl+Alt+Enter`

---

## What Happens During Training

### Phase 1: Setup (15 minutes)
- Cell 1: Environment check ✅
- Cell 2: Create directories ✅
- Cell 3: Inject API keys ✅
- Cell 4: Install dependencies (transformers, peft, trl, etc.) ✅
- Cell 5: Run tests (non-blocking) ⚠️

### Phase 2: Data (30 minutes)
- Cell 6: Download datasets (GSM8K, MATH, etc.) 📥
- Cell 7: **SKIP THIS** - Teacher labels take 4-10 hours (only if you want real difficulty labels)
- Cell 8: Build training data ✅

### Phase 3: Verifier Training (2-4 hours)
- Cell 9: Train 400M difficulty verifier 🎯
- Cell 10: Push verifier to HuggingFace ☁️

### Phase 4: Pilot Training (4 hours) - **THE MAIN EVENT**
- Cell 11: Smoke test (50 steps) ✅
- Cell 12: **PILOT RUN** (200 steps, ~4 hours) 🚀
- Cell 13: **GO/NO-GO DECISION** 🎯

### After Pilot:
- **If Cell 13 says "Case A"**: ✅ Change `GRPO_STEPS: 200` → `1500`, re-run Cell 12 (36h)
- **If Case B/C**: ⚠️ Tune hyperparams first

---

## Monitor Progress

### WandB Dashboard
https://wandb.ai/jeeth-kataria/adaptivethink-samsung

### Local Logs
```bash
tail -f logs/grpo_seed0.log
```

### VS Code Output
Check the output below each cell as it runs

---

## Checkpoints (Auto-Save)

- **Every 50 steps** saved to:
  - `outputs/router-seed0/checkpoint-X/`
  - HuggingFace Hub: `Jeeeeet123/router-1p5b-seed0`
- **If interrupted**: Just re-run the cell, it auto-resumes!
- **Max work lost**: 50 steps (~15 min)

---

## If Something Breaks

### "Module not found"
```bash
conda activate adaptivethink
pip install <missing-module>
```

### "CUDA out of memory"
Edit Cell 0:
```python
"GRPO_GROUP_SIZE": 4,
"GRPO_MAX_SEQ_LEN": 1024,
```

### "Tests failed"
**Ignore it!** Tests are non-blocking. Training continues.

### Cell stuck/hanging
- Click the stop button (⬛) next to the cell
- Re-run that cell

---

## Quick Commands Summary

```bash
# Setup
cd /media/ccps/EXT3/AdaptiveThink
conda activate adaptivethink

# Open notebook
code notebooks/AdaptiveThink_Train.ipynb

# Monitor
tail -f logs/grpo_seed0.log

# Check GPU
nvidia-smi

# If you need to restart
# Just click "Run All" again - checkpoints auto-resume!
```

---

## Timeline

| Task | Time | Can Skip? |
|------|------|-----------|
| Setup (Cells 1-6) | 15 min | No |
| Teacher labels (Cell 7) | 4-10 hours | **YES** - use stub data |
| Verifier training (Cell 9) | 2-4 hours | No |
| Pilot training (Cell 12) | 4 hours | No |
| **TOTAL (skipping Cell 7)** | **~7 hours** | |

---

## You're Ready! 🚀

1. ✅ Setup conda environment
2. ✅ Get 4 API keys
3. ✅ Open notebook in VS Code
4. ✅ Paste keys in Cell 0
5. ✅ Click "Run All"

**That's it!** The notebook handles everything else automatically.
