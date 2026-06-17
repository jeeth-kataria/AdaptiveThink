# 🚀 AdaptiveThink - Quick Reference

## ✅ Everything is Now on GitHub!

**Your repo:** https://github.com/jeeth-kataria/AdaptiveThink

---

## 📦 One-Command Setup

```bash
# On Azure VM (or any Linux with GPU)
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink
bash quick_start.sh
```

**This automatically:**
- ✅ Installs all dependencies (torch, transformers, trl, unsloth, etc.)
- ✅ Detects your GPU and configures settings
- ✅ Creates all required directories
- ✅ Shows you next steps

**Time:** ~5 minutes  
**Requirements:** Python 3.10+, CUDA 11.8+, 14GB+ VRAM

---

## 🔑 Get API Keys (5 minutes)

See `API_KEYS_GUIDE.md` for detailed steps, or:

**1. DeepInfra** (~$1.26 for teacher labels)
- https://deepinfra.com/dash/api_keys
- Sign up → Create key → Copy (starts with `sk-`)

**2. WandB** (FREE)
- https://wandb.ai/authorize
- Sign up → Copy key shown on page

**3. HuggingFace** (FREE)
- https://huggingface.co/settings/tokens
- New token → **Type: Write** → Copy (starts with `hf_`)

---

## 📓 Run the Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/AdaptiveThink_Train.ipynb
```

**In the notebook:**

1. **Cell 0:** Paste your 3 API keys
2. **Set pilot mode:**
   ```python
   "GRPO_SEEDS": [0],       # Single seed
   "GRPO_STEPS": 200,       # Pilot (4h)
   ```
3. **Click:** "Run All"
4. **Wait:** ~12 hours for pilot to complete

---

## 🖥️ Auto-Adapts to Your GPU

The notebook automatically detects and configures for:

| GPU | VRAM | Config Applied |
|-----|------|----------------|
| **T4** | 16GB | No vLLM, group_size=4, seq_len=1024 |
| **V100** | 16-32GB | No vLLM, group_size=4, seq_len=1024 |
| **A100** | 80GB | vLLM enabled, group_size=8, seq_len=2048 |

**You don't configure anything!** Just run the notebook.

---

## 💾 Auto-Saves Everything

**5 layers of protection:**

1. ✅ **Local checkpoints** every 50 steps
2. ✅ **HuggingFace Hub backup** every 50 steps
3. ✅ **Auto-resume** if interrupted
4. ✅ **Teacher labels cached** (never pay twice)
5. ✅ **Verifier saved** before GRPO starts

**Max work lost if VM crashes:** 50 steps (~15 minutes)

---

## 📊 Monitor Training

**WandB Dashboard:** https://wandb.ai/YOUR_USERNAME/adaptivethink
- Live reward curve
- Think rate
- KL divergence

**Or SSH into VM:**
```bash
tail -f logs/grpo_seed0.log
```

---

## 💰 Cost Breakdown

| Stage | VM | Time | Cost |
|-------|-----|------|------|
| Data + Verifier | T4 | 6h | $1.20 |
| **Pilot (200 steps)** | **T4** | **4h** | **$0.80** |
| Teacher labels API | - | - | $1.26 |
| **PILOT TOTAL** | | **10h** | **~$3.26** |
| | | | |
| **Full (1500 steps)** | **A100** | **36h** | **$27.00** |
| **FULL TOTAL** | | **46h** | **~$30.26** |

**Your $150 budget:** Enough for 5 complete runs

---

## 🎯 Recommended Flow

### **Run 1: Smoke Test** (30 min, FREE)
```python
# In Cell 0:
"GRPO_STEPS": 50,
```
- Validates environment works
- No API costs

### **Run 2: Pilot** (4h, ~$5)
```python
"GRPO_STEPS": 200,
```
- Real training with verifier
- Check if reward curve goes up
- **Decision gate:** GO or NO-GO for full training

### **Run 3: Full Training** (36h, ~$27)
Only if pilot succeeded!
```python
"GRPO_STEPS": 1500,
```
- Switch to A100 VM
- Your hackathon demo model

---

## 📚 Complete Guides

| Guide | What's Inside |
|-------|---------------|
| `API_KEYS_GUIDE.md` | Exact steps for all 3 API keys |
| `AZURE_SETUP_GUIDE.md` | Azure VM creation commands |
| `SAFETY_CHECKLIST.md` | Verify all protections work |
| `NOTEBOOK_GUIDE.md` | Detailed notebook walkthrough |
| `WHERE_TO_RUN.md` | Alternative environments |

---

## 🆘 Troubleshooting

**"No CUDA GPU detected"**
- Check: `nvidia-smi` shows GPU
- VM must be NC-series (T4/V100/A100)

**"API key invalid"**
- Check you pasted full key (no spaces)
- DeepInfra keys start with `sk-`
- HuggingFace tokens start with `hf_`

**"Out of Memory (OOM)"**
- T4 is running A100 config (shouldn't happen - auto-detect should fix)
- In Cell 0: Set `"GRPO_GROUP_SIZE": 4, "GRPO_MAX_SEQ_LEN": 1024`

**"Training stopped mid-run"**
- Check WandB: https://wandb.ai
- Checkpoints saved to HuggingFace Hub
- Just re-run Cell 12 → auto-resumes

---

## ✅ Pre-Flight Checklist

Before clicking "Run All":

- [ ] `git clone` completed
- [ ] `bash quick_start.sh` completed
- [ ] All 3 API keys obtained
- [ ] Cell 0 edited with API keys
- [ ] `GRPO_STEPS` set to 200 (pilot)
- [ ] Jupyter running at http://localhost:8888

**If all checked → Click "Run All" and relax!**

---

## 🎓 What You'll Get

**After pilot (200 steps):**
- ✅ Proof environment works
- ✅ Reward curve showing training is learning
- ✅ Think rate metric (routing health)
- ✅ Decision: proceed to full training or debug

**After full training (1500 steps):**
- ✅ Trained model (~65-70% GSM8K accuracy)
- ✅ GGUF quantized model (~1.2GB)
- ✅ Pareto chart (accuracy vs compute)
- ✅ Ready for hackathon demo

---

## 🔗 Links

- **GitHub Repo:** https://github.com/jeeth-kataria/AdaptiveThink
- **DeepInfra API:** https://deepinfra.com/dash/api_keys
- **WandB Dashboard:** https://wandb.ai/authorize
- **HuggingFace Tokens:** https://huggingface.co/settings/tokens
- **Azure Portal:** https://portal.azure.com

---

## 🚀 Ready? Go!

```bash
# On Azure VM
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink
bash quick_start.sh
jupyter notebook notebooks/AdaptiveThink_Train.ipynb
```

**Good luck with your hackathon! 🏆**
