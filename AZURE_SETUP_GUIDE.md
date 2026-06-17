# Azure VM Setup Guide - AdaptiveThink

## Prerequisites

1. **Azure Account** with $150 free credit
2. **Azure CLI** installed on your laptop
3. **SSH client** (built into macOS/Linux, Windows 10+)

---

## Step 1: Install Azure CLI (if needed)

```bash
# macOS
brew install azure-cli

# Or download from: https://aka.ms/installazureclimacos
```

---

## Step 2: Login to Azure

```bash
az login
# This opens a browser - sign in with your Azure account
```

---

## Step 3: Create Resource Group (one time)

```bash
az group create \
  --name adaptivethink-rg \
  --location eastus
```

---

## Step 4: Create T4 VM (for pilot training - $0.20/hr)

```bash
az vm create \
  --resource-group adaptivethink-rg \
  --name adaptivethink-t4 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --size Standard_NC4as_T4_v3 \
  --priority Spot \
  --max-price 0.50 \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard
```

**This will:**
- Create a VM with NVIDIA T4 GPU (16GB VRAM)
- Use spot pricing (~$0.20/hr instead of ~$0.70/hr)
- Auto-generate SSH keys (saved to `~/.ssh/`)
- Take ~3 minutes to create

**Output will show:**
```json
{
  "publicIpAddress": "XX.XX.XX.XX",
  ...
}
```

**Save this IP address!**

---

## Step 5: SSH into VM

```bash
# Replace XX.XX.XX.XX with your VM's IP from above
ssh azureuser@XX.XX.XX.XX
```

**First time:** It will ask "Are you sure you want to continue connecting?" → Type `yes`

---

## Step 6: Setup on VM (run these commands)

```bash
# Clone the repo
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink

# Run setup script
bash scripts/01_setup.sh
# This takes ~5 minutes, installs all dependencies

# Start Jupyter notebook
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

**Output will show:**
```
http://127.0.0.1:8888/?token=abc123xyz...
```

**Copy the full URL with token!**

---

## Step 7: Access Jupyter from Your Laptop

**Option A: SSH Tunnel (secure, recommended)**

Open a NEW terminal on your laptop (keep VM terminal running):

```bash
# Replace XX.XX.XX.XX with your VM IP
ssh -L 8888:localhost:8888 azureuser@XX.XX.XX.XX
```

Then open in browser: `http://localhost:8888` and paste the token from Step 6

**Option B: Direct Access (requires firewall rule)**

```bash
# On your laptop
az vm open-port \
  --resource-group adaptivethink-rg \
  --name adaptivethink-t4 \
  --port 8888 \
  --priority 1001
```

Then open: `http://XX.XX.XX.XX:8888/?token=abc123xyz...`

---

## Step 8: Open Notebook and Configure

In Jupyter, navigate to: `notebooks/AdaptiveThink_Train.ipynb`

**Edit Cell 0 only:**

```python
CFG = {
    "DEEPINFRA_API_KEY": "paste_your_key_here",
    "WANDB_API_KEY": "paste_your_key_here",
    "HF_TOKEN": "paste_your_key_here",
    
    "GRPO_SEEDS": [0],        # Single seed for hackathon
    "GRPO_STEPS": 200,        # Pilot first (4h)
    
    # Rest leave as default
}
```

**Get API keys:**
- DeepInfra: https://deepinfra.com → Sign up → Dashboard → API Keys
- WandB: https://wandb.ai/authorize (just click, it shows your key)
- HuggingFace: https://huggingface.co/settings/tokens → New token → Write permission

---

## Step 9: Run Pilot Training

**In Jupyter:**
1. Click "Cell" → "Run All" (or press Shift+Enter through each cell)
2. Watch for errors in cell outputs
3. Cells 1-11 take ~6 hours total:
   - Cell 4-6: Download data (~30 min)
   - Cell 7: Teacher labels (~2-4h, costs $1.26 API)
   - Cell 8-10: Verifier training (~2h)
   - Cell 11: Smoke test (~30 min)
4. Cell 12: Pilot training (~4h)
5. Cell 13: Evaluation (~20 min)

**Total time for pilot: ~10-12 hours**
**Total cost: ~$5 (T4 VM + API calls)**

---

## Step 10: Check Pilot Results

After Cell 13 finishes, look for:

✅ **Good Signs:**
- "✅ GO: Proceed to full training"
- Reward trend graph going UP
- Think rate between 30-70%
- Accuracy > 30% on GSM8K

❌ **Bad Signs:**
- "⚠️ NO-GO: Routing collapsed"
- Think rate < 10% or > 90%
- Reward flat or negative

---

## Step 11: Full Training (only if pilot succeeded)

**Stop T4 VM:**
```bash
az vm deallocate \
  --resource-group adaptivethink-rg \
  --name adaptivethink-t4
```

**Create A100 VM:**
```bash
az vm create \
  --resource-group adaptivethink-rg \
  --name adaptivethink-a100 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --size Standard_NC24ads_A100_v4 \
  --priority Spot \
  --max-price 1.50 \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard
```

**SSH into A100 VM, repeat Steps 6-7, then:**

**In notebook Cell 0:**
```python
"GRPO_STEPS": 1500,  # Change from 200 → 1500
```

**Run only Cell 12** (GRPO training)
- Takes ~36 hours
- Checkpoints save every 50 steps to HuggingFace
- If VM disconnects, just restart Cell 12 - it auto-resumes

---

## Monitoring Training

**WandB Dashboard:**
https://wandb.ai/YOUR_USERNAME/adaptivethink

Shows live:
- Reward curve
- Think rate
- KL divergence
- Token usage

**Check progress via SSH:**
```bash
ssh azureuser@<A100-VM-IP>
cd AdaptiveThink
tail -f logs/grpo_seed0.log
```

---

## Clean Up After Training

**Delete VMs to stop billing:**
```bash
# Delete T4 VM
az vm delete \
  --resource-group adaptivethink-rg \
  --name adaptivethink-t4 \
  --yes

# Delete A100 VM (after training finishes)
az vm delete \
  --resource-group adaptivethink-rg \
  --name adaptivethink-a100 \
  --yes
```

**Or just deallocate (keeps disk, can restart later):**
```bash
az vm deallocate --resource-group adaptivethink-rg --name adaptivethink-a100
```

---

## Troubleshooting

**Problem: "az: command not found"**
- Install Azure CLI (see Step 1)

**Problem: "No subscriptions found"**
- Run `az login` again
- Check you have an active Azure subscription

**Problem: "Quota exceeded" or "VM size not available"**
- Try different region: `--location westus2` or `--location westeurope`
- Or try on-demand instead of spot: remove `--priority Spot` lines

**Problem: SSH connection refused**
- Wait 2-3 minutes after VM creation
- Check VM is running: `az vm list -o table`

**Problem: Jupyter notebook doesn't load**
- Check the port 8888 is open (see Step 7 Option B)
- Or use SSH tunnel (Step 7 Option A)

**Problem: Cell fails with "API key invalid"**
- Double-check you copied the full key (no extra spaces)
- Test keys: https://deepinfra.com/dash/api_keys

**Problem: Out of Memory (OOM)**
- In Cell 0, change: `"GRPO_GROUP_SIZE": 4, "GRPO_MAX_SEQ_LEN": 1024`
- Re-run Cell 12

---

## Cost Summary

| Item | Cost |
|------|------|
| Teacher labels API | $1.26 |
| T4 VM (12h total) | $2.40 |
| A100 VM (36h) | $27.00 |
| **TOTAL** | **$30.66** |

**Your $150 budget allows 4 complete runs.**

---

## Need Help?

If you get stuck, paste the error message and I'll help debug!

Common places to get stuck:
1. Azure CLI login
2. SSH connection
3. Jupyter access
4. API key configuration
5. Notebook cell errors

Just show me the error and which step you're on.
