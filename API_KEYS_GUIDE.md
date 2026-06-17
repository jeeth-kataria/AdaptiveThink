# API Keys Setup Guide - Exact Steps

## You Need 3 API Keys (All FREE)

1. **DeepInfra** - For teacher difficulty labels (~$1.26 cost)
2. **WandB** - For training monitoring (100% free)
3. **HuggingFace** - For checkpoint backup (100% free)

---

## 1️⃣ DeepInfra API Key (~$1.26 credit needed)

### Step-by-Step:

1. **Go to:** https://deepinfra.com

2. **Sign up:**
   - Click "Sign In" (top right)
   - Choose "Sign up with Google" or "Sign up with GitHub"
   - Click through authorization

3. **Add payment method:**
   - Click your profile icon (top right) → "Billing"
   - Click "Add Payment Method"
   - Enter credit card (they'll charge ~$1.26 for teacher labels)
   - OR add $5 credit to start

4. **Get API key:**
   - Click your profile icon → "API Keys"
   - Click "Create API Key"
   - Name it: "AdaptiveThink"
   - Click "Create"
   - **Copy the key** (looks like: `sk-abc123xyz...`)
   - ⚠️ Save it now - you can't see it again!

**Cost:** ~$1.26 for 12,000 teacher labels (one-time)

---

## 2️⃣ WandB API Key (FREE)

### Step-by-Step:

1. **Go to:** https://wandb.ai/authorize

2. **Sign up:**
   - Click "Sign up"
   - Choose "Sign up with Google" or "Sign up with GitHub"
   - Click through authorization

3. **Get API key:**
   - After signup, you're automatically at: https://wandb.ai/authorize
   - **The key is right there on the page!**
   - It looks like: `abc123xyz...` (long string)
   - Click "Copy" button
   - That's it!

**Alternative if you're already logged in:**
- Go to: https://wandb.ai/settings
- Scroll down to "Danger Zone"
- Click "API keys" tab
- Copy the existing key or create new one

**Cost:** FREE (100GB storage, unlimited runs)

---

## 3️⃣ HuggingFace Token (FREE)

### Step-by-Step:

1. **Go to:** https://huggingface.co/join

2. **Sign up:**
   - Enter email, username, password
   - Verify email (check inbox)
   - Click verification link

3. **Get Write token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "AdaptiveThink"
   - **Type: Select "Write"** ← IMPORTANT! (Not "Read")
   - Click "Create token"
   - **Copy the token** (looks like: `hf_abc123xyz...`)
   - ⚠️ Save it - you can't see it again!

**Why "Write" permission?**
- Notebook needs to upload checkpoints to your HuggingFace account
- "Read" tokens won't work - uploads will fail

**Cost:** FREE (unlimited public models, private repos also free)

---

## Summary Checklist

After following all steps, you should have:

```
✅ DeepInfra key:    sk-abc123xyz...  (starts with "sk-")
✅ WandB key:        abc123xyz...     (just numbers/letters)
✅ HuggingFace token: hf_abc123xyz... (starts with "hf_")
```

---

## Where to Paste Them

**In Jupyter Notebook → Cell 0:**

```python
CFG = {
    "DEEPINFRA_API_KEY": "sk-abc123xyz...",      # ← Paste here
    "WANDB_API_KEY": "abc123xyz...",             # ← Paste here
    "HF_TOKEN": "hf_abc123xyz...",               # ← Paste here
    
    # Rest of config...
}
```

**⚠️ Important:**
- Remove quotes if they're in your paste
- No extra spaces before/after
- Full key (don't truncate)

---

## Test Your Keys (Optional)

Before running full notebook, test keys work:

**In a new Jupyter cell:**

```python
import os

# Test DeepInfra
from openai import OpenAI
try:
    client = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai"
    )
    models = client.models.list()
    print("✅ DeepInfra: Working! First model:", models.data[0].id)
except Exception as e:
    print("❌ DeepInfra failed:", str(e)[:100])

# Test WandB
import wandb
try:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    user = wandb.api.viewer()
    print("✅ WandB: Working! Logged in as:", user["username"])
except Exception as e:
    print("❌ WandB failed:", str(e)[:100])

# Test HuggingFace
from huggingface_hub import HfApi
try:
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()
    print("✅ HuggingFace: Working! Logged in as:", user["name"])
except Exception as e:
    print("❌ HuggingFace failed:", str(e)[:100])
```

**Expected output:**
```
✅ DeepInfra: Working! First model: deepseek-ai/DeepSeek-V3
✅ WandB: Working! Logged in as: your_username
✅ HuggingFace: Working! Logged in as: Your Name
```

If all three show ✅ → **You're ready to run!**

---

## Troubleshooting

### "DeepInfra: 401 Unauthorized"
- Key is wrong or expired
- Go back to https://deepinfra.com/dash/api_keys
- Create new key
- Copy it again carefully

### "WandB: Authentication failed"
- Key is wrong
- Go to https://wandb.ai/authorize
- Copy the key shown on that page (it's always visible)

### "HuggingFace: 403 Forbidden" or "Token does not have write access"
- You created "Read" token instead of "Write"
- Go to https://huggingface.co/settings/tokens
- Delete old token
- Create new token with **Type: Write**
- Copy the new one

### "Invalid API key format"
- Check you copied the FULL key (not truncated)
- Check no extra spaces or quotes
- DeepInfra keys start with `sk-`
- HuggingFace tokens start with `hf_`

---

## Cost Breakdown

| Service | Signup | Usage | Your Cost |
|---------|--------|-------|-----------|
| DeepInfra | FREE | Pay-per-use | ~$1.26 (teacher labels) |
| WandB | FREE | FREE | $0 |
| HuggingFace | FREE | FREE | $0 |
| **TOTAL** | | | **$1.26** |

**Plus Azure VM costs:**
- T4 pilot: ~$2.40 (12h @ $0.20/hr)
- A100 full: ~$27.00 (36h @ $0.75/hr)

**Grand total: ~$30.66 for complete run**

---

## Privacy & Security

**Safe:**
- ✅ These are YOUR keys for YOUR accounts
- ✅ Notebook only uses them for training (teacher labels, monitoring, checkpoints)
- ✅ Keys stored in notebook cell (not shared with anyone)
- ✅ Azure VM is yours (not shared)

**NOT safe:**
- ❌ Don't share keys in public repos
- ❌ Don't paste keys in Discord/Slack
- ❌ Don't commit keys to GitHub

**If keys leak:**
- DeepInfra: Someone could use your API credit
- WandB: Someone could see your training runs
- HuggingFace: Someone could upload to your repos

**To revoke:**
- DeepInfra: https://deepinfra.com/dash/api_keys → Delete
- WandB: https://wandb.ai/settings → Revoke token
- HuggingFace: https://huggingface.co/settings/tokens → Revoke

---

## Ready?

Once you have all 3 keys:

1. ✅ Paste them in Cell 0
2. ✅ Set `GRPO_STEPS: 200` (pilot)
3. ✅ Click "Run All"
4. ✅ Walk away for 12 hours

**Everything else is automatic!**
