# Groq API Setup for Teacher Labels

## What Changed

AdaptiveThink now supports **Groq** as a teacher label provider with **dual API key fallback** for rate limit protection.

**Benefits:**
- ✅ **FREE tier** (vs $1.26 for DeepInfra)
- ✅ **Fast inference** (llama-3.3-70b-versatile)
- ✅ **Automatic fallback** to key 2 if key 1 hits rate limit
- ✅ **Bimodal prompt** optimized to avoid 0.5 clustering

---

## Get Groq API Keys (2 minutes)

### Step 1: Create Account
1. Go to: https://groq.com
2. Click "Get Started" or "Sign Up"
3. Sign up with Google/GitHub

### Step 2: Get API Key 1
1. Go to: https://console.groq.com/keys
2. Click "Create API Key"
3. Name it: "AdaptiveThink-Key1"
4. Click "Create"
5. **Copy the key** (starts with `gsk_...`)
6. Save it somewhere safe

### Step 3: Get API Key 2
1. Same page: https://console.groq.com/keys
2. Click "Create API Key" again
3. Name it: "AdaptiveThink-Key2"
4. Click "Create"
5. **Copy the second key**

**Why two keys?**
- Groq free tier has rate limits
- If key 1 hits limit (429 error), script automatically retries with key 2
- Doubles your effective rate limit

---

## Configure Notebook

**In `notebooks/AdaptiveThink_Train.ipynb` Cell 0:**

```python
CFG = {
    # Paste your Groq keys here
    "GROQ_API_KEY_1": "gsk_abc123...",    # ← Your first key
    "GROQ_API_KEY_2": "gsk_xyz789...",    # ← Your second key
    
    # Other keys still needed
    "WANDB_API_KEY": "...",
    "HF_TOKEN": "hf_...",
    
    # Teacher provider is now Groq
    "TEACHER_PROVIDER": "groq",  # Already set by default
    
    # Rest of config...
}
```

**That's it!** The notebook will now use Groq instead of DeepInfra.

---

## How It Works

### Automatic Fallback

```python
# Try key 1
difficulty = call_groq(question, key=GROQ_API_KEY_1)

# If 429 rate limit or error
if difficulty is None:
    print("[fallback] Retrying with API key 2")
    difficulty = call_groq(question, key=GROQ_API_KEY_2)
```

### Bimodal Prompt

The Groq prompt is specially designed to push scores away from 0.5:

```
System:
You are a math question difficulty rater. Your only job is to output a single float between 0.0 and 1.0.
0.0 = trivially easy (single arithmetic step, no reasoning needed)
1.0 = extremely hard (multi-step reasoning, algebra, or word problem requiring planning)
Be bimodal — push easy questions toward 0.0–0.2 and hard ones toward 0.8–1.0. Avoid clustering near 0.5.
Respond with ONLY a float. No explanation, no units, no extra text.

User:
Rate the difficulty of this math question:
{question}
```

**Why bimodal?**
- Without this instruction, Llama tends to output 0.5 for everything
- Mode-collapsed labels kill the verifier (it can't learn)
- Cell 6 in notebook will warn you if distribution looks bad

### Model Used

- **Groq model:** `llama-3.3-70b-versatile`
- **Why this model?** Best Groq model for instruction following
- **Context window:** 32k tokens (more than enough for math questions)

---

## Cost Comparison

| Provider | Model | Cost | Notes |
|----------|-------|------|-------|
| **Groq** | llama-3.3-70b | **FREE** | Rate limited, dual keys help |
| DeepInfra | DeepSeek-V3 | $1.26 | 12k items × 3 calls |
| OpenAI | gpt-4o-mini | $5-10 | Fast but expensive |
| Together | Qwen2.5-72B | $2-3 | Mid-range |

**Groq saves you $1.26** on teacher labels (and it's faster!)

---

## Rate Limits (Free Tier)

**Groq free tier:**
- 14,400 requests/day per key
- With 2 keys: 28,800 requests/day
- Your workload: 12,000 items × 3 calls = 36,000 requests
- **Will take 2 days** (18k requests day 1, 18k requests day 2)

**Strategies:**
1. **Split the run:** Run notebook on Day 1, continue on Day 2
2. **Cache is smart:** Already-labeled items skip API calls
3. **Key fallback:** Second key doubles throughput

---

## Troubleshooting

### "Rate limit exceeded (429)"
- Normal! Script automatically switches to key 2
- If both keys exhausted, wait 24h and re-run
- Cache saves progress, you won't re-label same items

### "Invalid API key"
- Check you copied full key (starts with `gsk_`)
- No spaces before/after
- Go to https://console.groq.com/keys and verify key is active

### "Model not found"
- Check Groq status: https://status.groq.com
- Model should be: `llama-3.3-70b-versatile`
- If unavailable, switch to DeepInfra temporarily

### "All scores are 0.5"
- Groq prompt should prevent this (bimodal instruction)
- If it happens, Cell 6 will warn you
- Try increasing temperature in `teacher_labels.py` (line 12): `temperature=0.3`

---

## Switch Back to DeepInfra

If you prefer DeepInfra (paid but faster):

```python
CFG = {
    "DEEPINFRA_API_KEY": "sk_...",     # Get from deepinfra.com
    "TEACHER_PROVIDER": "deepinfra",   # Change from "groq"
    # Remove Groq keys or leave blank
}
```

---

## Technical Details

**File changed:** `src/adaptivethink/data/teacher_labels.py`

**What was added:**
1. New provider option: `"groq"` in `_get_client()`
2. Groq-specific system prompt with bimodal instruction
3. Dual key logic: `_get_client(provider, key_num=1 or 2)`
4. Automatic retry on 429 errors
5. Simple float response parsing (no JSON needed)

**Cell 0 validation logic:**
- Only requires keys for active provider
- `GROQ_API_KEY_1` required when `TEACHER_PROVIDER == "groq"`
- `DEEPINFRA_API_KEY` only required when `provider == "deepinfra"`

---

## Next Steps

1. ✅ Get 2 Groq API keys (https://console.groq.com/keys)
2. ✅ Paste keys in notebook Cell 0
3. ✅ Verify `TEACHER_PROVIDER: "groq"` (default)
4. ✅ Run notebook Cell 7 (teacher labels)
5. ✅ Check Cell 6 output for distribution quality

**Free, fast, and automatic fallback. Ready to use!**

---

## Summary

**Before:** DeepInfra, $1.26, single key  
**After:** Groq, FREE, dual keys with auto-fallback  

**Prompt optimized for Llama** to avoid 0.5 clustering (critical for verifier quality).

**Repository updated:** https://github.com/jeeth-kataria/AdaptiveThink
