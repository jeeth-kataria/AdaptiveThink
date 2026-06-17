# ── Cell 10: Push verifier + difficulty labels to HF Hub ─────────────────────
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
OUTPUTS_DIR = _AT_OUTPUTS_DIR
DATA_DIR    = _AT_DATA_DIR

# ── Verifier model ────────────────────────────────────────────────────────────
verifier_dir = OUTPUTS_DIR / "verifier-400m"
repo = CFG["VERIFIER_REPO"]
print(f"Pushing verifier to {repo} ...")
api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=str(verifier_dir), repo_id=repo,
                  commit_message="verifier v0.1 — automated push from notebook")

# Push model card
card = f"""---
license: apache-2.0
tags: [adaptivethink, difficulty-verifier, qwen2.5]
---
# {repo}

400M cross-encoder difficulty verifier distilled from DeepSeek-V3 teacher labels.
Outputs a calibrated difficulty score `d ∈ [0,1]` for a given question,
where 0 = trivial and 1 = requires full chain-of-thought.

Used as the gating term in the AdaptiveThink GRPO reward:
`r = correctness − λ_tok · tokens · (1−d)`

**Spearman ρ** on held-out set: see training logs in wandb project `{CFG['WANDB_PROJECT']}`.

## Usage
```python
from adaptivethink.verifier.model import load_verifier
model, tok = load_verifier("outputs/verifier-400m/best.pt")
scores = model.score(["What is 2+2?", "Prove Fermat's Last Theorem."], tok)
```

## Training
- Encoder: `Qwen/Qwen2.5-0.5B-Instruct`
- Labels: DeepSeek-V3 teacher, 3 calls/item, mean difficulty
- Loss: MSE + 0.2×BCE auxiliary
- License: Apache-2.0
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
print(f"  ✅ {repo}")

# ── Difficulty labels dataset ─────────────────────────────────────────────────
labels_repo = CFG["LABELS_REPO"]
api.create_repo(repo_id=labels_repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=str(DATA_DIR / "teacher_labels.jsonl"),
                path_in_repo="teacher_labels.jsonl", repo_id=labels_repo,
                repo_type="dataset",
                commit_message="difficulty labels v0.1 — 12k items")
print(f"  ✅ {labels_repo}")
print("\n✅ HF push complete.")
