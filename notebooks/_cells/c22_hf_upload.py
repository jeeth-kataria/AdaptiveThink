# ── Cell 22: Push all artefacts to HF Hub ────────────────────────────────────
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
OUTPUTS_DIR = _AT_OUTPUTS_DIR

uploads = [
    # (repo_id, local_path, repo_type, commit_msg)
    (CFG["VERIFIER_REPO"],
     OUTPUTS_DIR / "verifier-400m",
     "model", "verifier v1.0 final"),
    
    (f"{CFG['HF_ORG']}/router-1p5b-lora",
     _AT_BEST_SEED_DIR,
     "model", f"router LoRA v1.0 — best of 3 seeds"),
    
    (f"{CFG['HF_ORG']}/router-1p5b-merged",
     OUTPUTS_DIR / "router-merged",
     "model", "router merged FP16 v1.0"),
    
    (CFG["LABELS_REPO"],
     _AT_DATA_DIR / "teacher_labels.jsonl",
     "dataset", "difficulty labels v1.0 — 12k items"),
]

for repo, path, rtype, msg in uploads:
    if not Path(path).exists():
        print(f"  ⏭️  {path} does not exist — skipping {repo}")
        continue
    print(f"  Pushing {repo} ...")
    api.create_repo(repo_id=repo, repo_type=rtype, private=True, exist_ok=True)
    if Path(path).is_dir():
        api.upload_folder(folder_path=str(path), repo_id=repo,
                          repo_type=rtype, commit_message=msg)
    else:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name,
                        repo_id=repo, repo_type=rtype, commit_message=msg)
    print(f"  ✅ {repo}")

print("\n✅ All HF uploads complete.")
print(f"   Verifier:   https://huggingface.co/{CFG['VERIFIER_REPO']}")
print(f"   Router LoRA: https://huggingface.co/{CFG['HF_ORG']}/router-1p5b-lora")
print(f"   Merged:     https://huggingface.co/{CFG['HF_ORG']}/router-1p5b-merged")
print(f"   Labels:     https://huggingface.co/{CFG['LABELS_REPO']}")
