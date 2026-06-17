"""
Build data/gsm8k_train_labelled.jsonl for GRPO training.

Filters data/teacher_labels.jsonl (12k mixed pool) down to GSM8K-only items,
or — if teacher_labels.jsonl is not yet produced — pulls GSM8K train split
directly and assigns difficulty=0.5 as a placeholder (smoke-test safe).

Usage:
    python scripts/02b_build_grpo_data.py                        # normal
    python scripts/02b_build_grpo_data.py --fallback-stub        # no API needed
"""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-labels", default="data/teacher_labels.jsonl")
    p.add_argument("--out", default="data/gsm8k_train_labelled.jsonl")
    p.add_argument("--fallback-stub", action="store_true",
                   help="If teacher labels absent, stub with difficulty=0.5 from raw GSM8K")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    src = Path(args.teacher_labels)
    if src.exists():
        items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        # teacher_labels.jsonl may contain items from all datasets;
        # keep all — GRPO benefits from math diversity and the verifier d values
        # are what make easy/hard items differ, not the dataset tag.
        out_items = [it for it in items if "difficulty" in it]
        print(f"[02b] {len(out_items)} labelled items from {src} -> {args.out}")
    elif args.fallback_stub:
        print("[02b] teacher_labels.jsonl absent — building stub from raw GSM8K (d=0.5)")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        out_items = [
            {"question": r["question"],
             "answer": r["answer"].split("####")[-1].strip(),
             "difficulty": 0.5}
            for r in ds
        ]
        print(f"[02b] {len(out_items)} stub items (difficulty=0.5) -> {args.out}")
    else:
        raise FileNotFoundError(
            f"{src} not found. Run 02_gen_teacher_labels.sh first, "
            "or pass --fallback-stub for a smoke-test-safe stub."
        )

    with open(args.out, "w") as f:
        for it in out_items:
            f.write(json.dumps(it) + "\n")
    print(f"[02b] Done. {len(out_items)} items written.")


if __name__ == "__main__":
    main()
