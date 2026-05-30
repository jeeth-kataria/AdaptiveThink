"""Data loaders for GSM8K, MATH-500, StrategyQA, MMLU."""
from datasets import load_dataset


def load_gsm8k(split="train", seed=0, n=None):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]


def load_math500(seed=0, n=None):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["problem"], "answer": r["answer"]} for r in ds]


def load_strategyqa(seed=0, n=None):
    ds = load_dataset("wics/strategy-qa", split="test")
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": str(r["answer"])} for r in ds]


def load_mmlu(subjects=None, seed=0, n=None):
    subjects = subjects or ["high_school_mathematics", "college_mathematics", "abstract_algebra"]
    items = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        for r in ds:
            choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(r["choices"]))
            items.append({
                "question": f"{r['question']}\n{choices}",
                "answer": chr(65 + r["answer"]),
            })
    import random; rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n] if n else items


def build_verifier_pool(seed=0):
    """12k items for verifier distillation."""
    items = (
        load_gsm8k("train", seed, 6000)
        + load_math500(seed, 3000)
        + load_strategyqa(seed, 2000)
        + load_mmlu(seed=seed, n=1000)
    )
    import random; random.Random(seed).shuffle(items)
    return items


def build_verifier_eval(seed=42):
    """500-item held-out set, never used for training."""
    items = (
        load_gsm8k("test", seed, 125)
        + load_math500(seed, 125)
        + load_strategyqa(seed, 125)
        + load_mmlu(seed=seed, n=125)
    )
    import random; random.Random(seed).shuffle(items)
    return items
