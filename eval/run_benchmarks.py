"""
Evaluation harness (build step 6).

Measures Pass@1 (and optionally Pass@k) plus efficiency stats (avg tokens,
latency, think-rate) for any (model, adapter, route_mode) combination on
GSM8K / MMLU / StrategyQA / AIME24.

KPI workflow:
  1. Baseline  : base reasoner, no adapter, full CoT.
       python eval/run_benchmarks.py --benchmark all --route-mode always_think \
              --tag baseline --out results/baseline.json
  2. Router    : trained adapter, adaptive routing.
       python eval/run_benchmarks.py --benchmark all --adapter outputs/router-seed0 \
              --verifier-ckpt outputs/verifier-400m/best.pt --route-mode model \
              --tag router --out results/router.json
  3. Delta is computed by eval/plots.py (must be >= +5% on >= 2 benchmarks).

Pass@k uses the unbiased Chen et al. estimator over --n-samples sampled
completions per question.
"""
import argparse
import json
from pathlib import Path

from adaptivethink.data.loaders import load_benchmark
from adaptivethink.metrics import is_correct as _is_correct, pass_at_k
from adaptivethink.router.reward import extract_answer

BENCHMARKS = ["gsm8k", "mmlu", "strategyqa", "aime24"]


def eval_benchmark(pipeline, name, n, seed, k_list, n_samples):
    items = load_benchmark(name, seed=seed, n=n)
    per_item, n_correct, n_think, tok_total, lat_total = [], 0, 0, 0.0, 0.0
    pass_k_counts = {k: 0.0 for k in k_list}

    for it in items:
        q, gt = it["question"], it["answer"]
        # Pass@1 with a single (greedy) sample drives the headline accuracy + stats.
        res = pipeline.answer(q, greedy=True)
        correct = _is_correct(res.completion, gt)
        n_correct += int(correct)
        n_think += int(res.decision == "think")
        tok_total += res.n_tokens
        lat_total += res.latency_s

        if k_list:
            samples = [pipeline.answer(q, greedy=False) for _ in range(n_samples)]
            c = sum(_is_correct(s.completion, gt) for s in samples)
            for k in k_list:
                pass_k_counts[k] += pass_at_k(n_samples, c, k)

        per_item.append({
            "question": q[:200], "gt": gt, "pred": extract_answer(res.completion),
            "correct": correct, "decision": res.decision,
            "difficulty": round(res.difficulty, 3), "n_tokens": res.n_tokens,
            "latency_s": round(res.latency_s, 3),
        })

    m = len(items)
    return {
        "benchmark": name,
        "n": m,
        "pass@1": round(n_correct / m, 4) if m else 0.0,
        "pass@k": {str(k): round(v / m, 4) for k, v in pass_k_counts.items()} if k_list else {},
        "think_rate": round(n_think / m, 4) if m else 0.0,
        "avg_tokens": round(tok_total / m, 1) if m else 0.0,
        "avg_latency_s": round(lat_total / m, 4) if m else 0.0,
        "per_item": per_item,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--adapter", default=None, help="PEFT router adapter dir")
    p.add_argument("--verifier-ckpt", default=None)
    p.add_argument("--gguf", default=None, help="GGUF Q4_K_M file (on-device backend)")
    p.add_argument("--route-mode", default="model",
                   choices=["model", "threshold", "always_think", "never_think"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--benchmark", default="all",
                   choices=BENCHMARKS + ["all", "math500"])
    p.add_argument("--n", type=int, default=None, help="limit items per benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pass-k", default="", help='e.g. "1,8,64" — enables sampling')
    p.add_argument("--n-samples", type=int, default=8, help="samples for Pass@k")
    p.add_argument("--max-think-tokens", type=int, default=1024)
    p.add_argument("--max-answer-tokens", type=int, default=256)
    p.add_argument("--no-budget-force", action="store_true")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from adaptivethink.inference.pipeline import build_pipeline
    pipeline = build_pipeline(
        model_name=args.model_name,
        adapter_path=args.adapter,
        verifier_ckpt=args.verifier_ckpt,
        gguf_path=args.gguf,
        device=device,
        route_mode=args.route_mode,
        threshold=args.threshold,
        max_think_tokens=args.max_think_tokens,
        max_answer_tokens=args.max_answer_tokens,
        budget_force=not args.no_budget_force,
    )

    k_list = [int(x) for x in args.pass_k.split(",") if x.strip()] if args.pass_k else []
    targets = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    results = {"tag": args.tag, "route_mode": args.route_mode,
               "adapter": args.adapter, "gguf": args.gguf, "benchmarks": {}}
    for name in targets:
        print(f"[eval] {name} ...")
        r = eval_benchmark(pipeline, name, args.n, args.seed, k_list, args.n_samples)
        results["benchmarks"][name] = r
        print(f"  {name}: Pass@1={r['pass@1']:.3f}  think={r['think_rate']:.2f}  "
              f"tok={r['avg_tokens']:.0f}  lat={r['avg_latency_s']:.2f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
