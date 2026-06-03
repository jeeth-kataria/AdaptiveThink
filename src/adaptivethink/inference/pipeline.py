"""
On-device adaptive reasoning pipeline (Stage 3).

Flow:  question -> 400M verifier -> difficulty d -> route -> answer.

Routing modes:
  - "model":      the RL-trained router decides via its first emitted token
                  (<think> / <no_think>). This is the headline system.
  - "threshold":  hard-route on the verifier score (d < threshold -> System-1,
                  else System-2). Matches the brief's Stage-3 description and is
                  the deterministic fallback.
  - "always_think" / "never_think": fixed-policy baselines for the Pareto chart.

Hard (System-2) branches use s1-style budget-forced extension: if the model
tries to stop before producing a boxed answer and is still under the token
budget, we append a short "Wait" continuation to push more reasoning.

Two backends:
  - "hf":   transformers (+ optional PEFT router adapter). Default for eval.
  - "gguf": llama-cpp-python over a Q4_K_M file. The on-device path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from adaptivethink.router.prompt import make_prompt, make_forced_prompt
from adaptivethink.router.reward import extract_answer, decision_from_response

BUDGET_FORCE_TEXT = "\nWait, let me re-check the reasoning step by step.\n"


@dataclass(frozen=True)
class RouteResult:
    question: str
    answer: str | None
    decision: str | None       # "think" | "no_think" | None
    difficulty: float
    n_tokens: int
    latency_s: float
    completion: str


# ── Backends ───────────────────────────────────────────────────────────────────


class HFBackend:
    """transformers generation, optionally with a PEFT router adapter merged in."""

    def __init__(self, model_name: str, adapter_path: str | None, device: str, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype or torch.bfloat16
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        self.model = model.to(device).eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, greedy: bool) -> tuple[str, int]:
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_p=0.95 if not greedy else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][enc.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        return text, int(new_ids.shape[0])


class GGUFBackend:
    """llama-cpp-python over a quantised Q4_K_M file (the on-device path)."""

    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, greedy: bool) -> tuple[str, int]:
        out = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0 if greedy else temperature,
            top_p=0.95,
            echo=False,
        )
        text = out["choices"][0]["text"]
        n_tok = int(out.get("usage", {}).get("completion_tokens", 0)) or _rough_token_count(text)
        return text, n_tok


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


# ── Pipeline ────────────────────────────────────────────────────────────────────


class AdaptivePipeline:
    def __init__(
        self,
        backend,
        verifier=None,
        verifier_tok=None,
        device: str = "cuda",
        route_mode: str = "model",
        threshold: float = 0.5,
        max_think_tokens: int = 1024,
        max_answer_tokens: int = 256,
        budget_force: bool = True,
        budget_force_max_rounds: int = 2,
        temperature: float = 0.7,
    ):
        assert route_mode in {"model", "threshold", "always_think", "never_think"}
        self.backend = backend
        self.verifier = verifier
        self.verifier_tok = verifier_tok
        self.device = device
        self.route_mode = route_mode
        self.threshold = threshold
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.budget_force = budget_force
        self.budget_force_max_rounds = budget_force_max_rounds
        self.temperature = temperature

    # difficulty ---------------------------------------------------------------

    def difficulty(self, question: str) -> float:
        if self.verifier is None or self.verifier_tok is None:
            return 0.5
        return float(self.verifier.score([question], self.verifier_tok, device=self.device)[0])

    # routing ------------------------------------------------------------------

    def _resolve_decision(self, question: str, d: float) -> str | None:
        if self.route_mode == "always_think":
            return "think"
        if self.route_mode == "never_think":
            return "no_think"
        if self.route_mode == "threshold":
            return "think" if d >= self.threshold else "no_think"
        return None  # "model" mode: the model decides during generation

    # generation ---------------------------------------------------------------

    def answer(self, question: str, greedy: bool = True) -> RouteResult:
        t0 = time.time()
        d = self.difficulty(question)
        forced = self._resolve_decision(question, d)

        if forced is None:
            prompt = make_prompt(question)
            budget = self.max_think_tokens  # let the model pick; allow full budget
            completion, n_tok = self.backend.generate(prompt, budget, self.temperature, greedy)
            decision = decision_from_response(completion)
        else:
            prompt = make_forced_prompt(question, forced)
            budget = self.max_think_tokens if forced == "think" else self.max_answer_tokens
            completion, n_tok = self.backend.generate(prompt, budget, self.temperature, greedy)
            decision = forced
            # the routing token is prefilled, so it is part of the logical completion
            completion = ("<think>" if forced == "think" else "<no_think>") + completion

        is_think = decision == "think"
        if is_think and self.budget_force:
            completion, extra = self._budget_extend(question, prompt, completion, n_tok, greedy)
            n_tok += extra

        return RouteResult(
            question=question,
            answer=extract_answer(completion),
            decision=decision,
            difficulty=d,
            n_tokens=n_tok,
            latency_s=time.time() - t0,
            completion=completion,
        )

    def _budget_extend(self, question, prompt, completion, n_tok, greedy) -> tuple[str, int]:
        """s1-style forcing: if no boxed answer yet and budget remains, push more thinking."""
        extra_total = 0
        for _ in range(self.budget_force_max_rounds):
            if extract_answer(completion) is not None:
                break
            if n_tok + extra_total >= self.max_think_tokens:
                break
            cont_prompt = prompt + completion + BUDGET_FORCE_TEXT
            remaining = self.max_think_tokens - (n_tok + extra_total)
            more, more_tok = self.backend.generate(
                cont_prompt, min(remaining, self.max_answer_tokens), self.temperature, greedy
            )
            completion += BUDGET_FORCE_TEXT + more
            extra_total += more_tok
        return completion, extra_total


# ── Loaders ─────────────────────────────────────────────────────────────────────


def build_pipeline(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    adapter_path: str | None = None,
    verifier_ckpt: str | None = None,
    gguf_path: str | None = None,
    device: str = "cuda",
    **kwargs,
) -> AdaptivePipeline:
    if gguf_path:
        backend = GGUFBackend(gguf_path)
    else:
        backend = HFBackend(model_name, adapter_path, device)

    verifier = vtok = None
    if verifier_ckpt:
        from adaptivethink.verifier.model import load_verifier

        verifier, vtok = load_verifier(verifier_ckpt, device)

    return AdaptivePipeline(backend, verifier, vtok, device=device, **kwargs)
