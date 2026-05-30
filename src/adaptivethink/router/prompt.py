"""Prompt templates for the routing head."""

SYSTEM = (
    "You are a math/reasoning assistant. "
    "Before answering, decide whether this question needs chain-of-thought reasoning. "
    "Start your response with EXACTLY one of:\n"
    "  <think>   — if you need step-by-step reasoning\n"
    "  <no_think> — if you can answer directly\n"
    "Then provide your answer ending with \\boxed{answer}."
)


def make_prompt(question: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
