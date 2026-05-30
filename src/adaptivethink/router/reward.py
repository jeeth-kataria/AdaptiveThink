"""
Reward function for GRPO router training.

Novelty vs AdaptThink/CODA: difficulty `d` is from an *external* distilled
verifier, not internal model confidence or group-rollout pass-rate.
"""
import re

ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Also accept plain "The answer is X" as fallback
PLAIN_RE = re.compile(r"(?:answer is|answer:|=)\s*([^\n.]+)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = PLAIN_RE.search(text)
    return m.group(1).strip() if m else None


def _answers_match(pred: str, gt: str) -> bool:
    """Normalise and compare: strip $, commas, spaces, lowercase."""
    def norm(s):
        return s.lower().replace("$", "").replace(",", "").replace(" ", "").strip()
    return norm(pred) == norm(gt)


def decision_from_response(response: str) -> str | None:
    s = response.strip()
    if s.startswith("<think>"):
        return "think"
    if s.startswith("<no_think>"):
        return "no_think"
    return None


def compute_rewards(
    responses: list[str],
    ground_truths: list[str],
    difficulties: list[float],
    token_counts: list[int] | None = None,
    lambda_tok: float = 3e-3,   # 5e-4 was too weak; 3e-3 penalises 300-token easy responses by ~0.9
    lambda_obey: float = 0.05,  # small enough that wrong+honoured stays negative
) -> list[float]:
    """
    Returns one scalar reward per response.
    token_counts: actual output token counts from the model (preferred over word count).
    """
    rewards = []
    for i, (resp, gt, d) in enumerate(zip(responses, ground_truths, difficulties)):
        pred = extract_answer(resp)
        correct = float(pred is not None and _answers_match(pred, gt))

        # Use actual token count if provided, else word count as proxy
        n_tok = token_counts[i] if token_counts else len(resp.split())
        length_penalty = lambda_tok * n_tok * (1.0 - float(d))

        decision = decision_from_response(resp)
        if decision == "think":
            honoured = float("</think>" in resp)
        elif decision == "no_think":
            honoured = float("</think>" not in resp)
        else:
            honoured = 0.0

        rewards.append(correct - length_penalty + lambda_obey * honoured * correct)
    return rewards
