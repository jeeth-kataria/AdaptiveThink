"""
Unit tests for the difficulty verifier — no GPU required.

These run before any training to catch:
  1. Output range: sigmoid(logit) must be in [0, 1].
  2. d informativeness: trivially different inputs must produce different scores
     (guards against the "all 0.5" silent failure mode).
  3. load_verifier API: function exists and returns (model, tokenizer).
"""
import torch
import pytest
from adaptivethink.verifier.model import DifficultyVerifier


@pytest.fixture(scope="module")
def verifier():
    """CPU verifier with random weights — no checkpoint needed."""
    return DifficultyVerifier.__new__(DifficultyVerifier)


def test_output_in_unit_interval():
    """sigmoid(logit) must always be in [0, 1] regardless of input magnitude."""
    from adaptivethink.verifier.model import DifficultyVerifier
    model = DifficultyVerifier()
    # random token IDs; mask everything attended
    ids = torch.randint(0, 1000, (4, 16))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits = model(ids, mask)
        scores = torch.sigmoid(logits)
    assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), \
        f"Scores out of [0,1]: {scores.tolist()}"


def test_scores_not_all_identical():
    """Different questions must produce different scores (d must be informative).

    Uses random weights, so we only check that the model CAN differentiate —
    i.e. that the architecture doesn't collapse outputs. A real verifier with
    trained weights will be much more differentiated.
    """
    from adaptivethink.verifier.model import DifficultyVerifier
    from transformers import AutoTokenizer

    model = DifficultyVerifier()
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        pytest.skip("Tokenizer not available (network required)")

    easy = "What is 2 + 2?"
    hard = ("Compute the sum of all positive integers n ≤ 1000 such that "
            "n is divisible by 7 or 11 but not both, and prove your result.")

    scores = model.score([easy, hard], tok, device="cpu")
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores), f"Out of range: {scores}"
    # With random weights this difference will be small but non-zero
    # (if it's exactly 0 the head has collapsed — that's a bug)
    assert scores[0] != scores[1], \
        "Both inputs scored identically — verifier head may have collapsed"


def test_score_batch_consistency():
    """Batched scoring must match single-item scoring."""
    from adaptivethink.verifier.model import DifficultyVerifier
    from transformers import AutoTokenizer

    model = DifficultyVerifier()
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        pytest.skip("Tokenizer not available")

    questions = ["What is 1+1?", "What is 3+3?", "What is 5+5?"]
    batch_scores = model.score(questions, tok, device="cpu", batch_size=32)
    single_scores = [model.score([q], tok, device="cpu")[0] for q in questions]

    for b, s in zip(batch_scores, single_scores):
        assert abs(b - s) < 1e-5, f"Batch vs single mismatch: {b} vs {s}"


def test_load_verifier_api_exists():
    """load_verifier function must exist and have the right signature."""
    from adaptivethink.verifier.model import load_verifier
    import inspect
    sig = inspect.signature(load_verifier)
    assert "ckpt_path" in sig.parameters
    assert "device" in sig.parameters
