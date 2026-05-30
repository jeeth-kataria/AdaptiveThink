"""Unit tests for reward function — run before any training: pytest tests/test_reward.py"""
import pytest
from adaptivethink.router.reward import compute_rewards, extract_answer, decision_from_response, _answers_match


def test_extract_boxed():
    assert extract_answer("so \\boxed{42} is correct") == "42"
    assert extract_answer("no answer here") is None


def test_extract_plain_fallback():
    assert extract_answer("The answer is 42.") == "42"


def test_answers_match_normalise():
    assert _answers_match("$42$", "42")
    assert _answers_match("1,000", "1000")
    assert not _answers_match("42", "43")


def test_decision():
    assert decision_from_response("<think> reasoning...") == "think"
    assert decision_from_response("<no_think> 42") == "no_think"
    assert decision_from_response("random") is None


def test_correct_short_easy():
    r = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1])[0]
    assert r > 0.5, f"got {r}"


def test_correct_long_easy_penalised():
    long_resp = "<think> " + "word " * 300 + "</think> \\boxed{42}"
    r = compute_rewards([long_resp], ["42"], [0.1])[0]
    assert r < 0.5, f"long easy should be penalised, got {r}"


def test_correct_long_hard_not_penalised():
    long_resp = "<think> " + "word " * 300 + "</think> \\boxed{42}"
    r = compute_rewards([long_resp], ["42"], [0.95])[0]
    assert r > 0.5, f"long hard should not be penalised, got {r}"


def test_wrong_answer_negative():
    r = compute_rewards(["<no_think> \\boxed{99}"], ["42"], [0.5])[0]
    assert r < 0, f"wrong answer should give negative reward, got {r}"


def test_missing_routing_token_penalised():
    r_no_token = compute_rewards(["\\boxed{42}"], ["42"], [0.5])[0]
    r_with_token = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.5])[0]
    assert r_no_token < r_with_token


def test_token_counts_override():
    # When actual token counts provided, use them instead of word count
    r_word = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1])[0]
    r_tok  = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1], token_counts=[500])[0]
    assert r_tok < r_word  # more tokens → more penalty
