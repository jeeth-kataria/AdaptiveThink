"""Unit tests for eval + TTRL pure logic (no model load): pytest tests/test_eval.py"""
from adaptivethink.metrics import pass_at_k, is_correct as _is_correct
from adaptivethink.ttrl.vote import majority_vote_reward


def test_pass_at_k_all_correct():
    assert pass_at_k(n=8, c=8, k=1) == 1.0
    assert pass_at_k(n=8, c=8, k=8) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(n=8, c=0, k=1) == 0.0


def test_pass_at_k_monotonic_in_k():
    # more attempts -> higher chance of at least one correct
    assert pass_at_k(8, 2, 8) >= pass_at_k(8, 2, 1)


def test_pass_at_k_fewer_wrong_than_k():
    # if n-c < k it's guaranteed a hit
    assert pass_at_k(n=4, c=2, k=3) == 1.0


def test_is_correct_boxed():
    assert _is_correct("blah \\boxed{42}", "42")
    assert not _is_correct("blah \\boxed{7}", "42")


def test_majority_vote_reward_rewards_consensus():
    comps = ["\\boxed{42}", "\\boxed{42}", "\\boxed{42}", "\\boxed{7}"]
    r = majority_vote_reward(comps, lambda_tok=0.0)
    # the three agreeing with majority (42) score higher than the dissenter
    assert r[0] > r[3]
    assert r[0] == r[1] == r[2]


def test_majority_vote_confidence_scaling():
    high_conf = majority_vote_reward(["\\boxed{1}"] * 4, lambda_tok=0.0)
    low_conf = majority_vote_reward(
        ["\\boxed{1}", "\\boxed{1}", "\\boxed{2}", "\\boxed{3}"], lambda_tok=0.0)
    # full agreement -> reward 1.0; split -> scaled down
    assert high_conf[0] == 1.0
    assert low_conf[0] < 1.0


def test_majority_vote_no_valid_answers():
    assert majority_vote_reward(["no answer", "nope"]) == [0.0, 0.0]
