"""E1 pilot (exploratory): tokens-to-target interpolation and the predicted cost."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "e1"))
import analyse  # noqa: E402  (numpy-only, safe in the CPU suite)


def test_tokens_to_target_interpolates_first_crossing():
    tokens = np.array([0, 10, 20, 30])
    loss = np.array([5.0, 4.0, 3.0, 3.5])          # bounces back up after 20
    assert np.isclose(analyse.tokens_to_target(tokens, loss, 3.5), 15.0)
    assert np.isclose(analyse.tokens_to_target(tokens, loss, 3.0), 20.0)


def test_tokens_to_target_uses_running_minimum():
    tokens = np.array([0, 10, 20])
    loss = np.array([5.0, 3.0, 4.0])
    # a later rise must not move the crossing
    assert np.isclose(analyse.tokens_to_target(tokens, loss, 4.0), 5.0)


def test_tokens_to_target_never_reached_is_none():
    assert analyse.tokens_to_target(np.array([0, 10]), np.array([5.0, 4.0]), 3.0) is None


def test_loss_at_interpolates():
    assert np.isclose(analyse.loss_at(np.array([0, 10]), np.array([4.0, 2.0]), 7.5), 2.5)


def test_predicted_cost_zero_for_honest_and_positive_with_spread():
    assert abs(analyse.predicted_cost("honest", b_noise_micro=4.0)) < 1e-12
    assert analyse.predicted_cost("lognorm1.0", b_noise_micro=4.0) > 0.1
