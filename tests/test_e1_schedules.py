"""E1 pilot (exploratory): micro-step schedules match the E0 spike's distributions.

See notes/results/e0-work-variation-spike-findings.md for the reference numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "e1"))
import schedules  # noqa: E402  (numpy-only, safe in the CPU suite)

N = 400_000


def test_honest_is_constant_eight():
    G = schedules.make_schedule("honest", N, seed=0)
    assert G.dtype.kind == "i"
    assert np.all(G == 8)


def test_shipped_matches_e0_statistics():
    G = schedules.make_schedule("shipped0.7", N, seed=0)
    assert G.min() >= 1 and G.max() == 160
    assert abs(G.mean() - 25.0) < 0.3
    assert np.median(G) == 8
    assert abs((G == 160).mean() - 0.087) < 0.003


@pytest.mark.parametrize("arm,cv", [("lognorm0.8", 0.94), ("lognorm1.0", 1.27)])
def test_lognormal_keeps_mean_eight(arm, cv):
    G = schedules.make_schedule(arm, N, seed=0)
    assert G.min() >= 1 and G.max() <= 160
    assert abs(G.mean() - 8.0) < 0.1
    assert abs(G.std() / G.mean() - cv) < 0.03


def test_sqrt_lr_arm_shares_lognormal_schedule():
    a = schedules.make_schedule("lognorm1.0", 1000, seed=3)
    b = schedules.make_schedule("lognorm1.0_sqrtlr", 1000, seed=3)
    assert np.array_equal(a, b)


def test_lr_multiplier():
    G = np.array([2, 8, 32])
    assert np.allclose(schedules.lr_multiplier("lognorm1.0", G), 1.0)
    assert np.allclose(schedules.lr_multiplier("lognorm1.0_sqrtlr", G),
                       [0.5, 1.0, 2.0])


def test_schedule_is_seeded():
    a = schedules.make_schedule("lognorm1.0", 1000, seed=1)
    b = schedules.make_schedule("lognorm1.0", 1000, seed=1)
    c = schedules.make_schedule("lognorm1.0", 1000, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_stochastic_rounding_preserves_mean():
    rng = np.random.default_rng(0)
    g = schedules.stochastic_round(np.full(N, 3.25), rng)
    assert set(np.unique(g)) == {3, 4}
    assert abs(g.mean() - 3.25) < 0.01
    assert schedules.stochastic_round(np.array([0.2]), rng)[0] == 1
