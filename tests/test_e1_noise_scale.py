"""E1 pilot (exploratory): the McCandlish B_simple estimator recovers a known
noise scale on synthetic gradients."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "e1"))
import noise_scale  # noqa: E402  (numpy-only, safe in the CPU suite)


def test_b_simple_recovers_known_noise_scale():
    rng = np.random.default_rng(0)
    d, b_small, k = 200, 4, 8            # dim, examples per micro-batch, micro-batches
    # true gradient sized so B_noise (~14) sits between b_small and b_big, the
    # regime the pilot runs in (honest batch ~ B_noise)
    g = rng.normal(0.0, 1.0, d) * 0.3
    sigma = rng.uniform(0.5, 1.5, d)     # per-example noise std per coordinate
    true_b = float(np.sum(sigma**2) / np.sum(g**2))

    g2s, ss = [], []
    for _ in range(4000):
        micro = g + rng.normal(0.0, 1.0, (k, d)) * sigma / np.sqrt(b_small)
        small_sq = float(np.mean(np.sum(micro**2, axis=1)))
        big_sq = float(np.sum(micro.mean(axis=0) ** 2))
        g2, s = noise_scale.noise_scale_terms(small_sq, big_sq, b_small, k * b_small)
        g2s.append(g2)
        ss.append(s)
    est = noise_scale.b_simple(g2s, ss)
    assert abs(est / true_b - 1.0) < 0.05


def test_terms_reject_equal_batch_sizes():
    with pytest.raises(ValueError):
        noise_scale.noise_scale_terms(1.0, 1.0, 4, 4)
