"""ST1 phase-resampling tests (task 19.5): a wandering duty wave becomes
periodic in angle under the true phase, constant-f round trip is the identity,
the monotonicity guard trips on a bad path, and the three interpolants agree
on a smooth signal."""

from __future__ import annotations

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.st1.pipeline import stage2_known_phase
from powerladder.st1.resample import (angle_resample, angle_sample_times,
                               phase_from_path, random_smooth_phase_path)
from powerladder.typeb.synth import _duty_wave

FS = 20.0
N = 6001  # 300 s at 20 Hz


def _grid():
    return np.arange(N) / FS


def _concentration(x: np.ndarray) -> float:
    """Peak periodogram bin / total power (DC excluded) — line concentration."""
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    X = X[1:]
    return float(X.max() / X.sum())


def test_wandering_duty_wave_periodic_in_angle():
    """With the TRUE phase, angle resampling re-concentrates the smeared line."""
    t = _grid()
    rng = np.random.default_rng(0)
    _, theta = random_smooth_phase_path(t, rng, drift_hz=0.3)
    x = _duty_wave(theta, duty=0.6, n_harmonics=3) \
        + 0.05 * rng.normal(size=t.size)
    _, x_ang = angle_resample(t, x, theta, samples_per_cycle=32, interp="cubic")
    c_time, c_angle = _concentration(x), _concentration(x_ang)
    assert c_angle > 3.0 * c_time, f"time {c_time:.3f} vs angle {c_angle:.3f}"
    assert c_angle > 0.5, f"angle-domain concentration only {c_angle:.3f}"


def test_constant_f_round_trip_is_identity():
    """theta = 2*pi*f0*t: angle samples land at m/(spc*f0) and reproduce x."""
    t = _grid()
    f0 = 1.0
    theta = 2.0 * np.pi * f0 * t
    x = np.sin(2.0 * np.pi * f0 * t + 0.7)
    spc = 32
    theta_grid, t_m = angle_sample_times(t, theta, samples_per_cycle=spc)
    assert np.allclose(t_m, theta_grid / (2.0 * np.pi * f0), atol=1e-9)
    _, x_ang = angle_resample(t, x, theta, samples_per_cycle=spc, interp="cubic")
    assert np.max(np.abs(x_ang - np.sin(theta_grid + 0.7))) < 2e-3


def test_monotonicity_guard_raises():
    t = _grid()
    theta = 2.0 * np.pi * 1.0 * t
    theta[100:200] = theta[100]                    # flat = not strictly increasing
    with pytest.raises(ValueError, match="strictly increasing"):
        angle_resample(t, np.zeros_like(t), theta, samples_per_cycle=32,
                       interp="cubic")


def test_interps_agree_on_smooth_signal():
    t = _grid()
    rng = np.random.default_rng(1)
    _, theta = random_smooth_phase_path(t, rng, drift_hz=0.2)
    x = np.sin(theta) + 0.3 * np.cos(0.5 * theta)
    out = {
        interp: angle_resample(t, x, theta, samples_per_cycle=32,
                               interp=interp)[1]
        for interp in ("linear", "cubic", "sinc")
    }
    for a in out.values():
        assert np.all(np.isfinite(a))
    # interior comparison: the truncated sinc kernel is edge-affected over its
    # half-width, so exclude ~one kernel width at each end.
    sl = slice(32, -32)
    assert np.max(np.abs(out["cubic"][sl] - out["sinc"][sl])) < 2e-2
    assert np.max(np.abs(out["cubic"][sl] - out["linear"][sl])) < 8e-2


def test_phase_from_path_clips_at_floor_and_is_monotone():
    t = _grid()
    frame_times = np.array([0.0, 100.0, 200.0, 300.0])
    path = np.array([1.0, -0.5, 0.8, 1.2])        # dips below zero: must clip
    f_t, theta = phase_from_path(frame_times, path, t, floor_hz=0.05)
    assert f_t.min() >= 0.05
    assert np.all(np.diff(theta) > 0.0)
    # constant extrapolation at the edges
    f_edge, _ = phase_from_path(np.array([100.0, 200.0]), np.array([1.0, 1.0]),
                                t, floor_hz=0.05)
    assert np.allclose(f_edge, 1.0)


def test_stage2_known_phase_scores_signal_over_null():
    """Stage 2 with the true warp separates a wandering duty wave from noise."""
    t = _grid()
    rng = np.random.default_rng(2)
    _, theta = random_smooth_phase_path(t, rng, drift_hz=0.3)
    x_sig = 8.0 * _duty_wave(theta, duty=0.6, n_harmonics=3) \
        + rng.normal(0.0, 4.0, size=t.size)
    x_null = rng.normal(0.0, 4.0, size=t.size)
    p = DEFAULT.st1
    s_sig = stage2_known_phase(t, x_sig, 0.3, 1.7, params=p, theta_true=theta)
    s_null = stage2_known_phase(t, x_null, 0.3, 1.7, params=p, theta_true=theta)
    assert s_sig > 8.0 > s_null, f"sig {s_sig:.2f} vs null {s_null:.2f}"
