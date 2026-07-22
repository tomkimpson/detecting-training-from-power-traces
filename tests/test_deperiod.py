"""De-periodicisation measures tests — ST2 task 20.5.

training_F_meta exposes the generator's ground-truth iteration boundaries
(training_F stays a byte-identical wrapper); code.typeb.deperiod turns them
into the frontier's shared physical x-axis. Analytic anchors: i.i.d.
fractional period jitter sigma at cadence f0 gives CV ~ sigma and
D ~ (2*pi)^2 * sigma^2 * f0 (the windowed variance-growth estimator carries a
known mild downward detrending bias, so magnitude checks are loose and the
scaling checks are the point).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from powerladder.config import DEFAULT
from powerladder.forward import make_time_grid
from powerladder.ko_workload import training_F, training_F_meta
from powerladder.typeb.deperiod import cadence_cv, phase_diffusion_coeff

KO = DEFAULT.ko

REFS = json.loads(
    (Path(__file__).parent / "data" / "st2_prechange_refs.json").read_text()
)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    ).hexdigest()


def _boundaries(sigma, f0, seed, T=1500.0, **knobs):
    p = dataclasses.replace(KO, sigma_jitter=sigma)
    t = make_time_grid(T, 0.05)
    _, iter_starts = training_F_meta(t, p, np.random.default_rng(seed),
                                     f0=f0, eta_scale=0.0, **knobs)
    return iter_starts


def _mean_D(sigma, f0, seeds=range(5)):
    return float(np.mean([
        phase_diffusion_coeff(_boundaries(sigma, f0, s)) for s in seeds
    ]))


# --- training_F_meta wrapper ----------------------------------------------------

def test_training_F_is_byte_identical_wrapper_of_meta():
    """training_F == training_F_meta[0] under the same seed, and both still
    reproduce the pre-refactor reference digest."""
    t = make_time_grid(60.0, 0.05)
    plain = training_F(t, KO, np.random.default_rng(123), f_peak=1.0, f0=1.0)
    meta_F, iter_starts = training_F_meta(t, KO, np.random.default_rng(123),
                                          f_peak=1.0, f0=1.0)
    assert np.array_equal(plain, meta_F)
    assert _digest(plain) == REFS["training_f0_1.0_seed123"]
    # boundaries are the up-phase starts: monotone from 0, ~one per period
    assert iter_starts[0] == 0.0
    assert np.all(np.diff(iter_starts) > 0)
    assert abs(iter_starts.size - 60.0) < 15


# --- cadence CV ------------------------------------------------------------------

def test_cv_zero_for_jitter_free_generator():
    """sigma_jitter = 0 at fixed f0 -> exactly periodic boundaries, CV ~ 0
    (rho only splits the period, never changes it)."""
    assert cadence_cv(_boundaries(0.0, 1.0, 0)) < 1e-9


def test_cv_recovers_known_jitter_sigma():
    """Ko's i.i.d. fractional period jitter sigma is recovered as CV ~ sigma
    (loose: T = 1/(f0(1+xi)) skews the period CV slightly above sigma)."""
    cv = cadence_cv(_boundaries(0.2, 1.0, 1))
    assert 0.15 < cv < 0.3
    cv_small = cadence_cv(_boundaries(0.05, 1.0, 1))
    assert 0.035 < cv_small < 0.07


# --- phase diffusion coefficient --------------------------------------------------

def test_phase_diffusion_scales_as_sigma2_f0():
    """D ~ (2*pi)^2 * sigma^2 * f0: right magnitude (loose), right scaling in
    sigma (quadratic) and f0 (linear)."""
    D_ref = _mean_D(0.1, 1.0)
    pred = (2.0 * np.pi) ** 2 * 0.1 ** 2 * 1.0
    assert 0.3 * pred < D_ref < 1.5 * pred
    # quadratic in sigma
    assert 2.0 < _mean_D(0.2, 1.0) / D_ref < 8.0
    # linear in f0
    assert 1.3 < _mean_D(0.1, 2.0) / D_ref < 3.0


def test_phase_slip_registers_as_diffusion():
    """The task-20.4 phase-slip knob lands on the same D axis: slip sigma at
    fixed mean period diffuses like period jitter of the same sigma."""
    D = float(np.mean([
        phase_diffusion_coeff(_boundaries(0.0, 1.0, s, phase_slip_sigma=0.05))
        for s in range(5)
    ]))
    pred = (2.0 * np.pi) ** 2 * 0.05 ** 2 * 1.0
    assert 0.3 * pred < D < 1.5 * pred


def test_measures_deterministic_under_seed():
    """Pure functions of the seeded generator output: identical seed ->
    identical boundaries, CV and D."""
    a = _boundaries(0.1, 1.0, 7, T=300.0)
    b = _boundaries(0.1, 1.0, 7, T=300.0)
    assert np.array_equal(a, b)
    assert cadence_cv(a) == cadence_cv(b)
    assert phase_diffusion_coeff(a) == phase_diffusion_coeff(b)


def test_short_series_raise():
    """Too few boundaries for a meaningful measure is an error, not a NaN."""
    with pytest.raises(ValueError):
        cadence_cv(np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        phase_diffusion_coeff(np.arange(10.0))
