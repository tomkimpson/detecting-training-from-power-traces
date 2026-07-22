"""Phase integration and angle-domain resampling (ST1 task 19.5; memo Sec. 3.4).

Tacholess order tracking, applied to power traces: given a per-frame estimate
of the instantaneous iteration frequency f0(t) (from any tracker — the Viterbi
path in stages 3-4, or the TRUE warp in the stage-2 harness), form the phase

    theta(t) = 2 pi * integral_0^t f0(u) du,

strictly increasing because f0 is clipped at a positive floor, then resample
the trace at equal phase increments theta_m = m * 2pi / samples_per_cycle.
In the angle domain a frequency-wandering iteration signature becomes strictly
periodic with cycle frequency 1/samples_per_cycle per sample, so the
downstream DG statistic tests ONE fixed order — no residual frequency search.

The interpolation scheme ("linear" | "cubic" | "sinc") is a flagged open
choice (St1DetectorParams.resample_interp): a data-dependent warp plus an
interpolator can inject phase-locked structure at exactly the tested order.
Whether it does is measured, not assumed — the stage 1->2 delta isolates the
resampling operation alone (known, independent warp), and the interp is a
stage-8 sweep axis.

Also here: :func:`random_smooth_phase_path`, the stage-2 harness's known-warp
generator — an OU frequency wander around the Ko nominal cadence, drawn
INDEPENDENTLY of the null trace (under the null the trace has no line, so the
warp is just a warp; any FAR movement 1->2 is the resampling machinery).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import lfilter

from ..config import DEFAULT

# Windowed-sinc (Lanczos) half-width in samples: kernel support is
# [-_SINC_HALF, +_SINC_HALF] with a sinc(u/_SINC_HALF) taper. 16 taps per side
# is the usual high-quality audio-resampling operating point.
_SINC_HALF = 16

# Stage-2 known-warp defaults: OU wander around the Ko nominal cadence
# (KoWorkloadParams f0 band centre), mean-reversion per CYCLE matching
# KoWorkloadParams.f0_drift_theta, excursion scale drift_hz.
_KO_NOMINAL_F0_HZ = 0.5 * (DEFAULT.ko.f0_lo + DEFAULT.ko.f0_hi)


def phase_from_path(
    frame_times: np.ndarray,
    path_freqs_hz: np.ndarray,
    t: np.ndarray,
    *,
    floor_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a per-frame frequency path into a phase on the sample grid.

    PCHIP-interpolates ``path_freqs_hz`` (at ``frame_times``) onto ``t`` with
    CONSTANT extrapolation beyond the first/last frame (PCHIP is monotone
    shape-preserving — no spline overshoot into unphysical frequencies), clips
    the result at ``floor_hz`` > 0, and returns

        (f_t, theta)   with   theta = 2 pi * cumtrapz(f_t, t)  (theta[0] = 0).

    The clip makes theta STRICTLY increasing, which downstream inversion
    (angle resampling) requires and asserts.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    path = np.asarray(path_freqs_hz, dtype=float)
    t = np.asarray(t, dtype=float)
    if frame_times.size == 0:
        raise ValueError("phase_from_path needs at least one tracked frame")
    if frame_times.size == 1:
        f_t = np.full(t.shape, path[0])
    else:
        pchip = PchipInterpolator(frame_times, path)
        f_t = pchip(np.clip(t, frame_times[0], frame_times[-1]))
    f_t = np.clip(f_t, floor_hz, None)
    theta = 2.0 * np.pi * cumulative_trapezoid(f_t, t, initial=0.0)
    return f_t, theta


def angle_sample_times(
    t: np.ndarray,
    theta: np.ndarray,
    *,
    samples_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform angle grid over the covered phase range and its time preimage.

    theta_m = m * 2pi/samples_per_cycle for every m with theta_m inside
    [theta[0], theta[-1]]; t_m = theta^{-1}(theta_m) by monotone (PCHIP)
    interpolation of t against theta. Raises ValueError if theta is not
    strictly increasing (the guard a bad frequency path must trip loudly —
    a silently folded phase would alias structure into the tested order).
    """
    theta = np.asarray(theta, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("phase path theta must be strictly increasing "
                         "(clip the frequency path at a positive floor)")
    dth = 2.0 * np.pi / samples_per_cycle
    m0 = int(np.ceil(theta[0] / dth - 1e-9))
    m1 = int(np.floor(theta[-1] / dth + 1e-9))
    theta_grid = np.arange(m0, m1 + 1) * dth
    t_m = PchipInterpolator(theta, t)(theta_grid)
    return theta_grid, t_m


def _sinc_resample(t: np.ndarray, x: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Windowed band-limited (Lanczos) interpolation, vectorised.

    Kernel sinc(u) * sinc(u/_SINC_HALF) truncated at |u| = _SINC_HALF sample
    spacings, normalised by the kernel sum per output point (edge-safe).
    Assumes ``t`` uniformly spaced.
    """
    dt = t[1] - t[0]
    u = (np.asarray(t_new, dtype=float) - t[0]) / dt
    k0 = np.floor(u).astype(int)
    offs = np.arange(1 - _SINC_HALF, _SINC_HALF + 1)
    idx = k0[:, None] + offs[None, :]
    frac = u[:, None] - idx
    kern = np.sinc(frac) * np.sinc(frac / _SINC_HALF)
    kern[np.abs(frac) >= _SINC_HALF] = 0.0
    idx = np.clip(idx, 0, x.size - 1)
    return (x[idx] * kern).sum(axis=1) / (kern.sum(axis=1) + 1e-300)


def resample_at(
    t: np.ndarray, x: np.ndarray, t_new: np.ndarray, interp: str
) -> np.ndarray:
    """Sample x(t) at arbitrary times ``t_new`` with the chosen interpolant."""
    x = np.asarray(x, dtype=float)
    if interp == "linear":
        return np.interp(t_new, t, x)
    if interp == "cubic":
        return CubicSpline(t, x)(t_new)
    if interp == "sinc":
        return _sinc_resample(t, x, t_new)
    raise ValueError(f"unknown interp {interp!r} "
                     "(expected 'linear', 'cubic', or 'sinc')")


def angle_resample(
    t: np.ndarray,
    x: np.ndarray,
    theta: np.ndarray,
    *,
    samples_per_cycle: int,
    interp: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample x(t) at equal phase increments of theta(t).

    Returns ``(theta_grid, x_angle)``: theta_grid = m * 2pi/samples_per_cycle
    over the phase range theta covers, and x_angle = x(t(theta_m)) via the
    chosen interpolant. In the angle domain a signal periodic in theta has a
    fixed cycle frequency of 1/samples_per_cycle per (unit-spaced) sample.
    Raises ValueError on a non-monotone theta (see angle_sample_times).
    """
    theta_grid, t_m = angle_sample_times(t, theta,
                                         samples_per_cycle=samples_per_cycle)
    return theta_grid, resample_at(t, x, t_m, interp)


def random_smooth_phase_path(
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    f0_hz: float = _KO_NOMINAL_F0_HZ,
    drift_hz: float = 0.2,
    drift_theta_per_cycle: float = DEFAULT.ko.f0_drift_theta,
    floor_hz: float = DEFAULT.st1.f_path_floor_hz,
) -> tuple[np.ndarray, np.ndarray]:
    """Random smooth frequency path + phase for the stage-2 (known-warp) harness.

    f(t) = f0_hz * g(t) with g an OU walk around 1 — the ko_workload
    f0_drift_hz idiom re-expressed per SAMPLE: the per-cycle mean reversion
    ``drift_theta_per_cycle`` becomes theta_s = theta_cycle * f0 * dt per
    sample, and the innovation std is scaled so the stationary excursion of
    f is ~``drift_hz`` (matching ko_workload._drift_sigma). Clipped at
    ``floor_hz``; default drift 0.2 Hz is the pre-registered wander-regime
    scale (notes/results/st1-findings.md GO-reframe criterion).

    Drawn INDEPENDENTLY of any trace: under the null this is just a warp, so
    stage-2 FAR cells pair (null trace, fresh warp) per replicate and any FAR
    movement relative to stage 1 is attributable to the resampling machinery
    alone. Returns ``(f_t, theta)`` as in :func:`phase_from_path`.
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    th_s = drift_theta_per_cycle * f0_hz * dt
    sig_s = (drift_hz / f0_hz) * np.sqrt(2.0 * th_s - th_s * th_s)
    innov = rng.normal(0.0, sig_s, size=t.size)
    # OU recursion g - 1 <- (1 - th_s)(g - 1) + innov, vectorised as AR(1).
    g = 1.0 + lfilter([1.0], [1.0, -(1.0 - th_s)], innov)
    f_t = np.clip(f0_hz * g, floor_hz, None)
    theta = 2.0 * np.pi * cumulative_trapezoid(f_t, t, initial=0.0)
    return f_t, theta
