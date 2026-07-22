"""ST1 null suite: the traces the structural detectors must NOT fire on.

Graded difficulty, mirroring the staged ST1 design (plan-for-paper-2 Sec. 6):

    stage 5 (stationary Gaussian / coloured):  white, ar1, ar2_resonant
    stage 6 (stationary non-Gaussian):         ar1_t, lognormal, sq_gauss
    stage 7 (locally stationary / controller): tvar, am_walk, mean_drift,
                                               controller, controller_ar1

Every generator has the signature ``fn(n, fs, p, rng) -> TypeTrace`` with
``p: St1NullParams`` and emits ``P_base + fluctuation`` on the standard
300 s / 20 Hz grid unless (n, fs) say otherwise; label ``"null"``, ``f0 = nan``.
The registry :data:`NULLS` maps name -> generator; :data:`STAGE5` /
:data:`STAGE6` / :data:`STAGE7` group the names by stage.

INTERPRETATION CONTRACT (pre-registered; plan Sec. 6 ST1 stage 7): the DG /
order statistics' asymptotic chi-squared null assumes weak-dependence
stationarity. The stage-7 nulls (tvar, am_walk, mean_drift, controller,
controller_ar1) deliberately violate that assumption — realised-FAR inflation
on them is *information*, mapping the causal-attribution boundary of the
structural null (paper Sec. 1.4), NOT a defect of the implementation and NOT
a gate failure. Only stage-5/6 miscalibration counts against GO
(notes/results/st1-findings.md, pre-registered criteria).

The controller null mimics the measured A100 power-management limit cycle
(0.31-0.45 Hz, ~10x the iteration-line amplitude; B2Params.band_line_lo
provenance) as a noisy Van der Pol relaxation oscillation with the cycle
frequency re-jittered every cycle — nearly single-tone, non-sinusoidal,
non-stationary in phase: the hardest realistic "periodic but not training"
confuser we have measured.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import lfilter

from ..config import St1NullParams
from ..typeb.synth import TypeTrace

# Burn-in samples dropped from every recursive-filter null so the trace starts
# in the stationary regime (>> the longest AR memory 1/(1-phi) ~ 20 samples).
_BURN = 500

# mean_drift shape choices (documented, not magic): total ramp amplitude and
# per-step amplitude in units of p.sigma, and the number of level steps.
_DRIFT_RAMP_SIGMA = 2.0
_DRIFT_STEP_SIGMA = 1.0
_DRIFT_N_STEPS = 3

# controller: white measurement noise on top of the limit cycle, in units of
# p.sigma ("small" relative to ctrl_amp = 5x sigma), and the max RK4 substep
# in Van der Pol natural time (mu=4 relaxation is moderately stiff).
_CTRL_NOISE_FRAC = 1.0
_CTRL_MAX_DTAU = 0.05


def _grid(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs


def _trace(t: np.ndarray, fluct: np.ndarray, p: St1NullParams) -> TypeTrace:
    return TypeTrace(t=t, P_obs=p.P_base + fluct, label="null", f0=float("nan"))


def _ar1_series(n: int, phi: float, innov: np.ndarray) -> np.ndarray:
    """AR(1) with the given innovations (length n + _BURN), burn-in dropped."""
    x = lfilter([1.0], [1.0, -phi], innov)
    return x[_BURN:_BURN + n]


# --- stage 5: stationary Gaussian / coloured --------------------------------

def white(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """White Gaussian fluctuation, std sigma — the textbook null."""
    return _trace(_grid(n, fs), rng.normal(0.0, p.sigma, size=n), p)


def ar1(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """AR(1), phi = ar1_phi, stationary std sigma (innovations scaled by sqrt(1-phi^2))."""
    innov = rng.normal(0.0, p.sigma * np.sqrt(1.0 - p.ar1_phi**2), size=n + _BURN)
    return _trace(_grid(n, fs), _ar1_series(n, p.ar1_phi, innov), p)


def ar2_resonant(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """AR(2) with a spectral peak at ar2_peak_hz INSIDE the search band.

    Complex pole pair at radius r and angle w0 = 2*pi*f0/fs; the resonance
    half-power bandwidth is ~ f0/Q, giving r = exp(-pi*f0/(Q*fs)). Output is
    rescaled to std sigma (the peak location, not the scale, is the point).
    """
    w0 = 2.0 * np.pi * p.ar2_peak_hz / fs
    r = np.exp(-np.pi * p.ar2_peak_hz / (p.ar2_q * fs))
    innov = rng.normal(0.0, 1.0, size=n + _BURN)
    x = lfilter([1.0], [1.0, -2.0 * r * np.cos(w0), r**2], innov)[_BURN:_BURN + n]
    x = x / (x.std() + 1e-12) * p.sigma
    return _trace(_grid(n, fs), x, p)


# --- stage 6: stationary non-Gaussian ----------------------------------------

def ar1_t(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """AR(1) with Student-t(t_dof) innovations, variance-normalised.

    Innovations are divided by sqrt(dof/(dof-2)) so they have unit variance,
    then scaled as in :func:`ar1`; the heavy tails survive the normalisation.
    """
    innov = rng.standard_t(p.t_dof, size=n + _BURN)
    innov = innov / np.sqrt(p.t_dof / (p.t_dof - 2.0))
    innov = innov * p.sigma * np.sqrt(1.0 - p.ar1_phi**2)
    return _trace(_grid(n, fs), _ar1_series(n, p.ar1_phi, innov), p)


def lognormal(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """Exponentiated Gaussian AR(1), recentred and rescaled to std sigma.

    Stationary, coloured, positively skewed marginal — the asymmetric-tail
    stress on the Gaussian-ish moment assumptions.
    """
    innov = rng.normal(0.0, np.sqrt(1.0 - p.ar1_phi**2), size=n + _BURN)
    g = _ar1_series(n, p.ar1_phi, innov)          # unit-variance Gaussian AR(1)
    y = np.exp(g)
    y = (y - y.mean()) / (y.std() + 1e-12) * p.sigma
    return _trace(_grid(n, fs), y, p)


def sq_gauss(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """Squared Gaussian AR(1) (chi^2-marginal), recentred and rescaled to std sigma."""
    innov = rng.normal(0.0, np.sqrt(1.0 - p.ar1_phi**2), size=n + _BURN)
    g = _ar1_series(n, p.ar1_phi, innov)
    y = g**2
    y = (y - y.mean()) / (y.std() + 1e-12) * p.sigma
    return _trace(_grid(n, fs), y, p)


# --- stage 7: locally stationary / controller-like ---------------------------

def tvar(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """Time-varying AR(1): phi ramps linearly tvar_phi_lo -> tvar_phi_hi.

    Innovation std is held at sigma, so the local stationary variance
    sigma^2/(1-phi^2) GROWS along the trace — a locally-stationary null whose
    long-run covariance is not constant (the weak-dependence assumption's
    first casualty).
    """
    phi_t = np.linspace(p.tvar_phi_lo, p.tvar_phi_hi, n)
    innov = rng.normal(0.0, p.sigma, size=n)
    x = np.empty(n)
    prev = 0.0
    for i in range(n):
        prev = phi_t[i] * prev + innov[i]
        x[i] = prev
    return _trace(_grid(n, fs), x, p)


def _bounded_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Slow zero-mean random walk, demeaned / std-normalised / clipped to [-1, 1].

    The _wandering_phase idiom from code.typeb.synth: a cumsum of white steps,
    recentred and scale-normalised so the excursion is O(1) regardless of n,
    then clipped so multiplicative uses stay bounded.
    """
    walk = np.cumsum(rng.normal(0.0, 1.0, size=n))
    walk = walk - walk.mean()
    walk = walk / (walk.std() + 1e-12)
    return np.clip(walk, -1.0, 1.0)


def am_walk(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """Coloured (AR(1)) noise with sigma(t) a slow bounded random walk.

    sigma(t) = sigma * (1 + am_walk_frac * walk(t)) with walk in [-1, 1], so the
    amplitude wanders within [1-frac, 1+frac] x sigma. Amplitude modulation is
    exactly the kind of second-order structure the DG statistic keys on — but
    here it is aperiodic, so a calibrated detector must not fire.
    """
    innov = rng.normal(0.0, np.sqrt(1.0 - p.ar1_phi**2), size=n + _BURN)
    g = _ar1_series(n, p.ar1_phi, innov)          # unit-variance coloured noise
    sig_t = p.sigma * (1.0 + p.am_walk_frac * _bounded_walk(n, rng))
    return _trace(_grid(n, fs), sig_t * g, p)


def mean_drift(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """White noise plus a slow mean ramp and a few level steps.

    Ramp spans +-_DRIFT_RAMP_SIGMA x sigma over the window (random sign);
    _DRIFT_N_STEPS level shifts of +-_DRIFT_STEP_SIGMA x sigma land at uniform
    random times. First-order nonstationarity only — no periodic structure.
    """
    t = _grid(n, fs)
    x = rng.normal(0.0, p.sigma, size=n)
    ramp_amp = _DRIFT_RAMP_SIGMA * p.sigma * rng.choice([-1.0, 1.0])
    x = x + ramp_amp * np.linspace(-1.0, 1.0, n)
    for _ in range(_DRIFT_N_STEPS):
        i0 = rng.integers(1, n)
        x[i0:] += _DRIFT_STEP_SIGMA * p.sigma * rng.choice([-1.0, 1.0])
    return _trace(t, x, p)


@lru_cache(maxsize=8)
def _vdp_natural_period(mu: float) -> float:
    """Natural period of the Van der Pol oscillator x'' - mu(1-x^2)x' + x = 0.

    Measured once per mu by RK4-integrating in natural time and averaging the
    spacing of upward zero crossings after the transient (cached; deterministic).
    """
    dt = 0.001
    state = np.array([2.0, 0.0])

    def deriv(s):
        return np.array([s[1], mu * (1.0 - s[0] ** 2) * s[1] - s[0]])

    crossings = []
    prev_x = state[0]
    t_now = 0.0
    while len(crossings) < 12 and t_now < 500.0:
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_now += dt
        if prev_x < 0.0 <= state[0]:
            crossings.append(t_now)
        prev_x = state[0]
    periods = np.diff(crossings[2:])              # drop the transient
    return float(np.mean(periods))


def _controller_cycle(
    n: int, fs: float, p: St1NullParams, rng: np.random.Generator
) -> np.ndarray:
    """Noisy relaxation limit cycle: Van der Pol, frequency re-jittered per cycle.

    The oscillator runs in its natural (dimensionless) time tau; each output
    sample advances tau at rate T_mu * f_cyc per second so the fundamental
    lands at f_cyc, and f_cyc is re-drawn ~ U(ctrl_f_lo, ctrl_f_hi) at every
    upward zero crossing (per-cycle jitter, as the measured limit cycle shows).
    RK4 with substeps capped at _CTRL_MAX_DTAU. Van der Pol's limit-cycle
    amplitude is ~2, so the output is scaled by ctrl_amp/2.

    Implementation note (2026-07-21, task 19.9): the RK4 inner loop runs on
    scalar floats rather than 2-vectors — bit-identical output (same IEEE
    operation order, same rng draw sequence; verified against the original
    array form on multiple seeds) at ~16x the speed, which is what makes the
    M = 10^4 controller FAR cells tractable.
    """
    T_mu = _vdp_natural_period(p.ctrl_mu)
    mu = p.ctrl_mu

    x, v = 2.0, 0.0
    f_cyc = rng.uniform(p.ctrl_f_lo, p.ctrl_f_hi)
    out = np.empty(n)
    for i in range(n):
        out[i] = x
        dtau = T_mu * f_cyc / fs                  # natural time per output sample
        n_sub = max(1, int(np.ceil(dtau / _CTRL_MAX_DTAU)))
        h = dtau / n_sub
        for _ in range(n_sub):
            prev_x = x
            k1x = v
            k1v = mu * (1.0 - x ** 2) * v - x
            x2 = x + 0.5 * h * k1x
            v2 = v + 0.5 * h * k1v
            k2x = v2
            k2v = mu * (1.0 - x2 ** 2) * v2 - x2
            x3 = x + 0.5 * h * k2x
            v3 = v + 0.5 * h * k2v
            k3x = v3
            k3v = mu * (1.0 - x3 ** 2) * v3 - x3
            x4 = x + h * k3x
            v4 = v + h * k3v
            k4x = v4
            k4v = mu * (1.0 - x4 ** 2) * v4 - x4
            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            if prev_x < 0.0 <= x:                 # new cycle -> re-jitter f_cyc
                f_cyc = rng.uniform(p.ctrl_f_lo, p.ctrl_f_hi)
    return (p.ctrl_amp / 2.0) * out


def controller(n: int, fs: float, p: St1NullParams, rng: np.random.Generator) -> TypeTrace:
    """Power-management-style limit cycle + white noise (the measured confuser).

    Nearly single-tone at ~0.31-0.45 Hz, strongly non-sinusoidal (relaxation),
    ~5x sigma in amplitude — periodic structure that is NOT training. Nominal
    validity of the stationary-null asymptotics MAY fail here by design (see
    module docstring: that is information, not a defect).
    """
    fluct = _controller_cycle(n, fs, p, rng) + rng.normal(
        0.0, _CTRL_NOISE_FRAC * p.sigma, size=n
    )
    return _trace(_grid(n, fs), fluct, p)


def controller_ar1(
    n: int, fs: float, p: St1NullParams, rng: np.random.Generator
) -> TypeTrace:
    """Controller limit cycle riding on AR(1) coloured noise (limit cycle + colour)."""
    innov = rng.normal(0.0, p.sigma * np.sqrt(1.0 - p.ar1_phi**2), size=n + _BURN)
    fluct = _controller_cycle(n, fs, p, rng) + _ar1_series(n, p.ar1_phi, innov)
    return _trace(_grid(n, fs), fluct, p)


# --- registry -----------------------------------------------------------------

NULLS: dict = {
    "white": white,
    "ar1": ar1,
    "ar2_resonant": ar2_resonant,
    "ar1_t": ar1_t,
    "lognormal": lognormal,
    "sq_gauss": sq_gauss,
    "tvar": tvar,
    "am_walk": am_walk,
    "mean_drift": mean_drift,
    "controller": controller,
    "controller_ar1": controller_ar1,
}

STAGE5: tuple[str, ...] = ("white", "ar1", "ar2_resonant")
STAGE6: tuple[str, ...] = ("ar1_t", "lognormal", "sq_gauss")
STAGE7: tuple[str, ...] = ("tvar", "am_walk", "mean_drift", "controller",
                           "controller_ar1")
