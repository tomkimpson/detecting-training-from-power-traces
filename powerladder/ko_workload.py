"""Ko et al. (arXiv 2508.16457) stochastic workload generator for F(t).

Phase 2, task 1.1 (notes/development-notes/phase2-plan.md Sec. 2.1). Ko et al.
model AI-datacentre *power* as a piecewise-periodic process with realistic
inter-/intra-iteration stochasticity. We adopt the same waveform to drive the
latent **FLOP rate F(t)** the time-resolved tracker estimates: the compute/up
phase carries high F, the comm/down (or idle) phase low F. This is shared
infrastructure -- the B1 detector (tasks 3-4) reuses the same generator.

The model (parameters in code.config.KoWorkloadParams, == Ko Table I):

    Training (eqs 1-5):
        iteration period   T_i  = 1 / (f0 (1 + xi_i)),   xi_i ~ N(0, sigma_jitter^2)
        phase split        T_up = rho_i T_i,   T_down = (1 - rho_i) T_i
        up   level         1 + Delta_up   + eta_up(t),   Delta_up   ~ N(0, sigma_d^2)
        down level         1 - Delta_down + eta_down(t), Delta_down ~ N(mu_d, sigma_d^2)
    Fine-tuning (eqs 6-10): identical shape with f1, tail/idle phases, delta/mu_delta.
    Aggregate (eq 11): P^AI = P0_tr + sum_j P_j^tr + sum_k P_k^ft at the 9:0.5:0.5
        dominance ratio (the B1-agg positive class, task 4; ``aggregate_null_F``
        swaps the dominant slot for the inference null -- our construction, not Ko's).

NAMING: Ko's compute/comm phase ratio is "r"; we reserve r for the exchange rate
[J/FLOP], so the phase ratio is ``rho`` throughout.

RNG is always an injected ``np.random.Generator`` (repo convention; no globals).
``f_peak`` sets the absolute scale (FLOP/s); callers pass FloorParams.F_max times
a headroom fraction. With ``f0=None`` the characteristic frequency is drawn once
per trace from the configured band; pass an explicit ``f0`` for reproducible
periods (tests) or to relate an adversary's bandwidth to the line (task 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import KoWorkloadParams

# Smallest admissible jitter factor (1 + xi); guards the 1/(f0(1+xi)) period
# against a pathological deep-negative xi draw (sigma_jitter is small, so rare).
_MIN_JITTER_FACTOR = 0.05

# Floor on the drifting frequency factor g; f0_eff = f0*g never falls below this
# fraction of f0 (defends the period formula when the OU walk wanders low).
_MIN_DRIFT_FACTOR = 0.1


@dataclass(frozen=True)
class TrainingPhaseMetadata:
    """Evaluation-only boundaries for complete synthetic training iterations.

    The arrays describe the latent schedule before the observation map is
    applied.  Detector functions must never consume this object; it exists to
    measure whether local compute-to-communication events survive that map and
    to diagnose a blind event detector against generator ground truth.
    """

    compute_starts: np.ndarray
    communication_starts: np.ndarray
    compute_durations: np.ndarray
    communication_durations: np.ndarray

    @property
    def iteration_starts(self) -> np.ndarray:
        """Start time of every complete iteration."""
        return self.compute_starts


def _drift_sigma(f0_drift_hz: float, f0: float, theta: float) -> float:
    """Per-iteration OU innovation std giving an f0 excursion of ~f0_drift_hz.

    The drift factor g follows g <- g + theta*(1 - g) + N(0, sigma_g). Its
    stationary std is sigma_g / sqrt(2*theta - theta^2); we want that excursion,
    expressed as a fraction of f0, to equal f0_drift_hz / f0. Hence
    sigma_g = (f0_drift_hz / f0) * sqrt(2*theta - theta^2). Matches the
    ``wander_hz`` semantics of code.typeb.synth._wandering_phase (excursion scaled
    to the configured Hz) but re-expressed in the period-accumulation domain.
    """
    return (f0_drift_hz / f0) * np.sqrt(2.0 * theta - theta * theta)


def _periodic_F_meta(
    t: np.ndarray,
    f_peak: float,
    freq_lo: float,
    freq_hi: float,
    sigma_jitter: float,
    rho_lo: float,
    rho_hi: float,
    sigma_delta: float,
    mu_delta_down: float,
    eta_up_range: tuple[float, float],
    eta_down_range: tuple[float, float],
    rng: np.random.Generator,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
    f0_drift_theta: float = 0.02,
    work_sigma: float = 0.0,
    work_drift_hz: float = 0.0,
    work_shift_hz: float | None = None,
    work_base_accum: int = 8,
    phase_slip_sigma: float = 0.0,
    harmonic_smooth_s: float = 0.0,
    shape_fill_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise up/down (or tail/idle) FLOP-rate trace on grid ``t``, plus the
    phase-boundary times.

    Shared engine for :func:`training_F` and :func:`finetune_F` -- the two
    workloads differ only in their parameter set, not their structure. Returns
    ``(F, phase_starts)`` where ``phase_starts`` alternates up-phase and
    down-phase start times, so ``phase_starts[::2]`` are the iteration
    boundaries (generator ground truth for the ST2 de-periodicisation
    measures; :func:`training_F` discards it).

    ``eta_scale`` rescales Ko's intra-phase (sub-millisecond) noise. Ko's eta is a
    sub-ms power fluctuation; a meter sampling at tens of Hz averages it well below
    a sample, so a time-resolved tracker should down-weight it (eta_scale < 1) and
    represent the residual through the meter-noise channel instead. Left at 1.0 it
    reproduces Ko faithfully (e.g. for a sub-ms B1 detector).

    ``f0_drift_hz`` adds a slow OU random walk of the centre frequency on top of
    Ko's i.i.d. ``xi`` (B1 / task 3.2; config docstring). 0.0 (default) draws NO
    extra RNG and leaves the trace byte-identical to the no-drift generator -- the
    tracker scenarios (code.scenarios) and tests rely on this.

    Work variation (ST2 / task 20.3 -- the measured ~zero-cost attack): the
    knobs mirror :func:`code.b2.workloads.work_jitter_schedule` so synthetic
    levels map 1:1 onto the measured b2 anchors. Any of ``work_sigma > 0``,
    ``work_drift_hz > 0`` or ``work_shift_hz is not None`` quantises the UP
    phase into an integer number of real micro-steps: with
    ``t_micro = rho * T_base / work_base_accum`` (T_base = 1/f0_eff), each
    iteration draws a real-valued target around ``work_base_accum`` -- divided
    by ``(1 + xi_w)`` with ``xi_w ~ N(0, work_sigma)``, by an OU drift factor
    with excursion ``work_drift_hz`` (same innovation maths as ``f0_drift_hz``,
    sharing ``f0_drift_theta``), or retargeted so the MEAN cadence lands at
    ``work_shift_hz`` -- then stochastically rounds to ``G_i >= 1`` and sets
    ``T_up_i = G_i * t_micro``. ``T_down`` keeps its honest draw untouched (the
    comm/optimizer phase does not scale with accumulation count), so the duty
    ratio co-moves with ``G_i`` -- the physics of the zero-cost attack. NOTE
    one deliberate deviation from work_jitter_schedule's shift formula: there
    the whole period is G*t_micro, here the fixed T_down is subtracted first
    (``g_real = (1/work_shift_hz - T_down)/t_micro``) so the retargeted TOTAL
    cadence, not just the up phase, lands at work_shift_hz. All-defaults draws
    ZERO extra RNG (byte-identical guard in tests/test_ko_workload.py).

    De-periodicisation knobs (ST2 / task 20.4; all default no-op, zero RNG):

    ``phase_slip_sigma`` -- after each iteration the running clock slips by
    ``N(0, phase_slip_sigma * T_i)`` WITHOUT changing the target period:
    Brownian boundary diffusion at fixed mean cadence (distinct from Ko's
    ``sigma_jitter``, whose i.i.d. periods also random-walk the boundaries but
    tie the diffusion to the period variance; the slip decouples them). The
    clock is clamped so phase boundaries stay monotone (a deep negative slip
    cannot re-enter the previous phase).

    ``harmonic_smooth_s`` -- the piecewise level waveform is convolved with a
    raised-cosine (Hann) kernel of that total width AFTER the levels are laid
    onto the grid and BEFORE intra-phase noise: phase transitions ramp instead
    of step, so the harmonic comb dies while the fundamental survives (kernel
    edge-normalised; the attack's budget is the analytic energy delta).

    ``shape_fill_frac`` (phi, mirrors the measured b2 shaped-idle sweep) --
    lifts each down level toward its iteration's up level:
    ``level_down' = level_down + phi * (level_up - level_down)``; phi = 1
    collapses the up/down swing entirely (DC-matched shaping).
    """
    if f0 is None:
        f0 = float(rng.uniform(freq_lo, freq_hi))
    t_total = float(t[-1])

    # OU drift of the instantaneous f0: f0_eff = f0*g, g a slow mean-reverting walk
    # (only when drift is requested -- otherwise the RNG stream is untouched).
    drift = f0_drift_hz > 0.0
    sigma_g = _drift_sigma(f0_drift_hz, f0, f0_drift_theta) if drift else 0.0
    g = 1.0

    # Work variation: OU drift factor of the micro-step target (mirrors
    # work_jitter_schedule's g; only touched when the knobs are on).
    work = work_sigma > 0.0 or work_drift_hz > 0.0 or work_shift_hz is not None
    sigma_gw = _drift_sigma(work_drift_hz, f0, f0_drift_theta) \
        if work_drift_hz > 0.0 else 0.0
    g_w = 1.0

    # Lay down phases by accumulating iteration periods until the grid is covered.
    starts: list[float] = []
    levels: list[float] = []
    eta_sig: list[float] = []
    clock = 0.0
    while clock <= t_total:
        xi = rng.normal(0.0, sigma_jitter)
        if drift:
            g += f0_drift_theta * (1.0 - g) + rng.normal(0.0, sigma_g)
            f0_eff = max(f0 * g, f0 * _MIN_DRIFT_FACTOR)
        else:
            f0_eff = f0
        period = 1.0 / (f0_eff * max(1.0 + xi, _MIN_JITTER_FACTOR))
        rho = rng.uniform(rho_lo, rho_hi)
        t_up = rho * period
        t_down = period - t_up
        if work:
            # micro-step grid: the honest up phase is work_base_accum steps of
            # t_micro each (t_micro = rho * T_base / base_accum, T_base = 1/f0_eff)
            t_micro = rho / (f0_eff * work_base_accum)
            if work_shift_hz is not None:
                # retarget the MEAN total cadence to work_shift_hz: the fixed
                # T_down is subtracted before quantising the up phase.
                g_real = (1.0 / work_shift_hz - t_down) / t_micro
            else:
                xi_w = rng.normal(0.0, work_sigma) if work_sigma > 0.0 else 0.0
                if work_drift_hz > 0.0:
                    g_w += f0_drift_theta * (1.0 - g_w) + rng.normal(0.0, sigma_gw)
                    g_w = max(g_w, _MIN_DRIFT_FACTOR)
                g_real = work_base_accum / (g_w * max(1.0 + xi_w, _MIN_JITTER_FACTOR))
            # stochastic rounding to an integer micro-step count, floor 1
            # (== work_jitter_schedule._round_stoch)
            g_lo = np.floor(g_real)
            g_i = max(int(g_lo) + (1 if rng.random() < g_real - g_lo else 0), 1)
            t_up = g_i * t_micro
        # up / compute / tail phase: level ~ 1 + N(0, sigma_delta^2)
        level_up = 1.0 + rng.normal(0.0, sigma_delta)
        starts.append(clock)
        levels.append(level_up)
        eta_sig.append(rng.uniform(*eta_up_range))
        clock += t_up
        # down / comm / idle phase: level ~ 1 - N(mu_delta_down, sigma_delta^2)
        level_down = 1.0 - rng.normal(mu_delta_down, sigma_delta)
        if shape_fill_frac != 0.0:
            # amplitude shaping: lift the dip toward the up level (b2 phi)
            level_down += shape_fill_frac * (level_up - level_down)
        starts.append(clock)
        levels.append(level_down)
        eta_sig.append(rng.uniform(*eta_down_range))
        clock += t_down
        if phase_slip_sigma > 0.0:
            # Brownian boundary slip at fixed target period; clamp keeps the
            # boundary sequence monotone for searchsorted below.
            slip = rng.normal(0.0, phase_slip_sigma * period)
            clock = max(clock + slip, starts[-1])

    starts_a = np.asarray(starts)
    levels_a = np.asarray(levels)
    eta_sig_a = np.asarray(eta_sig)

    # Assign each sample to its phase, add per-sample sub-step Gaussian noise.
    idx = np.clip(np.searchsorted(starts_a, t, side="right") - 1, 0, levels_a.size - 1)
    base = levels_a[idx]
    if harmonic_smooth_s > 0.0:
        # raised-cosine smoothing of the level waveform (before intra-phase
        # noise): transitions ramp over ~harmonic_smooth_s, killing the comb.
        dt_grid = float(t[1] - t[0]) if t.size > 1 else 1.0
        n_k = int(round(harmonic_smooth_s / dt_grid))
        if n_k >= 2:
            kernel = np.hanning(n_k + 2)[1:-1]      # strictly positive interior
            kernel = kernel / kernel.sum()
            # edge-normalised 'same' convolution so partial overlap at the
            # trace ends does not sag the levels
            base = np.convolve(base, kernel, mode="same") \
                / np.convolve(np.ones_like(base), kernel, mode="same")
    eta = rng.normal(0.0, 1.0, size=t.shape) * (eta_scale * eta_sig_a[idx])
    F = f_peak * (base + eta)
    return np.clip(F, 0.0, None), starts_a


def _periodic_F(*args, **kwargs) -> np.ndarray:
    """:func:`_periodic_F_meta` without the phase-boundary metadata."""
    return _periodic_F_meta(*args, **kwargs)[0]


def training_F_meta(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
    work_sigma: float = 0.0,
    work_drift_hz: float = 0.0,
    work_shift_hz: float | None = None,
    phase_slip_sigma: float = 0.0,
    harmonic_smooth_s: float = 0.0,
    shape_fill_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Training FLOP-rate trace F(t) [FLOP/s] (Ko eqs 1-5), plus the iteration
    boundary times.

    Returns ``(F, iter_starts)``: ``iter_starts`` are the up-phase start times
    the generator laid down — GROUND-TRUTH iteration boundaries for the ST2
    de-periodicisation measures (:mod:`code.typeb.deperiod`), free of any
    estimation step. :func:`training_F` is this function without the metadata
    (byte-identical wrapper).

    ``f0_drift_hz`` (B1 / task 3.2) adds a slow OU centre-frequency wander on top
    of Ko's i.i.d. jitter; 0.0 (default) is byte-identical to the Ko-faithful trace.

    ``work_sigma`` / ``work_drift_hz`` / ``work_shift_hz`` (ST2 / task 20.3)
    vary the REAL work per iteration -- an integer micro-step count around
    ``params.work_base_accum`` -- mirroring the measured
    :func:`code.b2.workloads.work_jitter_schedule` (see
    :func:`_periodic_F_meta`).

    ``phase_slip_sigma`` / ``harmonic_smooth_s`` / ``shape_fill_frac`` (ST2 /
    task 20.4): Brownian boundary slip at fixed mean period, raised-cosine
    smoothing of phase transitions, and lifting the down level toward the up
    level (see :func:`_periodic_F_meta`). All defaults are byte-identical with
    zero extra RNG.
    """
    F, phase_starts = _periodic_F_meta(
        t, f_peak, params.f0_lo, params.f0_hi, params.sigma_jitter,
        params.rho_tr_lo, params.rho_tr_hi, params.sigma_delta_tr,
        params.mu_delta_tr, params.eta_up_tr, params.eta_down_tr, rng,
        f0=f0, eta_scale=eta_scale,
        f0_drift_hz=f0_drift_hz, f0_drift_theta=params.f0_drift_theta,
        work_sigma=work_sigma, work_drift_hz=work_drift_hz,
        work_shift_hz=work_shift_hz, work_base_accum=params.work_base_accum,
        phase_slip_sigma=phase_slip_sigma,
        harmonic_smooth_s=harmonic_smooth_s,
        shape_fill_frac=shape_fill_frac,
    )
    return F, phase_starts[::2]


def training_F_phase_meta(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
    work_sigma: float = 0.0,
    work_drift_hz: float = 0.0,
    work_shift_hz: float | None = None,
    phase_slip_sigma: float = 0.0,
    harmonic_smooth_s: float = 0.0,
    shape_fill_frac: float = 0.0,
) -> tuple[np.ndarray, TrainingPhaseMetadata]:
    """Training trace plus complete compute/communication phase metadata.

    This is a non-breaking, evaluation-only sibling of :func:`training_F_meta`.
    It calls the same generator once and exposes the alternating phase starts
    that the existing wrapper intentionally reduces to iteration boundaries.
    Only complete compute/down pairs are returned, so every duration is finite
    and lies within the generated schedule (the final partial iteration is
    omitted).
    """
    F, phase_starts = _periodic_F_meta(
        t, f_peak, params.f0_lo, params.f0_hi, params.sigma_jitter,
        params.rho_tr_lo, params.rho_tr_hi, params.sigma_delta_tr,
        params.mu_delta_tr, params.eta_up_tr, params.eta_down_tr, rng,
        f0=f0, eta_scale=eta_scale,
        f0_drift_hz=f0_drift_hz, f0_drift_theta=params.f0_drift_theta,
        work_sigma=work_sigma, work_drift_hz=work_drift_hz,
        work_shift_hz=work_shift_hz, work_base_accum=params.work_base_accum,
        phase_slip_sigma=phase_slip_sigma,
        harmonic_smooth_s=harmonic_smooth_s,
        shape_fill_frac=shape_fill_frac,
    )
    n_complete = max((phase_starts.size - 1) // 2, 0)
    compute_starts = phase_starts[:2 * n_complete:2].copy()
    communication_starts = phase_starts[1:2 * n_complete:2].copy()
    next_compute_starts = phase_starts[2:2 * n_complete + 1:2]
    metadata = TrainingPhaseMetadata(
        compute_starts=compute_starts,
        communication_starts=communication_starts,
        compute_durations=communication_starts - compute_starts,
        communication_durations=next_compute_starts - communication_starts,
    )
    return F, metadata


def training_F(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
    work_sigma: float = 0.0,
    work_drift_hz: float = 0.0,
    work_shift_hz: float | None = None,
    phase_slip_sigma: float = 0.0,
    harmonic_smooth_s: float = 0.0,
    shape_fill_frac: float = 0.0,
) -> np.ndarray:
    """Training FLOP-rate trace F(t) [FLOP/s] (Ko eqs 1-5).

    Thin wrapper over :func:`training_F_meta` (see there for every knob),
    discarding the iteration-boundary metadata — byte-identical to the
    pre-meta generator (guarded in tests/test_ko_workload.py).
    """
    return training_F_meta(
        t, params, rng, f_peak=f_peak, f0=f0, eta_scale=eta_scale,
        f0_drift_hz=f0_drift_hz,
        work_sigma=work_sigma, work_drift_hz=work_drift_hz,
        work_shift_hz=work_shift_hz,
        phase_slip_sigma=phase_slip_sigma,
        harmonic_smooth_s=harmonic_smooth_s,
        shape_fill_frac=shape_fill_frac,
    )[0]


def finetune_F(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
) -> np.ndarray:
    """Fine-tuning FLOP-rate trace F(t) [FLOP/s] (Ko eqs 6-10).

    ``f0_drift_hz`` (B1 / task 3.2): same slow centre-frequency wander as
    :func:`training_F`; 0.0 (default) is byte-identical to the Ko-faithful trace.
    """
    return _periodic_F(
        t, f_peak, params.f1_lo, params.f1_hi, params.sigma_jitter,
        params.rho_ft_lo, params.rho_ft_hi, params.sigma_delta_ft,
        params.mu_delta_ft, params.eta_tail_ft, params.eta_idle_ft, rng,
        f0=f0, eta_scale=eta_scale,
        f0_drift_hz=f0_drift_hz, f0_drift_theta=params.f0_drift_theta,
    )


def _band_limited_noise(
    t: np.ndarray, band_lo: float, band_hi: float, rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """White noise with its spectrum kept only in [band_lo, band_hi], scaled to ``rms``.

    Models the decode floor under continuous batching: the per-step cadence exists
    but the batch composition reshuffles every step, smearing it into broadband
    in-band power with NO coherent frequency path. A matched filter sees a raised
    floor; a line tracker finds no path through it. This is the diffuse bulk of the
    in-band power; the MoE envelope and prefill bursts add the quasi-periodic
    structure on top (so the null is structured, not merely white).
    """
    n = t.size
    if n == 0:
        return np.zeros(0)
    fs = 1.0 / (t[1] - t[0]) if n > 1 else 1.0
    w = rng.normal(0.0, 1.0, size=n)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    W[(freqs < band_lo) | (freqs > band_hi)] = 0.0
    x = np.fft.irfft(W, n=n)
    return x / (x.std() + 1e-12) * rms


def _ou_envelope(
    t: np.ndarray, tau_s: float, depth: float, rng: np.random.Generator,
) -> np.ndarray:
    """Slow Ornstein-Uhlenbeck amplitude envelope 1 + depth*x, x unit-std.

    Models MoE expert-load imbalance: a slowly-varying multiplicative envelope on
    the decode floor. Amplitude modulation spreads the decode energy into sidebands,
    making the inference null harder (more diffuse in-band) for the line detectors.
    """
    n = t.size
    if n == 0:
        return np.ones(0)
    dt = t[1] - t[0] if n > 1 else 1.0
    phi = np.exp(-dt / tau_s) if tau_s > 0 else 0.0
    x = np.empty(n)
    x[0] = rng.normal()
    innov = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, innov)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return 1.0 + depth * x


def _prefill_bursts(
    t: np.ndarray, rate_hz: float, amp: float, dur_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Aperiodic prefill-burst component: box pulses at Poisson arrival times.

    Prefill (the prompt-encoding phase) is compute-heavy and arrives irregularly
    under continuous batching. Poisson (memoryless) arrivals carry NO line -- they
    raise broadband in-band power, a key part of the hard null (spec.md Sec. 4 B1).
    """
    n = t.size
    out = np.zeros(n)
    if rate_hz <= 0 or n == 0:
        return out
    t_total = float(t[-1])
    clock = 0.0
    while True:
        clock += float(rng.exponential(1.0 / rate_hz))
        if clock > t_total:
            break
        out[(t >= clock) & (t < clock + dur_s)] += amp
    return out


def inference_F(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    eta_scale: float = 1.0,
) -> np.ndarray:
    """HARD inference-null FLOP-rate trace F(t) [FLOP/s] (B1 / task 3.3).

    NOT a Ko et al. workload (they model training/fine-tuning only). The Type null
    must be quasi-periodic with in-band power comparable to a training line but with
    NO concentrated trackable line (see :class:`KoInferenceParams` for provenance and
    model). Three superposed structures: a decode floor smeared into broadband
    in-band power by continuous batching, a slow MoE amplitude envelope, and
    aperiodic Poisson prefill bursts. Same units/scale convention as
    :func:`training_F` (dimensionless levels x ``f_peak``).
    """
    ip = params.inference
    n = t.size

    # 1. Decode floor: in-band power smeared to broadband by continuous batching
    #    (no coherent line). This is the diffuse bulk.
    decode = _band_limited_noise(t, ip.decode_lo, ip.decode_hi, ip.inf_amp_frac, rng)

    # 2. MoE-routing envelope: slow amplitude modulation -> structure (not white).
    decode = decode * _ou_envelope(t, ip.moe_tau_s, ip.moe_depth, rng)

    # 3. Aperiodic prefill bursts (Poisson arrivals): broadband in-band spikes.
    prefill = _prefill_bursts(
        t, ip.prefill_rate_hz, ip.prefill_amp, ip.prefill_dur_s, rng
    )

    # 4. Sub-step intra-phase noise (as Ko's eta).
    eta_lo, eta_hi = ip.eta_inf
    eta = rng.normal(0.0, 1.0, size=t.shape) * (eta_scale * 0.5 * (eta_lo + eta_hi))

    F = f_peak * (ip.base_level + decode + prefill + eta)
    return np.clip(F, 0.0, None)


def _aggregate_background(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float,
    n_tr: int,
    n_ft: int,
    eta_scale: float,
) -> np.ndarray:
    """The non-dominant part of the eq-11 aggregate: ``n_tr`` small trainings +
    ``n_ft`` fine-tunings at the configured ``w_tr``/``w_ft`` shares of ``f_peak``.

    Shared by :func:`aggregate_F` and :func:`aggregate_null_F` so the two B1-agg
    classes differ ONLY in their dominant slot. Note each small training draws its
    own ``f0`` from the same band as a dominant one -- the background carries real
    (weak) lines, so the aggregate null is not line-free.
    """
    _, w_tr, w_ft = params.aggregate_ratio
    total = sum(params.aggregate_ratio)
    F = np.zeros_like(t)
    for _ in range(n_tr):
        F = F + training_F(t, params, rng,
                           f_peak=f_peak * (w_tr / total) / n_tr,
                           eta_scale=eta_scale)
    for _ in range(n_ft):
        F = F + finetune_F(t, params, rng,
                           f_peak=f_peak * (w_ft / total) / n_ft,
                           eta_scale=eta_scale)
    return F


def aggregate_F(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    n_tr: int = 4,
    n_ft: int = 4,
    f0: float | None = None,
    eta_scale: float = 1.0,
    f0_drift_hz: float = 0.0,
) -> np.ndarray:
    """Superposed PDU-level FLOP rate (Ko eq 11): one dominant training workload +
    ``n_tr`` small trainings + ``n_ft`` fine-tunings, at the configured dominance
    ratio. The B1-agg positive class (task 4); peak total ~ ``f_peak``.

    ``f0`` and ``f0_drift_hz`` apply to the DOMINANT training only (the line under
    test); the background workloads always draw their own frequencies.
    ``eta_scale`` rescales the sub-step noise of every component (same meter-
    averaging argument as :func:`training_F`). Defaults draw the same RNG stream
    as before these knobs existed, so seeded aggregates are unchanged.
    """
    w_dom, _, _ = params.aggregate_ratio
    total = sum(params.aggregate_ratio)
    F = training_F(t, params, rng, f_peak=f_peak * w_dom / total,
                   f0=f0, eta_scale=eta_scale, f0_drift_hz=f0_drift_hz)
    return F + _aggregate_background(t, params, rng, f_peak, n_tr, n_ft, eta_scale)


def aggregate_null_F(
    t: np.ndarray,
    params: KoWorkloadParams,
    rng: np.random.Generator,
    f_peak: float = 1.0,
    n_tr: int = 4,
    n_ft: int = 4,
    eta_scale: float = 1.0,
) -> np.ndarray:
    """Aggregate with NO dominant training: the B1-agg null class (task 4).

    NOT a Ko et al. workload mix (their eq 11 is training-only): the Verifier's
    null hypothesis at a shared PDU is a datacentre whose dominant workload is
    inference, over the same small-training + fine-tuning background. The dominant
    slot is the hard inference null (:func:`inference_F`) at the same ``w_dom``
    share of ``f_peak`` as the positive class, so the two classes are power-matched
    and both contain the background's weak lines -- the detector must find the
    dominant line, not just any line.
    """
    w_dom, _, _ = params.aggregate_ratio
    total = sum(params.aggregate_ratio)
    F = inference_F(t, params, rng, f_peak=f_peak * w_dom / total,
                    eta_scale=eta_scale)
    return F + _aggregate_background(t, params, rng, f_peak, n_tr, n_ft, eta_scale)
