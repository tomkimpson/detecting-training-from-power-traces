"""Ko et al. workload-generator tests (Phase 2, task 1.1; GPU-free).

Checks the structural invariants the time-resolved tracker and the B1 detector
rely on: the spectral line sits at f0, levels and bounds are physical, the
aggregate superposes its components, and draws are reproducible under a fixed seed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from powerladder.config import DEFAULT
from powerladder.forward import make_time_grid
from powerladder.ko_workload import _periodic_F_meta, aggregate_F, finetune_F, training_F

KO = DEFAULT.ko
F_PEAK = 1.0e15

# Reference digests captured from the generator BEFORE the ST2 knobs existed
# (work variation, phase slip, harmonic smoothing, shaping, the meta refactor)
# — the byte-identical guard is against the pre-change code, not merely
# against the edited code called with explicit defaults.
REFS = json.loads(
    (Path(__file__).parent / "data" / "st2_prechange_refs.json").read_text()
)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    ).hexdigest()


def _psd_peak_hz(F: np.ndarray, dt: float) -> float:
    """Dominant non-DC frequency of F [Hz]."""
    x = F - F.mean()
    freqs = np.fft.rfftfreq(x.size, dt)
    psd = np.abs(np.fft.rfft(x)) ** 2
    return float(freqs[np.argmax(psd)])


def test_training_psd_peak_at_f0():
    """The training fluctuation line sits at the configured f0 (Ko eqs 1-5)."""
    dt = 0.02
    t = make_time_grid(300.0, dt)
    F = training_F(t, KO, np.random.default_rng(0), f_peak=F_PEAK, f0=1.0)
    assert abs(_psd_peak_hz(F, dt) - 1.0) < 0.1


def test_finetune_psd_peak_at_f1():
    """Fine-tuning carries its slower line at the configured frequency (eqs 6-10)."""
    dt = 0.02
    t = make_time_grid(300.0, dt)
    F = finetune_F(t, KO, np.random.default_rng(0), f_peak=F_PEAK, f0=0.5)
    assert abs(_psd_peak_hz(F, dt) - 0.5) < 0.1


def test_nonnegative_and_scaled():
    """F is non-negative and on the order of f_peak (never wildly above it)."""
    t = make_time_grid(120.0, 0.05)
    F = training_F(t, KO, np.random.default_rng(1), f_peak=F_PEAK, f0=1.0)
    assert F.min() >= 0.0
    assert F.max() <= 1.5 * F_PEAK
    assert 0.4 * F_PEAK < F.mean() < F_PEAK   # down phases pull the mean below peak


def test_finetune_idles_deeper_than_training():
    """Ko's mu_delta ordering: fine-tuning idles far lower than training's comm dip."""
    t = make_time_grid(200.0, 0.02)
    tr = training_F(t, KO, np.random.default_rng(2), f_peak=F_PEAK, f0=1.0)
    ft = finetune_F(t, KO, np.random.default_rng(2), f_peak=F_PEAK, f0=0.5)
    # 5th percentile = the low (down/idle) phase floor.
    assert np.percentile(ft, 5) < np.percentile(tr, 5)


def test_eta_scale_reduces_intraphase_variance():
    """Down-weighting Ko's intra-phase noise lowers sample-to-sample variance."""
    t = make_time_grid(120.0, 0.05)
    full = training_F(t, KO, np.random.default_rng(3), f_peak=F_PEAK, f0=1.0,
                      eta_scale=1.0)
    quiet = training_F(t, KO, np.random.default_rng(3), f_peak=F_PEAK, f0=1.0,
                       eta_scale=0.0)
    assert np.std(np.diff(quiet)) < np.std(np.diff(full))


def test_aggregate_superposes_components():
    """Eq-11 aggregate is non-negative and dominated by the training term."""
    t = make_time_grid(60.0, 0.05)
    agg = aggregate_F(t, KO, np.random.default_rng(4), f_peak=F_PEAK, n_tr=3, n_ft=3)
    w_dom, _, _ = KO.aggregate_ratio
    dom = training_F(t, KO, np.random.default_rng(4),
                     f_peak=F_PEAK * w_dom / sum(KO.aggregate_ratio))
    assert agg.min() >= 0.0
    # the dominant training workload supplies most of the aggregate mean.
    assert dom.mean() > 0.7 * agg.mean()


def test_reproducible_under_seed():
    """Identical seed + grid -> identical trace (reproducibility, CLAUDE.md)."""
    t = make_time_grid(30.0, 0.05)
    a = training_F(t, KO, np.random.default_rng(5), f_peak=F_PEAK, f0=1.0)
    b = training_F(t, KO, np.random.default_rng(5), f_peak=F_PEAK, f0=1.0)
    assert np.array_equal(a, b)


# --- ST2 byte-identical guards (tasks 20.3-20.5) ------------------------------

def test_prechange_reference_digests():
    """All-default calls reproduce the exact traces captured before any ST2
    knob or the _periodic_F_meta refactor existed (seeded sha256 equality)."""
    t = make_time_grid(60.0, 0.05)
    t30 = make_time_grid(30.0, 0.05)
    assert _digest(training_F(t, KO, np.random.default_rng(123),
                              f_peak=1.0, f0=1.0)) \
        == REFS["training_f0_1.0_seed123"]
    assert _digest(training_F(t, KO, np.random.default_rng(124), f_peak=1.0)) \
        == REFS["training_f0_None_seed124"]
    assert _digest(training_F(t, KO, np.random.default_rng(125), f_peak=1.0,
                              f0=1.0, f0_drift_hz=0.3)) \
        == REFS["training_drift_0.3_seed125"]
    assert _digest(finetune_F(t, KO, np.random.default_rng(126),
                              f_peak=1.0, f0=0.5)) \
        == REFS["finetune_f0_0.5_seed126"]
    assert _digest(aggregate_F(t30, KO, np.random.default_rng(127),
                               f_peak=1.0, n_tr=2, n_ft=2)) \
        == REFS["aggregate_seed127"]


def test_work_defaults_byte_identical():
    """Explicit no-op work knobs draw the same RNG stream as their absence."""
    t = make_time_grid(60.0, 0.05)
    base = training_F(t, KO, np.random.default_rng(123), f_peak=1.0, f0=1.0)
    expl = training_F(t, KO, np.random.default_rng(123), f_peak=1.0, f0=1.0,
                      work_sigma=0.0, work_drift_hz=0.0, work_shift_hz=None)
    assert np.array_equal(base, expl)
    assert _digest(base) == REFS["training_f0_1.0_seed123"]


# --- ST2 work-variation knobs (task 20.3) --------------------------------------

def _train_meta(rng, *, duration=600.0, f0=1.0, sigma_jitter=None,
                rho=None, **knobs):
    """training_F's parameter plumbing but through _periodic_F_meta, so the
    tests can read the generator's ground-truth phase boundaries."""
    p = KO
    if sigma_jitter is not None:
        p = dataclasses.replace(p, sigma_jitter=sigma_jitter)
    rho_lo, rho_hi = (p.rho_tr_lo, p.rho_tr_hi) if rho is None else (rho, rho)
    t = make_time_grid(duration, 0.05)
    return _periodic_F_meta(
        t, 1.0, p.f0_lo, p.f0_hi, p.sigma_jitter, rho_lo, rho_hi,
        p.sigma_delta_tr, p.mu_delta_tr, p.eta_up_tr, p.eta_down_tr, rng,
        f0=f0, f0_drift_theta=p.f0_drift_theta,
        work_base_accum=p.work_base_accum, **knobs,
    )


def _phases(starts):
    """(T_up, T_down) arrays from the alternating phase-boundary times."""
    n_iter = (starts.size - 1) // 2          # complete iterations
    up = starts[1:2 * n_iter:2] - starts[0:2 * n_iter:2]
    down = starts[2:2 * n_iter + 1:2] - starts[1:2 * n_iter:2]
    return up, down


def test_work_sigma_preserves_mean_cadence():
    """work_sigma jitters the period but the mean cadence stays at f0
    (the zero-cost attack moves NO average work)."""
    _, starts = _train_meta(np.random.default_rng(0), work_sigma=0.2)
    periods = np.diff(starts[::2])
    assert abs(periods.mean() - 1.0) < 0.06
    assert periods.std() / periods.mean() > 0.1   # and it does jitter


def test_work_shift_retargets_mean_cadence():
    """work_shift_hz parks the MEAN total cadence at the target frequency
    (T_down is subtracted before quantising the up phase)."""
    _, starts = _train_meta(np.random.default_rng(1), work_shift_hz=0.5)
    periods = np.diff(starts[::2])
    assert abs(periods.mean() - 2.0) < 0.1


def test_work_tdown_fixed_while_tup_varies():
    """The comm/optimizer phase does not scale with the accumulation count:
    T_down keeps its honest distribution while T_up carries the variation, so
    the duty ratio co-moves with G_i."""
    rho = 0.7
    _, s_honest = _train_meta(np.random.default_rng(2), rho=rho)
    _, s_work = _train_meta(np.random.default_rng(3), rho=rho, work_sigma=0.35)
    up_h, down_h = _phases(s_honest)
    up_w, down_w = _phases(s_work)
    # T_down distribution unchanged (same mean/std as the honest generator)
    assert abs(down_w.mean() - down_h.mean()) < 0.02 * down_h.mean()
    assert abs(down_w.std() - down_h.std()) < 0.3 * down_h.std()
    # T_up now varies much more than the honest up phase
    assert up_w.std() > 2.0 * up_h.std()
    # and with jitter-free honest timing T_down is exactly constant
    _, s_q = _train_meta(np.random.default_rng(4), rho=rho, sigma_jitter=0.0,
                         work_sigma=0.35)
    _, down_q = _phases(s_q)
    assert np.allclose(down_q, (1.0 - rho) / 1.0)


# --- ST2 phase-slip / harmonic-smoothing / shaping knobs (task 20.4) -----------

def _band_power(t, x, lo, hi):
    f = np.fft.rfftfreq(x.size, t[1] - t[0])
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return float(psd[(f >= lo) & (f <= hi)].sum())


def test_deperiod_defaults_byte_identical():
    """Explicit no-op task-20.4 knobs draw the same RNG stream as their absence
    AND reproduce the pre-change reference digest."""
    t = make_time_grid(60.0, 0.05)
    base = training_F(t, KO, np.random.default_rng(123), f_peak=1.0, f0=1.0)
    expl = training_F(t, KO, np.random.default_rng(123), f_peak=1.0, f0=1.0,
                      phase_slip_sigma=0.0, harmonic_smooth_s=0.0,
                      shape_fill_frac=0.0)
    assert np.array_equal(base, expl)
    assert _digest(expl) == REFS["training_f0_1.0_seed123"]


def test_shape_fill_one_collapses_swing():
    """phi = 1 lifts every down level onto its up level: the in-band RMS of the
    iteration swing collapses (residual: per-iteration level fluctuations)."""
    t = make_time_grid(300.0, 0.02)
    honest = training_F(t, KO, np.random.default_rng(0), f_peak=1.0, f0=1.0,
                        eta_scale=0.0)
    filled = training_F(t, KO, np.random.default_rng(0), f_peak=1.0, f0=1.0,
                        eta_scale=0.0, shape_fill_frac=1.0)
    ratio = _band_power(t, filled, 0.5, 1.5) / _band_power(t, honest, 0.5, 1.5)
    assert ratio < 0.05


def test_harmonic_smoothing_kills_comb_fundamental_survives():
    """Raised-cosine smoothing suppresses the 2nd/3rd harmonics far more than
    the fundamental (the comb dies, the line survives)."""
    p = dataclasses.replace(KO, sigma_jitter=0.0,
                            rho_tr_lo=0.65, rho_tr_hi=0.65)  # clean comb
    t = make_time_grid(300.0, 0.02)
    raw = training_F(t, p, np.random.default_rng(1), f_peak=1.0, f0=1.0,
                     eta_scale=0.0)
    smooth = training_F(t, p, np.random.default_rng(1), f_peak=1.0, f0=1.0,
                        eta_scale=0.0, harmonic_smooth_s=0.5)

    def h(x, k):
        return _band_power(t, x, k - 0.05, k + 0.05)

    assert h(smooth, 1.0) > 0.5 * h(raw, 1.0)     # fundamental survives
    assert h(smooth, 2.0) < 0.4 * h(raw, 2.0)     # 2nd harmonic dies
    assert h(smooth, 3.0) < 0.15 * h(raw, 3.0)    # 3rd harmonic dies harder


def test_phase_slip_boundary_variance_grows_linearly():
    """Brownian boundary slip: Var[t_{i+k} - t_i - k*T] ~ k * (sigma*T)^2, at
    an unchanged mean period."""
    sigma_s, f0 = 0.05, 1.0
    _, starts = _train_meta(np.random.default_rng(2), duration=1500.0, f0=f0,
                            sigma_jitter=0.0, rho=0.65,
                            phase_slip_sigma=sigma_s)
    it = starts[::2]
    t_bar = float(np.mean(np.diff(it)))
    assert abs(t_bar - 1.0 / f0) < 0.01           # mean period unchanged

    def var_at_lag(k):
        d = it[k:] - it[:-k] - k * t_bar
        return float(np.var(d))

    pred = (sigma_s / f0) ** 2                    # per-iteration slip variance
    assert abs(var_at_lag(5) - 5 * pred) < 0.5 * 5 * pred
    assert abs(var_at_lag(20) - 20 * pred) < 0.5 * 20 * pred
    # linear growth: quadrupling the lag ~quadruples the variance
    assert 2.0 < var_at_lag(20) / var_at_lag(5) < 8.0


def _work_jitter_schedule_replica(n, *, base_accum, jitter_sigma, rng):
    """Replicated from code.b2.workloads.work_jitter_schedule (jitter-only
    case). That module imports torch at module level, so it cannot be imported
    in this CPU-only suite; these are its exact lines for drift_hz=0,
    shift_to_hz=None (incl. the 0.05 jitter-factor floor and stochastic
    rounding with floor 1)."""
    g_real = np.empty(n)
    for i in range(n):
        xi = rng.normal(0.0, jitter_sigma)
        g_real[i] = base_accum / max(1.0 + xi, 0.05)
    lo = np.floor(g_real)
    g = lo + (rng.random(g_real.shape) < g_real - lo)
    return np.maximum(g, 1.0).astype(np.int64)


def test_work_G_distribution_matches_work_jitter_schedule():
    """The integer micro-step counts G_i realised by the generator follow the
    same distribution as the hardware campaign's work_jitter_schedule at the
    same (base_accum, sigma) — the 1:1 mapping onto measured b2 anchors."""
    rho, sigma, accum = 0.7, 0.35, KO.work_base_accum
    # jitter-free honest timing + fixed rho => t_micro is constant and G_i is
    # exactly recoverable from the up-phase durations
    _, starts = _train_meta(np.random.default_rng(5), duration=1200.0,
                            rho=rho, sigma_jitter=0.0, work_sigma=sigma)
    up, _ = _phases(starts)
    t_micro = rho / (1.0 * accum)
    G = np.rint(up / t_micro).astype(np.int64)
    assert np.allclose(G * t_micro, up)           # exact micro-step quantisation
    assert G.min() >= 1
    G_ref = _work_jitter_schedule_replica(
        8000, base_accum=accum, jitter_sigma=sigma,
        rng=np.random.default_rng(99))
    assert abs(G.mean() - G_ref.mean()) < 0.5
    # the 1/(1+xi) tail makes the std a noisy statistic -> relative tolerance
    assert abs(G.std() - G_ref.std()) < 0.25 * G_ref.std()
    # quartiles agree to within one integer micro-step bin
    assert np.all(np.abs(np.percentile(G, [25, 50, 75])
                         - np.percentile(G_ref, [25, 50, 75])) <= 1.0)
    assert ks_2samp(G, G_ref).pvalue > 0.005
