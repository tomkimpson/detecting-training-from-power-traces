"""Observation-channel (meter map) tests — ST2 task 1 (plan-for-paper-2 §2).

The load-bearing invariant is the first test: the default MeterParams is an
EXACT no-op that draws ZERO RNG, so wiring the meter through the trace
builders cannot perturb any existing seeded population. The rest checks each
stage against its analytic prediction: Butterworth attenuation, notch kill,
aliasing under undecimated sampling, the controller line, and the AR(1)
correlation of the meter noise.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from powerladder.config import MeterParams
from powerladder.forward import make_time_grid
from powerladder.observation import apply_meter

FS = 100.0


def _grid(T: float, fs: float = FS) -> np.ndarray:
    return make_time_grid(T, 1.0 / fs)


def _tone(t: np.ndarray, f: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * f * t)


def _peak_hz(t: np.ndarray, x: np.ndarray) -> float:
    """Dominant non-DC frequency of x."""
    x = x - x.mean()
    freqs = np.fft.rfftfreq(x.size, t[1] - t[0])
    psd = np.abs(np.fft.rfft(x)) ** 2
    return float(freqs[np.argmax(psd)])


def _bin_power(t: np.ndarray, x: np.ndarray, f: float) -> float:
    """|FFT|^2 at the bin nearest f."""
    freqs = np.fft.rfftfreq(x.size, t[1] - t[0])
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return float(psd[np.argmin(np.abs(freqs - f))])


# --- the critical invariant ---------------------------------------------------

def test_default_is_exact_noop_with_zero_rng():
    """MeterParams() returns (t, P) unchanged and leaves the RNG untouched."""
    t = _grid(30.0)
    p = 250.0 + _tone(t, 1.0, 30.0)
    rng = np.random.default_rng(42)
    state_before = rng.bit_generator.state
    out = apply_meter(t, p, MeterParams(), rng)
    assert rng.bit_generator.state == state_before
    assert np.array_equal(out.t, t)
    assert np.array_equal(out.P_meter, p)


def test_lti_stage_draws_zero_rng():
    """The deterministic stages (lp/notch/gain/integrate/sample) never touch
    the generator — noise ownership stays with the additive terms."""
    t = _grid(30.0)
    p = 250.0 + _tone(t, 1.0, 30.0)
    rng = np.random.default_rng(7)
    state_before = rng.bit_generator.state
    mp = MeterParams(lp_cutoff_hz=2.0, notch_hz=1.0, gain=1.5,
                     integ_window_s=0.2, sample_hz=10.0)
    apply_meter(t, p, mp, rng)
    assert rng.bit_generator.state == state_before


# --- LTI stage -----------------------------------------------------------------

def test_lowpass_attenuates_by_analytic_butterworth_factor():
    """A tone above the cutoff shrinks by |H|^2 = 1/(1 + (f/fc)^(2n))
    (filtfilt applies the Butterworth twice)."""
    t = _grid(200.0)
    f_tone, fc, order = 2.0, 1.0, 2
    p = _tone(t, f_tone)
    out = apply_meter(t, p, MeterParams(lp_cutoff_hz=fc, lp_order=order),
                      np.random.default_rng(0))
    # steady-state amplitude ratio, away from filtfilt edge transients
    mid = slice(t.size // 4, 3 * t.size // 4)
    ratio = np.std(out.P_meter[mid]) / np.std(p[mid])
    analytic = 1.0 / (1.0 + (f_tone / fc) ** (2 * order))
    assert abs(ratio - analytic) < 0.05 * analytic


def test_notch_kills_tone_at_notch_hz():
    """The notch removes the tone at notch_hz while an off-notch tone survives."""
    t = _grid(300.0, fs=20.0)
    f_notch, f_keep = 1.0, 0.4
    p = _tone(t, f_notch) + _tone(t, f_keep)
    out = apply_meter(t, p, MeterParams(notch_hz=f_notch, notch_q=5.0),
                      np.random.default_rng(0))
    assert _bin_power(t, out.P_meter, f_notch) < 1e-2 * _bin_power(t, p, f_notch)
    assert _bin_power(t, out.P_meter, f_keep) > 0.5 * _bin_power(t, p, f_keep)


def test_notch_depth_1_matches_full_null():
    """notch_depth=1.0 (default) reproduces the full iirnotch null — the
    backward-compatible default, so existing sweeps are unchanged."""
    t = _grid(300.0, fs=20.0)
    p = _tone(t, 1.0) + _tone(t, 0.4)
    rng = np.random.default_rng(0)
    default = apply_meter(t, p, MeterParams(notch_hz=1.0, notch_q=5.0), rng)
    explicit = apply_meter(t, p, MeterParams(notch_hz=1.0, notch_q=5.0,
                                             notch_depth=1.0), rng)
    assert np.allclose(default.P_meter, explicit.P_meter)


def test_notch_depth_0_is_identity():
    """notch_depth=0.0 disables the anti-resonance even with notch_hz set: the
    tone at the notch centre survives intact."""
    t = _grid(300.0, fs=20.0)
    p = _tone(t, 1.0)
    out = apply_meter(t, p, MeterParams(notch_hz=1.0, notch_q=5.0,
                                        notch_depth=0.0),
                      np.random.default_rng(0))
    assert np.allclose(out.P_meter, p)


def test_notch_depth_partial_attenuates_between():
    """0 < depth < 1 attenuates the notch-centre tone partially: less than the
    full null, more than untouched (monotone in depth)."""
    t = _grid(300.0, fs=20.0)
    p = _tone(t, 1.0)
    p0 = _bin_power(t, p, 1.0)

    def resid(depth):
        out = apply_meter(t, p, MeterParams(notch_hz=1.0, notch_q=5.0,
                                            notch_depth=depth),
                          np.random.default_rng(0))
        return _bin_power(t, out.P_meter, 1.0)

    full, half = resid(1.0), resid(0.5)
    assert full < half < p0


# --- integrate + sample ---------------------------------------------------------

def test_decimation_without_lowpass_aliases_to_predicted_bin():
    """ZOH decimation at sample_hz with no anti-alias filter folds a
    beyond-Nyquist tone to |f - sample_hz| — the deliberate aliasing case."""
    t = _grid(200.0)
    f_tone, f_samp = 3.0, 4.0          # Nyquist = 2 Hz; alias at |3 - 4| = 1 Hz
    p = _tone(t, f_tone)
    out = apply_meter(t, p, MeterParams(sample_hz=f_samp),
                      np.random.default_rng(0))
    assert out.t.size == int(np.floor(200.0 * f_samp)) + 1
    df = 1.0 / 200.0
    assert abs(_peak_hz(out.t, out.P_meter) - abs(f_tone - f_samp)) <= 2 * df


def test_boxcar_integration_nulls_a_full_period_tone():
    """A trailing boxcar spanning exactly one tone period sits at a sinc null:
    the tone vanishes while the DC level is preserved."""
    t = _grid(100.0)
    f_tone = 1.0
    p = 200.0 + _tone(t, f_tone, 10.0)
    out = apply_meter(t, p, MeterParams(integ_window_s=1.0 / f_tone),
                      np.random.default_rng(0))
    tail = out.P_meter[t > 2.0]        # skip the partial-window start-up
    assert np.std(tail) < 0.02 * 10.0
    assert abs(tail.mean() - 200.0) < 0.1


# --- additive interference + noise ----------------------------------------------

def test_controller_line_appears_at_controller_hz():
    """With a flat device trace, the output PSD peaks at the controller
    frequency (the interference line the meter-robustness family turns on)."""
    t = _grid(300.0, fs=20.0)
    p = np.full_like(t, 250.0)
    mp = MeterParams(controller_hz=0.4, controller_amp=5.0)
    out = apply_meter(t, p, mp, np.random.default_rng(3))
    df = 1.0 / 300.0
    assert abs(_peak_hz(t, out.P_meter) - 0.4) <= 2 * df


def test_eta_ar1_lag1_autocorrelation():
    """AR(1) meter noise has lag-1 autocorrelation exp(-dt/tau)."""
    fs, tau = 10.0, 5.0
    t = _grid(2000.0, fs=fs)
    p = np.zeros_like(t)
    mp = MeterParams(sigma_eta=2.0, eta_tau_s=tau)
    out = apply_meter(t, p, mp, np.random.default_rng(1))
    eta = out.P_meter
    r1 = float(np.corrcoef(eta[:-1], eta[1:])[0, 1])
    assert abs(r1 - np.exp(-1.0 / (fs * tau))) < 0.01
    # marginal std is the configured sigma
    assert abs(np.std(eta) - 2.0) < 0.15
    # and eta_tau_s = 0 falls back to white noise
    mp_white = dataclasses.replace(mp, eta_tau_s=0.0)
    out_w = apply_meter(t, p, mp_white, np.random.default_rng(1))
    r1_w = float(np.corrcoef(out_w.P_meter[:-1], out_w.P_meter[1:])[0, 1])
    assert abs(r1_w) < 0.05
