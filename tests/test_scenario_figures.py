"""Unit checks for the two scenario-model figure scripts.

Both scripts generate their own data rather than reading a frozen summary, so
the things worth testing are the physics reducers (does the metric say what the
caption claims?) and the writers' plumbing, not the rendering. The expensive
part is the 20-seed meter sweep; the smoke tests run it at n_seeds=1 into a
tmp_path so tracked figures/ is never touched.

Loaded by path via importlib because scripts/ is not an importable package (the
scripts carry sys.path shims instead) — the same idiom as tests/test_st2_pareto.py.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np
import pytest

from powerladder.config import DEFAULT, MeterParams

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


traces = _load("plot_scenario_traces")
meter = _load("plot_scenario_meter")


# --- plot_scenario_traces: the scenario gallery -------------------------------

def test_only_the_inference_null_lacks_a_cadence():
    """f0 is a real number for the three periodic classes and NaN for the null.

    The inference null is built from band-limited noise, an OU envelope and
    Poisson arrivals, none of which carries a line — the figure's whole point.
    """
    f0s = {k: traces.make_trace(k)["f0"] for k in traces.PANELS}
    assert np.isnan(f0s["inference"])
    for kind in ("training", "finetune", "aggregate"):
        assert np.isfinite(f0s[kind]) and f0s[kind] > 0.0


def test_cadence_lines_land_where_they_are_marked():
    """The spectral peak of each periodic class sits at the f0 the figure marks.

    Guards against a silent generator/label mismatch, which would put the
    dashed f0 marker somewhere the line is not.
    """
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    for kind in ("training", "finetune", "aggregate"):
        rec = traces.make_trace(kind)
        freqs, psd = traces.multitaper_psd(rec["t"], rec["p"])
        band = (freqs >= band_lo) & (freqs <= band_hi)
        peak = freqs[band][np.argmax(psd[band])]
        assert peak == pytest.approx(rec["f0"], abs=0.05)


def test_inference_null_is_loud_but_incoherent():
    """The null carries in-band POWER without a line — not merely a quiet trace.

    A null that was simply quiet would make Rung 1 trivial. This asserts the
    honest version: comparable (here, larger) in-band power, far lower
    peak-to-median contrast than training.
    """
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi

    def in_band(kind):
        rec = traces.make_trace(kind)
        freqs, psd = traces.multitaper_psd(rec["t"], rec["p"])
        band = (freqs >= band_lo) & (freqs <= band_hi)
        return (float(np.trapezoid(psd[band], freqs[band])),
                float(psd[band].max() / np.median(psd[band])))

    power_inf, contrast_inf = in_band("inference")
    power_tr, contrast_tr = in_band("training")
    assert power_inf > power_tr
    assert contrast_inf < 0.25 * contrast_tr


def test_traces_writer_smoke(tmp_path):
    out = traces.plot(traces.build(), fig_dir=tmp_path)
    assert out == tmp_path / "scenario_traces.pdf"
    assert out.exists() and (tmp_path / "scenario_traces.png").exists()


# --- plot_scenario_meter: the observation-map sweep ---------------------------

def test_nominal_channel_normalises_to_exactly_one():
    """The reference point is unity by construction, not approximately.

    Every panel contains the nominal setting, and the meter RNG is re-seeded
    per grid point, so that point must reproduce the reference bit for bit.
    Were it not exact, each curve would be normalised to a different channel.
    """
    trace = meter.device_trace(0)
    seed = [meter.SEED, 0, 0]
    ref = meter._score(trace, meter.NOMINAL_METER, seed)
    again = meter._score(trace, dataclasses.replace(meter.NOMINAL_METER), seed)
    assert ref > 0.0
    assert again == ref


def test_default_meter_is_the_exact_noop_path():
    """A default MeterParams passes the device trace through untouched.

    The sweep leans on this invariant: with every axis off, movement in the
    figure can only come from the axis being swept.
    """
    from powerladder.observation import apply_meter
    trace = meter.device_trace(0)
    mt = apply_meter(trace.t, trace.P_obs, MeterParams(),
                     np.random.default_rng(0))
    assert np.array_equal(mt.t, trace.t)
    assert np.array_equal(mt.P_meter, trace.P_obs)


def test_line_past_nyquist_scores_zero():
    """Decimation below 2*f0 makes the cadence unobservable, not merely faint."""
    trace = meter.device_trace(0)
    dead = dataclasses.replace(meter.NOMINAL_METER,
                               sample_hz=meter.F0_HZ, integ_window_s=0.0)
    assert meter._score(trace, dead, [meter.SEED, 0, 0]) == 0.0


def test_pinned_cadence_avoids_the_grid_coincidences():
    """f0 must not sit on a swept Nyquist frequency or a boxcar sinc null.

    At f0 = 1.0 Hz both the 2 Hz sample rate and the 1 s integration window
    would null the line exactly, reading as a knife-edge collapse that
    misrepresents each axis.
    """
    f0 = meter.F0_HZ
    sample_rates = [lv for lv in dict(
        (a["key"], a) for a in meter.AXES)["sample_hz"]["levels"] if lv]
    for fs in sample_rates:
        assert f0 != pytest.approx(0.5 * fs), f"f0 sits on the {fs} Hz Nyquist"
    windows = [lv for lv in dict(
        (a["key"], a) for a in meter.AXES)["integ_window_s"]["levels"] if lv]
    for w in windows:
        # Boxcar sinc nulls at integer multiples of 1/w.
        assert abs(f0 * w - round(f0 * w)) > 0.05, f"f0 sits on a {w}s sinc null"


@pytest.mark.parametrize("key", [a["key"] for a in meter.AXES])
def test_every_axis_degrades_the_cadence(key):
    """Each observation-map axis must cost cadence SNR at its extreme setting.

    The figure's claim is that all six axes attenuate the signature; an axis
    whose companion fields were wired up wrongly would be inert and sit flat at
    1.0, which this catches.
    """
    axis = next(a for a in meter.AXES if a["key"] == key)
    rec = meter.sweep_axis(axis, n_seeds=1)
    med = np.asarray(rec["median"])
    assert med[0] >= med[-1]
    assert med[-1] < 0.5 * med[0]


def test_meter_sweep_schema_and_writer_smoke(tmp_path):
    sweeps = {a["key"]: meter.sweep_axis(a, n_seeds=1) for a in meter.AXES}
    assert set(sweeps) == {a["key"] for a in meter.AXES}
    for axis in meter.AXES:
        rec = sweeps[axis["key"]]
        for field in ("levels", "median", "q1", "q3", "n_seeds"):
            assert field in rec
        assert len(rec["median"]) == len(axis["levels"])
        assert all(q1 <= m <= q3 for q1, m, q3
                   in zip(rec["q1"], rec["median"], rec["q3"]))

    out = meter.plot(sweeps, fig_dir=tmp_path)
    assert out == tmp_path / "scenario_meter.pdf"
    assert out.exists() and (tmp_path / "scenario_meter.png").exists()
