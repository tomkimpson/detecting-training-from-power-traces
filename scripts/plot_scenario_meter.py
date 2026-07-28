"""Sensitivity of the cadence signature to the observation map.

Paper figure for the scenario-models section. The generator produces a device
power trace; what a verifier actually reads is that trace after the observation
map -- power-delivery and reporting bandwidth, meter integration, sample
cadence, in-band transfer-function notches, additive noise, and periodic
controller interference. Each of those is a knob on MeterParams, and each can
attenuate the iteration cadence before any detector sees it.

This figure sweeps one axis at a time on a FIXED honest training workload and
reports how much of the cadence survives, as

    cadence signal-to-interference ratio
        = multitaper PSD at the known f0
          / in-band power the meter itself contributes

normalised to the value on the nominal channel (20 Hz, no filtering, no
decimation, sigma_eta = 4 W). The point at the nominal setting is 1 by
construction. Attenuating axes (bandwidth, integration, decimation, an in-band
notch) act on the numerator, additive ones (meter noise, controller
interference) on the denominator, so a single curve reads the same way on
every panel: down means less cadence reaches the verifier.

Two things to be clear about:

  * This is a property of the SIGNAL IN THE CHANNEL, not of a detector. It says
    how much cadence structure the observation map delivers, on an honest
    workload with no adversary. What each detector class does with what is left
    -- and hence the minimum meter specification -- is the frontier section's
    meter-boundary sweep, which grids the same axes against detection rate.
  * The axes here are the ones the code implements. Note that the realised
    quantisation is TEMPORAL (boxcar integration and zero-order-hold
    decimation); there is no amplitude-quantisation knob in the meter model.

The grids reuse config.St2MeterBoundaryParams where they overlap, so the axes
line up with the frontier section rather than inventing a second set.

Reproduce:
    python scripts/plot_scenario_meter.py
Outputs:
    figures/scenario_meter.{pdf,png}
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT, MeterParams  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid, simulate  # noqa: E402
from powerladder.ko_workload import training_F  # noqa: E402
from powerladder.noise import WhiteNoise  # noqa: E402
from powerladder.observation import apply_meter  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.st1.multitaper import dpss_eigencoef  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
SEED = 0
N_SEEDS = 20              # workload realisations per grid point
# Display floor for the log axis; grid points at or below it are marked dead
# (an exact zero means decimation pushed f0 to or past Nyquist).
YFLOOR = 1e-3
# Pinned mid-band training cadence. Deliberately NOT 1.0 Hz: that value sits
# exactly on the Nyquist frequency of the 2 Hz sample grid and exactly on the
# first sinc null of a 1 s boxcar, so it would read as a knife-edge collapse on
# two of the panels and misrepresent both axes.
F0_HZ = 0.9
_MB = DEFAULT.st2_meter_boundary

# The reference channel every panel normalises to: the nominal off-chip meter.
# Only meter noise is active, at the glue's sigma_eta; every other axis is off.
NOMINAL_METER = MeterParams(sigma_eta=DEFAULT.ko_typeb.sigma_eta)

# One entry per panel: (key, field, levels, x-label, log-x, tick labels).
# ``levels`` always CONTAINS the nominal value, so each panel has an exact
# unit reference point. Grids marked "== St2MeterBoundaryParams" are shared
# with the frontier section's meter-boundary sweep.
AXES: tuple[dict, ...] = (
    {
        "key": "lp_cutoff_hz",
        "field": "lp_cutoff_hz",
        # None == no power-delivery / reporting low-pass (the nominal channel).
        # Capped below the 10 Hz Nyquist of the 20 Hz device grid, which the
        # Butterworth design rejects.
        "levels": (None, 8.0, 5.0, 2.0, 1.0, 0.5),
        "xlabel": "low-pass cutoff [Hz]",
        "offlabel": "off",
    },
    {
        "key": "integ_window_s",
        "field": "integ_window_s",
        "levels": _MB.integ_window_grid,          # == St2MeterBoundaryParams
        "xlabel": "integration window [s]",
        "offlabel": None,
    },
    {
        "key": "sample_hz",
        "field": "sample_hz",
        # None == the 20 Hz device grid, i.e. no decimation (the nominal channel).
        "levels": (None,) + _MB.sample_hz_grid[1:],   # == St2MeterBoundaryParams
        "xlabel": "sample rate [Hz]",
        "offlabel": "20",
    },
    {
        "key": "notch_depth",
        "field": "notch_depth",
        "levels": (0.0,) + _MB.notch_depth_grid,  # == St2MeterBoundaryParams
        "xlabel": f"notch depth at $f_0$ (Q={_MB.notch_q:g})",
        "offlabel": None,
    },
    {
        "key": "sigma_eta",
        "field": "sigma_eta",
        "levels": (1.0, DEFAULT.ko_typeb.sigma_eta, 12.0, 30.0, 60.0, 120.0),
        "xlabel": r"meter noise $\sigma_\eta$ [W]",
        "offlabel": None,
    },
    {
        "key": "controller_amp",
        "field": "controller_amp",
        "levels": (0.0, 10.0, 30.0, 60.0, 120.0),
        "xlabel": "controller amplitude [W pk-pk]",
        "offlabel": None,
    },
)
# The controller and notch axes need a companion field set alongside the swept
# one, otherwise the stage is inert. Values match St2Params.meter_variants.
AXIS_COMPANIONS = {
    "notch_depth": {"notch_hz": F0_HZ, "notch_q": _MB.notch_q},
    "controller_amp": {"controller_hz": 0.38, "controller_duty": 0.4,
                       "controller_wander_hz": 0.05},
}


def device_trace(seed: int):
    """One honest training workload as a noiseless device trace.

    The workload is fixed across the meter sweep: only the observation map
    changes between grid points, so any movement in the figure is the channel's
    doing and not the generator's. Simulated noiselessly because the meter map
    owns all observation noise (the noise-ownership rule of
    powerladder.typeb.ko_synth); the glue's sigma_eta enters through
    NOMINAL_METER instead.
    """
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    rng = np.random.default_rng([SEED, seed])
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_flop = training_F(t, ko, rng, f_peak=glue.f_peak_frac * DEFAULT.floor.F_max,
                        f0=F0_HZ, eta_scale=glue.eta_scale)
    spec = TraceSpec(t=t, F=f_flop,
                     r=np.full_like(t, glue.r),
                     P0=np.full_like(t, glue.P0))
    return simulate(spec, WhiteNoise(0.0), rng)


def _psd(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Multitaper PSD of a trace, plus its sample rate."""
    fs = 1.0 / float(t[1] - t[0])
    freqs, y, _ = dpss_eigencoef(p, fs, params=DEFAULT.st1)
    return freqs, np.mean(np.abs(y) ** 2, axis=0) / fs, fs


def cadence_snr(t_sig, p_sig, t_int, p_int, f0: float) -> float:
    """Cadence power at f0 over the meter's own in-band interference power.

    The numerator is the PSD at the KNOWN f0 -- what the channel delivers at
    the cadence, not what a search would find. The denominator is the in-band
    power of the SAME meter realisation driven by a flat workload, i.e. the
    additive noise, baseline wander and controller interference the meter
    contributes on its own.

    Splitting it this way is what makes the measure monotone under both of the
    channel's mechanisms. Attenuation (bandwidth, integration, decimation, an
    in-band notch) shrinks the numerator; additive interference (meter noise, a
    controller limit cycle) grows the denominator. A ratio of the line to the
    in-band BACKGROUND would instead be blind to attenuation, because a boxcar
    that suppresses the line suppresses its skirt with it -- measured, and the
    reason that ratio is not used here.

    Returns 0.0 if decimation has pushed f0 to or past Nyquist: the line is
    then not in the observable band at all.
    """
    freqs, psd, fs = _psd(t_sig, p_sig)
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    if f0 >= 0.5 * fs:
        return 0.0
    band = (freqs >= band_lo) & (freqs <= min(band_hi, 0.5 * fs))
    if not np.any(band):
        return 0.0
    freqs_i, psd_i, _ = _psd(t_int, p_int)
    band_i = (freqs_i >= band_lo) & (freqs_i <= min(band_hi, 0.5 * fs))
    interference = float(np.mean(psd_i[band_i])) if np.any(band_i) else 0.0
    if interference <= 0.0:
        raise RuntimeError("meter contributes no in-band interference — "
                           "the SNR denominator is degenerate")
    return float(psd[np.argmin(np.abs(freqs - f0))] / interference)


def _meter_for(axis: dict, level) -> MeterParams:
    """The nominal meter with this axis' field (and companions) overridden."""
    overrides = dict(AXIS_COMPANIONS.get(axis["key"], {}))
    overrides[axis["field"]] = level
    return dataclasses.replace(NOMINAL_METER, **overrides)


def _score(spec_trace, meter: MeterParams, rng_seed) -> float:
    """Cadence SNR of one workload under one meter configuration.

    The interference run reuses the same RNG seed, so it is the same additive
    realisation the signal run saw -- the denominator is that meter's own
    contribution, not an independent draw.
    """
    mt = apply_meter(spec_trace.t, spec_trace.P_obs, meter,
                     np.random.default_rng(rng_seed))
    flat = np.full_like(spec_trace.P_obs, DEFAULT.ko_typeb.P0)
    mi = apply_meter(spec_trace.t, flat, meter, np.random.default_rng(rng_seed))
    return cadence_snr(mt.t, mt.P_meter, mi.t, mi.P_meter, F0_HZ)


def sweep_axis(axis: dict, n_seeds: int = N_SEEDS) -> dict:
    """Normalised survival curve for one observation-map axis.

    For each workload realisation the reference and every grid point are scored
    with a freshly seeded meter RNG, so the grid point that equals the nominal
    setting reproduces the reference exactly and normalises to 1.
    """
    curves = []
    for s in range(n_seeds):
        trace = device_trace(s)
        rng_seed = [SEED, s, AXES.index(axis)]
        ref = _score(trace, NOMINAL_METER, rng_seed)
        if ref <= 0.0:
            raise RuntimeError("reference channel scored zero — check NOMINAL_METER")
        curves.append([_score(trace, _meter_for(axis, lv), rng_seed) / ref
                       for lv in axis["levels"]])
    arr = np.asarray(curves, dtype=float)
    return {
        "key": axis["key"],
        "levels": list(axis["levels"]),
        "median": np.median(arr, axis=0).tolist(),
        "q1": np.percentile(arr, 25, axis=0).tolist(),
        "q3": np.percentile(arr, 75, axis=0).tolist(),
        "n_seeds": n_seeds,
    }


def build(n_seeds: int = N_SEEDS) -> dict[str, dict]:
    return {axis["key"]: sweep_axis(axis, n_seeds=n_seeds) for axis in AXES}


def _xpositions(axis: dict) -> tuple[list[int], list[str]]:
    """Categorical x positions and tick labels (levels include None)."""
    labels = []
    for lv in axis["levels"]:
        if lv is None:
            labels.append(axis["offlabel"] or "off")
        else:
            labels.append(f"{lv:g}")
    return list(range(len(labels))), labels


def plot(sweeps: dict[str, dict], fig_dir: pathlib.Path | None = None) -> pathlib.Path:
    apply_house_style()
    ncol = 3
    nrow = int(np.ceil(len(AXES) / ncol))
    fig, axarr = plt.subplots(nrow, ncol, figsize=(WIDTH_WIDE * 1.35, 3.4),
                              sharey=True)
    flat = axarr.ravel()

    for i, axis in enumerate(AXES):
        ax = flat[i]
        rec = sweeps[axis["key"]]
        xs, labels = _xpositions(axis)
        med = np.asarray(rec["median"])
        # Log axis: the sweeps span four decades, from a 16x gain on a quiet
        # meter to a hard zero once decimation puts f0 past Nyquist. Exact
        # zeros are clipped onto the floor line, which reads as "the cadence is
        # not observable in this channel at all".
        ax.fill_between(xs, np.clip(rec["q1"], YFLOOR, None),
                        np.clip(rec["q3"], YFLOOR, None),
                        color=C["skyblue"], alpha=0.35, lw=0)
        ax.plot(xs, np.clip(med, YFLOOR, None), color=C["blue"], lw=0.9,
                marker="o", ms=2.4)
        ax.axhline(1.0, color=C["grey"], ls=":", lw=0.6, zorder=0)
        dead = med <= YFLOOR
        if dead.any():
            ax.plot(np.asarray(xs)[dead], np.full(dead.sum(), YFLOOR),
                    ls="none", marker="x", ms=3.2, color=C["vermillion"],
                    zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_xlabel(axis["xlabel"], fontsize=6.5)
        ax.set_ylim(YFLOOR * 0.6, 40.0)
        if i % ncol == 0:
            ax.set_ylabel("cadence SNR\n(rel. nominal)", fontsize=6.5)

    for j in range(len(AXES), len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout()
    return save(fig, "scenario_meter") if fig_dir is None \
        else _save_to(fig, "scenario_meter", fig_dir)


def _save_to(fig, stem: str, fig_dir: pathlib.Path) -> pathlib.Path:
    """Write to an injectable directory (tests), matching plotstyle.save."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(fig_dir / f"{stem}.png")
    plt.close(fig)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS,
                    help="workload realisations per grid point")
    args = ap.parse_args()

    sweeps = build(n_seeds=args.n_seeds)
    for axis in AXES:
        rec = sweeps[axis["key"]]
        pairs = ", ".join(
            f"{'off' if lv is None else format(lv, 'g')}={m:.2f}"
            for lv, m in zip(rec["levels"], rec["median"])
        )
        print(f"{axis['key']:<16} {pairs}")

    out = plot(sweeps)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
