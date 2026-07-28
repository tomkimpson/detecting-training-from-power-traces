"""Scenario-model gallery: example generator traces and their spectra.

Paper figure for the scenario-models section: what the four workload classes
of the generator actually look like at the meter, in time and in frequency.
The four classes are the ones every later gate scores against --

  training        Ko et al. eqs 1-5   -- a jittered two-phase iteration cycle
  fine-tuning     Ko et al. eqs 6-10  -- same engine, tail/idle parameter set
  inference null  OURS                -- continuous-batching decode floor, slow
                                         MoE envelope, Poisson prefill bursts;
                                         in-band power but NO coherent line
  aggregate       Ko et al. eq 11     -- a dominant training run superposed on
                                         a background of smaller trainings and
                                         fine-tunings at one meter

The point the figure has to carry is the third row of that list. The inference
null is not "quiet": it puts comparable power in the same 0.3-1.7 Hz search
band as training. What it lacks is a coherent frequency path, which is why a
matched filter sees only a raised floor and a line tracker finds nothing to
follow. Reading the bottom row across, the training and aggregate panels carry
a line at the marked f0 and the inference panel does not.

This is a DESCRIPTIVE figure: it shows the signal the channel delivers, not any
detector's operating point. Detector behaviour is the structural-evidence and
frontier sections.

Everything runs on the nominal channel (20 Hz meter, no filtering or
decimation, sigma_eta = 4 W) so the four panels are comparable; the
observation-map axes are swept separately in plot_scenario_meter.py.

Reproduce:
    python scripts/plot_scenario_traces.py
Outputs:
    figures/scenario_traces.{pdf,png}
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import DEFAULT, MeterParams  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid, simulate  # noqa: E402
from powerladder.ko_workload import (  # noqa: E402
    aggregate_F,
    finetune_F,
    inference_F,
    training_F,
)
from powerladder.noise import WhiteNoise  # noqa: E402
from powerladder.observation import apply_meter  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.st1.multitaper import dpss_eigencoef  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
SEED = 0
# Cadences are pinned rather than drawn so the figure is stable and the marked
# f0 is exact. Both sit mid-band for their respective Ko bands (training
# f0 ~ U(0.5, 1.5); fine-tuning f1 ~ U(0.3, 0.7)). The training value matches
# plot_scenario_meter.py, which avoids 1.0 Hz because it coincides with both a
# Nyquist frequency and a boxcar sinc null on the swept meter grids.
F0_TRAIN_HZ = 0.9
F0_FINETUNE_HZ = 0.5
# Time-domain excerpt: long enough to show ~20 training iterations, short
# enough that individual up/down phases are legible.
EXCERPT_S = 20.0
# Aggregate superposition: Ko eq 11 background multiplicity.
N_BACKGROUND_TR = 4
N_BACKGROUND_FT = 4
# Nominal off-chip channel — no filtering, no decimation, meter noise only.
# Every other observation-map axis is swept in plot_scenario_meter.py.
NOMINAL_METER = MeterParams(sigma_eta=DEFAULT.ko_typeb.sigma_eta)

PANELS = ("training", "finetune", "inference", "aggregate")
PANEL_TITLE = {
    "training": "Training (Ko eqs 1–5)",
    "finetune": "Fine-tuning (Ko eqs 6–10)",
    "inference": "Inference null (ours)",
    "aggregate": "Aggregate (Ko eq 11)",
}
PANEL_COLOR = {
    "training": C["blue"],
    "finetune": C["green"],
    "inference": C["vermillion"],
    "aggregate": C["purple"],
}


def _f_of(kind: str, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """FLOP-rate waveform for one scenario class, and its cadence (NaN if none)."""
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    t_grid = t
    if kind == "training":
        f_flop = training_F(t_grid, ko, rng, f_peak=f_peak, f0=F0_TRAIN_HZ,
                            eta_scale=glue.eta_scale)
        return f_flop, F0_TRAIN_HZ
    if kind == "finetune":
        f_flop = finetune_F(t_grid, ko, rng, f_peak=f_peak, f0=F0_FINETUNE_HZ,
                            eta_scale=glue.eta_scale)
        return f_flop, F0_FINETUNE_HZ
    if kind == "inference":
        # No cadence by construction: the decode floor is band-limited noise and
        # the prefill arrivals are Poisson, so neither carries a line.
        f_flop = inference_F(t_grid, ko, rng, f_peak=f_peak, eta_scale=glue.eta_scale)
        return f_flop, float("nan")
    if kind == "aggregate":
        f_flop = aggregate_F(t_grid, ko, rng, f_peak=f_peak,
                             n_tr=N_BACKGROUND_TR, n_ft=N_BACKGROUND_FT,
                             f0=F0_TRAIN_HZ, eta_scale=glue.eta_scale)
        # f0 applies to the DOMINANT training only; the background carries its
        # own weaker lines drawn independently from the same band.
        return f_flop, F0_TRAIN_HZ
    raise ValueError(f"unknown scenario {kind!r} (expected one of {PANELS})")


def make_trace(kind: str, seed: int = SEED) -> dict:
    """One observed power trace for scenario ``kind`` on the nominal channel.

    Returns ``{t, p, f0}`` with ``p`` in watts on the meter grid. Follows the
    same generator -> device -> meter path as powerladder.typeb.ko_synth, and
    the same noise-ownership rule: the device simulation is noiseless and the
    meter map is the single owner of observation noise.
    """
    glue = DEFAULT.ko_typeb
    rng = np.random.default_rng([seed, PANELS.index(kind)])
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_flop, f0 = _f_of(kind, t, rng)
    spec = TraceSpec(t=t, F=f_flop,
                     r=np.full_like(t, glue.r),
                     P0=np.full_like(t, glue.P0))
    trace = simulate(spec, WhiteNoise(0.0), rng)
    meter = apply_meter(trace.t, trace.P_obs, NOMINAL_METER, rng)
    return {"kind": kind, "t": meter.t, "p": meter.P_meter, "f0": f0}


def multitaper_psd(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thomson multitaper PSD estimate [W^2/Hz] of a demeaned trace.

    Reuses the DPSS eigencoefficients the structural detector already computes
    (powerladder.st1.multitaper), so the spectra shown here are the ones the
    Rung-1 detector sees. With scipy's unit-energy tapers the eigenspectrum is
    |y_k(f)|^2 / fs, and the multitaper estimate averages over tapers.
    """
    fs = 1.0 / float(t[1] - t[0])
    freqs, y, _ = dpss_eigencoef(p, fs, params=DEFAULT.st1)
    psd = np.mean(np.abs(y) ** 2, axis=0) / fs
    return freqs, psd


def build(seed: int = SEED) -> dict[str, dict]:
    """All four scenario traces plus their spectra, keyed by scenario name."""
    out = {}
    for kind in PANELS:
        rec = make_trace(kind, seed=seed)
        rec["freqs"], rec["psd"] = multitaper_psd(rec["t"], rec["p"])
        out[kind] = rec
    return out


def plot(scenarios: dict[str, dict], fig_dir: pathlib.Path | None = None) -> pathlib.Path:
    apply_house_style()
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    fig, axes = plt.subplots(2, len(PANELS), figsize=(WIDTH_WIDE * 1.55, 3.4))

    for col, kind in enumerate(PANELS):
        rec = scenarios[kind]
        colour = PANEL_COLOR[kind]

        # --- top row: time-domain excerpt --------------------------------------
        ax = axes[0, col]
        t, p = rec["t"], rec["p"]
        win = t <= EXCERPT_S
        ax.plot(t[win], p[win], color=colour, lw=0.7)
        ax.set_title(PANEL_TITLE[kind], fontsize=7)
        ax.set_xlabel("time [s]")
        ax.set_xlim(0, EXCERPT_S)
        if col == 0:
            ax.set_ylabel("meter power [W]")

        # --- bottom row: multitaper spectrum over the search band --------------
        ax = axes[1, col]
        freqs, psd = rec["freqs"], rec["psd"]
        show = (freqs >= band_lo) & (freqs <= band_hi)
        ax.semilogy(freqs[show], psd[show], color=colour, lw=0.7)
        f0 = rec["f0"]
        if np.isfinite(f0):
            ax.axvline(f0, color=C["grey"], ls="--", lw=0.6, zorder=0)
            ax.annotate(r"$f_0$", xy=(f0, 0.94), xycoords=("data", "axes fraction"),
                        ha="left", va="top", fontsize=6, color=C["grey"],
                        xytext=(2, 0), textcoords="offset points")
        else:
            ax.annotate("no line", xy=(0.5, 0.94), xycoords="axes fraction",
                        ha="center", va="top", fontsize=6, color=C["grey"])
        ax.set_xlabel("frequency [Hz]")
        ax.set_xlim(band_lo, band_hi)
        if col == 0:
            ax.set_ylabel(r"multitaper PSD [W$^2$/Hz]")

    # Share the spectral y-range so the four panels are directly comparable.
    lows, highs = zip(*(axes[1, c].get_ylim() for c in range(len(PANELS))))
    for c in range(len(PANELS)):
        axes[1, c].set_ylim(min(lows), max(highs))

    fig.tight_layout()
    return save(fig, "scenario_traces", subdir=None) if fig_dir is None \
        else _save_to(fig, "scenario_traces", fig_dir)


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
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    scenarios = build(seed=args.seed)
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    print(f"{'scenario':<12} {'f0 [Hz]':>8} {'in-band power':>14} {'peak/median':>12}")
    for kind in PANELS:
        rec = scenarios[kind]
        freqs, psd = rec["freqs"], rec["psd"]
        band = (freqs >= band_lo) & (freqs <= band_hi)
        in_band = float(np.trapezoid(psd[band], freqs[band]))
        ratio = float(psd[band].max() / np.median(psd[band]))
        print(f"{kind:<12} {rec['f0']:>8.2f} {in_band:>14.3g} {ratio:>12.1f}")

    out = plot(scenarios)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
