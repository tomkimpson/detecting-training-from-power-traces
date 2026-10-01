"""Aggregate workload: where the background lines are, and when they show.

Companion to plot_scenario_traces.py, drafted as an alternative aggregate panel.
The aggregate of Ko et al. eq 11 is one dominant run over N_tr small trainings
and N_ft fine-tunings, each with its own cadence. At the default power shares
(9 : 0.5 : 0.5, so the dominant run takes 90% of peak) each background job gets
1.25% of peak, ~37 dB below the dominant line in power, and its line is buried
under the dominant run's broadband floor. This figure makes that visible, and
shows the case in which the background does matter:

  column 1  aggregate, dominant share 0.9 (default)  -- background buried
  column 2  aggregate, dominant share 0.5            -- background lines emerge
  column 3  null aggregate, dominant share 0.5       -- inference in the
            dominant slot over the SAME background (same draws as column 2),
            so it is not line-free

The background cadences are drawn here (uniform over the Ko bands, as the
generator does) rather than inside aggregate_F, so each one can be marked on
the spectrum. The composition otherwise matches aggregate_F /
aggregate_null_F: dominant slot at w_dom/total of peak, each background member
at (w/total)/n of peak. Share 0.5 is the co-resident control's value.

Everything runs on the nominal channel (20 Hz meter, no filtering or
decimation, sigma_eta = 4 W), as in plot_scenario_traces.py.

Reproduce:
    python scripts/plot_scenario_aggregate.py
Outputs:
    figures/scenario_aggregate.{pdf,png}
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from plot_scenario_traces import (  # noqa: E402
    EXCERPT_S,
    F0_TRAIN_HZ,
    N_BACKGROUND_FT,
    N_BACKGROUND_TR,
    NOMINAL_METER,
    _save_to,
    multitaper_psd,
)
from powerladder.config import DEFAULT  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid, simulate  # noqa: E402
from powerladder.ko_workload import finetune_F, inference_F, training_F  # noqa: E402
from powerladder.noise import WhiteNoise  # noqa: E402
from powerladder.observation import apply_meter  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
SEED = 0
# Dominant-slot share of peak power, w_main / (w_main + w_tr + w_ft). 0.9 is the
# default aggregate_ratio (9 : 0.5 : 0.5); 0.5 is the co-resident control.
SHARE_DEFAULT = 0.9
SHARE_LOW = 0.5

# (key, dominant slot, dominant share, title)
PANELS = (
    ("agg_default", "training", SHARE_DEFAULT, "Aggregate, dominant share 0.9"),
    ("agg_low", "training", SHARE_LOW, "Aggregate, dominant share 0.5"),
    ("null_low", "inference", SHARE_LOW, "Null aggregate, dominant share 0.5"),
)
PANEL_COLOR = {
    "agg_default": C["purple"],
    "agg_low": C["purple"],
    "null_low": C["vermillion"],
}
BG_COLOR = {"tr": C["blue"], "ft": C["green"]}


def draw_background(seed: int) -> dict[str, np.ndarray]:
    """Background cadences, uniform over the Ko bands, one per member.

    Drawn from their own stream so columns 2 and 3 share the same background.
    """
    ko = DEFAULT.ko
    rng = np.random.default_rng([seed, 1])
    return {
        "tr": rng.uniform(ko.f0_lo, ko.f0_hi, N_BACKGROUND_TR),
        "ft": rng.uniform(ko.f1_lo, ko.f1_hi, N_BACKGROUND_FT),
    }


def background_F(t: np.ndarray, cadences: dict[str, np.ndarray], share: float,
                 seed: int) -> np.ndarray:
    """Background FLOP rate at a given dominant share, with pinned cadences.

    The background weights (w_tr, w_ft) keep their default values; the dominant
    weight is set so that it takes ``share`` of the total.
    """
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    _, w_tr, w_ft = ko.aggregate_ratio
    total = (w_tr + w_ft) / (1.0 - share)
    rng = np.random.default_rng([seed, 2])
    F = np.zeros_like(t)
    for f in cadences["tr"]:
        F += training_F(t, ko, rng, f_peak=f_peak * (w_tr / total) / len(cadences["tr"]),
                        f0=float(f), eta_scale=glue.eta_scale)
    for f in cadences["ft"]:
        F += finetune_F(t, ko, rng, f_peak=f_peak * (w_ft / total) / len(cadences["ft"]),
                        f0=float(f), eta_scale=glue.eta_scale)
    return F


def dominant_F(t: np.ndarray, slot: str, share: float, seed: int) -> np.ndarray:
    """Dominant-slot FLOP rate: the training run (pinned f0) or inference."""
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max * share
    rng = np.random.default_rng([seed, 3])
    if slot == "training":
        return training_F(t, ko, rng, f_peak=f_peak, f0=F0_TRAIN_HZ,
                          eta_scale=glue.eta_scale)
    return inference_F(t, ko, rng, f_peak=f_peak, eta_scale=glue.eta_scale)


def make_trace(slot: str, share: float, cadences: dict, seed: int) -> dict:
    """One observed aggregate trace on the nominal channel (as plot_scenario_traces)."""
    glue = DEFAULT.ko_typeb
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    f_flop = dominant_F(t, slot, share, seed) + background_F(t, cadences, share, seed)
    rng = np.random.default_rng([seed, 4])
    spec = TraceSpec(t=t, F=f_flop, r=np.full_like(t, glue.r),
                     P0=np.full_like(t, glue.P0))
    trace = simulate(spec, WhiteNoise(0.0), rng)
    meter = apply_meter(trace.t, trace.P_obs, NOMINAL_METER, rng)
    freqs, psd = multitaper_psd(meter.t, meter.P_meter)
    f0 = F0_TRAIN_HZ if slot == "training" else float("nan")
    return {"t": meter.t, "p": meter.P_meter, "freqs": freqs, "psd": psd, "f0": f0}


def build(seed: int = SEED) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    cadences = draw_background(seed)
    out = {key: make_trace(slot, share, cadences, seed)
           for key, slot, share, _ in PANELS}
    return out, cadences


def plot(scenarios: dict[str, dict], cadences: dict[str, np.ndarray],
         fig_dir: pathlib.Path | None = None) -> pathlib.Path:
    apply_house_style()
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    fig, axes = plt.subplots(2, len(PANELS), figsize=(WIDTH_WIDE * 1.2, 3.4))

    for col, (key, _, _, title) in enumerate(PANELS):
        rec = scenarios[key]
        colour = PANEL_COLOR[key]

        # --- top row: time-domain excerpt --------------------------------------
        ax = axes[0, col]
        win = rec["t"] <= EXCERPT_S
        ax.plot(rec["t"][win], rec["p"][win], color=colour, lw=0.7)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("time [s]")
        ax.set_xlim(0, EXCERPT_S)
        if col == 0:
            ax.set_ylabel("meter power [W]")

        # --- bottom row: spectrum, dominant f0 and background cadences marked --
        ax = axes[1, col]
        freqs, psd = rec["freqs"], rec["psd"]
        show = (freqs >= band_lo) & (freqs <= band_hi)
        ax.semilogy(freqs[show], psd[show], color=colour, lw=0.7)
        for kind, fs in cadences.items():
            for f in fs:
                ax.plot(f, 0.02, marker="^", ms=3, color=BG_COLOR[kind],
                        transform=ax.get_xaxis_transform(), clip_on=False)
        f0 = rec["f0"]
        if np.isfinite(f0):
            ax.axvline(f0, color=C["grey"], ls="--", lw=0.6, zorder=0)
            ax.annotate(r"$f_0$", xy=(f0, 0.94), xycoords=("data", "axes fraction"),
                        ha="left", va="top", fontsize=6, color=C["grey"],
                        xytext=(2, 0), textcoords="offset points")
        else:
            ax.annotate("no dominant line", xy=(0.5, 0.94), xycoords="axes fraction",
                        ha="center", va="top", fontsize=6, color=C["grey"])
        ax.set_xlabel("frequency [Hz]")
        ax.set_xlim(band_lo, band_hi)
        if col == 0:
            ax.set_ylabel(r"multitaper PSD [W$^2$/Hz]")

    lows, highs = zip(*(axes[1, c].get_ylim() for c in range(len(PANELS))))
    for c in range(len(PANELS)):
        axes[1, c].set_ylim(min(lows), max(highs))

    handles = [plt.Line2D([], [], ls="", marker="^", ms=3, color=BG_COLOR[k], label=lab)
               for k, lab in (("tr", "background training"),
                              ("ft", "background fine-tuning"))]
    axes[1, -1].legend(handles=handles, loc="upper right", fontsize=5,
                       frameon=False, bbox_to_anchor=(1.0, 0.88))

    fig.tight_layout()
    return save(fig, "scenario_aggregate", subdir=None) if fig_dir is None \
        else _save_to(fig, "scenario_aggregate", fig_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    scenarios, cadences = build(seed=args.seed)
    print("background training cadences [Hz]:   ",
          np.array2string(np.sort(cadences["tr"]), precision=2))
    print("background fine-tuning cadences [Hz]:",
          np.array2string(np.sort(cadences["ft"]), precision=2))
    band_lo, band_hi = DEFAULT.ko_typeb.band_lo, DEFAULT.ko_typeb.band_hi
    print(f"{'panel':<12} {'in-band power':>14} {'peak/median':>12}")
    for key, *_ in PANELS:
        rec = scenarios[key]
        band = (rec["freqs"] >= band_lo) & (rec["freqs"] <= band_hi)
        in_band = float(np.trapezoid(rec["psd"][band], rec["freqs"][band]))
        ratio = float(rec["psd"][band].max() / np.median(rec["psd"][band]))
        print(f"{key:<12} {in_band:>14.3g} {ratio:>12.1f}")

    out = plot(scenarios, cadences)
    print(f"-> {out.with_suffix('')}.*")


if __name__ == "__main__":
    main()
