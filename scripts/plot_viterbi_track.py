"""How the Viterbi tracker works: its contrast map with the tracked path overlaid.

The Sec. 4.1 explainer figure. Three 300 s traces on the nominal channel, each
shown as the tracker sees it -- the band-restricted, median-normalised
log-contrast map L_kj (paper eq. contrast) on the tracker's own grid (16 s
frames at half overlap, 62.5 mHz bins, band B = [0.3, 1.7] Hz) -- with the best
path of eq. (viterbi) overlaid:

  panel 1  training, stationary cadence (f0_drift_hz = 0)
  panel 2  training, wandering cadence  (f0_drift_hz = WANDER_HZ)
  panel 3  inference null (no line by construction)

On the training panels the true cadence is overlaid for comparison: in each
tracker frame, 1 / (median iteration period) over the iteration starts the
generator laid down (training_F_meta), so it is ground truth free of any
estimation step. The path and the map come from viterbi_best_path, which uses
the same emission map, transition cost and window defaults as
viterbi_statistic, so the path drawn is exactly the one the score V rides.

Everything runs on the nominal channel (20 Hz meter, no filtering or
decimation, sigma_eta = 4 W), as in plot_scenario_traces.py.

Reproduce:
    python scripts/plot_viterbi_track.py [--seed 4]
Outputs:
    figures/viterbi_track.{pdf,png}
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from plot_scenario_traces import F0_TRAIN_HZ, NOMINAL_METER, _save_to  # noqa: E402
from powerladder.config import DEFAULT  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid, simulate  # noqa: E402
from powerladder.ko_workload import inference_F, training_F_meta  # noqa: E402
from powerladder.noise import WhiteNoise  # noqa: E402
from powerladder.observation import apply_meter  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style  # noqa: E402
from powerladder.typeb.detectors import viterbi_best_path, viterbi_statistic  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
# Pinned so the figure is stable. Seed 4 gives a wandering draw that spans most
# of the band without leaving it, so the panel shows tracking rather than a
# path pinned to the band edge.
SEED = 4
# Cadence wander of the middle panel: a point on the drift axis of
# fig:drift_ceiling where the matched filter has already lost most of its power
# (0.21) and the tracker holds 1.00.
WANDER_HZ = 0.4
# Colour range of the contrast map [log-contrast]. The lower clip keeps the
# deep troughs of the log-periodogram from dominating the scale; the upper is
# shared across panels so the null reads as dim against a line.
L_MIN, L_MAX = -2.0, 3.5

PANELS = ("stationary", "wandering", "inference")
PANEL_TITLE = {
    "stationary": "Training, stationary",
    "wandering": f"Training, wander {WANDER_HZ:g} Hz",
    "inference": "Inference null",
}


def _observe(t: np.ndarray, f_flop: np.ndarray, rng: np.random.Generator):
    """Generator -> device -> meter, as plot_scenario_traces.make_trace."""
    glue = DEFAULT.ko_typeb
    spec = TraceSpec(t=t, F=f_flop, r=np.full_like(t, glue.r),
                     P0=np.full_like(t, glue.P0))
    trace = simulate(spec, WhiteNoise(0.0), rng)
    meter = apply_meter(trace.t, trace.P_obs, NOMINAL_METER, rng)
    return meter.t, meter.P_meter


def _true_cadence(starts: np.ndarray, frame_t: np.ndarray, frame_s: float) -> np.ndarray:
    """1 / median iteration period within each tracker frame [Hz]."""
    periods = np.diff(starts)
    mids = 0.5 * (starts[1:] + starts[:-1])
    out = np.full(frame_t.size, np.nan)
    for j, tc in enumerate(frame_t):
        sel = np.abs(mids - tc) <= 0.5 * frame_s
        if sel.any():
            out[j] = 1.0 / np.median(periods[sel])
    return out


def build(seed: int = SEED) -> dict[str, dict]:
    ko, glue = DEFAULT.ko, DEFAULT.ko_typeb
    f_peak = glue.f_peak_frac * DEFAULT.floor.F_max
    t = make_time_grid(glue.duration_s, 1.0 / glue.fs)
    out = {}
    for i, kind in enumerate(PANELS):
        rng = np.random.default_rng([seed, i])
        starts = None
        if kind == "inference":
            f_flop = inference_F(t, ko, rng, f_peak=f_peak, eta_scale=glue.eta_scale)
        else:
            drift = WANDER_HZ if kind == "wandering" else 0.0
            f_flop, starts = training_F_meta(t, ko, rng, f_peak=f_peak, f0=F0_TRAIN_HZ,
                                             eta_scale=glue.eta_scale,
                                             f0_drift_hz=drift)
        tt, p = _observe(t, f_flop, rng)
        freqs, times, logp, path = viterbi_best_path(tt, p, glue.band_lo, glue.band_hi)
        frame_s = 2.0 * float(times[1] - times[0])  # half-overlap: frame = 2 hops
        rec = {"freqs": freqs, "times": times, "L": logp, "path": path,
               "V": viterbi_statistic(tt, p, glue.band_lo, glue.band_hi),
               "truth": None if starts is None else _true_cadence(starts, times, frame_s)}
        out[kind] = rec
    return out


def plot(panels: dict[str, dict], fig_dir: pathlib.Path | None = None) -> pathlib.Path:
    apply_house_style()
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_WIDE, 2.2), sharey=True,
                            constrained_layout=True)
    for ax, kind in zip(axs, PANELS):
        rec = panels[kind]
        im = ax.pcolormesh(rec["times"], rec["freqs"], rec["L"], shading="nearest",
                           cmap="viridis", vmin=L_MIN, vmax=L_MAX, rasterized=True)
        if rec["truth"] is not None:
            ax.plot(rec["times"], rec["truth"], color=C["vermillion"], ls="--",
                    lw=1.0, label="true cadence")
        # Dark outline keeps the white path legible over bright (yellow) bins.
        ax.plot(rec["times"], rec["path"], color="white", lw=1.2,
                label="Viterbi path",
                path_effects=[patheffects.withStroke(linewidth=2.2,
                                                     foreground=C["black"])])
        ax.set_title(PANEL_TITLE[kind], fontsize="small")
        ax.text(0.04, 0.05, f"$V = {rec['V']:.2f}$", transform=ax.transAxes,
                fontsize="small", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))
        ax.set_xlabel("Time [s]")
    axs[0].set_ylabel("Frequency [Hz]")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               fontsize="small", frameon=False)
    cb = fig.colorbar(im, ax=axs, pad=0.02, fraction=0.04)
    cb.set_label(r"$L_{kj}$")
    if fig_dir is not None:
        return _save_to(fig, "viterbi_track", fig_dir)
    return _save_to(fig, "viterbi_track",
                    pathlib.Path(__file__).resolve().parent.parent / "figures")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    panels = build(seed=args.seed)
    print(f"{'panel':<11} {'V':>6} {'frames':>7} {'bins':>5} {'max |path-truth| [Hz]':>22}")
    for kind in PANELS:
        rec = panels[kind]
        err = ("" if rec["truth"] is None
               else f"{np.nanmax(np.abs(rec['path'] - rec['truth'])):.3f}")
        print(f"{kind:<11} {rec['V']:>6.2f} {rec['times'].size:>7} "
              f"{rec['freqs'].size:>5} {err:>22}")
    print(plot(panels))


if __name__ == "__main__":
    main()
