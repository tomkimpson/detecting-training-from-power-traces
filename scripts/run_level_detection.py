"""Run-level detection: how likely is a concealing operator to be caught over a run?

Every detection rate in paper Sec. 6 is per 300 s window. A verifier watches a
whole run, so the operator must evade every window. This script simulates whole
runs and reports P(detected) against observation time (paper Sec. 7.1).

Each run is ONE continuous aggregate trace (Sec. 2 eq-11: dominant training over
an honest small-training + fine-tune background, ``ko_make_aggregate_trace``)
lasting ``RUN_HOURS``, cut into consecutive ``WINDOW_S`` windows and scored by
the Viterbi tracker. The run's cadence and background are drawn once and held
for the whole trace, so windows of one run are NOT independent draws -- in
particular a null facility whose background happens to look training-like
raises correlated false alarms. That is the point: it tests the independence
assumption the counting rule makes, instead of building it in.

Window decision: conformal p-value of the window score against ``N_CAL``
independent 300 s aggregate-null windows (the Sec. 6 null), detected iff
p <= ``WINDOW_FAR`` (the paper's false-alarm rate 1e-2).

Two run-level rules at run false-alarm rate ``RUN_FAR``:

  count     Fixed-horizon. After n windows, alarm iff the number of detected
            windows k satisfies P(Binom(n, WINDOW_FAR) >= k) <= RUN_FAR. Valid
            only if null windows are independent; the null runs measure how far
            that holds. Each observation time is a separate fixed-n test.
  eprocess  Anytime-valid. Running product of p-to-e calibrated window
            p-values (``powerladder.typeb.sequential``), alarm at the first
            crossing of 1/RUN_FAR (Ville). The verifier may stop whenever it
            likes. Also assumes independent window p-values under the null;
            measured the same way.

Arms: honest training, the Sec. 6 work variation at sigma_G = 0.7, and the
mean-preserving lognormal work variation at s = 1.0 (the E1 attacker, which
hides more per window); plus ``N_NULL_RUNS`` inference-dominant null runs for
the empirical run-level false-alarm rate.

Reproduce:
    python scripts/run_level_detection.py              # full run, ~20 min on 8 cores
    python scripts/run_level_detection.py --smoke      # plumbing check, < 1 min
Outputs:
    results/st2/run_level_summary.json                 (tracked)
    results/st2/run_level_raw.npz                      (per-window scores; ignored)
    figures/run_level_detection.{pdf,png}
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from powerladder.config import DEFAULT  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.typeb.detectors import viterbi_statistic  # noqa: E402
from powerladder.typeb.ko_synth import ko_make_aggregate_trace  # noqa: E402
from powerladder.typeb.sequential import (conformal_p,  # noqa: E402
                                          first_crossing, wealth_process)
from powerladder.typeb.st2 import _effective_meter  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
WINDOW_S = 300.0            # one scored window (as Sec. 6)
RUN_HOURS = 24.0            # simulated observation of each run
CHECK_HOURS = (1, 2, 3, 6, 12, 24)
WINDOW_FAR = 1e-2           # per-window false-alarm rate (paper operating point)
RUN_FAR = 1e-3              # run-level false-alarm rate
KAPPA = 0.5                 # p-to-e calibrator (sequential.py default; fixed a priori)
N_CAL = 5000                # independent null windows for the conformal p-values
N_RUNS = 200                # runs per training arm
N_NULL_RUNS = 2000          # null runs for the empirical run-level FAR
SEED = 20261002

ARMS = {                    # name -> dominant-training attack knobs
    "honest": {},
    "work0.7": {"work_sigma": 0.7},
    "lognorm1.0": {"work_lognorm_s": 1.0},
}
NULL = "null"
_STREAM = {name: i for i, name in enumerate((*ARMS, NULL, "cal"))}

_RESULTS = _REPO / "results" / "st2"
_LABEL = {"honest": "honest training", "work0.7": r"work variation $\sigma_G=0.7$",
          "lognorm1.0": r"lognormal work $s=1.0$", NULL: "inference null"}
_COLOR = {"honest": C["black"], "work0.7": C["blue"], "lognorm1.0": C["vermillion"],
          NULL: C["grey"]}


def _glue(duration_s: float):
    return dataclasses.replace(DEFAULT.ko_typeb, duration_s=duration_s)


def _score_windows(t: np.ndarray, p: np.ndarray, n_win: int) -> np.ndarray:
    g = DEFAULT.ko_typeb
    m = int(round(WINDOW_S * g.fs))
    return np.array([viterbi_statistic(t[i * m:(i + 1) * m] - t[i * m],
                                       p[i * m:(i + 1) * m], g.band_lo, g.band_hi)
                     for i in range(n_win)])


def _one_run(job: tuple[str, int, float]) -> tuple[float, np.ndarray]:
    """(dominant f0, per-window Viterbi scores) for one continuous run."""
    arm, idx, hours = job
    rng = np.random.default_rng([SEED, _STREAM[arm], idx])
    n_win = int(round(hours * 3600 / WINDOW_S))
    glue = _glue(n_win * WINDOW_S)
    meter = _effective_meter(DEFAULT.st2)
    if arm == NULL:
        tr = ko_make_aggregate_trace("infer", DEFAULT.ko, glue, rng, meter=meter)
    else:
        tr = ko_make_aggregate_trace("train", DEFAULT.ko, glue, rng, meter=meter,
                                     **ARMS[arm])
    return float(tr.f0), _score_windows(tr.t, tr.P_obs, n_win)


def _cal_window(idx: int) -> float:
    """One independent 300 s aggregate-null window (the Sec. 6 null)."""
    rng = np.random.default_rng([SEED, _STREAM["cal"], idx])
    tr = ko_make_aggregate_trace("infer", DEFAULT.ko, _glue(WINDOW_S), rng,
                                 meter=_effective_meter(DEFAULT.st2))
    return float(_score_windows(tr.t, tr.P_obs, 1)[0])


def count_threshold(n: int) -> int:
    """Smallest k with P(Binom(n, WINDOW_FAR) >= k) <= RUN_FAR."""
    k = np.arange(n + 2)
    return int(k[binom.sf(k - 1, n, WINDOW_FAR) <= RUN_FAR][0])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def analyse(scores: dict[str, np.ndarray], f0s: dict[str, np.ndarray],
            cal: np.ndarray) -> dict:
    """Window rates, and P(alarm by T) under both rules, per arm."""
    out: dict = {}
    for arm, s in scores.items():
        p = conformal_p(s.ravel(), cal).reshape(s.shape)
        det = p <= WINDOW_FAR
        rate = det.mean(axis=1)                         # per-run window rate
        with np.errstate(over="ignore"):    # wealth -> inf still crosses 1/RUN_FAR
            cross = [first_crossing(wealth_process(row, kappa=KAPPA), RUN_FAR)
                     for row in p]
        cross_win = np.array([np.inf if c is None else c + 1 for c in cross])
        curve = {}
        for h in CHECK_HOURS:
            n = int(round(h * 3600 / WINDOW_S))
            if n > s.shape[1]:
                continue
            k_star = count_threshold(n)
            hits_c = int((det[:, :n].sum(axis=1) >= k_star).sum())
            hits_e = int((cross_win <= n).sum())
            curve[f"{h:g}"] = {
                "n_windows": n, "count_threshold": k_star,
                "count": hits_c / len(s), "count_ci": _wilson(hits_c, len(s)),
                "eprocess": hits_e / len(s), "eprocess_ci": _wilson(hits_e, len(s)),
            }
        out[arm] = {
            "n_runs": int(len(s)),
            "window_detection_rate": float(det.mean()),
            "per_run_window_rate_quantiles": {
                f"{q:g}": float(np.quantile(rate, q)) for q in (0, .05, .25, .5, .75, 1)},
            "per_run_window_rate": rate.round(4).tolist(),
            "f0": [None if np.isnan(f) else round(float(f), 4) for f in f0s[arm]],
            "p_detected_by_hours": curve,
        }
    return out


def plot(summary: dict, stem: str) -> pathlib.Path:
    apply_house_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.3))
    arms = summary["arms"]
    for arm in ARMS:
        f0 = np.array(arms[arm]["f0"], dtype=float)
        a.scatter(f0, arms[arm]["per_run_window_rate"], s=4, color=_COLOR[arm],
                  label=_LABEL[arm], alpha=0.7, lw=0)
    a.axhline(WINDOW_FAR, color=C["grey"], lw=0.8, ls=":")
    a.set_xlabel(r"run cadence $f_0$ [Hz]")
    a.set_ylabel("fraction of windows detected")
    a.set_ylim(-0.02, 1.02)
    a.legend(loc="lower left", fontsize=6, frameon=False)
    for arm in (*ARMS, NULL):
        c = arms[arm]["p_detected_by_hours"]
        hrs = [float(h) for h in c]
        for rule, ls in (("count", "-"), ("eprocess", "--")):
            b.plot(hrs, [c[h][rule] for h in c], ls=ls, marker="o", ms=2,
                   color=_COLOR[arm], lw=1)
    b.axhline(RUN_FAR, color=C["grey"], lw=0.8, ls=":")
    b.set_xscale("log")
    b.set_yscale("symlog", linthresh=1e-3)
    b.set_xticks(CHECK_HOURS, [f"{h:g}" for h in CHECK_HOURS])
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("observation time [h]")
    b.set_ylabel("P(run flagged)")
    b.plot([], [], "k-", lw=1, label="count rule")
    b.plot([], [], "k--", lw=1, label="e-process")
    b.legend(loc="center right", fontsize=6, frameon=False)
    for ax, lab in ((a, "a"), (b, "b")):
        ax.text(-0.18, 1.02, lab, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    return save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from the existing summary JSON")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    if args.replot:
        summary = json.loads((_RESULTS / f"run_level{tag}_summary.json").read_text())
        print("wrote", plot(summary, f"run_level_detection{tag}"))
        return

    n_runs, n_null, n_cal, hours = ((4, 8, 200, 2.0) if args.smoke
                                    else (N_RUNS, N_NULL_RUNS, N_CAL, RUN_HOURS))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        cal = np.array(list(ex.map(_cal_window, range(n_cal), chunksize=50)))
        scores, f0s = {}, {}
        for arm, n in ((*((a, n_runs) for a in ARMS), (NULL, n_null))):
            res = list(ex.map(_one_run, [(arm, i, hours) for i in range(n)],
                              chunksize=4))
            f0s[arm] = np.array([r[0] for r in res])
            scores[arm] = np.stack([r[1] for r in res])
            print(f"{arm}: {n} runs done at {time.time() - t0:.0f}s", flush=True)

    summary = {
        "description": "Run-level detection of the Sec. 2 aggregate: P(run flagged) "
                       "vs observation time under a fixed-n count rule and an "
                       "anytime-valid e-process (paper Sec. 7.1).",
        "params": {"window_s": WINDOW_S, "run_hours": hours,
                   "window_far": WINDOW_FAR, "run_far": RUN_FAR, "kappa": KAPPA,
                   "n_cal": n_cal, "n_runs": n_runs, "n_null_runs": n_null,
                   "seed": SEED, "arms": ARMS, "detector": "viterbi",
                   "scenario": "aggregate", "smoke": args.smoke},
        "arms": analyse(scores, f0s, cal),
        "runtime_s": round(time.time() - t0, 1),
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / f"run_level{tag}_summary.json").write_text(
        json.dumps(summary, indent=1))
    np.savez_compressed(_RESULTS / f"run_level{tag}_raw.npz", cal=cal,
                        **{f"scores_{k}": v.astype(np.float32) for k, v in scores.items()},
                        **{f"f0_{k}": v for k, v in f0s.items()})
    print("wrote", plot(summary, f"run_level_detection{tag}"))
    for arm, r in summary["arms"].items():
        c = r["p_detected_by_hours"]
        print(f"{arm:11s} window rate {r['window_detection_rate']:.3f} | "
              + " ".join(f"{h}h c={v['count']:.3f} e={v['eprocess']:.3f}"
                         for h, v in c.items()))


if __name__ == "__main__":
    main()
