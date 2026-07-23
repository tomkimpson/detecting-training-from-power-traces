"""Meter-requirement boundary sweep: observation channel -> detector power.

Phase 2 (plan §2; spec.md "minimum meter specification"). ST2 established a
*two-point* meter finding — the ``integrating_1hz`` channel (1 s trailing boxcar
+ 1 Hz ZOH sampler) defeats every detector on UNMODIFIED honest training. This
module generalises that to a dense grid of :class:`code.config.MeterParams`
channels, run on honest training only, to locate where each detector class dies
as the channel degrades between the nominal 20 Hz meter and the 1 Hz integrating
sampler.

It is the ST2 ``meter`` family (:mod:`code.typeb.st2_attacks`) with a 2-D grid in
place of four named variants, reusing every scoring primitive unchanged:

    positives  make_positive_population("meter", None, ..., meter=mp)
               (nominal honest training pushed through the channel ``mp``)
    negatives  make_negative_population("meter", ..., meter=mp)
               (the hard inference null through the SAME channel — both classes
               must share the observation channel or the gate measures the
               channel, not the workload)
    score      code.typeb.gate.score_population + code.typeb.roc.{auc,tpr_at_far}

Physical CV/D measures are deliberately absent: the meter family does not touch
timing, so the sweep's axes are channel parameters, not de-periodicisation.

Cells are independent, crc-seeded (:func:`code.scripts.st1_far` idiom), so the
sweep runs as a Slurm array (one cell per task) with no cross-cell coupling.
"""

from __future__ import annotations

import zlib
from functools import partial

import numpy as np

from ..config import DEFAULT, KoTypeBParams, KoWorkloadParams, MeterParams, St2MeterBoundaryParams
from .gate import score_population
from .roc import auc, tpr_at_far
from .st2_attacks import make_negative_population, make_positive_population

# The two detector classes the boundary is reported against (the frontier's
# grouping, code.scripts.plot_st2_frontier): tracking methods follow the
# wandering line; fixed methods assume a stationary cadence.
TRACKING = ("viterbi", "dg_order_full", "dg_order_semicoh")
FIXED = ("spectral", "mtf")


def full_detector_set() -> dict:
    """The 5-detector 'full' set (== plot_st2_sweeps.detector_set("full")).

    The pre-ST1 pair (spectral matched filter, Viterbi line tracker) plus the
    ST1-gate detectors that survived calibration: the fixed multitaper F
    comparator and the tracked order/cyclostationary statistics (pooled and
    semi-coherent). Assembled from library modules only (no scripts import).
    """
    from ..st1.pipeline import ST1_DETECTORS, stage4_semicoherent
    from .gate import DETECTORS

    dets = dict(DETECTORS)
    dets["mtf"] = ST1_DETECTORS["mtf"]
    dets["dg_order_full"] = ST1_DETECTORS["dg_order_full"]
    dets["dg_order_semicoh"] = partial(stage4_semicoherent, params=DEFAULT.st1)
    return dets


def meter_grid(
    p: St2MeterBoundaryParams,
) -> list[tuple[str, MeterParams]]:
    """The flat cell list: ``(cell_name, MeterParams)`` per observation channel.

    Two orthogonal sweeps share the list:

    - the MAIN 2-D grid ``sample_hz_grid`` x ``integ_window_grid`` (no notch);
      the ``(20 Hz, 0 s)`` corner is the honest reference channel where every
      detector should sit at full power.
    - the NOTCH sub-sweep ``notch_hz_grid`` x ``notch_depth_grid`` at the
      nominal 20 Hz sampler / no integration, fixed ``notch_q``.

    Every channel carries ``sigma_eta`` W of meter noise (noise ownership lives
    in the meter, code.typeb.ko_synth._observe). Cell names are stable and
    unique so the crc seed and the per-cell artefact are reproducible.
    """
    cells: list[tuple[str, MeterParams]] = []
    for fs in p.sample_hz_grid:
        for iw in p.integ_window_grid:
            name = f"main_fs{fs:g}_iw{iw:g}"
            cells.append((name, MeterParams(
                sample_hz=fs, integ_window_s=iw, sigma_eta=p.sigma_eta)))
    for fn in p.notch_hz_grid:
        for depth in p.notch_depth_grid:
            name = f"notch_f{fn:g}_d{depth:g}"
            cells.append((name, MeterParams(
                sample_hz=20.0, integ_window_s=0.0,
                notch_hz=fn, notch_q=p.notch_q, notch_depth=depth,
                sigma_eta=p.sigma_eta)))
    return cells


def run_meter_cell(
    cell_name: str,
    mp: MeterParams,
    p: St2MeterBoundaryParams,
    ko_params: KoWorkloadParams,
    glue: KoTypeBParams,
    *,
    detectors: dict | None = None,
    base_seed: int | None = None,
) -> dict:
    """Score honest-training-vs-inference through one channel ``mp``.

    Reproducible off ``base_seed`` (or ``p.seed``): a crc of ``cell_name``
    derives the entropy so cells are order-independent and array-safe (the
    st1_far idiom). Negatives draw ``[seed, crc]``, positives ``[seed, crc, 1]``
    — distinct streams, both nominal HONEST training / inference (the attack is
    entirely in the channel).

    Returns the cell record: the channel parameters plus, per detector, AUC and
    TPR at each ``p.target_fars``.
    """
    detectors = full_detector_set() if detectors is None else detectors
    base = p.seed if base_seed is None else base_seed
    crc = zlib.crc32(cell_name.encode())

    rng_neg = np.random.default_rng([base, crc])
    negs = make_negative_population("meter", p.n_each, ko_params, glue,
                                    rng_neg, meter=mp)
    rng_pos = np.random.default_rng([base, crc, 1])
    pos = make_positive_population("meter", None, p.n_each, ko_params, glue,
                                   rng_pos, meter=mp)

    auc_d: dict[str, float] = {}
    tpr_d: dict[str, dict[str, float]] = {}
    for name, fn in detectors.items():
        pos_s = score_population(pos, fn, glue)
        neg_s = score_population(negs, fn, glue)
        auc_d[name] = auc(pos_s, neg_s)
        tpr_d[name] = {f"{far:g}": tpr_at_far(pos_s, neg_s, far)
                       for far in p.target_fars}

    return {
        "cell": cell_name,
        "sample_hz": mp.sample_hz,
        "integ_window_s": mp.integ_window_s,
        "notch_hz": mp.notch_hz,
        "notch_q": mp.notch_q,
        "notch_depth": mp.notch_depth,
        "sigma_eta": mp.sigma_eta,
        "auc": auc_d,
        "tpr_at_far": tpr_d,
    }
