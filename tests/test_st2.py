"""ST2 harness + attack-registry tests — task 20.6.

Tiny-n smokes (n_each=8, 2 levels, 60 s glue) prove every family runs
end-to-end with both stock detectors and returns finite, well-formed fields;
the structural guarantees (injected detectors, negatives scored once per
family and reused across levels, meter family sharing one channel across both
classes) each get a dedicated test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from powerladder.config import DEFAULT, MeterParams
from powerladder.typeb.st2 import St2Point, run_family
from powerladder.typeb.st2_attacks import (
    ATTACKS,
    FAMILY_ORDER,
    attack_families,
    make_negative_population,
    make_positive_population,
)

KO = DEFAULT.ko
# 60 s keeps the smokes fast (the CV/D measures use their own >= 240 s meta
# window, so short scored traces do not starve the diffusion fit).
GLUE = dataclasses.replace(DEFAULT.ko_typeb, duration_s=60.0)

SMALL = dataclasses.replace(
    DEFAULT.st2,
    n_each=8,
    jitter_levels=(0.1, 0.35),
    work_levels=(0.1, 0.35),
    drift_levels=(0.0, 0.4),
    phase_levels=(0.02, 0.1),
    harmonic_levels=(0.25, 0.75),
    shape_levels=(0.5, 0.95),
    relocate_f0s=(0.35, 1.4),
    dilute_shares=(0.35, 0.7),
    meter_variants=DEFAULT.st2.meter_variants[:2],
)


def test_registry_covers_all_families_and_grids():
    fams = attack_families(SMALL)
    assert tuple(fams) == FAMILY_ORDER
    assert fams["jitter"].levels == SMALL.jitter_levels
    assert fams["meter"].levels == ("notch_at_cadence", "integrating_1hz")
    # anchored families point at the b2 measured summaries; the rest are None
    assert fams["work"].cost_anchor == "results/b2/workjitter_summary.json"
    for f in ("phase", "relocate", "harmonic", "dilute", "meter"):
        assert fams[f].cost_anchor is None
    # module-level registry uses the default grids
    assert ATTACKS["jitter"].levels == DEFAULT.st2.jitter_levels


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_every_family_smoke(family):
    """End-to-end with both stock detectors: finite, well-formed St2Points."""
    points = run_family(family, SMALL, KO, GLUE, seed=1)
    fam = attack_families(SMALL)[family]
    assert len(points) == len(fam.levels)
    for pt, level in zip(points, fam.levels):
        assert isinstance(pt, St2Point)
        assert pt.family == family and pt.level == level
        assert np.isfinite(pt.cadence_cv) and pt.cadence_cv >= 0.0
        assert np.isfinite(pt.phase_diffusion_D)
        assert set(pt.auc) == {"spectral", "viterbi"}
        for det in ("spectral", "viterbi"):
            assert 0.0 <= pt.auc[det] <= 1.0
            assert set(pt.tpr_at_far[det]) == {"0.05", "0.01"}
            for v in pt.tpr_at_far[det].values():
                assert 0.0 <= v <= 1.0


def test_detector_injection():
    """A dummy statistic drops into the detector slot (the ST1 plug-in point)."""
    def meanpow(t, p_obs, band_lo, band_hi):
        return float(np.mean(p_obs))

    points = run_family("jitter", SMALL, KO, GLUE,
                        detectors={"meanpow": meanpow}, seed=2)
    for pt in points:
        assert set(pt.auc) == {"meanpow"}
        assert set(pt.tpr_at_far) == {"meanpow"}
        assert np.isfinite(pt.auc["meanpow"])


def test_negatives_scored_once_and_run_deterministic():
    """Negatives are scored once per family and reused across levels; and the
    whole run is a pure function of the seed."""
    calls = []

    def spy(t, p_obs, band_lo, band_hi):
        calls.append(float(p_obs[0]))
        return float(p_obs[0])

    run_family("jitter", SMALL, KO, GLUE, detectors={"spy": spy}, seed=3)
    n, n_levels = SMALL.n_each, len(SMALL.jitter_levels)
    # negatives once (n calls), positives per level (n per level)
    assert len(calls) == n + n_levels * n

    a = run_family("work", SMALL, KO, GLUE, seed=4)
    b = run_family("work", SMALL, KO, GLUE, seed=4)
    assert a == b


def test_meter_negatives_same_seed_identical_device_draws():
    """The meter family's per-level negatives reuse one fixed stream: two
    variants see the SAME device-side traces, only the channel differs."""
    variants = dict(SMALL.meter_variants)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    neg_notch = make_negative_population(
        "meter", 4, KO, GLUE, rng1, meter=variants["notch_at_cadence"])
    neg_noop = make_negative_population(
        "meter", 4, KO, GLUE, rng2, meter=MeterParams())
    # same seed -> same device draws; a pure LTI variant only reshapes them
    for a, b in zip(neg_notch, neg_noop):
        assert a.t.shape == b.t.shape
        assert not np.allclose(a.P_obs, b.P_obs)   # the channel acted
        assert np.isclose(a.P_obs.mean(), b.P_obs.mean(), rtol=0.05)


def test_meter_family_shares_channel_between_classes():
    """Positives and negatives pass through the SAME hostile channel: the
    1 Hz integrating sampler decimates both classes onto the 1 s grid."""
    m = dict(SMALL.meter_variants)["integrating_1hz"]
    rng = np.random.default_rng(11)
    pos = make_positive_population("meter", "integrating_1hz", 2, KO, GLUE,
                                   rng, meter=m)
    neg = make_negative_population("meter", 2, KO, GLUE, rng, meter=m)
    for tr in pos + neg:
        dt = np.diff(tr.t)
        assert np.allclose(dt, 1.0)


def test_dilute_uses_aggregate_builders():
    """Dilute positives sweep the dominant share; the null is the nominal
    inference-dominant aggregate (generated once, level-independent)."""
    rng = np.random.default_rng(13)
    pos = make_positive_population("dilute", 0.35, 2, KO, GLUE, rng,
                                   meter=None)
    neg = make_negative_population("dilute", 2, KO, GLUE, rng, meter=None)
    for tr in pos:
        assert tr.label == "train" and np.isfinite(tr.f0)
    for tr in neg:
        assert tr.label == "infer" and np.isnan(tr.f0)
