"""ko_synth meter wiring tests — ST2 task 20.2.

Two invariants:

1. ``meter=None`` is BYTE-IDENTICAL to the pre-change builders. The reference
   digests in tests/data/st2_prechange_refs.json were captured from the code
   BEFORE the meter kwarg existed (same seeds, same 30 s glue), so this guards
   against any accidental RNG-stream or arithmetic change, not just
   self-consistency of the edited code.

2. Noise ownership: with a meter, the glue's white sigma_eta is NOT applied —
   the meter's own sigma_eta is the single noise source (no double noise).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from powerladder.config import DEFAULT, MeterParams
from powerladder.typeb.ko_synth import ko_make_aggregate_trace, ko_make_trace

KO = DEFAULT.ko
GLUE30 = dataclasses.replace(DEFAULT.ko_typeb, duration_s=30.0)

REFS = json.loads(
    (Path(__file__).parent / "data" / "st2_prechange_refs.json").read_text()
)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    ).hexdigest()


def test_meter_none_is_byte_identical_to_prechange():
    """meter=None reproduces the exact traces captured before the meter kwarg
    existed (seeded sha256 equality; the ST2 byte-identical guard)."""
    tr = ko_make_trace("train", KO, GLUE30, np.random.default_rng(11))
    inf = ko_make_trace("infer", KO, GLUE30, np.random.default_rng(12))
    atr = ko_make_aggregate_trace("train", KO, GLUE30, np.random.default_rng(13))
    ainf = ko_make_aggregate_trace("infer", KO, GLUE30, np.random.default_rng(14))
    assert _digest(tr.P_obs) == REFS["ko_trace_train_seed11"]
    assert _digest(inf.P_obs) == REFS["ko_trace_infer_seed12"]
    assert _digest(atr.P_obs) == REFS["ko_agg_train_seed13"]
    assert _digest(ainf.P_obs) == REFS["ko_agg_infer_seed14"]
    assert tr.f0 == REFS["ko_trace_train_seed11_f0"]


def test_meter_owns_noise_no_double_application():
    """With a meter carrying only sigma_eta, the residual noise matches the
    METER noise level — the glue's sigma_eta is not also applied (which would
    inflate the std by sqrt(2))."""
    sigma = GLUE30.sigma_eta
    # Noiseless reference: a no-op meter zeroes the glue noise, so P_obs is the
    # clean device power r*F + P0 (same seed => same workload draw).
    clean = ko_make_trace("train", KO, GLUE30, np.random.default_rng(21),
                          meter=MeterParams())
    noisy = ko_make_trace("train", KO, GLUE30, np.random.default_rng(21),
                          meter=MeterParams(sigma_eta=sigma))
    resid = noisy.P_obs - clean.P_obs
    assert abs(np.std(resid) - sigma) < 0.15 * sigma          # not sqrt(2)*sigma
    # and the meter=None trace at the same seed carries the same noise LEVEL,
    # so meter-off and meter-on populations are noise-matched by construction.
    legacy = ko_make_trace("train", KO, GLUE30, np.random.default_rng(21))
    resid_legacy = legacy.P_obs - clean.P_obs
    assert abs(np.std(resid_legacy) - sigma) < 0.15 * sigma


def test_meter_grid_propagates_to_typetrace():
    """A decimating meter returns the trace on the METER grid."""
    mp = MeterParams(lp_cutoff_hz=2.0, sample_hz=5.0)
    tr = ko_make_trace("train", KO, GLUE30, np.random.default_rng(31), meter=mp)
    assert tr.t.size == int(np.floor(GLUE30.duration_s * 5.0)) + 1
    assert np.allclose(np.diff(tr.t), 1.0 / 5.0)
    assert tr.P_obs.size == tr.t.size
