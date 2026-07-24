"""Rung-2 decision rules + evaluation harness (plan §3.3).

Three decision rules on ONE physics feature vector (:mod:`code.typeb.
rung2_features`), so the reader sees where the training-vs-inference
discrimination comes from:

    (i)  physics_score   — PRESPECIFIED, no label fitting. Each feature is
         oriented larger = more training-like, robustly standardised by the
         STATED inference null's median/MAD, and summed with pre-registered
         weights (unit, except the fixed-cadence comparator ``dg_fixed`` which
         is pre-registered OUT on physics grounds: the declared cadence is drawn
         from a band, so a single-frequency comparator is not part of the
         a-priori score — it is kept as a reported feature the fitted rules may
         still use).
    (ii) fitted_discriminant — a regularised logistic discriminant FITTED to the
         scenario generator (physics vector -> train/infer). Cross-validated OOF
         within the stated population; fit-once / zero-shot-predict for transfer
         and the semantic controls.
    (iii)learned_reference — the flexible RandomForest baseline
         (:mod:`code.typeb.rf_baseline`, Rahman shape features), reused as-is. It
         exposes how much discrimination comes from the null choice rather than
         the physics method.

Reporting (plan §3.3, kept separate): per rule, threshold-free AUC and the
FPR/FNR at each ``target_far`` operating point on the STATED workload population
(training vs the inference null through the meter). Transfer / domain-shift and
the semantic falsification controls reuse the same rules via the
fit-nominal / apply-elsewhere protocol (:func:`run_rung2_cell`).

Every rule returns one score per trace (larger = more training-like), feeding
:mod:`code.typeb.roc` exactly like the Rung-1 detectors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..config import (DEFAULT, KoTypeBParams, KoWorkloadParams, MeterParams,
                      RfBaselineParams, Rung2Params, St1DetectorParams)
from .ko_synth import ko_make_population, ko_make_trace
from .rf_baseline import rf_fit_predict, rf_oof_scores
from .roc import auc
from .rung2_features import FEATURE_NAMES, feature_matrix
from .rung2_scenarios import CONTROL_ORDER, CONTROLS, make_control_population

# The fixed-cadence comparator is pre-registered OUT of the prespecified physics
# score (see module docstring): larger IS more training-like in principle, but a
# single declared cadence is not knowable a priori when f0 is drawn from a band.
FIXED_COMPARATOR_FEATURES: tuple[str, ...] = ("dg_fixed",)

# The three decision-rule names, in report order.
RULES: tuple[str, ...] = ("physics_score", "fitted_discriminant", "learned_reference")


# ---------------------------------------------------------------------------
# Rule (i): prespecified physics score
# ---------------------------------------------------------------------------

def physics_weights(names, override=None) -> np.ndarray:
    """Pre-registered weights for the physics score, in ``names`` order.

    ``override`` (Rung2Params.physics_weights) wins if given; otherwise unit
    weights with the fixed-cadence comparator zeroed.
    """
    if override is not None:
        w = np.asarray(override, dtype=float)
        if w.size != len(names):
            raise ValueError("physics_weights length != number of features")
        return w
    return np.array([0.0 if n in FIXED_COMPARATOR_FEATURES else 1.0
                     for n in names], dtype=float)


def _null_location_scale(ref_null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-feature (median, MAD-scale) of the stated inference null."""
    med = np.median(ref_null, axis=0)
    mad = np.median(np.abs(ref_null - med), axis=0)
    scale = np.where(mad > 0.0, mad, 1.0) * 1.4826    # ~ Gaussian sigma
    return med, scale


def physics_score(
    X: np.ndarray,
    ref_null: np.ndarray,
    names,
    *,
    weights=None,
) -> np.ndarray:
    """Prespecified score: sum of null-standardised, oriented features.

    ``ref_null`` is the STATED inference-null feature matrix that sets the
    location/scale (so the rule is calibrated on the stated null and then applied
    unchanged to any population). No label fitting.
    """
    med, scale = _null_location_scale(ref_null)
    w = physics_weights(names, weights)
    Z = (X - med) / scale
    return (Z * w).sum(axis=1)


# ---------------------------------------------------------------------------
# Rule (ii): fitted logistic discriminant on the physics vector
# ---------------------------------------------------------------------------

def _make_logistic(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )


def fitted_discriminant_oof(
    X: np.ndarray, y: np.ndarray, *, n_folds: int, seed: int
) -> np.ndarray:
    """StratifiedKFold OOF decision scores (one physics vector per trace).

    One feature vector per trace ⇒ no within-trace window leakage, so plain
    StratifiedKFold is sufficient (unlike the RF baseline's grouped CV).
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    n_splits = int(min(n_folds, np.bincount(y).min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(y.size, np.nan)
    for tr, te in skf.split(X, y):
        clf = _make_logistic(seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def fitted_discriminant_fit(X: np.ndarray, y: np.ndarray, *, seed: int):
    """Fit the logistic discriminant on the whole nominal population (for transfer)."""
    clf = _make_logistic(seed)
    clf.fit(X, np.asarray(y).astype(int))
    return clf


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def threshold_at_far(neg: np.ndarray, far: float) -> float:
    """Operating threshold: the fixed empirical-FAR quantile of the null scores.

    ``score >= thr`` calls "training". Uses the conservative 'higher' quantile
    (matching scripts/plot_st1_bakeoff.py) so the realised FPR does not exceed
    ``far`` by rounding.
    """
    return float(np.quantile(neg, 1.0 - far, method="higher"))


def fpr_fnr(pos: np.ndarray, neg: np.ndarray, thr: float) -> tuple[float, float]:
    """(FPR, FNR) at threshold ``thr``: FPR = neg>=thr, FNR = pos<thr."""
    fpr = float(np.mean(np.asarray(neg) >= thr))
    fnr = float(np.mean(np.asarray(pos) < thr))
    return fpr, fnr


def fraction_training(scores: np.ndarray, thr: float) -> float:
    """Fraction of a population scored as training (score >= thr) — for controls."""
    return float(np.mean(np.asarray(scores) >= thr))


# ---------------------------------------------------------------------------
# rule bank: fit on the stated population, score any feature matrix
# ---------------------------------------------------------------------------

@dataclass
class RuleBank:
    """The three rules calibrated on the stated (nominal) population.

    Holds the physics-score null reference + weights, the fitted logistic
    pipeline, and the nominal RF training set (features are re-extracted inside
    the RF baseline, so the bank keeps the traces + labels for zero-shot RF).
    """

    names: list[str]
    ref_null: np.ndarray                 # stated inference-null feature matrix
    weights: np.ndarray
    logistic: object
    rf_train_traces: list
    rf_train_labels: np.ndarray
    rf_params: RfBaselineParams
    rf_fs: float

    def score_physics(self, X: np.ndarray) -> np.ndarray:
        return physics_score(X, self.ref_null, self.names, weights=self.weights)

    def score_fitted(self, X: np.ndarray) -> np.ndarray:
        return self.logistic.decision_function(X)

    def score_rf(self, test_traces) -> np.ndarray:
        return rf_fit_predict(
            self.rf_train_traces, self.rf_train_labels, test_traces,
            self.rf_params, fs=self.rf_fs,
            window_s=self.rf_params.window_s, stride_s=self.rf_params.stride_s,
            drop_level=True)


def fit_rule_bank(
    train_traces,
    infer_traces,
    glue: KoTypeBParams,
    p: Rung2Params,
    *,
    st1: St1DetectorParams = DEFAULT.st1,
    rf_params: RfBaselineParams = DEFAULT.rf,
    seed: int,
) -> tuple[RuleBank, np.ndarray, np.ndarray]:
    """Calibrate all three rules on the stated (nominal) population.

    Returns ``(bank, X_train, X_infer)`` — the fitted bank plus the nominal
    physics feature matrices (reused by the stated-cell OOF path so features are
    extracted once).
    """
    X_train, names = feature_matrix(train_traces, glue, params=st1)
    X_infer, _ = feature_matrix(infer_traces, glue, params=st1)
    X = np.vstack([X_train, X_infer])
    y = np.concatenate([np.ones(len(train_traces)), np.zeros(len(infer_traces))])

    weights = physics_weights(names, p.physics_weights)
    logistic = fitted_discriminant_fit(X, y, seed=seed)
    rf_p = dataclasses.replace(rf_params, n_splits=p.n_folds)
    bank = RuleBank(
        names=names, ref_null=X_infer, weights=weights, logistic=logistic,
        rf_train_traces=list(train_traces) + list(infer_traces),
        rf_train_labels=y, rf_params=rf_p, rf_fs=glue.fs)
    return bank, X_train, X_infer


def _report_block(pos: np.ndarray, neg: np.ndarray,
                  target_fars) -> dict:
    """AUC + FPR/FNR at each target FAR for one rule's (pos, neg) scores."""
    out = {"auc": auc(pos, neg), "fpr": {}, "fnr": {}}
    for far in target_fars:
        thr = threshold_at_far(neg, far)
        fpr, fnr = fpr_fnr(pos, neg, thr)
        out["fpr"][f"{far:g}"] = fpr
        out["fnr"][f"{far:g}"] = fnr
    return out


def run_stated_cell(
    train_traces,
    infer_traces,
    glue: KoTypeBParams,
    p: Rung2Params,
    *,
    st1: St1DetectorParams = DEFAULT.st1,
    rf_params: RfBaselineParams = DEFAULT.rf,
    seed: int,
) -> dict:
    """Rung-2 FPR/FNR under the stated workload population, all three rules.

    Cross-validated / OOF so the stated-population numbers are not optimistic:
    physics_score is prespecified (no fit); the fitted discriminant uses
    StratifiedKFold OOF; the RF uses its grouped OOF.
    """
    X_train, names = feature_matrix(train_traces, glue, params=st1)
    X_infer, _ = feature_matrix(infer_traces, glue, params=st1)
    X = np.vstack([X_train, X_infer])
    y = np.concatenate([np.ones(len(train_traces)), np.zeros(len(infer_traces))])
    weights = physics_weights(names, p.physics_weights)

    # (i) prespecified — score directly, null-reference = the inference matrix.
    ps_pos = physics_score(X_train, X_infer, names, weights=weights)
    ps_neg = physics_score(X_infer, X_infer, names, weights=weights)

    # (ii) fitted discriminant OOF.
    oof = fitted_discriminant_oof(X, y, n_folds=p.n_folds, seed=seed)
    fd_pos, fd_neg = oof[y == 1], oof[y == 0]

    # (iii) RF learned reference OOF (grouped by trace).
    rf_p = dataclasses.replace(rf_params, n_splits=p.n_folds)
    rf = rf_oof_scores(
        list(train_traces) + list(infer_traces), y, rf_p, fs=glue.fs,
        window_s=rf_p.window_s, stride_s=rf_p.stride_s, drop_level=True,
        seed=seed)

    return {
        "physics_score": _report_block(ps_pos, ps_neg, p.target_fars),
        "fitted_discriminant": _report_block(fd_pos, fd_neg, p.target_fars),
        "learned_reference": _report_block(rf.pos, rf.neg, p.target_fars),
    }


# ---------------------------------------------------------------------------
# transfer / domain shift + semantic controls (fit nominal, apply elsewhere)
# ---------------------------------------------------------------------------

# transfer-shift parameter keys the harness understands.
_GLUE_KEYS = ("f_peak_frac", "duration_s", "eta_scale", "r", "P0")
_KO_KEYS = ("f0_lo", "f0_hi", "sigma_jitter")


def resolve_shift(
    pairs,
    glue: KoTypeBParams,
    ko: KoWorkloadParams,
    meter_variants,
) -> tuple[KoTypeBParams, KoWorkloadParams, MeterParams | None]:
    """Apply a transfer shift's ``((param, value), ...)`` to (glue, ko, meter).

    ``meter_variant`` resolves against ``meter_variants`` (== St2Params.
    meter_variants). Other keys override the KoTypeBParams glue or the
    KoWorkloadParams band. Returns the shifted (glue, ko, meter-or-None).
    """
    glue_over: dict = {}
    ko_over: dict = {}
    meter: MeterParams | None = None
    for k, v in pairs:
        if k in _GLUE_KEYS:
            glue_over[k] = float(v)
        elif k in _KO_KEYS:
            ko_over[k] = float(v)
        elif k == "meter_variant":
            meter = dict(meter_variants)[v]
        else:
            raise ValueError(f"unknown transfer-shift key {k!r}")
    glue2 = dataclasses.replace(glue, **glue_over) if glue_over else glue
    ko2 = dataclasses.replace(ko, **ko_over) if ko_over else ko
    return glue2, ko2, meter


def _stated_meter(p: Rung2Params) -> MeterParams | None:
    """The nominal observation channel (no-op MeterParams -> None, the fast path)."""
    return None if p.meter == MeterParams() else p.meter


def _score_pos_neg(bank: RuleBank, glue: KoTypeBParams, st1,
                   pos_traces, neg_traces, p: Rung2Params) -> dict:
    """Per-rule AUC + FPR/FNR of a (positive, negative) pair scored zero-shot."""
    Xp, _ = feature_matrix(pos_traces, glue, params=st1)
    Xn, _ = feature_matrix(neg_traces, glue, params=st1)
    return {
        "physics_score": _report_block(
            bank.score_physics(Xp), bank.score_physics(Xn), p.target_fars),
        "fitted_discriminant": _report_block(
            bank.score_fitted(Xp), bank.score_fitted(Xn), p.target_fars),
        "learned_reference": _report_block(
            bank.score_rf(pos_traces), bank.score_rf(neg_traces), p.target_fars),
    }


def _score_control(bank: RuleBank, glue: KoTypeBParams, st1,
                   control_traces, ref_null_traces, p: Rung2Params) -> dict:
    """Per-rule AUC(control vs null) + fraction-scored-as-training at each FAR.

    ``ref_null_traces`` is a HELD-OUT nominal inference-null sample (never in the
    RF fit set) so the operating threshold and the RF null scores are honest.
    """
    Xc, _ = feature_matrix(control_traces, glue, params=st1)
    Xn, _ = feature_matrix(ref_null_traces, glue, params=st1)
    out: dict = {}
    scorers = {
        "physics_score": (bank.score_physics(Xc), bank.score_physics(Xn)),
        "fitted_discriminant": (bank.score_fitted(Xc), bank.score_fitted(Xn)),
        "learned_reference": (bank.score_rf(control_traces),
                              bank.score_rf(ref_null_traces)),
    }
    for rule, (ctrl_s, null_s) in scorers.items():
        frac = {}
        for far in p.target_fars:
            thr = threshold_at_far(null_s, far)
            frac[f"{far:g}"] = fraction_training(ctrl_s, thr)
        out[rule] = {"auc": auc(ctrl_s, null_s), "frac_training": frac}
    return out


def run_rung2_cell(
    kind: str,
    name: str | None,
    ko: KoWorkloadParams,
    glue: KoTypeBParams,
    p: Rung2Params,
    *,
    st1: St1DetectorParams = DEFAULT.st1,
    rf_params: RfBaselineParams = DEFAULT.rf,
    meter_variants=(),
    seed: int,
) -> dict:
    """One Rung-2 evaluation cell.

    ``kind`` is ``"stated"`` (the stated workload population, OOF), ``"transfer"``
    (a domain shift ``name``, fit-nominal / zero-shot-predict), or ``"control"``
    (a semantic falsification control ``name`` vs a held-out nominal null). The
    cell is self-contained and reproducible off ``seed`` (crc-derived per cell in
    the driver script).
    """
    stated_meter = _stated_meter(p)
    rng_nom = np.random.default_rng([seed, 0])
    train_nom, infer_nom = ko_make_population(
        p.n_each, ko, glue, rng_nom, meter=stated_meter)

    if kind == "stated":
        rules = run_stated_cell(train_nom, infer_nom, glue, p,
                                st1=st1, rf_params=rf_params, seed=seed)
        return {"kind": "stated", "condition": "stated", "rules": rules}

    bank, _, _ = fit_rule_bank(train_nom, infer_nom, glue, p,
                               st1=st1, rf_params=rf_params, seed=seed)

    if kind == "transfer":
        pairs = dict(p.transfer_shifts)[name]
        glue2, ko2, meter2 = resolve_shift(pairs, glue, ko, meter_variants)
        m2 = meter2 if meter2 is not None else stated_meter
        rng_c = np.random.default_rng([seed, 1])
        train_s, infer_s = ko_make_population(p.n_each, ko2, glue2, rng_c, meter=m2)
        rules = _score_pos_neg(bank, glue2, st1, train_s, infer_s, p)
        return {"kind": "transfer", "condition": name, "rules": rules}

    if kind == "control":
        c = CONTROLS[name]
        rng_c = np.random.default_rng([seed, 1])
        control_pop = make_control_population(
            name, p.n_each, ko, glue, rng_c, p, meter=stated_meter)
        rng_ref = np.random.default_rng([seed, 2])
        ref_null = [ko_make_trace("infer", ko, glue, rng_ref, meter=stated_meter)
                    for _ in range(p.n_each)]
        rules = _score_control(bank, glue, st1, control_pop, ref_null, p)
        return {"kind": "control", "condition": name,
                "is_training": c.is_training, "expect": c.expect, "note": c.note,
                "rules": rules}

    raise ValueError(f"unknown cell kind {kind!r}")


def rung2_grid(p: Rung2Params) -> list[tuple[str, str, str | None]]:
    """The flat Rung-2 cell list: ``(cell_name, kind, condition_name)``.

    One ``stated`` cell (the Rung-2 FPR/FNR under the stated population), one
    cell per transfer shift, and one per semantic control. Names are stable and
    unique so the crc seed and per-cell artefacts are reproducible (the
    code.typeb.meter_boundary idiom).
    """
    cells: list[tuple[str, str, str | None]] = [("stated", "stated", None)]
    for name, _ in p.transfer_shifts:
        cells.append((f"transfer__{name}", "transfer", name))
    for name in CONTROL_ORDER:
        cells.append((f"control__{name}", "control", name))
    return cells
