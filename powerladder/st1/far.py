"""ST1 realised-FAR harness (task 19.4): the gate's measuring instrument.

Per cell (stage x null x calibration): M independent null traces (spawned
child seeds), one p-value each from the detector's corrected -log10 p; report
the realised false-alarm rate at each nominal level with Clopper-Pearson
binomial CIs, the KS distance of the p-values from U(0,1), and the raw p
array (re-readable at ANY level later). Covariance-conditioning failures are
read off the code.st1.cyclo counter around the cell — each cell runs wholly
inside one process under the pool-over-cells design, so the counter is
cell-local.

Sizing (pre-registered in notes/results/st1-findings.md): M = 10^4 resolves a 3x
inflation at the 1e-3 nominal level (CP CI at k ~ 10 is ~[0.5, 1.8]x10^-3).

Storage: raw p-values to results/st1/raw/{stage}_{null}_{calib}.npz (gitignored
— *.npz is untracked repo-wide); aggregates merged into
results/st1/far_summary.json (tracked), nested {stage: {null: {calib: {...}}}}
with M, per-level {k, far, ci}, KS stat/p, cov_failures, and a params hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta, kstest

from .cyclo import cov_failure_count, reset_cov_failures


def binomial_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson (exact) binomial CI on a proportion, via beta quantiles.

    lo = Beta(k, n-k+1).ppf(a/2)   (0 when k = 0)
    hi = Beta(k+1, n-k).ppf(1-a/2) (1 when k = n)
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def params_hash(*objs) -> str:
    """Short stable hash of the reprs of the param dataclasses driving a cell."""
    h = hashlib.sha256("|".join(repr(o) for o in objs).encode())
    return h.hexdigest()[:12]


@dataclass
class CellResult:
    """One (stage x null x calibration) cell of the calibration table."""

    stage: str
    null: str
    calib: str
    M: int
    seed: str                                   # the seed spec, stringified
    levels: dict = field(default_factory=dict)  # {level: {"k", "far", "ci"}}
    ks_stat: float = float("nan")
    ks_pvalue: float = float("nan")
    cov_failures: int = 0
    params_hash: str = ""
    p: np.ndarray = field(default_factory=lambda: np.empty(0))


def run_cell(
    detector_pfn,
    null_fn,
    M: int,
    seed,
    *,
    stage: str,
    null_name: str,
    calib: str,
    n: int,
    fs: float,
    null_params,
    band_lo: float,
    band_hi: float,
    nominal_levels: tuple[float, ...],
    ci_conf: float = 0.95,
    phash: str = "",
) -> CellResult:
    """Draw M null traces, score each, and assemble the realised-FAR record.

    ``detector_pfn(t, P_obs, band_lo, band_hi) -> float`` returns a corrected
    -log10 p (the shared gate signature); the per-trace p-value is 10**(-s).
    Traces are drawn with M independent spawned child generators of
    ``np.random.default_rng(seed)`` so the cell is reproducible and the
    traces are statistically independent.
    """
    rng = np.random.default_rng(seed)
    children = rng.spawn(M)
    reset_cov_failures()
    p_vals = np.empty(M)
    for i, child in enumerate(children):
        tr = null_fn(n, fs, null_params, child)
        score = detector_pfn(tr.t, tr.P_obs, band_lo, band_hi)
        p_vals[i] = 10.0 ** (-score)
    n_fail = cov_failure_count()

    levels = {}
    for lvl in nominal_levels:
        k = int(np.sum(p_vals <= lvl))
        lo, hi = binomial_ci(k, M, ci_conf)
        levels[float(lvl)] = {"k": k, "far": k / M, "ci": [lo, hi]}

    ks = kstest(p_vals, "uniform")
    return CellResult(
        stage=stage, null=null_name, calib=calib, M=M, seed=str(seed),
        levels=levels, ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        cov_failures=n_fail, params_hash=phash, p=p_vals,
    )


def save_cell_raw(cell: CellResult, results_dir: Path) -> Path:
    """Raw per-cell p-values -> results/st1/raw/{stage}_{null}_{calib}.npz."""
    raw_dir = Path(results_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{cell.stage}_{cell.null}_{cell.calib}.npz"
    np.savez_compressed(
        path, p=cell.p, M=cell.M, stage=cell.stage, null=cell.null,
        calib=cell.calib, cov_failures=cell.cov_failures,
        params_hash=cell.params_hash,
    )
    return path


def update_summary(cell: CellResult, summary_path: Path) -> None:
    """Merge one cell into the nested far_summary.json (creating it if absent)."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    entry = {
        "M": cell.M,
        "levels": {
            f"{lvl:g}": rec for lvl, rec in sorted(cell.levels.items(),
                                                   reverse=True)
        },
        "ks": {"stat": cell.ks_stat, "pvalue": cell.ks_pvalue},
        "cov_failures": cell.cov_failures,
        "params_hash": cell.params_hash,
    }
    summary.setdefault(cell.stage, {}).setdefault(cell.null, {})[cell.calib] = entry
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
