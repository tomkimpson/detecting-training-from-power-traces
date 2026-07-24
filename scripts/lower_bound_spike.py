"""Lower-bound feasibility spike — numerical sanity check.

Strategic gate (tasks.md, "Lower-bound feasibility spike"): is a TV/KL
(covertness) lower bound on hiding cost derivable for one attack family
(i.i.d. phase jitter) under our filtered low-rate meter? The analytic argument
lives in ``notes/discussion/lower-bound-feasibility-spike.md``; this script is
its LIGHTWEIGHT sanity check, not a new empirical frontier (the expensive ST2
grids stay on slurm — this is ~2500 short traces, seconds on CPU).

What it checks, end to end, against the existing generator + meter:

  1. The measured coherent line power tracks the analytic phase-diffusion
     prediction (panel A). i.i.d. period jitter ACCUMULATES (deperiod.py:
     D ~ (2*pi)^2 sigma^2 f0), so the coherent power at the KNOWN f0 washes out
     as a Lorentzian rolloff P_c(sigma)/P_c(0) ~ 1/(1 + kappa sigma^2),
     kappa ~ pi f0 T_obs. We confirm the form (R^2) and that the fitted kappa is
     within an O(1) width-convention factor of the parameter-free pi f0 T_obs.

  2. The covert-comms inequality is computable on a concrete verifier (panel B).
     Building training vs inference-null populations (code.ko_workload ->
     code.forward -> code.observation, exactly the code.typeb.ko_synth path) and
     scoring a fixed-bin statistic at f0, TV = 2*AUC - 1 is the standard
     hypothesis-testing cap: any such verifier's detection advantage <= TV. It
     decays with sigma with the same functional form but a much smaller effective
     kappa (AUC saturates while any f0 excess stays separable), so covertness
     thresholds sigma*(epsilon) are read off this CONSERVATIVE rolloff.

  3. The honest caveat (see the note): the demonstrated verifier is FIXED at the
     known cadence; the work-jitter series (also plotted) is the measured
     ~zero-throughput-cost escape, and it is MORE detectable at matched sigma
     (the fixed comm-barrier residual) — direct support for the lower bound. The
     bound against a TRACKING/optimal verifier is carried by the analytic
     argument, not by this fixed-detector check.

  4. The generator's de-periodicisation anchors hold: measured cadence CV ~=
     sigma and D ~= (2*pi)^2 sigma^2 f0 (deperiod.py), recorded in the summary.

Reproduce:
    python scripts/lower_bound_spike.py
Outputs:
    figures/lower_bound_spike.{pdf,png}
    results/spike/lower_bound_spike.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powerladder.config import KoTypeBParams, KoWorkloadParams, MeterParams  # noqa: E402
from powerladder.config import DEFAULT  # noqa: E402
from powerladder.forward import TraceSpec, make_time_grid, simulate  # noqa: E402
from powerladder.ko_workload import inference_F, training_F, training_F_meta  # noqa: E402
from powerladder.noise import WhiteNoise  # noqa: E402
from powerladder.observation import apply_meter  # noqa: E402
from powerladder.plotstyle import C, WIDTH_WIDE, apply_house_style, save  # noqa: E402
from powerladder.typeb.deperiod import cadence_cv, phase_diffusion_coeff  # noqa: E402

# --- parameters (no magic numbers below) -------------------------------------
SEED = 0
N_EACH = 80                       # traces per class per level (keep it fast)
F0_HZ = 1.0                       # fixed iteration cadence for a clean overlay
SIGMAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.35, 0.5)  # jitter grid
EPSILONS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)           # target detectabilities
GLUE = KoTypeBParams()            # fs=20 Hz, 300 s, r/P0/f_peak_frac, band
KO = KoWorkloadParams()           # Ko Table-I defaults
METER = MeterParams(sigma_eta=GLUE.sigma_eta)   # nominal channel: AR-free floor

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "results" / "spike"


def _observe(F: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> tuple:
    """F(t) -> (t_meter, P_meter) via the forward model + meter map.

    Mirrors code.typeb.ko_synth._observe with a meter set: simulate noiselessly
    (the meter owns all observation noise), then push through apply_meter.
    """
    spec = TraceSpec(t=t, F=F,
                     r=np.full_like(t, GLUE.r),
                     P0=np.full_like(t, GLUE.P0))
    trace = simulate(spec, WhiteNoise(0.0), rng)
    mt = apply_meter(trace.t, trace.P_obs, METER, rng)
    return mt.t, mt.P_meter


def _fixed_bin(t: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fixed-bin measurements at the KNOWN nominal cadence f0.

    The covert-comms warden is a FIXED verifier: it knows the nominal iteration
    cadence f0 and tests that DFT bin (as opposed to a tracking detector that
    searches the band for a wandering line). Returns:

      snr  = f0-bin power / robust (median) in-band floor — the detector
             statistic whose population AUC gives the TV cap.
      coh  = COHERENT excess power at f0 = f0-bin power - floor — the physical
             line power whose phase-diffusion decay the derivation predicts
             (1/(1 + kappa sigma^2), kappa ~ pi f0 T_obs).
    """
    p = p - p.mean()
    fs = 1.0 / (t[1] - t[0])
    freqs = np.fft.rfftfreq(p.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(p)) ** 2
    band = (freqs >= GLUE.band_lo) & (freqs <= GLUE.band_hi)
    if band.sum() < 3:
        return 1.0, 0.0
    k = int(np.argmin(np.abs(freqs - F0_HZ)))    # the fixed f0 bin
    floor = float(np.median(psd[band]))
    return float(psd[k] / (floor + 1e-30)), float(max(psd[k] - floor, 0.0))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic (rank-based, ties handled)."""
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    r_pos = r[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _train_scores(sigma: float, work: bool,
                  rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """(SNR scores, mean coherent f0 power) for N_EACH training traces.

    ``work=False``: i.i.d. period jitter (ko_p.sigma_jitter, the accumulating
    family the bound targets). ``work=True``: real-work variation (training_F
    work_sigma), the measured ~zero-cost escape, for comparison.
    """
    t = make_time_grid(GLUE.duration_s, 1.0 / GLUE.fs)
    f_max = DEFAULT.floor.F_max
    f_peak = GLUE.f_peak_frac * f_max
    snr = np.empty(N_EACH)
    coh = np.empty(N_EACH)
    for i in range(N_EACH):
        if work:
            F = training_F(t, KO, rng, f_peak=f_peak, f0=F0_HZ,
                           eta_scale=GLUE.eta_scale, work_sigma=sigma)
        else:
            ko = KoWorkloadParams(**{**KO.__dict__, "sigma_jitter": sigma})
            F = training_F(t, ko, rng, f_peak=f_peak, f0=F0_HZ,
                           eta_scale=GLUE.eta_scale)
        t_m, p_m = _observe(F, t, rng)
        snr[i], coh[i] = _fixed_bin(t_m, p_m)
    return snr, float(coh.mean())


def _infer_scores(rng: np.random.Generator) -> np.ndarray:
    """Fixed-bin SNR scores for N_EACH inference-null traces (the hard null)."""
    t = make_time_grid(GLUE.duration_s, 1.0 / GLUE.fs)
    f_max = DEFAULT.floor.F_max
    f_peak = GLUE.f_peak_frac * f_max
    scores = np.empty(N_EACH)
    for i in range(N_EACH):
        F = inference_F(t, KO, rng, f_peak=f_peak, eta_scale=GLUE.eta_scale)
        t_m, p_m = _observe(F, t, rng)
        scores[i], _ = _fixed_bin(t_m, p_m)
    return scores


def _deperiod_anchors(sigma: float, rng: np.random.Generator) -> tuple:
    """Measured (cadence CV, phase-diffusion D) for one long jittered trace.

    Confirms deperiod.py's analytic anchor CV ~ sigma, D ~ (2*pi)^2 sigma^2 f0.
    """
    if sigma == 0.0:
        return 0.0, 0.0
    t = make_time_grid(GLUE.duration_s, 1.0 / GLUE.fs)
    ko = KoWorkloadParams(**{**KO.__dict__, "sigma_jitter": sigma})
    _, iter_starts = training_F_meta(t, ko, rng, f_peak=1.0, f0=F0_HZ,
                                     eta_scale=0.0)
    if iter_starts.size < 20:
        return float("nan"), float("nan")
    return cadence_cv(iter_starts), phase_diffusion_coeff(iter_starts)


def _fit_kappa(sigmas: np.ndarray, tv: np.ndarray, tv0: float) -> float:
    """Least-squares kappa in TV(sigma) = tv0 / (1 + kappa sigma^2).

    Linearised: 1/TV = (1/tv0)(1 + kappa sigma^2), so
    tv0/TV - 1 = kappa sigma^2 is linear through the origin in sigma^2.
    Fit on the informative regime (0 < TV < ~tv0) only; the fully-washed-out
    tail carries no slope information and its floor noise would bias the fit.
    """
    y = tv0 / np.clip(tv, 1e-6, None) - 1.0
    x = sigmas ** 2
    m = (sigmas > 0) & (tv > 0.02 * tv0) & (tv < 0.98 * tv0)
    if m.sum() < 2:
        m = sigmas > 0
    return float((x[m] @ y[m]) / (x[m] @ x[m]))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination of ``pred`` against ``y``."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-30)


def compute() -> dict:
    """Run the sweep and return the summary record."""
    rng = np.random.default_rng(SEED)
    infer = _infer_scores(rng)

    # analytic rolloff constant: D*T/(4 pi) with D = (2 pi)^2 sigma^2 f0 gives
    # kappa_analytic = pi f0 T_obs (Lorentzian HWHM per DFT bin; the width
    # convention leaves an O(1) prefactor, hence we also FIT kappa below).
    kappa_analytic = np.pi * F0_HZ * GLUE.duration_s
    rows = []
    for sigma in SIGMAS:
        jit_snr, jit_coh = _train_scores(sigma, work=False, rng=rng)
        wrk_snr, _ = _train_scores(sigma, work=True, rng=rng)
        cv, d = _deperiod_anchors(sigma, rng)
        rows.append({
            "sigma": sigma,
            "auc_jitter": _auc(jit_snr, infer),
            "auc_work": _auc(wrk_snr, infer),
            "coherent_power_jitter": jit_coh,
            "cadence_cv": cv,
            "phase_diffusion_D": d,
            "D_analytic": (2.0 * np.pi) ** 2 * sigma ** 2 * F0_HZ,
        })

    sig = np.array([r["sigma"] for r in rows])
    # (a) PHYSICAL quantity: coherent f0 power, normalised to sigma -> 0.
    coh = np.array([r["coherent_power_jitter"] for r in rows])
    coh_ratio = coh / (coh[0] + 1e-30)
    kappa_coh = _fit_kappa(sig, coh_ratio, 1.0)     # vs analytic pi f0 T
    r2_coh = _r2(coh_ratio, 1.0 / (1.0 + kappa_coh * sig ** 2))
    # (b) DETECTABILITY: TV cap = 2 AUC - 1 (saturates while any f0 excess
    #     remains detectable, so its effective rolloff kappa is << the
    #     coherent-power kappa -- reported, not conflated).
    tv_jit = np.array([2.0 * r["auc_jitter"] - 1.0 for r in rows])
    tv0 = float(tv_jit[0])
    kappa_tv = _fit_kappa(sig, tv_jit, tv0)
    r2_tv = _r2(tv_jit, tv0 / (1.0 + kappa_tv * sig ** 2))

    # covertness threshold sigma*(eps) from the DETECTABILITY rolloff
    # TV0 / (1 + kappa_tv sigma^2) = eps. Flag extrapolation beyond the sweep.
    sig_max = float(sig.max())
    thresholds = []
    for eps in EPSILONS:
        arg = tv0 / eps - 1.0
        s = float(np.sqrt(arg / kappa_tv)) if arg > 0 else 0.0
        thresholds.append({"epsilon": eps, "sigma_star": s,
                           "extrapolated": s > sig_max})

    return {
        "seed": SEED, "n_each": N_EACH, "f0_hz": F0_HZ,
        "duration_s": GLUE.duration_s, "fs": GLUE.fs,
        "band_hz": [GLUE.band_lo, GLUE.band_hi],
        "meter": {"sigma_eta": METER.sigma_eta},
        "kappa_analytic": kappa_analytic,
        "coherent_power": {"kappa_fitted": kappa_coh, "r2": r2_coh},
        "detectability_tv": {"kappa_fitted": kappa_tv, "r2": r2_tv, "tv0": tv0},
        "sigma_max_swept": sig_max,
        "rows": rows,
        "covertness_thresholds": thresholds,
        "analytic_form": (
            "coherent f0 power ~ 1/(1 + kappa sigma^2), kappa_analytic = "
            "pi f0 T_obs (accumulating i.i.d. jitter, Lorentzian rolloff). "
            "Detectability TV = 2 AUC - 1 follows the same shape with a much "
            "smaller effective kappa because AUC saturates while any f0 excess "
            "stays separable from the line-free null; thresholds use that "
            "(conservative) detectability rolloff."),
    }


def plot(summary: dict) -> pathlib.Path:
    """(A) coherent f0 power vs sigma with the analytic Lorentzian rolloff;
    (B) detectability cap TV = 2 AUC - 1 with covertness thresholds + escape."""
    apply_house_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 2.6))

    rows = summary["rows"]
    sig = np.array([r["sigma"] for r in rows])
    grid = np.linspace(sig.min(), sig.max(), 300)

    # --- panel A: coherent line power (the derivation's core claim) -----------
    coh = np.array([r["coherent_power_jitter"] for r in rows])
    coh_ratio = coh / (coh[0] + 1e-30)
    k_coh = summary["coherent_power"]["kappa_fitted"]
    k_an = summary["kappa_analytic"]
    axA.plot(grid, 1.0 / (1.0 + k_an * grid ** 2), color=C["grey"], lw=1.0,
             ls=":", zorder=0,
             label=rf"analytic $\kappa=\pi f_0 T={k_an:.0f}$")
    axA.plot(grid, 1.0 / (1.0 + k_coh * grid ** 2), color=C["grey"], lw=1.0,
             ls="-", zorder=0,
             label=rf"fit $\kappa={k_coh:.0f}$ "
                   f"($R^2$={summary['coherent_power']['r2']:.2f})")
    axA.plot(sig, coh_ratio, color=C["vermillion"], marker="o", ms=3, lw=1.1,
             label="coherent $f_0$ power (measured)")
    axA.set_xlabel(r"fractional period jitter $\sigma$")
    axA.set_ylabel(r"coherent line power  $P_c(\sigma)/P_c(0)$")
    axA.set_yscale("log")
    axA.set_ylim(1e-3, 2.0)
    axA.legend(frameon=False, loc="lower left")

    # --- panel B: detectability cap + covertness thresholds + escape ----------
    tv_jit = np.array([2.0 * r["auc_jitter"] - 1.0 for r in rows])
    tv_wrk = np.array([2.0 * r["auc_work"] - 1.0 for r in rows])
    tv0 = summary["detectability_tv"]["tv0"]
    k_tv = summary["detectability_tv"]["kappa_fitted"]
    axB.plot(grid, tv0 / (1.0 + k_tv * grid ** 2), color=C["grey"], lw=1.0,
             ls="-", zorder=0,
             label=rf"fit ($R^2$={summary['detectability_tv']['r2']:.2f})")
    axB.plot(sig, tv_jit, color=C["vermillion"], marker="o", ms=3, lw=1.1,
             label="i.i.d. period jitter")
    axB.plot(sig, tv_wrk, color=C["blue"], marker="s", ms=3, lw=1.1, ls="--",
             label="real-work jitter (escape)")
    for th in summary["covertness_thresholds"]:
        if th["epsilon"] in (0.1, 0.5):
            axB.axhline(th["epsilon"], color=C["grey"], lw=0.5, ls=":", zorder=0)
    axB.set_xlabel(r"fractional period jitter $\sigma$")
    axB.set_ylabel(r"detectability cap  $TV \approx 2\,\mathrm{AUC}-1$")
    axB.set_ylim(-0.05, 1.02)
    axB.legend(frameon=False, loc="upper right")

    fig.suptitle("Lower-bound spike: covertness of i.i.d. phase jitter "
                 f"(fixed-bin verifier, $f_0$={F0_HZ:g} Hz, "
                 f"$T$={GLUE.duration_s:g} s)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "lower_bound_spike")


def main() -> None:
    summary = compute()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "lower_bound_spike.json").write_text(json.dumps(summary, indent=2))
    fig_pdf = plot(summary)

    print("i.i.d. period jitter (fixed-bin verifier):")
    for r in summary["rows"]:
        print(f"  sigma={r['sigma']:.2f}  TV_jit={2*r['auc_jitter']-1:+.3f}  "
              f"TV_work={2*r['auc_work']-1:+.3f}  "
              f"CV={r['cadence_cv']:.3f} (~sigma)  "
              f"D={r['phase_diffusion_D']:.2f} vs {r['D_analytic']:.2f} analytic")
    cp, tv = summary["coherent_power"], summary["detectability_tv"]
    print(f"coherent-power rolloff: kappa={cp['kappa_fitted']:.0f} "
          f"(analytic pi f0 T = {summary['kappa_analytic']:.0f}), "
          f"R^2={cp['r2']:.3f}")
    print(f"detectability rolloff:  kappa={tv['kappa_fitted']:.0f}, "
          f"R^2={tv['r2']:.3f}, TV0={tv['tv0']:.3f}")
    print("covertness thresholds sigma*(eps):")
    for t in summary["covertness_thresholds"]:
        flag = " [extrapolated]" if t["extrapolated"] else ""
        print(f"  eps={t['epsilon']:.2f} -> sigma*={t['sigma_star']:.3f}{flag}")
    print(f"-> {_OUT / 'lower_bound_spike.json'} ; "
          f"{fig_pdf.with_suffix('')}.*")


if __name__ == "__main__":
    main()
