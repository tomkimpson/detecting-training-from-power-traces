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

  2. A concrete verifier's ACHIEVED advantage decays with sigma (panel B), which
     is the direction a necessary-distortion claim needs. Building training vs
     inference-null populations (code.ko_workload -> code.forward ->
     code.observation, exactly the code.typeb.ko_synth path) and scoring a
     fixed-bin statistic at f0, the achieved advantage of the best threshold test
     on that statistic is the Youden index J = max_thr (TPR - FPR), i.e. the
     two-sample KS distance of the score populations. By data processing
     J <= TV(P_train, P_null), so J(sigma) > epsilon CERTIFIES that sigma is not
     epsilon-covert: epsilon-covertness REQUIRES sigma >= sigma*(epsilon), read
     off the J rolloff. Enlarging the verifier class can only raise sigma*.

     Note the direction carefully. Pinsker (TV <= sqrt(KL/2)) runs the other way
     and would give a SUFFICIENT distortion, not a necessary one, so it is not
     what the threshold is computed from. The Gini index 2*AUC - 1 is also
     retained in the summary (it is what the deliberation note quotes), but it is
     a paired/rank statistic that can EXCEED J, so it is not itself an achieved
     single-trace advantage and does not carry the bound.

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


def _youden(pos: np.ndarray, neg: np.ndarray) -> float:
    """Youden index J = max_thr (TPR - FPR): the ACHIEVED advantage of the best
    threshold test on this statistic (equivalently the two-sample KS distance of
    the score populations).

    This is the quantity the necessary-distortion argument runs on: a threshold
    test is an explicit single-trace verifier, and by data processing
    J <= TV(P_train, P_null). So J > epsilon certifies non-epsilon-covertness.
    Contrast :func:`_auc`, whose Gini index 2*AUC-1 can exceed J and is therefore
    not an achieved single-trace advantage.
    """
    thr = np.unique(np.concatenate([pos, neg]))
    tpr = (pos[:, None] >= thr[None, :]).mean(axis=0)
    fpr = (neg[:, None] >= thr[None, :]).mean(axis=0)
    return float(np.max(tpr - fpr))


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
            "youden_jitter": _youden(jit_snr, infer),
            "youden_work": _youden(wrk_snr, infer),
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
    # (b) ACHIEVED ADVANTAGE of the explicit fixed-bin threshold test: the Youden
    #     index J <= TV. This is what the necessary-distortion claim runs on.
    j_jit = np.array([r["youden_jitter"] for r in rows])
    j0 = float(j_jit[0])
    kappa_j = _fit_kappa(sig, j_jit, j0)
    r2_j = _r2(j_jit, j0 / (1.0 + kappa_j * sig ** 2))
    # (c) The Gini index 2 AUC - 1, retained for continuity with the
    #     deliberation note. NOT an achieved single-trace advantage (it can
    #     exceed J), so it is reported alongside and does not set the threshold.
    tv_jit = np.array([2.0 * r["auc_jitter"] - 1.0 for r in rows])
    tv0 = float(tv_jit[0])
    kappa_tv = _fit_kappa(sig, tv_jit, tv0)
    r2_tv = _r2(tv_jit, tv0 / (1.0 + kappa_tv * sig ** 2))

    # covertness threshold sigma*(eps): invert the ACHIEVED-advantage rolloff
    # J0 / (1 + kappa_j sigma^2) = eps. Flag extrapolation beyond the sweep --
    # and note the generator's linear jitter anchors break by sigma ~ 0.35 (the
    # ko_workload period clamp), so extrapolated rows sit outside the model.
    sig_max = float(sig.max())

    def _invert(k: float, y0: float) -> list[dict]:
        out = []
        for eps in EPSILONS:
            arg = y0 / eps - 1.0
            s = float(np.sqrt(arg / k)) if arg > 0 else 0.0
            out.append({"epsilon": eps, "sigma_star": s,
                        "extrapolated": s > sig_max})
        return out

    thresholds = _invert(kappa_j, j0)
    thresholds_gini = _invert(kappa_tv, tv0)

    return {
        "seed": SEED, "n_each": N_EACH, "f0_hz": F0_HZ,
        "duration_s": GLUE.duration_s, "fs": GLUE.fs,
        "band_hz": [GLUE.band_lo, GLUE.band_hi],
        "meter": {"sigma_eta": METER.sigma_eta},
        "kappa_analytic": kappa_analytic,
        "coherent_power": {"kappa_fitted": kappa_coh, "r2": r2_coh},
        "detectability_youden": {"kappa_fitted": kappa_j, "r2": r2_j, "j0": j0},
        "detectability_tv": {"kappa_fitted": kappa_tv, "r2": r2_tv, "tv0": tv0},
        "sigma_max_swept": sig_max,
        "rows": rows,
        "covertness_thresholds": thresholds,
        "covertness_thresholds_gini": thresholds_gini,
        "analytic_form": (
            "coherent f0 power ~ 1/(1 + kappa sigma^2), kappa_analytic = "
            "pi f0 T_obs (accumulating i.i.d. jitter, Lorentzian rolloff). "
            "The ACHIEVED advantage of the explicit fixed-bin threshold test, "
            "J = max_thr (TPR - FPR), follows the same shape with a much smaller "
            "effective kappa because J saturates while any f0 excess stays "
            "separable from the line-free null. Since J <= TV, J(sigma) > eps "
            "certifies non-eps-covertness, so covertness_thresholds inverts the "
            "(conservative) J rolloff: eps-covertness REQUIRES sigma >= "
            "sigma*(eps). covertness_thresholds_gini repeats the inversion on "
            "2 AUC - 1 for continuity with the deliberation note; that index can "
            "exceed J and is not an achieved single-trace advantage, so it does "
            "not carry the bound."),
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

    # --- panel B: achieved advantage + covertness thresholds + escape ---------
    j_jit = np.array([r["youden_jitter"] for r in rows])
    j_wrk = np.array([r["youden_work"] for r in rows])
    j0 = summary["detectability_youden"]["j0"]
    k_j = summary["detectability_youden"]["kappa_fitted"]
    axB.plot(grid, j0 / (1.0 + k_j * grid ** 2), color=C["grey"], lw=1.0,
             ls="-", zorder=0,
             label=rf"fit $\kappa={k_j:.0f}$ "
                   f"($R^2$={summary['detectability_youden']['r2']:.2f})")
    axB.plot(sig, j_jit, color=C["vermillion"], marker="o", ms=3, lw=1.1,
             label="i.i.d. period jitter")
    axB.plot(sig, j_wrk, color=C["blue"], marker="s", ms=3, lw=1.1, ls="--",
             label="real-work jitter (escape)")
    for th in summary["covertness_thresholds"]:
        if th["epsilon"] in (0.1, 0.5):
            axB.axhline(th["epsilon"], color=C["grey"], lw=0.5, ls=":", zorder=0)
    axB.set_xlabel(r"fractional period jitter $\sigma$")
    axB.set_ylabel(r"achieved advantage  $J \leq \mathrm{TV}$")
    axB.set_ylim(-0.05, 1.02)
    axB.legend(frameon=False, loc="lower left")

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
        print(f"  sigma={r['sigma']:.2f}  J_jit={r['youden_jitter']:.3f}  "
              f"J_work={r['youden_work']:.3f}  "
              f"Gini_jit={2*r['auc_jitter']-1:+.3f}  "
              f"CV={r['cadence_cv']:.3f} (~sigma)  "
              f"D={r['phase_diffusion_D']:.2f} vs {r['D_analytic']:.2f} analytic")
    cp = summary["coherent_power"]
    jd, tv = summary["detectability_youden"], summary["detectability_tv"]
    print(f"coherent-power rolloff: kappa={cp['kappa_fitted']:.0f} "
          f"(analytic pi f0 T = {summary['kappa_analytic']:.0f}), "
          f"R^2={cp['r2']:.3f}")
    print(f"achieved-advantage (J) rolloff: kappa={jd['kappa_fitted']:.0f}, "
          f"R^2={jd['r2']:.3f}, J0={jd['j0']:.3f}   <- carries the bound")
    print(f"Gini (2 AUC - 1) rolloff:       kappa={tv['kappa_fitted']:.0f}, "
          f"R^2={tv['r2']:.3f}, TV0={tv['tv0']:.3f}   (reference only)")
    print("covertness thresholds sigma*(eps) from J:")
    for t in summary["covertness_thresholds"]:
        flag = " [extrapolated]" if t["extrapolated"] else ""
        print(f"  eps={t['epsilon']:.2f} -> sigma*={t['sigma_star']:.3f}{flag}")
    print(f"-> {_OUT / 'lower_bound_spike.json'} ; "
          f"{fig_pdf.with_suffix('')}.*")


if __name__ == "__main__":
    main()
