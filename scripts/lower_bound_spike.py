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
     as a Lorentzian rolloff P_c(sigma)/P_c(0) ~ 1/(1 + kappa sigma^2) with the
     PARAMETER-FREE kappa = pi^2 f0 T_obs (the HWHM and bin-width algebra is
     carried through in compute(); nothing is left as an O(1) factor). We confirm
     both the form (R^2) and the constant.

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
import math
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
# 600 traces/class/level. The original 80 was too few: the H0 floor of the
# one-sided KS statistic is 0.25 at n=80 (see _ks_null_floor), larger than every
# epsilon being inverted, which left kappa_J varying ~4x across seeds and the
# derived sigma*(eps) unstable by ~2x. At 600 the floor drops to 0.093.
N_EACH = 600
F0_HZ = 1.0                       # fixed iteration cadence for a clean overlay
SIGMAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.35, 0.5)  # jitter grid
EPSILONS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)           # target detectabilities
N_BOOT = 400                      # bootstrap resamples for the sigma* intervals
FLOOR_Z = 3.0                     # keep a fit point only this far above the H0 floor
CV_TOL = 0.25                     # |measured CV / sigma - 1| allowed to call sigma "realised"
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
    """Youden index J = max_thr (TPR - FPR) for this statistic: the ONE-SIDED
    two-sample Kolmogorov-Smirnov distance of the score populations.

    (One-sided, not the usual two-sided KS distance: training traces score high,
    so only sup(F_train - F_null) is a detection advantage. One-sided <= two-sided
    <= TV, so the inequality the argument needs still runs the right way.)

    The POPULATION value of this quantity is what the necessary-distortion
    argument runs on: each threshold test is an explicit single-trace verifier, so
    by data processing J <= TV(P_train, P_null), and J > epsilon therefore
    certifies non-epsilon-covertness. Contrast :func:`_auc`, whose Gini index
    2*AUC-1 can exceed J and so is not attained by any single-trace test.

    NOTE this plug-in estimator maximises over thresholds on the same sample it
    evaluates, so it is upward-biased by ~:func:`_ks_null_floor` at finite n.
    Fit points must clear that floor (see :func:`_fit_kappa`) and the reported
    sigma* carry bootstrap intervals.
    """
    thr = np.unique(np.concatenate([pos, neg]))
    tpr = (pos[:, None] >= thr[None, :]).mean(axis=0)
    fpr = (neg[:, None] >= thr[None, :]).mean(axis=0)
    return float(np.max(tpr - fpr))


def _ks_null_floor(n: int, z: float = FLOOR_Z) -> float:
    """Upper edge of the H0 sampling fluctuation of :func:`_youden` at size n.

    For two equal samples of size n the limiting law of sqrt(n/2)*D+ has survival
    exp(-2x^2), hence mean sqrt(pi/8) and sd sqrt(1/2 - pi/8). Returns
    (mean + z*sd)/sqrt(n/2): a measured J below this is indistinguishable from no
    separation. Such points must be excluded from the rolloff fit -- J cannot go
    negative, so a null point still reads as a small POSITIVE advantage, and
    because the fit linearises as J0/J - 1 (which diverges as J -> 0) it would
    otherwise dominate the fit. (At n=80 this floor is 0.25, which is why the
    original sigma=0.35 and 0.5 points had to go; at n=600 it is 0.093.)
    """
    mean, sd = math.sqrt(math.pi / 8.0), math.sqrt(0.5 - math.pi / 8.0)
    return float((mean + z * sd) / math.sqrt(n / 2.0))


def _train_scores(sigma: float, work: bool,
                  rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """(SNR scores, mean coherent f0 power) for N_EACH training traces.

    ``work=False``: i.i.d. period jitter (ko_p.sigma_jitter, the accumulating
    family the bound targets). ``work=True``: real-work variation (training_F
    work_sigma), the measured ~zero-cost escape, for comparison.

    BOTH arms set sigma_jitter explicitly so the comparison is at MATCHED total
    distortion: the work arm zeroes KoWorkloadParams.sigma_jitter (default 0.1)
    before applying work_sigma. Leaving the default in place would give the work
    arm 0.1 of period jitter on top of its work variation, which is not a matched
    comparison -- it made J_work(0) read as J_jitter(0.1) and inverted the
    ordering of the two curves below sigma = 0.1.
    """
    t = make_time_grid(GLUE.duration_s, 1.0 / GLUE.fs)
    f_max = DEFAULT.floor.F_max
    f_peak = GLUE.f_peak_frac * f_max
    snr = np.empty(N_EACH)
    coh = np.empty(N_EACH)
    for i in range(N_EACH):
        if work:
            ko = KoWorkloadParams(**{**KO.__dict__, "sigma_jitter": 0.0})
            F = training_F(t, ko, rng, f_peak=f_peak, f0=F0_HZ,
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


def _fit_mask(sigmas: np.ndarray, y: np.ndarray, y0: float,
              floor: float) -> np.ndarray:
    """Points carrying slope information: 0 < sigma, and y clear of both ends.

    ``floor`` is the H0 fluctuation of the statistic (:func:`_ks_null_floor` for
    J, ~0 for the rank-based Gini, which can go negative and so screens itself).
    Excluding the saturated end (y >= 0.98*y0) and the null end (y <= floor) is
    what keeps the linearised fit from being driven by points that carry no
    separation -- the defect that made the n=80 kappa_J 97% determined by two
    washed-out points.
    """
    return (sigmas > 0) & (y > floor) & (y < 0.98 * y0)


def _fit_kappa(sigmas: np.ndarray, y: np.ndarray, y0: float,
               floor: float) -> tuple[float, np.ndarray]:
    """Least-squares kappa in y(sigma) = y0 / (1 + kappa sigma^2).

    Linearised: y0/y - 1 = kappa sigma^2 is linear through the origin in sigma^2.
    Returns (kappa, mask) so callers can report R^2 over the points actually
    fitted rather than over the whole sweep.
    """
    m = _fit_mask(sigmas, y, y0, floor)
    if m.sum() < 2:
        raise SystemExit(
            f"only {m.sum()} of {sigmas.size} sweep points clear the null floor "
            f"{floor:.3f} -- raise N_EACH (currently {N_EACH}) or extend SIGMAS; "
            "refusing to fit a rolloff constant to washed-out points")
    lin = y0 / y[m] - 1.0
    x = sigmas[m] ** 2
    return float((x @ lin) / (x @ x)), m


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination of ``pred`` against ``y``."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-30)


def _covertness_thresholds(kappa: float, y0: float,
                           sig_max: float) -> list[dict]:
    """sigma*(eps) from y0 / (1 + kappa sigma^2) = eps, per epsilon in EPSILONS.

    Flags ``extrapolated`` when the threshold falls beyond the swept sigma range,
    since the fitted rolloff has no support there -- and, past sigma ~ 0.35, the
    generator's own jitter model no longer holds either.
    """
    out = []
    for eps in EPSILONS:
        arg = y0 / eps - 1.0
        s = float(math.sqrt(arg / kappa)) if arg > 0 else 0.0
        out.append({"epsilon": eps, "sigma_star": s,
                    "extrapolated": s > sig_max})
    return out


def compute() -> dict:
    """Run the sweep and return the summary record."""
    rng = np.random.default_rng(SEED)
    infer = _infer_scores(rng)

    # Analytic rolloff constant, carried through rather than left as an O(1):
    # accumulating jitter gives D = (2 pi)^2 sigma^2 f0, so the line is a
    # Lorentzian of HWHM gamma = D/(4 pi) = pi sigma^2 f0 [Hz]. A record of
    # length T resolves 1/T, so the coherent power inside one bin is the
    # Lorentzian mass within a half-width B = 1/(2T), i.e.
    # (2/pi) arctan(B/gamma), whose large-sigma form is 1/(pi^2 f0 T sigma^2).
    # Hence kappa = pi^2 f0 T -- a parameter-free prediction, not a fitted scale.
    kappa_analytic = np.pi ** 2 * F0_HZ * GLUE.duration_s
    rows, scores = [], {}
    for sigma in SIGMAS:
        jit_snr, jit_coh = _train_scores(sigma, work=False, rng=rng)
        wrk_snr, _ = _train_scores(sigma, work=True, rng=rng)
        cv, d = _deperiod_anchors(sigma, rng)
        scores[sigma] = jit_snr          # kept for the bootstrap below
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
    sig_max = float(sig.max())
    floor = _ks_null_floor(N_EACH)
    # (a) PHYSICAL quantity: coherent f0 power, normalised to sigma -> 0. A power
    #     ratio has no sampling floor of its own, so only the saturated end is cut.
    coh = np.array([r["coherent_power_jitter"] for r in rows])
    coh_ratio = coh / (coh[0] + 1e-30)
    kappa_coh, m_coh = _fit_kappa(sig, coh_ratio, 1.0, 0.0)
    r2_coh = _r2(coh_ratio[m_coh],
                 1.0 / (1.0 + kappa_coh * sig[m_coh] ** 2))
    # (b) ACHIEVED ADVANTAGE of the explicit fixed-bin threshold test: the Youden
    #     index J <= TV. This is what the necessary-distortion claim runs on, and
    #     the only series whose fit sets the reported thresholds.
    j_jit = np.array([r["youden_jitter"] for r in rows])
    j0 = float(j_jit[0])
    kappa_j, m_j = _fit_kappa(sig, j_jit, j0, floor)
    r2_j = _r2(j_jit[m_j], j0 / (1.0 + kappa_j * sig[m_j] ** 2))
    # (c) The Gini index 2 AUC - 1, retained for continuity with the
    #     deliberation note. NOT an achieved single-trace advantage (it can
    #     exceed J), so it is reported alongside and does not set the threshold.
    tv_jit = np.array([2.0 * r["auc_jitter"] - 1.0 for r in rows])
    tv0 = float(tv_jit[0])
    kappa_tv, m_tv = _fit_kappa(sig, tv_jit, tv0, 0.0)
    r2_tv = _r2(tv_jit[m_tv], tv0 / (1.0 + kappa_tv * sig[m_tv] ** 2))

    # Largest swept sigma at which the generator still REALISES the nominal
    # distortion (measured cadence CV within CV_TOL of sigma). Past it the
    # reciprocal-Gaussian period draw and the ko_workload clamp take over, so
    # both the sigma^2 parameterisation and the cost anchors keyed by sigma stop
    # meaning what they say. Reported so a sigma* beyond it is flagged in the
    # artefact rather than only in prose.
    faithful = [r["sigma"] for r in rows
                if r["sigma"] > 0 and abs(r["cadence_cv"] / r["sigma"] - 1.0) <= CV_TOL]
    sig_faithful = float(max(faithful)) if faithful else 0.0
    thresholds = _covertness_thresholds(kappa_j, j0, sig_max)
    for th in thresholds:
        th["beyond_faithful_sigma"] = th["sigma_star"] > sig_faithful
    thresholds_gini = _covertness_thresholds(kappa_tv, tv0, sig_max)

    # Sensitivity: refit using only the points inside the faithful regime. The
    # headline fit needs unsaturated points and so reaches past it; quoting both
    # is how the reader sees that dependence instead of having to guess it.
    m_faith = _fit_mask(sig, j_jit, j0, floor) & (sig <= sig_faithful)
    if m_faith.sum() >= 2:
        lin = j0 / j_jit[m_faith] - 1.0
        x = sig[m_faith] ** 2
        kappa_j_faithful = float((x @ lin) / (x @ x))
    else:
        kappa_j_faithful = float("nan")

    # Bootstrap the whole J -> kappa_J -> sigma* chain over the score
    # populations, so the reported thresholds carry sampling intervals rather
    # than a spurious two-significant-figure precision. Resampling both classes
    # with replacement; the fit mask is recomputed per resample.
    boot_rng = np.random.default_rng(SEED + 1)
    k_boot, star_boot = [], {e: [] for e in EPSILONS}
    for _ in range(N_BOOT):
        neg_b = infer[boot_rng.integers(0, infer.size, infer.size)]
        j_b = np.array([
            _youden(scores[s][boot_rng.integers(0, N_EACH, N_EACH)], neg_b)
            for s in SIGMAS])
        y0_b = float(j_b[0])
        if not _fit_mask(sig, j_b, y0_b, floor).sum() >= 2:
            continue
        k_b, _ = _fit_kappa(sig, j_b, y0_b, floor)
        k_boot.append(k_b)
        for th in _covertness_thresholds(k_b, y0_b, sig_max):
            star_boot[th["epsilon"]].append(th["sigma_star"])
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    for th in thresholds:
        th["ci95"] = pct(star_boot[th["epsilon"]])

    return {
        "seed": SEED, "n_each": N_EACH, "f0_hz": F0_HZ,
        "duration_s": GLUE.duration_s, "fs": GLUE.fs,
        "band_hz": [GLUE.band_lo, GLUE.band_hi],
        "meter": {"sigma_eta": METER.sigma_eta},
        "kappa_analytic": kappa_analytic,
        "youden_null_floor": floor,
        "sigma_faithful_max": sig_faithful,
        "n_boot": len(k_boot),
        "coherent_power": {"kappa_fitted": kappa_coh, "r2": r2_coh,
                           "sigmas_fitted": sig[m_coh].tolist()},
        "detectability_youden": {"kappa_fitted": kappa_j, "r2": r2_j, "j0": j0,
                                 "sigmas_fitted": sig[m_j].tolist(),
                                 "kappa_ci95": pct(k_boot),
                                 "kappa_faithful_only": kappa_j_faithful},
        "detectability_tv": {"kappa_fitted": kappa_tv, "r2": r2_tv, "tv0": tv0,
                             "sigmas_fitted": sig[m_tv].tolist()},
        "sigma_max_swept": sig_max,
        "rows": rows,
        "covertness_thresholds": thresholds,
        "covertness_thresholds_gini": thresholds_gini,
        "analytic_form": (
            "coherent f0 power ~ 1/(1 + kappa sigma^2) with the PARAMETER-FREE "
            "kappa_analytic = pi^2 f0 T_obs (accumulating i.i.d. jitter: HWHM "
            "gamma = D/(4 pi) = pi sigma^2 f0, bin half-width 1/(2T), in-bin mass "
            "(2/pi) arctan(B/gamma) -> 1/(pi^2 f0 T sigma^2)). The ACHIEVED "
            "advantage of the explicit fixed-bin threshold test, J = max_thr "
            "(TPR - FPR) = the one-sided two-sample KS distance, follows the same "
            "shape with a much smaller effective kappa because J saturates while "
            "any f0 excess stays separable from the line-free null. Since J <= TV, "
            "J(sigma) > eps certifies non-eps-covertness, so covertness_thresholds "
            "inverts the J rolloff: eps-covertness REQUIRES sigma >= sigma*(eps). "
            "J is an in-sample maximum and so upward-biased at finite n; points "
            "below youden_null_floor carry no separation and are excluded from the "
            "fit (sigmas_fitted records which were used), and every sigma* carries "
            "a bootstrap ci95. covertness_thresholds_gini repeats the inversion on "
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
             label=rf"analytic $\kappa=\pi^2 f_0 T={k_an:.0f}$")
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
    # the H0 fluctuation of J: points below it carry no separation and are
    # excluded from the fit, so show the reader where that line sits
    axB.axhline(summary["youden_null_floor"], color=C["vermillion"], lw=0.6,
                ls="-.", zorder=0,
                label=rf"$H_0$ floor at $n$={summary['n_each']}")
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
          f"(analytic pi^2 f0 T = {summary['kappa_analytic']:.0f}, "
          f"{abs(cp['kappa_fitted']/summary['kappa_analytic']-1)*100:.2f}% off), "
          f"R^2={cp['r2']:.3f} over sigma={cp['sigmas_fitted']}")
    print(f"J null floor at n={summary['n_each']}: "
          f"{summary['youden_null_floor']:.3f}")
    print(f"achieved-advantage (J) rolloff: kappa={jd['kappa_fitted']:.1f} "
          f"[{jd['kappa_ci95'][0]:.1f}, {jd['kappa_ci95'][1]:.1f}], "
          f"R^2={jd['r2']:.3f}, J0={jd['j0']:.3f}   <- carries the bound")
    print(f"  fitted on sigma={jd['sigmas_fitted']} "
          f"({summary['n_boot']} bootstrap resamples)")
    print(f"  generator realises sigma up to {summary['sigma_faithful_max']:.2f} "
          f"(|CV/sigma-1| <= {CV_TOL}); faithful-only refit kappa="
          f"{jd['kappa_faithful_only']:.1f}")
    print(f"Gini (2 AUC - 1) rolloff:       kappa={tv['kappa_fitted']:.1f}, "
          f"R^2={tv['r2']:.3f}, TV0={tv['tv0']:.3f}   (reference only)")
    print("covertness thresholds sigma*(eps) from J, with 95% bootstrap CI:")
    for t in summary["covertness_thresholds"]:
        flag = " [extrapolated]" if t["extrapolated"] else ""
        if t["beyond_faithful_sigma"]:
            flag += " [beyond realised sigma]"
        print(f"  eps={t['epsilon']:.2f} -> sigma*={t['sigma_star']:.3f} "
              f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]{flag}")
    print(f"-> {_OUT / 'lower_bound_spike.json'} ; "
          f"{fig_pdf.with_suffix('')}.*")


if __name__ == "__main__":
    main()
