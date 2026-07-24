"""Parameters for the Phase 1 power-only simulator.

Single source of truth — scripts import :data:`DEFAULT` rather than hardcoding
numbers (CLAUDE.md: "no magic numbers"). Every value carries its units and a
provenance note.

PLACEHOLDER VALUES. The exchange-rate band, overhead band, and noise level are
order-of-magnitude placeholders from Horowitz, "Computing's Energy Problem (and
What We Can Do About It)", ISSCC 2014, as recorded in
notes/development-notes/phase1-plan.md Sec. 2.2.
They are explicitly to be replaced by the A2 single-GPU bench measurements
(Experiments 1-3, paper Sec. 3.6). Absolute numbers here are illustrative; the
*structure* (band ratios driving beta > 1) is what the A0 sanity check shows.

Notation matches spec.md / paper:
    r        exchange rate / energy-per-op            [J/FLOP]
    F        FLOP rate                                [FLOP/s]
    P0       non-compute overhead power               [W]
    E0       overhead *energy* over the window = P0 T [J]
    C_max    capacity ceiling = F_max T               [FLOP]
    dE0      overhead *energy* band width             [J]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloorParams:
    """Bands that set the closed-form verification floor beta.

    The declared-side exchange-rate band [r_lo, r_hi] is the product of the CMOS
    factors r ~ kappa(precision) * alpha(operands) * C_eff(locality) * V^2
    (paper Sec. 5). The headline ratio r_hi / r_lo ~ 10-20x comes from precision
    (fp32 -> int8) compounded by DVFS (V^2, ~1.5-2x) and operand locality.
    """

    # Declared-side exchange-rate band [J/FLOP].
    r_lo: float = 1.0e-12          # cheapest declared op (~A100 fp16 ballpark)
    r_hi: float = 15.0e-12         # ~15x band (mid of the 10-20x range)

    # Cheapest covert edge r_lo^cov (the divisor of beta). For the bare power
    # meter the covert edge coincides with the declared cheapest rate.
    r_lo_cov: float = 1.0e-12

    # Declared hardware FLOP utilisation HFU_dec = C_dec / C_max (dimensionless).
    hfu_dec: float = 1.0

    # Overhead *power* band [W]; idle/cooling/leakage draw the verifier cannot
    # subtract. The overhead energy band is (P0_hi - P0_lo) * T -> see dE0.
    P0_lo: float = 50.0
    P0_hi: float = 150.0

    # Capacity scale.
    F_max: float = 1.0e15          # 1 PFLOP/s ceiling
    T: float = 3600.0              # 1 h monitoring window [s]

    @property
    def dE0(self) -> float:
        """Overhead energy band width Delta E_0 = (P0_hi - P0_lo) * T  [J]."""
        return (self.P0_hi - self.P0_lo) * self.T

    @property
    def C_max(self) -> float:
        """Capacity ceiling C_max = F_max * T  [FLOP]."""
        return self.F_max * self.T


@dataclass(frozen=True)
class ChannelParams:
    """Verifier-channel residual parameters for the A1 channel-residual study.

    These set how far each *irreducible* budget term (u_dec, u_m, u_0) shrinks as
    the corresponding channel is deployed (spec.md Sec. 4 A1 deliverable #1: "the
    paper states the allowances only schematically; this study makes them curves").
    The channel-residual sweeps over n / m / q0 live in the plotting script; these
    are the fixed companions.

    PLACEHOLDER VALUES, grounded by A2 hardware (spec.md Sec. 4 A1 / Sec. 5).
    """

    # Confidence multiplier z_gamma at level 1 - gamma for the re-execution
    # standard error sigma_dec / sqrt(n). Two-sided 95% normal quantile
    # (spec.md Sec. 4 line 154); expose gamma's effect by editing this.
    z_gamma: float = 1.96

    # Declared-typical per-FLOP rate std [J/FLOP]. PLACEHOLDER: a *narrow* sub-band
    # of the full declared band (r_hi - r_lo) = 1.4e-11 -- only the activity-factor
    # alpha jitter on declared operands survives re-execution, not the full
    # precision x DVFS x locality band. To be primed by A2 Experiment 2 restricted
    # to declared-typical operands (spec.md line 195).
    sigma_dec: float = 3.0e-13

    # Representative declared rate scaling the transfer band 2m [J/FLOP].
    # Mid-band (r_lo + r_hi)/2 of FloorParams; a strict worst-case variant uses
    # r_hi. Not r_lo^cov (which enters the divisor of beta_composed instead).
    r_bar_dec: float = 8.0e-12


@dataclass(frozen=True)
class SimParams:
    """Synthetic-trace generation knobs."""

    dt: float = 1.0                # sample period [s] (NVML-cadence-like)
    sigma_eta: float = 5.0         # meter-noise std [W] (PLACEHOLDER)
    seed: int = 0                  # RNG seed for reproducibility


@dataclass(frozen=True)
class KoInferenceParams:
    """HARD inference null for the Type detector (B1 / task 3.3).

    NOT from Ko et al. (who model training/fine-tuning/aggregate only). Provenance:
    LLM-serving systems -- continuous batching (Orca, OSDI'22; vLLM, SOSP'23),
    prefill/decode disaggregation, and MoE expert routing. The Type null is not
    featureless (spec.md Sec. 4 B1 premise 2): batched inference carries its own
    QUASI-PERIODIC in-band structure that can mimic a training line. The hard null
    puts in-band power COMPARABLE to a training line but with NO single concentrated,
    trackable line, so the detectors must discriminate concentration-vs-diffuseness
    rather than mere presence of power (b0-state-space §6).

    Three superposed structures, each in-band but individually non-trackable:
      1. decode fluctuation spread across a broad band that spans BEYOND the search
         band -- continuous batching reshuffles the batch every step, so token-
         generation power carries no concentrated cadence at the meter. The breadth
         is calibrated so the null retains MODERATE in-band concentration: enough
         that a peak-to-background filter confuses it with an honestly-SMEARED
         training line (the hard false-positive case, task 3.3), but with no
         temporally coherent path, so the Viterbi tracker still rejects it;
      2. prefill bursts at a slow Poisson arrival rate -> aperiodic in-band spikes;
      3. MoE-routing amplitude envelope (slow expert-load imbalance) -> structure
         on the decode floor (so the null is structured, not merely white).

    Amplitudes are dimensionless fractions of f_peak (the caller's FLOP-rate scale),
    matching the training waveform's 1 +/- delta convention in KoWorkloadParams.

    Calibrated (decode_hi, inf_amp_frac) so on the gate: a STATIONARY training line
    beats the null on both detectors (stationary tie at high TPR), while a DRIFTED
    line collapses the spectral filter to the null's level yet the Viterbi tracker
    holds -- the B0 verdict reproduced on the faithful Ko generator (task 3.1-3.2).
    """

    # Decode fluctuation band [Hz]: from below the search band up to ABOVE it. The
    # breadth sets the null's in-band concentration -- wide enough to carry no line
    # but not so wide that the null becomes trivially separable from a smeared line.
    decode_lo: float = 0.1
    decode_hi: float = 4.0         # < Nyquist (fs/2 = 10 Hz at the 20 Hz meter)
    # Prefill-burst Poisson arrival rate [Hz], FLOP-rate amplitude (rel. f_peak),
    # and duration [s] (prefill is a compute-heavy phase, not an instant spike).
    prefill_rate_hz: float = 0.08
    prefill_amp: float = 0.15
    prefill_dur_s: float = 0.1
    # MoE load-imbalance amplitude envelope: OU timescale [s] and modulation depth.
    moe_tau_s: float = 5.0
    moe_depth: float = 0.4
    # Decode fluctuation amplitude (rel. f_peak): the steady token-generation draw.
    # Sets the in-band RMS; calibrated comparable to the training line by
    # test_inference_inband_power_comparable (B1 task 3.3 / gate calibration).
    inf_amp_frac: float = 0.40
    # Intra-step Gaussian noise range (sub-step fluctuation, as Ko's eta).
    eta_inf: tuple[float, float] = (0.02, 0.05)
    # Mean decode level (rel. f_peak): the DC compute floor inference runs at.
    base_level: float = 0.5


@dataclass(frozen=True)
class KoWorkloadParams:
    """Ko et al. (arXiv 2508.16457) stochastic workload model — Table I.

    The Phase 2 time-resolved tracker drives the *latent FLOP rate* F(t) with Ko
    et al.'s piecewise-periodic waveform (phase2-plan.md Sec. 1.1: "their workload
    model for F(t) rather than a generic random walk"), reused by B1 (tasks 3-4).
    Their model emits *power*; we repurpose the same shape for F (up/compute phase
    -> high F, down/comm phase -> low F). See code.ko_workload.

    NAMING: Ko's compute/comm *phase ratio* is called "r" in the paper. Our project
    reserves r for the exchange rate [J/FLOP], so the phase ratio is ``rho`` here.

    Values are dimensionless (jitter, deltas, intra-phase noise) or in Hz
    (frequencies); the absolute F scale (F_peak) is supplied by the caller from
    FloorParams.F_max. Independently reviewed by AWS/Meta (Ko et al. Remark 2),
    which pre-empts the "invented signatures" referee.
    """

    # --- Training (eqs 1-5) ---------------------------------------------------
    # Iteration fluctuation frequency f0 ~ U(f0_lo, f0_hi) [Hz]; period 1/f0.
    f0_lo: float = 0.5
    f0_hi: float = 1.5
    # Per-iteration period jitter xi ~ N(0, sigma_jitter^2): T_i = 1/(f0(1+xi)).
    sigma_jitter: float = 0.1

    # --- f0 drift (B1 / task 3.2) --------------------------------------------
    # Slow random walk of the CENTRE frequency on top of Ko's i.i.d. xi. Ko's xi
    # is mean-reverting to f0 every iteration -- it broadens the line but does NOT
    # accumulate, so the line does not drift. Honest async/pipeline parallelism,
    # gradient accumulation and compute-comm overlap make f0 itself wander slowly
    # (a false-negative risk before any adversary; spec.md Sec. 4 B1 premise 1).
    # The wander axis is what decides spectral-vs-Viterbi (b0-state-space §6). The
    # drift is an Ornstein-Uhlenbeck walk g_i of the instantaneous f0 = f0*g_i:
    # OU (not a pure cumsum) keeps the line in-band over the window and makes the
    # excursion scale physical rather than dependent on the iteration count.
    # 0.0 = stationary line (Ko-faithful, default -- the no-drift code path draws
    # no extra RNG, so existing traces stay byte-identical).
    f0_drift_hz: float = 0.0       # std of the f0 excursion over the window [Hz]
    f0_drift_theta: float = 0.02   # OU mean-reversion per iteration (keeps in-band)
    # --- work-variation attack (ST2 / task 20.3) ------------------------------
    # Honest number of gradient-accumulation micro-steps per iteration: the
    # up phase is work_base_accum micro-steps of t_micro = rho*T_base/
    # work_base_accum each. The work-variation knobs on training_F (work_sigma,
    # work_drift_hz, work_shift_hz) vary the INTEGER micro-step count G_i per
    # iteration (stochastic rounding, floor 1) exactly as the measured
    # code.b2.workloads.work_jitter_schedule does on hardware, so synthetic
    # levels map 1:1 to the measured b2 anchors. == B2Params-era grad_accum
    # granularity; 8 matches the hardware campaign's TrainLoop default.
    work_base_accum: int = 8

    # Compute/comm phase ratio rho ~ U(rho_tr_lo, rho_tr_hi): T_up = rho*T_i.
    rho_tr_lo: float = 0.55
    rho_tr_hi: float = 0.8
    # Per-phase level deltas: up ~ N(0, sigma_delta_tr^2), down ~ N(mu_delta_tr, .).
    sigma_delta_tr: float = 0.05
    mu_delta_tr: float = 0.3
    # Intra-phase (sub-step) Gaussian noise std, drawn per phase from these ranges.
    eta_up_tr: tuple[float, float] = (0.02, 0.05)
    eta_down_tr: tuple[float, float] = (0.01, 0.03)

    # --- Fine-tuning (eqs 6-10) ----------------------------------------------
    # Cycle frequency f1 ~ U(f1_lo, f1_hi) [Hz]; jitter reuses sigma_jitter (Ko
    # Table I lists sigma_xi only for training; we apply the same to fine-tuning).
    f1_lo: float = 0.3
    f1_hi: float = 0.7
    # Tail/idle phase ratio rho ~ U(rho_ft_lo, rho_ft_hi): T_tail = rho*T_i.
    rho_ft_lo: float = 0.7
    rho_ft_hi: float = 0.9
    # Per-phase deltas: tail ~ N(0, sigma_delta_ft^2), idle ~ N(mu_delta_ft, .).
    sigma_delta_ft: float = 0.03
    mu_delta_ft: float = 0.8
    eta_tail_ft: tuple[float, float] = (0.01, 0.03)
    eta_idle_ft: tuple[float, float] = (0.005, 0.02)

    # Aggregate (eq 11) dominance ratio  P0_tr : sum P_tr : sum P_ft.
    aggregate_ratio: tuple[float, float, float] = (9.0, 0.5, 0.5)

    # --- Inference null (B1 / task 3.3; NOT Ko -- see KoInferenceParams) ------
    inference: KoInferenceParams = KoInferenceParams()


@dataclass(frozen=True)
class TrackerParams:
    """EKF/UKF joint-estimation knobs (compute_cluster_tracking.md; paper Sec 4.1).

    State x = [F, r, P0]; random-walk dynamics x_k = x_{k-1} + w, w ~ N(0, Q);
    nonlinear measurement h(x) = F*r + P0, z = P_obs. The filter resolves the
    one-equation/three-unknown system *only* through the timescale-separation prior
    encoded in Q (compute_cluster_tracking Sec. 5.1): F may jump (q_F LARGE), r
    drifts over seconds-minutes (q_r MEDIUM), P0 over tens of minutes (q_P0 SMALL).
    Loosening that ordering is exactly the non-injectivity collapse of task 1.4.

    Process noise is given as a random-walk std PER sqrt(second) in each state's
    physical units, so Q = diag([q_F^2, q_r^2, q_P0^2]) * dt is dt-consistent.
    PLACEHOLDER magnitudes (scaled to FloorParams); the *ratios* carry the physics.
    """

    # Random-walk volatilities (std per sqrt(s)); q_F >> q_r >> q_P0. The ratio
    # q_F/q_r is the timescale-separation prior; task 1.4 sweeps q_r up toward q_F.
    q_F: float = 3.0e14            # FLOP/s : F can swing a large fraction of F_max/s
    q_r: float = 1.0e-14          # J/FLOP : r drifts ~few % over the window (medium)
    q_P0: float = 0.1             # W      : baseline barely moves second-to-second

    # Meter-noise variance R [W^2]. Ties to SimParams.sigma_eta (R = sigma_eta^2).
    R: float = 25.0               # = 5.0^2, matches SimParams.sigma_eta default

    # Initial-covariance std (state-relative) for P_0 = diag(.^2). The r prior is
    # tight (~2.5% of a mid-band rate): the verifier knows the declared operating
    # point reasonably well (spec.md "r is partially seeable"); this is what makes
    # the cooperative case identifiable -- a loose r prior leaves F/r ambiguous.
    p0_F: float = 1.0e15
    p0_r: float = 2.0e-13
    p0_P0: float = 100.0

    # State constraints (compute_cluster_tracking Sec. 5.3): clip after each update.
    r_min: float = 1.0e-15        # J/FLOP > 0
    P0_min: float = 0.0           # W >= 0
    # F is clipped to [0, F_max]; F_max comes from FloorParams.

    # Unscented-transform tuning (van der Merwe sigma points).
    ukf_alpha: float = 1.0e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0


@dataclass(frozen=True)
class BenchConfig:
    """Knobs for the A2 single-GPU bench harness (code/bench/, scripts/run_bench).

    Single source of truth for the three paper-Sec.-3.6 experiments that measure
    the bands placeholdered in :class:`FloorParams` / :class:`ChannelParams`:

        Exp 1  exchange-rate band r_hi/r_lo : vary precision x locality (x DVFS)
        Exp 2  covert edge r_lo^cov, operand core w0 : vary operand VALUES only
        Exp 3  overhead band dE0 + affine check : vary utilisation idle->saturated

    No magic numbers in the runner — every shape, dtype, repeat count lives here.

    Locality is set by matrix size relative to the A100 L2 cache (40 MB), not a
    flag. For a (M,K)*(K,N) matmul the operand+result footprint is
    (M*K + K*N + M*N) * bytes(dtype). ``shape_cache`` keeps all three resident in
    L2 (low C_eff); ``shape_dram`` is far larger than L2 so every op pays a DRAM
    fetch (high C_eff). Residency arithmetic for the defaults:
        cache (2048^3, fp16 2 B): 3 * 2048^2 * 2 B = 24 MB  < 40 MB  -> L2-resident
        dram  (8192^3, fp16 2 B): 3 * 8192^2 * 2 B = 384 MB >> 40 MB -> DRAM-bound
    1024^3 was too small — launch-bound, sub-ms windows that miss the NVML energy
    counter (2026-06-18). 2048^3 fits L2 for fp16/bf16 yet runs long enough to
    measure; fp32 (48 MB) spills slightly but sits at the high-r end regardless.
    """

    # Matmul shapes (M, N, K). FLOP count per matmul is 2*M*N*K (see
    # code.bench.workload.flops_matmul).
    shape_cache: tuple[int, int, int] = (2048, 2048, 2048)   # L2-resident, low C_eff
    shape_dram: tuple[int, int, int] = (8192, 8192, 8192)    # DRAM-bound, high C_eff

    # Exp 1 precision axis (the kappa factor; fp32 -> int8 is the precision span).
    # "fp32" = tf32 OFF, "tf32" = fp32 tensors with tf32 allowed (a backend flag,
    # not a dtype). Handled in code.bench.workload.make_matmul.
    dtypes: tuple[str, ...] = ("fp32", "tf32", "fp16", "bf16", "int8")

    # Exp 1 locality axis (the C_eff factor).
    localities: tuple[str, ...] = ("cache", "dram")

    # Exp 1 DVFS axis (the V^2 factor): SM clocks [MHz] to attempt to lock. None
    # leaves the clock to the governor (the expected shared-cluster case — clock
    # locking usually needs elevated privileges; see code.bench.device). Achieved
    # clocks are recorded per run regardless, so any passive variation is captured.
    dvfs_sm_clocks: tuple[int, ...] | None = None

    # Exp 2 operand-value axis (the alpha factor). Held at one dtype/shape/clock so
    # ONLY the switching activity moves. "declared_typical" primes sigma_dec.
    operand_dists: tuple[str, ...] = (
        "zeros", "sparse_struct", "low_entropy",
        "dense_random", "adversarial_toggle", "declared_typical",
    )
    exp2_dtype: str = "fp16"          # fixed precision for the operand sweep
    exp2_locality: str = "dram"       # fixed shape for the operand sweep

    # Exp 3 utilisation axis: duty cycle d in [0, 1] of a fixed busy kernel. Mean
    # FLOP rate F = d * F_peak and mean power P = P0 + d*(P_busy - P0), so a linear
    # fit P = r*F + P0 has intercept = idle/overhead power P0 (the dE0 band edge).
    util_fracs: tuple[float, ...] = (
        0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
    )
    exp3_dtype: str = "fp16"
    exp3_locality: str = "dram"
    exp3_period_s: float = 0.5        # duty-cycle period [s] (busy + idle per cycle)
    exp3_n_periods: int = 8           # periods per measured window

    # Measurement window. The NVML total-energy counter updates coarsely (tens of
    # ms), so a window must span MANY updates for the delta to be accurate — a few
    # ms reads stale/zero deltas (probe finding 2026-06-18). Since dtypes differ in
    # throughput by ~30x, a fixed matmul count can't equalise window length; the
    # runner instead CALIBRATES iters per cell to ~target_window_s of wall time
    # (code.bench.runner). ``iters`` is only the fallback when calibration is off.
    target_window_s: float = 1.5      # wall time per measured window [s]
    calib_matmuls: int = 5            # matmuls timed to estimate per-matmul cost
    max_iters: int = 100000           # safety cap on calibrated iters
    iters: int = 64                   # fallback iters (target_window_s <= 0)
    warmup: int = 3                   # discarded warmup windows (JIT/autotune/soak)
    repeats: int = 5                  # measured windows per sweep point
    discard_frac: float = 0.2         # trim fraction (slowest/highest-energy tail)

    # Thermal-drift control (code.bench.runner): randomise sweep order and splice a
    # fixed reference point every ``ref_every`` runs so slow drift is regressable.
    ref_every: int = 8
    seed: int = 0                     # RNG seed for operand fills + sweep shuffle


@dataclass(frozen=True)
class TypeBParams:
    """Knobs for the B0 Type-detector decision gate (code/typeb/, scripts/plot_b0_*).

    Track B (Type) asks "training vs inference?" — a *temporal* signature, not an
    amplitude (spec.md Sec. 4 B0). The discriminator is a spectral line at the
    training iteration cadence f0 (the f_step line) that the inference null lacks.

    PLACEHOLDER / B0-LOCAL. These drive a *minimal* synthetic generator
    (code.typeb.synth) whose only job is to exercise the two detectors on a
    decision-gate ROC. The frequency bands and phase ratio follow Ko et al.
    (arXiv 2508.16457, eqs 1-10) so the structure is faithful, but this is NOT
    the shared Ko et al. trace generator (that is Task 1.1, built in a separate
    worktree and reused by B1). When it merges, the detectors — which consume
    only (t, P_obs) arrays — plug onto it unchanged; only the script's import of
    the generator changes. Absolute amplitudes here are illustrative.

    Sampling must resolve f0 ~ 1 Hz: fs >> 2*f0 (Nyquist), and the window must be
    long enough for many iterations so the line is resolvable (df ~ 1/duration).
    """

    # --- sampling / window ---
    fs: float = 20.0               # meter sampling rate [Hz] (Nyquist >> f0)
    duration_s: float = 300.0      # trace length [s] (df ~ 1/duration resolution)

    # --- training signature (Ko et al. eq-1..10 training band) ---
    f0_train_lo: float = 0.5       # iteration-cadence f0 band [Hz], training
    f0_train_hi: float = 1.5
    duty_train_lo: float = 0.55    # compute/comm phase ratio r (Ko et al.), train
    duty_train_hi: float = 0.80
    amp_train: float = 30.0        # iteration power swing [W] (line amplitude)
    n_harmonics: int = 3           # square-ish iteration -> fundamental + harmonics

    # --- inference null (in-band power comparable to training, but DIFFUSE) ---
    # The hard null (spec.md Sec. 4 B1 premise 2): batched inference / prefill-decode
    # / MoE routing put quasi-periodic *broadband* power into the same f0 band, with
    # no single concentrated line. Set the in-band RMS comparable to the training
    # line's RMS so the detectors must discriminate concentration-vs-diffuseness,
    # not mere presence of power -- the only regime in which the gate is meaningful.
    amp_infer_band: float = 14.0   # diffuse in-band broadband RMS [W]
    amp_infer_slow: float = 16.0   # slow out-of-band batch swing [W]
    f0_infer_lo: float = 0.03      # slow prefill/decode / batch cadence [Hz]
    f0_infer_hi: float = 0.15

    # --- shared baseline + noise ---
    P_base: float = 250.0          # mean board power [W]
    intra_sigma: float = 8.0       # intra-phase Gaussian fluctuation [W] (Ko et al.)
    sigma_eta: float = 4.0         # off-chip meter noise std [W]

    # --- adversary / honest-smearing knobs (the regime axis of the gate) ---
    # f0 random-walk amplitude over the window [Hz]: 0 = stationary line (matched
    # filter's best case); > 0 = wandering line (the CW-Viterbi regime, also the
    # jitter-adversary regime, plan Sec. 3 B0). This is the axis the gate sweeps.
    wander_hz: float = 0.0
    phase_jitter: float = 0.05     # per-iteration phase jitter sigma [rad] (e^{-w^2 s^2})

    # --- detector search band (around the training f0) ---
    band_lo: float = 0.3           # Hz; matched-filter / Viterbi search band
    band_hi: float = 1.7

    seed: int = 0                  # RNG seed for reproducible traces


@dataclass(frozen=True)
class KoTypeBParams:
    """Ko F(t) -> observed power glue + the B1 gate knobs (task 3).

    B1 re-runs the B0 Type detectors (which consume only (t, P_obs)) on the
    FAITHFUL Ko generator (code.ko_workload) instead of the B0-local placeholder
    (code.typeb.synth) -- turning B0's method decision into real numbers. The glue
    is the forward model P = r*F + P0 + meter noise (code.forward), exactly as
    validated in tests/test_typeb.py::test_detectors_fire_on_ko_training and used in
    code.scenarios. Constants below DELIBERATELY mirror those single sources:
    f_peak_frac == scenarios.F_PEAK_FRAC, eta_scale == scenarios.ETA_SCALE_METER,
    r == FloorParams.r_lo, and fs/band/sigma_eta == TypeBParams. F_peak itself reads
    FloorParams.F_max at call time rather than being copied here.

    Exposes ``band_lo``/``band_hi`` so the gate's score_population (which reads only
    those two fields off its params arg) consumes this in place of TypeBParams.
    """

    # sampling / window (== TypeBParams.fs / duration_s; resolves f0 ~ 1 Hz)
    fs: float = 20.0
    duration_s: float = 300.0

    # forward-model operating point (single-GPU declared point)
    r: float = 1.0e-12             # J/FLOP (== FloorParams.r_lo, A100 fp16 ballpark)
    P0: float = 250.0              # W board/overhead power (== TypeBParams.P_base)
    f_peak_frac: float = 0.85      # F_peak = frac * FloorParams.F_max (scenarios.F_PEAK_FRAC)
    eta_scale: float = 0.15        # sub-ms eta averaged below the 20 Hz meter (scenarios.ETA_SCALE_METER)
    sigma_eta: float = 4.0         # off-chip meter-noise std [W] (== TypeBParams.sigma_eta)

    # detector search band [Hz] (== TypeBParams.band_lo/band_hi)
    band_lo: float = 0.3
    band_hi: float = 1.7

    # gate sweep knobs
    n_each: int = 200              # traces per class per drift level
    target_far: float = 0.05       # operational false-alarm rate
    seed: int = 0


@dataclass(frozen=True)
class MeterParams:
    """Observation-channel (meter) map knobs (ST2; plan-for-paper-2 §2).

    The device emits P_device = r F + P0 (code.forward); an external sensor sees

        P_meter(t) = (h * P_device)(t) + P_other(t) + eta(t)

    where h collects power-delivery dynamics, meter integration, reporting
    filters and sampling, P_other is interference (controller limit cycle,
    baseline wander) and eta is meter noise. :func:`code.observation.apply_meter`
    realises this map; composition order is documented there.

    EVERY default is an exact no-op that draws ZERO RNG — apply_meter with
    ``MeterParams()`` returns (t, P) unchanged and leaves the generator state
    untouched (the ST2 sweeps run meter-off for b0/b1 comparability, plus one
    dedicated meter-robustness family with hostile features on).
    """

    # --- LTI stage (device grid) ---------------------------------------------
    # Butterworth low-pass on the device-grid power (power-delivery + reporting
    # bandwidth). None = no filter.
    lp_cutoff_hz: float | None = None
    lp_order: int = 2
    # Transfer-function notch near the cadence (anti-resonance in the delivery
    # network). None = no notch; notch_q is the quality factor.
    notch_hz: float | None = None
    notch_q: float = 5.0
    # Notch depth as a linear blend between the unfiltered and fully-notched
    # signal: x -> x + notch_depth * (notch(x) - x). 1.0 = the full iirnotch
    # null (the original behaviour); 0.0 = notch off; intermediate values give
    # a partial anti-resonance. Only consulted when notch_hz is not None, so the
    # exact-no-op invariant is untouched.
    notch_depth: float = 1.0
    # Stable meter gain (calibration factor).
    gain: float = 1.0

    # --- integrate + sample ---------------------------------------------------
    # Trailing boxcar integration window [s] (meter averages the recent past);
    # 0.0 = no integration.
    integ_window_s: float = 0.0
    # ZOH decimation of the integrated signal onto a sample_hz grid. None = keep
    # the device grid. Decimation WITHOUT lp_cutoff_hz is the deliberate
    # aliasing case (plan §2: "sample cadence and aliasing").
    sample_hz: float | None = None

    # --- additive interference + noise (meter grid) ---------------------------
    # Periodic controller interference (template: the measured A100 power-
    # management limit cycle, ~0.31-0.45 Hz): a zero-mean square-ish wave of
    # peak-to-peak controller_amp [W] at controller_hz with duty
    # controller_duty; controller_wander_hz lets the line wander slowly (OU
    # excursion of the instantaneous frequency). amp = 0 or hz = 0 disables.
    controller_hz: float = 0.0
    controller_amp: float = 0.0
    controller_duty: float = 0.5
    controller_wander_hz: float = 0.0
    # OU baseline wander (time-varying operating point): marginal std [W] and
    # correlation time [s]. sigma = 0 disables.
    baseline_sigma: float = 0.0
    baseline_tau_s: float = 600.0
    # Meter noise eta: AR(1) with marginal std sigma_eta [W] and correlation
    # time eta_tau_s [s]; 0 = white (reuses code.noise.WhiteNoise/AR1Noise).
    sigma_eta: float = 0.0
    eta_tau_s: float = 0.0

    # Index of the meter's independent RNG substream. apply_meter itself uses
    # the generator it is handed; the ST2 sweep harness uses this field to
    # derive a per-trace child stream so meter noise is decoupled from the
    # workload draw.
    seed_stream: int = 0


@dataclass(frozen=True)
class St2Params:
    """ST2 de-periodicisation frontier sweep knobs (task 20.6; plan §4-5).

    One attack family per level grid below; the registry that maps families to
    generator knobs lives in :mod:`code.typeb.st2_attacks`, the sweep harness
    in :mod:`code.typeb.st2`. Level grids deliberately echo earlier sweeps so
    the synthetic frontier keys onto existing anchors:

    - ``jitter_levels`` / ``work_levels``: superset of B2Params'
      spoof/work-jitter sigma grid (0.05-0.5) extended to 0.7, so measured
      throughput-overhead anchors exist at every shared level;
    - ``drift_levels``: within the B1 gate's 0-1.5 Hz drift grid
      (plot_b1_roc.py), sampled log-ish;
    - ``shape_levels``: == B2Params.shaped_fidelity_levels (measured phi);
    - ``dilute_shares``: within the b1agg dominant-share grid (0.2-0.9);
    - ``relocate_f0s``: cadences moved toward/onto the detector search-band
      edges (band 0.3-1.7 Hz; honest Ko band 0.5-1.5);
    - ``phase_levels`` / ``harmonic_levels``: analytic-only families (no
      hardware anchor) spanning gentle to line-destroying.

    ``meter``: the observation channel every non-meter family runs through.
    The default ``MeterParams()`` is the exact no-op — the harness maps it to
    ``meter=None`` in the builders so sweeps stay byte-comparable with the
    b0/b1 gates (glue-owned white noise). ``meter_variants`` is the dedicated
    meter-robustness family: named HOSTILE channels the nominal honest
    training must survive. Each variant carries its own ``sigma_eta`` because
    noise ownership moves to the meter (code.observation): without it the
    metered traces would be noiseless, flattering every detector.
    """

    n_each: int = 200                       # traces per class per level
    target_fars: tuple[float, ...] = (0.05, 0.01)   # 0.01 = 1/200 granularity
    seed: int = 0

    # Observation channel for all non-meter families (default: exact no-op ==
    # the b0/b1-comparable meter-off path; see class docstring).
    meter: MeterParams = MeterParams()

    # --- per-family level grids ------------------------------------------------
    jitter_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7)
    work_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7)
    drift_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.5)
    phase_levels: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
    harmonic_levels: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)  # seconds
    shape_levels: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)      # phi
    relocate_f0s: tuple[float, ...] = (0.35, 0.45, 0.6, 1.4, 1.6, 1.65)  # Hz
    dilute_shares: tuple[float, ...] = (0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9)

    # --- meter-robustness family: named hostile observation channels ----------
    # (name, MeterParams). sigma_eta = 4.0 W mirrors KoTypeBParams.sigma_eta
    # (the glue noise the meter replaces); hostile knobs on top:
    #   notch_at_cadence      transfer-function notch at the Ko band centre
    #                         (1.0 Hz), low Q so it bites across the band;
    #   integrating_1hz       1 s trailing boxcar + 1 Hz ZOH sampler (a slow
    #                         facility meter; band 0.3-0.5 Hz survives Nyquist);
    #   controller_on         A100-power-management-style limit cycle at
    #                         0.38 Hz inside the search band, 30 W pk-pk
    #                         (~the line swing), slowly wandering;
    #   coloured_noise_heavy  3x AR(1) meter noise (tau 2 s) + 20 W OU
    #                         baseline wander (tau 60 s).
    meter_variants: tuple[tuple[str, MeterParams], ...] = (
        ("notch_at_cadence",
         MeterParams(notch_hz=1.0, notch_q=1.5, sigma_eta=4.0)),
        ("integrating_1hz",
         MeterParams(integ_window_s=1.0, sample_hz=1.0, sigma_eta=4.0)),
        ("controller_on",
         MeterParams(controller_hz=0.38, controller_amp=30.0,
                     controller_duty=0.4, controller_wander_hz=0.05,
                     sigma_eta=4.0)),
        ("coloured_noise_heavy",
         MeterParams(sigma_eta=12.0, eta_tau_s=2.0,
                     baseline_sigma=20.0, baseline_tau_s=60.0)),
    )


@dataclass(frozen=True)
class St2MeterBoundaryParams:
    """Meter-requirement boundary sweep (Phase 2; plan §2, spec.md "minimum
    meter specification").

    Generalises the ST2 ``meter`` family (four named hostile variants) to a
    dense grid of observation channels, run on HONEST training only, to locate
    where each detector class dies as the channel degrades between the nominal
    20 Hz meter and the 1 Hz integrating sampler. Two orthogonal sweeps share
    one flat cell list (see :mod:`code.typeb.meter_boundary`):

    - the main 2-D grid ``sample_hz_grid`` x ``integ_window_grid`` (ZOH sample
      cadence x trailing-boxcar integration window), no notch;
    - a notch sub-sweep ``notch_hz_grid`` x ``notch_depth_grid`` at the nominal
      20 Hz sampler (in-band transfer-function anti-resonance), fixed Q.

    Every cell carries ``sigma_eta`` W of meter noise (noise ownership moves
    into the meter, matching St2Params.meter_variants — otherwise the metered
    traces would be noiseless and flatter every detector). ``n_each`` /
    ``target_fars`` / ``seed`` mirror St2Params so the boundary sweep is scored
    at the same operating point as the frontier.
    """

    n_each: int = 200                       # traces per class per cell
    target_fars: tuple[float, ...] = (0.05, 0.01)
    seed: int = 0
    sigma_eta: float = 4.0                  # W; == KoTypeBParams.sigma_eta

    # Main grid: ZOH sample cadence [Hz] x trailing boxcar window [s]. The
    # nominal channel is fs=20, integ=0; integrating_1hz sits at fs=1, integ=1.
    sample_hz_grid: tuple[float, ...] = (20.0, 10.0, 5.0, 2.0, 1.0, 0.5)
    integ_window_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

    # Notch sub-sweep (at the nominal 20 Hz sampler): notch centre across the
    # Ko cadence band (f0 ~ U(0.5, 1.5) Hz) x blend depth, fixed Q.
    notch_hz_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
    notch_depth_grid: tuple[float, ...] = (0.5, 0.9, 1.0)
    notch_q: float = 1.5                    # == St2Params notch_at_cadence Q


@dataclass(frozen=True)
class Rung2Params:
    """Rung 2 training-vs-inference classification (Phase 2; plan §3.3).

    Rung 1 asks "is there iteration-structured cyclicity?"; Rung 2 asks the
    conditional-classification question "is this trace more training-like than
    the STATED inference population?" — a specified statistical task at the cost
    of an explicit inference null (:func:`code.ko_workload.inference_F`). Three
    decision rules are compared on ONE fixed interpretable physics feature vector
    (:mod:`code.typeb.rung2_features`, the Rung-1 quantities), so the reader sees
    where power comes from:

      (i)  a prespecified physics score (no label fitting: each feature oriented
           larger = more training-like, standardised by the stated inference
           null's robust spread, summed with ``physics_weights``);
      (ii) a model-based discriminant FITTED to the scenario generator (LDA /
           logistic on the physics vector; StratifiedKFold OOF scores — one
           feature vector per trace, so no window-leakage concern);
      (iii)a flexible LEARNED reference (:mod:`code.typeb.rf_baseline`, the
           Rahman RandomForest on its own statistical-shape features) — exposes
           how much comes from the null choice rather than the physics method.

    Reporting is kept separate (plan §3.3): Rung-2 FPR/FNR under the stated
    workload population (train vs inference null through ``meter``); TRANSFER —
    fit on the nominal population, evaluate zero-shot when hardware / meter /
    workload parameters leave it (``transfer_shifts``); and the SEMANTIC
    falsification controls (:mod:`code.typeb.rung2_scenarios`), the paper's
    central question of what the meter can vs cannot certify.

    ``n_each`` / ``target_fars`` / ``seed`` / ``meter`` mirror St2Params so Rung 2
    is scored at the same operating point as the frontier. The heavy sweep runs
    on Slurm (crc-seeded cells, code.scripts.rung2_eval); local runs are smoke
    sizes only (repo policy).
    """

    n_each: int = 200                       # traces per class (train / infer)
    target_fars: tuple[float, ...] = (0.05, 0.01)
    seed: int = 0
    n_folds: int = 5                        # StratifiedKFold folds for rule (ii)/(iii)

    # Observation channel for the stated population (default: exact no-op ==
    # the b0/b1-comparable meter-off path). Transfer meter shifts swap in a
    # hostile St2Params.meter_variant by name.
    meter: MeterParams = MeterParams()

    # Rule (i) per-feature weights in the prespecified physics score (in
    # code.typeb.rung2_features.FEATURE_NAMES order). None == unit weights (all
    # features oriented so larger = more training-like, so +1 each).
    physics_weights: tuple[float, ...] | None = None

    # --- transfer / domain-shift grid -----------------------------------------
    # Each shift is (name, ((param, value), ...)). The harness interprets:
    #   f_peak_frac / duration_s / eta_scale -> override the KoTypeBParams glue;
    #   f0_lo / f0_hi                         -> override the KoWorkloadParams band;
    #   meter_variant                         -> pick that St2Params.meter_variant.
    # Fitted rules train on the nominal population, then score the shifted
    # population zero-shot; the prespecified rule is simply re-evaluated.
    transfer_shifts: tuple[tuple[str, tuple[tuple[str, float | str], ...]], ...] = (
        ("f_peak_lo",        (("f_peak_frac", 0.70),)),
        ("f_peak_hi",        (("f_peak_frac", 0.95),)),
        ("duration_short",   (("duration_s", 150.0),)),
        ("band_shift_hi",    (("f0_lo", 0.8), ("f0_hi", 1.8))),
        ("meter_controller", (("meter_variant", "controller_on"),)),
        ("meter_coloured",   (("meter_variant", "coloured_noise_heavy"),)),
    )

    # --- semantic falsification control knobs (plan §3.3) ---------------------
    # gradient_only: forward/backward WITHOUT the optimizer-step swing -> the
    # per-phase level deltas collapse toward flat (attenuated mu/sigma_delta_tr).
    grad_only_mu_delta: float = 0.05
    grad_only_sigma_delta: float = 0.02
    # nonml_kernel_loop: a training-SHAPED non-ML kernel with the same cadence
    # but a rounded (comb-suppressed) waveform (training_F harmonic_smooth_s).
    nonml_harmonic_smooth_s: float = 0.5
    # controller_cycle: a bare periodic NON-COMPUTE load (F built directly:
    # base + square limit cycle, no training/inference structure). Amplitudes
    # are fractions of f_peak (the KoWorkloadParams convention).
    controller_hz: float = 0.4
    controller_amp: float = 0.3
    controller_duty: float = 0.4
    controller_base: float = 0.5
    # async_training: genuine training strongly de-periodicised (OU centre-freq
    # drift + Brownian boundary slip) — the scoped-OUT efficient-family edge.
    async_f0_drift_hz: float = 1.2
    async_phase_slip_sigma: float = 0.15
    # periodic_inference: the inference null modulated by a PERIODIC request
    # envelope (a periodic request generator), req rate [Hz] and depth (rel.).
    periodic_inf_req_hz: float = 0.9
    periodic_inf_amp: float = 0.5
    # coresident_mixture: dominant training share of the co-resident aggregate.
    coresident_share: float = 0.5


@dataclass(frozen=True)
class B2Params:
    """B2 hardware campaign (task 5): measured Type signatures on the A100.

    Capture-side knobs for code.bench.powertrace + code.b2.workloads and the
    analysis scripts. fs_uniform/band/duration/target_far DELIBERATELY mirror
    KoTypeBParams so measured traces are scored at the exact operating point
    the synthetic B0/B1 gates used; f_step_lo/hi mirror KoWorkloadParams.f0_lo/
    f0_hi so the measured training population samples the same cadence band.
    NVML power refreshes at ~50-100 ms on the A100 (notes/a2-nvml-probe.md);
    poll_hz = 50 oversamples that refresh 3-5x so sensor_cadence can measure
    it rather than assume it.
    """

    # power capture
    poll_hz: float = 50.0          # logger poll rate (oversamples the sensor)
    fs_uniform: float = 20.0       # resample grid for the detectors (== KoTypeBParams.fs)
    duration_s: float = 300.0      # per-trace capture window (== KoTypeBParams.duration_s)
    warmup_s: float = 30.0         # thermal/clock soak before logging starts

    # detector operating point (== KoTypeBParams / TypeBParams)
    band_lo: float = 0.3
    band_hi: float = 1.7
    target_far: float = 0.05
    # measured A100 traces carry a ~0.31-0.45 Hz power-management limit-cycle
    # component under time-varying load (2026-07-04 signatures campaign),
    # ~10x the f_step line. The line-only band starts above it, so the hard
    # question -- is the *iteration line itself* detectable -- is scored
    # without that assist. Costs the bottom of the f_step draw range (0.5).
    band_line_lo: float = 0.55

    # measured populations
    n_each: int = 24               # traces per class (FAR granularity 1/24 vs 0.05)
    f_step_lo: float = 0.5         # target iteration-cadence band [Hz]
    f_step_hi: float = 1.5         #   (== KoWorkloadParams.f0_lo/f0_hi)

    # LM workload (single-A100 scale; real fwd/bwd/AdamW; batches are random
    # on-GPU tokens so no dataloader/disk periodicity contaminates the line)
    n_layer: int = 8
    d_model: int = 1024
    n_head: int = 16
    seq_len: int = 512
    vocab: int = 32768
    micro_batch: int = 8

    # serving null (continuous-batching decode; Orca/vLLM-style provenance,
    # measured counterpart of KoInferenceParams). 64 slots: a *loaded* server
    # (smoke at 32 was host-bound at 111 W; bigger decode batch = more GPU
    # work per step, the realistic high-load regime)
    serve_slots: int = 64
    serve_mean_out_tokens: int = 200
    serve_prompt_lo: int = 64
    serve_prompt_hi: int = 512
    serve_arrival_mean_s: float = 0.25   # Exp wait before a completed slot refills

    smoke_duration_s: float = 120.0      # shortened per-trace window, smoke phase

    # 5.2 passive-spoofing sweep (jitter = KoWorkloadParams.sigma_jitter axis;
    # drift = f0_drift_hz axis; shift moves f_step below band_lo entirely)
    spoof_jitter_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5)
    spoof_drift_levels: tuple[float, ...] = (0.2, 0.5, 1.0)
    spoof_shift_hz: float = 0.2    # out-of-band target cadence (< band_lo)
    spoof_n_each: int = 8          # positives per spoof level

    # 6.2 top-up: extra work-jitter positives at the line-band-leaky levels
    # only (halves TPR granularity to 1/16 where the Viterbi leak lives).
    # Names continue the base indices (_008..) and the rng stream is offset,
    # so nothing collides with the original 72 workjitter traces.
    topup_jitter_levels: tuple[float, ...] = (0.1, 0.2, 0.35)
    topup_drift_levels: tuple[float, ...] = (0.2, 0.5)
    topup_n_each: int = 8
    topup_seed_offset: int = 1

    # 6.1 shaped idle: power-matched busy-work in the pads, swept over
    # matching fidelity phi (pad target power = P_idle + phi*(P_train -
    # P_idle)) on two timing-spoof bases — the strongest in-band jitter
    # level (must be a 5.2 sweep level so the phi=0 anchor cross-checks the
    # existing spoof traces) and the band shift. Levels are geometric in
    # *residual* swing (~280/140/56/14/0 W nominal), bracketing the ~11 W
    # honest ripple near phi=0.95 where the detection transition must live.
    # The chop quantum is randomized per cycle so the 10 Hz ZOH sampler
    # can't alias a coherent chop into the 0.3-1.7 Hz band.
    shaped_fidelity_levels: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    shaped_jitter_sigma: float = 0.35
    shaped_cal_s: float = 10.0     # per-probe calibration window (train/idle/burst)
    shaped_quantum_lo_s: float = 0.015
    shaped_quantum_hi_s: float = 0.035
    shaped_m: int = 4096           # pad-kernel matmul dim (== duty-drive's)

    # 9.2 distribution-matching adversary via DILUTION (Rahman et al.
    # arXiv 2606.19262 class K/L, "training-as-inference"): bury real training
    # under the serving null itself. Task 9.1 showed every cadence-only spoof
    # leaves the gross power ENVELOPE (~219-370 W) far above the serving null
    # (~119 W), so a power-only amplitude RF is unbeaten. The one move that
    # collapses it is to match the power *distribution* to the null while still
    # training. We realise it as dilution: interleave TrainLoop micro-steps
    # among ServeLoop decode steps finer than the ~100 ms NVML integration
    # window, so most windows read pure serving and the aggregate distribution
    # (mean AND higher moments -- the dilutant is real serving) matches the null
    # by construction. The residual optimizer-step cadence is the tracker's last
    # tell. Fidelity phi sweeps the dilution: phi=0 = the adversary's pure
    # (undiluted) training anchor; phi->1 = mostly serving (~119 W). Mirrors the
    # 6.1 shaped sweep so the two plot on one axis. (The honest ~350 W / mb=8
    # reference is the signatures population, captured separately.)
    dilute_fidelity_levels: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    dilute_n_each: int = 8         # positives per fidelity level (== spoof_n_each)
    dilute_cal_s: float = 10.0     # per-probe calibration window (train/serve)
    # Train-side micro-batch for the ENTIRE dilute sweep (serving is unaffected
    # -- it batches by serve_slots, not micro_batch). The first capture at the
    # global micro_batch=8 stalled: one train micro-step (~0.20 s) exceeds the
    # ~0.1 s NVML window, so it reads as a sparse ~350 W SPIKE (trivial for a
    # peak detector, hostile to a coherent line-tracker) rather than the
    # intended continuous low-amplitude cadence, and the envelope floored ~50 W
    # above the null. A smaller batch shrinks t_micro below the meter window so
    # the step averages down; dilute_schedule adapts automatically (smaller
    # t_micro -> larger grad_accum -> more, shorter steps per period). Probe
    # (14273112, 2026-07-14) picked mb=1: it pulls the envelope onto the null
    # (mean 121 W vs 114 W null, KS 0.17 vs 0.40 at mb=8) -- the amplitude tell
    # 9.1 called unbeatable is erased. NB the residual stays a narrow periodic
    # SPIKE, not a ripple: at phi->1 grad_accum floors to 1, so a serving-matched
    # ~1 Hz cadence is one short high-power burst/period by construction (the
    # envelope-vs-spike dilemma; see notes/b2-dilution-9.2-findings.md).
    dilute_micro_batch: int = 1
    # Interleave quantum: number of consecutive serve decode steps between
    # train micro-steps is chosen per level, but the serve/train step lengths
    # themselves (~10-40 ms) already sit below the 100 ms NVML window; this caps
    # how many serve steps we sprinkle per train micro-step so one train
    # micro-step never dominates a meter sample (validated by the GPU test).
    dilute_max_serve_per_micro: int = 64
    dilute_seed_offset: int = 2    # disjoint rng stream from spoof/shaped/topup

    # 9.3 LoRA positive class: low-footprint fine-tuning. Rahman et al.
    # (arXiv 2606.19262) report LoRA + dilution as their single sub-50% cell
    # (42.6% LOO). A LoRA fine-tune leaves the forward pass -- the dominant
    # power draw -- unchanged, but freezes the base weights (cheaper backward:
    # no base weight-gradient matmuls) and optimises only the small low-rank
    # adapters (a tiny AdamW step). The optimizer-step boundary is what
    # modulates power into the f_step line, so the cadence tell is plausibly
    # WEAKER than full fine-tuning -- the empirical question this phase settles
    # (either the tracker still catches it, or an honest shared-limitation
    # backstopped by the amount query). RANK is the footprint knob (adapters
    # are negligible vs the base at any rank, so the sweep is expected ~flat --
    # a clean rank-invariance result either way). alpha sets the update scaling
    # alpha/rank; targets are attention-only (qkv/proj), the standard LoRA
    # placement. Negatives + the full-FT anchor reuse the signatures phase, so
    # only the positives are captured here. Mirrors the dilute block's shape.
    lora_ranks: tuple[int, ...] = (4, 16, 64)
    lora_n_each: int = 8           # positives per rank (== dilute_n_each)
    lora_alpha: float = 16.0       # update scaling alpha/rank
    lora_targets: tuple[str, ...] = ("qkv", "proj")   # attention adapters
    lora_seed_offset: int = 3      # disjoint rng stream (topup=1, dilute=2)

    # 5.3 cadence survival: commanded duty-drive line + decimation ladder
    drive_freqs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0)
    drive_duration_s: float = 120.0
    decimate_fs: tuple[float, ...] = (10.0, 5.0, 2.0, 1.0, 0.5)

    # 6.4 sequential verifier (code.typeb.sequential): 60 s blocks keep ~6
    # overlapped Viterbi frames per block yet give 5 exchangeable blocks per
    # 300 s trace (24 serving traces -> 120 null blocks, conformal floor
    # 1/121); horizon 1800 s = six chained traces answers "does watching
    # longer recover the line band"; kappa = 0.5 is the standard sqrt
    # p-to-e calibrator.
    seq_block_s: float = 60.0
    seq_kappa: float = 0.5
    seq_alpha: float = 0.05
    seq_beta: float = 0.05
    seq_horizon_s: float = 1800.0
    seq_n_boot: int = 200
    seq_seed: int = 20260705

    seed: int = 20260704


@dataclass(frozen=True)
class RfBaselineParams:
    """Task 9.1: Rahman et al. (arXiv 2606.19262) RandomForest, power channel only.

    Rahman detect hidden ML training with a RandomForest over 166 windowed
    features from 9 NVML channels @ 1 Hz (headline 98.2%). 9.1 replicates their
    method *restricted to the one channel an off-chip meter sees* — the power
    draw — on our existing B2 A100 traces, reported alongside our Viterbi tracker
    and matched filter. Two purposes: (1) quantify the information gap (how much
    of their 98.2% came from the non-power channels a meter is blind to), and
    (2) establish the supervised ceiling on the trusted channel our physics
    detector must beat. Their code is not public and expects their 1 Hz
    9-channel format, so we replicate the feature vocabulary (App. A.3) and
    RF hyperparameters here rather than reuse it.

    Feature vocabulary (App. A.3), power-channel survivors:
      base statistics (13): mean, std, min, max, p25/50/75/95, IQR, range, CV,
                            skew, kurtosis;
      autocorrelation (5): lags in ``autocorr_lags_s`` (Rahman lag it in samples
                           at 1 Hz; we express in SECONDS so the physical lag is
                           preserved at either rate);
      epoch periodicity (1): dominant-bin power / mean-bin power of the FFT;
      duty / idle (2): fraction above the window midpoint / near the window min.
    ``drop_level`` (the "shape-only" variant) removes the absolute-level features
    (mean, min, max, percentiles, range) — Rahman's own train-vs-infer stage
    keeps only shape/periodicity/autocorrelation terms, and our task IS
    train-vs-infer, so shape-only is the faithful analogue; the full set is
    reported as an upper bound.

    Sampling rate is a first-class axis (``rates_hz``): 1 Hz matches Rahman's
    NVML cadence for the information-gap comparison against their reported 0.982
    (at 1 Hz the ~1.075 Hz iteration line is beyond the 0.5 Hz Nyquist and
    aliases away — an informative result, not a bug), while 20 Hz
    (== B2Params.fs_uniform) is what the physics detectors see, the fair rate for
    the "must beat" comparison.
    """

    # RandomForest hyperparameters (Rahman App. A.3 / their "RF 30 s").
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"

    # windowing (Rahman headline 30 s window, 15 s stride).
    window_s: float = 30.0
    stride_s: float = 15.0
    window_sweep_s: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)

    # autocorrelation lags [s] (Rahman lags 1/2/5/10/20 samples at 1 Hz).
    autocorr_lags_s: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)

    # sampling rates [Hz]: 1.0 = Rahman info-gap arm; 20.0 = must-beat arm.
    rates_hz: tuple[float, ...] = (1.0, 20.0)

    # grouped cross-validation (all windows of a trace share one fold — the
    # anti-leakage guard, since windows within a trace are highly correlated).
    n_splits: int = 5

    # "idle" threshold: a sample is idle if within idle_frac_tol of the window's
    # dynamic range above its minimum (a documented modelling choice).
    idle_frac_tol: float = 0.05

    seed: int = 20260714


@dataclass(frozen=True)
class St1DetectorParams:
    """ST1 gate (paper-2 Phase 0): adaptive structural-detector knobs.

    Drives code/st1/ — the multitaper harmonic F-test, the Dandawate–Giannakis
    (DG) cyclostationary statistic, and (tasks 19.5+) phase resampling and
    sample splitting. The sampling grid and search band DELIBERATELY mirror
    TypeBParams / KoTypeBParams so ST1 detectors score the exact operating
    point the B0/B1/B2 gates used. Full design: the approved Phase 0 plan and
    notes/discussion/power-verification-paths-forward.md Sec. 3.1-3.2.
    """

    # sampling / window (== TypeBParams.fs / duration_s)
    fs: float = 20.0
    duration_s: float = 300.0

    # detector search band [Hz] (== TypeBParams.band_lo / band_hi)
    band_lo: float = 0.3
    band_hi: float = 1.7

    # --- multitaper (Thomson harmonic F-test) ---------------------------------
    nw: float = 4.0                # DPSS time-bandwidth product NW
    n_tapers: int = 7              # K tapers (K <= 2*NW - 1)
    n_fft_pad: int = 4             # zero-pad factor: n_fft = pad * next_pow2(n)
    n_harmonics_comb: int = 3      # harmonic-comb Fisher combination over (f0..H*f0)

    # --- DG cyclostationary statistic ------------------------------------------
    # Prespecified lag set (plan: prespecified recommended over adaptive).
    lag_set: tuple[int, ...] = (0, 1, 2, 3, 5, 8)
    # Long-run covariance estimator: "bartlett" (Newey-West HAC) | "batch_means".
    cov_estimator: str = "bartlett"
    # Bartlett bandwidth as a fraction of N; 0.0 -> the default b = floor(N**(1/3)).
    cov_bandwidth_frac: float = 0.0
    # Explicit Bartlett bandwidth b in LAGS (stage-8 sweep axis, task 19.9);
    # 0 -> use the cov_bandwidth_frac rule above. Takes precedence when > 0.
    cov_bandwidth_b: int = 0
    # Diagonal shrinkage lambda: Sigma <- (1-l)*Sigma + l*diag(Sigma).
    cov_shrinkage: float = 0.05

    # --- phase resampling (task 19.5; carried here so the config is complete) --
    samples_per_cycle: int = 32    # angle-domain grid: samples per 2*pi cycle
    resample_interp: str = "cubic" # "linear" | "cubic" | "sinc" (a swept open choice)
    f_path_floor_hz: float = 0.05  # clip floor on the tracked f0 path [Hz]

    # --- sample splitting (task 19.6) ------------------------------------------
    split_block_s: float = 30.0    # alternating A/B estimation/test block length [s]
    split_guard_s: float = 2.0     # guard gap dropped at each block boundary [s]

    # --- tracker (stages 3-4; stage-8 sweep axis, task 19.9) -------------------
    # Spectrogram window as a multiple of the typeb default (~16 s); 1.0 keeps
    # the exact viterbi_best_path/heldout_freq_path default (nperseg=None path).
    tracker_nperseg_factor: float = 1.0

    # --- semi-coherent DG variant (task 19.10; dg_order_semicoh) ---------------
    # Per-block DG on the de-warped angle series, Fisher-combined over blocks.
    semicoh_block_s: float = 60.0  # time-domain block length [s]


@dataclass(frozen=True)
class St1NullParams:
    """ST1 null suite (code/st1/nulls.py): the traces the detectors must NOT fire on.

    Baseline power and fluctuation scale mirror TypeBParams (P_base, intra_sigma)
    so the nulls live at the same operating point as the B0/B1 populations. The
    controller null mimics the measured A100 power-management limit cycle:
    0.31-0.45 Hz at ~10x the iteration-line amplitude (B2Params.band_line_lo
    provenance note; 2026-07-04 signatures campaign).
    """

    P_base: float = 250.0          # mean board power [W] (== TypeBParams.P_base)
    sigma: float = 8.0             # fluctuation scale [W] (== TypeBParams.intra_sigma)

    # stage 5: stationary Gaussian / coloured
    ar1_phi: float = 0.9           # AR(1) pole
    ar2_peak_hz: float = 0.8       # AR(2) resonant peak, INSIDE the search band [Hz]
    ar2_q: float = 5.0             # resonance quality factor (sets the pole radius)

    # stage 6: stationary non-Gaussian
    t_dof: float = 3.0             # Student-t innovation dof for ar1_t

    # stage 7: locally stationary / controller-like
    tvar_phi_lo: float = 0.5       # AR(1) phi ramp start (time-varying AR)
    tvar_phi_hi: float = 0.95      # AR(1) phi ramp end
    am_walk_frac: float = 0.5      # bounded sigma(t) random-walk depth (fraction)
    ctrl_f_lo: float = 0.31        # controller limit-cycle band [Hz] (measured A100)
    ctrl_f_hi: float = 0.45
    ctrl_amp: float = 40.0         # controller amplitude [W] (~10x line scale)
    ctrl_mu: float = 4.0           # Van der Pol nonlinearity (relaxation regime)


@dataclass(frozen=True)
class St1FarParams:
    """ST1 realised-FAR harness (code/st1/far.py, scripts/st1_far.py).

    Sizing: M = 10^4 null traces per cell resolves a 3x inflation at the 1e-3
    nominal level (Clopper-Pearson CI on k ~ 10 is about [0.5, 1.8]x10^-3).
    Surrogate cells run M = 2,000 x S = 999 and are scoped to 0.05 / 1e-2 only
    (min surrogate p = 1/(S+1) = 1e-3 is degenerate at the 1e-3 level).
    """

    n_null: int = 10_000                 # M null traces per (stage x null x calib) cell
    n_null_surrogate_cells: int = 2_000  # M for surrogate-calibrated cells (task 19.8)
    n_surrogates: int = 999              # S surrogates per trace (task 19.8)
    nominal_levels: tuple[float, ...] = (0.05, 1e-2, 1e-3)
    ci_conf: float = 0.95                # Clopper-Pearson confidence level
    seed: int = 0


@dataclass(frozen=True)
class NpCeilingParams:
    """NP-optimal LRT ceiling for the ST1 detector bake-off (Phase 2, optional).

    The strategy memo (``notes/discussion/method-soundness-and-prior-art.md`` §2.2)
    asks: since we OWN the generators, compute the Neyman–Pearson optimal
    likelihood-ratio detector between the training and null generators and report
    each corpus-free detector as a *fraction* of it — "X% of NP-optimal power at
    Y% of the information cost". This quantifies "is our detector good?".

    Method (user decision 2026-07-24): the **Whittle spectral LRT**. There is no
    closed-form likelihood (both generators are black-box samplers), so we work in
    the frequency domain under the stationary-Gaussian (Whittle) approximation: the
    in-band periodogram ordinates are ~independent Exponentials with mean the PSD
    ``S(f)``, giving a per-hypothesis log-likelihood
    ``ℓ_H(x) = −Σ_f [log S_H(f) + I_x(f)/S_H(f)]``. The class PSDs are estimated
    from the generators by Monte-Carlo (mean periodogram over ``n_mc`` traces): one
    ``S_neg`` per negative class, and a per-f₀ TEMPLATE BANK ``S_tr(f; f0_k)`` for
    the training class (f₀ is the known nuisance, marginalised). The ceiling score
    is ``logsumexp_k ℓ_tr(x|f0_k) − ℓ_neg(x)`` (uniform f₀ prior).

    CAVEAT carried into every report: this is NP-optimal *under the Whittle model*.
    It discards harmonic-phase coherence and the null's non-Gaussian burst/OU
    structure, so the true optimum can exceed it — where a tracking detector
    (Viterbi / DG-order) approaches or beats it at high drift, that is a finding
    about wandering-line structure the spectral template cannot see, not a bug.

    The template bank is built POOLED over the eval drift grid (drift drawn from
    ``drift_grid``), so the ceiling — like the deployable detectors — does not know
    the adversary's drift. Eval mirrors the bake-off (``n_each`` / ``fars`` /
    ``drifts`` == ``scripts/plot_st1_bakeoff.py``) so "fraction of optimal" is
    apples-to-apples. The MC corpus draws from a DISJOINT rng stream
    (``corpus_seed_offset`` off the eval seed): the ceiling is fit once, then frozen
    and applied to the eval populations. Heavy runs go on Slurm (repo policy); local
    runs are ``--smoke`` sizes only.
    """

    # Monte-Carlo corpus for the class PSD estimates (traces averaged per template).
    n_mc: int = 2000
    # Training f₀ template bank: f0_n points across the Ko band [f0_lo, f0_hi].
    f0_n: int = 61
    # Drift grid the template corpus pools over (== bake-off DRIFTS_HZ).
    drift_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.5)
    # Eval populations (mirror scripts/plot_st1_bakeoff.py).
    n_each: int = 200
    fars: tuple[float, ...] = (0.05, 0.01)
    drifts: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.5)
    eval_seed: int = 20260721                # == bake-off seed (parity)
    corpus_seed_offset: int = 90_000_000     # disjoint MC-corpus rng stream
    # Whittle band == the ST1 detector band (DEFAULT.st1.band_lo/hi) at call time.
    psd_floor: float = 1e-12                 # guard log / division against a zero PSD bin


@dataclass(frozen=True)
class Config:
    floor: FloorParams = FloorParams()
    channel: ChannelParams = ChannelParams()
    sim: SimParams = SimParams()
    ko: KoWorkloadParams = KoWorkloadParams()
    tracker: TrackerParams = TrackerParams()
    bench: BenchConfig = BenchConfig()
    typeb: TypeBParams = TypeBParams()
    ko_typeb: KoTypeBParams = KoTypeBParams()
    b2: B2Params = B2Params()
    rf: RfBaselineParams = RfBaselineParams()
    st1: St1DetectorParams = St1DetectorParams()
    st1_null: St1NullParams = St1NullParams()
    st1_far: St1FarParams = St1FarParams()
    meter: MeterParams = MeterParams()
    st2: St2Params = St2Params()
    st2_meter_boundary: St2MeterBoundaryParams = St2MeterBoundaryParams()
    rung2: Rung2Params = Rung2Params()
    np_ceiling: NpCeilingParams = NpCeilingParams()


DEFAULT = Config()
