# Handoff — 2026-07-24 (Strategic gate: lower-bound feasibility spike done)

## What happened this session
Ran the **lower-bound feasibility spike** — the kill/continue strategic gate that
was blocking Phase 1 number-freeze. Branch `docs/lower-bound-spike` (not yet
merged / pushed). Deliverable: `notes/discussion/lower-bound-feasibility-spike.md`
plus a lightweight reproducible sanity check.

**Verdict: a bound IS derivable → the CONTRIBUTION side of the fork.** For i.i.d.
phase jitter:
- Coherent line power rolls off as `1/(1+κσ²)`, `κ≈πf₀T`, from phase diffusion
  `D≈(2π)²σ²f₀` (`deperiod.py` anchor). `detector power ≤ TV`, Pinsker → a finite
  covertness threshold `σ*(ε)`.
- **Physics anchor makes it a *lower* bound, not just an upper bound on our
  detector:** the synchronous all-reduce down-phase `T_down` is fixed regardless of
  work variation, so the free-work-jitter escape (≈0 throughput cost) leaves a
  residual comm-cadence line. Reaching `σ*(ε)` costs throughput OR learning
  efficiency — cannot be both zero.

Sanity check `scripts/lower_bound_spike.py` (CPU, seeded, `n_each=80`, ~seconds)
→ `figures/lower_bound_spike.*`, `results/spike/lower_bound_spike.json`:
- coherent-power rolloff form **R²=0.97**; fitted κ≈2950 within ~3× of the
  parameter-free πf₀T≈940 (O(1) Lorentzian-width prefactor);
- work-jitter escape measurably **more** detectable at matched σ than period
  jitter (TV_work > TV_jit for σ≥0.1) — confirms the comm-barrier residual;
- CV≈σ and D≈(2π)²σ²f₀ hold in the small-σ (threshold-relevant) regime.

## Current status
- Branch `docs/lower-bound-spike` — **not yet merged / pushed**. Additive only
  (new script, note, figure, JSON, `tasks.md` tick, this handoff); no library code
  touched.
- `tasks.md` "Lower-bound feasibility spike" ticked with a full Result. With the
  ST0 audit already done, **both strategic gates are now closed** → the paper's
  claim strength is settled on the contribution side.
- **Environment:** this treehouse worktree had NO Python env; created a venv at
  `<scratchpad>/venv` (numpy 2.2.6 / scipy 1.15.3 / matplotlib / sklearn /
  scienceplots from `requirements.txt`). The scratchpad path is session-local — a
  future session in this worktree must recreate it (or use the `.venv/` the prior
  handoff mentions on the shared dev node).
- `pytest -p no:debugging -m "not gpu"` → **5 failed, 171 passed** = the exact
  known baseline (BLAS-sensitive byte-identity digests; env, not a bug). No new
  failures from this session.

## Next steps
1. **`check-PR` + merge `docs/lower-bound-spike` → main.** Self-contained.
   Consider `check-refs`/`check-arxiv-llm-compliance` are NOT needed (no paper
   prose or bib entries touched this session).
2. **Phase 4 write-up of the identifiability section** (plan §7 / §10 Q2): promote
   §3 of the spike note to a **proposition-level covertness cost bound** for the
   i.i.d.-jitter family; firm up the fixed-verifier → tracking/optimal-verifier
   step (the one analytic gap flagged in the note §5); cite `lower_bound_spike.*`.
3. **Open sub-question from the spike:** metric choice (TV vs KL vs Hellinger) for
   the tightest closed-form bound — KL/Hellinger tensorise over the many-iteration
   product law and may beat the TV-via-AUC form used here.
4. **Still-open Phase 1 number-freeze** (unchanged, blocking a paper freeze):
   slurm ST1 surrogate S=999/M=10⁴ incl. stage-6 non-Gaussian nulls; from-scratch
   ST2 re-verification; regenerate the byte-identity digest fixtures in the
   canonical env. Do NOT run sweeps on the dev node — slurm only.
5. **Phase 2 remainder** (from prior handoff, unchanged): merge
   `feat/phase2-meter-boundary` if not yet done; full-resolution frontier with
   cost anchors; Rung 2 (still scaffold).

## Key file locations
- Deliverable: `notes/discussion/lower-bound-feasibility-spike.md`.
- Sanity check: `scripts/lower_bound_spike.py` → `figures/lower_bound_spike.*`,
  `results/spike/lower_bound_spike.json`.
- Reused (unchanged): `powerladder/ko_workload.py`, `powerladder/typeb/deperiod.py`,
  `powerladder/observation.py`, `powerladder/typeb/ko_synth.py` (the trace path the
  script mirrors), `powerladder/config.py` (`KoTypeBParams`, `MeterParams`).
- Context read: `notes/discussion/st0-prior-art-audit.md` §4/§6,
  `notes/discussion/method-soundness-and-prior-art.md` §3/§4,
  `notes/discussion/north-star-and-positioning.md` §8.
