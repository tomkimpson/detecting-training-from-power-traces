# Measured cost anchors

Three JSON summaries of **measured** GPU throughput overhead, carried over as static
input data from the source monorepo (`analogue-sensors-for-ai-verification`), where
they were produced by the **B2 single-A100 capability campaign** (real hardware, on
OzSTAR). They are the empirical systems-cost anchors overlaid on the ST2
de-periodicisation frontier (`scripts/plot_st2_frontier.py`).

**These files are NOT regenerable in this repo** — this repo is CPU-only and
synthetic, and the GPU bench harness that produced them was intentionally left behind
in the monorepo. They are committed here as provenance-tracked input data.

| File | What it measures | Frontier family it anchors |
|---|---|---|
| `spoof_summary.json` | Idle-pad realisation of timing jitter / cadence drift (the *expensive* way to de-periodicise) | `jitter`, `drift` |
| `workjitter_summary.json` | Real-work (gradient-accumulation) realisation of the same σ grid — measured ≈ zero throughput cost within noise | `work` |
| `shaped_summary.json` | Amplitude-shaping (shaped-jitter) overhead; captured on a σ=0.35 timing base, so it *upper-bounds* shaping-only cost | `shape` |

Consumed via the frontier script's `--b2-dir` flag:

```
python scripts/plot_st2_frontier.py --b2-dir data/measured_cost_anchors
```

Families without an entry here are analytic/qualitative only on the frontier.

## Provenance: off-grid top-up (2026-10-01)

`spoof_summary.json` and `workjitter_summary.json` were extended with the frontier
levels the July campaign never ran: work jitter 0.7, pad jitter 0.7, and pad drift
0.1 / 0.4 / 0.8 / 1.5 Hz (8 traces each). These were captured on OzSTAR `milan-gpu`
(A100-SXM4-80GB at 500 W, the same model as the original anchors; slurm job
17811800), using the monorepo's `offgrid` phase (branch `feat/b2-offgrid-topup`,
commits `732479a` and `86b412a`). None of the traces were throttled. Each trace's
overhead is measured against its own honest baseline, so no new negatives were
needed. Every pre-existing key is byte-identical to the July values; the new
levels are only appended.

The drift overheads are heavy-tailed (at 0.8 and 1.5 Hz the mean is well above the
median), because a slow random wander occasionally parks a 300 s trace in long pads.
The frontier quotes the mean, as for every other anchor.
