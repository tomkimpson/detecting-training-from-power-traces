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
