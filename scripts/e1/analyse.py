"""Analyse the E1 learning-cost pilot (exploratory).

Reads results/e1/<arm>_m<m>_s<seed>/log.jsonl for one micro-batch size and
reports, per arm and seed, the extra tokens needed to reach a target validation
loss relative to honest training:

    target         the seed-matched honest run's val loss at TARGET_FRAC * D
    tokens_to_hit  first crossing of the running-minimum val-loss curve (linear
                   interpolation between evals); None if never reached
    cost           tokens_arm / tokens_honest - 1  (a lower bound if never reached)

Because every attack arm missed that target in the pilot, the summary also
reports (added after the runs, so not pre-registered):

    equiv_cost     D / (tokens honest needed to reach the arm's FINAL loss) - 1

It is conservative in the attacker's favour: the honest run at fewer tokens
still has an undecayed learning rate, so it reaches a given loss later than an
honest run with a shorter schedule would.

It also estimates the gradient noise scale B_simple from the second half of the
honest runs' probes (or, if they did not probe, the honest *_bnoise scan at this m) and puts it into the McCandlish penalty used in E0, to
compare predicted with measured cost.

    python scripts/e1/analyse.py --micro-batch 16
        -> results/e1/summary.json, figures/e1_loss_curves.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import noise_scale
import schedules

REPO = Path(__file__).resolve().parents[2]
TARGET_FRAC = 0.75
N_PRED = 200_000            # G draws for the predicted penalty


def tokens_to_target(tokens: np.ndarray, loss: np.ndarray, target: float) -> float | None:
    """Tokens at which the running minimum of ``loss`` first reaches ``target``."""
    run = np.minimum.accumulate(np.asarray(loss, dtype=float))
    hit = np.nonzero(run <= target)[0]
    if hit.size == 0:
        return None
    j = int(hit[0])
    if j == 0:
        return float(tokens[0])
    t0, t1, l0, l1 = tokens[j - 1], tokens[j], run[j - 1], run[j]
    return float(t0 + (t1 - t0) * (l0 - target) / (l0 - l1))


def loss_at(tokens: np.ndarray, loss: np.ndarray, t: float) -> float:
    return float(np.interp(t, tokens, loss))


def predicted_cost(arm: str, b_noise_micro: float, seed: int = 0) -> float:
    """McCandlish extra-token fraction vs honest G = 8 (notes/plans, §2), with
    B_noise in micro-batch units."""
    G = schedules.make_schedule(arm, N_PRED, seed).astype(float)
    h = lambda b: b / (b + b_noise_micro)  # noqa: E731
    honest = h(schedules.BASE_ACCUM) / schedules.BASE_ACCUM
    return float(honest / (np.mean(h(G)) / G.mean()) - 1)


def _read(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open()]
    ev = [r for r in rows if r["kind"] == "eval"]
    return dict(
        tokens=np.array([r["tokens"] for r in ev], dtype=float),
        loss=np.array([r["val_loss"] for r in ev]),
        probes=[r for r in rows if r["kind"] == "probe"],
        done=next((r for r in rows if r["kind"] == "done"), None),
    )


def _b_simple(probes: list[dict]) -> float | None:
    if not probes:
        return None
    terms = [noise_scale.noise_scale_terms(p["small_sq"], p["big_sq"],
                                           p["b_small"], p["b_big"]) for p in probes]
    return noise_scale.b_simple([t[0] for t in terms], [t[1] for t in terms])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--results", type=Path, default=REPO / "results" / "e1")
    ap.add_argument("--fig-dir", type=Path, default=REPO / "figures")
    a = ap.parse_args()
    m = a.micro_batch

    runs: dict[tuple[str, int], dict] = {}
    for arm in schedules.ARMS:
        for d in sorted(a.results.glob(f"{arm}_m{m}_s*")):
            if d.name.endswith(("_smoke", "_bnoise")) or not (d / "log.jsonl").exists():
                continue
            seed = int(d.name.rsplit("_s", 1)[1])
            runs[(arm, seed)] = _read(d / "log.jsonl")
    seeds = sorted({s for (arm, s) in runs if arm == "honest"})
    if not seeds:
        raise SystemExit(f"no honest runs for micro-batch {m} in {a.results}")

    # Noise scale: the second half of the probes (B_noise grows as loss falls;
    # mid-to-late training is where the batch choice matters), from the honest
    # runs if they probed, else from the honest *_bnoise scan at this m. Probing
    # does not change training, so the scan is the same honest configuration.
    def second_half(p):
        return p[len(p) // 2:]

    bnoise_runs = {d.name: _read(d / "log.jsonl")["probes"]
                   for d in sorted(a.results.glob("honest_m*_s*_bnoise"))
                   if (d / "log.jsonl").exists()}
    probes = [p for s in seeds for p in second_half(runs[("honest", s)]["probes"])]
    b_source = "honest runs"
    if not probes:
        probes = [p for name, ps in bnoise_runs.items()
                  if name.startswith(f"honest_m{m}_") for p in second_half(ps)]
        b_source = f"honest_m{m}_*_bnoise"
    b_seq = _b_simple(probes)
    bnoise = {name: _b_simple(second_half(ps)) for name, ps in bnoise_runs.items()}

    table = []
    for (arm, seed), r in sorted(runs.items()):
        h = runs[("honest", seed)]
        D = h["tokens"][-1]
        target = loss_at(h["tokens"], h["loss"], TARGET_FRAC * D)
        t_h = tokens_to_target(h["tokens"], h["loss"], target)
        t_a = tokens_to_target(r["tokens"], r["loss"], target)
        cost = (t_a / t_h - 1) if t_a is not None else (r["tokens"][-1] / t_h - 1)
        t_eq = tokens_to_target(h["tokens"], h["loss"], float(r["loss"][-1]))
        equiv = (r["tokens"][-1] / t_eq - 1) if t_eq else None
        table.append(dict(
            arm=arm, seed=seed, target_loss=target, tokens_to_target=t_a,
            cost=cost, cost_is_lower_bound=t_a is None,
            honest_tokens_to_final=t_eq, equiv_cost=equiv,
            final_loss=float(r["loss"][-1]), final_gap=float(r["loss"][-1] - h["loss"][-1]),
            tokens_per_s=r["done"]["tokens_per_s"] if r["done"] else None,
            mean_G=r["done"]["mean_G"] if r["done"] else None,
        ))

    per_arm = {}
    for arm in schedules.ARMS:
        rows = [t for t in table if t["arm"] == arm]
        if not rows:
            continue
        c = np.array([t["cost"] for t in rows])
        e = np.array([t["equiv_cost"] for t in rows], dtype=float)
        per_arm[arm] = dict(
            n_seeds=len(rows), cost_mean=float(c.mean()),
            cost_sd=float(c.std(ddof=1)) if len(c) > 1 else None,
            any_lower_bound=any(t["cost_is_lower_bound"] for t in rows),
            equiv_cost_mean=float(np.nanmean(e)),
            equiv_cost_sd=float(np.nanstd(e, ddof=1)) if len(e) > 1 else None,
            final_gap_mean=float(np.mean([t["final_gap"] for t in rows])),
            tokens_per_s_mean=float(np.mean([t["tokens_per_s"] for t in rows
                                             if t["tokens_per_s"]] or [np.nan])),
            predicted_cost=(predicted_cost(arm, b_seq / m) if b_seq else None),
        )

    summary = dict(
        micro_batch=m, target_frac=TARGET_FRAC, honest_batch_seq=schedules.BASE_ACCUM * m,
        b_simple_seq=b_seq, b_simple_source=b_source, b_simple_micro=(b_seq / m if b_seq else None),
        b_simple_bnoise_runs_second_half=bnoise, per_arm=per_arm, runs=table,
    )
    out = a.results / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    _plot(runs, seeds, a.fig_dir)


def _plot(runs, seeds, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(REPO))
    from powerladder.plotstyle import apply_house_style
    apply_house_style()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.1, 5.0), sharex=True)
    for arm in schedules.ARMS:
        curves = [runs[(arm, s)] for s in seeds if (arm, s) in runs]
        if not curves:
            continue
        hon = [runs[("honest", s)] for s in seeds if (arm, s) in runs]
        t = curves[0]["tokens"]
        L = np.mean([np.interp(t, c["tokens"], c["loss"]) for c in curves], axis=0)
        H = np.mean([np.interp(t, h["tokens"], h["loss"]) for h in hon], axis=0)
        ax0.plot(t / 1e6, L, label=arm)
        ax1.plot(t / 1e6, L - H, label=arm)
    ax0.set_ylabel("validation loss")
    ax1.set_ylabel("loss minus honest")
    ax1.set_xlabel("tokens [M]")
    ax1.axhline(0, color="k", lw=0.5)
    ax0.legend(fontsize=6)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"e1_loss_curves.{ext}", dpi=200)


if __name__ == "__main__":
    main()
