"""E1 learning-cost pilot (exploratory): train a small GPT under one micro-step
schedule and log validation loss against tokens consumed.

Every arm with the same seed sees the same initial weights and the same
micro-batches in the same order. Arms differ only in how many micro-batches are
grouped into each optimiser step (scripts/e1/schedules.py), and, for the
best-response arm, in the per-step learning rate. The learning-rate schedule is
indexed by tokens, so all arms share it.

    python scripts/e1/train.py --arm honest --seed 0 --micro-batch 16 \
        --tokens 400e6 --data-dir /fred/oz022/tkimpson/e1_data

Writes results/e1/<arm>_m<m>_s<seed>[_<tag>][_smoke]/{config.json,log.jsonl}. Log rows:
    {"kind": "step",  "step", "tokens", "G", "lr", "loss"}
    {"kind": "eval",  "step", "tokens", "val_loss"}
    {"kind": "probe", "step", "tokens", "small_sq", "big_sq", "b_small", "b_big"}
    {"kind": "done",  ...throughput and G statistics}
See notes/plans/work-variation-hidden-cost-experiments.md (E1, E2).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import schedules

REPO = Path(__file__).resolve().parents[2]

# Model and optimiser (fixed for the pilot; see the plan).
SEQ_LEN = 512
MODEL = dict(n_layer=6, n_head=6, n_embd=384, block_size=SEQ_LEN,
             vocab_size=50304, dropout=0.0, bias=False)
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.02
MIN_LR_FRAC = 0.1
EVAL_BATCH = 64


def lr_at(tokens: float, total: float, lr_max: float) -> float:
    """Linear warmup over the first 2% of tokens, then cosine to 10% at ``total``."""
    warm = WARMUP_FRAC * total
    if tokens < warm:
        return lr_max * tokens / warm
    frac = min((tokens - warm) / (total - warm), 1.0)
    return lr_max * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))


def _grad_vector(model) -> torch.Tensor:
    return torch.cat([p.grad.detach().flatten().float() for p in model.parameters()
                      if p.grad is not None])


def accumulate(model, batches, G: int, loss_fn, *, probe: bool = False,
               ctx=contextlib.nullcontext):
    """Run G micro-steps, each loss divided by G, so the parameters' ``.grad`` hold
    the mean gradient (as the hardware TrainLoop does).

    Returns (mean loss, probe), where probe is None or (mean squared norm of the
    micro-batch gradients, squared norm of their mean) for the B_simple estimator.
    Probing reads gradient differences after each micro-step; it does not change
    the accumulated gradient.
    """
    total = 0.0
    prev = None
    sq = []
    for _ in range(G):
        x, y = next(batches)
        with ctx():
            loss = loss_fn(model, x, y)
        (loss / G).backward()
        total += loss.item()
        if probe:
            cur = _grad_vector(model)
            micro = (cur - prev) * G if prev is not None else cur * G
            sq.append(float(micro.pow(2).sum()))
            prev = cur
    out = None
    if probe:
        out = (float(np.mean(sq)), float(prev.pow(2).sum()))
    return total / G, out


class MicroBatches:
    """Endless, seeded stream of (x, y) micro-batches of ``m`` random windows."""

    def __init__(self, path: Path, m: int, seed: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.m, self.device = m, device
        self.rng = np.random.default_rng([seed, 0xDA7A])

    def __iter__(self):
        return self

    def __next__(self):
        ix = self.rng.integers(0, len(self.data) - SEQ_LEN - 1, self.m)
        x = np.stack([self.data[i:i + SEQ_LEN] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + SEQ_LEN] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x, y


@torch.no_grad()
def evaluate(model, val: np.ndarray, device: str, ctx) -> float:
    """Mean loss over consecutive, non-overlapping windows of the fixed val slice."""
    model.eval()
    n = (len(val) - 1) // SEQ_LEN
    losses = []
    for start in range(0, n, EVAL_BATCH):
        idx = range(start, min(start + EVAL_BATCH, n))
        x = np.stack([val[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in idx]).astype(np.int64)
        y = np.stack([val[i * SEQ_LEN + 1:(i + 1) * SEQ_LEN + 1] for i in idx]).astype(np.int64)
        with ctx():
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        losses.append(loss.item() * len(idx))
    model.train()
    return sum(losses) / n


def _gpt_loss(model, x, y):
    return model(x, y)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=schedules.ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--micro-batch", type=int, required=True, help="sequences per micro-step")
    ap.add_argument("--tokens", type=float, default=400e6, help="token budget D")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "e1")
    ap.add_argument("--eval-every", type=float, default=5e6, help="tokens between evals")
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--probe-every", type=float, default=0,
                    help="tokens between noise-scale probes (0 = off)")
    ap.add_argument("--tag", default="", help="suffix for the run directory (e.g. bnoise)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2M tokens, frequent evals, tagged _smoke")
    a = ap.parse_args()
    if a.smoke:
        a.tokens, a.eval_every, a.val_tokens = 2e6, 0.5e6, 0.2e6
        a.probe_every = a.probe_every and 0.25e6

    from model import GPT, GPTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
           if device == "cuda" else contextlib.nullcontext)

    tag = (f"{a.arm}_m{a.micro_batch}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
           + ("_smoke" if a.smoke else ""))
    out = a.out / tag
    out.mkdir(parents=True, exist_ok=True)

    tok_per_micro = a.micro_batch * SEQ_LEN
    n_micro = int(a.tokens // tok_per_micro)
    total_tokens = n_micro * tok_per_micro
    sched = schedules.make_schedule(a.arm, n_micro, a.seed)   # >= steps needed

    torch.manual_seed(a.seed)                 # same init for every arm of a seed
    model = GPT(GPTConfig(**MODEL)).to(device)
    opt = model.configure_optimizers(WEIGHT_DECAY, a.lr, BETAS, device)
    fwd = model if a.no_compile or device != "cuda" else torch.compile(model)
    train_batches = MicroBatches(a.data_dir / "train.bin", a.micro_batch, a.seed, device)
    val = np.memmap(a.data_dir / "val.bin", dtype=np.uint16, mode="r")[: int(a.val_tokens) + 1]

    config = dict(vars(a), data_dir=str(a.data_dir), out=str(out), model=MODEL,
                  seq_len=SEQ_LEN, weight_decay=WEIGHT_DECAY, betas=BETAS,
                  grad_clip=GRAD_CLIP, warmup_frac=WARMUP_FRAC, min_lr_frac=MIN_LR_FRAC,
                  n_micro=n_micro, total_tokens=total_tokens, device=device,
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                  n_params=model.get_num_params(), torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    log = (out / "log.jsonl").open("w")

    def emit(**row):
        log.write(json.dumps(row) + "\n")

    used, step, tokens = 0, 0, 0
    next_eval, next_probe = 0.0, 0.0
    train_s, train_tok, Gs = 0.0, 0, []
    while used < n_micro:
        if tokens >= next_eval:
            emit(kind="eval", step=step, tokens=tokens, val_loss=evaluate(fwd, val, device, ctx))
            next_eval += a.eval_every
        G = int(min(sched[step], n_micro - used))
        probe = bool(a.probe_every) and tokens >= next_probe and G >= 2
        lr = lr_at(tokens, total_tokens, a.lr) * float(schedules.lr_multiplier(a.arm, [G])[0])
        for group in opt.param_groups:
            group["lr"] = lr

        t0 = time.perf_counter()
        loss, pr = accumulate(fwd, train_batches, G, _gpt_loss, probe=probe, ctx=ctx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        used += G
        tokens += G * tok_per_micro
        step += 1
        Gs.append(G)
        emit(kind="step", step=step, tokens=tokens, G=G, lr=lr, loss=loss)
        if pr is not None:
            emit(kind="probe", step=step, tokens=tokens, small_sq=pr[0], big_sq=pr[1],
                 b_small=a.micro_batch, b_big=G * a.micro_batch)
            next_probe += a.probe_every
        elif step > 3:                       # skip compile warm-up; probes cost extra
            train_s += dt
            train_tok += G * tok_per_micro
    emit(kind="eval", step=step, tokens=tokens, val_loss=evaluate(fwd, val, device, ctx))
    G_arr = np.asarray(Gs)
    emit(kind="done", steps=step, tokens=tokens, tokens_per_s=train_tok / train_s,
         mean_G=float(G_arr.mean()), median_G=float(np.median(G_arr)),
         frac_G_cap=float((G_arr >= schedules.CAP).mean()))
    log.close()
    print(f"{tag}: {step} steps, {tokens:.3g} tokens, {train_tok / train_s:.3g} tok/s")


if __name__ == "__main__":
    main()
