# Why Rahman et al. is not a long-term verification path — a light critique

**Status:** discussion note (2026-07-23), distilling a conversation on how to
criticise Rahman et al. (arXiv 2606.19262) — a good paper and a good research
project — without overreaching. The claim is not that the paper is wrong; it is
that the *approach* (supervised classifier over rich on-chip telemetry) is not a
good path forward for the long-term verification problem, and that its own
appendices point toward ours. Companion to the deep-dive
[`rahman2606.19262-comparison.md`](rahman2606.19262-comparison.md) and the
framing note
[`physics-informed-vs-learned-contribution.md`](physics-informed-vs-learned-contribution.md).
Intended uses: spin-out positioning (`paper-viterbi/` Phase 4), referee-style
commentary, talk material.

---

## Two precision fixes before criticising

Get these right or the critique is dismissible on contact:

1. **It is not a neural network.** It is a RandomForest (400 trees) over 166
   hand-crafted features from nine NVML counters at 1 Hz, and the learned
   mapping is telemetry → label. The feature vocabulary is in fact
   *semi*-physics-informed (FFT epoch periodicity, cumulative energy,
   memory-plateau features) — the paper drifting toward mechanism without
   committing to it. The right criticism is of the *epistemics* of a fitted
   detector, not of end-to-end learning they didn't do.
2. **The naive "hardware gap" is pre-empted.** Leave-one-GPU-out
   generalisation is ≥ 99.4% across 9 GPU models and 4 generations (Ampere →
   Blackwell). "It won't transfer across hardware" loses to that table. The
   gap that survives is narrower — see §5 below.

## The critique, in order of severity

### 1. The trust boundary is the fatal long-term problem, not the learning

Every bit of their signal flows through NVML — the prover-owned, on-chip,
software-readable estimate of the card's own state. Against their own
*hardware operator* adversary, the entire result is conditional on
hardware-enabled mechanisms (tamper-resistant read path, signed telemetry,
attested monitoring) that they themselves say do not exist yet. As a long-term
verification path the paper does not solve verification — it **relocates** it
into an unsolved hardware-security problem. To their credit they say so openly
("we advance governance from the ML problem to the hardware security problem"),
but it means the headline 98.2% is an accuracy on a channel the adversary who
matters can forge wholesale.

The one-sentence version: *they get nine rich channels by assuming a trust
anchor that doesn't exist; we get one poor channel that is trustworthy
precisely because it is off-chip — and it is deployable today.*

### 2. The robustness is inductive, and their own protocol shows the failure mode

Leave-one-strategy-out is generalisation *within their own 20-family catalog* —
a catalog they authored. It says nothing about strategies outside the space
they imagined. Their white-box round demonstrates the dynamic: an evader
holding the classifier's internals achieved up to 94% evasion against the
un-hardened model, which then had to be retrained. That is a red-queen loop —
the classifier is fooled by each genuinely new attack class until it sees
examples. Retrain-per-round is a fine posture for spam filtering; it is a
strange one for *verification*, where the point is a guarantee the other party
cannot invalidate by being cleverer than your training set.

The counterpoint is our tracker: a fixed public statistic, never retrained,
that handled every adversary we implemented — because its robustness arguments
are conservation-law-shaped (you cannot hide the optimizer cadence without
paying joules or throughput), not empirical bets.

### 3. No deployed-mode error control

Every number in the paper is a per-window supervised accuracy. There is no
false-alarm guarantee on an unlabelled stream, no sequential operation, no
anytime-valid anything. A regulator watching one stream needs "FAR ≤ α no
matter when I stop looking" — what our conformal e-process provides. And we
have a concrete cautionary tale for why fitted models are dangerous in this
seat: our SPRT comparison showed the Wald test wrongly accepting the null in
48% of chains under an adversary its likelihood model didn't anticipate. Same
disease, different symptom.

### 4. Evasion cost is asserted, never measured

They wave at dilution being "economically unviable at frontier scale" but
produce no cost curve. We measured the exchange rate: 2–5.5× energy per
trained token for power shaping that buys a near-zero miss rate. "Concealment
is uneconomic" is a quantitative claim in our paper and a hope in theirs.

### 5. The hardware/workload gaps that actually survive

The defensible version of the generalisation worry:

- NVIDIA-only; the learned mapping is tied to NVML's specific counter
  semantics, so it does not obviously survive a vendor telemetry change, let
  alone TPUs or non-NVML accelerators. Cross-GPU-model transfer *within one
  vendor's telemetry stack* is not hardware generality.
- Single-node only (≤ 8 GPUs, no InfiniBand) — nothing about frontier-scale
  distributed training as it actually runs.
- PyTorch-level adversaries only; no custom CUDA kernels.
- Their own worst case (white-box LoRA + dilution, 42.6% LOO) shows the
  in-catalog robustness already cracking at the low-footprint end.

## Keeping it light: the steelman and the self-critique

**Steelman.** The red-teaming is genuinely serious — five monitor–evader
co-evolution rounds, LOO evaluation, a white-box round; most papers do none of
this. And their breadth embarrasses ours: 9 GPUs vs our one A100, 40 real
inference workloads vs our one serving null, 20 adversary families vs our one
axis. Criticise their generalisation and a reviewer asks us the mirror
question; the honest answer is depth over breadth (guarantees, pricing, a
trusted channel) and it should be said explicitly.

**The paper criticises itself in our direction** — rhetorically the strongest
move. Their App. B wall-plug experiment found CPU-side evasion invisible to
NVML that shows up at the wall, and they conclude layered wall/rack-power
monitoring is needed. Their hardening loop drove them to add cumulative-energy
and power-correlation features, which they call "physics-based
evasion-resistant features," and their §6.1 argues energy is the invariant you
cannot equalise (backward ≈ 2× forward FLOPs). So the critique is not "they're
wrong" but: *follow their own appendices to their limit and you arrive at our
paper — the physics-derived features on the off-chip channel are what remains
when the learned features and the trusted telemetry are stripped away.*

## The caveat that keeps us honest

Our own RF-on-power-only baseline (task 9.1,
`results/b2/rf_summary.json`) showed that on our measured threat model a
Rahman-style forest is at ceiling too — the information gap between learned and
physics-informed detectors was ~0 on that data. So the critique must never be
"learned methods detect worse." It is that they cannot tell you *why* they
will keep working, cannot bound their false alarms in deployment, cannot price
evasion, and stand on a trust anchor that does not exist. Detection accuracy is
the one axis where they are fine — which is exactly why the paper looks strong,
and why the criticism has to be about verification epistemology, not
performance.
