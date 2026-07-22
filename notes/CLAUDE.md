# CLAUDE.md — notes/

Rules for this directory. **Every markdown note lives in exactly one of the three
subdirectories below.** No loose `.md` files at `notes/` root (this file is the only
exception).

## The three categories

- **`plans/`** — forward-looking: what we intend to do and why. Project/phase plans
  and their reviews (`plan-for-paper-2.md`, `plan-for-paper-2-review.md`). A plan and
  its review live side by side.
- **`results/`** — written findings: what we found and what it means. Gate records,
  investigation write-ups (`st1-findings.md`, `issue54-investigation.md`). These are
  prose records with interpretation — **not** the same as top-level `results/`, which
  holds machine-generated experiment outputs (JSON/npz) regenerable from `scripts/`.
- **`discussion/`** — rationale and deliberation: options considered, open questions,
  background memos (`power-verification-paths-forward.md`).

## Rules

1. **New note → pick the single best-fit directory.** If a note mixes findings with a
   plan, split it or file it by its primary purpose.
2. **Date + status header at the top** of every note (existing docs follow this).
3. **Notes are records, not trackers.** Status lives in `tasks.md`, scope in
   `spec.md`, session state in `handoff.md`. A note captures rationale or findings at
   a point in time; prefer appending dated sections over silently rewriting history.
4. **Cross-referencing:** use repo-rooted paths (`notes/plans/…`) from outside
   `notes/`, relative paths within it. Code docstrings and the manuscript cite these
   paths — when moving or renaming a note, grep the whole repo and fix references.
5. **Dangling monorepo citations are deliberate.** Some `powerladder/` docstrings cite
   source-monorepo notes that were not carried over (e.g.
   `development-notes/phase1-plan.md`, `phase2-plan.md`, `b2-extensions.md`). They are
   provenance pointers into `analogue-sensors-for-ai-verification`; leave them as-is.
