"""Track B (Type) — the "training vs inference?" detector and its decision gate.

This package is the B0 deliverable (spec.md Sec. 4 B0; tasks.md Phase 2 Task 2):
a written state-space spec plus a *spectral baseline* and a *CW-style HMM/Viterbi*
detector, compared on ROC at a fixed false-alarm rate to decide whether the richer
machinery earns its place (the go/no-go gate).

Layout::

    synth      minimal, B0-LOCAL labeled-trace generator (placeholder for the
               shared Ko et al. generator built in Task 1.1 / a separate worktree)
    ko_synth   B1 builder: Ko et al. F(t) -> labeled power TypeTrace populations
               (the faithful generator behind the B0 placeholder; tasks.md Task 3)
    detectors  spectral_statistic (PSD peak matched filter) and viterbi_statistic
               (spectrogram line-tracker); both consume only (t, P_obs) arrays
    roc        ROC helpers (roc_points, tpr_at_far, auc) — no extra dependencies
    gate       run both detectors over a frequency-wander sweep (figure == tests)

The detectors are generator-agnostic by design: the B1 gate (Task 3) re-runs the
B0 comparison on the faithful Ko generator via ko_synth with no detector change.
"""
