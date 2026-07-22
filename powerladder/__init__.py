"""analogue-sensors-for-ai-verification — Phase 1 power-only simulator.

Reusable library modules. Standalone entry points live in ``scripts/``.

Stage 1 (this scaffold):
    config    parameter dataclasses (placeholders until A2 bench measurement)
    noise     meter-noise models (white now; AR(1) stub for the Floor-A study)
    forward   the forward model P(t) = r(t) F(t) + P0(t), observed with noise
    inversion energy-integral inversion -> covert-compute identified set
    floor     closed-form verification floor beta (the headline vacuity result)
"""
