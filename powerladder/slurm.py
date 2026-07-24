"""Slurm array helpers shared by the sweep driver scripts.

The ST1/ST2 driver scripts (`st1_far.py`, `plot_st2_sweeps.py`,
`st2_meter_boundary.py`) all run one work item per array task: an explicit
``--array-id`` (or, on the cluster, ``SLURM_ARRAY_TASK_ID``) selects a single
cell/family out of the deterministically-built work list. Keep that resolution
in one place so the three drivers cannot drift.
"""

from __future__ import annotations

import os


def resolve_array_id(explicit: int | None, n: int) -> int | None:
    """Resolve the single work-item index for one array task.

    Prefers ``explicit`` (the ``--array-id`` flag); falls back to
    ``SLURM_ARRAY_TASK_ID``; returns ``None`` when neither is set (meaning: run
    the full grid, not array mode). Raises ``SystemExit`` if the resolved index
    is outside ``[0, n)`` so an out-of-range array task fails loudly.
    """
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    idx = explicit if explicit is not None else (
        int(env_id) if env_id is not None else None)
    if idx is not None and not 0 <= idx < n:
        raise SystemExit(f"array id {idx} out of range [0, {n})")
    return idx
