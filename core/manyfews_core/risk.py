"""
Reducing an inundation field to a single headline risk number.

Ports ``flood_risk.calculate_risk_percentages``. This is the least defensible
part of the original pipeline and the port reproduces it faithfully rather than
quietly improving it - see :class:`~manyfews_core.config.RiskConfig`.
"""

from __future__ import annotations

import numpy as np

from .config import RiskConfig

__all__ = ["wet_cell_count", "risk_fraction"]


def wet_cell_count(depth: np.ndarray, threshold_m: float = 0.01) -> int:
    """Number of cells wetter than ``threshold_m``, ignoring NaN."""
    return int(np.count_nonzero(np.nan_to_num(np.asarray(depth)) > threshold_m))


def risk_fraction(n_wet: int, cfg: RiskConfig = RiskConfig()) -> float:
    """
    Map a wet-cell count onto 0..1.

    The legacy formula is ``n / (large - channel)``, which the original applies
    without subtracting the channel baseline from the numerator - so a catchment
    at exactly its channel cell count reports 7% risk rather than zero. Two
    further oddities carry over from the Django settings: the constants do not
    match the shipped parameter file at all (the grid holds 302,748 cells, so
    ``large_flood_count`` of 1,440,811 is unreachable and risk saturates around
    0.225), and neither has a documented provenance.

    Set ``cfg.legacy_formula = False`` for the presumably intended
    ``(n - channel) / (large - channel)``.
    """
    span = cfg.large_flood_count - cfg.channel_cell_count
    if span <= 0:
        raise ValueError(
            f"large_flood_count ({cfg.large_flood_count}) must exceed "
            f"channel_cell_count ({cfg.channel_cell_count})"
        )
    numerator = n_wet if cfg.legacy_formula else n_wet - cfg.channel_cell_count
    return float(np.clip(numerator / span, 0.0, 1.0))
