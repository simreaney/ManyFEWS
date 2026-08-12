"""
What-if scenarios: synthetic storms and direct flow overrides.

Two entry points into the pipeline at different depths. :func:`inject_storm`
rewrites the weather and lets the full hydrology respond, which is the honest
route. :func:`constant_flow_samples` skips the hydrology entirely and drives the
emulator from a chosen flow, which is instant and useful for exploring what the
inundation model alone says.

Both exist because a normal forecast for this catchment floods nothing at all:
measured flows run 9.5-30.6 m3/s against a lowest emulator threshold of 50.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import StormConfig
from .weather import WeatherSeries

logger = logging.getLogger(__name__)

__all__ = [
    "inject_storm",
    "inject_storm_ensemble",
    "scale_precip",
    "constant_flow_samples",
]


def inject_storm(
    series: WeatherSeries, issue_time: datetime, storm: StormConfig
) -> WeatherSeries:
    """
    Replace one day's rainfall with a design storm totalling ``storm.total_mm``.

    Mirrors ``generate_river_flows._testStormOverrides``: the target is the UTC
    calendar day ``storm.days_ahead`` days after ``issue_time``, the total is
    spread evenly across that day's buckets, and existing rainfall is *replaced*
    rather than added to. Returns a copy; a no-op if the day is outside the
    series or the storm is disabled.

    .. note::
       ``StormConfig.total_mm`` defaults to 100 mm for parity with the Django
       ``TestModeSettings``, but 100 mm produces a peak p90 flow of about
       36 m3/s - below every emulator threshold, so the map stays empty. Around
       200 mm is the point where flooding appears.
    """
    if not storm.enabled:
        return series

    start = (
        issue_time.astimezone(timezone.utc) + timedelta(days=storm.days_ahead)
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    lo = np.datetime64(start.replace(tzinfo=None), "s")
    hi = np.datetime64(end.replace(tzinfo=None), "s")
    target = np.flatnonzero((series.times >= lo) & (series.times < hi))

    if target.size == 0:
        logger.warning(
            "Storm day %s is outside the forecast range %s..%s; no storm injected",
            start.date(),
            series.times[0],
            series.times[-1],
        )
        return series

    precip = series.data[:, WeatherSeries.PRECIP].copy()
    precip[target] = storm.total_mm / target.size
    logger.info(
        "Injected %.0f mm storm on %s across %d bucket(s) of member %s",
        storm.total_mm,
        start.date(),
        target.size,
        series.member,
    )
    return series.with_precip(precip)


def inject_storm_ensemble(
    forecast: list[WeatherSeries], issue_time: datetime, storm: StormConfig
) -> list[WeatherSeries]:
    """Apply the same design storm to every ensemble member."""
    return [inject_storm(member, issue_time, storm) for member in forecast]


def scale_precip(series: WeatherSeries, factor: float) -> WeatherSeries:
    """Multiply all rainfall by ``factor``, for sensitivity testing."""
    return series.with_precip(series.data[:, WeatherSeries.PRECIP] * factor)


def constant_flow_samples(flow_m3s: float, n: int = 1000) -> np.ndarray:
    """
    A degenerate flow population for driving the emulator directly.

    Every sample is identical, so every percentile collapses to the depth at that
    flow - which is what makes a flow slider behave the way a reader expects.
    """
    return np.full(n, float(flow_m3s), dtype=np.float64)
