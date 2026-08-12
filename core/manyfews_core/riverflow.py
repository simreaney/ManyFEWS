"""
Running the catchment model: weather in, river flow ensemble out.

Ports ``GenerateRiverFlows`` from the Django module with three substitutions -
the catchment constants that were hardcoded in the function body become
configuration, the parameter file is injected rather than re-read from disk on
every call, and the celery logger becomes a stdlib one.

The uncertainty here is two-dimensional and gets pooled before percentiles are
taken: 100 calibrated rainfall-runoff parameter sets times N weather ensemble
members, giving roughly 1,000 flow samples per forecast time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import CatchmentConfig, DEFAULT_CATCHMENT, DEFAULT_INITIAL_STATE
from .data import PARAMETERS_CSV, data_path
from .hydrology import FAO56, ModelFun
from .weather import WeatherSeries, align_series

logger = logging.getLogger(__name__)

__all__ = [
    "RiverFlowResult",
    "EnsembleFlows",
    "load_parameters",
    "default_initial_state",
    "generate_river_flows",
    "spin_up",
    "run_ensemble",
]

# Surface roughness equivalent to the FAO56 reference crop, used to bring the
# 10 m wind that Open-Meteo reports down to the 2 m the formula wants.
_Z0 = 0.006247
_Z2 = 2.0
_Z10 = 10.0


@dataclass
class RiverFlowResult:
    """Output of one weather realisation through the catchment model."""

    times: np.ndarray  # (n_steps,) datetime64[s]
    flow_m3s: np.ndarray  # (n_steps, n_param_sets)
    rainfall_mm_day: np.ndarray  # (n_steps,)
    pet_mm_day: np.ndarray  # (n_steps,)
    open_water_evap: np.ndarray  # (n_steps,) - E0, which the Django app discards
    state: np.ndarray  # (n_param_sets, 3) end-of-run F0
    member: str = "control"


@dataclass
class EnsembleFlows:
    """
    Every ensemble member's flow, on a common time axis.

    ``flow_m3s`` is (n_members, n_steps, n_param_sets). Pooling the last two axes
    at a given step reproduces exactly what ``flood_risk.run_flood_model_for_time``
    queries out of the database before computing depth percentiles.
    """

    times: np.ndarray  # (n_steps,)
    members: list[str]
    flow_m3s: np.ndarray  # (n_members, n_steps, n_param_sets)
    rainfall_mm_day: np.ndarray  # (n_members, n_steps)
    pet_mm_day: np.ndarray  # (n_members, n_steps)

    @property
    def n_steps(self) -> int:
        return int(self.flow_m3s.shape[1])

    def pooled(self, step: int) -> np.ndarray:
        """All flow samples at one time step: (n_members * n_param_sets,)."""
        return self.flow_m3s[:, step, :].ravel()

    def percentiles(
        self, pcts: tuple[float, ...] = (10.0, 30.0, 50.0, 90.0)
    ) -> np.ndarray:
        """Flow percentiles across pooled samples: (n_steps, len(pcts))."""
        pooled = self.flow_m3s.transpose(1, 0, 2).reshape(self.n_steps, -1)
        return np.percentile(pooled, list(pcts), axis=1).T

    def median_rainfall(self) -> np.ndarray:
        """Median rainfall across members: (n_steps,)."""
        return np.median(self.rainfall_mm_day, axis=0)

    def peak_step(self, pct: float = 90.0) -> int:
        """Index of the step with the highest flow at the given percentile."""
        pooled = self.flow_m3s.transpose(1, 0, 2).reshape(self.n_steps, -1)
        return int(np.argmax(np.percentile(pooled, pct, axis=1)))

    def to_csv(self, path: str | Path) -> None:
        """Write a tidy per-step summary."""
        pct = self.percentiles()
        header = "time,rainfall_mm_day,flow_p10,flow_p30,flow_p50,flow_p90"
        rows = np.column_stack([self.median_rainfall(), pct])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            for t, row in zip(self.times, rows):
                handle.write(f"{t}," + ",".join(f"{v:.6g}" for v in row) + "\n")


def load_parameters(path: str | Path | None = None) -> np.ndarray:
    """
    Load the 100 calibrated PDM parameter sets: ``(n_sets, 4)`` of
    Smax (mm), qmax (mm/day), k (mm/day), Tr (days).
    """
    path = Path(path) if path else data_path(PARAMETERS_CSV)
    params = np.loadtxt(open(path), delimiter=",", usecols=range(4))
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError(f"expected (n, 4) parameters from {path}, got {params.shape}")
    return params


def default_initial_state(n: int = 100) -> np.ndarray:
    """Seed state tiled across parameter sets, as ``initialModelSetUp`` does."""
    return np.tile(np.array(DEFAULT_INITIAL_STATE, dtype=np.float64), (n, 1))


def _wind_10m_to_2m(wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    """Log-profile conversion of 10 m wind magnitude to 2 m."""
    u10 = np.sqrt(wind_u**2 + wind_v**2)
    u0 = 0.0
    uTAU = ((u10 - u0) / 2.5) / (math.log(_Z10 / _Z0))
    return 2.5 * uTAU * (math.log(_Z2 / _Z0)) + u0


def generate_river_flows(
    weather: WeatherSeries,
    state: np.ndarray,
    params: np.ndarray,
    catchment: CatchmentConfig = DEFAULT_CATCHMENT,
    prediction_date: datetime | None = None,
) -> RiverFlowResult:
    """
    Run one weather realisation through FAO56, PDM and the routing stores.

    :param weather: one ensemble member or an observed history
    :param state: ``(n_sets, 3)`` initial state; **copied**, never mutated
    :param params: ``(n_sets, 4)`` from :func:`load_parameters`
    :param prediction_date: start of the series; defaults to ``weather.start``.
        Must be timezone-aware - FAO56 derives day-of-year from it.
    """
    weather.validate()
    dt = catchment.timestep_days

    if prediction_date is None:
        prediction_date = weather.start
    if prediction_date.tzinfo is None:
        raise ValueError("prediction_date must be timezone-aware")
    prediction_date = prediction_date.astimezone(timezone.utc)

    N = len(weather)
    RH = weather.data[:, WeatherSeries.RH]
    TempMax = weather.data[:, WeatherSeries.TMAX] - 273.15
    TempMin = weather.data[:, WeatherSeries.TMIN] - 273.15
    T = (TempMin + TempMax) / 2

    # Daily min/max: collapse each day's four buckets, then broadcast back.
    Tmin = np.repeat(TempMin.reshape(N // 4, 4).min(axis=1), 4)
    Tmax = np.repeat(TempMax.reshape(N // 4, 4).max(axis=1), 4)

    u2 = _wind_10m_to_2m(
        weather.data[:, WeatherSeries.U], weather.data[:, WeatherSeries.V]
    )

    # Precipitation is mm per bucket; the model wants mm/day.
    qp = weather.data[:, WeatherSeries.PRECIP] / dt

    Ep, E0 = FAO56(
        dt,
        prediction_date,
        Tmin,
        Tmax,
        catchment.altitude_m,
        catchment.latitude_deg,
        T,
        u2,
        RH,
    )

    # ModelFun mutates its F0 argument in place, so hand it a private copy.
    state_out = np.array(state, dtype=np.float64, copy=True)
    Q, state_out = ModelFun(qp, Ep, dt, catchment.area_km2, params, state_out)

    return RiverFlowResult(
        times=weather.times,
        flow_m3s=Q,
        rainfall_mm_day=qp,
        pet_mm_day=Ep,
        open_water_evap=E0,
        state=state_out,
        member=weather.member,
    )


def spin_up(
    history: WeatherSeries,
    params: np.ndarray,
    catchment: CatchmentConfig = DEFAULT_CATCHMENT,
    state: np.ndarray | None = None,
) -> np.ndarray:
    """
    Replay observed weather to bring the model's internal state to something
    realistic, returning ``(n_sets, 3)``.

    Matches ``initialModelSetUp``, which runs a single pass over the whole
    ``INITIAL_BACKTIME`` window rather than stepping day by day. The seed state
    is deliberately crude - roughly a month of real rainfall pulls it to
    something physically sensible regardless of where it started.
    """
    if state is None:
        state = default_initial_state(params.shape[0])
    logger.info(
        "Spinning up over %d buckets (%.0f days)", len(history), len(history) / 4
    )
    return generate_river_flows(history, state, params, catchment).state


def run_ensemble(
    forecast: list[WeatherSeries],
    state: np.ndarray,
    params: np.ndarray,
    catchment: CatchmentConfig = DEFAULT_CATCHMENT,
    prediction_date: datetime | None = None,
) -> EnsembleFlows:
    """
    Run every ensemble member forward from the same initial state.

    Each member starts from a *copy* of ``state``: ``ModelFun`` writes its
    end-of-run state back into the array it was given, so sharing one array
    between members would silently chain them together.
    """
    if not forecast:
        raise ValueError("forecast is empty")

    aligned = align_series(forecast)
    if len(aligned[0]) == 0:
        raise ValueError("no whole days remain after aligning ensemble members")

    results = [
        generate_river_flows(member, state, params, catchment, prediction_date)
        for member in aligned
    ]

    logger.info(
        "Ran %d member(s) x %d step(s) x %d parameter set(s)",
        len(results),
        len(aligned[0]),
        params.shape[0],
    )
    return EnsembleFlows(
        times=aligned[0].times,
        members=[r.member for r in results],
        flow_m3s=np.stack([r.flow_m3s for r in results]),
        rainfall_mm_day=np.stack([r.rainfall_mm_day for r in results]),
        pet_mm_day=np.stack([r.pet_mm_day for r in results]),
    )
