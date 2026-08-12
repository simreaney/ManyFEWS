"""
Configuration for the ManyFEWS core pipeline.

Every magic number that the Django application buries in ``settings.py``, in
module-level constants, or (worst) inline in a function body lands here as a
field on a frozen dataclass. Nothing in this package reads global state.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CatchmentConfig",
    "ForecastConfig",
    "StormConfig",
    "EmulatorConfig",
    "RiskConfig",
    "DEFAULT_CATCHMENT",
    "DEFAULT_INITIAL_STATE",
]


# Seed state for the three PDM/routing state variables, tiled across all
# parameter sets to start a spin-up. From tasks.initialModelSetUp:
#     np.tile(np.array([20.556992, 3.86579, 1.862992]), (100, 1))
# Order is (storage_level mm, slow_flow_rate mm/day, fast_flow_rate mm/day).
DEFAULT_INITIAL_STATE = (20.556992, 3.86579, 1.862992)


@dataclass(frozen=True)
class CatchmentConfig:
    """
    Physical description of the catchment being modelled.

    ``latitude_deg`` / ``altitude_m`` / ``area_km2`` were hardcoded inside
    ``GenerateRiverFlows`` in the Django app (generate_river_flows.py:423-425),
    which is why that function only ever worked for Majalaya. They are catchment
    *means* and are used by FAO56 for solar geometry and atmospheric pressure.

    ``weather_lat`` / ``weather_lon`` are a different thing: the point at which
    weather is sampled from Open-Meteo. In the Django app these are the separate
    ``LAT_VALUE`` / ``LON_VALUE`` settings, and they do not equal the catchment
    centroid. Keeping the two pairs distinct is deliberate.
    """

    name: str = "Majalaya"

    # Catchment means - drive the hydrology.
    latitude_deg: float = -7.125
    altitude_m: float = 1157.0
    area_km2: float = 212.2640

    # Weather sampling point - drives the API request. Not the same location.
    weather_lat: float = -7.05
    weather_lon: float = 107.758

    timestep_days: float = 0.25
    n_param_sets: int = 100

    def __post_init__(self) -> None:
        if self.timestep_days != 0.25:
            raise ValueError(
                f"timestep_days must be 0.25, got {self.timestep_days}. "
                "Four 6-hour buckets per day is baked into the numerics in two "
                "places: FAO56 computes day-of-year as "
                "`np.arange(0, np.size(Tmax) / 4, dt)`, and GenerateRiverFlows "
                "groups days with `reshape(N / 4, 4)`. Any other value silently "
                "produces a mismatched day grouping rather than an error."
            )
        if self.area_km2 <= 0:
            raise ValueError(f"area_km2 must be positive, got {self.area_km2}")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude_deg out of range: {self.latitude_deg}")


DEFAULT_CATCHMENT = CatchmentConfig()


@dataclass(frozen=True)
class ForecastConfig:
    """Open-Meteo fetching behaviour."""

    model: str = "gfs_seamless"
    forecast_days: int = 16
    max_members: int | None = 10  # None or 0 keeps every member available
    spinup_days: int = 29  # Django's INITIAL_BACKTIME

    # Open-Meteo's ERA5T archive lags real time by roughly five days; requests
    # closer than this return nulls, which _bucket_to_6h drops, which shifts the
    # day grouping without any error. Django's initialModelSetUp asks for data up
    # to *yesterday* and is exposed to exactly this.
    archive_lag_days: int = 6

    timeout_s: float = 60.0
    retries: int = 3
    backoff_s: float = 5.0


@dataclass(frozen=True)
class StormConfig:
    """
    Synthetic design storm, mirroring the Django ``TestModeSettings`` singleton.

    ``total_mm`` defaults to 100.0 for parity with ``TestModeSettings.STORM_TOTAL_MM``,
    but note that 100 mm does *not* generate any flooding in this catchment: it
    produces a peak p90 flow of about 36 m3/s against a lowest emulator threshold
    of 50. Around 200 mm is needed to see anything on the map.
    """

    enabled: bool = False
    total_mm: float = 100.0
    days_ahead: int = 2


@dataclass(frozen=True)
class EmulatorConfig:
    """
    Flood-depth emulator behaviour.

    ``q_cap_m3s`` has no counterpart in the Django app, which clamps only
    negative depths (flood_risk.py). The fitted cubics diverge above their
    calibration range - measured maxima are 15 m at Q=300, 118 m at Q=500 and
    1122 m at Q=800, and the wet-cell count actually *falls* above 300 as cubics
    go negative. Clamping the input rather than the output also keeps the
    response monotone, which is what makes the exact percentile shortcut in
    ``emulator.depth_percentiles`` valid.
    """

    q_cap_m3s: float = 300.0
    percentiles: tuple[float, ...] = (10.0, 30.0, 50.0, 90.0)
    mask_channel: bool = True
    method: Literal["auto", "hybrid", "brute"] = "auto"


@dataclass(frozen=True)
class RiskConfig:
    """
    Wet-cell count to headline risk percentage.

    Both constants come from Django settings and neither matches the shipped
    parameter file: the grid holds 302,748 cells in total, so
    ``large_flood_count`` of 1,440,811 is unreachable and risk saturates at
    about 0.225. ``legacy_formula`` reproduces the Django behaviour exactly
    (``n / (large - channel)``); setting it False uses the presumably intended
    ``(n - channel) / (large - channel)``.
    """

    channel_cell_count: int = 93794
    large_flood_count: int = 1440811
    legacy_formula: bool = True
